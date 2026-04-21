from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PatternToken:
    """Minimal token used by pattern templates."""

    index: int
    kinds: FrozenSet[str]
    mem_reads: Tuple[str, ...]         # base regs read (from memory operand)
    mem_writes: Tuple[str, ...]        # base regs written (from memory operand)
    opcode_variant: str
    # NEW: non-memory register side effects, used to reason about dependency chains
    #   dst_regs: registers overwritten by the instruction (ALU dst / load dst)
    #   src_regs: registers read by the instruction (ALU src)
    dst_regs: Tuple[str, ...] = ()
    src_regs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PatternTemplate:
    """Configurable template for vulnerability-specific code patterns."""

    name: str
    sequence: Tuple[str, ...]
    max_gap: int
    base_reward: float
    require_store_load_alias: bool = False
    max_matches: int = 2


@dataclass
class PatternMatchResult:
    score: float
    matches: Dict[str, int]


class VulnerabilityPatternMatcher:
    """
    Match configurable instruction templates and return a shaping score.
    Templates are grouped by vulnerability type for easy extension.

    For Spectre v4 (SSB) we want the agent to produce the canonical gadget:
        (A) slow-addr producer: writes the register that is about to be used as a store base
            via a load or long-latency ALU (mul/div/mov_rm/add_rm).
        (B) store:               memory write whose base is that slow register.
        (C) bypass load:         memory read that aliases the store (same base reg) AND
                                 the base register is NOT overwritten between B and C.
        (D) transmitter:         memory access whose base register is the DESTINATION of
                                 the bypass load (C), so the stale value is encoded into
                                 the cache, producing a side-channel signal.

    The rewards below are layered: pair-level shaping remains for early exploration,
    and the full gadget gets a large, sparse bonus to drive convergence toward v4.
    """

    TEMPLATE_LIBRARY: Dict[str, List[PatternTemplate]] = {
        "spectre_v4": [
            # Core store-to-load forwarding candidate with likely alias.
            # max_gap doubled to account for sandbox "and <reg>, mask" instrumentation
            # inserted between every user instruction.
            PatternTemplate(
                name="stl_alias_pair",
                sequence=("store", "load"),
                max_gap=4,
                base_reward=10.0,
                require_store_load_alias=True,
                max_matches=2,
            ),
            # Weaker candidate when alias cannot be inferred.
            PatternTemplate(
                name="stl_pair",
                sequence=("store", "load"),
                max_gap=8,
                base_reward=4.0,
                require_store_load_alias=False,
                max_matches=2,
            ),
        ]
    }

    # Extra rule-level shaping terms (independent from templates).
    SAME_REG_STORE_LOAD_BONUS: float = 3.0
    SLOW_STORE_FAST_LOAD_BONUS: float = 5.0
    REPEATED_INSTR_PENALTY: float = 2.0
    SLOW_ADDR_LOOKBACK: int = 4
    FAST_LOAD_MAX_GAP: int = 4

    # v4-specific shaping
    V4_FULL_GADGET_BONUS: float = 60.0          # store -> bypass-load -> transmitter all present
    V4_SLOW_STORE_PRODUCER_BONUS: float = 20.0  # load/mul/div writes store's base right before store
    V4_TRANSMITTER_BONUS: float = 30.0          # memory access with base = bypass-load's dst
    V4_FAST_LOAD_MAX_GAP: int = 6               # bypass load within this many tokens after store
    V4_TRANSMITTER_MAX_GAP: int = 4             # transmitter within this many tokens after bypass load
    V4_BASE_OVERWRITE_PENALTY: float = 15.0     # store's base reg written between store and load
    V4_EXPLICIT_MASK_PENALTY: float = 10.0      # explicit mov_ri / xor_rr(same) on store's base between
    V4_TIGHT_WINDOW: int = 2                    # adjacency window for timing-critical rewards
    V4_NEAR_SLOW_PRODUCER_BONUS: float = 12.0   # slow producer immediately before store
    V4_TIGHT_BYPASS_BONUS: float = 10.0         # bypass load appears right after store
    V4_NEAR_TRANSMITTER_BONUS: float = 12.0     # transmitter appears right after bypass load
    V4_BYPASS_DST_CLOBBER_PENALTY: float = 20.0 # bypass-loaded value overwritten before transmitter

    # Opcode variants that look like long-latency producers (used for slow-addr detection).
    _LOAD_PRODUCER_OPS: FrozenSet[str] = frozenset(
        {"mov_rm", "add_rm", "sub_rm", "cmp_rm"}
    )
    _LATENCY_PRODUCER_OPS: FrozenSet[str] = frozenset(
        {"mul", "div"}
    )
    _SLOWISH_OPS: FrozenSet[str] = frozenset(
        {
            "mul",
            "div",
            "mov_mr",
            "mov_mi",
            "add_mr",
            "add_mi",
            "sbb_mr",
            "sbb_mi",
            "cmp_mr",
            "cmp_rm",
        }
    )

    def __init__(self, vulnerability_type: str = "spectre_v4"):
        self.vulnerability_type = vulnerability_type
        self.templates = self.TEMPLATE_LIBRARY.get(vulnerability_type, [])

    @classmethod
    def register_templates(cls, vulnerability_type: str, templates: List[PatternTemplate]) -> None:
        """Allow callers to inject/replace templates for other vulnerabilities."""
        cls.TEMPLATE_LIBRARY[vulnerability_type] = templates

    def score(self, tokens: Sequence[PatternToken]) -> PatternMatchResult:
        total = 0.0
        counts: Dict[str, int] = {}
        for template in self.templates:
            match_count = self._count_matches(tokens, template)
            if match_count <= 0:
                continue
            counts[template.name] = match_count
            total += template.base_reward * float(match_count)

        # Rule 1: prefer store/load pairs that share the same memory base register.
        same_reg_pairs = self._count_same_reg_store_load_pairs(tokens)
        if same_reg_pairs > 0:
            counts["same_reg_store_load_pairs"] = same_reg_pairs
            total += self.SAME_REG_STORE_LOAD_BONUS * float(same_reg_pairs)

        # Rule 1b: prefer pairs where store side looks slower and load side looks faster.
        slow_fast_pairs = self._count_slow_store_fast_load_pairs(tokens)
        if slow_fast_pairs > 0:
            counts["slow_store_fast_load_pairs"] = slow_fast_pairs
            total += self.SLOW_STORE_FAST_LOAD_BONUS * float(slow_fast_pairs)

        # Rule 2: penalize consecutive repeated instructions.
        repeated_adjacent = self._count_adjacent_repeated_instructions(tokens)
        if repeated_adjacent > 0:
            counts["adjacent_repeated_instr"] = repeated_adjacent
            total -= self.REPEATED_INSTR_PENALTY * float(repeated_adjacent)

        # v4-specific structured shaping (only meaningful when we target v4).
        if self.vulnerability_type == "spectre_v4":
            v4_score, v4_counts = self._score_v4_structured(tokens)
            total += v4_score
            counts.update(v4_counts)

        return PatternMatchResult(score=total, matches=counts)

    # ------------------------------------------------------------------
    # Generic template matching
    # ------------------------------------------------------------------
    def _count_matches(self, tokens: Sequence[PatternToken], template: PatternTemplate) -> int:
        if not template.sequence or not tokens:
            return 0

        matches = 0
        for start in range(len(tokens)):
            if template.sequence[0] not in tokens[start].kinds:
                continue
            end_idx = self._follow_template(tokens, template, start, 1)
            if end_idx is None:
                continue
            if template.require_store_load_alias:
                if not self._has_store_load_alias(tokens[start], tokens[end_idx]):
                    continue
                # For alias template, also require that the store-base register is not
                # overwritten by any intervening user instruction (otherwise the second
                # `and base, mask` will use a different value and the addresses no longer
                # alias byte-for-byte).
                if self._base_overwritten_between(tokens, start, end_idx):
                    continue
            matches += 1
            if matches >= template.max_matches:
                break
        return matches

    def _follow_template(
        self,
        tokens: Sequence[PatternToken],
        template: PatternTemplate,
        current_idx: int,
        seq_pos: int,
    ) -> Optional[int]:
        if seq_pos >= len(template.sequence):
            return current_idx

        target_kind = template.sequence[seq_pos]
        max_next = min(len(tokens) - 1, current_idx + template.max_gap + 1)
        for nxt in range(current_idx + 1, max_next + 1):
            if target_kind not in tokens[nxt].kinds:
                continue
            out = self._follow_template(tokens, template, nxt, seq_pos + 1)
            if out is not None:
                return out
        return None

    # ------------------------------------------------------------------
    # Helpers shared across rules
    # ------------------------------------------------------------------
    @staticmethod
    def _has_store_load_alias(store_token: PatternToken, load_token: PatternToken) -> bool:
        if "store" not in store_token.kinds or "load" not in load_token.kinds:
            return False
        return bool(set(store_token.mem_writes).intersection(load_token.mem_reads))

    @staticmethod
    def _shared_base_regs(store_token: PatternToken, load_token: PatternToken) -> FrozenSet[str]:
        return frozenset(set(store_token.mem_writes).intersection(load_token.mem_reads))

    def _base_overwritten_between(
        self,
        tokens: Sequence[PatternToken],
        store_idx: int,
        load_idx: int,
    ) -> bool:
        """
        Return True if any base register shared between store and load is (re)written
        by a user instruction in (store_idx, load_idx).  Instrumentation tokens are
        already filtered out upstream (is_instrumentation=True are dropped in the
        env's _build_pattern_tokens), so any write here is an agent-chosen one and
        would break the byte-level alias that SSB requires.
        """
        if load_idx - store_idx <= 1:
            return False
        shared = self._shared_base_regs(tokens[store_idx], tokens[load_idx])
        if not shared:
            return False
        for k in range(store_idx + 1, load_idx):
            dst = set(tokens[k].dst_regs)
            # Also consider memory-store bases as "writes" of that reg only if the
            # store instruction's mem op is a dst — but the store base is NOT a
            # GPR write, it's a memory write.  So only dst_regs matters here.
            if shared.intersection(dst):
                return True
        return False

    def _count_same_reg_store_load_pairs(self, tokens: Sequence[PatternToken]) -> int:
        """Count nearby store->load pairs that share at least one base register."""
        pairs = 0
        for i, token in enumerate(tokens):
            if "store" not in token.kinds:
                continue
            max_next = min(len(tokens), i + 5)
            for j in range(i + 1, max_next):
                nxt = tokens[j]
                if "load" not in nxt.kinds:
                    continue
                if self._has_store_load_alias(token, nxt):
                    pairs += 1
                    break
        return pairs

    def _count_slow_store_fast_load_pairs(self, tokens: Sequence[PatternToken]) -> int:
        """
        Heuristic for SSB-favorable timing:
        - store has likely slow address path (complex ops in lookback window)
        - load happens quickly afterwards and with no complex ops in between
        - store/load still alias on base register
        """
        pairs = 0
        for i, token in enumerate(tokens):
            if "store" not in token.kinds:
                continue
            if not self._has_slow_store_context(tokens, i):
                continue

            max_next = min(len(tokens), i + self.FAST_LOAD_MAX_GAP + 2)
            for j in range(i + 1, max_next):
                nxt = tokens[j]
                if "load" not in nxt.kinds:
                    continue
                if not self._has_store_load_alias(token, nxt):
                    continue
                if not self._has_fast_load_context(tokens, i, j):
                    continue
                pairs += 1
                break
        return pairs

    def _has_slow_store_context(self, tokens: Sequence[PatternToken], store_idx: int) -> bool:
        start = max(0, store_idx - self.SLOW_ADDR_LOOKBACK)
        store_bases = set(tokens[store_idx].mem_writes)
        for k in range(start, store_idx):
            t = tokens[k]
            if t.opcode_variant in self._SLOWISH_OPS:
                return True
            if store_bases and (
                store_bases.intersection(t.mem_reads) or store_bases.intersection(t.mem_writes)
            ):
                return True
        return False

    def _has_fast_load_context(self, tokens: Sequence[PatternToken], store_idx: int, load_idx: int) -> bool:
        if (load_idx - store_idx) > self.FAST_LOAD_MAX_GAP:
            return False
        for k in range(store_idx + 1, load_idx):
            if tokens[k].opcode_variant in self._SLOWISH_OPS:
                return False
        return True

    def _count_adjacent_repeated_instructions(self, tokens: Sequence[PatternToken]) -> int:
        """Count how often two adjacent instructions use the same opcode variant."""
        exempt_pairs = self._build_adjacent_key_pair_exemptions(tokens)
        repeated = 0
        for i in range(1, len(tokens)):
            if (i - 1, i) in exempt_pairs:
                continue
            if tokens[i].opcode_variant == tokens[i - 1].opcode_variant:
                repeated += 1
        return repeated

    def _build_adjacent_key_pair_exemptions(self, tokens: Sequence[PatternToken]) -> FrozenSet[Tuple[int, int]]:
        """
        Do not penalize adjacent key pair store->load alias sequences; they are
        exactly what we want to encourage for SSB exploration.
        """
        exempt = set()
        for i in range(1, len(tokens)):
            prev_t = tokens[i - 1]
            cur_t = tokens[i]
            if "store" not in prev_t.kinds or "load" not in cur_t.kinds:
                continue
            if self._has_store_load_alias(prev_t, cur_t):
                exempt.add((i - 1, i))
        return frozenset(exempt)

    # ------------------------------------------------------------------
    # Spectre v4 structured gadget scoring
    # ------------------------------------------------------------------
    def _score_v4_structured(
        self, tokens: Sequence[PatternToken]
    ) -> Tuple[float, Dict[str, int]]:
        """
        Walk the token stream and reward the canonical v4 gadget structure:
            slow-addr producer -> store -> bypass load -> transmitter.

        Partial credit is handed out for each of the three structural pieces so
        that PPO gets a dense signal long before the full gadget appears, while
        the full-gadget bonus is large enough to dominate once it's assembled.
        """
        total = 0.0
        counts: Dict[str, int] = {
            "v4_slow_store_producer": 0,
            "v4_near_slow_store_producer": 0,
            "v4_bypass_load": 0,
            "v4_tight_bypass_load": 0,
            "v4_transmitter": 0,
            "v4_near_transmitter": 0,
            "v4_full_gadget": 0,
            "v4_base_overwritten": 0,
            "v4_explicit_base_mask": 0,
            "v4_bypass_dst_clobbered": 0,
        }

        full_gadgets = 0
        # Scan each candidate store
        for i, store_tok in enumerate(tokens):
            if "store" not in store_tok.kinds:
                continue
            store_bases = set(store_tok.mem_writes)
            if not store_bases:
                continue

            slow_producer = self._has_slow_addr_producer(tokens, i, store_bases)
            if slow_producer:
                counts["v4_slow_store_producer"] += 1
                total += self.V4_SLOW_STORE_PRODUCER_BONUS
                if self._has_near_slow_addr_producer(tokens, i, store_bases):
                    counts["v4_near_slow_store_producer"] += 1
                    total += self.V4_NEAR_SLOW_PRODUCER_BONUS

            # Find nearest aliasing load, ensuring base isn't overwritten in between
            bypass_idx = self._find_bypass_load(tokens, i, store_bases)
            if bypass_idx is None:
                continue
            counts["v4_bypass_load"] += 1
            if (bypass_idx - i) <= self.V4_TIGHT_WINDOW:
                counts["v4_tight_bypass_load"] += 1
                total += self.V4_TIGHT_BYPASS_BONUS

            # Mild penalty if the agent explicitly smashes the store base between
            # store and load (and still happens to alias by luck).
            if self._has_explicit_base_reset(tokens, i, bypass_idx, store_bases):
                counts["v4_explicit_base_mask"] += 1
                total -= self.V4_EXPLICIT_MASK_PENALTY

            # Look for a transmitter: a memory access within a small window whose
            # BASE register is a register written by the bypass load.
            transmitter_idx = self._find_transmitter(tokens, bypass_idx)
            if transmitter_idx is not None:
                if self._bypass_dst_clobbered_between(tokens, bypass_idx, transmitter_idx):
                    counts["v4_bypass_dst_clobbered"] += 1
                    total -= self.V4_BYPASS_DST_CLOBBER_PENALTY
                    transmitter_idx = None
                else:
                    counts["v4_transmitter"] += 1
                    total += self.V4_TRANSMITTER_BONUS
                    if (transmitter_idx - bypass_idx) <= self.V4_TIGHT_WINDOW:
                        counts["v4_near_transmitter"] += 1
                        total += self.V4_NEAR_TRANSMITTER_BONUS

            if slow_producer and transmitter_idx is not None:
                full_gadgets += 1

        if full_gadgets > 0:
            counts["v4_full_gadget"] = full_gadgets
            total += self.V4_FULL_GADGET_BONUS * float(full_gadgets)

        # Prune zero-count shaping keys so logs are readable.
        counts = {k: v for k, v in counts.items() if v}
        return total, counts

    def _has_slow_addr_producer(
        self,
        tokens: Sequence[PatternToken],
        store_idx: int,
        store_bases: FrozenSet[str],
    ) -> bool:
        """Look back a few tokens for an instr that WRITES the store's base reg via a load/mul/div."""
        start = max(0, store_idx - self.SLOW_ADDR_LOOKBACK)
        for k in range(start, store_idx):
            t = tokens[k]
            if t.opcode_variant not in self._LOAD_PRODUCER_OPS and \
               t.opcode_variant not in self._LATENCY_PRODUCER_OPS:
                continue
            if set(t.dst_regs).intersection(store_bases):
                return True
        return False

    def _has_near_slow_addr_producer(
        self,
        tokens: Sequence[PatternToken],
        store_idx: int,
        store_bases: FrozenSet[str],
    ) -> bool:
        """Prefer timing-critical producer->store adjacency."""
        start = max(0, store_idx - self.V4_TIGHT_WINDOW)
        for k in range(start, store_idx):
            t = tokens[k]
            if t.opcode_variant not in self._LOAD_PRODUCER_OPS and \
               t.opcode_variant not in self._LATENCY_PRODUCER_OPS:
                continue
            if set(t.dst_regs).intersection(store_bases):
                return True
        return False

    def _find_bypass_load(
        self,
        tokens: Sequence[PatternToken],
        store_idx: int,
        store_bases: FrozenSet[str],
    ) -> Optional[int]:
        max_next = min(len(tokens), store_idx + self.V4_FAST_LOAD_MAX_GAP + 1)
        for j in range(store_idx + 1, max_next):
            t = tokens[j]
            if "load" not in t.kinds:
                continue
            if not store_bases.intersection(t.mem_reads):
                continue
            # Reject if any intervening user instruction overwrites the base reg.
            if self._base_overwritten_between(tokens, store_idx, j):
                # record the overwrite event for diagnostics via a noop: the caller
                # will still count nothing for this pair.
                return None
            return j
        return None

    def _find_transmitter(
        self, tokens: Sequence[PatternToken], bypass_idx: int
    ) -> Optional[int]:
        """
        A transmitter is a memory access whose base register is a register
        written by the bypass load (i.e., its dst_regs).  That base is exactly
        the stale/speculated value read from the store buffer, so the
        subsequent memory access depends on the secret value.
        """
        bypass_dst = set(tokens[bypass_idx].dst_regs)
        if not bypass_dst:
            return None
        max_next = min(len(tokens), bypass_idx + self.V4_TRANSMITTER_MAX_GAP + 1)
        for j in range(bypass_idx + 1, max_next):
            t = tokens[j]
            if "memory" not in t.kinds:
                continue
            bases = set(t.mem_reads).union(t.mem_writes)
            if bypass_dst.intersection(bases):
                return j
        return None

    def _bypass_dst_clobbered_between(
        self, tokens: Sequence[PatternToken], bypass_idx: int, transmitter_idx: int
    ) -> bool:
        """
        If bypass-load destination is overwritten before transmitter, the stale value
        is no longer used as address and v4 observability collapses.
        """
        if transmitter_idx - bypass_idx <= 1:
            return False
        bypass_dst = set(tokens[bypass_idx].dst_regs)
        if not bypass_dst:
            return False
        for k in range(bypass_idx + 1, transmitter_idx):
            if bypass_dst.intersection(tokens[k].dst_regs):
                return True
        return False

    def _has_explicit_base_reset(
        self,
        tokens: Sequence[PatternToken],
        store_idx: int,
        load_idx: int,
        store_bases: FrozenSet[str],
    ) -> bool:
        """Detect explicit resets like `mov base, imm` or `xor base, base` between store and load."""
        for k in range(store_idx + 1, load_idx):
            t = tokens[k]
            if t.opcode_variant == "mov_ri" and set(t.dst_regs).intersection(store_bases):
                return True
            if t.opcode_variant == "xor" and set(t.dst_regs).intersection(store_bases) \
                    and set(t.src_regs).intersection(store_bases):
                # xor <base>, <base> zeroes the base
                return True
        return False
