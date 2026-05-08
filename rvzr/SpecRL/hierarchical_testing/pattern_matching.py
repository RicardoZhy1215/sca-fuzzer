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
        # [V4-only templates commented out — re-enable for v4 training.]
        # "spectre_v4": [
        #     # Core store-to-load forwarding candidate with likely alias.
        #     # max_gap doubled to account for sandbox "and <reg>, mask" instrumentation
        #     # inserted between every user instruction.
        #     PatternTemplate(
        #         name="stl_alias_pair",
        #         sequence=("store", "load"),
        #         max_gap=4,
        #         base_reward=10.0,
        #         require_store_load_alias=True,
        #         max_matches=2,
        #     ),
        #     # Weaker candidate when alias cannot be inferred.
        #     PatternTemplate(
        #         name="stl_pair",
        #         sequence=("store", "load"),
        #         max_gap=8,
        #         base_reward=4.0,
        #         require_store_load_alias=False,
        #         max_matches=2,
        #     ),
        # ],
        # spectre_v1 templates not yet defined; use generic rule-level shaping
        # (same_reg_store_load / slow-store-fast-load / repeated-instr) until
        # v1-specific gadget patterns are added here.
    }

    # Extra rule-level shaping terms (independent from templates).
    SAME_REG_STORE_LOAD_BONUS: float = 3.0
    SLOW_STORE_FAST_LOAD_BONUS: float = 5.0
    REPEATED_INSTR_PENALTY: float = 2.0
    # P1.1: canonical v4 needs >=10 slow-addr producers on the store base
    # before the store itself, so the lookback has to be wide enough to
    # actually reach them. Old value (4) only covered the last 4 user-instrs.
    # Bumped 20 -> 30 so chain-length counting and slow-producer detection
    # can see all of the ~26 lea chain in tests/x86_tests/asm/spectre_v4.asm.
    SLOW_ADDR_LOOKBACK: int = 30
    FAST_LOAD_MAX_GAP: int = 4

    # [V4-only] Spectre v4 structured-gadget shaping constants. Kept here
    # commented for reference; not used while training Spectre v1.
    # V4_FULL_GADGET_BONUS: float = 200.0
    # V4_SLOW_STORE_PRODUCER_BONUS: float = 30.0
    # V4_TRANSMITTER_BONUS: float = 50.0
    # V4_FAST_LOAD_MAX_GAP: int = 8
    # V4_TRANSMITTER_MAX_GAP: int = 4
    # V4_BASE_OVERWRITE_PENALTY: float = 15.0
    # V4_EXPLICIT_MASK_PENALTY: float = 10.0
    # V4_TIGHT_WINDOW: int = 2
    # V4_NEAR_SLOW_PRODUCER_BONUS: float = 12.0
    # V4_TIGHT_BYPASS_BONUS: float = 10.0
    # V4_NEAR_TRANSMITTER_BONUS: float = 12.0
    # V4_BYPASS_DST_CLOBBER_PENALTY: float = 20.0
    # V4_SLOW_CHAIN_LINK_BONUS: float = 3.0
    # V4_SLOW_CHAIN_CAP: int = 30
    # V4_MIN_CHAIN_LEN_FOR_FULL: int = 5
    # V4_CHAIN_LEN_SATURATION: int = 15
    # V4_SLOW_CHAIN_SMASHED_PENALTY: float = 25.0
    # _FAST_RESET_OPS: FrozenSet[str] = frozenset({"mov_ri", "mov_rr", "xor_rr"})

    # Opcode variants that look like long-latency producers (used for slow-addr detection).
    # lea_rrr (`lea rd, [rs + rd + 1]`) chains rd through itself which gives
    # exactly the serial dependency the canonical v4 gadget uses to delay
    # the store's address resolution. Treat it as a slow address producer.
    _LOAD_PRODUCER_OPS: FrozenSet[str] = frozenset(
        {"mov_rm", "add_rm", "sub_rm", "cmp_rm", "lea_rrr"}
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
            "lea_rrr",
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

        # [V4-only] structured-gadget shaping. Disabled while training v1.
        # if self.vulnerability_type == "spectre_v4":
        #     v4_score, v4_counts = self._score_v4_structured(tokens)
        #     total += v4_score
        #     counts.update(v4_counts)

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

        P0.1: ALSO exempt adjacent slow-addr producers that chain on the same
        register (e.g. `lea_rrr rbx,... ; lea_rrr rbx,...` or `mul ; mul`).
        Stacking these is exactly the canonical v4 delay chain — the old
        blanket REPEATED_INSTR_PENALTY was actively training the policy
        AGAINST the single most important v4 pattern.
        """
        exempt = set()
        slow_ops = self._LOAD_PRODUCER_OPS | self._LATENCY_PRODUCER_OPS
        for i in range(1, len(tokens)):
            prev_t = tokens[i - 1]
            cur_t = tokens[i]
            if "store" in prev_t.kinds and "load" in cur_t.kinds \
                    and self._has_store_load_alias(prev_t, cur_t):
                exempt.add((i - 1, i))
                continue
            if prev_t.opcode_variant in slow_ops and cur_t.opcode_variant in slow_ops:
                # Only exempt when both writes land on the same register(s);
                # otherwise the "repeat" really is a wasted duplicate op.
                if set(prev_t.dst_regs).intersection(cur_t.dst_regs):
                    exempt.add((i - 1, i))
        return frozenset(exempt)

#     # ------------------------------------------------------------------
#     # Spectre v4 structured gadget scoring
#     # ------------------------------------------------------------------
#     def _score_v4_structured(
#         self, tokens: Sequence[PatternToken]
#     ) -> Tuple[float, Dict[str, int]]:
#         """
#         Walk the token stream and reward the canonical v4 gadget structure:
#             slow-addr producer -> store -> bypass load -> transmitter.
# 
#         Partial credit is handed out for each of the three structural pieces so
#         that PPO gets a dense signal long before the full gadget appears, while
#         the full-gadget bonus is large enough to dominate once it's assembled.
#         """
#         total = 0.0
#         counts: Dict[str, int] = {
#             "v4_slow_store_producer": 0,
#             "v4_near_slow_store_producer": 0,
#             "v4_bypass_load": 0,
#             "v4_tight_bypass_load": 0,
#             "v4_transmitter": 0,
#             "v4_near_transmitter": 0,
#             "v4_full_gadget": 0,
#             "v4_base_overwritten": 0,
#             "v4_explicit_base_mask": 0,
#             "v4_bypass_dst_clobbered": 0,
#             # New: counts how often the agent built a slow chain and then
#             # smashed it with a fast-reset op before the store (the reward-hack
#             # pattern). Should drop to near-zero once the policy converges.
#             "v4_slow_chain_smashed": 0,
#         }
# 
#         full_gadgets = 0
#         # Scan each candidate store
#         for i, store_tok in enumerate(tokens):
#             if "store" not in store_tok.kinds:
#                 continue
#             store_bases = set(store_tok.mem_writes)
#             if not store_bases:
#                 continue
# 
#             # Detect the "farm slow chain, then smash with mov_ri/xor_rr"
#             # reward-hack pattern and penalize it BEFORE crediting any slow
#             # chain / full gadget bonuses for this store. The actual slow
#             # checks below use last-writer semantics, so once a smash is
#             # detected slow_producer will already evaluate to False, but we
#             # still want the explicit penalty to create a dedicated negative
#             # gradient against the pattern.
#             if self._slow_chain_smashed(tokens, i, store_bases):
#                 counts["v4_slow_chain_smashed"] += 1
#                 total -= self.V4_SLOW_CHAIN_SMASHED_PENALTY
# 
#             slow_producer = self._has_slow_addr_producer(tokens, i, store_bases)
#             if slow_producer:
#                 counts["v4_slow_store_producer"] += 1
#                 total += self.V4_SLOW_STORE_PRODUCER_BONUS
#                 if self._has_near_slow_addr_producer(tokens, i, store_bases):
#                     counts["v4_near_slow_store_producer"] += 1
#                     total += self.V4_NEAR_SLOW_PRODUCER_BONUS
# 
#             # P1.2: dense credit per extra slow producer that writes the
#             # store's base within the lookback window. Gives the agent a
#             # monotone "+3 for every additional lea_rrr/mul" gradient, so
#             # PPO can climb from a 1-link chain to a 10+-link chain without
#             # each intermediate step looking reward-flat.
#             chain_len = self._count_slow_chain_length(tokens, i, store_bases)
#             if chain_len > 0:
#                 counts["v4_slow_chain_len"] = max(
#                     counts.get("v4_slow_chain_len", 0), chain_len
#                 )
#                 total += self.V4_SLOW_CHAIN_LINK_BONUS * min(
#                     chain_len, self.V4_SLOW_CHAIN_CAP
#                 )
# 
#             # Find nearest aliasing load, ensuring base isn't overwritten in between
#             bypass_idx = self._find_bypass_load(tokens, i, store_bases)
#             if bypass_idx is None:
#                 continue
#             counts["v4_bypass_load"] += 1
#             if (bypass_idx - i) <= self.V4_TIGHT_WINDOW:
#                 counts["v4_tight_bypass_load"] += 1
#                 total += self.V4_TIGHT_BYPASS_BONUS
# 
#             # Mild penalty if the agent explicitly smashes the store base between
#             # store and load (and still happens to alias by luck).
#             if self._has_explicit_base_reset(tokens, i, bypass_idx, store_bases):
#                 counts["v4_explicit_base_mask"] += 1
#                 total -= self.V4_EXPLICIT_MASK_PENALTY
# 
#             # Look for a transmitter: a memory access within a small window whose
#             # BASE register is a register written by the bypass load.
#             transmitter_idx = self._find_transmitter(tokens, bypass_idx)
#             if transmitter_idx is not None:
#                 if self._bypass_dst_clobbered_between(tokens, bypass_idx, transmitter_idx):
#                     counts["v4_bypass_dst_clobbered"] += 1
#                     total -= self.V4_BYPASS_DST_CLOBBER_PENALTY
#                     transmitter_idx = None
#                 else:
#                     counts["v4_transmitter"] += 1
#                     total += self.V4_TRANSMITTER_BONUS
#                     if (transmitter_idx - bypass_idx) <= self.V4_TIGHT_WINDOW:
#                         counts["v4_near_transmitter"] += 1
#                         total += self.V4_NEAR_TRANSMITTER_BONUS
# 
#             # Gate full_gadget on chain_len >= V4_MIN_CHAIN_LEN_FOR_FULL and
#             # scale bonus linearly to saturation at V4_CHAIN_LEN_SATURATION.
#             # Calibration (canonical v4 diagnostic):
#             #   chain_len ~= 26 -> ~2-4% fire rate (detected by fuzzer)
#             #   chain_len ~= 1  -> ~0% fire rate (never detected)
#             # So giving full_gadget bonus at chain_len<5 is a reward-hack trap.
#             if slow_producer and transmitter_idx is not None:
#                 if chain_len >= self.V4_MIN_CHAIN_LEN_FOR_FULL:
#                     full_gadgets += 1
#                     scale = min(chain_len, self.V4_CHAIN_LEN_SATURATION) \
#                         / float(self.V4_CHAIN_LEN_SATURATION)
#                     total += self.V4_FULL_GADGET_BONUS * scale
#                 else:
#                     # Structurally complete but chain too short to fire SSB.
#                     # Track separately so wandb shows the agent is getting
#                     # the shape right but needs to extend the chain.
#                     counts["v4_shortchain_gadget"] = counts.get(
#                         "v4_shortchain_gadget", 0
#                     ) + 1
# 
#         if full_gadgets > 0:
#             counts["v4_full_gadget"] = full_gadgets
#             # NOTE: full_gadget bonus is now applied per-store inside the
#             # loop (scaled by chain_len). Do NOT re-add the flat bonus here.
# 
#         # Prune zero-count shaping keys so logs are readable.
#         counts = {k: v for k, v in counts.items() if v}
#         return total, counts
# 
#     def _is_slow_op(self, opcode_variant: str) -> bool:
#         return (opcode_variant in self._LOAD_PRODUCER_OPS
#                 or opcode_variant in self._LATENCY_PRODUCER_OPS)
# 
#     def _last_base_writer(
#         self,
#         tokens: Sequence[PatternToken],
#         store_idx: int,
#         store_bases: FrozenSet[str],
#         window: int,
#     ) -> Optional[int]:
#         """
#         Return the index of the MOST RECENT user-instr within `window` tokens
#         before store_idx that writes any register in store_bases. Returns None
#         if no such writer exists. This is the token whose output will actually
#         flow into the store's base at runtime.
#         """
#         start = max(-1, store_idx - window - 1)
#         for k in range(store_idx - 1, start, -1):
#             if set(tokens[k].dst_regs).intersection(store_bases):
#                 return k
#         return None
# 
#     def _has_slow_addr_producer(
#         self,
#         tokens: Sequence[PatternToken],
#         store_idx: int,
#         store_bases: FrozenSet[str],
#     ) -> bool:
#         """
#         Last-writer semantics: return True only if the MOST RECENT writer of
#         any store-base register within the lookback window is itself a slow
#         producer. If a faster op (mov_ri / mov_rr / xor_rr, etc.) writes the
#         store base AFTER the slow producer, the slow-chain value never reaches
#         the store and the addr is known before the store issues — no SSB.
# 
#         This is stricter than the old "any slow write somewhere in lookback"
#         check and rules out reward-hacking patterns like
#             lea rsi, [rcx+rsi+1]   # slow
#             mov rsi, 7560          # smashes the chain
#             mov [r14+rsi], ...     # fast store, no speculation window
#         """
#         last_writer = self._last_base_writer(
#             tokens, store_idx, store_bases, self.SLOW_ADDR_LOOKBACK
#         )
#         if last_writer is None:
#             return False
#         return self._is_slow_op(tokens[last_writer].opcode_variant)
# 
#     def _has_near_slow_addr_producer(
#         self,
#         tokens: Sequence[PatternToken],
#         store_idx: int,
#         store_bases: FrozenSet[str],
#     ) -> bool:
#         """
#         Last-writer semantics within the tight adjacency window. Same rule
#         as _has_slow_addr_producer but restricted to V4_TIGHT_WINDOW tokens
#         before the store.
#         """
#         last_writer = self._last_base_writer(
#             tokens, store_idx, store_bases, self.V4_TIGHT_WINDOW
#         )
#         if last_writer is None:
#             return False
#         return self._is_slow_op(tokens[last_writer].opcode_variant)
# 
#     def _count_slow_chain_length(
#         self,
#         tokens: Sequence[PatternToken],
#         store_idx: int,
#         store_bases: FrozenSet[str],
#     ) -> int:
#         """
#         Walk BACKWARD from the store, counting consecutive slow-producer writes
#         to the store base. As soon as we see a token that writes the store base
#         but is NOT a slow producer, the chain is considered broken and we stop
#         — any earlier slow writes have been clobbered and don't reach the store.
# 
#         Tokens that don't touch store_bases are transparent (allowed to sit
#         between chain links), because they don't disturb the dataflow into
#         the store's addr-gen.
#         """
#         start = max(0, store_idx - self.SLOW_ADDR_LOOKBACK)
#         count = 0
#         for k in range(store_idx - 1, start - 1, -1):
#             t = tokens[k]
#             writes_base = bool(set(t.dst_regs).intersection(store_bases))
#             if not writes_base:
#                 continue
#             if self._is_slow_op(t.opcode_variant):
#                 count += 1
#                 continue
#             break
#         return count
# 
#     def _slow_chain_smashed(
#         self,
#         tokens: Sequence[PatternToken],
#         store_idx: int,
#         store_bases: FrozenSet[str],
#     ) -> bool:
#         """
#         Detect the reward-hack pattern: some slow producer in the lookback
#         wrote the store base, but a LATER fast-reset op (mov_ri / mov_rr /
#         xor_rr) clobbered it before the store. The matcher's chain-length
#         counter would previously still reward the earlier slow write; this
#         signal lets the caller apply V4_SLOW_CHAIN_SMASHED_PENALTY to
#         neutralize the farm-then-clobber strategy.
#         """
#         start = max(0, store_idx - self.SLOW_ADDR_LOOKBACK)
#         saw_slow = False
#         for k in range(start, store_idx):
#             t = tokens[k]
#             if not set(t.dst_regs).intersection(store_bases):
#                 continue
#             if self._is_slow_op(t.opcode_variant):
#                 saw_slow = True
#                 continue
#             # Non-slow write to a store base.
#             if saw_slow and t.opcode_variant in self._FAST_RESET_OPS:
#                 return True
#             # Any other non-slow writer also breaks the chain, but we only
#             # flag the farm-then-reset case (fast-reset ops) as "smashed"
#             # because those are the ones the agent uses to force a constant
#             # store address. Generic writes (e.g. add_ri) fall into the
#             # chain-truncation path via _count_slow_chain_length instead.
#         return False
# 
#     def _find_bypass_load(
#         self,
#         tokens: Sequence[PatternToken],
#         store_idx: int,
#         store_bases: FrozenSet[str],
#     ) -> Optional[int]:
#         max_next = min(len(tokens), store_idx + self.V4_FAST_LOAD_MAX_GAP + 1)
#         for j in range(store_idx + 1, max_next):
#             t = tokens[j]
#             if "load" not in t.kinds:
#                 continue
#             if not store_bases.intersection(t.mem_reads):
#                 continue
#             # Reject if any intervening user instruction overwrites the base reg.
#             if self._base_overwritten_between(tokens, store_idx, j):
#                 # record the overwrite event for diagnostics via a noop: the caller
#                 # will still count nothing for this pair.
#                 return None
#             return j
#         return None
# 
#     def _find_transmitter(
#         self, tokens: Sequence[PatternToken], bypass_idx: int
#     ) -> Optional[int]:
#         """
#         A transmitter is a memory access whose base register is a register
#         written by the bypass load (i.e., its dst_regs).  That base is exactly
#         the stale/speculated value read from the store buffer, so the
#         subsequent memory access depends on the secret value.
#         """
#         bypass_dst = set(tokens[bypass_idx].dst_regs)
#         if not bypass_dst:
#             return None
#         max_next = min(len(tokens), bypass_idx + self.V4_TRANSMITTER_MAX_GAP + 1)
#         for j in range(bypass_idx + 1, max_next):
#             t = tokens[j]
#             if "memory" not in t.kinds:
#                 continue
#             bases = set(t.mem_reads).union(t.mem_writes)
#             if bypass_dst.intersection(bases):
#                 return j
#         return None
# 
#     def _bypass_dst_clobbered_between(
#         self, tokens: Sequence[PatternToken], bypass_idx: int, transmitter_idx: int
#     ) -> bool:
#         """
#         If bypass-load destination is overwritten before transmitter, the stale value
#         is no longer used as address and v4 observability collapses.
#         """
#         if transmitter_idx - bypass_idx <= 1:
#             return False
#         bypass_dst = set(tokens[bypass_idx].dst_regs)
#         if not bypass_dst:
#             return False
#         for k in range(bypass_idx + 1, transmitter_idx):
#             if bypass_dst.intersection(tokens[k].dst_regs):
#                 return True
#         return False
# 
#     def _has_explicit_base_reset(
#         self,
#         tokens: Sequence[PatternToken],
#         store_idx: int,
#         load_idx: int,
#         store_bases: FrozenSet[str],
#     ) -> bool:
#         """Detect explicit resets like `mov base, imm` or `xor base, base` between store and load."""
#         for k in range(store_idx + 1, load_idx):
#             t = tokens[k]
#             if t.opcode_variant == "mov_ri" and set(t.dst_regs).intersection(store_bases):
#                 return True
#             if t.opcode_variant == "xor" and set(t.dst_regs).intersection(store_bases) \
#                     and set(t.src_regs).intersection(store_bases):
#                 # xor <base>, <base> zeroes the base
#                 return True
#         return False
