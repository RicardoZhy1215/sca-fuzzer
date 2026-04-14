from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PatternToken:
    """Minimal token used by pattern templates."""

    index: int
    kinds: FrozenSet[str]
    mem_reads: Tuple[str, ...]
    mem_writes: Tuple[str, ...]
    opcode_variant: str


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
    """

    TEMPLATE_LIBRARY: Dict[str, List[PatternTemplate]] = {
        "spectre_v4": [
            # Core store-to-load forwarding candidate with likely alias.
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
                max_gap=6,
                base_reward=4.0,
                require_store_load_alias=False,
                max_matches=2,
            ),
        ]
    }

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
        return PatternMatchResult(score=total, matches=counts)

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

    @staticmethod
    def _has_store_load_alias(store_token: PatternToken, load_token: PatternToken) -> bool:
        if "store" not in store_token.kinds or "load" not in load_token.kinds:
            return False
        return bool(set(store_token.mem_writes).intersection(load_token.mem_reads))
