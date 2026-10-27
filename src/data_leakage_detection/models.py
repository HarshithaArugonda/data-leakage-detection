from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Action(StrEnum):
    """What to do with a single detected entity."""

    ALLOW = "allow"
    MASK = "mask"  # replace with <TYPE_N>
    BLOCK = "block"  # reject the whole message


class Decision(StrEnum):
    """Verdict for the whole message."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass(frozen=True)
class Finding:
    entity_type: str
    start: int
    end: int  # exclusive
    text: str
    score: float
    source: str  # "regex" / "presidio"

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Finding) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class ScanResult:
    decision: Decision
    redacted_text: str  # also filled on BLOCK so it's safe to log
    findings: list[Finding] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)  # one per finding
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        # for the audit log -- deliberately leaves out the matched text
        return {
            "decision": self.decision.value,
            "redacted_text": self.redacted_text,
            "findings": [
                {
                    "entity_type": f.entity_type,
                    "start": f.start,
                    "end": f.end,
                    "score": round(f.score, 4),
                    "source": f.source,
                    "action": a.value,
                }
                for f, a in zip(self.findings, self.actions, strict=True)
            ],
            "reasons": self.reasons,
        }
