"""Core data types shared by every scanner.

A scanner produces `Finding`s (what it saw and where). The policy turns each
finding into an `Action`. The overall result for a piece of text is a
`Decision` plus the redacted text, wrapped in a `ScanResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Action(StrEnum):
    """What to do with one detected entity."""

    ALLOW = "allow"  # leave it in the text
    MASK = "mask"  # replace it with a placeholder like <EMAIL_ADDRESS_1>
    BLOCK = "block"  # refuse the whole request/response


class Decision(StrEnum):
    """Final verdict for a whole piece of text."""

    ALLOW = "allow"  # nothing to change
    REDACT = "redact"  # pass through, but with masked spans
    BLOCK = "block"  # do not pass through at all


@dataclass(frozen=True)
class Finding:
    """One sensitive span found in the text.

    Attributes:
        entity_type: Label such as "EMAIL_ADDRESS" or "AWS_ACCESS_KEY".
        start: Start character offset (inclusive).
        end: End character offset (exclusive).
        text: The matched substring, text[start:end].
        score: Confidence in [0, 1]. Regex rules use fixed scores.
        source: Which layer found it: "regex" or "presidio".
    """

    entity_type: str
    start: int
    end: int
    text: str
    score: float
    source: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Finding) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class ScanResult:
    """Output of scanning one piece of text.

    Attributes:
        decision: allow / redact / block for the whole text.
        redacted_text: Text with MASK entities replaced. Equals the input on ALLOW.
            On BLOCK it is still filled in so it can be logged safely.
        findings: Every finding kept after overlap resolution.
        actions: Action chosen for each finding (same order as `findings`).
        reasons: Human-readable reasons, one per non-ALLOW finding.
    """

    decision: Decision
    redacted_text: str
    findings: list[Finding] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form for the audit log. Never includes raw matched text."""
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
