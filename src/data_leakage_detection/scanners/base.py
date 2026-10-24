"""Scanner interface and the generic regex scanner.

Every detection layer (regex, Presidio, later the fine-tuned model) implements
the same `Scanner` protocol: take text, return a list of `Finding`s. That lets
the hybrid scanner and the eval harness treat layers interchangeably, which is
what makes the ablation table (rules -> +Presidio -> +model) possible.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from data_leakage_detection.models import Finding


class Scanner(Protocol):
    """Anything that can find sensitive spans in text."""

    name: str

    def analyze(self, text: str) -> list[Finding]:
        """Return all findings in `text`. Overlaps are allowed here."""
        ...


@dataclass(frozen=True)
class RegexRule:
    """One regex detector.

    Attributes:
        entity_type: Label to emit, e.g. "AWS_ACCESS_KEY".
        pattern: Compiled regex. If it has a group named "value", only that
            group is reported (so `password = hunter2` reports just `hunter2`).
        score: Fixed confidence for matches of this rule.
        validator: Optional function that must return True on the reported
            value, e.g. a Luhn check. Used to drop false positives.
    """

    entity_type: str
    pattern: re.Pattern[str]
    score: float
    validator: Callable[[str], bool] | None = None

    def find(self, text: str) -> Iterable[Finding]:
        for m in self.pattern.finditer(text):
            if "value" in self.pattern.groupindex:
                start, end = m.span("value")
            else:
                start, end = m.span()
            value = text[start:end]
            if not value:
                continue
            if self.validator is not None and not self.validator(value):
                continue
            yield Finding(
                entity_type=self.entity_type,
                start=start,
                end=end,
                text=value,
                score=self.score,
                source="regex",
            )


class RegexScanner:
    """Runs a list of `RegexRule`s over the text."""

    def __init__(self, rules: Iterable[RegexRule], name: str = "regex") -> None:
        self.rules = list(rules)
        self.name = name

    def analyze(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        for rule in self.rules:
            findings.extend(rule.find(text))
        return sorted(findings, key=lambda f: (f.start, -f.length))
