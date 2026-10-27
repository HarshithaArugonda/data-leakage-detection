from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from data_leakage_detection.models import Finding


class Scanner(Protocol):
    """Common interface for every detection layer (regex, presidio, model later)."""

    name: str

    def analyze(self, text: str) -> list[Finding]: ...


@dataclass(frozen=True)
class RegexRule:
    entity_type: str
    pattern: re.Pattern[str]
    score: float
    # extra check on the matched value, e.g. luhn for cards
    validator: Callable[[str], bool] | None = None

    def find(self, text: str) -> Iterable[Finding]:
        for m in self.pattern.finditer(text):
            # if the pattern has a "value" group, report only that part
            # (so "password = hunter2" flags "hunter2", not the whole thing)
            if "value" in self.pattern.groupindex:
                start, end = m.span("value")
            else:
                start, end = m.span()
            value = text[start:end]
            if not value:
                continue
            if self.validator is not None and not self.validator(value):
                continue
            yield Finding(self.entity_type, start, end, value, self.score, "regex")


class RegexScanner:
    def __init__(self, rules: Iterable[RegexRule], name: str = "regex") -> None:
        self.rules = list(rules)
        self.name = name

    def analyze(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        for rule in self.rules:
            findings.extend(rule.find(text))
        return sorted(findings, key=lambda f: (f.start, -f.length))
