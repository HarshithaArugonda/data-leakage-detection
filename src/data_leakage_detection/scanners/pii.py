"""Layer 1b (regex PII rules) and Layer 2 (Microsoft Presidio).

Regex handles PII with a rigid format: emails, phone numbers, SSNs, card
numbers, IP addresses. Validators (Luhn, SSN rules) remove most junk matches.

Presidio adds what regex cannot do: names, locations and other entities that
need an NER model (spaCy under the hood), plus its own pattern recognizers
with context-word boosting ("my SSN is ..." raises the score).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

from data_leakage_detection.models import Finding
from data_leakage_detection.scanners.base import RegexRule, RegexScanner
from data_leakage_detection.scanners.validators import luhn_valid, ssn_valid

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine


def _ipv4_valid(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 for p in parts)


PII_RULES: list[RegexRule] = [
    RegexRule(
        "EMAIL_ADDRESS",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
        0.95,
    ),
    # North American numbers: (214) 555-0123, 214-555-0123, +1 214 555 0123
    RegexRule(
        "PHONE_NUMBER",
        re.compile(
            r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[\s.-]?[2-9]\d{2}[\s.-]?\d{4}(?![\w])"
        ),
        0.7,
    ),
    # International numbers that start with + and a country code: +44 20 7946 0958
    RegexRule(
        "PHONE_NUMBER",
        re.compile(r"(?<![\w+])\+(?:[2-9]\d{0,2})[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)"),
        0.65,
    ),
    # SSN with separators: 123-45-6789 or 123 45 6789 (same separator both times).
    RegexRule(
        "US_SSN",
        re.compile(r"(?<![\d-])\d{3}([- ])\d{2}\1\d{4}(?![\d-])"),
        0.85,
        validator=ssn_valid,
    ),
    # Bare 9 digits only when "SSN" / "social security" is nearby, to avoid
    # flagging every 9-digit order number.
    RegexRule(
        "US_SSN",
        re.compile(r"(?i)\b(?:ssn|social security(?: number| no\.?| #)?)\D{0,20}?(?P<value>\d{9})\b"),
        0.85,
        validator=ssn_valid,
    ),
    # 13-19 digits, optionally split by spaces or dashes, must pass Luhn.
    RegexRule(
        "CREDIT_CARD",
        re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])"),
        0.9,
        validator=luhn_valid,
    ),
    RegexRule(
        "IP_ADDRESS",
        re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
        0.6,
        validator=_ipv4_valid,
    ),
]


def pii_regex_scanner() -> RegexScanner:
    """Regex scanner with only the structured-PII rules."""
    return RegexScanner(PII_RULES, name="regex-pii")


DEFAULT_SPACY_MODEL = "en_core_web_lg"


@lru_cache(maxsize=4)
def _build_analyzer(spacy_model: str) -> AnalyzerEngine:
    """Build a Presidio AnalyzerEngine once per spaCy model (loading is slow)."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }
    )
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


class PresidioScanner:
    """Wraps Presidio's AnalyzerEngine behind the `Scanner` protocol.

    Args:
        entities: Presidio entity names to look for. Entities Presidio does
            not support (e.g. our secret types) are silently skipped.
        score_threshold: Drop Presidio results below this score.
        spacy_model: spaCy pipeline. Defaults to $DLD_SPACY_MODEL, then
            en_core_web_lg.
    """

    def __init__(
        self,
        entities: Iterable[str] | None = None,
        score_threshold: float = 0.5,
        spacy_model: str | None = None,
    ) -> None:
        self.name = "presidio"
        self.spacy_model = spacy_model or os.environ.get("DLD_SPACY_MODEL", DEFAULT_SPACY_MODEL)
        self.score_threshold = score_threshold
        self._analyzer = _build_analyzer(self.spacy_model)
        supported = set(self._analyzer.get_supported_entities(language="en"))
        wanted = set(entities) if entities is not None else supported
        self.entities = sorted(wanted & supported)

    def analyze(self, text: str) -> list[Finding]:
        if not text.strip() or not self.entities:
            return []
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self.entities,
            score_threshold=self.score_threshold,
        )
        findings = [
            Finding(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                text=text[r.start : r.end],
                score=float(r.score),
                source="presidio",
            )
            for r in results
        ]
        return sorted(findings, key=lambda f: (f.start, -f.length))
