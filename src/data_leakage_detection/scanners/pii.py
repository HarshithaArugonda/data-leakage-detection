# Regex rules for fixed-format PII (email, phone, ssn, card, ip) + a Presidio
# wrapper for the stuff regex can't do (names, locations -> needs NER).
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
    # US/Canada: (214) 555-0123, 214-555-0123, +1 214 555 0123
    RegexRule(
        "PHONE_NUMBER",
        re.compile(
            r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[\s.-]?[2-9]\d{2}[\s.-]?\d{4}(?![\w])"
        ),
        0.7,
    ),
    # international with + prefix: +44 20 7946 0958
    RegexRule(
        "PHONE_NUMBER",
        re.compile(r"(?<![\w+])\+(?:[2-9]\d{0,2})[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)"),
        0.65,
    ),
    # 123-45-6789 or 123 45 6789 (\1 forces the same separator)
    RegexRule(
        "US_SSN",
        re.compile(r"(?<![\d-])\d{3}([- ])\d{2}\1\d{4}(?![\d-])"),
        0.85,
        validator=ssn_valid,
    ),
    # bare 9 digits only counts if "ssn"/"social security" is right before it,
    # otherwise every order number would match
    RegexRule(
        "US_SSN",
        re.compile(r"(?i)\b(?:ssn|social security(?: number| no\.?| #)?)\D{0,20}?(?P<value>\d{9})\b"),
        0.85,
        validator=ssn_valid,
    ),
    # 13-19 digits, spaces/dashes allowed, must pass luhn
    RegexRule(
        "CREDIT_CARD",
        re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])"),
        0.9,
        validator=luhn_valid,
    ),
    # same digits without luhn, but only right after a card word. lower score.
    # (nemotron's fake cards mostly fail luhn -- only 56/443 in val pass)
    RegexRule(
        "CREDIT_CARD",
        re.compile(
            r"(?i)\b(?:card|visa|mastercard|amex|credit|debit|cc)\b[^\d\n]{0,25}"
            r"(?P<value>\d(?:[ -]?\d){12,18})(?![\d-])"
        ),
        0.75,
    ),
    # bank account numbers have no fixed format, so require "account number" etc. before it
    RegexRule(
        "US_BANK_NUMBER",
        re.compile(
            r"(?i)\b(?:account|acct|a/c)(?:\s*(?:number|no\.?|num|#))?[^\w\n]{0,4}(?:(?:is|was)\s+)?"
            r"(?P<value>[A-Z]{0,3}\d[\d-]{5,20}\d)\b"
        ),
        0.8,
    ),
    RegexRule(
        "IP_ADDRESS",
        re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
        0.6,
        validator=_ipv4_valid,
    ),
]


def pii_regex_scanner() -> RegexScanner:
    return RegexScanner(PII_RULES, name="regex-pii")


DEFAULT_SPACY_MODEL = "en_core_web_lg"


@lru_cache(maxsize=4)
def _build_analyzer(spacy_model: str) -> AnalyzerEngine:
    # cached: loading the spacy model takes a few seconds
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
        # presidio errors on unknown entity names, so drop ours (AWS_ACCESS_KEY etc.)
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
