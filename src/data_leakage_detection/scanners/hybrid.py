"""Combine detection layers and apply the policy.

Pipeline for one piece of text:
    1. Run every layer (regex secrets, regex PII, Presidio, later the model).
    2. Drop findings below policy.min_score.
    3. Resolve overlaps so each character belongs to at most one finding.
    4. Look up each finding's action in the policy.
    5. Decide: any BLOCK -> block; else any MASK -> redact; else allow.
    6. Build redacted text with stable placeholders (<EMAIL_ADDRESS_1>).

The same class gives us the baselines for free: pass only the regex layers
for "regex-only", only Presidio for "Presidio-only", or all of them for the
hybrid. Same policy, same overlap logic, so the comparison is fair.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from data_leakage_detection.models import Action, Decision, Finding, ScanResult
from data_leakage_detection.policy import Policy, load_policy
from data_leakage_detection.scanners.base import RegexScanner, Scanner
from data_leakage_detection.scanners.pii import PII_RULES, PresidioScanner
from data_leakage_detection.scanners.secrets import SECRET_RULES

_SEVERITY = {Action.BLOCK: 2, Action.MASK: 1, Action.ALLOW: 0}


def resolve_overlaps(findings: Sequence[Finding], policy: Policy) -> list[Finding]:
    """Keep a non-overlapping subset of findings.

    When two findings overlap, keep the one that is stricter under the policy
    (block > mask > allow), then higher score, then longer span. Choosing the
    stricter one first means a secret is never hidden behind a weaker label,
    e.g. Presidio calling part of an API key a PERSON.
    """

    def priority(f: Finding) -> tuple[int, float, int]:
        return (_SEVERITY[policy.action_for(f.entity_type)], f.score, f.length)

    kept: list[Finding] = []
    for f in sorted(findings, key=priority, reverse=True):
        if not any(f.overlaps(k) for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)


def redact(text: str, findings: Sequence[Finding], to_mask: Sequence[bool]) -> str:
    """Replace selected findings with numbered placeholders.

    The same value always gets the same placeholder within one text, so
    "email a@x.com, then cc a@x.com" becomes "<EMAIL_ADDRESS_1> ... <EMAIL_ADDRESS_1>".
    The LLM can still tell the two mentions refer to the same thing.
    """
    ids: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    labels: list[str | None] = []
    for f, masked in zip(findings, to_mask, strict=True):
        if not masked:
            labels.append(None)
            continue
        key = (f.entity_type, f.text)
        if key not in ids:
            counters[f.entity_type] = counters.get(f.entity_type, 0) + 1
            ids[key] = f"<{f.entity_type}_{counters[f.entity_type]}>"
        labels.append(ids[key])

    out = text
    # Replace right-to-left so earlier offsets stay valid.
    for f, label in sorted(zip(findings, labels, strict=True), key=lambda p: -p[0].start):
        if label is not None:
            out = out[: f.start] + label + out[f.end :]
    return out


class SensitiveDataScanner:
    """Runs several detection layers and applies one policy.

    Args:
        policy: Validated policy. Loaded from configs/policy.yaml if omitted.
        layers: Scanners to run. If omitted, built from the policy:
            regex secrets + regex PII + Presidio (if enabled).
    """

    def __init__(self, policy: Policy | None = None, layers: Sequence[Scanner] | None = None):
        self.policy = policy or load_policy()
        self.layers: list[Scanner] = list(layers) if layers is not None else self._default_layers(self.policy)

    @staticmethod
    def _default_layers(policy: Policy) -> list[Scanner]:
        layers: list[Scanner] = [RegexScanner(SECRET_RULES + PII_RULES, name="regex")]
        if policy.presidio.enabled:
            layers.append(_presidio_from_policy(policy))
        return layers

    @classmethod
    def regex_only(cls, policy: Policy | None = None) -> SensitiveDataScanner:
        """Baseline: rules only, no NER."""
        return cls(policy, [RegexScanner(SECRET_RULES + PII_RULES, name="regex")])

    @classmethod
    def presidio_only(cls, policy: Policy | None = None) -> SensitiveDataScanner:
        """Baseline: stock Presidio, none of our rules."""
        policy = policy or load_policy()
        return cls(policy, [_presidio_from_policy(policy)])

    def find(self, text: str) -> list[Finding]:
        """All findings after score filtering and overlap resolution."""
        raw: list[Finding] = []
        for layer in self.layers:
            raw.extend(layer.analyze(text))
        raw = [f for f in raw if f.score >= self.policy.min_score]
        return resolve_overlaps(raw, self.policy)

    def scan(self, text: str) -> ScanResult:
        """Scan `text` and return the decision plus redacted text."""
        findings = self.find(text)
        actions = [self.policy.action_for(f.entity_type) for f in findings]

        if Action.BLOCK in actions:
            decision = Decision.BLOCK
        elif Action.MASK in actions:
            decision = Decision.REDACT
        else:
            decision = Decision.ALLOW

        # On BLOCK we still mask block+mask spans so the audit log never
        # stores the raw secret.
        to_mask = [a is not Action.ALLOW for a in actions]
        redacted = redact(text, findings, to_mask) if decision is not Decision.ALLOW else text

        reasons = [
            f"{f.entity_type} found by {f.source} (score {f.score:.2f}) -> {a.value}"
            for f, a in zip(findings, actions, strict=True)
            if a is not Action.ALLOW
        ]
        return ScanResult(decision, redacted, findings, actions, reasons)


def _presidio_from_policy(policy: Policy) -> PresidioScanner:
    model = os.environ.get("DLD_SPACY_MODEL") or policy.presidio.spacy_model
    # Ask Presidio only for entity types the policy names; it skips any it
    # does not support (our secret types).
    return PresidioScanner(
        entities=list(policy.entity_actions),
        score_threshold=policy.presidio.score_threshold,
        spacy_model=model,
    )
