# Runs all detection layers, resolves overlaps, applies the policy, redacts.
# regex_only() / presidio_only() reuse the same logic so baselines are comparable.
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
    # on overlap keep: stricter action first (block > mask > allow), then score,
    # then length. stricter-first so presidio tagging half an API key as PERSON
    # can't downgrade a block to a mask.

    def priority(f: Finding) -> tuple[int, float, int]:
        return (_SEVERITY[policy.action_for(f.entity_type)], f.score, f.length)

    kept: list[Finding] = []
    for f in sorted(findings, key=priority, reverse=True):
        if not any(f.overlaps(k) for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)


def redact(text: str, findings: Sequence[Finding], to_mask: Sequence[bool]) -> str:
    # same value -> same placeholder, so the LLM can still tell that two
    # mentions of a@x.com are the same person
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
    # go right to left so earlier offsets don't shift
    for f, label in sorted(zip(findings, labels, strict=True), key=lambda p: -p[0].start):
        if label is not None:
            out = out[: f.start] + label + out[f.end :]
    return out


class SensitiveDataScanner:
    def __init__(self, policy: Policy | None = None, layers: Sequence[Scanner] | None = None):
        self.policy = policy or load_policy()
        self.layers: list[Scanner] = list(layers) if layers is not None else self._default_layers(self.policy)

    @staticmethod
    def _default_layers(policy: Policy) -> list[Scanner]:
        layers: list[Scanner] = [RegexScanner(SECRET_RULES + PII_RULES, name="regex")]
        if policy.presidio.enabled:
            layers.append(presidio_from_policy(policy))
        return layers

    @classmethod
    def regex_only(cls, policy: Policy | None = None) -> SensitiveDataScanner:
        return cls(policy, [RegexScanner(SECRET_RULES + PII_RULES, name="regex")])

    @classmethod
    def presidio_only(cls, policy: Policy | None = None) -> SensitiveDataScanner:
        policy = policy or load_policy()
        return cls(policy, [presidio_from_policy(policy)])

    def find(self, text: str) -> list[Finding]:
        raw: list[Finding] = []
        for layer in self.layers:
            raw.extend(layer.analyze(text))
        raw = [f for f in raw if f.score >= self.policy.min_score]
        return resolve_overlaps(raw, self.policy)

    def scan(self, text: str) -> ScanResult:
        findings = self.find(text)
        actions = [self.policy.action_for(f.entity_type) for f in findings]

        if Action.BLOCK in actions:
            decision = Decision.BLOCK
        elif Action.MASK in actions:
            decision = Decision.REDACT
        else:
            decision = Decision.ALLOW

        # still redact on BLOCK so the raw secret never ends up in the audit log
        to_mask = [a is not Action.ALLOW for a in actions]
        redacted = redact(text, findings, to_mask) if decision is not Decision.ALLOW else text

        reasons = [
            f"{f.entity_type} found by {f.source} (score {f.score:.2f}) -> {a.value}"
            for f, a in zip(findings, actions, strict=True)
            if a is not Action.ALLOW
        ]
        return ScanResult(decision, redacted, findings, actions, reasons)


def presidio_from_policy(policy: Policy) -> PresidioScanner:
    model = os.environ.get("DLD_SPACY_MODEL") or policy.presidio.spacy_model
    return PresidioScanner(
        entities=list(policy.entity_actions),
        score_threshold=policy.presidio.score_threshold,
        spacy_model=model,
    )
