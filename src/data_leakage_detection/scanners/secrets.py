# Regex rules for secrets/credentials.
# Prefix-based rules (AKIA, ghp_, sk-ant-...) are precise -> high score.
# The "password = ..." / "api_key = ..." rules are noisier -> lower score.
# Scores are only used to rank overlapping matches, they're not probabilities.
from __future__ import annotations

import re

from data_leakage_detection.scanners.base import RegexRule, RegexScanner

SECRET_RULES: list[RegexRule] = [
    # PEM private keys. END line optional so a truncated paste still matches
    RegexRule(
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----"
            r"(?:[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----)?"
        ),
        0.99,
    ),
    # AKIA = long-term key, ASIA = temporary (STS)
    RegexRule("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), 0.95),
    # classic tokens (ghp_, gho_, ...) + fine-grained PATs
    RegexRule(
        "GITHUB_TOKEN",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,})\b"),
        0.97,
    ),
    # has to come before the openai rule, both start with sk-
    RegexRule("ANTHROPIC_API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{32,}"), 0.97),
    # sk-, sk-proj-, sk-svcacct-. lookahead skips anthropic keys
    RegexRule(
        "OPENAI_API_KEY",
        re.compile(r"\bsk-(?!ant-)(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}"),
        0.9,
    ),
    RegexRule("SLACK_TOKEN", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), 0.95),
    RegexRule("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), 0.95),
    RegexRule("STRIPE_KEY", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), 0.97),
    # JWT: header.payload.sig, header and payload both start with eyJ ({" in base64)
    RegexRule(
        "JWT",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        0.9,
    ),
    RegexRule(
        "BEARER_TOKEN",
        re.compile(r"(?i)\bbearer\s+(?P<value>[A-Za-z0-9\-._~+/]{20,}=*)"),
        0.85,
    ),
    # password inside a connection string: postgres://user:PASSWORD@host
    RegexRule(
        "PASSWORD",
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:/@]+:(?P<value>[^\s@/]+)@"
        ),
        0.9,
    ),
    # password = x / pwd: x / "password is x"
    RegexRule(
        "PASSWORD",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|passcode)\b[\"']?\s*(?:[:=]|\bis\b)\s*[\"']?(?P<value>[^\s\"',;]{4,})"
        ),
        0.75,
    ),
    RegexRule(
        "GENERIC_SECRET",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|client[_-]?secret)\b"
            r"[\"']?\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9_\-./+=]{16,})"
        ),
        0.7,
    ),
]


def secrets_scanner() -> RegexScanner:
    return RegexScanner(SECRET_RULES, name="regex-secrets")
