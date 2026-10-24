"""Layer 1a: regex rules for secrets and credentials.

Most real secrets have a fixed, documented prefix (AKIA for AWS, ghp_ for
GitHub, sk-ant- for Anthropic). Prefix rules are precise, so they get high
scores. The two "assignment" rules (password = ..., api_key = ...) catch
secrets without a known prefix; they are noisier, so they score lower.

Scores are fixed per rule. They are ranking hints for overlap resolution,
not calibrated probabilities.
"""

from __future__ import annotations

import re

from data_leakage_detection.scanners.base import RegexRule, RegexScanner

SECRET_RULES: list[RegexRule] = [
    # Private key blocks (PEM). END line is optional so truncated pastes still match.
    RegexRule(
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----"
            r"(?:[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----)?"
        ),
        0.99,
    ),
    # AWS access key ID (long-term AKIA, temporary ASIA).
    RegexRule("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), 0.95),
    # GitHub classic tokens (ghp_, gho_, ghu_, ghs_, ghr_) and fine-grained PATs.
    RegexRule(
        "GITHUB_TOKEN",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,})\b"),
        0.97,
    ),
    # Anthropic keys. Checked before OpenAI because both start with "sk-".
    RegexRule("ANTHROPIC_API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{32,}"), 0.97),
    # OpenAI keys: sk-..., sk-proj-..., sk-svcacct-... The lookahead skips sk-ant-.
    RegexRule(
        "OPENAI_API_KEY",
        re.compile(r"\bsk-(?!ant-)(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}"),
        0.9,
    ),
    # Slack bot/user/app tokens.
    RegexRule("SLACK_TOKEN", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), 0.95),
    # Google API keys.
    RegexRule("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), 0.95),
    # Stripe secret and restricted keys.
    RegexRule("STRIPE_KEY", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), 0.97),
    # JSON Web Tokens: three base64url parts, first two start with eyJ ('{"').
    RegexRule(
        "JWT",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        0.9,
    ),
    # Authorization: Bearer <token>
    RegexRule(
        "BEARER_TOKEN",
        re.compile(r"(?i)\bbearer\s+(?P<value>[A-Za-z0-9\-._~+/]{20,}=*)"),
        0.85,
    ),
    # Credentials inside a database/broker URI: scheme://user:PASSWORD@host
    RegexRule(
        "PASSWORD",
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:/@]+:(?P<value>[^\s@/]+)@"
        ),
        0.9,
    ),
    # password = ..., pwd: ..., "password is ..."
    RegexRule(
        "PASSWORD",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|passcode)\b[\"']?\s*(?:[:=]|\bis\b)\s*[\"']?(?P<value>[^\s\"',;]{4,})"
        ),
        0.75,
    ),
    # api_key = ..., client_secret: ..., access_token = ...
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
    """Regex scanner with only the secret/credential rules."""
    return RegexScanner(SECRET_RULES, name="regex-secrets")
