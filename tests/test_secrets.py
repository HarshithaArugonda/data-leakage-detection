import pytest

from data_leakage_detection.scanners.secrets import secrets_scanner
from tests.conftest import fake

SCANNER = secrets_scanner()


def types_in(text: str) -> list[str]:
    return [f.entity_type for f in SCANNER.analyze(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (fake("key: ", "AKIA", "Z7QK3M9P2XWB4N6T"), "AWS_ACCESS_KEY"),
        (fake("token ", "ghp_", "a" * 36), "GITHUB_TOKEN"),
        (fake("github_pat_", "11ABCDEFG", "x" * 70), "GITHUB_TOKEN"),
        (fake("sk-ant-", "api03-", "Ab3_" * 12), "ANTHROPIC_API_KEY"),
        (fake("OPENAI_KEY=", "sk-proj-", "Zx9" * 10), "OPENAI_API_KEY"),
        (fake("xoxb-", "1234567890-abcdefghij"), "SLACK_TOKEN"),
        (fake("AIza", "S" * 35), "GOOGLE_API_KEY"),
        (fake("sk_", "live_", "4eC39HqLyjWDarjtT1zdp7dc"), "STRIPE_KEY"),
        (fake("eyJ", "hbGciOiJIUzI1NiJ9", ".", "eyJ", "zdWIiOiIxMjM0NSJ9", ".", "sig_abc123XYZ"), "JWT"),
        (
            fake("-----BEGIN RSA ", "PRIVATE KEY-----\nMIIEow\n-----END RSA ", "PRIVATE KEY-----"),
            "PRIVATE_KEY",
        ),
        (fake("-----BEGIN OPENSSH ", "PRIVATE KEY-----\nb3BlbnNzaC1r"), "PRIVATE_KEY"),
        ("password = Hunter2!x", "PASSWORD"),
        ('"pwd": "s3cr3tPass"', "PASSWORD"),
        ("my password is correcthorse", "PASSWORD"),
        ("postgres://admin:Sup3rS3cret@db.internal:5432/app", "PASSWORD"),
        ('client_secret: "a1b2c3d4e5f6g7h8i9j0"', "GENERIC_SECRET"),
    ],
)
def test_detects_secret(text: str, expected: str) -> None:
    assert expected in types_in(text)


def test_password_rule_reports_only_the_value() -> None:
    [finding] = SCANNER.analyze("password = Hunter2!x")
    assert finding.text == "Hunter2!x"


def test_bearer_token_reports_only_the_token() -> None:
    token = fake("abc123", "DEF456", "ghi789", "JKL")
    findings = SCANNER.analyze(f"Authorization: Bearer {token}")
    assert any(f.entity_type == "BEARER_TOKEN" and f.text == token for f in findings)


def test_anthropic_key_is_not_labelled_openai() -> None:
    types = types_in(fake("sk-ant-", "api03-", "Ab3_" * 12))
    assert "OPENAI_API_KEY" not in types


@pytest.mark.parametrize(
    "text",
    [
        "Please reset my password tomorrow.",  # word "password" with no value
        "The task-ant-colony algorithm is neat.",  # looks like sk-ant- but isn't
        "AKIA is a prefix used by AWS.",  # prefix alone
        "Use a bearer of good news.",  # 'bearer' without a token
        "See section sk-1 of the doc.",  # too short for an OpenAI key
    ],
)
def test_no_false_positive_on_benign_text(text: str) -> None:
    assert types_in(text) == []
