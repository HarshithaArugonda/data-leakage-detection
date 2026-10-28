import pytest

from data_leakage_detection.scanners.pii import pii_regex_scanner

SCANNER = pii_regex_scanner()


def found(text: str) -> list[tuple[str, str]]:
    return [(f.entity_type, f.text) for f in SCANNER.analyze(text)]


@pytest.mark.parametrize(
    ("text", "entity", "value"),
    [
        ("mail me at jane.doe+work@mail.example.co.uk", "EMAIL_ADDRESS", "jane.doe+work@mail.example.co.uk"),
        ("call (214) 555-0182 today", "PHONE_NUMBER", "(214) 555-0182"),
        ("cell: 214-555-0182", "PHONE_NUMBER", "214-555-0182"),
        ("cell: +1 214 555 0182", "PHONE_NUMBER", "+1 214 555 0182"),
        ("london office +44 20 7946 0958", "PHONE_NUMBER", "+44 20 7946 0958"),
        ("SSN: 123-45-6788", "US_SSN", "123-45-6788"),
        ("my social security number is 501234567", "US_SSN", "501234567"),
        ("card 4111 1111 1111 1111 exp 12/29", "CREDIT_CARD", "4111 1111 1111 1111"),
        ("amex 378282246310005", "CREDIT_CARD", "378282246310005"),
        ("server at 192.168.1.20", "IP_ADDRESS", "192.168.1.20"),
        # non-luhn card number is ok when a card word is right before it
        ("Credit/Debit Card Number: 4929 3756 1245 6789", "CREDIT_CARD", "4929 3756 1245 6789"),
        ("My account number is 982375612489.", "US_BANK_NUMBER", "982375612489"),
        ("**Account Number:** 527-834-1692", "US_BANK_NUMBER", "527-834-1692"),
        ("the account number C752198463.", "US_BANK_NUMBER", "C752198463"),
        ("Bank Account: 5983214760", "US_BANK_NUMBER", "5983214760"),
    ],
)
def test_detects_pii(text: str, entity: str, value: str) -> None:
    assert (entity, value) in found(text)


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("order 4111111111111112", "CREDIT_CARD"),  # fails Luhn
        ("SSN: 000-12-3456", "US_SSN"),  # invalid area
        ("tracking 501234567", "US_SSN"),  # 9 digits, no SSN context
        ("version 999.1.2.3", "IP_ADDRESS"),  # octet > 255
        ("call 123-555-0182", "PHONE_NUMBER"),  # area code can't start with 1
        ("dates 2026-10-22", "PHONE_NUMBER"),
        ("tracking number 1234567890123456", "CREDIT_CARD"),  # no luhn, no card word
        ("the account is 3 days old", "US_BANK_NUMBER"),
        ("open account settings", "US_BANK_NUMBER"),
    ],
)
def test_rejects_lookalikes(text: str, entity: str) -> None:
    assert entity not in [e for e, _ in found(text)]
