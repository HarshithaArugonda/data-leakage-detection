import pytest

from data_leakage_detection.scanners.validators import digits_only, luhn_valid, ssn_valid


def test_digits_only_strips_separators() -> None:
    assert digits_only("4111-1111 1111.1111") == "4111111111111111"


@pytest.mark.parametrize(
    "number",
    [
        "4111111111111111",  # Visa test number
        "4111 1111 1111 1111",
        "5500-0000-0000-0004",  # Mastercard test number
        "378282246310005",  # Amex test number (15 digits)
    ],
)
def test_luhn_accepts_valid_cards(number: str) -> None:
    assert luhn_valid(number)


@pytest.mark.parametrize(
    "number",
    [
        "4111111111111112",  # last digit changed
        "1234567890123",  # random 13 digits
        "411111111111",  # too short (12)
        "41111111111111111111",  # too long (20)
    ],
)
def test_luhn_rejects_invalid(number: str) -> None:
    assert not luhn_valid(number)


@pytest.mark.parametrize("ssn", ["123-45-6788", "212 67 4321", "501234567"])
def test_ssn_accepts_valid(ssn: str) -> None:
    assert ssn_valid(ssn)


@pytest.mark.parametrize(
    "ssn",
    [
        "000-12-3456",  # area 000
        "666-12-3456",  # area 666
        "912-12-3456",  # area 9xx
        "123-00-4567",  # group 00
        "123-45-0000",  # serial 0000
        "078-05-1120",  # famous wallet-card SSN
        "123-45-678",  # wrong length
    ],
)
def test_ssn_rejects_invalid(ssn: str) -> None:
    assert not ssn_valid(ssn)
