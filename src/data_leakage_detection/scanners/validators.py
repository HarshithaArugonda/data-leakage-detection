"""Checksum and format validators used to cut regex false positives.

A 16-digit number is not automatically a card number, and a 9-digit number is
not automatically an SSN. These checks reject matches that cannot be real.
"""

from __future__ import annotations

import re

_NON_DIGIT = re.compile(r"\D")


def digits_only(value: str) -> str:
    """Strip spaces, dashes and anything else that is not a digit."""
    return _NON_DIGIT.sub("", value)


def luhn_valid(number: str) -> bool:
    """Return True if `number` passes the Luhn checksum used by payment cards.

    Walk the digits right to left, double every second digit, subtract 9 from
    any doubled value above 9, and sum. Valid numbers sum to a multiple of 10.
    About 90% of random digit strings fail, which is the point.
    """
    digits = digits_only(number)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def ssn_valid(value: str) -> bool:
    """Return True if `value` is a structurally valid US SSN.

    Rules from the SSA: area (first 3) is not 000, 666, or 900-999; group
    (middle 2) is not 00; serial (last 4) is not 0000. Also rejects a few
    well-known fake/advertising numbers.
    """
    digits = digits_only(value)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area.startswith("9"):
        return False
    if group == "00" or serial == "0000":
        return False
    return digits not in {"078051120", "219099999", "123456789"}
