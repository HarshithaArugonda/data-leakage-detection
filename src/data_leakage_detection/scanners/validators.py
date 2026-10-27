from __future__ import annotations

import re

_NON_DIGIT = re.compile(r"\D")


def digits_only(value: str) -> str:
    return _NON_DIGIT.sub("", value)


def luhn_valid(number: str) -> bool:
    # standard luhn: from the right, double every 2nd digit (minus 9 if > 9),
    # sum must be divisible by 10. ~90% of random numbers fail.
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
    # SSA rules: area != 000/666/9xx, group != 00, serial != 0000
    digits = digits_only(value)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area.startswith("9"):
        return False
    if group == "00" or serial == "0000":
        return False
    # well-known fake / sample SSNs
    return digits not in {"078051120", "219099999", "123456789"}
