"""Map dataset PII labels to this project's entity types.

Our policy and scanners use Presidio-style names (PERSON, EMAIL_ADDRESS, ...).
Nemotron-PII uses its own lowercase labels. Only labels with a clear 1:1
meaning get mapped. Everything else maps to None and is excluded from
scoring, so we never grade a scanner on a type it was not built to find.
"""

from __future__ import annotations

NEMOTRON_TO_DLD: dict[str, str] = {
    "first_name": "PERSON",
    "last_name": "PERSON",
    "email": "EMAIL_ADDRESS",
    "phone_number": "PHONE_NUMBER",
    "ssn": "US_SSN",
    "ipv4": "IP_ADDRESS",
    "password": "PASSWORD",
    "credit_debit_card": "CREDIT_CARD",
    "credit_card_number": "CREDIT_CARD",
    "account_number": "US_BANK_NUMBER",
    "street_address": "LOCATION",
    "city": "LOCATION",
    "county": "LOCATION",
    "state": "LOCATION",
    "country": "LOCATION",
    "url": "URL",
}


def map_nemotron_label(label: str) -> str | None:
    """Return our entity type for a Nemotron label, or None if unmapped."""
    return NEMOTRON_TO_DLD.get(label.strip().lower())
