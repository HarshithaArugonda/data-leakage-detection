# Nemotron label -> our entity type. Only clear 1:1 matches are mapped;
# anything else becomes None and isn't scored.
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
    return NEMOTRON_TO_DLD.get(label.strip().lower())
