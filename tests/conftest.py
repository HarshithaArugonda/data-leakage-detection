from __future__ import annotations

import importlib.util
import os

import pytest

from data_leakage_detection.policy import Policy, load_policy


def fake(*parts: str) -> str:
    # build fake keys at runtime -- a literal "ghp_..." in the repo would trip
    # GitHub push protection / secret scanners
    return "".join(parts)


def _spacy_model_available() -> bool:
    model = os.environ.get("DLD_SPACY_MODEL", "en_core_web_lg")
    return importlib.util.find_spec(model) is not None


requires_presidio = pytest.mark.skipif(
    not _spacy_model_available(),
    reason="spaCy model for Presidio not installed",
)


@pytest.fixture(scope="session")
def policy() -> Policy:
    return load_policy()
