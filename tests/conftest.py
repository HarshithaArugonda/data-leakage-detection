"""Shared test helpers.

Fake secrets are built by joining pieces at runtime. If a test file had a
literal token like "ghp_..." in it, GitHub push protection could reject the
push, and other secret scanners would flag the repo. Joining strings keeps
the pattern out of the source while the test still sees the full token.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from data_leakage_detection.policy import Policy, load_policy


def fake(*parts: str) -> str:
    """Join parts into a fake secret so the literal never appears in source."""
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
