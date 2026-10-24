from pathlib import Path

import pytest

from data_leakage_detection.models import Action
from data_leakage_detection.policy import Policy, PolicyError, load_policy


def test_repo_policy_loads(policy: Policy) -> None:
    assert policy.action_for("AWS_ACCESS_KEY") is Action.BLOCK
    assert policy.action_for("EMAIL_ADDRESS") is Action.MASK
    assert policy.action_for("DATE_TIME") is Action.ALLOW


def test_unknown_entity_uses_default(policy: Policy) -> None:
    assert policy.action_for("SOMETHING_NEW") is policy.default_action


def test_actions_are_case_insensitive() -> None:
    p = Policy.from_dict({"entities": {"EMAIL_ADDRESS": "BLOCK"}})
    assert p.action_for("EMAIL_ADDRESS") is Action.BLOCK


def test_bad_action_raises() -> None:
    with pytest.raises(PolicyError):
        Policy.from_dict({"entities": {"EMAIL_ADDRESS": "shred"}})


def test_bad_min_score_raises() -> None:
    with pytest.raises(PolicyError):
        Policy.from_dict({"min_score": 2})


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PolicyError):
        load_policy(tmp_path / "nope.yaml")


def test_env_var_overrides_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "p.yaml"
    custom.write_text("default_action: block\n")
    monkeypatch.setenv("DLD_POLICY", str(custom))
    assert load_policy().default_action is Action.BLOCK
