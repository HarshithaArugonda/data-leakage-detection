from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from data_leakage_detection.models import Action

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "policy.yaml"


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class PresidioConfig:
    enabled: bool = True
    spacy_model: str = "en_core_web_lg"
    score_threshold: float = 0.5


@dataclass(frozen=True)
class Policy:
    entity_actions: dict[str, Action] = field(default_factory=dict)
    default_action: Action = Action.MASK  # for types not listed in the yaml
    min_score: float = 0.5
    presidio: PresidioConfig = field(default_factory=PresidioConfig)

    def action_for(self, entity_type: str) -> Action:
        return self.entity_actions.get(entity_type, self.default_action)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Policy:
        if not isinstance(raw, dict):
            raise PolicyError("policy must be a mapping")
        entities = raw.get("entities") or {}
        if not isinstance(entities, dict):
            raise PolicyError("'entities' must be a mapping of ENTITY: action")
        try:
            actions = {str(k): Action(str(v).lower()) for k, v in entities.items()}
            default = Action(str(raw.get("default_action", "mask")).lower())
        except ValueError as exc:
            raise PolicyError(f"bad action in policy: {exc}") from exc

        min_score = float(raw.get("min_score", 0.5))
        if not 0.0 <= min_score <= 1.0:
            raise PolicyError("min_score must be between 0 and 1")

        p = raw.get("presidio") or {}
        presidio = PresidioConfig(
            enabled=bool(p.get("enabled", True)),
            spacy_model=str(p.get("spacy_model", "en_core_web_lg")),
            score_threshold=float(p.get("score_threshold", 0.5)),
        )
        return cls(actions, default, min_score, presidio)


def load_policy(path: str | Path | None = None) -> Policy:
    # explicit path > $DLD_POLICY > configs/policy.yaml
    policy_path = Path(path or os.environ.get("DLD_POLICY") or DEFAULT_POLICY_PATH)
    if not policy_path.exists():
        raise PolicyError(f"policy file not found: {policy_path}")
    with policy_path.open(encoding="utf-8") as fh:
        return Policy.from_dict(yaml.safe_load(fh) or {})
