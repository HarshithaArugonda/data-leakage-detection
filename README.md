# data-leakage-detection

Guardrails for LLM agents: catches PII / secrets going into or out of the model, prompt injection, and unsafe tool calls. Work in progress.

## Setup

```bash
uv sync
uv run pytest -q
```

## Usage

```python
from data_leakage_detection.scanners import SensitiveDataScanner

scanner = SensitiveDataScanner()  # uses configs/policy.yaml
r = scanner.scan("I'm Priya, email priya@acme.io")
r.decision       # redact
r.redacted_text  # "I'm <PERSON_1>, email <EMAIL_ADDRESS_1>"
```

The scanner runs regex rules (secrets, emails, phones, SSNs, cards, IPs) and Presidio (names, locations), then applies the actions in `configs/policy.yaml`.

## Data

```bash
uv run python -m eval.prepare_data   # -> data/processed/
uv run python -m eval.pii_baseline   # -> results/
```

Datasets:
- [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) (CC-BY-4.0): PII spans
- [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) (Apache-2.0)
- [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) (MIT)
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC-BY-SA-3.0): benign prompts

Data isn't committed. `prepare_data` pins each dataset to a commit and writes file hashes to `data/processed/manifest.json`.
