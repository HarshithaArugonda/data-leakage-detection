# Data Leakage Detection

Data leakage detection for LLM agents. It sits between users, an agent, and the agent's tools, and blocks or redacts sensitive data, prompt injections, hidden-context leaks, and unsafe tool calls. Targets OWASP Top 10 for LLM Applications: LLM01 (Prompt Injection) and LLM02 (Sensitive Information Disclosure).

> Work in progress. Results tables are added only from real runs of `python -m eval.run_all`.

## Status

| Component | Status |
|---|---|
| PII / secret scanner (regex + Presidio) | done |
| Dataset prep (download + splits) | done |
| Regex / Presidio baselines | next |
| LoRA PII + injection classifiers | planned |
| Tool-call policy proxy, canaries, output scanner | planned |
| API, Docker, demo, MCP server | planned |

## Quickstart

```bash
uv sync          # installs deps, including the spaCy en_core_web_lg model
uv run pytest -q
```

```python
from data_leakage_detection.scanners import SensitiveDataScanner

scanner = SensitiveDataScanner()  # loads configs/policy.yaml
result = scanner.scan("I'm Priya, email priya@acme.io")
result.decision  # Decision.REDACT
result.redacted_text  # "I'm <PERSON_1>, email <EMAIL_ADDRESS_1>"
result.reasons  # why each span was masked
```

## How the sensitive-data scanner works

1. **Regex rules** (`scanners/secrets.py`, `scanners/pii.py`): known secret formats (AWS, GitHub, OpenAI, Anthropic, Slack, Google, Stripe, JWT, PEM keys, passwords) and structured PII (email, phone, SSN, card, IP). Card numbers must pass Luhn; SSNs must pass SSA structure rules.
2. **Presidio** (`scanners/pii.py`): NER-based entities like PERSON and LOCATION, plus Presidio's own recognizers.
3. **Combine** (`scanners/hybrid.py`): drop low scores, resolve overlaps (stricter action wins, then score, then length), apply `configs/policy.yaml`, and redact with stable placeholders.

Decision: any `block` entity blocks the message; otherwise any `mask` entity redacts it; otherwise allow.

## Data

```bash
uv run python -m eval.prepare_data   # downloads, splits, writes data/processed/ + manifest.json
```

| Task | Source | License | Splits |
|---|---|---|---|
| PII spans | [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | CC-BY-4.0 | val sampled from official train; test sampled from official test |
| Prompt injection | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | val carved from train; official test |
| Prompt injection (attacks) | [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | MIT | official train/val/test |
| Benign traffic | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC-BY-SA-3.0 | 80/10/10 |

- Splits use a salted hash of the normalised text, so they're the same on every machine.
- Any text in test is removed from train/val, and any text in val from train.
- Each dataset is pinned to its Hugging Face commit, and every output file's SHA-256 is in `manifest.json`.
- Data is not committed. Rebuild it with the command above.
- ai4privacy PII datasets were not used: their license forbids derivative works (including fine-tuned models) without written permission.

## Repo layout

```
configs/policy.yaml         entity -> allow / mask / block
src/data_leakage_detection/models.py     Finding, Action, Decision, ScanResult
src/data_leakage_detection/policy.py     policy loader + validation
src/data_leakage_detection/scanners/     regex rules, Presidio wrapper, hybrid combiner
eval/                       dataset prep (prepare_data.py), evaluation scripts
tests/                      unit + integration tests
```
