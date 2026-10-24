# Data Leakage Detection

Data leakage detection for LLM agents. It sits between users, an agent, and the agent's tools, and blocks or redacts sensitive data, prompt injections, hidden-context leaks, and unsafe tool calls. Targets OWASP Top 10 for LLM Applications: LLM01 (Prompt Injection) and LLM02 (Sensitive Information Disclosure).

> Work in progress. Results tables are added only from real runs of `python -m eval.run_all`.

## Status

| Component | Status |
|---|---|
| PII / secret scanner (regex + Presidio) | done |
| Dataset prep + regex/Presidio baselines | next |
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

## Repo layout

```
configs/policy.yaml         entity -> allow / mask / block
src/data_leakage_detection/models.py     Finding, Action, Decision, ScanResult
src/data_leakage_detection/policy.py     policy loader + validation
src/data_leakage_detection/scanners/     regex rules, Presidio wrapper, hybrid combiner
tests/                      unit + integration tests
```
