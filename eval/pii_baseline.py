"""PII baselines: regex only vs presidio only vs both.

usage: uv run python -m eval.pii_baseline [--split val|test] [--limit N]

Defaults to val. Only run --split test for the final numbers, otherwise we end
up tuning regex rules against the test set.

Scoring (overlap match, per entity type):
- recall: gold span counts as found if a same-type prediction overlaps it
- precision: prediction counts as correct if it overlaps a same-type gold span
- overlap instead of exact match because the data splits "Jason Smith" into
  two spans and presidio returns one. exact-match recall is still reported.
- predictions on gold spans we don't map (postcode, blood_type, ...) are ignored
- predicted types we don't score (DATE_TIME etc.) are ignored
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_leakage_detection.models import Finding
from data_leakage_detection.policy import Policy, load_policy
from data_leakage_detection.scanners.base import RegexScanner, Scanner
from data_leakage_detection.scanners.hybrid import presidio_from_policy, resolve_overlaps
from data_leakage_detection.scanners.pii import PII_RULES
from data_leakage_detection.scanners.secrets import SECRET_RULES
from eval.io import read_jsonl, sha256_file
from eval.label_map import NEMOTRON_TO_DLD

SCORED_TYPES: tuple[str, ...] = tuple(sorted(set(NEMOTRON_TO_DLD.values())))
CONFIGS: dict[str, tuple[str, ...]] = {
    "regex_only": ("regex",),
    "presidio_only": ("presidio",),
    "regex_presidio": ("regex", "presidio"),
}


# --- scoring ---


@dataclass
class Counts:
    gold: int = 0  # gold spans of this type
    gold_found: int = 0  # gold spans overlapped by a same-type prediction
    gold_exact: int = 0  # gold spans matched with exact boundaries
    pred: int = 0  # scored predictions (ignore-region hits excluded)
    pred_correct: int = 0  # predictions overlapping a same-type gold span

    def add(self, other: Counts) -> None:
        self.gold += other.gold
        self.gold_found += other.gold_found
        self.gold_exact += other.gold_exact
        self.pred += other.pred
        self.pred_correct += other.pred_correct

    def metrics(self) -> dict[str, float | int | None]:
        p = self.pred_correct / self.pred if self.pred else None
        r = self.gold_found / self.gold if self.gold else None
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) > 0 else 0.0
        return {
            "precision": p,
            "recall": r,
            "f1": f1 if self.gold else None,
            "exact_recall": self.gold_exact / self.gold if self.gold else None,
            "support": self.gold,
            "predicted": self.pred,
        }


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def score_document(
    gold_spans: Sequence[dict[str, Any]],
    preds: Sequence[Finding],
    scored_types: Iterable[str] = SCORED_TYPES,
) -> dict[str, Counts]:
    scored = set(scored_types)
    counts: dict[str, Counts] = {t: Counts() for t in scored}
    gold = [g for g in gold_spans if g["label"] in scored]
    ignore = [g for g in gold_spans if g["label"] is None]
    preds = [p for p in preds if p.entity_type in scored]

    for p in preds:
        hit = any(
            g["label"] == p.entity_type and _overlaps(p.start, p.end, g["start"], g["end"]) for g in gold
        )
        if hit:
            counts[p.entity_type].pred += 1
            counts[p.entity_type].pred_correct += 1
        elif any(_overlaps(p.start, p.end, g["start"], g["end"]) for g in ignore):
            continue
        else:
            counts[p.entity_type].pred += 1

    for g in gold:
        c = counts[g["label"]]
        c.gold += 1
        same = [p for p in preds if p.entity_type == g["label"]]
        if any(_overlaps(p.start, p.end, g["start"], g["end"]) for p in same):
            c.gold_found += 1
        if any(p.start == g["start"] and p.end == g["end"] for p in same):
            c.gold_exact += 1
    return counts


def postprocess(raw: Iterable[Finding], policy: Policy) -> list[Finding]:
    # same steps as SensitiveDataScanner.find
    return resolve_overlaps([f for f in raw if f.score >= policy.min_score], policy)


# --- run ---


@dataclass
class ConfigResult:
    per_type: dict[str, Counts] = field(default_factory=lambda: {t: Counts() for t in SCORED_TYPES})
    latency_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        micro = Counts()
        for c in self.per_type.values():
            micro.add(c)
        lat = sorted(self.latency_ms)
        return {
            "per_type": {t: c.metrics() for t, c in sorted(self.per_type.items())},
            "micro": micro.metrics(),
            "latency_ms": {
                "p50": statistics.median(lat) if lat else None,
                "p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
                "mean": statistics.fmean(lat) if lat else None,
            },
        }


def evaluate(
    docs: Iterable[dict[str, Any]], layers: dict[str, Scanner], policy: Policy, progress_every: int = 500
) -> dict[str, ConfigResult]:
    results = {name: ConfigResult() for name in CONFIGS}
    for i, doc in enumerate(docs, 1):
        text = doc["text"]
        raw: dict[str, list[Finding]] = {}
        took: dict[str, float] = {}
        for name, layer in layers.items():
            t0 = time.perf_counter()
            raw[name] = layer.analyze(text)
            took[name] = (time.perf_counter() - t0) * 1000

        for cfg, used in CONFIGS.items():
            if not all(u in layers for u in used):
                continue
            t0 = time.perf_counter()
            preds = postprocess((f for u in used for f in raw[u]), policy)
            post_ms = (time.perf_counter() - t0) * 1000
            results[cfg].latency_ms.append(sum(took[u] for u in used) + post_ms)
            for t, c in score_document(doc["spans"], preds).items():
                results[cfg].per_type[t].add(c)

        if progress_every and i % progress_every == 0:
            print(f"  {i} docs")
    return {k: v for k, v in results.items() if v.latency_ms}


# --- reporting ---


def _git_commit() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "eval", "configs"],
            capture_output=True,
            text=True,
            check=True,
        )
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _fmt(x: float | int | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def to_markdown(report: dict[str, Any]) -> str:
    cfgs = list(report["configs"])
    head = "| Entity | Support | " + " | ".join(f"{c} P / R / F1" for c in cfgs) + " |"
    sep = "|---|---:|" + "---|" * len(cfgs)
    lines = [
        f"PII detection on `{report['split']}` split, {report['n_docs']} docs "
        f"(Nemotron-PII), overlap match. Commit `{report['meta']['git_commit']}`.",
        "",
        head,
        sep,
    ]
    types = list(report["configs"][cfgs[0]]["per_type"])
    for t in [*types, "micro"]:
        if t == "micro":
            rows = [report["configs"][c]["micro"] for c in cfgs]
        else:
            rows = [report["configs"][c]["per_type"][t] for c in cfgs]
        cells = [f"{_fmt(m['precision'])} / {_fmt(m['recall'])} / {_fmt(m['f1'])}" for m in rows]
        name = "**micro avg**" if t == "micro" else t
        lines.append(f"| {name} | {rows[0]['support']} | " + " | ".join(cells) + " |")
    lines += ["", "| Latency per doc (ms) | " + " | ".join(cfgs) + " |", "|---|" + "---:|" * len(cfgs)]
    for k in ("p50", "p95"):
        lines.append(
            f"| {k} | " + " | ".join(_fmt(report["configs"][c]["latency_ms"][k]) for c in cfgs) + " |"
        )
    lines.append("")
    lines.append(f"Hardware: {report['meta']['machine']}. Reproduce: `{report['meta']['command']}`")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="PII baselines (regex / Presidio / both).")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--limit", type=int, default=0, help="score only the first N docs (quick check)")
    args = ap.parse_args(argv)

    path = args.data / "pii" / f"{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: uv run python -m eval.prepare_data")
    docs = list(read_jsonl(path))
    if args.limit:
        docs = docs[: args.limit]

    policy = load_policy()
    print("Loading Presidio (spaCy model)...")
    layers: dict[str, Scanner] = {
        "regex": RegexScanner(SECRET_RULES + PII_RULES, name="regex"),
        "presidio": presidio_from_policy(policy),
    }
    layers["presidio"].analyze("warm up the model")  # first call is slow; keep it out of latency

    print(f"Scoring {len(docs)} docs from {path}")
    results = evaluate(docs, layers, policy)

    command = "uv run python -m eval.pii_baseline --split " + args.split
    if args.limit:
        command += f" --limit {args.limit}"
    report = {
        "split": args.split,
        "n_docs": len(docs),
        "scored_types": list(SCORED_TYPES),
        "matching": "overlap, same type; unmapped gold spans are ignore regions",
        "configs": {k: v.summary() for k, v in results.items()},
        "meta": {
            "git_commit": _git_commit(),
            "data_file_sha256": sha256_file(path),
            "policy_min_score": policy.min_score,
            "presidio_score_threshold": policy.presidio.score_threshold,
            "spacy_model": getattr(layers["presidio"], "spacy_model", None),
            "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
            "command": command,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"pii_baseline_{args.split}" + (f"_limit{args.limit}" if args.limit else "")
    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    md = to_markdown(report)
    (args.out / f"{stem}.md").write_text(md)
    print("\n" + md)
    print(f"Wrote {args.out / (stem + '.json')} and .md")


if __name__ == "__main__":
    main()
