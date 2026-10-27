"""Download the datasets and write train/val/test splits to data/processed/.

usage: uv run python -m eval.prepare_data

Notes on the splits:
- official test splits are kept as test; val always comes out of train
- rows are assigned by hashing the text (not random.shuffle) so the split is
  the same on any machine regardless of row order
- anything in test is removed from train/val, anything in val from train
- dataset revisions + output file hashes go into manifest.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.io import sha256_file, write_jsonl
from eval.label_map import map_nemotron_label

Row = dict[str, Any]
Splits = dict[str, list[Row]]

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_OUT = Path("data/processed")
DEFAULT_SALT = "dld-v1"


@dataclass(frozen=True)
class Source:
    repo: str
    license: str
    note: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}"


SOURCES: dict[str, Source] = {
    "nemotron_pii": Source(
        "nvidia/Nemotron-PII",
        "CC-BY-4.0",
        "synthetic docs w/ char-offset PII spans",
    ),
    "deepset": Source(
        "deepset/prompt-injections",
        "Apache-2.0",
        "1 = injection, 0 = normal",
    ),
    "gandalf": Source(
        "Lakera/gandalf_ignore_instructions",
        "MIT",
        "gandalf game attacks, all label 1",
    ),
    "dolly": Source(
        "databricks/databricks-dolly-15k",
        "CC-BY-SA-3.0",
        "instructions only, used as benign traffic",
    ),
}


# --- splitting helpers ---


def text_key(text: str) -> str:
    # lowercase + collapse whitespace, used for dedup and hashing
    return " ".join(text.lower().split())


def text_id(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.sha1(text_key(text).encode()).hexdigest()[:12]}"


def bucket(text: str, salt: str) -> float:
    # stable number in [0, 1) for a given text
    digest = hashlib.sha256(f"{salt}|{text_key(text)}".encode()).hexdigest()
    return int(digest[:15], 16) / 16**15


def hash_sample(rows: Sequence[Row], n: int, salt: str) -> tuple[list[Row], list[Row]]:
    # pick n rows by hash order (like a seeded shuffle but order-independent).
    # n <= 0 means take everything
    if n <= 0 or n >= len(rows):
        return list(rows), []
    ranked = sorted(rows, key=lambda r: bucket(r["text"], salt))
    return ranked[:n], ranked[n:]


def hash_split(rows: Iterable[Row], val_fraction: float, salt: str) -> tuple[list[Row], list[Row]]:
    train: list[Row] = []
    val: list[Row] = []
    for r in rows:
        (val if bucket(r["text"], salt) < val_fraction else train).append(r)
    return train, val


def dedupe(rows: Iterable[Row]) -> tuple[list[Row], int]:
    seen: set[str] = set()
    kept: list[Row] = []
    removed = 0
    for r in rows:
        k = text_key(r["text"])
        if k in seen:
            removed += 1
        else:
            seen.add(k)
            kept.append(r)
    return kept, removed


def finalize(splits: Splits) -> dict[str, int]:
    # dedup within each split, then remove overlap: test wins over val, val over train.
    # sorting by id at the end keeps the output files byte-identical between runs
    stats: dict[str, int] = {}
    for name in SPLIT_NAMES:
        splits[name], removed = dedupe(splits[name])
        stats[f"{name}_internal_duplicates"] = removed

    test_keys = {text_key(r["text"]) for r in splits["test"]}
    val_before = len(splits["val"])
    splits["val"] = [r for r in splits["val"] if text_key(r["text"]) not in test_keys]
    stats["val_removed_overlap_with_test"] = val_before - len(splits["val"])

    held_out = test_keys | {text_key(r["text"]) for r in splits["val"]}
    train_before = len(splits["train"])
    splits["train"] = [r for r in splits["train"] if text_key(r["text"]) not in held_out]
    stats["train_removed_overlap_with_val_or_test"] = train_before - len(splits["train"])

    for name in SPLIT_NAMES:
        splits[name].sort(key=lambda r: r["id"])
    return stats


# --- PII (Nemotron) ---


def parse_spans(raw: Any) -> list[dict[str, Any]]:
    # spans column is a string, sometimes JSON, sometimes a python literal w/ single quotes
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = ast.literal_eval(raw)  # literal_eval, not eval -- doesn't run code
    if not isinstance(parsed, list):
        raise ValueError("spans must be a list")
    return parsed


def convert_pii_row(row: Row) -> Row | None:
    # returns None (row dropped) if any span's offsets don't match its text.
    #
    # first version dropped ~9.5% of rows. checked all 1.67M spans -- offsets
    # were fine, the mismatches were two quirks:
    #   - age / cvv values stored as ints (42 not "42") -> compare str()
    #   - some span texts lowercased ("spanish" vs "Spanish") -> ignore case
    text = row.get("text") or ""
    if not text.strip():
        return None
    try:
        raw_spans = parse_spans(row.get("spans"))
    except (ValueError, SyntaxError):
        return None

    spans: list[dict[str, Any]] = []
    for s in raw_spans:
        start, end, label = int(s["start"]), int(s["end"]), str(s["label"])
        if not 0 <= start < end <= len(text):
            return None
        if "text" in s and text[start:end].lower() != str(s["text"]).lower():
            return None
        spans.append({"start": start, "end": end, "label": map_nemotron_label(label), "source_label": label})
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return {
        "id": text_id("pii", text),
        "text": text,
        "spans": spans,
        "source": SOURCES["nemotron_pii"].repo,
    }


def build_pii(
    train_rows: Iterable[Row], test_rows: Iterable[Row], val_size: int, test_size: int, salt: str
) -> tuple[Splits, dict[str, int]]:
    stats: Counter[str] = Counter()

    def convert(rows: Iterable[Row]) -> list[Row]:
        out: list[Row] = []
        for r in rows:
            c = convert_pii_row(r)
            if c is None:
                stats["dropped_bad_spans"] += 1
            else:
                out.append(c)
        return out

    train_all = convert(train_rows)
    test_all = convert(test_rows)
    val, train = hash_sample(train_all, val_size, salt)
    test, _ = hash_sample(test_all, test_size, salt)
    splits: Splits = {"train": train, "val": val, "test": test}
    stats.update(finalize(splits))
    return splits, dict(stats)


# --- prompt injection ---


def _labelled(rows: Iterable[Row], prefix: str, source: str, label: int | None = None) -> list[Row]:
    out: list[Row] = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        y = int(r["label"]) if label is None else label
        out.append({"id": text_id(prefix, text), "text": text, "label": y, "source": source})
    return out


def build_injection(
    deepset_train: Iterable[Row],
    deepset_test: Iterable[Row],
    gandalf_train: Iterable[Row],
    gandalf_val: Iterable[Row],
    gandalf_test: Iterable[Row],
    val_fraction: float,
    salt: str,
) -> tuple[Splits, dict[str, int]]:
    # deepset has no val split so take one from its train. gandalf has all 3 splits.
    ds, gd = SOURCES["deepset"].repo, SOURCES["gandalf"].repo
    d_train, d_val = hash_split(_labelled(deepset_train, "inj", ds), val_fraction, salt)
    splits: Splits = {
        "train": d_train + _labelled(gandalf_train, "inj", gd, label=1),
        "val": d_val + _labelled(gandalf_val, "inj", gd, label=1),
        "test": _labelled(deepset_test, "inj", ds) + _labelled(gandalf_test, "inj", gd, label=1),
    }
    return splits, finalize(splits)


# --- benign ---


def build_benign(rows: Iterable[Row], salt: str) -> tuple[Splits, dict[str, int]]:
    # only the instruction field -- it looks like what a user would type. 80/10/10
    splits: Splits = {name: [] for name in SPLIT_NAMES}
    src = SOURCES["dolly"].repo
    for r in rows:
        text = (r.get("instruction") or "").strip()
        if not text:
            continue
        b = bucket(text, salt)
        name = "train" if b < 0.8 else "val" if b < 0.9 else "test"
        splits[name].append(
            {"id": text_id("ben", text), "text": text, "category": r.get("category"), "source": src}
        )
    return splits, finalize(splits)


# --- download ---


def resolve_revision(repo: str) -> str:
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo).sha
    if not sha:
        raise RuntimeError(f"could not resolve revision for {repo}")
    return sha


def hf_loader(revisions: dict[str, str]) -> Callable[[str, str], list[Row]]:
    from datasets import load_dataset

    def load(key: str, split: str) -> list[Row]:
        repo = SOURCES[key].repo
        print(f"  downloading {repo} [{split}] @ {revisions[key][:10]}")
        ds = load_dataset(repo, split=split, revision=revisions[key])
        return [dict(r) for r in ds]

    return load


def label_counts(task: str, rows: list[Row]) -> dict[str, int]:
    if task == "pii":
        c: Counter[str] = Counter(s["label"] or "(unmapped)" for r in rows for s in r["spans"])
    elif task == "injection":
        c = Counter("injection" if r["label"] == 1 else "normal" for r in rows)
    else:
        c = Counter(str(r.get("category")) for r in rows)
    return dict(sorted(c.items()))


def write_task(out_dir: Path, task: str, splits: Splits, stats: dict[str, int]) -> dict[str, Any]:
    info: dict[str, Any] = {"stats": stats, "splits": {}}
    for name in SPLIT_NAMES:
        path = out_dir / task / f"{name}.jsonl"
        n = write_jsonl(path, splits[name])
        info["splits"][name] = {
            "rows": n,
            "sha256": sha256_file(path),
            "label_counts": label_counts(task, splits[name]),
        }
    return info


def run(
    load: Callable[[str, str], list[Row]],
    revisions: dict[str, str],
    out_dir: Path,
    salt: str,
    pii_val_size: int,
    pii_test_size: int,
    injection_val_fraction: float,
) -> dict[str, Any]:
    # load is passed in so tests can use a fake loader instead of HF
    tasks: dict[str, Any] = {}

    print("PII")
    splits, stats = build_pii(
        load("nemotron_pii", "train"), load("nemotron_pii", "test"), pii_val_size, pii_test_size, salt
    )
    tasks["pii"] = write_task(out_dir, "pii", splits, stats)

    print("Prompt injection")
    splits, stats = build_injection(
        load("deepset", "train"),
        load("deepset", "test"),
        load("gandalf", "train"),
        load("gandalf", "validation"),
        load("gandalf", "test"),
        injection_val_fraction,
        salt,
    )
    tasks["injection"] = write_task(out_dir, "injection", splits, stats)

    print("Benign")
    splits, stats = build_benign(load("dolly", "train"), salt)
    tasks["benign"] = write_task(out_dir, "benign", splits, stats)

    manifest = {
        "salt": salt,
        "params": {
            "pii_val_size": pii_val_size,
            "pii_test_size": pii_test_size,
            "injection_val_fraction": injection_val_fraction,
        },
        "sources": {k: {**asdict(s), "url": s.url, "revision": revisions.get(k)} for k, s in SOURCES.items()},
        "tasks": tasks,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def print_summary(manifest: dict[str, Any]) -> None:
    print("\nRows per split")
    print(f"{'task':<10} {'train':>8} {'val':>8} {'test':>8}")
    for task, info in manifest["tasks"].items():
        counts = [info["splits"][s]["rows"] for s in SPLIT_NAMES]
        print(f"{task:<10} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8}")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--salt", default=DEFAULT_SALT, help="changes every split assignment")
    p.add_argument("--pii-val-size", type=int, default=5000)
    p.add_argument(
        "--pii-test-size",
        type=int,
        default=5000,
        help="sample of the 100k official test rows; 0 = use all (slow for Presidio on CPU)",
    )
    p.add_argument("--injection-val-fraction", type=float, default=0.15)
    p.add_argument(
        "--revisions",
        type=Path,
        help="JSON file {source_key: commit_sha} to reproduce an earlier run exactly",
    )
    args = p.parse_args(argv)

    if args.revisions:
        revisions = json.loads(args.revisions.read_text())
    else:
        print("Resolving dataset revisions")
        revisions = {k: resolve_revision(s.repo) for k, s in SOURCES.items()}

    manifest = run(
        hf_loader(revisions),
        revisions,
        args.out,
        args.salt,
        args.pii_val_size,
        args.pii_test_size,
        args.injection_val_fraction,
    )
    print_summary(manifest)
    print(f"\nWrote {args.out}/  (manifest: {args.out / 'manifest.json'})")


if __name__ == "__main__":
    main()
