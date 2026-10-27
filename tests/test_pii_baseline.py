import json
from pathlib import Path

from data_leakage_detection.models import Finding
from data_leakage_detection.policy import load_policy
from data_leakage_detection.scanners.base import RegexScanner
from data_leakage_detection.scanners.pii import PII_RULES
from data_leakage_detection.scanners.secrets import SECRET_RULES
from eval.io import write_jsonl
from eval.pii_baseline import Counts, evaluate, postprocess, score_document, to_markdown


def gold(start: int, end: int, label: str | None) -> dict[str, object]:
    return {"start": start, "end": end, "label": label, "source_label": "x"}


def pred(entity: str, start: int, end: int, score: float = 0.9) -> Finding:
    return Finding(entity, start, end, "x", score, "test")


def test_exact_match_counts_everywhere() -> None:
    c = score_document([gold(0, 5, "EMAIL_ADDRESS")], [pred("EMAIL_ADDRESS", 0, 5)])["EMAIL_ADDRESS"]
    assert (c.gold, c.gold_found, c.gold_exact, c.pred, c.pred_correct) == (1, 1, 1, 1, 1)


def test_one_person_pred_covers_first_and_last_name() -> None:
    # "Jason Smith" as one PERSON should cover both gold spans
    c = score_document([gold(0, 5, "PERSON"), gold(6, 11, "PERSON")], [pred("PERSON", 0, 11)])["PERSON"]
    assert c.gold_found == 2 and c.gold == 2
    assert c.pred == 1 and c.pred_correct == 1
    assert c.gold_exact == 0  # boundaries differ, and exact recall shows it


def test_wrong_type_is_fp_and_fn() -> None:
    counts = score_document([gold(0, 5, "LOCATION")], [pred("PERSON", 0, 5)])
    assert counts["PERSON"].pred == 1 and counts["PERSON"].pred_correct == 0
    assert counts["LOCATION"].gold == 1 and counts["LOCATION"].gold_found == 0


def test_pred_on_unmapped_gold_is_ignored() -> None:
    counts = score_document([gold(0, 5, None)], [pred("LOCATION", 0, 5)])
    assert counts["LOCATION"].pred == 0


def test_pred_on_nothing_is_false_positive() -> None:
    counts = score_document([], [pred("PHONE_NUMBER", 10, 20)])
    assert counts["PHONE_NUMBER"].pred == 1 and counts["PHONE_NUMBER"].pred_correct == 0


def test_unscored_pred_types_ignored() -> None:
    counts = score_document([], [pred("DATE_TIME", 0, 5), pred("AWS_ACCESS_KEY", 6, 26)])
    assert all(c.pred == 0 for c in counts.values())


def test_metrics_math() -> None:
    m = Counts(gold=10, gold_found=8, gold_exact=6, pred=16, pred_correct=8).metrics()
    assert m["precision"] == 0.5 and m["recall"] == 0.8
    assert abs(m["f1"] - 2 * 0.5 * 0.8 / 1.3) < 1e-9  # type: ignore[operator]
    assert m["exact_recall"] == 0.6


def test_no_predictions_gives_none_precision_zero_f1() -> None:
    m = Counts(gold=5).metrics()
    assert m["precision"] is None and m["recall"] == 0.0 and m["f1"] == 0.0


def test_postprocess_drops_low_scores_and_overlaps() -> None:
    policy = load_policy()
    out = postprocess(
        [pred("PERSON", 0, 5, 0.3), pred("EMAIL_ADDRESS", 10, 20), pred("PERSON", 12, 18)], policy
    )
    assert [f.entity_type for f in out] == ["EMAIL_ADDRESS"]


def test_evaluate_regex_layer_end_to_end(tmp_path: Path) -> None:
    docs = [
        {"text": "Mail ana@x.io now", "spans": [gold(5, 13, "EMAIL_ADDRESS")]},
        {"text": "Ana, SSN 123-45-6788", "spans": [gold(0, 3, "PERSON"), gold(9, 20, "US_SSN")]},
    ]
    write_jsonl(tmp_path / "d.jsonl", docs)
    layers = {"regex": RegexScanner(SECRET_RULES + PII_RULES)}
    res = evaluate(docs, layers, load_policy(), progress_every=0)  # type: ignore[arg-type]
    assert list(res) == ["regex_only"]  # configs needing Presidio are skipped
    s = res["regex_only"].summary()
    assert s["per_type"]["EMAIL_ADDRESS"]["recall"] == 1.0
    assert s["per_type"]["US_SSN"]["recall"] == 1.0
    assert s["per_type"]["PERSON"]["recall"] == 0.0
    md = to_markdown(
        {
            "split": "val",
            "n_docs": 2,
            "configs": {"regex_only": s},
            "meta": {"git_commit": "abc", "machine": "test", "command": "x"},
        }
    )
    assert "| EMAIL_ADDRESS | 1 |" in md and "**micro avg**" in md
    json.dumps(s)  # report must be JSON-serialisable
