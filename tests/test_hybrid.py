# most tests use a stub layer so they don't need spacy
from data_leakage_detection.models import Action, Decision, Finding
from data_leakage_detection.policy import Policy
from data_leakage_detection.scanners.hybrid import SensitiveDataScanner, redact, resolve_overlaps
from tests.conftest import fake, requires_presidio

POLICY = Policy.from_dict(
    {
        "min_score": 0.5,
        "default_action": "mask",
        "entities": {"SECRET": "block", "EMAIL_ADDRESS": "mask", "PERSON": "mask", "DATE_TIME": "allow"},
    }
)


def f(entity: str, start: int, end: int, score: float = 0.9, text: str = "x") -> Finding:
    return Finding(entity, start, end, text, score, "stub")


class StubLayer:
    name = "stub"

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings

    def analyze(self, text: str) -> list[Finding]:
        return [
            Finding(x.entity_type, x.start, x.end, text[x.start : x.end], x.score, x.source)
            for x in self.findings
        ]


# ---------- overlap resolution ----------


def test_stricter_action_wins_overlap() -> None:
    kept = resolve_overlaps([f("PERSON", 0, 10, 0.99), f("SECRET", 5, 20, 0.6)], POLICY)
    assert [k.entity_type for k in kept] == ["SECRET"]


def test_higher_score_wins_when_same_action() -> None:
    kept = resolve_overlaps([f("PERSON", 0, 10, 0.6), f("EMAIL_ADDRESS", 0, 10, 0.95)], POLICY)
    assert [k.entity_type for k in kept] == ["EMAIL_ADDRESS"]


def test_longer_span_wins_on_tie() -> None:
    kept = resolve_overlaps([f("PERSON", 0, 4), f("PERSON", 0, 10)], POLICY)
    assert [(k.start, k.end) for k in kept] == [(0, 10)]


def test_non_overlapping_all_kept_in_order() -> None:
    kept = resolve_overlaps([f("PERSON", 10, 15), f("EMAIL_ADDRESS", 0, 5)], POLICY)
    assert [k.start for k in kept] == [0, 10]


# ---------- redaction ----------


def test_same_value_gets_same_placeholder() -> None:
    text = "a@x.com then b@y.com then a@x.com"
    findings = [
        Finding("EMAIL_ADDRESS", 0, 7, "a@x.com", 0.9, "s"),
        Finding("EMAIL_ADDRESS", 13, 20, "b@y.com", 0.9, "s"),
        Finding("EMAIL_ADDRESS", 26, 33, "a@x.com", 0.9, "s"),
    ]
    out = redact(text, findings, [True, True, True])
    assert out == "<EMAIL_ADDRESS_1> then <EMAIL_ADDRESS_2> then <EMAIL_ADDRESS_1>"


def test_unmasked_findings_left_alone() -> None:
    text = "on Monday email a@x.com"
    findings = [
        Finding("DATE_TIME", 3, 9, "Monday", 0.9, "s"),
        Finding("EMAIL_ADDRESS", 16, 23, "a@x.com", 0.9, "s"),
    ]
    assert redact(text, findings, [False, True]) == "on Monday email <EMAIL_ADDRESS_1>"


# ---------- decisions ----------


def test_block_beats_mask() -> None:
    s = SensitiveDataScanner(POLICY, [StubLayer([f("EMAIL_ADDRESS", 0, 3), f("SECRET", 4, 8)])])
    r = s.scan("abc defg")
    assert r.decision is Decision.BLOCK
    assert r.redacted_text == "<EMAIL_ADDRESS_1> <SECRET_1>"  # safe to log


def test_mask_gives_redact() -> None:
    s = SensitiveDataScanner(POLICY, [StubLayer([f("PERSON", 0, 3)])])
    r = s.scan("Bob is here")
    assert r.decision is Decision.REDACT
    assert r.redacted_text == "<PERSON_1> is here"
    assert r.actions == [Action.MASK]


def test_allow_only_findings_give_allow() -> None:
    s = SensitiveDataScanner(POLICY, [StubLayer([f("DATE_TIME", 0, 6)])])
    r = s.scan("Monday works")
    assert r.decision is Decision.ALLOW
    assert r.redacted_text == "Monday works"
    assert r.reasons == []


def test_low_score_dropped() -> None:
    s = SensitiveDataScanner(POLICY, [StubLayer([f("SECRET", 0, 3, score=0.3)])])
    assert s.scan("abc").decision is Decision.ALLOW


def test_audit_dict_has_no_raw_values() -> None:
    s = SensitiveDataScanner(POLICY, [StubLayer([f("SECRET", 0, 6)])])
    d = s.scan("hunter later")
    assert "hunter" not in str(d.to_dict())


# ---------- real stack ----------


def test_regex_only_blocks_secret() -> None:
    s = SensitiveDataScanner.regex_only()
    r = s.scan(fake("use ", "AKIA", "Z7QK3M9P2XWB4N6T", " for deploy"))
    assert r.decision is Decision.BLOCK
    assert "AKIA" not in r.redacted_text


@requires_presidio
def test_full_stack_masks_contact_pii() -> None:
    s = SensitiveDataScanner()
    r = s.scan("I'm Priya Patel, reach me at priya.patel@acme.io or (214) 555-0182.")
    assert r.decision is Decision.REDACT
    for raw in ("Priya Patel", "priya.patel@acme.io", "555-0182"):
        assert raw not in r.redacted_text
    assert "<EMAIL_ADDRESS_1>" in r.redacted_text


@requires_presidio
def test_full_stack_allows_plain_question() -> None:
    r = SensitiveDataScanner().scan("How do I reverse a linked list in Python?")
    assert r.decision is Decision.ALLOW


@requires_presidio
def test_presidio_only_misses_prefixed_secret() -> None:
    # this is why the regex layer exists
    r = SensitiveDataScanner.presidio_only().scan(fake("key ", "AKIA", "Z7QK3M9P2XWB4N6T"))
    assert all(x.entity_type != "AWS_ACCESS_KEY" for x in r.findings)
