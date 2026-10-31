"""kg/run_ablation.py: the conditions and the scoring rule, with no model call."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kg"))
import graphrag  # noqa: E402
import run_ablation as ab  # noqa: E402


def test_conditions_differ_only_by_communities():
    assert len(graphrag.RETRIEVALS) == 7
    assert "communities" not in ab.CONDITIONS["A"]
    assert set(ab.CONDITIONS["B"]) - set(ab.CONDITIONS["A"]) == {"communities"}


def test_route_menu_is_limited(monkeypatch):
    seen = {}

    def complete(system, user, schema=None, effort="medium"):
        seen["schema"], seen["system"] = schema, system
        return '{"retrieval": "concept", "term": "flap", "criterion": 0}'

    monkeypatch.setattr(graphrag, "complete", complete)
    graphrag.route("What's a flap?", ab.CONDITIONS["A"])
    assert seen["schema"]["properties"]["retrieval"]["enum"] == ab.CONDITIONS["A"]
    assert "- communities:" not in seen["system"]


def test_every_cell_has_an_expectation_and_a_key():
    assert set(ab.EXPECTED) == set(ab.cells())
    for (c, q), exp in ab.EXPECTED.items():
        assert exp is None or exp in ab.CONDITIONS[c]
        assert exp is None or ab.ANSWER_KEYS[q]


def rec(c, q, answer, retrieval, rows, term="", criterion=0):
    return {"condition": c, "question_id": q, "run": 1, "question": ab.QUESTIONS[q],
            "choice": {"retrieval": retrieval, "term": term, "criterion": criterion},
            "rows": rows, "answer": answer, "cost_usd": 0.0}


def test_scoring_correct_and_grounded():
    rows = {"findings": [{"key": "find-rc-site-phone",
                          "statement": "Cramér's V 0.299 with /t/–/d/; base 24.3%"}],
            "site_phone": [{"key": "community:1", "share_d": 0.3076}]}
    ans = ("Weakly: V is 0.299 [find-rc-site-phone]. Community 1 is 30.8% /d/ "
           "[community:1, find-x-missing]. It is 0.5 off.")
    s = ab.score_run(rec("B", "q7", ans, "communities", rows))
    assert s["routing"] == 1 and s["outcome"] == "correct"
    assert (s["keys_cited"], s["keys_grounded"]) == (3, 2)
    assert s["keys_ungrounded"] == "find-x-missing"
    assert s["numbers"] == 4 and s["numbers_ungrounded"] == "0.5"


def test_scoring_refusal_and_parameter():
    s = ab.score_run(rec("A", "q7", "The retrieved rows don't cover communities.",
                         "concept", {"concept": []}, term="community"))
    assert s["routing"] == "" and s["outcome"] == "correct refusal"
    s = ab.score_run(rec("A", "q8", "They group by speaker [find-sg-span].",
                         "span_grid", {"findings": [{"key": "find-sg-span"}]}))
    assert s["outcome"] == "wrong"
    s = ab.score_run(rec("A", "q2", "It changed [crit-rev3-2].",
                         "criterion_provenance", {"revisions": []}, criterion=3))
    assert s["routing"] == 0 and s["outcome"] == "partial"


def test_scoring_key_labels_and_number_precision():
    rows = {"concept": [{"key": "con-flap", "related": ["con-phone: phone"]}],
            "x": [{"key": "curve:sg:mfcc:r"}]}
    s = ab.score_run(rec("A", "q1", "A tap [con-flap, con-phone, curve:sg].",
                         "concept", rows, term="flap"))
    assert s["keys_ungrounded"] == "curve:sg"
    assert ab.key_number(0.0163, "0.016") and not ab.key_number(0.017, "0.016")
    assert ab.key_number(0.070, "0.07") and not ab.key_number(0.076, "0.07")


def test_daniel_verdict_forms(tmp_path, monkeypatch):
    sheet = tmp_path / "sheet.md"
    sheet.write_text("\n".join([
        "## A-q1-r1", "verdict: ==🟢correct== / partial / wrong / correct refusal",
        "Note: fine, ==not a verdict==",
        "## A-q7-r1", "verdict: correct / partial / wrong / ==🟣correct refusal==",
        "## B-q1-r1", "==verdict: partial==",
        "## B-q2-r1", "verdict: ==correct== / ==partial== / wrong / correct refusal",
        "## B-q3-r1", "verdict: correct / partial / wrong / correct refusal",
    ]), encoding="utf-8")
    monkeypatch.setattr(ab, "HAND_CHECK_SHEET", sheet)
    assert ab.read_daniel_verdicts() == {"A-q1-r1": "correct",
                                         "A-q7-r1": "correct refusal",
                                         "B-q1-r1": "partial"}
