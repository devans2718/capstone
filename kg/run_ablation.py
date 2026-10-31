#!/usr/bin/env python3
"""The Challenge 4 Co-Design ablation: GraphRAG with and without `communities`.

HOW TO RUN (from the repo root, in WSL, with neo4j-capstone up)
    .venv-app/bin/python kg/run_ablation.py run --cells A-q6,B-q7   # 3 runs each
    .venv-app/bin/python kg/run_ablation.py run                     # every cell
    .venv-app/bin/python kg/run_ablation.py score

    `run` skips runs already saved, so it can be stopped and resumed. Needs
    ANTHROPIC_API_KEY, as kg/graphrag.py does. `score` makes no model call.

THE QUESTION
    Does community information help GraphRAG answer this project's
    questions? Condition A is the original menu: the six reviewed retrievals
    without `communities`. Condition B is all seven. Both run on the same
    graph, through graphrag.py's own route -> retrieve -> answer, with the
    same model; the only difference is the router's menu. --community (the
    10-28 context injection) is not a condition (Daniel, 2026-10-31).

FIXED BEFORE ANY RUN (Part A of the co-design brief, approved 2026-10-31)
    QUESTIONS, EXPECTED (the right retrieval per question and condition) and
    ANSWER_KEYS are committed before the first run, and the scoring rule
    below is applied the same way to every run.

SCORING, PER RUN
    routing   the chosen retrieval is EXPECTED's, and for q1 and q2 so is the
              parameter. A-q7 and A-q8 have no right retrieval: not scored.
    keys      cited keys ([...], comma lists split) that appear in the rows,
              over all cited keys.
    numbers   numbers in the answer (outside citations, and not in the
              question) that some number in the rows rounds to at the
              answer's precision (a percentage also as a fraction), over all.
    outcome   ANSWER_KEYS items hit (a key that must be cited, a number that
              must appear, at the key's precision or finer): all = correct, some = partial, none = wrong.
              Where the expected outcome is a refusal (A-q7, A-q8): correct
              refusal if the answer says the rows don't answer and cites no
              Finding; otherwise wrong. The hand check can override a label
              (a contradiction, an upgraded status); both are reported.
    variance  per cell, the share of runs on the modal retrieval and on the
              modal outcome.

OUTPUT
    kg/graphrag-runs/codesign-ablation/: one JSON per run (A-q6-r1.json ...),
    with the rows, answer and token usage; scores.csv (per run);
    ablation-table.md and .csv (aggregates). The hand-check sheet for Daniel
    goes to the vault (HAND_CHECK_SHEET), written once and not overwritten;
    his verdicts are read back from it. Claude's own check is
    hand-check-claude.json beside the runs.

THE BUCKEYE RULE
    The rows are graphrag.py's: the design layer, plus the Site-Phone
    communities' aggregates. Nothing per site.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kg"))
import graphrag  # noqa: E402

OUT = REPO / "kg" / "graphrag-runs" / "codesign-ablation"
VAULT = Path("/mnt/c/Users/danie/Documents/Obsidian/general/Projects/Capstone/"
             "Challenges/challenge-4")
HAND_CHECK_SHEET = VAULT / "ablation-hand-check.md"
CLAUDE_CHECK = OUT / "hand-check-claude.json"
RUNS_PER_CELL = 3
SAMPLE_SEED = 20261031
PRICE_IN, PRICE_OUT = 4.00 / 1e6, 20.00 / 1e6   # claude-opus-5-5, $/token

CONDITIONS = {
    "A": [n for n in graphrag.RETRIEVALS if n != "communities"],
    "B": list(graphrag.RETRIEVALS),
}

QUESTIONS = {
    "q1": "What's a flap?",
    "q2": "How did criterion 2 get its final form?",
    "q3": "Why did the pilot fail, and what's planned to test it?",
    "q4": "How does WavJEPA's best layer compare with MFCC under each split "
          "condition?",
    "q5": "What does registration revision 3 specify that the pilot never used?",
    "q6": "What did the span grid find?",
    "q7": "Do the flaps' phonetic environments group into communities, and do "
          "those communities line up with underlying /t/ versus /d/?",
    "q8": "What do the encoders' representations group the flaps by?",
}

# The right retrieval per (condition, question); None = no retrieval in that
# condition can answer, so the expected outcome is a correct refusal.
EXPECTED: dict[tuple[str, str], str | None] = {}
for c in CONDITIONS:
    EXPECTED.update({(c, "q1"): "concept", (c, "q2"): "criterion_provenance",
                     (c, "q3"): "why_failed", (c, "q4"): "word_axis",
                     (c, "q5"): "registered_not_run", (c, "q6"): "span_grid"})
EXPECTED.update({("A", "q7"): None, ("A", "q8"): None,
                 ("B", "q7"): "communities", ("B", "q8"): "communities"})
PARAMS = {"q1": lambda ch: "flap" in ch["term"].lower(),
          "q2": lambda ch: ch["criterion"] == 2}

# What a correct answer must contain, from the reviewed rows: ("key", k) must
# be cited, ("num", x) must appear. Not used where a refusal is expected.
ANSWER_KEYS: dict[str, list[tuple[str, str]]] = {
    "q1": [("key", "con-flap")],
    "q2": [("key", "crit-rev1-2"), ("key", "crit-rev3-2"), ("num", "0.10")],
    "q3": [("key", "crit-rev3-1"), ("key", "find-words"), ("key", "find-width"),
           ("num", "0.016")],
    "q4": [("key", "cond-joint"), ("num", "0.016"), ("num", "0.07")],
    "q5": [("key", "rep-wav2vec2"), ("key", "run-span-grid")],
    "q6": [("key", "find-sg-span"), ("key", "find-sg-encoders"),
           ("key", "find-sg-c7"), ("num", "0.616")],
    "q7": [("key", "find-rc-site-phone"), ("num", "0.299")],
    "q8": [("key", "find-rc-mfcc"), ("key", "find-rc-speaker"), ("num", "0.234")],
}

REFUSAL = re.compile(
    r"\b(rows?|retrieved|results?)\b[^.]{0,80}\b(don't|do not|doesn't|does not|"
    r"aren't|are not|isn't|is not|contain no|include no|say nothing|have no|"
    r"lack|cannot|can't)\b|\bnothing (in|about)\b|\bno (rows?|information|data)\b",
    re.I)
OUTCOMES = ("correct", "partial", "wrong", "correct refusal")


def cells() -> list[tuple[str, str]]:
    return [(c, q) for q in QUESTIONS for c in CONDITIONS]


def run_path(c: str, q: str, r: int) -> Path:
    return OUT / f"{c}-{q}-r{r}.json"


def cost(usage: list[dict]) -> float:
    return sum((u["input_tokens"] + u["cache_creation_input_tokens"]
                + u["cache_read_input_tokens"]) * PRICE_IN
               + u["output_tokens"] * PRICE_OUT for u in usage)

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_one(c: str, q: str, r: int) -> dict:
    question = QUESTIONS[q]
    graphrag.USAGE.clear()
    choice = graphrag.route(question, CONDITIONS[c])
    rows = graphrag.retrieve(choice)
    text = graphrag.answer(question, rows)
    rec = {"time": datetime.now().strftime("%Y%m%d-%H%M%S"),
           "model": graphrag.MODEL, "condition": c, "question_id": q, "run": r,
           "question": question, "menu": CONDITIONS[c], "choice": choice,
           "rows": rows, "answer": text, "usage": list(graphrag.USAGE),
           "cost_usd": round(cost(graphrag.USAGE), 5)}
    OUT.mkdir(parents=True, exist_ok=True)
    run_path(c, q, r).write_text(
        json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n")
    return rec


def cmd_run(args) -> None:
    graphrag.load_dotenv()
    todo = cells()
    if args.cells:
        want = [tuple(x.split("-")) for x in args.cells.split(",")]
        todo = [cq for cq in todo if cq in want]
    spent = 0.0
    for c, q in todo:
        for r in range(1, args.runs + 1):
            if run_path(c, q, r).exists():
                continue
            rec = run_one(c, q, r)
            spent += rec["cost_usd"]
            print(f"{c}-{q}-r{r}  {rec['choice']['retrieval']:<22} "
                  f"${rec['cost_usd']:.4f}  (this call ${spent:.3f})", flush=True)
    saved = load_runs()
    total = sum(x["cost_usd"] for x in saved)
    print(f"\n{len(saved)} runs saved, ${total:.3f} in all; "
          f"projected for {len(cells()) * RUNS_PER_CELL}: "
          f"${total / len(saved) * len(cells()) * RUNS_PER_CELL:.2f}"
          if saved else "no runs saved")

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?%?")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
CITE = re.compile(r"\[([^\[\]]+)\]")
KEYLIKE = re.compile(r"^[a-z][a-z0-9]*(?:[-:_][a-z0-9_]+)+$")


def load_runs() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(OUT.glob("[AB]-q*-r*.json"))]


def cited_keys(answer: str) -> list[str]:
    out = []
    for group in CITE.findall(answer):
        for tok in re.split(r"[,;]\s*|\s+", group.strip()):
            tok = tok.strip("`*. ")
            if KEYLIKE.match(tok):
                out.append(tok)
    return out


def numbers_in(text: str) -> list[tuple[float, int, bool, str]]:
    """(value, decimals, is_percent, as written), dates and citations removed."""
    text = DATE.sub(" ", CITE.sub(" ", text))
    out = []
    for m in NUM.finditer(text):
        s = m.group().rstrip(",")
        pct = s.endswith("%")
        core = s.rstrip("%").replace(",", "").replace("−", "-").lstrip("+")
        try:
            v = float(core)
        except ValueError:
            continue
        dec = len(core.split(".")[1]) if "." in core else 0
        out.append((abs(v), dec, pct, s))
    return out


def row_numbers(rows) -> list[float]:
    vals: list[float] = []

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            vals.append(abs(float(x)))
        elif isinstance(x, str):
            vals.extend(v for v, _, _, _ in numbers_in(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(rows)
    return vals


def grounded_number(v: float, dec: int, pct: bool, pool: list[float]) -> bool:
    targets = [(v, dec)] + ([(v / 100, dec + 2)] if pct else [])
    return any(abs(round(r, d) - t) < 10 ** -(d + 3) for t, d in targets for r in pool)


def key_number(v: float, x: str) -> bool:
    """An answer number matches an answer-key number if it rounds to it at the
    key's precision: +0.0163 (the rows' own figure) is the key's 0.016."""
    dec = len(x.split(".")[1]) if "." in x else 0
    return abs(round(v, dec) - float(x)) < 10 ** -(dec + 3)


def score_run(rec: dict) -> dict:
    c, q, ans = rec["condition"], rec["question_id"], rec["answer"]
    rows_json = json.dumps(rec["rows"], ensure_ascii=False, default=str)
    exp = EXPECTED[c, q]
    ch = rec["choice"]
    if exp is None:
        routing = ""
    else:
        routing = int(ch["retrieval"] == exp and PARAMS.get(q, lambda _: True)(ch))

    keys = cited_keys(ans)
    # A key followed by ":" and a word character is a longer key
    # (curve:sg vs curve:sg:mfcc:r); followed by ": " it is a label
    # ("con-phone: phone" in concept's related list).
    k_ok = [k for k in keys if re.search(r"(?<![\w:-])" + re.escape(k) + r"(?![\w-]|:\w)",
                                         rows_json)]
    q_nums = {v for v, _, _, _ in numbers_in(rec["question"])}
    nums = [n for n in numbers_in(ans) if n[0] not in q_nums]
    pool = row_numbers(rec["rows"])
    n_bad = [n[3] for n in nums if not grounded_number(n[0], n[1], n[2], pool)]

    ans_vals = [v for v, _, _, _ in numbers_in(ans)]
    if exp is None:
        refuses = bool(REFUSAL.search(ans))
        cites_finding = any(k.startswith("find-") for k in keys)
        hits, outcome = "", ("correct refusal" if refuses and not cites_finding
                             else "wrong")
    else:
        items = ANSWER_KEYS[q]
        hit = [x for kind, x in items
               if (kind == "key" and x in keys)
               or (kind == "num" and any(key_number(v, x) for v in ans_vals))]
        hits = f"{len(hit)}/{len(items)}"
        outcome = ("correct" if len(hit) == len(items)
                   else "partial" if hit else "wrong")
    return {"run_id": f"{c}-{q}-r{rec['run']}", "condition": c, "question_id": q,
            "run": rec["run"], "retrieval": ch["retrieval"], "expected": exp or "",
            "routing": routing, "keys_cited": len(keys), "keys_grounded": len(k_ok),
            "keys_ungrounded": " ".join(sorted(set(keys) - set(k_ok))),
            "numbers": len(nums), "numbers_grounded": len(nums) - len(n_bad),
            "numbers_ungrounded": " ".join(n_bad), "answer_key_hits": hits,
            "outcome": outcome, "cost_usd": rec["cost_usd"]}


def hand_sample(scores: list[dict]) -> list[str]:
    """All A-q7/A-q8 runs, plus one seeded-random run from every other cell."""
    rng = random.Random(SAMPLE_SEED)
    out = []
    for c, q in cells():
        ids = sorted(s["run_id"] for s in scores
                     if s["condition"] == c and s["question_id"] == q)
        if not ids:
            continue
        out += ids if (c, q) in (("A", "q7"), ("A", "q8")) else [rng.choice(ids)]
    return out


def _label(text: str) -> str | None:
    """An outcome label, with any leading emoji or marks stripped."""
    v = re.sub(r"^[^a-z]+", "", text.strip().lower()).strip()
    return v if v in OUTCOMES else None


def read_daniel_verdicts() -> dict[str, str]:
    """Daniel's verdicts from the vault sheet, in either form he may use:
        ==verdict: correct==                              (the sheet's form)
        verdict: ==🟢correct== / partial / wrong / ...    (one option highlighted)
    A verdict line with no highlight, or more than one, or an unknown label,
    is reported and left out rather than guessed. Note lines aren't read."""
    if not HAND_CHECK_SHEET.exists():
        return {}
    out, cur, problems = {}, None, []
    for line in HAND_CHECK_SHEET.read_text(encoding="utf-8").splitlines():
        if (m := re.match(r"^## ([AB]-q\d-r\d)\b", line)):
            cur = m[1]
        elif cur and (m := re.search(r"==verdict:\s*([^=]*?)\s*==", line)):
            if (v := _label(m[1])):
                out[cur] = v
        elif cur and re.match(r"^\W*verdict\W*:", line, re.I):
            marked = re.findall(r"==([^=]+)==", line)
            v = _label(marked[0]) if len(marked) == 1 else None
            if v:
                out[cur] = v
            elif marked:
                problems.append(f"{cur}: {line.strip()}")
    for p in problems:
        print(f"hand-check sheet: unreadable verdict, left out: {p}", file=sys.stderr)
    return out


def pct(a: int, b: int) -> str:
    return f"{a}/{b}" if b else "–"


def counts(c: Counter) -> str:
    return ", ".join(f"{v} {k}" for k, v in sorted(c.items(),
                                                   key=lambda kv: OUTCOMES.index(kv[0])))


def cell_summary(ss: list[dict], daniel: dict[str, str]) -> dict:
    """One cell's aggregates. Outcomes are the rule's; Daniel's verdicts are
    reported beside them, over the runs he checked, never blended in."""
    n = len(ss)
    routed = [s for s in ss if s["routing"] != ""]
    auto = Counter(s["outcome"] for s in ss)
    checked = [daniel[s["run_id"]] for s in ss if s["run_id"] in daniel]
    rc = Counter(s["retrieval"] for s in ss)
    return {
        "runs": n,
        "routing": pct(sum(s["routing"] for s in routed), len(routed)) if routed
                   else "n/a (" + ", ".join(f"{k} {v}" for k, v in rc.most_common()) + ")",
        "keys": pct(sum(s["keys_grounded"] for s in ss), sum(s["keys_cited"] for s in ss)),
        "numbers": pct(sum(s["numbers_grounded"] for s in ss), sum(s["numbers"] for s in ss)),
        "outcome": counts(auto),
        "daniel": (counts(Counter(checked)) + f" (of {len(checked)} checked)"
                   if checked else ""),
        "same_retrieval": pct(rc.most_common(1)[0][1], n),
        "same_outcome": pct(auto.most_common(1)[0][1], n),
    }


def rows_summary(rows: dict) -> str:
    parts = []
    for name, rs in rows.items():
        keys = [r.get("key") for r in rs if isinstance(r, dict) and r.get("key")]
        shown = ", ".join(f"`{k}`" for k in keys[:8]) + (" …" if len(keys) > 8 else "")
        parts.append(f"{name}: {len(rs)} rows" + (f" ({shown})" if keys else ""))
    return "; ".join(parts)


def write_sheet(runs: dict[str, dict], scores: dict[str, dict], sample: list[str],
                existing: dict[str, str]) -> None:
    out = ["Hand check of the Co-Design ablation, for Daniel. Generated by "
           "`kg/run_ablation.py score`; your verdicts are read back from the "
           "`==verdict: …==` highlights, so keep that form.",
           "",
           "**How to fill it in:** replace the text inside each highlight with one of "
           "`correct`, `partial`, `wrong`, `correct refusal`, judged against the rows "
           "the answer had (summarized under each run). Add a note on the line below "
           "if useful. The rule: correct = what the question asks, from the rows, "
           "nothing contradicting them, statuses kept; partial = some of it, nothing "
           "contradicting; wrong = a claim the rows contradict, a status upgraded, or "
           "nothing useful; correct refusal = the rows can't answer, and the answer "
           "says so without inventing. A refusal where the rows could answer is wrong.",
           "",
           f"Runs: all six A-q7/A-q8 runs, plus one per other cell (seed {SAMPLE_SEED}). "
           "A is the original six retrievals; B adds `communities`.",
           ""]
    for rid in sample:
        rec, s = runs[rid], scores[rid]
        mine = json.loads(CLAUDE_CHECK.read_text()).get(rid, {}) if CLAUDE_CHECK.exists() else {}
        v = existing.get(rid, "correct / partial / wrong / correct refusal")
        out += [f"## {rid}", "",
                f"- **Question:** {rec['question']}",
                f"- **Condition:** {rec['condition']} ("
                + ("six retrievals, no `communities`" if rec["condition"] == "A"
                   else "seven, `communities` included") + ")",
                f"- **Retrieval chosen:** `{rec['choice']['retrieval']}`"
                + (f" (term {rec['choice']['term']!r})" if rec['choice']['term'] else "")
                + (f" (criterion {rec['choice']['criterion']})" if rec['choice']['criterion'] else "")
                + f"; expected `{s['expected'] or 'none (refusal expected)'}`",
                f"- **Rows it had:** {rows_summary(rec['rows'])}",
                f"- **Automatic label:** {s['outcome']}"
                + (f" (answer-key items {s['answer_key_hits']})" if s['answer_key_hits'] else "")
                + f"; keys grounded {s['keys_grounded']}/{s['keys_cited']}, numbers "
                  f"grounded {s['numbers_grounded']}/{s['numbers']}"
                + (f" (not in the rows: {s['numbers_ungrounded']})" if s['numbers_ungrounded'] else ""),
                f"- **Claude's check:** {mine.get('verdict', '–')}"
                + (f". {mine['note']}" if mine.get("note") else ""),
                "", "**Answer (verbatim):**", ""]
        out += ["> " + l if l else ">" for l in rec["answer"].splitlines()]
        out += ["", f"==verdict: {v}==", "Note: ", ""]
    HAND_CHECK_SHEET.write_text("\n".join(out), encoding="utf-8", newline="\n")


def cmd_score(args) -> None:
    runs = {f"{r['condition']}-{r['question_id']}-r{r['run']}": r for r in load_runs()}
    scores = {k: score_run(r) for k, r in runs.items()}
    with (OUT / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(scores.values()))))
        w.writeheader()
        w.writerows(scores.values())

    mine = json.loads(CLAUDE_CHECK.read_text()) if CLAUDE_CHECK.exists() else {}
    daniel = read_daniel_verdicts()
    sample = hand_sample(list(scores.values()))
    # Written once; after that only with --rewrite-sheet, so Daniel's notes
    # are never overwritten (his verdicts are carried over if it is).
    if args.rewrite_sheet or not HAND_CHECK_SHEET.exists():
        write_sheet(runs, scores, sample, daniel)

    table = []
    for q in QUESTIONS:
        row = {"question": q}
        for c in CONDITIONS:
            ss = [s for s in scores.values() if s["condition"] == c and s["question_id"] == q]
            if ss:
                for k, v in cell_summary(ss, daniel).items():
                    row[f"{c}_{k}"] = v
        table.append(row)
    with (OUT / "ablation-table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in table for k in r},
                                                 key=lambda k: (k != "question", k)))
        w.writeheader()
        w.writerows(table)

    cols = ["routing", "keys", "numbers", "outcome", "daniel", "same_retrieval",
            "same_outcome"]
    md = ["# GraphRAG ablation: without and with `communities`", "",
          f"{len(runs)} runs, {RUNS_PER_CELL} per question × condition, "
          f"{graphrag.MODEL}; total ${sum(r['cost_usd'] for r in runs.values()):.2f}. "
          "A = the six reviewed retrievals; B = seven, `communities` included. "
          "`outcome` and `same outcome` are the scoring rule's labels only. `daniel` is "
          "Daniel's hand-check verdicts, over the runs he checked (empty until he has); "
          "Claude's check is in the hand-check sheet and the agreement line, not in "
          "the counts.", "",
          "| question | " + " | ".join(f"{c} {k.replace('_', ' ')}" for c in CONDITIONS
                                        for k in cols) + " |",
          "|---|" + "---|" * (len(cols) * len(CONDITIONS))]
    for r in table:
        md.append(f"| {r['question']} | " + " | ".join(str(r.get(f"{c}_{k}", "")) for c in CONDITIONS
                                                     for k in cols) + " |")
    md += ["", "Questions: " + "; ".join(f"{q} {t}" for q, t in QUESTIONS.items()), ""]
    agree = []
    for name, other in (("Claude's check", {k: v["verdict"] for k, v in mine.items()}),
                        ("Daniel's check", daniel)):
        both = [rid for rid in other if rid in scores]
        if both:
            n = sum(other[rid] == scores[rid]["outcome"] for rid in both)
            agree.append(f"automatic vs {name}: {n}/{len(both)}")
    both = [rid for rid in daniel if rid in mine]
    if both:
        agree.append("Claude vs Daniel: "
                     f"{sum(daniel[r] == mine[r]['verdict'] for r in both)}/{len(both)}")
    md.append("Agreement on the hand-checked runs: " + ("; ".join(agree) or "none yet") + ".")
    if daniel:
        md.append("Daniel's notes on individual runs are in the hand-check sheet "
                  f"(`{HAND_CHECK_SHEET.name}`, in the vault), not in this table.")
    (OUT / "ablation-table.md").write_text("\n".join(md) + "\n", encoding="utf-8",
                                           newline="\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cells", help="comma list like A-q6,B-q7 (default: all)")
    r.add_argument("--runs", type=int, default=RUNS_PER_CELL)
    s = sub.add_parser("score")
    s.add_argument("--rewrite-sheet", action="store_true",
                   help="rewrite the vault's hand-check sheet even if it exists "
                        "(verdicts are kept, notes are not)")
    args = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
