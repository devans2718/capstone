#!/usr/bin/env python3
"""Build the knowledge graph's design layer as one Cypher script.

HOW TO RUN (from the repo root, in WSL)
    python3 kg/build_design.py            # writes kg/design.cypher, prints a summary
    docker exec -i neo4j-capstone cypher-shell -u neo4j -p phonoprobe-local \
        < kg/design.cypher

WHAT IT DOES
    Takes the hand-written facts in kg/facts.py and adds what the repository
    already records, so none of it is retyped:
      - the Run, from results/flapping/run-manifest-pilot.json and the
        criteria report (one commit per run_probes.py stage)
      - 42 Curves, from results/flapping/probe-results-20260926.csv: one per
        representation x pooling x condition x (full | subsampled), with the
        per-layer values as lists, so an encoder with any number of layers fits
      - 4 Outcomes, parsed from results/flapping/criteria-20260926.txt
      - the span grid's Run and 17 Curves, from results/span-grid/: its
        manifest, span-grid-results-20261029.csv, and the summary's per-block
        table for the control test's p-value and C6's best-layer interval
      - the 2026-10-30 representation-communities Run (exploratory, not
        registered), from kg/communities/representation/summary.json, with
        each representation's AMI profile on its USED edge; and a check that
        every number the find-rc-* Findings quote matches that file and
        kg/communities/summary.json
      - each Artifact's commit, from `git log`
    Then it checks that every edge points at a node that exists, and writes
    kg/design.cypher. Standard library only.

RELOADING
    Every node written here carries layer = "design". The script starts by
    deleting exactly those nodes, so a reload replaces the design layer and
    leaves the corpus layer alone. Edits to the facts take effect on reload.
    The edges joining the layers (IN_DATASET, SAMPLES) are dropped with the
    design nodes, so re-run kg/build_corpus.py after reloading this one.

THE BUCKEYE RULE
    Only aggregate files are read. Nothing in the output carries words,
    speaker codes or timestamps; see kg/facts.py.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kg"))
import facts  # noqa: E402

RESULTS = REPO / "results" / "flapping"
CSV_PATH = RESULTS / "probe-results-20260926.csv"
CRITERIA_PATH = RESULTS / "criteria-20260926.txt"
MANIFEST_PATH = RESULTS / "run-manifest-pilot.json"
OUT = REPO / "kg" / "design.cypher"
COMMIT_MAP = REPO / "docs" / "provenance" / "commit-map.txt"

SG_RESULTS = REPO / "results" / "span-grid"
SG_CSV = SG_RESULTS / "span-grid-results-20261029.csv"
SG_COMPARISONS = SG_RESULTS / "span-grid-comparisons-20261029.csv"
SG_SUMMARY = SG_RESULTS / "span-grid-summary-20261029.txt"
SG_MANIFEST = SG_RESULTS / "run-manifest-span-grid.json"

RC_SUMMARY = REPO / "kg" / "communities" / "representation" / "summary.json"
COMM_SUMMARY = REPO / "kg" / "communities" / "summary.json"
RC_SCRIPT = "kg/representation_communities.py"
RC_GROUPINGS = ["label", "vowel", "following", "pair", "speaker", "word",
                "lemma", "site_phone"]

REP_KEY = {"wavjepa": "rep-wavjepa", "mfcc": "rep-mfcc", "wav2vec2": "rep-wav2vec2",
           "duration": "rep-duration"}
REP_NAME = {"wavjepa": "WavJEPA", "mfcc": "MFCC", "wav2vec2": "wav2vec2-base",
            "duration": "Duration", "mfcc_duration": "MFCC + duration"}
SPAN_KEY = {"r": "env-flap", "vr": "env-v-flap", "rv": "env-flap-v", "vrv": "env-v-flap-v"}
SPAN_NAME = {"r": "ɾ", "vr": "Vɾ", "rv": "ɾV", "vrv": "VɾV"}
COND_KEY = {"random": "cond-random", "speaker_disjoint": "cond-speaker",
            "word_disjoint": "cond-word", "joint": "cond-joint"}
POOL_KEY = {"mean": "pool-mean", "max": "pool-max", "midpoint": "pool-midpoint"}

# Per-layer columns stored as lists on a Curve, and per-curve scalars.
LAYER_COLUMNS = ["balanced_acc", "ci_lo", "ci_hi", "f1_d",
                 "control_balanced_acc", "selectivity", "speaker_balanced_acc"]
SCALAR_COLUMNS = ["n", "n_d", "lookup_balanced_acc"]


# ---------------------------------------------------------------------------
# Generated nodes
# ---------------------------------------------------------------------------

def public_commit(sha: str) -> str:
    """This repository's 7-character hash for a commit the run records recorded.

    The pilot ran in the private predecessor repository, so the manifest and
    criteria report name its hashes. They stay as recorded; the graph carries
    the public counterpart from the rewrite's commit map, so every commit in
    the graph resolves here.
    """
    for line in COMMIT_MAP.read_text().splitlines()[1:]:
        old, new = line.split()
        if old.startswith(sha):
            if set(new) == {"0"}:
                sys.exit(f"commit {sha} was dropped by the rewrite")
            return new[:7]
    sys.exit(f"commit {sha} is not in {COMMIT_MAP.name}")


def run_nodes(nodes: list, edges: list) -> None:
    """The pilot run, from its manifest and criteria report, and what it used.

    run_probes.py records HEAD at each stage, and the stages ran at different
    commits on 2026-09-26: `gate` (the criteria report) at one, `assemble`
    (the manifest and CSVs) at a later one whose only code change is inside
    assemble(). So the Run carries a commit per stage, not one code_commit.
    """
    m = json.loads(MANIFEST_PATH.read_text())
    report = CRITERIA_PATH.read_text()
    gate = re.search(r"^code commit\s+([0-9a-f]+)", report, re.M)
    feats = re.search(r"\(extracted at ([0-9a-f]+)\)", report)
    if not (gate and feats):
        sys.exit(f"couldn't read the stage commits from {CRITERIA_PATH.name}")
    props = {
        "name": m["run_id"], "run_id": m["run_id"], "date": "2026-09-26",
        "features_commit": public_commit(feats[1]),
        "gate_commit": public_commit(gate[1]),
        "assemble_commit": public_commit(m["code_commit"]),
        "checkpoint_revision": m["checkpoint_revision"],
        "context_ms": m["context_ms"], "inventory_hash": m["inventory_hash"],
        "n_sites_scored": m["n_sites_scored"], "synthetic": m["synthetic"],
        "criteria_commit": "b163059", "verdict": "failed, 1 of 4 criteria",
    }
    props.update({f"seed_{k}": v for k, v in m["seeds"].items()})
    nodes.append({"key": "run-pilot", "labels": ["Run"], **props})

    e = edges.append
    e(("exp-flapping", "HAS_RUN", "run-pilot", {}))
    e(("run-pilot", "REGISTERED_BY", "rev3", {}))
    e(("run-pilot", "USED", "ds-buckeye", {}))
    e(("run-pilot", "USED", "rep-wavjepa", {"role": "primary"}))
    e(("run-pilot", "USED", "rep-mfcc", {"role": "baseline"}))
    e(("run-pilot", "USED", "ctrl-lookup", {"role": "baseline"}))
    e(("run-pilot", "USED", "probe-linear", {"role": "primary"}))
    e(("run-pilot", "USED", "cond-joint", {"role": "gated"}))
    for c in ("cond-random", "cond-speaker", "cond-word"):
        e(("run-pilot", "USED", c, {"role": "reported"}))
    e(("run-pilot", "USED", "pool-mean", {"role": "primary"}))
    e(("run-pilot", "USED", "pool-max", {"role": "secondary"}))
    e(("run-pilot", "USED", "pool-midpoint", {"role": "secondary"}))
    e(("run-pilot", "USED", "env-flap", {"role": "primary"}))
    e(("run-pilot", "REALIZES", "task-voicing", {}))
    e(("run-pilot", "REALIZES", "task-control", {}))
    e(("run-pilot", "REALIZES", "task-speaker", {}))


def _num(s: str):
    if s in ("", None):
        return None
    if s in ("True", "False"):
        return s == "True"
    return float(s) if any(ch in s for ch in ".eE") else int(s)


def curve_nodes(nodes: list, edges: list) -> dict:
    """One Curve per representation x pooling x condition x subsample."""
    groups: dict[tuple, list] = defaultdict(list)
    with CSV_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            sub = "subsampled" if row["subsampled"] == "True" else "full"
            groups[(row["encoder"], row["pooling"], row["condition"], sub)].append(row)

    curves = {}
    for (enc, pool, cond, sub), rows in sorted(groups.items()):
        rows.sort(key=lambda r: int(r["layer"]))
        key = f"curve:{enc}:{pool}:{cond}:{sub}"
        props = {
            "name": f"{REP_NAME[enc]} · {cond} · {pool}"
                    + (" · subsampled" if sub == "subsampled" else ""),
            "representation": enc, "pooling": pool, "condition": cond,
            "subsampled": sub == "subsampled", "n_layers": len(rows),
        }
        for col in LAYER_COLUMNS:
            vals = [_num(r[col]) for r in rows]
            if all(v is not None for v in vals):
                props[col] = vals
        for col in SCALAR_COLUMNS:
            vals = {_num(r[col]) for r in rows}
            if len(vals) == 1 and None not in vals:
                props[col] = vals.pop()
        ba = props["balanced_acc"]
        best = max(range(len(ba)), key=lambda i: ba[i])  # first layer on ties
        props["best_layer"] = best
        props["best_balanced_acc"] = ba[best]
        props["spread"] = round(max(ba) - min(ba), 6)
        props["n_nonconverged"] = sum(int(r["n_nonconverged"]) for r in rows)

        nodes.append({"key": key, "labels": ["Curve"], **props})
        edges.append((key, "FROM_RUN", "run-pilot", {}))
        edges.append((key, "FOR_REPRESENTATION", REP_KEY[enc], {}))
        edges.append((key, "UNDER_CONDITION", COND_KEY[cond], {}))
        edges.append((key, "WITH_POOLING", POOL_KEY[pool], {}))
        edges.append((key, "IN_ENVIRONMENT", "env-flap", {}))   # the pilot pooled the flap only
        curves[key] = props
    return curves


def span_grid_run(nodes: list, edges: list) -> None:
    """The span grid's pooled-design run, from its manifest and summary.

    It ran in this repository, so its commits need no public_commit mapping.
    """
    m = json.loads(SG_MANIFEST.read_text())
    feats = re.search(r"\(extracted at ([0-9a-f]+)\)", SG_SUMMARY.read_text())
    if not feats:
        sys.exit(f"couldn't read the features commit from {SG_SUMMARY.name}")
    props = {
        "name": m["run_id"], "run_id": m["run_id"], "date": "2026-10-29",
        "code_commit": m["code_commit"][:7], "features_commit": feats[1][:7],
        "context_ms": m["context_ms"], "inventory_hash": m["inventory_hash"],
        "n_sites_scored": m["n_sites_scored"], "synthetic": m["synthetic"],
        "verdict": "gates nothing (§6); reported",
    }
    props.update({f"seed_{k}": v for k, v in m["seeds"].items()})
    nodes.append({"key": "run-span-grid", "labels": ["Run"], **props})

    e = edges.append
    e(("exp-flapping", "HAS_RUN", "run-span-grid", {}))
    e(("run-span-grid", "REGISTERED_BY", "sg-rev1", {}))
    e(("run-span-grid", "USED", "ds-buckeye", {}))
    e(("run-span-grid", "USED", "rep-wavjepa", {"role": "compared"}))
    e(("run-span-grid", "USED", "rep-wav2vec2", {"role": "compared"}))
    e(("run-span-grid", "USED", "rep-mfcc", {"role": "baseline"}))
    e(("run-span-grid", "USED", "rep-duration", {"role": "baseline"}))
    e(("run-span-grid", "USED", "probe-linear", {"role": "primary"}))
    e(("run-span-grid", "USED", "cond-joint", {"role": "only condition"}))
    e(("run-span-grid", "USED", "pool-mean", {"role": "only pooling"}))
    for env in SPAN_KEY.values():
        e(("run-span-grid", "USED", env, {"role": "span"}))
    e(("run-span-grid", "REALIZES", "task-voicing", {}))
    e(("run-span-grid", "REALIZES", "task-control", {}))


SG_LAYER_COLUMNS = ["balanced_acc", "ci_lo", "ci_hi", "control_balanced_acc",
                    "lemma_unseen_balanced_acc", "unclipped_balanced_acc"]
SG_BLOCK_LINE = re.compile(r"^(\w+?)_(r|vr|rv|vrv|clipped|full|vowel)\s+(\d+)\s+[\d.]+\s+"
                           r"\[[\d.]+, [\d.]+\]\s+\[(\d+), (\d+)\]\s+([\d.]+)\s")


def span_grid_curves(nodes: list, edges: list) -> dict:
    """One Curve per span x representation and per baseline, from the results CSV.

    The control test's p-value and C6's best-layer interval are written only in
    the summary's per-block table, so they are parsed from there. No Curve here
    carries `subsampled`, so the pilot's queries on it can't pick these up.
    """
    summary = {}
    for line in SG_SUMMARY.read_text().splitlines():
        if (mt := SG_BLOCK_LINE.match(line)):
            summary[mt[1], mt[2]] = {"best_layer": int(mt[3]), "best_layer_lo": int(mt[4]),
                                     "best_layer_hi": int(mt[5]), "control_p": float(mt[6])}
    groups: dict[tuple, list] = defaultdict(list)
    with SG_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            groups[(row["representation"], row["span"])].append(row)
    if set(groups) != set(summary) or len(groups) != 17:
        sys.exit(f"{SG_CSV.name} and {SG_SUMMARY.name} don't list the same 17 blocks")

    curves = {}
    for (rep, span), rows in sorted(groups.items()):
        rows.sort(key=lambda r: int(r["layer"]))
        key = f"curve:sg:{rep}:{span}"
        props = {"name": f"{REP_NAME[rep]} · {SPAN_NAME.get(span, span)} · span grid",
                 "representation": rep, "pooling": "mean", "condition": "joint",
                 "n_layers": len(rows)}
        if span in SPAN_KEY:
            props["span"] = span
        else:
            props["variant"] = span
        for col in SG_LAYER_COLUMNS:
            props[col] = [_num(r[col]) for r in rows]
        for col in ("n", "n_d"):
            props[col] = _num(rows[0][col])
        ba = props["balanced_acc"]
        best = max(range(len(ba)), key=lambda i: ba[i])
        if best != summary[rep, span]["best_layer"]:
            sys.exit(f"{key}: best layer {best} in the CSV, "
                     f"{summary[rep, span]['best_layer']} in the summary")
        props.update(summary[rep, span])
        props["best_balanced_acc"] = ba[best]
        props["best_ci_lo"] = props["ci_lo"][best]
        props["best_ci_hi"] = props["ci_hi"][best]
        props["spread"] = round(max(ba) - min(ba), 6)
        props["n_nonconverged"] = sum(int(r["n_nonconverged"]) for r in rows)

        nodes.append({"key": key, "labels": ["Curve"], **props})
        e = edges.append
        e((key, "FROM_RUN", "run-span-grid", {}))
        e((key, "UNDER_CONDITION", "cond-joint", {}))
        e((key, "WITH_POOLING", "pool-mean", {}))
        if rep == "mfcc_duration":
            e((key, "FOR_REPRESENTATION", "rep-mfcc", {}))
            e((key, "FOR_REPRESENTATION", "rep-duration", {}))
        else:
            e((key, "FOR_REPRESENTATION", REP_KEY[rep], {}))
        if span in SPAN_KEY:
            e((key, "IN_ENVIRONMENT", SPAN_KEY[span], {}))
        curves[key] = props
    return curves


def _git_commit(path: str) -> str:
    return subprocess.run(["git", "log", "-1", "--format=%h", "--", path],
                          cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def rep_communities_run(nodes: list, edges: list) -> None:
    """The 2026-10-30 communities over the encoders' features, from its summary.

    Exploratory, not registered: no REGISTERED_BY edge, so the span_grid
    retrieval (which follows REGISTERED_BY sg-rev1) can't pick it up. Each
    representation's results go on its USED edge, which says which run and
    which layer they belong to; the Representation nodes stay run-neutral.
    """
    d = json.loads(RC_SUMMARY.read_text())
    st, sp = d["setup"], d["site_phone_reference"]
    props = {
        "name": "representation-communities-20261030",
        "run_id": "representation-communities-20261030", "date": "2026-10-30",
        "code_commit": _git_commit(RC_SCRIPT), "exploratory": True,
        "synthetic": False, "n_sites_scored": d["sites"]["n"],
        "span": st["span"], "k": st["k"], "k_sensitivity": st["k_sensitivity"],
        "metric": st["metric"], "algorithm": st["algorithm"],
        "resolution": st["resolution"], "seed": st["seed"],
        "verdict": "exploratory; not registered",
    }
    props.update({f"site_phone_ami_{g}": sp["comparison"][g]
                  for g in RC_GROUPINGS if g != "site_phone"})
    nodes.append({"key": "run-rep-communities", "labels": ["Run"], **props})

    e = edges.append
    e(("exp-flapping", "HAS_RUN", "run-rep-communities", {}))
    e(("run-rep-communities", "USED", "ds-buckeye", {}))
    e(("run-rep-communities", "USED", "env-flap", {"role": "span"}))
    for rep, r in d["representations"].items():
        c, stab, iv = r["comparison"], r["stability"], r["intervals"]
        p = {"role": "baseline" if rep == "mfcc" else "compared",
             "layer": r["layer"], "n_communities": r["run"]["n_communities"],
             "modularity": r["run"]["modularity"],
             "ari_seed_median": stab["ari_vs_seeded_median"],
             "ari_seed_min": stab["ari_vs_seeded_min"],
             "v_label": c["label_cramers_v"],
             "v_label_corrected": c["label_cramers_v_corrected"],
             "exclude_base_wilson": iv["exclude_base_wilson"],
             "exclude_base_word": iv["exclude_base_word_resampled"]}
        for g in RC_GROUPINGS:
            p[f"ami_{g}"] = c[g]
            p[f"ami_{g}_seed_range"] = stab["ami_range"][g]
        e(("run-rep-communities", "USED", REP_KEY[rep], p))


def check_rc_findings(nodes: list) -> list[tuple[str, str]]:
    """Every number the find-rc-* statements quote, recomputed; stop on a miss.

    Formatted as the statements write them: AMI to 3 decimals, /t/-/d/ AMI
    and seed ranges to 4. Also checks the orderings the statements assert.
    """
    d = json.loads(RC_SUMMARY.read_text())
    comm = json.loads(COMM_SUMMARY.read_text())
    R = d["representations"]
    w, j, m = R["wav2vec2"], R["wavjepa"], R["mfcc"]
    sp = d["site_phone_reference"]
    f3 = lambda x: f"{x:.3f}"
    f4 = lambda x: f"{x:.4f}"
    rng = lambda r, g: "–".join(f4(x) for x in r["stability"]["ami_range"][g])
    C = lambda r, g: r["comparison"][g]
    want = {
        "find-rc-mfcc": [f"{d['sites']['n']:,}", f"k = {d['setup']['k']}",
                         *(f3(C(r, g)) for g in ("pair", "vowel", "following")
                           for r in (m, w, j)),
                         f"layer {w['layer']}", f"layer {j['layer']}",
                         rng(m, "pair"), rng(w, "pair"),
                         f3(sp["comparison"]["pair"])],
        "find-rc-speaker": [f3(C(j, "speaker")), rng(j, "speaker"),
                            f3(C(m, "speaker")), f3(C(w, "speaker")),
                            f3(C(j, "pair")), f"layer {w['layer']}"],
        "find-rc-voicing": [*(f4(C(r, "label")) for r in (w, m, j)),
                            *(f3(C(r, "label_cramers_v_corrected")) for r in (w, m, j)),
                            rng(w, "label"), rng(j, "label"), rng(m, "label"),
                            f4(w["k_sensitivity"]["5"]["ami"]["label"]),
                            f4(m["k_sensitivity"]["5"]["ami"]["label"]),
                            f4(w["k_sensitivity"]["30"]["ami"]["label"]),
                            f4(m["k_sensitivity"]["30"]["ami"]["label"]),
                            f4(sp["comparison"]["label"]),
                            f3(d["label_ami_of_the_labels_themselves"]["pair"])],
        "find-rc-stability": [*(f3(r["stability"]["ari_vs_seeded_median"])
                                for r in (w, j, m)),
                              f3(comm["stability"]["ari_vs_seeded_median"]),
                              *(f3(C(r, g)) for r in (m, w, j)
                                for g in ("word", "pair")),
                              f3(sp["comparison"]["word"])],
        "find-rc-site-phone": [f"The {comm['run']['n_communities']} ",
                               f4(comm["run"]["modularity"]),
                               f3(comm["label_association"]["community_x_label"]["cramers_v"]),
                               f3(comm["label_association"]["pair_x_label"]["cramers_v"]),
                               f3(comm["label_association"]["vowel_x_label"]["cramers_v"]),
                               f"{sp['intervals']['exclude_base_word_resampled']} of the "
                               f"{sp['intervals']['n_communities']}",
                               f"{100 * comm['graph']['base_share_d']:.1f}%",
                               f"at {sp['intervals']['exclude_base_wilson']}"],
    }
    # The two summaries must describe the same Site-Phone partition.
    if sp["intervals"]["n_communities"] != comm["run"]["n_communities"]:
        sys.exit("the two summaries describe different Site-Phone partitions")
    # The orderings the statements assert.
    phon = ("vowel", "following", "pair")
    pairs = [(rep, g) for rep in R for g in RC_GROUPINGS]
    disjoint = lambda g, a, b: (a["stability"]["ami_range"][g][1] < b["stability"]["ami_range"][g][0]
                                or b["stability"]["ami_range"][g][1] < a["stability"]["ami_range"][g][0])
    claims = {
        "MFCC above both encoders on every phonetic grouping":
            all(C(m, g) > max(C(w, g), C(j, g)) for g in phon),
        "WavJEPA's speaker AMI is the largest value in the table":
            all(C(j, "speaker") >= C(R[r], g) for r, g in pairs),
        "WavJEPA lowest of the three on every phonetic grouping":
            all(C(j, g) < min(C(w, g), C(m, g)) for g in phon),
        "seed ranges disjoint for every grouping but /t/-/d/":
            all(disjoint(g, a, b) for g in RC_GROUPINGS if g != "label"
                for a, b in ((w, j), (w, m), (j, m))),
        "/t/-/d/: wav2vec2 and MFCC overlap; WavJEPA below both":
            not disjoint("label", w, m) and disjoint("label", w, j),
    }
    bad = [c for c, ok in claims.items() if not ok]
    statements = {n["key"]: n["statement"] for n in nodes if n["key"] in want}
    missing = [(k, x) for k, xs in want.items() for x in xs if x not in statements[k]]
    if missing or bad:
        sys.exit(f"find-rc-* statements disagree with the summaries: {missing} {bad}")
    return [(k, x) for k, xs in want.items() for x in xs] + [(c, "holds") for c in claims]


CRITERION_LINE = re.compile(r"^Criterion (\d) — (.+): (PASS|FAIL)$")


def outcome_nodes(nodes: list, edges: list) -> None:
    """One Outcome per rev 3 criterion, parsed from the criteria report."""
    lines = CRITERIA_PATH.read_text().splitlines()
    found = 0
    for i, line in enumerate(lines):
        m = CRITERION_LINE.match(line)
        if not m:
            continue
        n, _, verdict = int(m[1]), m[2], m[3]
        key = f"outcome-{n}"
        nodes.append({"key": key, "labels": ["Outcome"],
                      "name": f"C{n}: {verdict}", "number": n,
                      "passed": verdict == "PASS", "verdict": verdict,
                      "detail": lines[i + 1].strip()})
        edges.append((key, "EVALUATES", f"crit-rev3-{n}", {}))
        edges.append((key, "FROM_RUN", "run-pilot", {}))
        edges.append((key, "SCORED_FROM", "curve:wavjepa:mean:joint:full", {}))
        if n == 1:
            edges.append((key, "SCORED_FROM", "curve:mfcc:mean:joint:full", {}))
        found += 1
    if found != 4:
        sys.exit(f"expected 4 criteria in {CRITERIA_PATH.name}, parsed {found}")


def artifact_links(nodes: list, edges: list, curves: dict, sg_curves: dict) -> None:
    """Commits from git; what each artifact reports; which came from the run."""
    for n in nodes:
        if "Artifact" in n["labels"]:
            out = subprocess.run(
                ["git", "log", "-1", "--format=%h", "--", n["path"]],
                cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
            n["commit"] = out or None
            n["name"] = Path(n["path"]).name

    def curves_where(**want):
        return [k for k, c in curves.items()
                if all(c[f] == v for f, v in want.items())]

    reports = {
        "art-criteria": [f"outcome-{i}" for i in (1, 2, 3, 4)],
        "art-results": list(curves),
        # Figure contents per scripts/plot_layer_curves.py's docstring.
        "art-fig-curves": curves_where(pooling="mean", subsampled=False),
        "art-fig-drop": curves_where(representation="wavjepa", pooling="mean"),
        "art-fig-controls": curves_where(pooling="mean", subsampled=False),
        "art-fig-pooling": curves_where(representation="wavjepa",
                                        condition="joint", subsampled=False),
    }
    for art, targets in reports.items():
        for t in targets:
            edges.append((art, "REPORTS", t, {}))
    for art in ("art-criteria", "art-results", "art-folds", "art-manifest",
                "art-fig-curves", "art-fig-drop", "art-fig-controls",
                "art-fig-pooling", "art-audit-sample", "art-audit-report"):
        edges.append((art, "FROM_RUN", "run-pilot", {}))
    for art in ("art-sg-results", "art-sg-summary"):
        for t in sg_curves:
            edges.append((art, "REPORTS", t, {}))
    for art in ("art-sg-results", "art-sg-cells", "art-sg-comparisons",
                "art-sg-summary", "art-sg-manifest"):
        edges.append((art, "FROM_RUN", "run-span-grid", {}))
    edges.append(("art-rc-summary", "FROM_RUN", "run-rep-communities", {}))


# ---------------------------------------------------------------------------
# Checks and Cypher
# ---------------------------------------------------------------------------

def check(nodes: list, edges: list) -> dict:
    keys = Counter(n["key"] for n in nodes)
    dupes = [k for k, c in keys.items() if c > 1]
    if dupes:
        sys.exit(f"duplicate node keys: {dupes}")
    label_of = {n["key"]: n["labels"][0] for n in nodes}
    missing = sorted({k for s, _, d, _ in edges for k in (s, d) if k not in label_of})
    if missing:
        sys.exit(f"edges point at nodes that don't exist: {missing}")
    return label_of


def lit(v) -> str:
    """A Cypher literal. JSON's string escapes are valid Cypher escapes."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    raise TypeError(f"no Cypher literal for {type(v).__name__}: {v!r}")


def props_map(d: dict) -> str:
    items = [f"{k}: {lit(v)}" for k, v in d.items() if v is not None]
    return "{" + ", ".join(items) + "}"


def write_cypher(nodes: list, edges: list, label_of: dict) -> None:
    labels = sorted({l for n in nodes for l in n["labels"]})
    out = ["// Generated by kg/build_design.py. Do not edit; edit kg/facts.py.",
           "// Replaces the design layer only (layer = 'design').", ""]
    for l in labels:
        out.append(f"CREATE CONSTRAINT {l.lower()}_key IF NOT EXISTS "
                   f"FOR (n:{l}) REQUIRE n.key IS UNIQUE;")
    out += ["", "MATCH (n) WHERE n.layer = 'design' DETACH DELETE n;", ""]
    for n in nodes:
        first, *rest = n["labels"]
        props = {k: v for k, v in n.items() if k not in ("key", "labels")}
        props["layer"] = "design"
        extra = "".join(f" SET n:{l}" for l in rest)
        out.append(f"MERGE (n:{first} {{key: {lit(n['key'])}}}) "
                   f"SET n += {props_map(props)}{extra};")
    out.append("")
    for s, rel, d, p in edges:
        line = (f"MATCH (a:{label_of[s]} {{key: {lit(s)}}}), "
                f"(b:{label_of[d]} {{key: {lit(d)}}}) MERGE (a)-[r:{rel}]->(b)")
        if p:
            line += f" SET r += {props_map(p)}"
        out.append(line + ";")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    nodes = [dict(n) for n in facts.NODES]
    edges = list(facts.EDGES)
    run_nodes(nodes, edges)
    curves = curve_nodes(nodes, edges)
    outcome_nodes(nodes, edges)
    span_grid_run(nodes, edges)
    sg_curves = span_grid_curves(nodes, edges)
    rep_communities_run(nodes, edges)
    rc_checked = check_rc_findings(nodes)
    # find-rc-speaker: "Layer 3 is WavJEPA's best on ɾ in the span grid, but
    # that best-layer interval is wide"; the layers chosen are the grid's best.
    rc_layers = json.loads(RC_SUMMARY.read_text())["setup"]["layers"]
    for rep in ("wav2vec2", "wavjepa"):
        c = sg_curves[f"curve:sg:{rep}:r"]
        if c["best_layer"] != rc_layers[rep]:
            sys.exit(f"{rep}: communities used layer {rc_layers[rep]}, the grid's "
                     f"best on ɾ is {c['best_layer']}")
    c = sg_curves["curve:sg:wavjepa:r"]
    rc_checked.append(("WavJEPA's best-layer interval on ɾ",
                       f"[{c['best_layer_lo']}, {c['best_layer_hi']}]"))
    artifact_links(nodes, edges, curves, sg_curves)
    label_of = check(nodes, edges)
    write_cypher(nodes, edges, label_of)

    print(f"wrote {OUT.relative_to(REPO)}: {len(nodes)} nodes, {len(edges)} edges")
    for label, c in sorted(Counter(l for n in nodes for l in n["labels"]).items()):
        print(f"  {label:<18} {c}")
    print(f"  {len({r for _, r, _, _ in edges})} relationship types: "
          + ", ".join(sorted({r for _, r, _, _ in edges})))

    # The numbers the Findings quote, recomputed from the CSV for review.
    print("\nbest WavJEPA layer, mean pooling, full run:")
    for cond in COND_KEY:
        c = curves[f"curve:wavjepa:mean:{cond}:full"]
        print(f"  {cond:<17} layer {c['best_layer']:>2}  {c['best_balanced_acc']:.4f}"
              f"  spread {c['spread']:.4f}")
    print("subsampling cost at the best layer (full - subsampled):")
    for cond in ("random", "speaker_disjoint", "word_disjoint"):
        full = curves[f"curve:wavjepa:mean:{cond}:full"]["best_balanced_acc"]
        sub = curves[f"curve:wavjepa:mean:{cond}:subsampled"]["best_balanced_acc"]
        print(f"  {cond:<17} {full - sub:+.4f}")

    # The numbers the span grid's Findings quote, from results/span-grid/.
    print("\nspan grid, best layer per block (find-sg-*):")
    for key, c in sg_curves.items():
        print(f"  {key:<30} layer {c['best_layer']:>2} [{c['best_layer_lo']}, "
              f"{c['best_layer_hi']}]  {c['best_balanced_acc']:.4f}  "
              f"[{c['best_ci_lo']:.4f}, {c['best_ci_hi']:.4f}]  control p {c['control_p']:.4f}")
    print("span grid comparisons (C1, C4, C5, C7):")
    with SG_COMPARISONS.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["comparison"] != "C6":
                print(f"  {r['comparison']}  {r['a']:<16} - {r['b']:<20} "
                      f"{float(r['gap']):+.4f}  [{float(r['ci_lo']):+.4f}, "
                      f"{float(r['ci_hi']):+.4f}]  {r['call']}")

    print("\nrepresentation communities (find-rc-*), each found in its statement:")
    for key, x in rc_checked:
        print(f"  {key:<56} {x}")


if __name__ == "__main__":
    main()
