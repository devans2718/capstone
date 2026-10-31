#!/usr/bin/env python3
"""GraphRAG over the capstone knowledge graph: question -> Cypher -> rows -> answer.

HOW TO RUN (from the repo root, in WSL, with neo4j-capstone up)
    .venv-app/bin/python kg/graphrag.py "Why did the pilot fail?"
    .venv-app/bin/python kg/graphrag.py --save kg/graphrag-runs "What's a flap?"

    Needs ANTHROPIC_API_KEY, from the environment or from .env at the repo
    root (gitignored). Neo4j connection as in kg/build_corpus.py.

WHAT IT DOES
    1. Route. The LLM picks one retrieval from RETRIEVALS below, plus its
       parameter. It chooses; it never writes Cypher. Every query that can
       run is one a human has read.
    2. Retrieve. The chosen Cypher runs against the graph.
    3. Answer. The LLM gets the question and the returned rows, and is told
       to answer from the rows only, citing node keys.
    Rows are printed before the answer, so the evidence is always on screen.

SWAPPING THE MODEL
    Both LLM calls go through complete(). Moving to another endpoint (e.g. a
    local Ollama model) means rewriting that one function.

THE COMMUNITY SUMMARY (--community)
    Adds COMMUNITY_STATEMENTS' rows to whatever the router retrieved: the
    Louvain communities that kg/build_communities.py writes over the flaps'
    phonetic environments. Kept out of RETRIEVALS on purpose, so the router's
    menu, and the five evaluated questions the app caches, stay as they were.

THE BUCKEYE RULE
    The retrievals read the design layer (aggregates, no words, speaker
    codes or timestamps), so a saved run is safe to commit. The Site-Phone
    community summary (the `communities` retrieval's site_phone statement,
    and --community) reads the community layer, which is built from the
    corpus, but returns aggregates only: community sizes, /t/-/d/ counts and
    phone labels, never a Site, Token, Lemma or Speaker key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parents[1]
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "phonoprobe-local"))
MODEL = "claude-opus-5-5"

# ---------------------------------------------------------------------------
# Retrievals. Each is a description (what the router sees) and one or more
# named Cypher statements. Parameters: $term (string), $criterion (integer).
# ---------------------------------------------------------------------------

RETRIEVALS: dict[str, dict] = {
    "concept": {
        "description": "Basic lookup: define a linguistics or ML concept "
                       "(flap, underlying form, probing, ...) and what it "
                       "relates to. Needs `term`.",
        "statements": {
            "concept": """
                MATCH (c:Concept) WHERE toLower(c.term) CONTAINS toLower($term)
                OPTIONAL MATCH (c)-[:BROADER|INVOLVES]-(d:Concept)
                OPTIONAL MATCH (e:Experiment)-[:INVOLVES]->(c)
                RETURN c.key AS key, c.term AS term, c.definition AS definition,
                       collect(DISTINCT d.key + ': ' + d.term) AS related,
                       collect(DISTINCT e.name) AS experiments
                ORDER BY term""",
        },
    },
    "criterion_provenance": {
        "description": "Multi-hop: how one pre-registered success criterion "
                       "evolved across registration revisions, which paper "
                       "justified it, and what it superseded. Needs "
                       "`criterion` (1-4).",
        "statements": {
            "revisions": """
                MATCH (r:Registration)-[:SETS_CRITERION]->(c:Criterion {number: $criterion})
                OPTIONAL MATCH (c)-[:SUPERSEDES]->(prev:Criterion)
                OPTIONAL MATCH (p:Paper)-[j:JUSTIFIES]->(c)
                RETURN r.key AS registration, r.revision AS revision,
                       r.commit AS commit, r.date AS date,
                       c.key AS criterion_key, c.statement AS statement,
                       c.threshold AS threshold, c.note AS note,
                       prev.key AS supersedes,
                       p.key AS paper, p.short AS paper_short, j.section AS section
                ORDER BY revision""",
        },
    },
    "why_failed": {
        "description": "Research question: why the pilot failed. Failed "
                       "criteria and their results, the candidate "
                       "explanations (Findings) with their status and full "
                       "statement, and what each rests on, including "
                       "planned analyses.",
        "statements": {
            "failed_criteria": """
                MATCH (:Run {run_id: 'pilot-20260926'})<-[:FROM_RUN]-(o:Outcome {passed: false})
                      -[:EVALUATES]->(c:Criterion)
                RETURN c.key AS key, c.number AS criterion, c.statement AS statement,
                       c.threshold AS threshold, o.detail AS result
                ORDER BY criterion""",
            "findings": """
                MATCH (:Run {run_id: 'pilot-20260926'})<-[:FROM_RUN]-(o:Outcome {passed: false})
                      -[:EVALUATES]->(c:Criterion)
                MATCH (f:Finding)-[:EXPLAINS]->(o)
                WITH f, collect(DISTINCT c.number) AS explains_failed_criteria
                OPTIONAL MATCH (f)-[e:EVIDENCED_BY]->(x)
                RETURN f.key AS key, f.candidate_cause AS cause, f.status AS status,
                       f.statement AS statement, explains_failed_criteria,
                       collect(DISTINCT labels(x)[0] + ' ' + x.key
                               + coalesce(' (status: ' + x.status + ')', '')
                               + coalesce(' - ' + x.description, '')
                               + coalesce(' ' + e.section, '')) AS evidenced_by
                ORDER BY key""",
        },
    },
    "word_axis": {
        "description": "Results: WavJEPA's best layer against the MFCC "
                       "baseline under each split condition (random, "
                       "speaker-disjoint, word-disjoint, joint).",
        "statements": {
            "curves": """
                MATCH (w:Curve {representation: 'wavjepa', pooling: 'mean', subsampled: false})
                      -[:UNDER_CONDITION]->(k:Condition)<-[:UNDER_CONDITION]-
                      (m:Curve {representation: 'mfcc', pooling: 'mean', subsampled: false})
                RETURN k.key AS condition_key, k.name AS condition, k.held_out AS held_out,
                       w.key AS wavjepa_curve, w.best_layer AS best_layer,
                       round(w.best_balanced_acc, 3) AS wavjepa,
                       round(m.best_balanced_acc, 3) AS mfcc,
                       round(w.best_balanced_acc - m.best_balanced_acc, 3) AS gap,
                       w.lookup_balanced_acc AS word_lookup
                ORDER BY wavjepa DESC""",
        },
    },
    "registered_not_run": {
        # Every specified item, flagged, since 2026-10-29 (approved by
        # Daniel): once the span grid used wav2vec2-base, the old "never
        # USED" filter returned nothing, and the model could not tell "all
        # used" from a gap in the graph.
        "description": "Plans: everything pre-registration revision 3 "
                       "specifies (encoders, analysis windows, probes, "
                       "controls), each flagged used or not used, with the "
                       "runs that used it, so what the pilot never used is "
                       "the items no pilot run used.",
        "statements": {
            "specified": """
                MATCH (:Registration {revision: 3})-[s:SPECIFIES]->(x)
                OPTIONAL MATCH (r:Run)-[:USED|REALIZES]->(x)
                WITH x, s, r ORDER BY r.key
                WITH x, s, collect(DISTINCT r.key) AS used_by
                RETURN x.key AS key, labels(x)[0] AS kind,
                       coalesce(x.name, x.key) AS item, s.role AS role,
                       size(used_by) > 0 AS used, used_by,
                       x.status AS status, x.note AS note
                ORDER BY used, key""",
        },
    },
    # Added 2026-10-29, approved by Daniel; live-only until it is evaluated,
    # so the app's five cached questions don't include it.
    "span_grid": {
        "description": "Results: what the span grid (the follow-up "
                       "registration, sg-rev1) found. Its findings, one per "
                       "registered comparison, with status and full "
                       "statement, and each span x representation's best "
                       "layer, balanced accuracy, interval and control-task "
                       "p-value.",
        "statements": {
            "findings": """
                MATCH (:Registration {key: 'sg-rev1'})<-[:REGISTERED_BY]-(:Run)
                      <-[:FROM_RUN]-(f:Finding)
                OPTIONAL MATCH (f)-[:EVIDENCED_BY]->(x)
                RETURN f.key AS key, f.question AS question, f.status AS status,
                       f.statement AS statement,
                       collect(DISTINCT x.key) AS evidenced_by
                ORDER BY key""",
            "curves": """
                MATCH (:Registration {key: 'sg-rev1'})<-[:REGISTERED_BY]-(r:Run)
                      <-[:FROM_RUN]-(c:Curve)
                OPTIONAL MATCH (c)-[:IN_ENVIRONMENT]->(e:Environment)
                RETURN c.key AS key, c.representation AS representation,
                       coalesce(e.name, c.variant) AS span,
                       c.best_layer AS best_layer,
                       [c.best_layer_lo, c.best_layer_hi] AS best_layer_range,
                       round(c.best_balanced_acc, 4) AS balanced_acc,
                       [round(c.best_ci_lo, 4), round(c.best_ci_hi, 4)] AS ci_95,
                       c.control_p AS control_p
                ORDER BY representation, span""",
        },
    },
}

# The Site-Phone communities' summary (community layer, aggregates only).
# Shared by the `communities` retrieval and by --community.
SITE_PHONE_SUMMARY = """
        MATCH (c:Community)
        WITH collect(c) AS cs, sum(c.n_d) AS d, sum(c.n_sites) AS n
        UNWIND cs AS c
        RETURN c.key AS key, c.n_sites AS sites, c.n_t AS t, c.n_d AS d,
               c.share_d AS share_d, round(toFloat(d) / n, 4) AS base_share_d,
               c.phones AS phones, c.modularity AS modularity
        ORDER BY c.number"""

# Added 2026-10-31, approved by Daniel; live-only until it is evaluated.
# Communities are not retrieval objects (unstable across seeds, confounded
# with word; decided 2026-10-30): this returns what was found about them,
# never community members as context for other questions. Its Site-Phone
# statement is named site_phone so it can't collide with --community's.
RETRIEVALS["communities"] = {
    "description": "Results about communities: Louvain groupings of the "
                   "flaps, (a) over the encoders' features (wav2vec2-base, "
                   "WavJEPA, MFCC; exploratory, 2026-10-30), with how much "
                   "each one's communities share with /t/-/d/, the phonetic "
                   "environment, speaker and word (AMI), and (b) the "
                   "Site-Phone communities over the flaps' phonetic "
                   "environments (built 10-28, re-run sorted 10-29), with "
                   "each one's size and /d/ share. What was found about "
                   "communities, not communities as evidence for other "
                   "questions.",
    "statements": {
        "findings": """
                MATCH (:Run {key: 'run-rep-communities'})<-[:FROM_RUN]-(f:Finding)
                OPTIONAL MATCH (f)-[:EVIDENCED_BY]->(x)
                RETURN f.key AS key, f.question AS question, f.status AS status,
                       f.statement AS statement,
                       collect(DISTINCT x.key) AS evidenced_by
                ORDER BY key""",
        "profile": """
                MATCH (r:Run {key: 'run-rep-communities'})-[u:USED]->(rep:Representation)
                RETURN rep.key AS key, rep.name AS representation, u.layer AS layer,
                       u.n_communities AS communities, u.modularity AS modularity,
                       u.ari_seed_median AS ari_across_seeds,
                       u.ami_label AS ami_t_d, u.ami_vowel AS ami_vowel,
                       u.ami_following AS ami_following, u.ami_pair AS ami_pair,
                       u.ami_speaker AS ami_speaker, u.ami_word AS ami_word,
                       u.ami_lemma AS ami_lemma, u.ami_site_phone AS ami_site_phone,
                       u.v_label_corrected AS cramers_v_t_d_corrected,
                       r.site_phone_ami_label AS site_phone_ami_t_d,
                       r.site_phone_ami_pair AS site_phone_ami_pair
                ORDER BY key""",
        "site_phone": SITE_PHONE_SUMMARY,
    },
}

# Not a retrieval the router can pick: added by --community (see the docstring).
COMMUNITY_STATEMENTS: dict[str, str] = {"communities": SITE_PHONE_SUMMARY}

# ---------------------------------------------------------------------------
# The one model call. Everything model-specific lives here.
# ---------------------------------------------------------------------------


def load_dotenv() -> None:
    """Put KEY=value lines from the repo's .env into the environment, if unset."""
    path = REPO / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))


_client: anthropic.Anthropic | None = None


def complete(system: str, user: str, schema: dict | None = None,
             effort: str = "medium") -> str:
    """Send one system + user message; return the text reply.

    With `schema`, the reply is constrained to JSON matching it.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    output_config: dict = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    response = _client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config=output_config,
    )
    if response.stop_reason == "refusal":
        raise SystemExit(f"model declined: {response.stop_details}")
    return "".join(b.text for b in response.content if b.type == "text")

# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


ROUTE_SYSTEM = """You route questions about a research study to one retrieval \
over its Neo4j knowledge graph. The study probes a speech encoder (WavJEPA) \
for underlying /t/ vs /d/ at English flaps, against an MFCC baseline, under a \
pre-registered design. Pick the single retrieval that best answers the \
question, and fill its parameter. Use "" for an unused `term` and 0 for an \
unused `criterion`.

Retrievals:
{menu}"""

ANSWER_SYSTEM = """You answer questions about a research study using only \
rows retrieved from its knowledge graph. Rules:
- Use only the rows. If they don't answer the question, say what's missing.
- Cite node keys in square brackets after each claim, e.g. [find-width].
- Keep each finding's status as the rows give it (open, ruled out, \
limitation, ...); don't upgrade a candidate explanation to a conclusion.
- Plain prose, at most about 200 words."""


def route(question: str) -> dict:
    menu = "\n".join(f"- {name}: {r['description']}"
                     for name, r in RETRIEVALS.items())
    schema = {
        "type": "object",
        "properties": {
            "retrieval": {"type": "string", "enum": list(RETRIEVALS)},
            "term": {"type": "string"},
            "criterion": {"type": "integer"},
        },
        "required": ["retrieval", "term", "criterion"],
        "additionalProperties": False,
    }
    reply = complete(ROUTE_SYSTEM.format(menu=menu), question, schema, effort="low")
    return json.loads(reply)


def statements(choice: dict) -> dict[str, str]:
    out = dict(RETRIEVALS[choice["retrieval"]]["statements"])
    if choice.get("community"):
        out.update(COMMUNITY_STATEMENTS)
    return out


def retrieve(choice: dict) -> dict[str, list[dict]]:
    params = {"term": choice["term"], "criterion": choice["criterion"]}
    out: dict[str, list[dict]] = {}
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        for name, cypher in statements(choice).items():
            records, _, _ = driver.execute_query(cypher, params)
            out[name] = [r.data() for r in records]
    return out


def answer(question: str, rows: dict[str, list[dict]]) -> str:
    context = json.dumps(rows, indent=1, ensure_ascii=False, default=str)
    return complete(ANSWER_SYSTEM,
                    f"Question: {question}\n\nRetrieved rows:\n{context}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question")
    ap.add_argument("--save", type=Path,
                    help="directory to write this run to, as JSON")
    ap.add_argument("--community", action="store_true",
                    help="add the community summary to the retrieved rows")
    args = ap.parse_args()
    load_dotenv()

    choice = route(args.question)
    if args.community:
        choice["community"] = True
    rows = retrieve(choice)
    text = answer(args.question, rows)

    print(f"QUESTION   {args.question}")
    print(f"RETRIEVAL  {choice['retrieval']}  "
          f"(term={choice['term']!r}, criterion={choice['criterion']})"
          + ("  + community summary" if args.community else ""))
    for name, cypher in statements(choice).items():
        print(f"\n--- {name}: Cypher")
        print("\n".join(l[16:] if l.startswith(" " * 16) else l
                        for l in cypher.strip("\n").splitlines()))
        print(f"--- {name}: {len(rows[name])} rows")
        for r in rows[name]:
            print(json.dumps(r, ensure_ascii=False, default=str))
    print(f"\nANSWER ({MODEL})\n{text}")

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = args.save / (f"{stamp}-{choice['retrieval']}"
                             + ("+community" if args.community else "") + ".json")
        path.write_text(json.dumps({
            "time": stamp, "model": MODEL, "question": args.question,
            "choice": choice, "rows": rows, "answer": text,
        }, indent=1, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8", newline="\n")
        print(f"\nsaved {path.relative_to(REPO) if path.is_absolute() else path}")


if __name__ == "__main__":
    main()
