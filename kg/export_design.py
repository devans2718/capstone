#!/usr/bin/env python3
"""Export the design layer and GraphRAG's reviewed retrievals to JSON, for the GNN notebook.

HOW TO RUN (from the repo root, in WSL, with the server up: kg/compose.yaml)
    .venv-app/bin/python kg/export_design.py
    Writes kg/gnn/design-graph.json. Connection as in kg/graphrag.py.

WHAT IT WRITES
    nodes      every node with layer = 'design': key, label, and its string
               properties joined into one `text` (the GNN's input feature)
    edges      every relationship between two design nodes: source key,
               type, target key. Edges into the corpus layer (the audit's
               SAMPLES) are left out, with their word and clip properties
    questions  one per retrieval in graphrag.RETRIEVALS, with the parameters
               fixed below. `gold` is every design-node key that appears in
               that retrieval's rows: the evidence the reviewed Cypher
               returns, which the GNN rankers are scored against

    Questions 1-3 are the saved runs in kg/graphrag-runs/ (same retrieval,
    same parameters). 4 and 5 are new, and the retrieval is assigned here,
    not chosen by the router, so no LLM call is made.

READ-ONLY
    Every query runs with routing_=READ, so the server refuses any write.
    The design layer holds no words, speaker codes or timestamps (the Buckeye
    rule in kg/graphrag.py), so the JSON is safe to commit.
"""

from __future__ import annotations

import json
from pathlib import Path

from neo4j import GraphDatabase, RoutingControl

from graphrag import AUTH, REPO, RETRIEVALS, URI

OUT = REPO / "kg" / "gnn" / "design-graph.json"

QUESTIONS = [
    ("What's a flap?", "concept", {"term": "flap", "criterion": 0}),
    ("How did criterion 2 get its final form?", "criterion_provenance", {"term": "", "criterion": 2}),
    ("Why did the pilot fail, and what's planned to test it?", "why_failed", {"term": "", "criterion": 0}),
    ("How does WavJEPA's best layer compare with MFCC under each split condition?", "word_axis",
     {"term": "", "criterion": 0}),
    ("What does registration revision 3 specify that the pilot never used?", "registered_not_run",
     {"term": "", "criterion": 0}),
]

NODES = """
    MATCH (n {layer: 'design'})
    RETURN n.key AS key, labels(n)[0] AS label, properties(n) AS props
    ORDER BY key"""
EDGES = """
    MATCH (a {layer: 'design'})-[r]->(b {layer: 'design'})
    RETURN a.key AS source, type(r) AS type, b.key AS target
    ORDER BY source, type, target"""
SKIP_PROPS = {"key", "layer", "commit"}   # identifiers, not content


def strings(x):
    """Every string inside a row value, recursing into lists and maps."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from strings(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from strings(y)


def gold_keys(rows: dict[str, list[dict]], keys: set[str]) -> list[str]:
    """Design-node keys named anywhere in the rows, whole values or tokens.

    Tokens matter for `evidenced_by`, whose entries read "Artifact <key> (status: ...)".
    """
    found = set()
    for s in strings(rows):
        for tok in [s, *s.replace(":", " ").split()]:
            if tok in keys:
                found.add(tok)
    return sorted(found)


def main() -> None:
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        q = lambda cypher, **p: driver.execute_query(cypher, p, routing_=RoutingControl.READ).records
        nodes = []
        for r in q(NODES):
            text = " ".join(str(v) for k, v in sorted(r["props"].items())
                            if k not in SKIP_PROPS and isinstance(v, str))
            nodes.append({"key": r["key"], "label": r["label"], "text": text})
        keys = {n["key"] for n in nodes}
        edges = [r.data() for r in q(EDGES)]
        questions = []
        for question, retrieval, params in QUESTIONS:
            rows = {name: [rec.data() for rec in q(cypher, **params)]
                    for name, cypher in RETRIEVALS[retrieval]["statements"].items()}
            questions.append({"question": question, "retrieval": retrieval, "params": params,
                              "gold": gold_keys(rows, keys)})

    assert len(keys) == len(nodes), "duplicate design keys"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"nodes": nodes, "edges": edges, "questions": questions},
                              indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(nodes)} nodes, {len(edges)} edges, {len(set(e['type'] for e in edges))} relationship types -> {OUT}")
    for qq in questions:
        print(f"  {qq['retrieval']:22s} {len(qq['gold']):3d} gold  {qq['question']}")


if __name__ == "__main__":
    main()
