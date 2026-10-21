# kg/ — the study as a knowledge graph, and GraphRAG over it

The flapping pilot's design, results and reasoning in Neo4j: the pre-registration
and its revisions, criteria and outcomes, the findings that explain each failure
and what each rests on, plus a corpus layer of sites, tokens and speakers.
`kg/graphrag.py` answers natural-language questions from it (Challenge 3
Co-Design).

| File | What it is |
|---|---|
| `facts.py` | The hand-written facts: nodes and edges. Edit here, then rebuild |
| `build_design.py` | Adds what the repo already records (run, curves, outcomes, commits) and writes `design.cypher` |
| `design.cypher` | Generated. Loads the design layer, replacing it on reload |
| `build_corpus.py` | Loads the corpus layer straight into Neo4j; checks it against the pilot run |
| `compose.yaml` | The Neo4j server, pinned |
| `queries/` | Nine demo queries (Challenge 3 AI Design); `run_all.sh` runs them all |
| `graphrag.py` | Question → retrieval → Cypher → rows → LLM answer, with the evidence printed |
| `graphrag-runs/` | Saved GraphRAG runs (JSON) |
| `export_design.py` | Read-only export of the design layer and the five retrievals' evidence, for the GNN |
| `gnn/` | The GNN stage: `design-graph.json` (generated) and `design_gnn.ipynb` |

## Setup

From the repo root, in WSL, with Docker Desktop running.

```bash
# 1. Python environment (once). Includes the neo4j driver and anthropic.
uv venv .venv-app && uv pip install --python .venv-app/bin/python -r requirements-app.txt

# 2. Neo4j (once: create the volume; then start the server)
docker volume create neo4j-capstone-data
docker compose -f kg/compose.yaml up -d
#    Browser http://localhost:7474 · Bolt bolt://localhost:7687 · neo4j / phonoprobe-local

# 3. Load the design layer
python3 kg/build_design.py
docker exec -i neo4j-capstone cypher-shell -u neo4j -p phonoprobe-local < kg/design.cypher

# 4. Load the corpus layer (optional for GraphRAG; needed for queries 08-09)
.venv-app/bin/python kg/build_corpus.py

# 5. API key for the LLM calls, in .env at the repo root (gitignored)
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

Reloading the design layer drops the edges that join it to the corpus layer, so
re-run step 4 after step 3. The corpus layer reads Buckeye-derived files that
this repository doesn't carry. With a registered copy, `scripts/count_flap_sites.py`
rebuilds the site inventory; the lemma table and the listening audit don't
regenerate, so without them the script loads the rest and says what it skipped.
See the docstring in `build_corpus.py`.

## GraphRAG

```bash
.venv-app/bin/python kg/graphrag.py "Why did the pilot fail, and what's planned to test it?"
.venv-app/bin/python kg/graphrag.py --save kg/graphrag-runs "What's a flap?"
```

1. **Route.** The LLM picks one of five retrievals and fills its parameter. It
   chooses; it never writes Cypher, so every query that runs is one a human has
   read.
2. **Retrieve.** The chosen Cypher runs against the graph.
3. **Answer.** The LLM answers from the returned rows only, citing node keys,
   and keeps each finding's status as the graph gives it.

| Retrieval | Kind | Path |
|---|---|---|
| `concept` | basic | Concept, its related concepts and experiments |
| `criterion_provenance` | multi-hop | Registration → Criterion → SUPERSEDES → Criterion; Paper → JUSTIFIES |
| `why_failed` | research | Run → failed Outcome → Criterion; Finding → EXPLAINS; Finding → EVIDENCED_BY |
| `word_axis` | results | WavJEPA and MFCC curves per split condition |
| `registered_not_run` | plans | Revision 3 SPECIFIES, never USED by a run |

Model: `claude-opus-5-5`. Both LLM calls go through `complete()`, so another
endpoint means rewriting one function. The retrievals read the design layer
only, which holds aggregates and no Buckeye words, speakers or timestamps, so
saved runs are safe to commit.

### The co-design iteration (2026-10-14)

The `find-width` finding said a cue outside a site's 2–4 frames "is invisible to
the probe". The first GraphRAG answer to the research question repeated it with
a citation (`graphrag-runs/20261014-213925-why_failed.json`). Checked against
the code, it holds for MFCC only. WavJEPA encodes the flap ± 200 ms as one
excerpt and the frames are selected afterwards
(`scripts/extract_site_features.py:206-230`), so attention can carry the vowel
into the flap frames. The finding was corrected in `facts.py` and the graph
reloaded. The re-run answer separates the two encoders
(`graphrag-runs/20261014-215107-why_failed.json`).

## GNN ranker (Challenge 3, GNN stage)

Does a GNN rank the evidence better than text similarity? This applies the course
notebook's recipe (TF-IDF features, two-layer GraphSAGE, link prediction,
seed → expand → score) to the design layer, and scores it against what the
reviewed Cypher returns.

```bash
# 1. Export (needs the server; read-only)
.venv-app/bin/python kg/export_design.py      # -> kg/gnn/design-graph.json

# 2. The notebook needs no Neo4j. It runs in an environment with torch,
#    torch_geometric, scikit-learn, networkx and matplotlib, from kg/gnn/
#    (GNN_FIG_DIR=<dir> also saves the figures)
```

- **Gold** for each of five questions is every design-node key in its
  retrieval's rows. Questions 1–3 are the saved runs; 4–5 are assigned their
  retrieval directly, so the export makes no LLM call.
- **Homogeneous GraphSAGE, not `to_hetero`.** The design layer has 31
  relationship types over 146 nodes, and 16 of them have five edges or fewer.
- **Result:** at the course notebook's settings, no ranker gets the reviewed
  evidence into its top 5 (mean hits@5 / possible: 0.10 vector only, 0.00 graph
  expansion, 0.00 graph + GNN). Text seeding is where it fails: the questions'
  words aren't the evidence nodes' words, so the gold set is mostly never a
  candidate. Section 4b varies features, seeds and hops; the GNN ordering
  only pulls ahead (0.22–0.32) when raw TF-IDF and five seeds widen the
  candidate set. That's five questions, so it's a pattern, not a result, and
  the evaluation isn't question-disjoint (the GNN trains on the gold edges).
