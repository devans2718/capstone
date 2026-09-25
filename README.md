# capstone

Layer-wise probing of frozen audio-JEPA representations. Primary encoder:
WavJEPA. Baselines: wav2vec 2.0, HuBERT, WavLM behind one extraction
interface. Research questions, scope, and design live in the Obsidian vault
(`Projects/Capstone/`) — this repo is the executable side only.

## Layout

| Path | What | Committed? |
|---|---|---|
| `Dockerfile`, `compose.yaml` | The environment. Same image locally and on cloud GPUs. | yes |
| `configs/` | `local.yaml` / `cloud.yaml` — differ only in values, never structure | yes |
| `src/capstone/` | Package: `encoders/` (4-model layer-wise interface), `data/`, `alignment/`, `probes/` | yes |
| `scripts/` | Thin entrypoints (encode, probe, eda), plus `smoke_*` / `probe_*` diagnostics that establish facts about a checkpoint | yes |
| `app/` | Streamlit interface for the Challenge 2 slice. Runs on `requirements-app.txt` alone — no GPU, no CUDA image. | yes |
| `tests/` | Frame arithmetic and other pure logic. Deliberately torch-free so it runs in seconds anywhere. | yes |
| `stimuli/` | Self-recorded audio stimuli (код/кот). Deliberately committed — irreplaceable, GitHub is their offsite backup. | yes |
| `results/` | Promoted metrics and figures | yes |
| `notebooks/` | Exploration | yes |
| `ops/` | Backup script (runs nightly inside WSL) | yes |
| `/data` (mount) | `corpora/`, `embeddings/`, `checkpoints/` — never inside the repo | no |

## Dev workflow

```bash
docker compose run --rm dev                          # shell, GPU attached
docker compose run --rm dev python -m pytest         # fast tests (~2 s)
docker compose run --rm dev python -m pytest -m model # + checkpoint-dependent
```

Tests marked `model` need the 785 MB checkpoint and a GPU, so they are
deselected by default (see `pyproject.toml`). Run them after any bump to
`transformers` or the pinned checkpoint revision: WavJEPA loads with
`trust_remote_code=True`, so upstream can change its frame arithmetic under
already-cached embeddings, and that pair of tests is the tripwire.

The repo bind-mounts to `/workspace`, data to `/data`. Configs reference
`/data` by absolute path, so a cloud instance with the same mounts runs the
same code with `configs/cloud.yaml` and zero edits.

## Challenge 2 — the flapping evidence slice

**What it does.** Answers questions about which flap sites a probe got wrong,
and hands back the evidence rather than a summary: the rows, the denominators
they must be read against, and the audio each row describes. The worked
question is the condition differential — *which /d/ sites are classified
correctly when word types recur between train and test, and stop being correct
when they do not?* — because that is the one a probe leaning on word identity
should fail.

**How to run it.**

```bash
uv venv .venv-app && uv pip install --python .venv-app/bin/python -r requirements-app.txt
.venv-app/bin/python scripts/count_flap_sites.py --parquet results/flapping/sites-20260924.parquet
.venv-app/bin/python scripts/make_synthetic_predictions.py
.venv-app/bin/streamlit run app/streamlit_app.py
```

No GPU and no CUDA image. `plan_excerpt`, `Excerpt` and `FrameGeometry` import
torch only under `TYPE_CHECKING` and inside function bodies, so the alignment
arithmetic the interface depends on is reachable from a plain virtualenv.
(`python -m venv` needs `python3.12-venv` installed; `uv` does not.)

**What data it uses.** The Buckeye inventory of 10,392 flap sites — 7,863
underlying /t/, 2,529 /d/ — built by `scripts/count_flap_sites.py` and
committed as `results/flapping/sites-20260924.parquet`. Predictions are written
to `/data/embeddings/`, never into the repository; the run manifest is
committed beside the inventory so a figure can be traced to a seed, an
inventory hash and a commit. Corpus audio mounts at `/data` and stays there —
Buckeye is distributed under a registration agreement with OSU.

**Every probability in this slice is synthetic.** No probe has been trained.
The pre-registration in `experiments/pilot-flapping/README.md` finalizes three
provisional thresholds in a revision that has not been written, and running
before it is pushed voids the pre-registration. Synthetic predictions cost
nothing that matters here: retrieval is exactly as testable against noise as
against a real result, because retrieval does not depend on whether a
probability is meaningful — only on whether it round-trips with its provenance
intact.

**What AI capability it demonstrates.** Tool-mediated retrieval with enforced
provenance. A language model may do exactly two things: turn a question into a
typed tool call, and narrate rows in prose where every number is traceable to a
returned field. It computes no statistic, infers no label, and asserts no
cause; if the tools return no rows the answer is "no rows". The mechanisms that
make that more than an intention:

- A **coverage block on every response**, including uninformative ones. 40 of
  10,392 sites are scored, and a 40-row table with no denominator reads exactly
  like a complete result.
- **Three eligibility states kept distinct** — discarded by the split, scored,
  and never reached. The third is the *absence* of a row, and collapsing any
  two of them into "no data" is the defect test case 4 exists to catch.
- **Clips planned, never re-derived.** `plan_clip` goes through `plan_excerpt`
  and `excerpt_samples`, the same call `encode_excerpt` makes, so what you hear
  is what the encoder would consume. Planning from the rounded integer anchor
  instead disagrees on 13% of windows.
- **`synthetic` asserted in four places**, one of them a guard that raises
  rather than a flag callers must remember.
- **Two uncertainty channels, never merged.** Label-and-population uncertainty
  is real and citable today; probe uncertainty is noise. A blended figure would
  be unfixable, because you could not tell which half had moved.

**What remains.** Deferred by design and all additive — `group_errors` and
contingency tables, bootstrap confidence intervals, feature-space nearest
neighbours, manifest hash *refusal*, the word-identity lookup baseline,
cross-encoder comparison, and any trained probe. Grouping errors by word is
deliberately not offered at all: under a word-disjoint condition it
hand-reconstructs the 99.5% lexical lookup the split exists to prevent.

Open decisions are tracked in
`docs/challenges/challenge-2-slice-decisions.md`. D1 is the one that reaches
beyond the slice: `frames_for_interval` disagrees with its own docstring when a
target starts exactly on a frame onset, which `plan_excerpt`'s 200 ms centring
makes true of every site.

## Cloud

Build and push the image, pull it on the instance (Lambda / Vast / RunPod),
populate `/data` with the download scripts, run with `--gpus all` and
`configs/cloud.yaml`. The image is the portability artifact — if it needs a
rebuild to run remotely, that is a bug.

## Backups

Nightly restic from WSL: tier 1 (this repo) + tier 2 (`/data/embeddings`,
`/data/checkpoints`) to D: and Backblaze B2. `/data/corpora` is deliberately
not backed up — re-downloadable. Full design: vault note
`Reference/Desktop/Capstone Dev Environment (Docker + WSL).md`.

## Successor-repo note

This is the summer repo. If a clean repo is started for the semester, fork or
copy this layout wholesale — the `/data` contract, config split, and stimuli
policy are the parts that must survive the move.
