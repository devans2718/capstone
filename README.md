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
| `app/` | Streamlit interface over the flapping pilot (Challenge 2). Runs on `requirements-app.txt` alone — no GPU, no CUDA image. | yes |
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

## Challenge 2 — can a speech model hear what a flap was?

In American English the *t* in **water** and the *d* in **ladder** are both
said as the same quick tap, a *flap*. The research question: **does a
self-supervised speech encoder (WavJEPA) keep a trace of whether a flap
was underlyingly /t/ or /d/, once memorised words and speakers are ruled
out, and does it do better than plain acoustic features (MFCC)?** The
pre-registration, fixed before any probe was trained, is
`experiments/pilot-flapping/README.md`. The run and its result are recorded
in `docs/challenges/challenge-2-enhancement-brief.md`, Amendments 3 and 4.

### What the application does

A Streamlit app (`app/streamlit_app.py`) over the pilot run. It takes a
reader from the pre-registered result to the individual sites behind it, the
audio of each one, and a human judgement of that audio. Every answer carries
its provenance: the run, the code commit, and the denominator it has to be
read against.

| Tab | What it shows |
|---|---|
| **Result** | The four pre-registered criteria parsed from `results/flapping/criteria-20260926.txt`, the committed figures, and a split selector that moves every other tab to that split |
| **Browse and listen** | The sites matching a filter (split, layer, pooling, /t/ or /d/, right or wrong), each playable as the exact audio samples the encoder read |
| **Compare two splits** | Sites whose prediction changed between two splits, e.g. right under `random`, wrong under `joint` |
| **What this can and can't tell you** | What the inventory left out, and per-layer 95% intervals from the run |
| **Look up one site** | One site's state: scored, excluded before scoring, or not reached, never shown as a zero |
| **Audit (blind)** | The human review: a pre-registered random sample of sites, judged by ear with nothing from the model on screen |
| **Ask the graph** | A question about the study answered from its knowledge graph by `kg/graphrag.py` (route → retrieve → answer): the chosen retrieval, its Cypher and rows, then the answer citing node keys. Needs Neo4j and `ANTHROPIC_API_KEY` (see `kg/README.md`); without them only this tab says so |

### The Human–AI workflow

**Human input → tools → result + evidence → human review.**

1. The human picks a split, a layer and filters. They are shown the
   pre-registered result first: *PILOT: FAILED, 1 of 4 criteria pass*.
2. Typed, tested tools (`capstone.retrieval.SliceStore`, DuckDB over parquet)
   return the matching sites. There is no language model and no free-form
   query path.
3. Every table comes with its denominators (shown / scored / eligible / in
   the corpus), and every site with its clip. The clip is planned through
   the same function the encoder used, so what you hear is what the model
   read.
4. The human reviews the **data** behind the result: is there a flap, is the
   word the one transcribed, is the clip centred? Doing this blind on a random
   sample, registered before listening (pre-registration §7), makes the review
   a measurement rather than a widget. Verdicts recorded while browsing are
   kept but never counted.

### Data used

- **Buckeye Corpus** of conversational American English (Pitt et al., 2007),
  40 speakers. The inventory holds **10,392 flap sites (7,863 /t/, 2,529
  /d/)**, built by `scripts/count_flap_sites.py` from the hand-corrected phone
  tier. It is written to `results/flapping/sites-20260924.parquet`
  (identifiers, times and labels, no audio) and stays local.
- **Nothing derived from Buckeye per site is in this repository.** OSU
  distributes the corpus only through a free registration, so the corpus,
  features and predictions live under `/data`, and the inventory, the
  lemma-overlap table and the audit sample are ignored (`.gitignore`). What
  is committed: the results CSVs, the criteria and audit reports, the
  figures, and run manifests with seeds and commits.
- **To rebuild from a registered copy:** register at
  [buckeyecorpus.osu.edu](https://buckeyecorpus.osu.edu), place the archives
  as `scripts/setup_buckeye.sh` describes and run it, then
  `scripts/count_flap_sites.py` regenerates the inventory deterministically.

### AI capability

- **Encoder:** WavJEPA, a joint-embedding predictive architecture trained on
  raw waveforms, frozen. Its 13 layers are read at each flap site, pooled
  over the site (mean, max or midpoint).
- **Probes:** a logistic-regression probe per layer, trained under four splits
  that close the shortcuts one at a time:
  - `random`;
  - `speaker_disjoint` (no test speaker seen in training);
  - `word_disjoint` (no test word form seen);
  - `joint` (neither). This is the pre-registered test.

  Predictions are out-of-fold, so every site is scored by a probe that never
  trained on it.
- **The app:** typed retrieval with enforced provenance. Every response carries a
  coverage block. Three eligibility states stay distinct. The run's own
  manifest decides the banner (real or synthetic). And a query that fails to
  pin one prediction per site raises an error instead of double-counting.

### Baseline and evaluation

- **Baselines:**
  - **MFCC** features through the same probe: acoustics with no learned
    structure.
  - A **word-identity lookup**, which uses no audio at all. It scores 0.94 on
    `random` because the label is almost always determined by the word
    (pre-registration §1a), and collapses to exactly 0.50 under `joint`.
  - The **Hewitt & Liang control task**, used here as a leak check.
- **Metric:** balanced accuracy (chance 0.50), plus /d/ F1. Intervals come from a
  10,000-resample bootstrap over word types.
- **Pre-registered criteria**, all scored on `joint` (§5):
  1. The best layer beats MFCC by ≥ 0.05, with the interval's lower bound
     above 0.
  2. The control task shows no leak.
  3. The layer curve is not flat.
  4. /d/ F1 is at least 0.45.
- **Result: the pilot failed, 1 of 4.** Only the leak check passed. The
  numbers, figures and full report are on the app's **Result** tab and in
  `results/flapping/`. Closing word identity costs most of the score, and
  closing speakers much less (brief, A3.2).
- **Human evaluation:** the blind listening audit of 80 sampled sites (40 /t/,
  40 /d/), **all 80 reviewed**. Every word matched the transcript and every
  flap sat centred in its clip; a flap was heard at 38 of 40 /t/ sites and at
  all 34 judged /d/ sites (6 can't tell). Report:
  `results/flapping/audit-20260928.txt`.
- **Tests:** `pytest`, in two environments:
  - the app environment runs retrieval, splits, probes and audit;
  - the CUDA image runs frame geometry and the checkpoint tests (`-m model`).

### How to run the application

**Deployment:** runs locally because Buckeye's license restricts
redistributing its audio. A live demo is available in person.

It needs no GPU and no CUDA image:

```bash
uv venv .venv-app && uv pip install --python .venv-app/bin/python -r requirements-app.txt
.venv-app/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
.venv-app/bin/python -m pytest tests          # app-environment tests
```

- **Inputs it expects under `/data`:**
  - the corpus at `/data/corpora/buckeye`, obtained from its distributor under
    its license;
  - the pilot's predictions and manifest at
    `/data/embeddings/probe-run-20260926/`.
- **Rebuilding the predictions:** the docstrings of
  `scripts/extract_site_features.py`, `scripts/run_probes.py` and
  `scripts/plot_layer_curves.py` say how (extraction runs in the CUDA image).
- **Without `/data`:** the Result tab, figures and criteria still render from
  the committed files.
- **The audit:** after listening, `.venv-app/bin/python
  scripts/summarize_audit.py` writes the audit report under
  `results/flapping/`.

### Limitations

- **The pilot failed.** On `joint`, the best WavJEPA layer (3) reaches 0.544
  balanced accuracy against MFCC's 0.527. The gap is +0.016, its interval
  includes 0, and the layer curve spans only 0.025. This does not show the
  encoder lacks the information, but this design did not find it.
- **"Unseen words" means unseen word forms.** The split is disjoint by word
  form, not lemma: under `joint`, **25.1% of test sites** have their lemma
  (`get` / `got`, `do` / `did`) in the probe's training set with the same
  label. Criterion 1 re-scored on the other 7,586 sites does not pass either.
- **Limited power.** At this size, MFCC's own interval under `joint` is about
  **±0.05** balanced accuracy, as wide as the effect criterion 1 asks for. A
  real but small advantage would not be distinguishable.
- **One encoder so far.** WavJEPA only. wav2vec 2.0 was registered as
  exploratory and has not run, so nothing here compares training objectives.
- **The inventory is the tidy half.** 21% of flap-bearing words were excluded
  because their transcribed and canonical pronunciations differ in length.
  These are the most reduced tokens.
- **The audit clears the sites, not the labels, and it has one untrained
  listener.** It found no sign that the failure comes from badly cut or
  mislocated sites: the lowest Wilson lower bound is 0.835 (flap, /t/). But
  the underlying /t/ or /d/ comes from the word's dictionary form and cannot
  be heard under a flap, so the audit does not check it. Its rates bound what
  one ear agrees with, nothing more.

### Next steps

Each is a new registration. §5 forbids adjusting this one and rerunning
it.

- **A lemma-disjoint split.**
- **A different site:** wider than 2–4 frames, or with context beyond the
  flap interval.
- **A stronger probe.**
- **The comparison encoders** (wav2vec 2.0, then HuBERT and WavLM), for the
  layer-wise comparison across training objectives.
- **ABX discrimination** as a probe-free measure.

## References

**Dataset**
- Pitt, M. A., Dilley, L., Johnson, K., Kiesling, S., Raymond, W., Hume, E.,
  & Fosler-Lussier, E. (2007). *Buckeye Corpus of Conversational Speech*
  (2nd release). Department of Psychology, Ohio State University. Distributed
  under a license; not redistributed here.

**Papers**
- Pasad, A., Chou, J.-C., & Livescu, K. (2022). Layer-wise analysis of a
  self-supervised speech representation model. ASRU 2021;
  [arXiv:2107.04734](https://arxiv.org/abs/2107.04734). The layer-wise method
  this project applies to a JEPA encoder.
- Pasad, A., Shi, B., & Livescu, K. (2023). Comparative layer-wise analysis
  of self-supervised speech models. ICASSP 2023;
  [arXiv:2211.03929](https://arxiv.org/abs/2211.03929). The cross-model
  comparison that the encoder baselines extend.
- Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with
  control tasks. EMNLP 2019; [arXiv:1909.03368](https://arxiv.org/abs/1909.03368).
  The source of the control task. Under this design it becomes a leak check
  (pre-registration §5).
- Yuksel, Guetschel, Tangermann, van Gerven, & van der Heijden (2025). WavJEPA:
  Semantic learning unlocks robust audio foundation models for raw waveforms.
  [arXiv:2509.23238](https://arxiv.org/abs/2509.23238). The primary encoder.

**Code**
- [labhamlet/wavjepa](https://github.com/labhamlet/wavjepa): WavJEPA code and
  checkpoints.
- [s3prl/s3prl](https://github.com/s3prl/s3prl): a toolkit for self-supervised
  speech models and the SUPERB benchmark; a reference for loading
  encoders layer by layer.
- [ankitapasad/layerwise-analysis](https://github.com/ankitapasad/layerwise-analysis):
  code for the Pasad et al. layer-wise analyses.
- [zerospeech/libri-light-abx2](https://github.com/zerospeech/libri-light-abx2):
  ABX evaluation, the probe-free measure listed under next steps.

## Cloud

Build and push the image, pull it on the instance (Lambda / Vast / RunPod),
populate `/data` (`scripts/download_common_voice.sh`; Buckeye by registration,
then `scripts/setup_buckeye.sh`), run with `--gpus all` and
`configs/cloud.yaml`. The image is the portability artifact — if it needs a
rebuild to run remotely, that is a bug.

## Backups

Nightly restic from WSL: tier 1 (this repo) + tier 2 (`/data/embeddings`,
`/data/checkpoints`) to D: and Backblaze B2. `/data/corpora` is deliberately
not backed up — re-downloadable. Full design: vault note
`Reference/Desktop/Capstone Dev Environment (Docker + WSL).md`.

## This repository and its history

This is the public successor to a private working repository. Its history is
that repository's, rewritten with `git filter-repo` on 2026-10-16 to remove the
Buckeye download automation and every Buckeye-derived per-site file, and to
redact example site identifiers and word lists. Commit hashes therefore differ
from the private repository's: `docs/provenance/commit-map.txt` maps each
private commit to its public counterpart (an all-zero hash means the commit
touched only removed files). The private repository stays the record that the
pre-registration preceded the pilot; the course instructor is a collaborator
on it.

## License

MIT; see `LICENSE`. The Buckeye Corpus is not covered by it and is not
redistributed here.
