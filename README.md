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
