#!/usr/bin/env bash
# Nightly capstone backup — runs inside WSL (Ubuntu-24.04), invoked by the
# Windows scheduled task "Restic Backup - Capstone" at 9:15pm. Design and
# recovery chain: vault note "Capstone Dev Environment (Docker + WSL)".
#
# Tiers — each is its own tagged snapshot stream with its own retention:
#   tier 1  ~/code/capstone     irreplaceable — code, results, stimuli, annotations
#   tier 2  /data/embeddings    regenerable, at GPU-hours cost
#   tier 2  /data/checkpoints   re-downloadable today, not guaranteed tomorrow
#   tier 3  /data/corpora       NOT backed up — re-downloadable by script
#
# Legs: local restic repo on D: (always) + Backblaze B2 (once creds exist).
set -uo pipefail

VAULT_DESKTOP="/mnt/c/Users/danie/Documents/Obsidian/general/Reference/Desktop"
export RESTIC_PASSWORD_FILE="$VAULT_DESKTOP/restic-password-capstone.txt"

# Overwritten each run — this is the "did last night work" file.
LOG_DIR="$HOME/.local/state/capstone-backup"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/last-run.log" 2>&1
echo "=== capstone backup $(date -Is) ==="

# Two snapshot streams, tagged, because the tiers want different retention and
# a single snapshot can only have one policy.
#
# Tier 1 is text: six months of monthlies dedups to almost nothing, and the
# history is the point. Tier 2 is incompressible float data — one hour of
# 13-layer fp16 embeddings is ~7 GB and restic cannot pack it down. Under the
# tier-1 policy a single re-extraction (new geometry, added layer, fp32) would
# pin the superseded copy for six months, because none of the bytes dedup.
# Keeping the last two says what is actually wanted: the current extraction and
# one step back, since anything older is cheaper to regenerate than to store.
TIER1=("$HOME/code/capstone")
TIER2=(/data/embeddings /data/checkpoints)

KEEP_TIER1=(--keep-daily 7 --keep-weekly 4 --keep-monthly 6)
KEEP_TIER2=(--keep-last 2)

# restic does not read .gitignore, so its scratch entries are mirrored here —
# and only those. Deliberately NOT excluded: .git (full history — the only
# copy anywhere until the GitHub push) and stimuli/ (tier 1, irreplaceable).
EXCLUDES=(
  --exclude '**/__pycache__'
  --exclude '**/.ipynb_checkpoints'
  --exclude "$HOME/code/capstone/.venv"
  --exclude "$HOME/code/capstone/outputs"
  --exclude "$HOME/code/capstone/wandb"
)

run_leg() {
  local repo="$1"
  echo "--- leg: $repo"

  restic -r "$repo" backup --tag tier1 "${EXCLUDES[@]}" "${TIER1[@]}" || return 1
  restic -r "$repo" backup --tag tier2 "${EXCLUDES[@]}" "${TIER2[@]}" || return 1

  # --group-by tag, not the default host,paths: it states the intent directly,
  # and it keeps working if a tier ever gains or loses a path (which would
  # otherwise silently start a fresh group with its own full retention).
  #
  # An untagged snapshot matches NEITHER filter and is therefore immortal.
  # The ten pre-2026-08-21 snapshots were retagged tier1 once, by hand, when
  # this split landed. Anything creating untagged snapshots here is a bug.
  restic -r "$repo" forget --tag tier1 --group-by tag "${KEEP_TIER1[@]}" || return 1
  restic -r "$repo" forget --tag tier2 --group-by tag "${KEEP_TIER2[@]}" || return 1

  # One prune for both forgets rather than --prune on each: pruning is the
  # expensive step and there is no reason to run it twice per leg per night.
  restic -r "$repo" prune || return 1

  # Metadata-only check: unlike the vault, this repo is too big for a nightly
  # --read-data (it would re-download the entire B2 repo every night).
  restic -r "$repo" check || return 1
}

status=0
run_leg /mnt/d/Backups/capstone/restic-repo || status=1

# B2 leg activates the moment the creds file exists — no script change needed.
B2_ENV="$VAULT_DESKTOP/restic-b2-capstone.env"
if [[ -f "$B2_ENV" ]]; then
  set -a; source "$B2_ENV"; set +a
  run_leg "$RESTIC_B2_REPO" || status=1
else
  echo "B2 leg skipped: $B2_ENV not found (see vault note for one-time setup)"
fi

echo "=== done status=$status $(date -Is) ==="
exit $status
