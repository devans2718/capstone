#!/usr/bin/env bash
# Nightly capstone backup — runs inside WSL (Ubuntu-24.04), invoked by the
# Windows scheduled task "Restic Backup - Capstone" at 9:15pm. Design and
# recovery chain: vault note "Capstone Dev Environment (Docker + WSL)".
#
# Tiers:
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

SOURCES=("$HOME/code/capstone" /data/embeddings /data/checkpoints)
KEEP=(--keep-daily 7 --keep-weekly 4 --keep-monthly 6)

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
  echo "--- leg: $1"
  restic -r "$1" backup "${EXCLUDES[@]}" "${SOURCES[@]}" || return 1
  restic -r "$1" forget --prune "${KEEP[@]}" || return 1
  # Metadata-only check: unlike the vault, this repo is too big for a nightly
  # --read-data (it would re-download the entire B2 repo every night).
  restic -r "$1" check || return 1
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
