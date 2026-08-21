#!/usr/bin/env bash
#
# download_common_voice.sh — fetch and extract Common Voice Scripted Speech 26.0 (Russian)
#
# WHY THIS SCRIPT EXISTS
#   /data/corpora is tier-3: not backed up. This script IS the backup. If the
#   corpus is lost, re-running this must reproduce it byte-for-byte — which is
#   why the dataset ID is pinned below rather than the human-readable "26.0",
#   and why the sha256 of the archive is recorded on first success and verified
#   on every re-run.
#
# PROVENANCE
#   Source:     Mozilla Data Collective  <https://mozilladatacollective.com>
#   Dataset:    Common Voice Scripted Speech 26.0 - Russian
#   Dataset ID: cmqinj9g500vsnr07qf4hmr3j     <-- the real version pin
#   Release:    cv-corpus-26.0-2026-06-12
#   Locale:     ru
#   License:    CC0-1.0
#   Format:     MP3 (32/48 kHz). Decode+resample to 16 kHz WAV in preprocessing;
#               do NOT align against one decode path and extract from another —
#               MP3 encoder delay (~576 samples + padding) is decoder-dependent
#               and a mismatch silently shifts every alignment timestamp.
#
#   Cite Common Voice per Mozilla's request:
#     Ardila et al. (2020), "Common Voice: A Massively-Multilingual Speech Corpus", LREC.
#
# CREDENTIALS
#   Requires an API key, supplied via the environment — never inline, never
#   committed. This repo is private, but the successor repo is the publishable
#   artifact and publishing a repo publishes its history.
#     export MDC_API_KEY=...            (or put it in a gitignored .env and source it)
#
# USAGE
#   MDC_API_KEY=... ./scripts/download_common_voice.sh
#   SKIP_SPACE_CHECK=1 ./scripts/download_common_voice.sh    # override the preflight
#
# IDEMPOTENCY
#   Safe to re-run. A verified archive is not re-downloaded; a verified
#   extraction is not re-extracted. A partial download resumes via a .part file
#   (never via -C - against a complete archive — that requests a range at EOF,
#   gets 416, and -f turns it into a spurious failure).

set -euo pipefail

# ---------------------------------------------------------------- configuration

DATASET_ID="cmqinj9g500vsnr07qf4hmr3j"
API_BASE="https://mozilladatacollective.com/api/datasets"

# Constants, not parameters. Making the locale a real parameter is the separate
# task that the English pull falls out of; these two exist so the paths below
# stop hardcoding the release layout in string literals.
RELEASE="cv-corpus-26.0-2026-06-12"
LOCALE="ru"

DEST="${DEST:-/data/corpora/common-voice}"
ARCHIVE_NAME="common-voice-scripted-speech-26-0-russian.tar.gz"

ARCHIVE="${DEST}/${ARCHIVE_NAME}"
PARTIAL="${ARCHIVE}.part"
SHA_FILE="${ARCHIVE}.sha256"
# The tarball carries its own `cv-corpus-26.0-2026-06-12/ru` top level, so tar
# unpacks into DEST directly. Naming a second per-locale directory here buried
# the corpus one level deeper than the release layout, which is harmless until
# something is rebuilt from scratch and every path written against the release
# layout misses by one component.
EXTRACT_DIR="${DEST}"
CORPUS_DIR="${EXTRACT_DIR}/${RELEASE}/${LOCALE}"

# Extraction needs room for the archive plus the unpacked tree. MP3 is already
# compressed, so gzip buys little — assume the extracted size ~= archive size.
SPACE_MULTIPLIER=21   # tenths, i.e. 2.1x the archive size

# ---------------------------------------------------------------------- helpers

log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die()  { printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

# Ask the API for a fresh presigned download URL.
# Presigned URLs expire, so this is called once per run and never cached to disk.
request_download_url() {
    local response url
    response=$(curl -fsS -X POST "${API_BASE}/${DATASET_ID}/download" \
        -H "Authorization: Bearer ${MDC_API_KEY}" \
        -H "Content-Type: application/json") \
        || die "download-URL request failed (bad or expired API key?)"

    # Quoted here-string, not 'echo $response' — the payload is JSON and must
    # not be word-split.
    url=$(jq -r '.downloadUrl // empty' <<< "$response") \
        || die "could not parse API response as JSON"

    [[ -n "$url" ]] || die "API response contained no downloadUrl: ${response:0:200}"
    printf '%s' "$url"
}

# Report the archive's size in bytes, or empty string if the server won't say.
# Tries HEAD first; falls back to a 1-byte range GET, because S3 presigned URLs
# are signed for a specific method and often reject HEAD with 403.
remote_size() {
    local url="$1" size

    size=$(curl -fsSIL "$url" 2>/dev/null \
           | tr -d '\r' \
           | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }') || true
    if [[ -n "${size:-}" ]]; then printf '%s' "$size"; return; fi

    size=$(curl -fsS -r 0-0 -o /dev/null -D - "$url" 2>/dev/null \
           | tr -d '\r' \
           | awk -F/ 'tolower($0) ~ /^content-range:/ { print $2 }') || true
    printf '%s' "${size:-}"
}

check_space() {
    local need_bytes="$1" avail
    [[ -n "$need_bytes" ]] || { log "WARN: server did not report a size; skipping space preflight"; return 0; }

    avail=$(df -PB1 "$DEST" | awk 'NR==2 { print $4 }')
    log "space: need ~$(( need_bytes / 1000000000 )) GB, available $(( avail / 1000000000 )) GB"

    if (( avail < need_bytes )); then
        die "insufficient space on $(df -P "$DEST" | awk 'NR==2 {print $6}'): \
need ~$(( need_bytes / 1000000000 )) GB, have $(( avail / 1000000000 )) GB. \
Set SKIP_SPACE_CHECK=1 to override."
    fi
}

verify_archive() {
    [[ -f "$ARCHIVE" && -f "$SHA_FILE" ]] || return 1
    log "verifying existing archive against recorded sha256..."
    ( cd "$DEST" && sha256sum -c --status "$(basename "$SHA_FILE")" )
}

# ------------------------------------------------------------------- preflight

require_cmd curl
require_cmd jq
require_cmd tar
require_cmd sha256sum
require_cmd awk

[[ -n "${MDC_API_KEY:-}" ]] || die "MDC_API_KEY is not set. Export it (or source a gitignored .env) — do not hardcode it."

mkdir -p "$DEST"

# ------------------------------------------------------------------- download

if verify_archive; then
    log "archive present and sha256 verified — skipping download"
else
    if [[ -f "$ARCHIVE" && ! -f "$SHA_FILE" ]]; then
        # An archive with no recorded checksum predates this script's bookkeeping.
        # Keep it, hash it below, and adopt it rather than re-downloading 7-70 GB.
        log "archive present but unverified (no .sha256 on record) — adopting and hashing it"
    else
        log "requesting presigned download URL for dataset ${DATASET_ID}..."
        DOWNLOAD_URL=$(request_download_url)
        log "presigned URL acquired (expires; not cached)"

        if [[ "${SKIP_SPACE_CHECK:-0}" != "1" ]]; then
            SIZE=$(remote_size "$DOWNLOAD_URL")
            if [[ -n "$SIZE" ]]; then
                log "server reports archive size: $SIZE bytes ($(( SIZE / 1000000000 )) GB)"
                check_space "$(( SIZE * SPACE_MULTIPLIER / 10 ))"
            else
                log "WARN: server did not report Content-Length; skipping space preflight"
            fi
        fi

        # Resume against .part only. The completed archive is never a -C - target,
        # which is the 416-on-complete-file trap that made download_buckeye.sh
        # report 40/40 speakers FAILED on a healthy corpus.
        if [[ -f "$PARTIAL" ]]; then
            log "resuming partial download ($(stat -c%s "$PARTIAL") bytes already present)"
            curl -fL --retry 5 --retry-delay 5 --retry-connrefused \
                 -C - -o "$PARTIAL" "$DOWNLOAD_URL"
        else
            log "starting download..."
            curl -fL --retry 5 --retry-delay 5 --retry-connrefused \
                 -o "$PARTIAL" "$DOWNLOAD_URL"
        fi

        mv "$PARTIAL" "$ARCHIVE"
        log "download complete: $(stat -c%s "$ARCHIVE") bytes"
    fi

    # Record the checksum on first success; this is what future runs verify against.
    log "computing sha256 (this takes a minute on a large archive)..."
    ( cd "$DEST" && sha256sum "$ARCHIVE_NAME" > "$(basename "$SHA_FILE")" )
    log "recorded: $(cat "$SHA_FILE")"
fi

# -------------------------------------------------------------------- extract

# Marker ties the extraction to the exact archive it came from, so a corpus
# re-release with a new checksum forces a clean re-extract instead of silently
# mixing two versions in one tree.
ARCHIVE_SHA=$(awk '{print $1}' "$SHA_FILE")
MARKER="${EXTRACT_DIR}/.extracted-${ARCHIVE_SHA}"

if [[ -f "$MARKER" ]]; then
    log "already extracted for this archive checksum — nothing to do"
else
    log "extracting to ${EXTRACT_DIR}..."
    mkdir -p "$EXTRACT_DIR"
    tar -xzf "$ARCHIVE" -C "$EXTRACT_DIR"
    touch "$MARKER"
    log "extraction complete"
fi

# ------------------------------------------------------------------- summary

log "summary:"
log "  archive : ${ARCHIVE}"
log "  sha256  : ${ARCHIVE_SHA}"
log "  extracted to: ${EXTRACT_DIR}"
log "  corpus root : ${CORPUS_DIR}"
log "  clips (mp3 count): $(find "$CORPUS_DIR" -name '*.mp3' | wc -l)"
log "  tsv manifests: $(find "$CORPUS_DIR" -name '*.tsv' -printf '%f\n' | sort | tr '\n' ' ')"
log ""
log "NEXT: establish speaker-disjoint splits from the client_id column in the TSVs"
log "      BEFORE caching any embeddings. Splitting after extraction is how"
log "      speaker leakage gets baked into results."
