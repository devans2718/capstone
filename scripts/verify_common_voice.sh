#!/usr/bin/env bash
#
# verify_common_voice.sh — measure what the Common Voice tarball actually contains
#
# WHY THIS EXISTS
#   The corpus figures recorded in the vault (1,138,688 clips / 1,643 validated
#   hours / 67.4 GB) came from Mozilla's per-release JSON. The 67.4 GB figure was
#   disproven on 2026-08-05 — the real archive is 7,060,217,902 bytes, ~10x
#   smaller — which removes the corroboration the hours figure was resting on.
#
#   That matters beyond bookkeeping. 7.06 GB across 1,643 hours implies ~9.5 kbps,
#   and MP3 encoders at that rate lowpass hard (often to 4-5 kHz), which would
#   degrade burst spectra and frication detail. Closure voicing (<300 Hz) and
#   vowel duration survive regardless, so RQ1's core cue is safe either way — but
#   "the codec is fine here" is a claim about a bitrate, and this script measures
#   it instead of assuming it.
#
#   Output is written to a dated report so the numbers land in the record rather
#   than scrolling past. Correct the vault TODO from that report, not from memory.
#
# USAGE
#   ./scripts/verify_common_voice.sh [corpus_root]
#
# DEPENDENCIES
#   ffprobe (apt install ffmpeg) for the audio measurements. Everything else
#   degrades gracefully if it is missing — the clip and TSV counts still run.

set -euo pipefail

ROOT="${1:-/data/corpora/common-voice}"
SAMPLE_N="${SAMPLE_N:-50}"
REPORT_DIR="${REPORT_DIR:-$HOME/code/capstone/results/corpus-verification}"
REPORT="${REPORT_DIR}/common-voice-ru-$(date -u +%Y%m%d).txt"

[[ -d "$ROOT" ]] || { echo "ERROR: corpus root not found: $ROOT" >&2; exit 1; }
mkdir -p "$REPORT_DIR"

# Everything below is written to stdout AND the report file.
exec > >(tee "$REPORT") 2>&1

echo "Common Voice verification"
echo "  root: $ROOT"
echo "  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# ------------------------------------------------------------ archive on disk

echo "=== archive ==="
find "$ROOT" -maxdepth 2 -name '*.tar.gz' -printf '%p\n  %s bytes\n' 2>/dev/null || true
find "$ROOT" -maxdepth 2 -name '*.sha256' -exec cat {} \; 2>/dev/null || true
echo

# ------------------------------------------------------------------ file tree

echo "=== layout (top 3 levels) ==="
find "$ROOT" -maxdepth 3 -type d -printf '%p\n' | head -40
echo

# --------------------------------------------------------------- clip counting

echo "=== clips ==="
MP3_COUNT=$(find "$ROOT" -name '*.mp3' | wc -l)
echo "  mp3 files on disk : $MP3_COUNT"
echo "  vault claim       : 1138688"
if [[ "$MP3_COUNT" -gt 0 ]]; then
    echo "  ratio to claim    : $(awk -v a="$MP3_COUNT" 'BEGIN { printf "%.3f", a/1138688 }')"
fi
echo

# ---------------------------------------------------------------------- TSVs

echo "=== TSV manifests ==="
# Row counts minus the header line. These are the corpus's own account of itself
# and should be trusted over the file tree if the two disagree.
while IFS= read -r tsv; do
    rows=$(( $(wc -l < "$tsv") - 1 ))
    printf '  %-28s %10d rows\n' "$(basename "$tsv")" "$rows"
done < <(find "$ROOT" -name '*.tsv' | sort)
echo

# clip_durations.tsv ships with newer releases and is the only exact duration
# source that does not require decoding every file.
DUR_TSV=$(find "$ROOT" -name 'clip_durations.tsv' | head -1)
if [[ -n "$DUR_TSV" ]]; then
    echo "=== total duration (from clip_durations.tsv, exact) ==="
    awk -F'\t' 'NR > 1 { ms += $2 } END {
        printf "  clips: %d\n  total: %.1f hours\n", NR-1, ms/1000/3600
    }' "$DUR_TSV"
    echo "  vault claim: 1643 validated hours / 1845 total"
    echo
fi

# ------------------------------------------------------------ audio properties

echo "=== audio properties (sample of $SAMPLE_N) ==="
if ! command -v ffprobe >/dev/null 2>&1; then
    echo "  ffprobe not installed — skipping. Install with: sudo apt install ffmpeg"
    echo "  THIS IS THE MEASUREMENT THAT MATTERS. Re-run after installing."
else
    # Sample across the tree rather than taking the head, which would be biased
    # toward whichever speaker sorts first.
    SAMPLE=$(find "$ROOT" -name '*.mp3' | shuf -n "$SAMPLE_N" 2>/dev/null \
             || find "$ROOT" -name '*.mp3' | head -n "$SAMPLE_N")

    TMP=$(mktemp)
    while IFS= read -r f; do
        [[ -n "$f" ]] || continue
        ffprobe -v error \
            -show_entries stream=sample_rate,channels,bit_rate \
            -show_entries format=duration,bit_rate \
            -of default=nw=1:nk=0 "$f" 2>/dev/null \
        | awk -F= -v OFS='\t' '
            $1 == "sample_rate" { sr = $2 }
            $1 == "channels"    { ch = $2 }
            $1 == "bit_rate" && br == "" { br = $2 }
            $1 == "duration"    { d  = $2 }
            END { print sr, ch, br, d }' >> "$TMP"
    done <<< "$SAMPLE"

    echo "  sample rates:"
    awk -F'\t' '{ print "    " $1 " Hz" }' "$TMP" | sort | uniq -c | sort -rn
    echo "  channels:"
    awk -F'\t' '{ print "    " $2 }' "$TMP" | sort | uniq -c | sort -rn
    echo "  bitrate (bits/sec):"
    awk -F'\t' '$3 != "" { print "    " int($3/1000) " kbps" }' "$TMP" | sort -n | uniq -c | sort -rn
    echo
    awk -F'\t' '
        { n++; d += $4; b += $3 }
        END {
            if (n) {
                printf "  mean clip duration : %.3f s   (vault claim: 5.203 s)\n", d/n
                printf "  mean bitrate       : %.1f kbps\n", b/n/1000
            }
        }' "$TMP"
    rm -f "$TMP"

    echo
    echo "  INTERPRETATION:"
    echo "    >= 32 kbps  -> codec is not a concern for RQ1; earlier assessment holds."
    echo "    ~16-24 kbps -> voicing and duration cues intact; note the lowpass as a"
    echo "                   limitation for any spectral measure above ~7 kHz."
    echo "    <= 12 kbps  -> check the effective lowpass before trusting burst or"
    echo "                   frication features. Closure voicing and vowel duration"
    echo "                   still survive, so RQ1 is workable, but say so explicitly."
fi

echo
echo "=== speakers ==="
# client_id is the speaker key and the basis for speaker-disjoint splits.
VAL_TSV=$(find "$ROOT" -name 'validated.tsv' | head -1)
if [[ -n "$VAL_TSV" ]]; then
    echo "  distinct client_id in validated.tsv: $(awk -F'\t' 'NR>1 { print $1 }' "$VAL_TSV" | sort -u | wc -l)"
    echo "  vault claim: 36393 speakers"
else
    echo "  validated.tsv not found; skipping"
fi

echo
echo "report written to: $REPORT"
