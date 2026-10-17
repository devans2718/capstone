#!/usr/bin/env bash
# Extract a registered copy of the Buckeye Speech Corpus into /data/corpora/buckeye.
#
# This script does not download anything. Ohio State distributes the corpus
# through a free registration (name and a short research purpose) at
# https://buckeyecorpus.osu.edu, and that gate is theirs to keep. Register,
# download the 40 speaker archives (sNN.zip, NN = 01..40) and the POS-tagged
# words archive the site offers, and place them in $DEST/zips/:
#
#   /data/corpora/buckeye/zips/s01.zip ... s40.zip
#   /data/corpora/buckeye/zips/tagged_words.zip   (any name ending tagged_words*.zip)
#
# Then run this script. It produces the layout every mining script expects:
#   /data/corpora/buckeye/sNN/          (extracted, nested per-track zips flattened)
#   /data/corpora/buckeye/tagged_words/ (POS-tagged .words.tagged files, nesting kept)
#
#   Version:  Buckeye Speech Corpus 2.0 — 40 talkers, ~3 GB,
#             spontaneous American English interview speech
#   Cite:     Pitt, M.A., Dilley, L., Johnson, K., Kiesling, S., Raymond, W.,
#             Hume, E. and Fosler-Lussier, E. (2007) Buckeye Corpus of
#             Conversational Speech (2nd release) [www.buckeyecorpus.osu.edu]
#             Columbus, OH: Department of Psychology, Ohio State University
#             (Distributor).
#
# Resumable: re-running skips speakers already extracted.
set -uo pipefail

DEST=${BUCKEYE_DEST:-/data/corpora/buckeye}
Z="$DEST/zips"

failed=()
missing=()
for i in $(seq -w 1 40); do
  [[ -f "$Z/s$i.zip" ]] || missing+=("s$i.zip")
done
tz=$(find "$Z" -maxdepth 1 -name 'tagged_words*.zip' 2>/dev/null | head -1)
[[ -n "$tz" ]] || missing+=("tagged_words*.zip")
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing from $Z: ${missing[*]}"
  echo "Register at https://buckeyecorpus.osu.edu, download them, and place them there."
  exit 1
fi

# The POS tags feed RQ1's flap-site mining, so a corpus without them is
# silently incomplete. Extracted WITHOUT -j: the archive nests the .tagged
# files one directory down, and the mining code is written against that layout.
if [[ ! -d "$DEST/tagged_words" ]]; then
  mkdir -p "$DEST/tagged_words.tmp"
  unzip -qo "$tz" -d "$DEST/tagged_words.tmp" \
    && mv "$DEST/tagged_words.tmp" "$DEST/tagged_words" \
    || { failed+=("tagged_words-extract"); rm -rf "$DEST/tagged_words.tmp"; }
else
  echo "extracted already: tagged_words"
fi

for i in $(seq -w 1 40); do
  z="$Z/s$i.zip"
  unzip -tqq "$z" > /dev/null || { echo "corrupt: s$i.zip"; failed+=("s$i"); continue; }
  d="$DEST/s$i"
  [[ -d "$d" ]] && { echo "extracted already: s$i"; continue; }
  mkdir -p "$d.tmp"
  unzip -qo "$z" -d "$d.tmp" || { failed+=("s$i-extract"); continue; }
  find "$d.tmp" -name '*.zip' | while read -r nz; do
    unzip -qo "$nz" -d "$d.tmp" && rm -f "$nz"
  done
  mv "$d.tmp" "$d"
done

echo "=== summary ==="
echo "speaker dirs: $(find "$DEST" -maxdepth 1 -type d -name 's[0-9][0-9]' | wc -l) / 40"
echo "tagged files: $(find "$DEST/tagged_words" -name '*.tagged' 2>/dev/null | wc -l)"
echo "total size: $(du -sh "$DEST" | cut -f1)"
[[ ${#failed[@]} -gt 0 ]] && echo "FAILED: ${failed[*]}" || echo "no failures"
