#!/usr/bin/env python3
"""How often is a held-out word's lemma in the probe's training set anyway?

Read-only, and no probe, features or audio are involved: a measurement of the
corpus and the registered split, taken before any run.

WHY THIS EXISTS
    The split is disjoint by word type, keyed by capstone.splits.type_key():
    the word form, lowercased, with contractions merged onto their host. It is
    not disjoint by lemma. `need` can train while `needed` is tested, and
    `started` while `starting` is. A lemma-disjoint split would reshuffle every
    fold and was not attempted (decided 2026-09-26). This measures how much of
    "unseen words" is unseen only as a form, so the limitation is registered as
    a number.

METHOD
    Each site's lemma comes from NLTK's WordNet lemmatizer, applied to the site's
    type key and guided by Buckeye's own part-of-speech tag: the first part of
    the tag, so VBP_RB -> VBP. The mapping is V* -> verb, N* -> noun,
    J* -> adjective, R* -> adverb. Any other tag (function words, modals)
    leaves the key as its own lemma. Under `joint` and `word_disjoint`, a test
    site OVERLAPS if its lemma occurs among the lemmas of the sites its probe
    trains on. Shares are pooled over the 10,389 eligible sites.

    Buckeye's tags are noisy (`need` appears as NN), and WordNet's lemmatizer is
    a dictionary lookup, not a morphological analyzer. Read the figures as a
    good estimate, not an exact count.

RUNNING IT
    NLTK is not in either requirements file. Run it in an overlay:
        NLTK_DATA=/data/nltk_data uv run --no-project \\
            --python .venv-app/bin/python --with nltk \\
            python scripts/measure_lemma_overlap.py
    It downloads WordNet into $NLTK_DATA on first use.
"""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from capstone.splits import iter_splits, type_key  # noqa: E402

DEFAULT_INVENTORY = (Path(__file__).resolve().parent.parent
                     / "results/flapping/sites-20260924.parquet")
INVENTORY = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INVENTORY

import nltk  # noqa: E402
from nltk.corpus import wordnet  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

data_dir = os.environ.get("NLTK_DATA", "/data/nltk_data")
try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download("wordnet", download_dir=data_dir, quiet=True)
    wordnet.ensure_loaded()

WN_POS = {"V": "v", "N": "n", "J": "a", "R": "r"}
lemmatizer = WordNetLemmatizer()


def lemma(word: str, tag: str) -> str:
    key = type_key(word)
    pos = WN_POS.get(tag.split("_")[0][:1])
    return lemmatizer.lemmatize(key, pos) if pos else key


inv = pq.read_table(INVENTORY, columns=["site_id", "speaker", "word", "underlying", "pos",
                                        "excerpt_clamped"]).to_pandas()
eligible = ~inv.excerpt_clamped.to_numpy()
keys = np.array([type_key(w) for w in inv.word])
lemmas = np.array([lemma(w, t) for w, t in zip(inv.word, inv.pos)])
y = inv.underlying.to_numpy()

print(f"nltk {nltk.__version__}, WordNet {wordnet.get_version()}")
print(f"eligible sites {int(eligible.sum())}, type keys "
      f"{len(set(keys[eligible]))}, lemmas {len(set(lemmas[eligible]))}")

family = defaultdict(set)
for k, l in zip(keys[eligible], lemmas[eligible]):
    family[l].add(k)
multi = {l: ks for l, ks in family.items() if len(ks) > 1}
in_multi = np.isin(lemmas, list(multi)) & eligible
print(f"lemmas spanning 2+ type keys: {len(multi)}, covering "
      f"{sum(len(k) for k in multi.values())} keys and {int(in_multi.sum())} sites")
sizes = Counter({l: int(((lemmas == l) & eligible).sum()) for l in multi})
print("  largest: " + ", ".join(f"{l} {{{', '.join(sorted(multi[l]))}}} {n}"
                                for l, n in sizes.most_common(12)))

# Two readings. LEMMA SEEN: the lemma occurs in training at all. SAME LABEL:
# it occurs there with the test site's own underlying label. Only the second
# is exploitable by a probe that memorizes lemmas, and it excludes families
# whose members flap on different segments (`good` /d/, `better` /t/).
for condition in ("word_disjoint", "joint"):
    seen_any = np.zeros(len(inv), dtype=bool)
    seen_same = np.zeros(len(inv), dtype=bool)
    for s in iter_splits(condition, inv.speaker, inv.word, inv.underlying,
                         eligible=eligible):
        train = set(lemmas[s.train])
        train_labelled = set(zip(lemmas[s.train], y[s.train]))
        seen_any[s.test] = [l in train for l in lemmas[s.test]]
        seen_same[s.test] = [(l, c) in train_labelled
                             for l, c in zip(lemmas[s.test], y[s.test])]
        # The registered split's own guarantee, re-checked here by key.
        assert not set(keys[s.test]) & set(keys[s.train])
    n = int(eligible.sum())
    print(f"\n{condition}: test sites whose lemma occurs in their probe's training set")
    for name, hit in (("lemma seen", seen_any), ("same label", seen_same)):
        parts = [f"all {int(hit[eligible].sum()):5d} of {n} ({hit[eligible].mean():.1%})"]
        for cls in ("t", "d"):
            m = eligible & (y == cls)
            parts.append(f"/{cls}/ {int(hit[m].sum())} of {int(m.sum())} ({hit[m].mean():.1%})")
        print(f"  {name:10s}  " + "   ".join(parts))
    top = Counter(lemmas[seen_same])
    print("  same label, by lemma: "
          + ", ".join(f"{l} {c}" for l, c in top.most_common(10)))
    if condition == "joint":
        joint_seen_any, joint_seen_same = seen_any, seen_same

# The per-site record under `joint`, committed so the lemma-unseen subset that
# pre-registration §5 re-scores is fixed now rather than re-derived after the
# run, when a different NLTK or WordNet could lemmatize differently.
OUT = (Path(__file__).resolve().parent.parent
       / "results/flapping/lemma-overlap-joint-20260926.tsv")
with open(OUT, "w") as fh:
    fh.write("site_id\ttype_key\tlemma\tlemma_in_train\tlemma_in_train_same_label\n")
    for i in np.flatnonzero(eligible):
        fh.write(f"{inv.site_id.iloc[i]}\t{keys[i]}\t{lemmas[i]}\t"
                 f"{str(bool(joint_seen_any[i])).lower()}\t"
                 f"{str(bool(joint_seen_same[i])).lower()}\n")
unseen = eligible & ~joint_seen_any
print(f"\nlemma-unseen subset under joint (lemma nowhere in the probe's training): "
      f"{int(unseen.sum())} sites, /d/ {int((unseen & (y == 'd')).sum())}")
print(f"wrote {OUT.relative_to(OUT.parents[2])}")
