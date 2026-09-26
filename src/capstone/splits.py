"""The four registered split conditions, built in one place and shared.

WHY THIS IS A MODULE
    Until revision 3 the §3 construction lived only inside
    scripts/analyze_flap_lexical_structure.py, and the slice drew its fold
    numbers at random. A split described in one file and implemented in another
    is two splits the moment either one changes. The probe run, the slice and
    the analysis script now all take their folds from here, and
    tests/test_splits.py holds the figures the pre-registration quotes.

THE TWO AXES, AND WHY EVERY CONDITION BUT ONE REUSES THEM
    Speaker folds: speakers sorted, speaker i -> fold i mod 5.
    Type folds: word types ranked by site count, highest first, ties broken by
    spelling in code-point order; the i-th ranked type -> fold i mod 5.

WHAT A "WORD TYPE" IS (decided 2026-09-26)
    The type key, computed by type_key(), is the inventory's `word` column (the
    orthographic word form), lowercased, with contracted forms merged onto the
    host word that carries the flap: `don't` -> `do`, `city's` -> `city`.
    Without the merge, `do` in training and `don't` in test would be the same
    lexical item on both sides of a word-disjoint split. The merge is an
    explicit table, not "cut at the apostrophe": that rule turns `don't` into
    `don`, and it would also merge `i'd` onto `i`, whose flap is in the `'d`,
    not the host. Forms whose flap lies in the clitic keep their own key.

    The key is by word form, not lemma: `need` and `needed` are different
    types. A lemma-disjoint split is not attempted. How often a test site's
    lemma occurs in its probe's training set is measured instead, by
    scripts/measure_lemma_overlap.py.

    speaker_disjoint holds out one speaker fold, word_disjoint one type fold,
    and joint one cell of their 5 x 5 grid. Joint's train set for cell (a, b)
    is then exactly the intersection of speaker_disjoint's train set for a and
    word_disjoint's for b, so a difference between conditions is attributable
    to the constraint rather than to differently drawn folds.

    random is the one condition with a seed: shuffled 5-fold, stratified by
    underlying label. Tokens of one word landing in both train and test is what
    it exists to show, not a defect.

FOLDS ARE ASSIGNED OVER THE ELIGIBLE SITES ONLY (decided 2026-09-25)
    Sites excluded before scoring (pre-registration §1; today the 3 clamped
    excerpts) are removed BEFORE any fold is assigned: they are not ranked, not
    counted, and belong to no fold. A site that is never scored should not
    decide where other sites go, and it would: counting the clamped sites keeps
    `nineties` in the ranking and moves the 191 singleton types behind it. The
    grid is therefore built over 10,389 sites, not 10,392.

    iter_splits takes the whole inventory plus an `eligible` mask and returns
    indices into the whole inventory, so a caller holding one array per site
    indexes it directly. `eligible` has no default: a caller that forgets it
    would get the rejected assignment over every site, silently. site_folds
    assigns over exactly what it is handed, so its callers filter first.

THE AXES ARE PURE PYTHON
    speaker_folds, rank_types and type_folds use only the standard library, so
    the analysis script keeps running under a bare python3. numpy and
    scikit-learn are imported only where index arrays or the random condition
    need them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

CONDITIONS = ("random", "speaker_disjoint", "word_disjoint", "joint")

N_FOLDS = 5

# The secondary run's training fraction for the single-axis conditions. They
# train on ~80% of sites and joint on ~64%, so part of any drop from
# word_disjoint to joint is only less data. 80% of 80% puts them level.
SUBSAMPLE_FRACTION = 0.8


# ------------------------------------------------------------------- the axes


def speaker_folds(speakers: Iterable[str]) -> dict[str, int]:
    """Speaker -> speaker fold: sorted, then i mod 5."""
    return {s: i % N_FOLDS for i, s in enumerate(sorted(set(speakers)))}


# Contracted forms in the inventory whose flap lies in the host word, mapped to
# that host. Every contracted form in the inventory is listed, either here or in
# OWN_KEY, so a new one cannot slip through unclassified (tests/test_splits.py
# checks). Where the flap is: 's, 'll and 've contain no /t/ or /d/, so the flap
# is in the host. For it'd and that'd the label is /t/, the host's. For the n't
# forms, the phone index puts the flap in the host (don't: [dx ow n tq]).
CONTRACTIONS = {
    # 's
    "adriatico's": "adriatico", "anybody's": "anybody", "body's": "body",
    "building's": "building", "catalog's": "catalog", "city's": "city",
    "computer's": "computer", "daddy's": "daddy", "daughter's": "daughter",
    "doty's": "doty", "everybody's": "everybody", "it's": "it",
    "media's": "media", "nobody's": "nobody", "reader's": "reader",
    "somebody's": "somebody", "that's": "that", "title's": "title",
    "today's": "today",
    # 'll
    "it'll": "it", "that'll": "that", "what'll": "what",
    # 've
    "could've": "could", "might've": "might", "should've": "should",
    "that've": "that", "would've": "would",
    # 'd after a /t/-final host: the flap is the host's /t/
    "it'd": "it", "that'd": "that",
    # n't: the flap is the host's /d/
    "couldn't": "could", "didn't": "did", "doesn't": "does", "don't": "do",
    "wouldn't": "would",
}

# Contracted forms whose flap is in the clitic, which keep their own key. The
# 'd after a vowel-final pronoun is a reduced "would" or "had": a link at the
# level of the lemma, which this key does not claim to cover.
OWN_KEY = frozenset({"he'd", "i'd", "they'd", "there'd", "you'd", "haven't"})


def type_key(word: str) -> str:
    """The word type a site belongs to: lowercased, contractions onto their host."""
    w = word.lower()
    return CONTRACTIONS.get(w, w)


def rank_types(words: Iterable[str]) -> list[tuple[str, int]]:
    """(word, site count), highest count first, ties by code-point order.

    Ranks whatever strings it is given; type_folds keys them first. Not
    Counter.most_common(): that orders ties by first appearance, and 985 of the
    1,022 word forms share a count with another, so the folds would ride on the
    inventory's row order.
    """
    return sorted(Counter(words).items(), key=lambda kv: (-kv[1], kv[0]))


def type_folds(words: Iterable[str], *, key=type_key) -> dict[str, int]:
    """Type key -> type fold: the i-th ranked key goes to fold i mod 5.

    Look a site up with typ[type_key(word)]. `key=str` ranks raw word forms, as
    revision 2 did; it exists only so that history can still be measured.
    """
    return {w: i % N_FOLDS for i, (w, _) in enumerate(rank_types(map(key, words)))}


# ------------------------------------------------------------- per-site folds


def _check(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")


def site_folds(condition: str, speakers, words, labels=None, *, seed: int | None = None):
    """Each site's test fold under `condition`, as an int array.

    0-4 for the single-axis conditions and random; 0-24 for joint, where cell
    (a, b) is 5a + b. This is the `fold` column of predictions.parquet.

    Assigns over every site passed in. Pass the eligible sites only; see the
    module docstring for why that is the registered set.

    random needs `labels` and `seed`, and refuses to run without them rather
    than falling back to a default: a seed nobody chose is a seed nobody
    recorded.
    """
    import numpy as np

    _check(condition)
    speakers, words = list(speakers), list(words)

    if condition == "random":
        if labels is None or seed is None:
            raise ValueError("the random condition needs labels and an explicit seed")
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(list(labels))
        fold = np.empty(len(y), dtype=np.int64)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (_, test) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold[test] = k
        return fold

    spk, typ = speaker_folds(speakers), type_folds(words)
    a = np.array([spk[s] for s in speakers], dtype=np.int64)
    b = np.array([typ[type_key(w)] for w in words], dtype=np.int64)
    return {"speaker_disjoint": a, "word_disjoint": b, "joint": N_FOLDS * a + b}[condition]


# ------------------------------------------------------------ train/test pairs


@dataclass(frozen=True)
class Split:
    """One probe's worth of indices into the arrays that were passed in."""

    condition: str
    fold: int
    train: "object"         # np.ndarray of int, sorted
    test: "object"          # np.ndarray of int, sorted
    subsampled: bool


def iter_splits(condition: str, speakers, words, labels, *, eligible,
                seed: int | None = None, subsample: bool = False) -> Iterator[Split]:
    """(train, test) for every probe of a condition: 5, or 25 for joint.

    Single-axis conditions and random train on every other fold. Joint trains
    on sites sharing neither the test cell's speaker fold nor its type fold;
    sites sharing exactly one are left out of that probe's training and scored
    by their own cell.

    `subsample` is the registered secondary run: each training set cut to
    SUBSAMPLE_FRACTION of itself, stratified by label, drawn from a generator
    keyed on (seed, fold) so no fold's draw depends on the order the folds are
    iterated in. Joint is the level the others are brought down to, so asking
    to subsample it is refused rather than quietly obeyed.

    `eligible`, a boolean per site, is required: ineligible sites are removed
    before folds are assigned and appear in no train or test set. Pass None
    only when every site passed in is already eligible. Indices always refer
    to the arrays as passed, ineligible sites included.
    """
    import numpy as np

    _check(condition)
    speakers, words, labels = list(speakers), list(words), list(labels)
    y = np.asarray(labels)
    n = len(y)
    if not (len(speakers) == len(words) == n):
        raise ValueError("speakers, words and labels must be the same length")
    if subsample and condition == "joint":
        raise ValueError("the subsample applies to the single-axis conditions, not joint")
    if subsample and seed is None:
        raise ValueError("subsample needs an explicit seed")

    keep = np.ones(n, dtype=bool) if eligible is None else np.asarray(list(eligible), dtype=bool)
    if len(keep) != n:
        raise ValueError(f"eligible has {len(keep)} entries for {n} sites")
    idx = np.flatnonzero(keep)                    # position in the subset -> in the input
    spk = [speakers[i] for i in idx]
    wrd = [words[i] for i in idx]

    if condition == "joint":
        a = site_folds("speaker_disjoint", spk, wrd)
        b = site_folds("word_disjoint", spk, wrd)
        for i in range(N_FOLDS):
            for j in range(N_FOLDS):
                yield Split(condition, N_FOLDS * i + j,
                            train=idx[(a != i) & (b != j)],
                            test=idx[(a == i) & (b == j)],
                            subsampled=False)
        return

    fold = site_folds(condition, spk, wrd, y[idx], seed=seed)
    for k in range(N_FOLDS):
        train = idx[fold != k]
        if subsample:
            train = _subsample(train, y, seed, k)
        yield Split(condition, k, train=train, test=idx[fold == k], subsampled=subsample)


def _subsample(idx, y, seed: int, fold: int):
    """SUBSAMPLE_FRACTION of idx, per class, without replacement, sorted."""
    import numpy as np

    rng = np.random.default_rng([seed, fold])
    parts = []
    for cls in np.unique(y[idx]):
        members = idx[y[idx] == cls]
        parts.append(rng.choice(members, size=round(SUBSAMPLE_FRACTION * len(members)),
                                replace=False))
    return np.sort(np.concatenate(parts))
