#!/usr/bin/env python3
"""Is a word-type-disjoint split feasible on the Buckeye flap inventory?

Read-only. Answers five questions: how many word types carry the 10,392 sites,
how concentrated they are, whether `underlying` is a deterministic function of
the word type, what survives a split that is type-disjoint and
speaker-disjoint at the same time, and whether the full 5 x 5 grid behind that
split can score every site instead of only the diagonal.

Word types are ranked by site count, ties broken alphabetically, so every fold
assignment here is a function of the inventory's contents and not its row order.
The ranking and the §5 folds come from capstone.splits, the same code the probe
run and the slice use, so this report describes the split that is actually run.

§1-§4 describe the whole inventory, 10,392 sites. §5 builds the registered grid
over the 10,389 eligible ones: the 3 clamped sites are removed before any fold
is assigned (decided 2026-09-25). That needs the excerpt_clamped column, which
only the parquet inventory carries, so this reads the parquet and runs in
.venv-app. Until 2026-09-25 it read the TSV, which lists the same sites in the
same order (tests/test_splits.py checks that).
"""
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capstone.splits import rank_types, speaker_folds, type_folds  # noqa: E402

DEFAULT_INVENTORY = (Path(__file__).resolve().parent.parent
                     / "results/flapping/sites-20260924.parquet")
INVENTORY = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INVENTORY

import pyarrow.parquet as pq  # noqa: E402

rows = pq.read_table(INVENTORY, columns=["speaker", "word", "underlying",
                                         "excerpt_clamped"]).to_pylist()

print(f"sites: {len(rows)}")
print(f"class counts: {Counter(r['underlying'] for r in rows)}")
print(f"speakers: {len({r['speaker'] for r in rows})}")

# --- 1. types and concentration -----------------------------------------
by_type = Counter(r["word"] for r in rows)
print(f"\ndistinct word types: {len(by_type)}")

# Count descending, ties broken by spelling in code-point order (Python's own
# string order, so 'Daddy' sorts before every lowercase word). Not
# most_common(): that orders ties by first appearance in the TSV, which would
# make the §3 and §5 fold assignment depend on row order. See §5.
ranked = rank_types(r["word"] for r in rows)
total = len(rows)
for n in (10, 50, 100, 200):
    share = sum(c for _, c in ranked[:n]) / total
    print(f"  top {n:>3} types cover {share:6.1%} of sites")
singletons = sum(1 for _, c in ranked if c == 1)
print(f"  types with exactly 1 site: {singletons} "
      f"({singletons / len(by_type):.1%} of types, {singletons / total:.1%} of sites)")
print(f"  top 10: {ranked[:10]}")

# --- 2. is the label determined by the word type? -----------------------
labels_of_type = defaultdict(set)
for r in rows:
    labels_of_type[r["word"]].add(r["underlying"])
mixed = {w: labs for w, labs in labels_of_type.items() if len(labs) > 1}
mixed_sites = sum(by_type[w] for w in mixed)
print(f"\ntypes carrying both classes: {len(mixed)} "
      f"({mixed_sites} sites, {mixed_sites / total:.2%})")
if mixed:
    print(f"  examples: {sorted(mixed)[:15]}")

# A probe that memorizes word -> label perfectly scores this on the real task:
type_major = sum(max(Counter(r["underlying"] for r in rows if r["word"] == w).values())
                 for w in mixed)
lexical_ceiling = (total - mixed_sites + type_major) / total
print(f"  accuracy of a pure word-identity lookup on the real task: {lexical_ceiling:.4f}")

# types per class
t_types = {w for w, labs in labels_of_type.items() if labs == {"t"}}
d_types = {w for w, labs in labels_of_type.items() if labs == {"d"}}
print(f"  /t/-only types: {len(t_types)}   /d/-only types: {len(d_types)}")

# --- 3. joint speaker- and type-disjoint split --------------------------
# Test fold = held-out speakers X held-out types. Train = the complement on
# BOTH axes. Sites matching one axis but not the other must be discarded, and
# that discard is the real cost of the design.
speakers = sorted({r["speaker"] for r in rows})
types = sorted(by_type)

print("\njoint speaker- + type-disjoint 5-fold (held-out = 20% speakers, 20% types)")
rng = random.Random(0)

for label, type_pick in (("types drawn at random", "random"),
                         ("types drawn stratified by frequency", "strat")):
    print(f"\n  {label}:")
    for fold in range(5):
        test_spk = set(speakers[fold::5])
        if type_pick == "random":
            shuffled = types[:]
            rng.shuffle(shuffled)
            test_typ = set(shuffled[: len(types) // 5])
        else:
            # walk the frequency-ranked list so head words spread across folds
            test_typ = {w for i, (w, _) in enumerate(ranked) if i % 5 == fold}

        tr = [r for r in rows if r["speaker"] not in test_spk and r["word"] not in test_typ]
        te = [r for r in rows if r["speaker"] in test_spk and r["word"] in test_typ]
        tr_c, te_c = Counter(r["underlying"] for r in tr), Counter(r["underlying"] for r in te)
        print(f"    fold {fold}: train {len(tr):>5} (t={tr_c['t']:>4} d={tr_c['d']:>4})  "
              f"test {len(te):>4} (t={te_c['t']:>3} d={te_c['d']:>3})  "
              f"discarded {total - len(tr) - len(te):>5} "
              f"({(total - len(tr) - len(te)) / total:.0%})")

# --- 4. type-disjoint only, ignoring speaker ----------------------------
print("\ntype-disjoint only (speakers ignored), 5-fold stratified by frequency:")
for fold in range(5):
    test_typ = {w for i, (w, _) in enumerate(ranked) if i % 5 == fold}
    tr = [r for r in rows if r["word"] not in test_typ]
    te = [r for r in rows if r["word"] in test_typ]
    tr_c, te_c = Counter(r["underlying"] for r in tr), Counter(r["underlying"] for r in te)
    print(f"  fold {fold}: train {len(tr):>5} (t={tr_c['t']:>4} d={tr_c['d']:>4})  "
          f"test {len(te):>5} (t={te_c['t']:>4} d={te_c['d']:>4})")

# --- 5. the full 5 x 5 grid behind §3's stratified split -----------------
# §3 tests only the diagonal: fold k holds out speaker fold k AND type fold k,
# so 5 of the 25 (speaker fold, type fold) cells are ever scored. Every cell can
# serve as a test set the same way: test = the cell, train = sites sharing
# neither its speaker fold nor its type fold. Speaker and type folds are both
# partitions, so each site sits in exactly one cell, and scoring all 25 gives
# every site one out-of-fold prediction. Cell (k, k)'s train set is §3's fold k
# train set exactly, so no probe sees less data than under the diagonal design.
#
# Built over the eligible sites only. The clamped sites are excluded before
# scoring (pre-registration §1), and since 2026-09-25 before fold assignment as
# well, so they neither enter the ranking nor sit in any cell.
eligible = [r for r in rows if not r["excerpt_clamped"]]
spk_fold = speaker_folds(r["speaker"] for r in eligible)
typ_fold = type_folds(r["word"] for r in eligible)
cell_of = [(spk_fold[r["speaker"]], typ_fold[r["word"]]) for r in eligible]

print("\nfull 5 x 5 grid: speaker fold x type fold (§3's stratified assignment)")
print(f"  eligible sites: {len(eligible)} of {total} "
      f"({total - len(eligible)} clamped, removed before folds are assigned)")
per_cell = Counter(cell_of)
print(f"  cells populated: {len(per_cell)} of 25; "
      f"sites assigned to a cell: {sum(per_cell.values())} of {len(eligible)}")

grid = {}
for a in range(5):
    for b in range(5):
        te = [r for r, c in zip(eligible, cell_of) if c == (a, b)]
        tr = [r for r, (sa, tb) in zip(eligible, cell_of) if sa != a and tb != b]
        leak = (len({r["speaker"] for r in te} & {r["speaker"] for r in tr})
                + len({r["word"] for r in te} & {r["word"] for r in tr}))
        grid[a, b] = dict(
            n=len(te),
            d=sum(r["underlying"] == "d" for r in te),
            d_types=len({r["word"] for r in te if r["underlying"] == "d"}),
            train=len(tr),
            train_d=sum(r["underlying"] == "d" for r in tr),
            leak=leak,
        )

for field, title in (("n", "test sites"), ("d", "test /d/ sites"),
                     ("d_types", "distinct /d/ word types in test")):
    print(f"\n  {title} (rows = speaker fold, cols = type fold):")
    print("         " + "".join(f"{b:>6}" for b in range(5)))
    for a in range(5):
        print(f"    {a:>3}  " + "".join(f"{grid[a, b][field]:>6}" for b in range(5)))

cells = grid.values()
diag = [grid[k, k] for k in range(5)]
print(f"\n  test sites per cell {min(c['n'] for c in cells)}-{max(c['n'] for c in cells)}, "
      f"/d/ per cell {min(c['d'] for c in cells)}-{max(c['d'] for c in cells)}, "
      f"/d/ types per cell {min(c['d_types'] for c in cells)}-{max(c['d_types'] for c in cells)}")
print(f"  train sites per cell {min(c['train'] for c in cells)}-{max(c['train'] for c in cells)}, "
      f"train /d/ {min(c['train_d'] for c in cells)}-{max(c['train_d'] for c in cells)}")
print(f"  speakers or types shared between a cell's test and train: {sum(c['leak'] for c in cells)}")
print(f"  scored, diagonal only (§3): {sum(c['n'] for c in diag)} sites, "
      f"{sum(c['d'] for c in diag)} /d/")
print(f"  scored, full grid:          {sum(c['n'] for c in cells)} sites, "
      f"{sum(c['d'] for c in cells)} /d/")

# Why the tie-break is registered rather than left implicit. Until 2026-09-24
# `ranked` was most_common(), which orders equal counts by first appearance in
# the TSV, so the fold assignment rode on the inventory's row order. This
# measures how much would move back under that rule, over all 10,392 sites as
# when it was first measured.
typ_fold_all = type_folds(r["word"] for r in rows)
row_order = by_type.most_common()
typ_fold_row = {w: i % 5 for i, (w, _) in enumerate(row_order)}
moved = {w for w in typ_fold_all if typ_fold_all[w] != typ_fold_row[w]}
tied = sum(n for n in Counter(by_type.values()).values() if n > 1)
print(f"\n  types sharing a site count with another type: {tied} of {len(by_type)}")
print(f"  types whose fold would differ under the old row-order tie-break: "
      f"{len(moved)} ({sum(by_type[w] for w in moved)} sites)")

# Why the clamped sites leave before folds are assigned, not only before
# scoring: counted, they would still decide where other sites go.
moved = {w for w in typ_fold if typ_fold[w] != typ_fold_all[w]}
dropped = set(typ_fold_all) - set(typ_fold)
print(f"  types whose fold would differ if the clamped sites were counted: "
      f"{len(moved)} ({sum(by_type[w] for w in moved)} sites); "
      f"types with no eligible site: {len(dropped)} {sorted(dropped)}")
