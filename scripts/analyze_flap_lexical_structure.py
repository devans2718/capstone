#!/usr/bin/env python3
"""Is a word-type-disjoint split feasible on the Buckeye flap inventory?

Read-only. Answers four questions: how many word types carry the 10,392 sites,
how concentrated they are, whether `underlying` is a deterministic function of
the word type, and what survives a split that is type-disjoint and
speaker-disjoint at the same time.
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_TSV = (Path(__file__).resolve().parent.parent
               / "results/corpus-verification/buckeye-flap-sites-20260821.tsv")
TSV = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TSV

with open(TSV, newline="") as fh:
    rows = list(csv.DictReader(fh, delimiter="\t"))

print(f"sites: {len(rows)}")
print(f"class counts: {Counter(r['underlying'] for r in rows)}")
print(f"speakers: {len({r['speaker'] for r in rows})}")

# --- 1. types and concentration -----------------------------------------
by_type = Counter(r["word"] for r in rows)
print(f"\ndistinct word types: {len(by_type)}")

ranked = by_type.most_common()
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
