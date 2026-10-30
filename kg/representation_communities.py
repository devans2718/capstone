#!/usr/bin/env python3
"""Communities over the encoders' features: what each representation puts together.

EXPLORATORY, NOT REGISTERED. No probe is trained, retrained or tuned, and none
of the span grid's registration or results is touched. This is a training-free
look at the pooled features the span grid already scored (Challenge 4, the
representation-communities brief of 2026-10-30).

HOW TO RUN (from the repo root, in WSL)
    .venv-app/bin/python kg/representation_communities.py
    Needs networkx in .venv-app (see build_communities.py), the span features
    under /data/embeddings/spans-b378c303ca90/, the inventory and lemma table
    under results/flapping/ (local, gitignored), and the Buckeye .phones files
    for the 10-28 partition. No Neo4j: nothing is read from or written to it.
    Then kg/plot_representation_communities.py draws the figures.

WHAT IT DOES
    Sites are nodes. For each representation, on the ɾ span, at its best
    ɾ layer in results/span-grid/ (wav2vec2-base 9, WavJEPA 3, MFCC 0):
      - z-score each dimension over the 10,334 grid sites, then join each site
        to its K nearest neighbours by cosine distance; undirected (union of
        the directed kNN edges), unweighted. Nodes are added in inventory
        order, so a seed fixes the partition (the 10-29 lesson).
      - Louvain at resolution 1, SEED; the stability check re-runs it over
        STABILITY_SEEDS and reports the spread of community counts and
        modularity, and the ARI against the seeded partition. K_SENSITIVITY
        re-runs the seeded partition at a smaller and a larger k.
    Each partition is then compared, by adjusted mutual information (AMI),
    with /t/-/d/, the preceding vowel, the following segment, the vowel-
    following pair, speaker, word (type_key), lemma, and the 10-28 Site-Phone
    partition. AMI is corrected for chance on both sides, so a 2-way label and
    a 983-way one can sit in the same table; plain Cramer's V grows with the
    number of categories. For /t/-/d/ only, Cramer's V is also given plain and
    bias-corrected (Bergsma 2013), to sit beside the 10-28 run's 0.299.

    The 10-28 partition is recomputed here, not read from Neo4j: the same
    environments (build_communities.environments), the graph built in the same
    sorted order its EXPORT returns, the same seed. It is used only if it
    reproduces kg/communities/summary.json exactly (count, modularity, sizes).

THE TOKEN STRUCTURE
    The sites are tokens of repeated words from 40 speakers, so a few frequent
    words can carry a community's /t/-/d/ skew. Per community: its distinct-
    word count, the top-5 words' share of its tokens (no words shown), and its
    /d/ share with two 95% intervals: Wilson, which treats sites as independent
    (what kg/plot_communities.py draws), and one resampled by word (type_key:
    words drawn with replacement, all of a word's tokens kept; BOOT resamples,
    SEED, percentile).

THE BUCKEYE RULE
    Per-site rows stay in memory. kg/communities/representation/summary.json
    holds aggregates only: counts, shares, AMI and V values, stability. No
    words, lemmas, site IDs, speaker codes or timestamps.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "kg"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

import build_communities as bc  # noqa: E402
import run_probes as rp  # noqa: E402

OUT = REPO / "kg" / "communities" / "representation" / "summary.json"
SITE_PHONE = REPO / "kg" / "communities" / "summary.json"

SPAN = "r"
LAYERS = {"wav2vec2": 9, "wavjepa": 3, "mfcc": 0}   # best on ɾ, span-grid-summary-20261029.txt
K = 15
K_SENSITIVITY = (5, 30)
SEED = 20261030          # the day this was run; not a registered seed
STABILITY_SEEDS = range(1, 21)
BOOT = 10_000
TARGETS = ("label", "vowel", "following", "pair", "speaker", "word", "lemma", "site_phone")


def fail(msg: str) -> None:
    sys.exit(f"representation_communities: {msg}")


# ---------------------------------------------------------------------------
# The sites and what they are compared against
# ---------------------------------------------------------------------------

def grid_sites(d) -> pd.DataFrame:
    """One row per grid site, in inventory order: every target as a column."""
    g = np.flatnonzero(d.in_grid)
    inv = d.inv.iloc[g]
    sites = pd.DataFrame({"idx": g, "site_id": inv.site_id.to_numpy(),
                          "label": inv.underlying.to_numpy(),
                          "speaker": inv.speaker.to_numpy(),
                          "word": [d.keys[i] for i in g]})
    lem = pd.read_csv(rp.LEMMA_TSV, sep="\t", usecols=["site_id", "type_key", "lemma"])
    sites = sites.merge(lem, on="site_id", how="left", validate="one_to_one")
    if sites.lemma.isna().any() or (sites.type_key != sites.word).any():
        fail("the lemma table does not cover the grid's sites with the same type keys")
    env = {e["site"].removeprefix("site:"): e for e in bc.environments()}
    if set(env) != set(sites.site_id):
        fail("build_communities.environments() is not the grid's 10,334 sites")
    sites["vowel"] = [env[s]["vowel"] for s in sites.site_id]
    sites["following"] = [env[s]["following"] for s in sites.site_id]
    sites["pair"] = sites.vowel + " " + sites.following
    return sites.drop(columns="type_key")


def site_phone_partition(sites: pd.DataFrame) -> np.ndarray:
    """The 10-28 partition, rebuilt offline and checked against its summary.json."""
    G = nx.Graph()
    # build_communities.EXPORT: ORDER BY site, role; FOLLOWED_BY sorts first.
    for s in sorted(sites.itertuples(), key=lambda r: "site:" + r.site_id):
        G.add_edge("site:" + s.site_id, "phone:" + s.following)
        G.add_edge("site:" + s.site_id, "phone:" + s.vowel)
    parts = bc.louvain(G, bc.SEED)
    Q = nx.community.modularity(G, parts, resolution=bc.RESOLUTION)
    ref = json.loads(SITE_PHONE.read_text())
    sizes = [sum(n.startswith("site:") for n in c) for c in parts]
    if (len(parts), round(Q, 4), sizes) != (ref["run"]["n_communities"], ref["run"]["modularity"],
                                            [c["n_sites"] for c in ref["communities"]]):
        fail(f"the offline Site-Phone partition ({len(parts)} communities, Q {Q:.4f}) is not "
             f"the one in {SITE_PHONE.relative_to(REPO)}: read it from Neo4j instead")
    return np.array(bc.membership(parts, ["site:" + s for s in sites.site_id]))


# ---------------------------------------------------------------------------
# The kNN graph and its communities
# ---------------------------------------------------------------------------

def features(d, rep: str, sites: pd.DataFrame) -> np.ndarray:
    X = np.load(rp.features_path(rep, SPAN, rp.SPANS_DIR), mmap_mode="r")
    X = np.asarray(X[sites.idx.to_numpy(), LAYERS[rep]], dtype=np.float64)
    if np.isnan(X).any():
        fail(f"{rep}_{SPAN} layer {LAYERS[rep]} has NaN rows among the grid's sites")
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def neighbours(X: np.ndarray, k: int) -> np.ndarray:
    """Each row's k nearest other rows by cosine distance, (n, k)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(X)
    _, idx = nn.kneighbors(X)
    out = np.empty((len(X), k), dtype=np.int64)
    for i, row in enumerate(idx):
        # Itself, normally first; a duplicate row can tie with it.
        others = row[row != i]
        out[i] = others[:k]
    return out


def knn_graph(nbrs: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(nbrs)))
    G.add_edges_from((i, int(j)) for i in range(len(nbrs)) for j in nbrs[i])
    return G


def run(G: nx.Graph, seed: int) -> tuple[np.ndarray, float, int]:
    parts = bc.louvain(G, seed)
    Q = nx.community.modularity(G, parts, resolution=bc.RESOLUTION)
    return np.array(bc.membership(parts, list(range(G.number_of_nodes())))), Q, len(parts)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def cramers_v(table: np.ndarray) -> tuple[float, float]:
    """(plain V, bias-corrected V, Bergsma 2013) for a contingency table."""
    table = table[table.sum(axis=1) > 0]
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    plain = math.sqrt(phi2 / (min(r, k) - 1))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return plain, math.sqrt(phi2c / min(kc - 1, rc - 1))


def label_table(membership: np.ndarray, is_d: np.ndarray) -> np.ndarray:
    t = np.zeros((membership.max() + 1, 2), dtype=int)
    np.add.at(t, (membership, is_d.astype(int)), 1)
    return t


def compare(membership: np.ndarray, targets: dict) -> dict:
    out = {t: round(float(adjusted_mutual_info_score(targets[t], membership)), 4)
           for t in TARGETS}
    plain, corrected = cramers_v(label_table(membership, targets["label"] == "d"))
    out["label_cramers_v"] = round(plain, 3)
    out["label_cramers_v_corrected"] = round(corrected, 3)
    return out


# ---------------------------------------------------------------------------
# Token structure
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def token_structure(membership: np.ndarray, word: np.ndarray, is_d: np.ndarray,
                    draws: np.ndarray) -> list[dict]:
    """Per community, by size: counts, word concentration, both /d/ intervals.

    draws: (BOOT, W) multiplicity of each word in each resample, shared by
    every partition so their intervals rest on the same resamples.
    """
    words, w = np.unique(word, return_inverse=True)
    C = membership.max() + 1
    tokens = np.zeros((C, len(words)))
    d_tok = np.zeros((C, len(words)))
    np.add.at(tokens, (membership, w), 1)
    np.add.at(d_tok, (membership, w), is_d)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = (draws @ d_tok.T) / (draws @ tokens.T)          # (BOOT, C)
    out = []
    for c in range(C):
        n, k = int(tokens[c].sum()), int(d_tok[c].sum())
        per_word = np.sort(tokens[c][tokens[c] > 0])[::-1]
        lo, hi = wilson(k, n)
        s = share[:, c][~np.isnan(share[:, c])]
        out.append({"number": c + 1, "n_sites": n, "n_d": k, "share_d": round(k / n, 4),
                    "n_words": int(len(per_word)),
                    "top5_word_share": round(float(per_word[:5].sum() / n), 4),
                    "wilson_95": [round(lo, 4), round(hi, 4)],
                    "word_resampled_95": [round(float(np.percentile(s, 2.5)), 4),
                                          round(float(np.percentile(s, 97.5)), 4)],
                    "resamples_without_tokens": int(BOOT - len(s))})
    return out


def interval_summary(cs: list[dict], base: float) -> dict:
    """How many communities' intervals exclude the base rate, and how much wider."""
    def excludes(iv):
        return iv[0] > base or iv[1] < base
    ratio = [(c["word_resampled_95"][1] - c["word_resampled_95"][0])
             / (c["wilson_95"][1] - c["wilson_95"][0]) for c in cs]
    return {"n_communities": len(cs),
            "exclude_base_wilson": sum(excludes(c["wilson_95"]) for c in cs),
            "exclude_base_word_resampled": sum(excludes(c["word_resampled_95"]) for c in cs),
            "width_ratio_word_over_wilson_median": round(float(np.median(ratio)), 2),
            "sites_in_communities_whose_top5_words_hold_half": int(sum(
                c["n_sites"] for c in cs if c["top5_word_share"] >= 0.5))}


# ---------------------------------------------------------------------------

def main() -> None:
    d = rp.SpanData()
    sites = grid_sites(d)
    n = len(sites)
    if n != 10_334:
        fail(f"{n:,} grid sites, not 10,334")
    targets = {t: sites[t].to_numpy() for t in TARGETS if t != "site_phone"}
    targets["site_phone"] = site_phone_partition(sites)
    is_d = targets["label"] == "d"
    base = float(is_d.mean())
    print(f"{n:,} sites, {int(is_d.sum()):,} /d/; {len(set(targets['speaker']))} speakers, "
          f"{len(set(targets['word']))} words, {len(set(targets['lemma']))} lemmas; "
          f"Site-Phone partition reproduced ({targets['site_phone'].max() + 1} communities)")

    rng = np.random.default_rng(SEED)
    W = len(set(targets["word"]))
    draws = np.zeros((BOOT, W))
    for b in range(BOOT):
        draws[b] = np.bincount(rng.integers(0, W, W), minlength=W)

    reps, memberships = {}, {}
    for rep in LAYERS:
        X = features(d, rep, sites)
        # Nearest first, so a smaller k is a prefix of the largest.
        widest = neighbours(X, max(K, *K_SENSITIVITY))
        nbrs = {k: widest[:, :k] for k in (K, *K_SENSITIVITY)}
        G = knn_graph(nbrs[K])
        seeded, Q, nc = run(G, SEED)
        memberships[rep] = seeded
        stab = [run(G, s) for s in STABILITY_SEEDS]
        sens = {}
        for k in K_SENSITIVITY:
            m, q, c = run(knn_graph(nbrs[k]), SEED)
            sens[str(k)] = {"n_communities": c, "modularity": round(q, 4),
                            "ari_vs_k15": round(float(adjusted_rand_score(seeded, m)), 3),
                            "ami": compare(m, targets)}
        cs = token_structure(seeded, targets["word"], is_d, draws)
        reps[rep] = {
            "layer": LAYERS[rep], "dims": X.shape[1],
            "graph": {"edges": G.number_of_edges(),
                      "max_degree": max(dg for _, dg in G.degree()),
                      "median_degree": float(np.median([dg for _, dg in G.degree()]))},
            "run": {"seed": SEED, "n_communities": nc, "modularity": round(Q, 4)},
            "stability": {"seeds": f"{STABILITY_SEEDS.start}-{STABILITY_SEEDS.stop - 1}",
                          "n_communities": sorted(Counter(c for *_, c in stab).items()),
                          "modularity_min": round(min(q for _, q, _ in stab), 4),
                          "modularity_max": round(max(q for _, q, _ in stab), 4),
                          "ari_vs_seeded_min": round(min(
                              adjusted_rand_score(seeded, m) for m, *_ in stab), 3),
                          "ari_vs_seeded_median": round(float(np.median(
                              [adjusted_rand_score(seeded, m) for m, *_ in stab])), 3),
                          # The partition can move between seeds while what it
                          # shares with each target does not: the range of each AMI.
                          "ami_range": {t: [min(a), max(a)] for t, a in zip(TARGETS, zip(*(
                              [round(float(adjusted_mutual_info_score(targets[t], m)), 4)
                               for t in TARGETS] for m, *_ in stab)))}},
            "k_sensitivity": sens,
            "comparison": compare(seeded, targets),
            "intervals": interval_summary(cs, base),
            "communities": cs}
        r = reps[rep]
        print(f"{rep} L{LAYERS[rep]}: {nc} communities, Q {Q:.4f}; stability ARI median "
              f"{r['stability']['ari_vs_seeded_median']} (min {r['stability']['ari_vs_seeded_min']})"
              f"; AMI " + ", ".join(f"{t} {r['comparison'][t]}" for t in TARGETS))

    sp = token_structure(targets["site_phone"], targets["word"], is_d, draws)
    reference = {"comparison": compare(targets["site_phone"], targets),
                 "intervals": interval_summary(sp, base), "communities": sp}
    # The environment labels themselves, as a ceiling on what any grouping of
    # them can share with /t/-/d/.
    ceiling = {t: round(float(adjusted_mutual_info_score(targets["label"], targets[t])), 4)
               for t in ("vowel", "following", "pair", "speaker", "word", "lemma")}
    between = {f"{a}-{b}": round(float(adjusted_rand_score(memberships[a], memberships[b])), 3)
               for i, a in enumerate(LAYERS) for b in list(LAYERS)[i + 1:]}

    words = Counter(targets["word"])
    summary = {
        "exploratory": "not registered; no probe trained, retrained or tuned",
        "sites": {"n": n, "n_d": int(is_d.sum()), "base_share_d": round(base, 4),
                  "speakers": len(set(targets["speaker"])), "words": W,
                  "lemmas": len(set(targets["lemma"])),
                  "top5_word_share": round(sum(c for _, c in words.most_common(5)) / n, 4)},
        "setup": {"features": str(rp.SPANS_DIR), "span": SPAN, "layers": LAYERS,
                  "standardize": "z-score per dimension over the grid's sites",
                  "metric": "cosine", "k": K, "k_sensitivity": list(K_SENSITIVITY),
                  "graph": "undirected union of kNN edges, unweighted",
                  "algorithm": "louvain (networkx " + nx.__version__ + ")",
                  "resolution": bc.RESOLUTION, "seed": SEED,
                  "comparison": "adjusted mutual information (sklearn, arithmetic); "
                                "Cramer's V plain and bias-corrected (Bergsma 2013) for /t/-/d/",
                  "word_bootstrap": {"unit": "type_key", "resamples": BOOT, "seed": SEED}},
        "representations": reps,
        "site_phone_reference": reference,
        "label_ami_of_the_labels_themselves": ceiling,
        "ari_between_representations": between,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("Site-Phone reference AMI: " + ", ".join(
        f"{t} {reference['comparison'][t]}" for t in TARGETS))
    print(f"/t/-/d/ AMI of the labels themselves: {ceiling}")
    print(f"ARI between representations: {between}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
