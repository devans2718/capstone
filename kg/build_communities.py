#!/usr/bin/env python3
"""Spell out each flap's phonetic environment as Phone nodes, then find communities.

HOW TO RUN (from the repo root, in WSL, with the server up: kg/compose.yaml)
    .venv-app/bin/python kg/build_communities.py
    Needs networkx in .venv-app (installed locally, not in
    requirements-app.txt: `uv pip install --python .venv-app/bin/python
    networkx`), the Buckeye .phones files at scripts/count_flap_sites.py's
    DEFAULT_ROOT, and the corpus layer already loaded.

    ORDER: AFTER build_corpus.py. build_corpus.py starts with DETACH DELETE
    on the corpus layer, which drops every edge this script hangs on a Site
    and leaves the Phone and Community nodes orphaned. After any corpus
    reload, run this again. The design layer is not touched.

WHAT IT BUILDS (Challenge 4, AI Design; not part of the registered study)
    Phone      one per segment label that occurs as a span-grid site's
               preceding vowel or following segment. Properties: label,
               in_v (the label is in measure_span_grid.V), n_preceding,
               n_following. layer = 'phone'
    Site -PRECEDED_BY {coda}-> Phone   the vowel the preceding sonorant
               stretch ends on; `coda` is the sonorant coda between it and
               the flap, "" when there is none
    Site -FOLLOWED_BY-> Phone          the following segment, whatever it is
    Community  one per Louvain community over the Site-Phone graph, numbered
               1.. by size. Properties: n_sites, n_t, n_d, phones, and the
               run's seed, resolution and modularity. layer = 'community'
    Site -IN_COMMUNITY-> Community, Phone -IN_COMMUNITY-> Community

    The environments are the span grid's own (experiments/span-grid/README.md
    §1-3): scripts/measure_span_grid.py:locate_spans, over the eligible sites,
    refused unless its exclusion list hashes to the registered one. So a site
    here has exactly the vowel and following segment the span grid cuts its
    spans from, and the 55 excluded sites get no Phone edges.

    One Phone node per label, with the role on the edge type: a vowel that
    precedes one flap and follows another is the same node.

WHAT IS LEFT OUT, AND WHY
    - The /t/-/d/ label stays a Site property. As a node it would be the
      strongest structure in the graph, and community detection would
      rediscover it and present it as a finding. It is used only afterwards,
      to ask whether the communities line up with it.
    - No [ɾ] node: every site has one, so it would be a hub joined to all of
      them and carry no structure.
    - No Token, TypeKey or Lemma edges. They bring word identity in, and word
      identity is already known to dominate the /t/-/d/ signal (the word-axis
      result). The communities here are phonetic environment only.

THE COMMUNITY RUN
    Louvain (networkx.community.louvain_communities) over the undirected
    Site-Phone graph read back from Neo4j, resolution 1, SEED below. The seed
    is not a registered one (capstone.seeds): this is an exploratory graph
    analysis, and the stability check re-runs it over STABILITY_SEEDS and
    reports the spread of modularity and the adjusted Rand index against the
    seeded partition, so the result does not rest on one draw.

    Then the cross-tabs: community x /t/-/d/ (chi-square, Cramer's V), set
    against vowel-following pair x /t/-/d/, which is what the communities
    group. If the communities only coarsen the pairs, their V can't exceed
    the pairs'.

THE BUCKEYE RULE
    Rows go from the inventory and the .phones tier to the server over Bolt.
    The file this writes, kg/communities/summary.json, holds aggregates only:
    community sizes, phone labels (ARPAbet segments, not words) and counts.
    No words, lemmas, site IDs, speaker codes or timestamps.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "kg"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from neo4j import GraphDatabase, RoutingControl  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from build_corpus import AUTH, URI, read_inventory  # noqa: E402
from measure_span_grid import EXPECTED_SHA256, V, exclusion_digest, locate_spans  # noqa: E402

OUT = REPO / "kg" / "communities" / "summary.json"

SEED = 20261028          # the day this was designed; not a registered seed
RESOLUTION = 1.0         # networkx's default, i.e. standard modularity
STABILITY_SEEDS = range(1, 21)
BATCH = 2_000
EXPECTED_SITES = {"kept": 10_334, "t": 7_821, "d": 2_513}   # span-grid README §1


def fail(msg: str) -> None:
    sys.exit(f"build_communities: {msg}")


# ---------------------------------------------------------------------------
# Environments, from the span grid's own code
# ---------------------------------------------------------------------------

def environments() -> list[dict]:
    """One row per span-grid site: its Site key, vowel, coda and following segment."""
    inv = read_inventory()   # refused unless it is the pilot run's inventory
    rows = inv[~inv.excerpt_clamped][["site_id", "track", "start", "underlying"]]
    kept, excluded = locate_spans(rows.to_dict("records"))
    if exclusion_digest(excluded) != EXPECTED_SHA256:
        fail("the span grid's exclusion list differs from the registered one")
    labels = Counter(r["underlying"] for r, *_ in kept)
    got = {"kept": len(kept), "t": labels["t"], "d": labels["d"]}
    if got != EXPECTED_SITES:
        fail(f"span-grid sites {got}, registered {EXPECTED_SITES}")
    return [{"site": f"site:{r['site_id']}", "vowel": v.label,
             "coda": " ".join(reversed(codas)), "following": f.label}
            for r, v, codas, f in kept]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

LOAD_PHONES = """
UNWIND $rows AS r
MERGE (p:Phone {key: r.key})
SET p += r, p.layer = 'phone'
"""

LOAD_EDGES = """
UNWIND $rows AS r
MATCH (s:Site {key: r.site})
MATCH (v:Phone {key: 'phone:' + r.vowel}), (f:Phone {key: 'phone:' + r.following})
MERGE (s)-[e:PRECEDED_BY]->(v) SET e.coda = r.coda
MERGE (s)-[:FOLLOWED_BY]->(f)
"""


def clear(driver) -> None:
    with driver.session() as s:
        s.run("MATCH (n) WHERE n.layer IN ['phone', 'community'] "
              "CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()


def load_phones(driver, envs: list[dict]) -> None:
    for label in ("Phone", "Community"):
        driver.execute_query(f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
                             f"FOR (n:{label}) REQUIRE n.key IS UNIQUE")
    pre = Counter(e["vowel"] for e in envs)
    fol = Counter(e["following"] for e in envs)
    phones = [{"key": f"phone:{p}", "label": p, "in_v": p in V,
               "n_preceding": pre[p], "n_following": fol[p]}
              for p in sorted(set(pre) | set(fol))]
    driver.execute_query(LOAD_PHONES, rows=phones)
    for i in range(0, len(envs), BATCH):
        driver.execute_query(LOAD_EDGES, rows=envs[i:i + BATCH])

    recs, _, _ = driver.execute_query(
        "MATCH (s:Site) RETURN count { (s)-[:PRECEDED_BY]->() } AS pre, "
        "count { (s)-[:FOLLOWED_BY]->() } AS fol, count(*) AS n")
    shape = Counter((r["pre"], r["fol"]) for r in recs for _ in range(r["n"]))
    if shape[(1, 1)] != len(envs):
        fail(f"{shape[(1, 1)]} Sites have one PRECEDED_BY and one FOLLOWED_BY, "
             f"not {len(envs)}: is the corpus layer loaded?")
    print(f"loaded {len(phones)} Phone nodes ({len(pre)} preceding, {len(fol)} following), "
          f"{2 * len(envs):,} edges from {len(envs):,} Sites")


# ---------------------------------------------------------------------------
# Communities, over the graph as read back
# ---------------------------------------------------------------------------

EXPORT = """
MATCH (s:Site)-[e:PRECEDED_BY|FOLLOWED_BY]->(p:Phone)
RETURN s.key AS site, s.underlying AS label, type(e) AS role, p.label AS phone
"""


def export(driver):
    recs, _, _ = driver.execute_query(EXPORT, routing_=RoutingControl.READ)
    G = nx.Graph()
    label, env = {}, defaultdict(dict)
    for r in recs:
        G.add_edge(r["site"], "phone:" + r["phone"])
        label[r["site"]] = r["label"]
        env[r["site"]]["vowel" if r["role"] == "PRECEDED_BY" else "following"] = r["phone"]
    return G, label, env


def louvain(G, seed: int) -> list[set]:
    parts = nx.community.louvain_communities(G, resolution=RESOLUTION, seed=seed)
    return sorted(parts, key=lambda c: (-len(c), min(c)))


def membership(parts: list[set], nodes: list[str]) -> list[int]:
    of = {n: i for i, c in enumerate(parts) for n in c}
    return [of[n] for n in nodes]


def cramers_v(table: np.ndarray) -> tuple[float, float]:
    """(Cramer's V, chi-square p) for a k x 2 table, empty rows dropped."""
    table = table[table.sum(axis=1) > 0]
    chi2, p, _, _ = chi2_contingency(table, correction=False)
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))), float(p)


def crosstab(keys: list, label: dict, sites: list[str]) -> np.ndarray:
    idx = {k: i for i, k in enumerate(sorted(set(keys)))}
    t = np.zeros((len(idx), 2), dtype=int)
    for k, s in zip(keys, sites):
        t[idx[k], 0 if label[s] == "t" else 1] += 1
    return t


WRITE_COMMUNITIES = """
UNWIND $rows AS r
CREATE (c:Community {key: r.key})
SET c += r, c.layer = 'community'
"""

WRITE_MEMBERS = """
UNWIND $rows AS r
MATCH (c:Community {key: r.community})
OPTIONAL MATCH (s:Site {key: r.node})
OPTIONAL MATCH (p:Phone {key: r.node})
WITH c, coalesce(s, p) AS n
MERGE (n)-[:IN_COMMUNITY]->(c)
"""


def main() -> None:
    envs = environments()
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        clear(driver)
        load_phones(driver, envs)
        G, label, env = export(driver)

        parts = louvain(G, SEED)
        Q = nx.community.modularity(G, parts, resolution=RESOLUTION)
        sites = sorted(label)
        seeded = membership(parts, sites)

        # Stability: the same run under other seeds.
        stab = []
        for s in STABILITY_SEEDS:
            p = louvain(G, s)
            stab.append({"seed": s, "n": len(p),
                         "modularity": nx.community.modularity(G, p, resolution=RESOLUTION),
                         "ari": adjusted_rand_score(seeded, membership(p, sites))})

        communities, members = [], []
        for i, c in enumerate(parts, 1):
            key = f"community:{i}"
            cs = [n for n in c if n.startswith("site:")]
            ph = sorted(n.removeprefix("phone:") for n in c if n.startswith("phone:"))
            n_d = sum(label[s] == "d" for s in cs)
            communities.append({
                "key": key, "number": i, "n_sites": len(cs), "n_t": len(cs) - n_d,
                "n_d": n_d, "share_d": round(n_d / len(cs), 4) if cs else None,
                "phones": ph, "seed": SEED, "resolution": RESOLUTION,
                "modularity": round(Q, 4)})
            members += [{"node": n, "community": key} for n in c]
        driver.execute_query(WRITE_COMMUNITIES, rows=communities)
        for i in range(0, len(members), BATCH):
            driver.execute_query(WRITE_MEMBERS, rows=members[i:i + BATCH])

    # The cross-tabs.
    comm_v, comm_p = cramers_v(crosstab(seeded, label, sites))
    pairs = [(env[s]["vowel"], env[s]["following"]) for s in sites]
    pair_v, pair_p = cramers_v(crosstab(pairs, label, sites))
    vowel_v, _ = cramers_v(crosstab([v for v, _ in pairs], label, sites))
    fol_v, _ = cramers_v(crosstab([f for _, f in pairs], label, sites))
    # How much of each community its phones account for: sites whose vowel AND
    # following segment are both members, against those split across two.
    of = {n: i for i, c in enumerate(parts) for n in c}
    whole = sum(of["phone:" + v] == of["phone:" + f] == of[s]
                for s, (v, f) in zip(sites, pairs))

    for c in communities:
        cs = [s for s in sites if of[s] == c["number"] - 1]
        c["top_vowels"] = Counter(env[s]["vowel"] for s in cs).most_common(5)
        c["top_following"] = Counter(env[s]["following"] for s in cs).most_common(5)
        c["n_pairs"] = len({(env[s]["vowel"], env[s]["following"]) for s in cs})

    summary = {
        "graph": {"sites": len(sites), "phones": G.number_of_nodes() - len(sites),
                  "edges": G.number_of_edges(),
                  "distinct_pairs": len(set(pairs)),
                  "base_share_d": round(sum(label[s] == "d" for s in sites) / len(sites), 4)},
        "run": {"algorithm": "louvain (networkx " + nx.__version__ + ")", "seed": SEED,
                "resolution": RESOLUTION, "modularity": round(Q, 4),
                "n_communities": len(parts),
                "sites_with_vowel_and_following_in_own_community": int(whole)},
        "stability": {"seeds": f"{STABILITY_SEEDS.start}-{STABILITY_SEEDS.stop - 1}",
                      "n_communities": sorted(Counter(s["n"] for s in stab).items()),
                      "modularity_min": round(min(s["modularity"] for s in stab), 4),
                      "modularity_max": round(max(s["modularity"] for s in stab), 4),
                      "ari_vs_seeded_min": round(min(s["ari"] for s in stab), 3),
                      "ari_vs_seeded_median": round(float(np.median([s["ari"] for s in stab])), 3)},
        "label_association": {
            "community_x_label": {"cramers_v": round(comm_v, 3), "p": comm_p},
            "pair_x_label": {"cramers_v": round(pair_v, 3), "p": pair_p},
            "vowel_x_label": {"cramers_v": round(vowel_v, 3)},
            "following_x_label": {"cramers_v": round(fol_v, 3)}},
        "communities": [{k: v for k, v in c.items()
                         if k not in ("key", "seed", "resolution", "modularity")}
                        for c in communities],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")

    r, a = summary["run"], summary["label_association"]
    print(f"louvain, seed {SEED}: {r['n_communities']} communities, modularity {r['modularity']}")
    print(f"  stability over seeds {summary['stability']['seeds']}: "
          f"modularity {summary['stability']['modularity_min']}-"
          f"{summary['stability']['modularity_max']}, ARI vs seeded median "
          f"{summary['stability']['ari_vs_seeded_median']} (min "
          f"{summary['stability']['ari_vs_seeded_min']})")
    print(f"  sites whose vowel and following segment share their community: "
          f"{whole:,} of {len(sites):,}")
    print(f"{'#':>3} {'sites':>6} {'/t/':>6} {'/d/':>6} {'%d':>6}  phones")
    for c in communities:
        print(f"{c['number']:>3} {c['n_sites']:>6} {c['n_t']:>6} {c['n_d']:>6} "
              f"{100 * (c['share_d'] or 0):>5.1f}%  {' '.join(c['phones'])}")
    print(f"base rate /d/ {100 * summary['graph']['base_share_d']:.1f}%")
    print(f"Cramer's V with /t/-/d/: community {a['community_x_label']['cramers_v']}, "
          f"vowel-following pair {a['pair_x_label']['cramers_v']}, "
          f"vowel {a['vowel_x_label']['cramers_v']}, "
          f"following {a['following_x_label']['cramers_v']}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
