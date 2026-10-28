#!/usr/bin/env python3
"""Draw the community layer: the collapsed community graph and the /t/-/d/ cross-tab.

HOW TO RUN (from the repo root, in WSL, after kg/build_communities.py)
    .venv-app/bin/python kg/plot_communities.py OUT_DIR
    Writes OUT_DIR/community-graph.png and OUT_DIR/community-crosstab.png.
    Read-only against the server (routing_=READ).

WHAT THEY SHOW
    community-graph     one node per Louvain community, area by its sites,
                        coloured by how far its /d/ share sits from the base
                        rate (blue below, orange above, grey at it). An edge
                        joins two communities when sites have their preceding
                        vowel in one and their following segment in the other;
                        width by how many.
    community-crosstab  each community's /d/ share with a 95% Wilson
                        interval, against the base rate. The cross-tab's
                        counts are in the tick labels.

THE BUCKEYE RULE
    Aggregates only. Node and tick labels are community numbers and phone
    labels (ARPAbet segments); no words, lemmas, site IDs or speaker codes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from neo4j import GraphDatabase, RoutingControl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphrag import AUTH, URI  # noqa: E402

# The dataviz skill's reference palette: diverging blue / orange, grey midpoint.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, MID = "#2a78d6", "#eb6834", "#c9c8c2"
DIVERGING = LinearSegmentedColormap.from_list("t-d", [BLUE, MID, ORANGE])

COMMUNITIES = """
MATCH (c:Community)
RETURN c.number AS number, c.n_sites AS n, c.n_t AS t, c.n_d AS d,
       c.share_d AS share_d, c.phones AS phones, c.modularity AS modularity
ORDER BY number
"""

# Sites whose vowel and following segment sit in two different communities.
BRIDGES = """
MATCH (cv:Community)<-[:IN_COMMUNITY]-(:Phone)<-[:PRECEDED_BY]-(s:Site)
      -[:FOLLOWED_BY]->(:Phone)-[:IN_COMMUNITY]->(cf:Community)
WHERE cv <> cf
WITH CASE WHEN cv.number < cf.number THEN [cv.number, cf.number]
          ELSE [cf.number, cv.number] END AS pair
RETURN pair[0] AS a, pair[1] AS b, count(*) AS n
"""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def phones_text(phones: list[str], limit: int = 5) -> str:
    shown = " ".join(phones[:limit])
    return shown + (f" +{len(phones) - limit}" if len(phones) > limit else "")


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


def graph_figure(cs: list[dict], bridges: list[dict], base: float, out: Path) -> None:
    G = nx.Graph()
    for c in cs:
        G.add_node(c["number"], **c)
    for b in bridges:
        G.add_edge(b["a"], b["b"], weight=b["n"])
    # A circle in community order (largest first, clockwise from the top):
    # a force layout piles the large, densely bridged communities on top of
    # each other.
    order = [c["number"] for c in cs]
    pos = {n: (math.sin(2 * math.pi * i / len(order)), math.cos(2 * math.pi * i / len(order)))
           for i, n in enumerate(order)}

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_xlim(-1.5, 1.85)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    wmax = max(d["weight"] for *_, d in G.edges(data=True))
    for a, b, d in G.edges(data=True):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], color=INK_2, alpha=0.35,
                lw=0.5 + 6 * d["weight"] / wmax, zorder=1, solid_capstyle="round")
    norm = TwoSlopeNorm(vmin=0, vcenter=base, vmax=0.6)
    for n, c in G.nodes(data=True):
        x, y = pos[n]
        ax.scatter(x, y, s=60 + 2.2 * c["n"], color=DIVERGING(norm(c["share_d"])),
                   edgecolors=SURFACE, linewidths=2, zorder=2)
        r = math.sqrt(60 + 2.2 * c["n"]) / 2 + 6
        ax.annotate(f"C{n}  {phones_text(c['phones'], 4)}\n"
                    f"{c['n']:,} sites, {100 * c['share_d']:.0f}% /d/",
                    (x, y), xytext=(r * x, r * y), textcoords="offset points",
                    ha="left" if x > 0.15 else "right" if x < -0.15 else "center",
                    va="bottom" if y > 0.15 else "top" if y < -0.15 else "center",
                    fontsize=8, color=INK, zorder=3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    cb.set_label(f"share /d/  (grey = base rate, {100 * base:.1f}%)", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2)
    ax.set_title(f"{len(cs)} Louvain communities over the flaps' phonetic environments "
                 f"(modularity {cs[0]['modularity']:.2f})",
                 color=INK, fontsize=13, loc="left", pad=28)
    ax.text(0, -0.02, "Node area: sites. Edge width: sites whose preceding vowel and "
            "following segment fall in different communities. Labels: ARPAbet phones.",
            transform=ax.transAxes, color=INK_2, fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def crosstab_figure(cs: list[dict], base: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=SURFACE)
    style(ax)
    y = list(range(len(cs)))[::-1]
    for yi, c in zip(y, cs):
        lo, hi = wilson(c["d"], c["n"])
        colour = ORANGE if lo > base else BLUE if hi < base else MID
        ax.plot([lo, hi], [yi, yi], color=colour, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(c["share_d"], yi, s=60, color=colour, edgecolors=SURFACE,
                   linewidths=2, zorder=3)
    ax.axvline(base, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(base, -0.9, f" base rate {100 * base:.1f}% /d/",
            color=INK_2, fontsize=9, va="center")
    ax.set_yticks(y, [f"C{c['number']}  {phones_text(c['phones'])}   "
                      f"{c['t']:,} /t/ · {c['d']:,} /d/" for c in cs], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("share of the community's sites with underlying /d/  (95% Wilson interval)",
                  color=INK_2)
    ax.set_ylim(-1.3, len(cs) - 0.5)
    ax.set_title("Community × underlying /t/–/d/", color=INK, fontsize=13, loc="left", pad=12)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=l)
               for c, l in ((BLUE, "below base rate"), (MID, "interval covers it"),
                            (ORANGE, "above base rate"))]
    ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK_2,
              fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        cs = [r.data() for r in driver.execute_query(
            COMMUNITIES, routing_=RoutingControl.READ).records]
        bridges = [r.data() for r in driver.execute_query(
            BRIDGES, routing_=RoutingControl.READ).records]
    if not cs:
        sys.exit("plot_communities: no Community nodes; run kg/build_communities.py first")
    base = sum(c["d"] for c in cs) / sum(c["n"] for c in cs)
    graph_figure(cs, bridges, base, out / "community-graph.png")
    crosstab_figure(cs, base, out / "community-crosstab.png")
    print(f"wrote {out / 'community-graph.png'}, {out / 'community-crosstab.png'}")


if __name__ == "__main__":
    main()
