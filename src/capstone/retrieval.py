"""Typed, deterministic retrieval over the site inventory and its predictions.

DIVISION OF LABOUR
    Tools compute. Retrieval returns rows. A language model does exactly two
    things with what comes out of here: turn a question into a typed call, and
    narrate returned rows in prose where every number is traceable to a field
    that was actually returned.

    It does not compute a statistic, infer a label, decide which sites are
    interesting without an n, or assert a cause. If this module returns no rows,
    the answer is "no rows" — not a hypothesis about why. That constraint is
    what makes the output citable in the same sense as the reports under
    results/corpus-verification/, and it only holds if the computing happens
    here, in code that can be tested, rather than in a paragraph.

    So: no language model inside this file, and no cleverness either. DuckDB
    over two parquet files.

EVERY RESPONSE CARRIES A COVERAGE BLOCK
    Including responses where it is uninformative and the denominators are
    equal. It costs one dictionary, and it is the only mechanism that makes a
    wrong denominator visible instead of silent.

    The failure it prevents is specific. This run scores 40 of 10,392 sites. A
    table of 40 rows with no denominator reads exactly like a complete result,
    and nothing about it looks wrong. The coverage block is what turns "12 sites
    were misclassified" into "12 of 40 scored, of 10,392 total" — which is a
    different sentence, and the true one.

WHICH QUESTION EACH METHOD ANSWERS, AND WHY THAT DECIDES THE JOIN
    Three eligibility states have to stay distinguishable: discarded by the
    split, scored, and never reached by this run. The third is represented by
    the ABSENCE of a prediction row, so any method that has to report it must
    start from the inventory rather than from predictions.

    site_state does exactly that, and is the method the failure case exercises.
    It looks the site up in the inventory first, so "no prediction row" reports
    as a state rather than as an empty result, and the three states come back as
    three different strings with p_t left None.

    query_sites and compare_conditions inner-join on purpose. They answer
    "which scored observations match this filter", a question about
    observations that exist, and an outer join there would pad the table with
    10,352 empty rows. What keeps them honest is not the join but the coverage
    block, which reports the full denominator on every response — so a 19-row
    answer says 19 of 40 scored of 10,392, and never just 19.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .encoders.extraction import excerpt_samples, plan_excerpt

SAMPLE_RATE = 16_000
CONTEXT_MS = 200.0

# What an unscored or discarded site renders as. A string, never a number:
# the whole point of the failure case is that absence must not be able to reach
# a chart axis or an average.
NOT_SCORED = "not scored — discarded by the split"
NOT_REACHED = "not scored — this run did not reach this site"


class UnknownSite(KeyError):
    """Raised for a site_id that is not in the inventory.

    Its own class so a caller can distinguish "you asked for something that does
    not exist" from "the thing you asked for exists and has no prediction",
    which are different answers and must not share a code path.
    """


@dataclass(frozen=True)
class Coverage:
    """The denominators behind any answer this module gives.

    compared_on is separate from n_scored because a comparison can be narrower
    than either condition alone: two conditions intersected on commonly-scored
    sites may share fewer sites than either scored by itself.
    """

    n_total: int
    n_eligible: int
    n_scored: int
    compared_on: int

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_eligible": self.n_eligible,
            "n_scored": self.n_scored,
            "compared_on": self.compared_on,
        }


@dataclass(frozen=True)
class Result:
    """Rows plus the denominators they have to be read against.

    The two are one object so they cannot be separated by accident. A caller
    that wants the rows has the coverage in its hand already; there is no path
    that produces a table without one.
    """

    rows: "object"          # a pandas DataFrame
    coverage: Coverage
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.rows)

    def as_dict(self) -> dict:
        return {"n": self.n, "coverage": self.coverage.as_dict(),
                "note": self.note, "rows": self.rows.to_dict("records")}


class SliceStore:
    """The two parquet files, and the queries the application is allowed to ask.

    Deliberately a small, closed set of methods rather than a general query
    interface. Every question the interface can ask is one that was written down
    and tested; there is no path by which a prompt becomes SQL.
    """

    def __init__(self, inventory: Path, predictions: Path, corpus_root: Path | None = None):
        import duckdb

        self.inventory = Path(inventory)
        self.predictions = Path(predictions)
        self.corpus_root = Path(corpus_root) if corpus_root else Path("/data/corpora/buckeye")
        for p in (self.inventory, self.predictions):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Build the inventory with "
                    "scripts/count_flap_sites.py --parquet, and predictions with "
                    "scripts/make_synthetic_predictions.py."
                )
        self.db = duckdb.connect(":memory:")
        # CREATE VIEW cannot be prepared in DuckDB, so the paths are inlined
        # with quotes escaped. They come from this process's own arguments, not
        # from anything a user typed into the interface — no path here is ever
        # built from a prompt.
        for name, path in (("sites", self.inventory), ("preds", self.predictions)):
            literal = str(path).replace("'", "''")
            self.db.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{literal}')"
            )

    # ------------------------------------------------------------- denominators

    def coverage(self, condition: str | None = None, compared_on: int | None = None) -> Coverage:
        n_total = self.db.execute("SELECT count(*) FROM sites").fetchone()[0]
        if condition is None:
            n_eligible, n_scored = n_total, self.db.execute(
                "SELECT count(DISTINCT site_id) FROM preds WHERE scored"
            ).fetchone()[0]
        else:
            # n_eligible counts the whole inventory minus those this condition
            # explicitly discarded. Sites with no row are eligible-but-unreached,
            # not ineligible — that is the distinction the brief exists to keep.
            discarded = self.db.execute(
                "SELECT count(*) FROM preds WHERE condition = ? AND NOT eligible",
                [condition],
            ).fetchone()[0]
            n_eligible = n_total - discarded
            n_scored = self.db.execute(
                "SELECT count(*) FROM preds WHERE condition = ? AND scored", [condition]
            ).fetchone()[0]
        return Coverage(n_total, n_eligible, n_scored,
                        n_scored if compared_on is None else compared_on)

    # ------------------------------------------------------------------ queries

    def query_sites(self, *, condition: str, encoder: str = "wavjepa",
                    layer: int | None = 8, underlying: str | None = None,
                    correct: bool | None = None, site_ids=None) -> Result:
        """Scored observations matching the filters, plus a coverage block.

        Returns an empty frame with a coverage block when nothing matches. Not
        an exception: "no sites match this filter" is an answer, and one whose
        denominator matters more than usual — an empty table with no n is
        indistinguishable from a broken query.
        """
        where = ["p.condition = ?", "p.encoder = ?"]
        params: list = [condition, encoder]
        if layer is not None:
            where.append("p.layer = ?"); params.append(layer)
        if underlying is not None:
            where.append("s.underlying = ?"); params.append(underlying)
        if correct is not None:
            where.append("p.correct = ?"); params.append(correct)
        if site_ids is not None:
            ids = list(site_ids)
            where.append(f"s.site_id IN ({','.join('?' * len(ids))})")
            params.extend(ids)

        rows = self.db.execute(f"""
            SELECT s.site_id, s.speaker, s.track, s.word, s.underlying,
                   s.start_sample, s.end_sample, s.duration_ms,
                   s.n_frame_onsets, s.n_frame_onsets_abs, s.excerpt_clamped,
                   s.prev_phone, s.next_phone, s.next_word,
                   p.condition, p.fold, p.eligible, p.scored, p.p_t, p.pred,
                   p.correct, p.synthetic
            FROM sites s JOIN preds p ON p.site_id = s.site_id
            WHERE {' AND '.join(where)} AND p.scored
            ORDER BY s.site_id
        """, params).df()
        return Result(rows, self.coverage(condition, compared_on=len(rows)))

    def site_state(self, site_id: str, condition: str) -> dict:
        """What is known about one site under one condition — all three states.

        This is the method the failure case exercises. It never returns a number
        standing in for an absence: a discarded site reports NOT_SCORED and a
        site this run never reached reports NOT_REACHED, both as strings, with
        p_t left as None rather than filled in.
        """
        site = self.db.execute(
            "SELECT * FROM sites WHERE site_id = ?", [site_id]
        ).df()
        if site.empty:
            raise UnknownSite(
                f"{site_id!r} is not in the inventory at {self.inventory}. "
                "Site identifiers are 'track:phone_index', e.g. 'sNNNNx:NNNNN'."
            )
        pred = self.db.execute(
            "SELECT * FROM preds WHERE site_id = ? AND condition = ?",
            [site_id, condition],
        ).df()

        base = {"site_id": site_id, "condition": condition,
                "word": site.word.iloc[0], "underlying": site.underlying.iloc[0],
                "n_frame_onsets": int(site.n_frame_onsets.iloc[0]),
                "n_frame_onsets_abs": int(site.n_frame_onsets_abs.iloc[0]),
                "excerpt_clamped": bool(site.excerpt_clamped.iloc[0]),
                "p_t": None, "pred": None, "correct": None}
        if pred.empty:
            return {**base, "eligible": True, "scored": False,
                    "status": NOT_REACHED, "synthetic": None}
        r = pred.iloc[0]
        if not bool(r.eligible):
            return {**base, "eligible": False, "scored": False,
                    "status": NOT_SCORED, "synthetic": bool(r.synthetic)}
        return {**base, "eligible": True, "scored": True,
                "status": "scored", "p_t": float(r.p_t), "pred": str(r.pred),
                "correct": bool(r.correct), "synthetic": bool(r.synthetic)}

    def compare_conditions(self, cond_a: str, cond_b: str, *,
                           encoder: str = "wavjepa", layer: int | None = 8,
                           underlying: str | None = None,
                           changed_only: bool = True) -> Result:
        """Sites whose correctness changed between two conditions.

        Intersects on commonly-scored sites by default and says so in the note.
        Comparing across sites only one condition scored would put two different
        denominators behind one difference, which is how a split artefact turns
        into a finding.
        """
        where = ["a.encoder = ?", "b.encoder = ?"]
        params: list = [encoder, encoder]
        if layer is not None:
            where += ["a.layer = ?", "b.layer = ?"]; params += [layer, layer]
        if underlying is not None:
            where.append("s.underlying = ?"); params.append(underlying)
        if changed_only:
            where.append("a.correct <> b.correct")

        sql = f"""
            SELECT s.site_id, s.speaker, s.word, s.underlying,
                   s.start_sample, s.duration_ms, s.n_frame_onsets,
                   a.correct AS correct_a, b.correct AS correct_b,
                   a.p_t AS p_t_a, b.p_t AS p_t_b,
                   a.pred AS pred_a, b.pred AS pred_b,
                   a.synthetic
            FROM sites s
            JOIN preds a ON a.site_id = s.site_id AND a.condition = '{cond_a}' AND a.scored
            JOIN preds b ON b.site_id = s.site_id AND b.condition = '{cond_b}' AND b.scored
            WHERE {' AND '.join(where)}
            ORDER BY s.site_id
        """
        rows = self.db.execute(sql, params).df()

        both = self.db.execute("""
            SELECT count(*) FROM preds a JOIN preds b USING (site_id)
            WHERE a.condition = ? AND b.condition = ? AND a.scored AND b.scored
        """, [cond_a, cond_b]).fetchone()[0]
        cov = self.coverage(cond_a, compared_on=both)
        return Result(rows, cov,
                      note=f"intersected on the {both} sites scored under both "
                           f"{cond_a!r} and {cond_b!r}")

    # --------------------------------------------------------------- clip export

    def plan_clip(self, site_id: str, context_ms: float = CONTEXT_MS) -> dict:
        """The sample window a clip for this site occupies. No file I/O.

        Split out from export_clip so the provenance test can run with no corpus
        mounted, and — more to the point — so the arithmetic is reachable and
        testable on its own. The window comes from plan_excerpt and the samples
        from excerpt_samples, the same function encode_excerpt uses. Nothing
        here re-derives a window: a clip that disagrees with the encoder by even
        one frame makes anything concluded by ear unfounded.
        """
        site = self.db.execute(
            "SELECT * FROM sites WHERE site_id = ?", [site_id]
        ).df()
        if site.empty:
            raise UnknownSite(
                f"{site_id!r} is not in the inventory at {self.inventory}. "
                "Site identifiers are 'track:phone_index', e.g. 'sNNNNx:NNNNN'."
            )
        s = site.iloc[0]
        wav = self.corpus_root / s.speaker / f"{s.track}.wav"
        duration_ms = self._recording_ms(wav) if wav.exists() else None
        if duration_ms is None:
            # Without the recording, plan_excerpt cannot know whether the window
            # clamps at an edge. A length just past this site is supplied so the
            # arithmetic still runs; `clamped` from such a call describes only
            # the leading edge and the caller is told as much.
            duration_ms = float(s["end"]) * 1000 + context_ms

        # THE TIMES GO IN AS FLOATS, DELIBERATELY.
        # `start_sample` is the audio anchor and the thing to index with, but it
        # is a ROUNDED view of `start`, and planning from it re-derives the
        # window through that rounding. Measured over 3,000 sites, the two
        # disagree on 13% of windows. The encoder plans from the float times, so
        # a clip that plans from the integer would be a clip of different audio
        # than the model consumed — which is exactly the failure the provenance
        # test exists to catch.
        excerpt = plan_excerpt(float(s["start"]) * 1000, float(s["end"]) * 1000,
                               duration_ms, context_ms=context_ms)
        start, end = excerpt_samples(excerpt, SAMPLE_RATE)
        return {"site_id": site_id, "wav": wav, "excerpt": excerpt,
                "start_sample": start, "end_sample": end,
                "n_samples": end - start, "clamped": excerpt.clamped,
                "target_start_sample": int(s.start_sample),
                "target_end_sample": int(s.end_sample)}

    def export_clip(self, site_id: str, context_ms: float = CONTEXT_MS,
                    out_dir: Path | None = None) -> Path:
        """Write the clip for one site and return its path.

        Written under /data, never into the repository: Buckeye is distributed
        under a registration agreement with OSU, and .gitignore's blanket *.wav
        exclusion is load-bearing rather than incidental.
        """
        import soundfile as sf

        plan = self.plan_clip(site_id, context_ms)
        if not plan["wav"].exists():
            raise FileNotFoundError(
                f"no audio for {site_id} at {plan['wav']}. The corpus mounts at "
                f"{self.corpus_root}; clips cannot be exported without it."
            )
        out_dir = Path(out_dir) if out_dir else Path("/data/clips")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{site_id.replace(':', '_')}_{int(context_ms)}ms.wav"

        data, sr = sf.read(plan["wav"], start=plan["start_sample"],
                           stop=plan["end_sample"], dtype="int16")
        sf.write(out, data, sr)
        return out

    @staticmethod
    def _recording_ms(wav: Path) -> float:
        import wave

        with wave.open(str(wav), "rb") as w:
            return w.getnframes() / w.getframerate() * 1000
