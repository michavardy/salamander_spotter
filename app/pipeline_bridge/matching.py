"""Matching seam onto ``pipeline/spot_transformer`` + ``pipeline/spot_embedding`` (spec §7.3).

* :class:`MatchingBridge` — the single active model scores a query image against
  every enrolled individual's photos -> per-pair similarity.
* :class:`CorrespondenceBridge` — an always-available geometric matcher that
  produces spot-to-spot assignments for the Match-lines view (visualisation only).

Tests use the ``Fake*`` variants; the ``Pipeline*`` variants call the ML code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PairScore:
    individual_id: str
    best_photo_id: str
    similarity: float


@dataclass(frozen=True)
class SpotLink:
    query_spot_id: int
    cand_image_id: str
    cand_spot_id: int
    score: float


@runtime_checkable
class MatchingBridge(Protocol):
    def score(
        self, query_image_id: str, gallery: list[str], contours_db_path: Path
    ) -> list[PairScore]:
        """One :class:`PairScore` per gallery individual, best photo chosen."""
        ...


@runtime_checkable
class CorrespondenceBridge(Protocol):
    def correspond(
        self, query_image_id: str, cand_image_id: str, contours_db_path: Path
    ) -> list[SpotLink]:
        ...


# --------------------------------------------------------------------------- #
class FakeMatchingBridge:
    """Deterministic scores driven by a supplied table, else a stable hash."""

    def __init__(self, scores: dict[tuple[str, str], float] | None = None, model_name: str = "fake_matcher"):
        self.scores = scores or {}
        self.model_name = model_name
        self.calls: list[str] = []

    def score(self, query_image_id, gallery, contours_db_path):  # noqa: D102
        self.calls.append(query_image_id)
        out: list[PairScore] = []
        for individual_id in gallery:
            key = (query_image_id, individual_id)
            if key in self.scores:
                sim = self.scores[key]
            else:
                # unspecified gallery members read as clear non-matches
                h = abs(hash(f"{query_image_id}|{individual_id}")) % 1000 / 1000.0
                sim = round(0.05 + 0.25 * h, 4)
            out.append(PairScore(individual_id=individual_id, best_photo_id=f"{individual_id}_1", similarity=sim))
        out.sort(key=lambda p: p.similarity, reverse=True)
        return out


class FakeCorrespondenceBridge:
    def correspond(self, query_image_id, cand_image_id, contours_db_path):  # noqa: D102
        return [
            SpotLink(query_spot_id=i, cand_image_id=cand_image_id, cand_spot_id=i, score=0.8)
            for i in range(1, 4)
        ]


# --------------------------------------------------------------------------- #
class StubMatchingBridge:
    """The safe default until a matcher is actually validated (spec M3).

    ``PipelineMatchingBridge`` below is a real, working implementation — but
    checked against this app's own imported dataset it ranks the true match at
    ~population-chance (median rank ≈ gallery_size/2 across gallery sizes
    5/20/60 — no measurable discriminating signal), matching the pipeline's own
    docs calling plain soft-chamfer a weak baseline "to improve upon", not a
    deployable matcher. Serving that as if it worked would produce confident-
    looking but essentially random review suggestions — worse than not
    matching. Use ``Bridges.experimental_matcher()`` to opt in anyway.
    """

    def score(self, query_image_id, gallery, contours_db_path):  # noqa: D102
        raise NotImplementedError(
            "No validated matcher is wired in yet. PipelineMatchingBridge exists and runs "
            "(soft-chamfer over contours.db's spot_embeddings) but measured at population-"
            "chance rank-1 on this dataset — not trustworthy for review suggestions. "
            "See app/bridges.py Bridges.production() and docs/salamander_spotter_spec.md M3."
        )


class StubCorrespondenceBridge:
    def correspond(self, query_image_id, cand_image_id, contours_db_path):  # noqa: D102
        raise NotImplementedError(
            "No validated correspondence matcher is wired in yet — see StubMatchingBridge."
        )


# --------------------------------------------------------------------------- #
#  real matcher: soft-chamfer over precomputed per-spot embeddings                #
# --------------------------------------------------------------------------- #
# `pipeline/spot_transformer/core/embeddings.py` (`get_spot_embeddings`, combine="concat")
# writes each spot as two independently L2-normalized 31-dim halves — shape, then
# body position — concatenated. So for two stored rows, `dot(v1, v2)` lands in
# [-2, 2] and equals `cos_shape + cos_pos`; dividing by 2 reproduces exactly the
# "(cos_shape + cos_pos)/2 -- an ADDITIVE, compensatory rule" that module documents
# (measured: every stored row has ||v|| == sqrt(2), confirming the two unit halves).
# No re-embedding, no GPU, no LLM call — this table was written once when the
# dataset/photo was extracted; here it is only read and multiplied.
_ADDITIVE_COSINE_DIVISOR = 2.0


def _table_exists(con, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
    )


def _spot_embeddings(con, table: str, image_id: str) -> tuple[list[int], "np.ndarray"]:
    import numpy as np

    rows = con.execute(
        f"SELECT spot_id, embedding FROM {table} WHERE salamander_id = ? ORDER BY spot_id",
        [image_id],
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0))
    spot_ids = [r[0] for r in rows]
    return spot_ids, np.array([r[1] for r in rows], dtype=np.float64)


def _real_photos_of(con, individual_id: str) -> list[str]:
    """Every non-synthetic photo already extracted for this individual — synthetic
    views are training-only, never a match candidate (spec §7.7, D11)."""
    return [
        r[0]
        for r in con.execute(
            "SELECT salamander_id FROM images WHERE starts_with(salamander_id, ?) AND NOT is_synthetic",
            [f"{individual_id}_"],
        ).fetchall()
    ]


class PipelineMatchingBridge:
    """Real matcher (spec M3): soft-chamfer mean-best-cosine over each photo's
    precomputed per-spot embeddings (see module note above). Scores every real
    photo of every gallery individual and keeps that individual's best.

    Needs the dataset's ``spot_embeddings`` table (built once by
    ``pixi run build-embeddings`` — already present in a dataset transferred via
    §7.9 A, since that copies ``contours.db`` verbatim).
    """

    model_name = "soft_chamfer_embeddings"

    def __init__(self, embedding_table: str = "spot_embeddings"):
        self.embedding_table = embedding_table

    def score(self, query_image_id, gallery, contours_db_path):  # noqa: D102
        import duckdb

        con = duckdb.connect(str(contours_db_path), read_only=True)
        try:
            if not _table_exists(con, self.embedding_table):
                raise RuntimeError(
                    f"contours.db has no '{self.embedding_table}' table — run "
                    f"`pixi run build-embeddings --combine both` on the dataset first "
                    f"(see docs/running.md)."
                )
            _, eq = _spot_embeddings(con, self.embedding_table, query_image_id)
            if eq.shape[0] == 0:
                return []

            out: list[PairScore] = []
            for individual_id in gallery:
                best_sim: float | None = None
                best_photo: str | None = None
                for photo_id in _real_photos_of(con, individual_id):
                    if photo_id == query_image_id:
                        continue
                    _, ec = _spot_embeddings(con, self.embedding_table, photo_id)
                    if ec.shape[0] == 0:
                        continue
                    sim = float((eq @ ec.T / _ADDITIVE_COSINE_DIVISOR).max(axis=1).mean())
                    if best_sim is None or sim > best_sim:
                        best_sim, best_photo = sim, photo_id
                if best_photo is not None:
                    out.append(PairScore(individual_id=individual_id, best_photo_id=best_photo, similarity=best_sim))
            return out
        finally:
            con.close()


class PipelineCorrespondenceBridge:
    """Geometric spot-to-spot correspondence for Match-lines (spec §7.3) — always
    available, independent of whichever matcher is active. Mutual-nearest-neighbour
    assignment (``pipeline/spot_transformer/core/strict_match.correspondences``,
    the module's shipped default) over the same precomputed embeddings."""

    def __init__(self, embedding_table: str = "spot_embeddings"):
        self.embedding_table = embedding_table

    def correspond(self, query_image_id, cand_image_id, contours_db_path):  # noqa: D102
        import duckdb

        from pipeline.spot_transformer.core.strict_match import correspondences

        con = duckdb.connect(str(contours_db_path), read_only=True)
        try:
            if not _table_exists(con, self.embedding_table):
                raise RuntimeError(f"contours.db has no '{self.embedding_table}' table.")
            q_ids, eq = _spot_embeddings(con, self.embedding_table, query_image_id)
            c_ids, ec = _spot_embeddings(con, self.embedding_table, cand_image_id)
            if not q_ids or not c_ids:
                return []
            s = eq @ ec.T / _ADDITIVE_COSINE_DIVISOR
            return [
                SpotLink(query_spot_id=q_ids[i], cand_image_id=cand_image_id,
                         cand_spot_id=c_ids[j], score=float(s[i, j]))
                for i, j in correspondences(s)
            ]
        finally:
            con.close()
