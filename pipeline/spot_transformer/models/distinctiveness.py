"""Supervised **distinctiveness** — learn which spots a human calls "special".

The whole re-ID track so far treats every spot as equal evidence. But the manual pair review
(``artifacts/pair_review/.../comments.json``) is emphatic that they are *not*:

    "the small round one should be penalized because it's not very shapy, it's very round"
    "spot 96 is so characteristic that the lack of this spot on the other should be penalized"
    "the shapes are very round so they should be penalized"

and the same intuition was captured directly as labels by the ``interesting-spot-selector`` web
app: for 362 images a human clicked the spots that "draw the eye", giving ~8.7k per-spot
binary labels (``artifacts/interesting_spots/<folder>/interesting_spots.json``, keyed by the DB
``spot_id``). That is real supervision for the notion ``strict_match.distinctiveness`` only ever
*guessed* at with hand weights.

This module:

* turns those clicks into a per-spot label aligned to the ``ImageSet`` tokens,
* computes the same six interpretable factors ``strict_match`` uses (size, elongation,
  non-circularity, irregularity, rarity, isolation),
* fits a tiny logistic regression factor -> P(interesting), **individual-split** so the score
  generalizes to unseen animals, and
* exposes :func:`predict_weights` so the strict voter can weight each spot by learned
  distinctiveness instead of the hand blend.

Running it as a script is the *inspection*: it prints held-out AUROC (how learnable "special"
is), the learned coefficients (which factor makes a spot special, in the human's eyes) and how
the hand-weighted ``strict_match`` score compares.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# flat imports — the package is mid-refactor and eval/aggregator_e2e only resolve this way
_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

import data as d                              # noqa: E402
import review_labels as rl                    # noqa: E402
import strict_match as sm                     # noqa: E402
from aggregator import train_aggregator, _logit, _prob   # noqa: E402
from census import auroc                       # noqa: E402

# ``rarity`` is out of the learned feature set as well as out of ``strict_match.DEFAULT_WEIGHTS``.
# It is not merely a weight of zero: with the hand blend skipping the neighbour search, the attached
# ``rarity`` column is a constant, and a constant feature contributes nothing but a shifted
# intercept. Keeping it would also quietly re-enable the O(n log n) population search the moment
# anyone raised its weight. ``ALL_FACTOR_NAMES`` retains it so the A/B that justified removing it
# stays runnable: :func:`main` loads the factors under ``LEGACY_WEIGHTS`` (the one caller that
# does), which computes the rarity column for real, and prints the with/without comparison on every
# ``pixi run distinctiveness``. There is no flag to turn the A/B off — a claim nobody re-checks is
# how ``DEFAULT_WEIGHTS`` and the sweeps drifted apart in the first place.
FACTOR_NAMES = ["size", "elongation", "noncircularity", "irregularity", "isolation"]
ALL_FACTOR_NAMES = ["size", "elongation", "noncircularity", "irregularity", "rarity", "isolation"]

INTERESTING_JSON = (d.REPO_ROOT / "artifacts" / "interesting_spots" / "all_sasa_norm"
                    / "interesting_spots.json")


# ----------------------------------------------------------------------------- labels
def load_interesting(path: Path | str | None = None,
                     dataset: str | None = None) -> dict[str, set[int]]:
    """``salamander_id -> {spot_id, ...}`` of human-tagged interesting spots.

    Reads the **preprocessing review store** by default (``review_labels``), not the legacy
    ``interesting_spots.json`` export: the export is a union snapshot written only when someone runs
    ``preprocess-export``, so training on it silently uses whatever the labels looked like at the
    last export rather than now. On this dataset that gap is 385 vs 477 images (2,521 vs 2,973
    clicks) — a quarter of the supervision, invisible.

    Either store keys spots by DB ``spot_id`` (the UI resolves a click through the mask table), so
    these align directly with :attr:`ImageSet.spot_ids` — no positional guessing. Pass ``path`` to
    force the legacy file (the ablation in ``compare_strict``'s ``LABELS=legacy`` uses it).
    """
    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return {sid: set(int(i) for i in ids) for sid, ids in raw["labels"].items()}
    return rl.interesting_spots(dataset or d.dataset_name)


# ----------------------------------------------------------------------------- factors
def _as_contour(c) -> np.ndarray:
    """DuckDB ``DOUBLE[][]`` -> (k, 2) boundary array, orientation-normalized.

    Stored as a list of ``[x, y]`` pairs; a few rows are empty/ragged, so build defensively and
    fall back to an empty (0, 2) contour (``shape_descriptors`` returns zeros for < 3 points)."""
    if c is None:
        return np.zeros((0, 2))
    try:
        a = np.array([list(p) for p in c], float)
    except (TypeError, ValueError):
        try:
            a = np.asarray(c, float)
        except (TypeError, ValueError):
            return np.zeros((0, 2))
    if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:   # stored as [xs, ys]
        a = a.T
    if a.ndim != 2 or a.shape[1] != 2:
        return np.zeros((0, 2))
    return a


def blend(frame: pd.DataFrame, weights: dict) -> np.ndarray:
    """Re-blend already-computed factor columns with a different weight set, in [0, 1].

    The factors are the expensive part (shape descriptors over every spot); the blend is a weighted
    mean. Separating them means an A/B over weights costs one pass, not one pass per variant — and
    guarantees both arms see numerically identical factors.

    One trap: re-blending with a NON-ZERO ``rarity`` is only meaningful on a frame loaded under
    ``LEGACY_WEIGHTS``. Under the default the neighbour search is skipped and the ``rarity`` column
    is the constant 0.5, so the "legacy" arm would silently come out as the shipped blend plus an
    offset. :func:`main` is the caller that gets this right.
    """
    names = [n for n in ALL_FACTOR_NAMES if n in frame.columns]
    wsum = sum(weights.get(n, 0.0) for n in names) or 1.0
    return sum(weights.get(n, 0.0) * frame[n].to_numpy(float) for n in names) / wsum


def load_spot_factors(sets, db_path=None, weights: dict | None = None) -> pd.DataFrame:
    """One row per spot (aligned to the concatenation of every set's tokens) carrying the six
    :mod:`strict_match` factors, computed **once over the whole population** (size is
    dataset-relative). Returns a frame with ``sid``, ``spot_id``, the ``FACTOR_NAMES`` columns and
    the blended ``distinctiveness``.

    ``weights`` blends that ``distinctiveness`` column and defaults to the **shipped**
    ``strict_match.DEFAULT_WEIGHTS`` — see the note at the call below for why that default is not a
    detail. Isolation is measured in body-normalized (axis_t, lateral-offset) coords, per individual
    — exactly as the reviewer described "not many other spots around".
    """
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        raw = con.execute(
            "SELECT salamander_id, spot_id, area_pixels, local_contour, axis_t, axis_offset "
            "FROM spots ORDER BY salamander_id, spot_id"
        ).df()
    finally:
        con.close()
    raw["local_contour"] = raw["local_contour"].map(_as_contour)

    # build the emb array aligned to `sets` token order, and a parallel spots-frame in the SAME
    # order so strict_match's population statistics match the matcher's tokens exactly.
    rows, embs = [], []
    key_to_row = {(r.salamander_id, int(r.spot_id)): r for r in raw.itertuples(index=False)}
    for s in sets:
        for j, sid_int in enumerate(s.spot_ids):
            r = key_to_row.get((s.sid, int(sid_int)))
            if r is None:
                continue
            rows.append(dict(sid=s.sid, salamander_id=s.sid, spot_id=int(sid_int),
                             area_pixels=r.area_pixels, local_contour=r.local_contour,
                             axis_t=r.axis_t, axis_offset=r.axis_offset))
            embs.append(s.spots[j])
    frame = pd.DataFrame(rows)
    emb = sm._rank01  # noqa: F841  (keep import warm; not used directly)
    E = np.stack(embs).astype(np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    # DEFAULT_WEIGHTS — the SHIPPED blend — unless a caller explicitly asks for something else.
    # This used to default to LEGACY_WEIGHTS so that :func:`main`'s A/B had a real rarity column to
    # compare against, and that default silently undid the change everywhere else: the
    # ``distinctiveness`` column written here IS what :func:`hand_weight_lookup` hands the matcher,
    # so `WEIGHT_MODE=hand` in ``compare_strict`` and ``sweep_bakeoff``'s unlabelled fallback both
    # kept scoring on rarity at 1.5 — while paying the O(n log n) neighbour search (~240 s) to do
    # it. The A/B is the one place that wants the factor computed, and it now asks for it by name.
    factors = sm.distinctiveness(frame, E, weights=weights or sm.DEFAULT_WEIGHTS, attach=True)
    return factors


def attach_labels(factors: pd.DataFrame,
                  interesting: dict[str, set[int]]) -> pd.DataFrame:
    """Add ``y`` (1 = interesting) and ``labeled`` (this image was reviewed) to the factor frame.

    Only images the human actually opened count as supervision: an unlabeled image is *unknown*,
    not *all-boring*, so its spots must be excluded from training rather than treated as negatives.
    """
    out = factors.copy()
    out["labeled"] = out["sid"].isin(interesting.keys())
    out["y"] = [int(sid in interesting and int(spid) in interesting[sid])
                for sid, spid in zip(out["sid"], out["spot_id"])]
    return out


# ----------------------------------------------------------------------------- model
def fit_distinctiveness(frame: pd.DataFrame, *, seed: int = 0):
    """Logistic regression over the six factors on the LABELED spots. Returns ``(model, scaler)``.

    Deliberately tiny/linear: the point is an interpretable, generalizable weight — the coefficients
    say *which* factor makes a spot special — not to squeeze the last AUROC point out of the label.
    """
    lab = frame[frame["labeled"]]
    X = lab[FACTOR_NAMES].to_numpy(float)
    y = lab["y"].to_numpy(float)
    return train_aggregator(X, y, hidden=0, seed=seed)


def predict_weights(model, scaler, frame: pd.DataFrame) -> np.ndarray:
    """P(interesting) in [0, 1] for every row of ``frame`` (aligned to its order)."""
    X = frame[FACTOR_NAMES].to_numpy(float)
    return _prob(model, scaler, X)


def weight_lookup(model, scaler, frame: pd.DataFrame) -> dict[tuple[str, int], float]:
    """``(sid, spot_id) -> learned distinctiveness weight`` for the strict voter to gate on."""
    w = predict_weights(model, scaler, frame)
    return {(sid, int(spid)): float(wi)
            for sid, spid, wi in zip(frame["sid"], frame["spot_id"], w)}


def hand_weight_lookup(frame: pd.DataFrame) -> dict[tuple[str, int], float]:
    """The same lookup from the frame's ``distinctiveness`` column — the hand-tuned blend, no
    fitting. That column is whatever :func:`load_spot_factors` blended, i.e.
    ``strict_match.DEFAULT_WEIGHTS`` unless the caller asked for another weight set.

    The ablation baseline for "does learning the weight from the clicks beat guessing it". Note this
    is *not* an unsupervised control: the hand weights were themselves chosen by reading the pair
    review, so this comparison measures fitting-vs-eyeballing the same human intuition, not
    supervision-vs-none. :func:`uniform_weight_lookup` is the no-supervision control.
    """
    return {(sid, int(spid)): float(w)
            for sid, spid, w in zip(frame["sid"], frame["spot_id"], frame["distinctiveness"])}


def uniform_weight_lookup(frame: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Every spot weighted 0.5 — equal-weight voting, the no-distinctiveness control.

    ``strict_voter`` already defaults an unknown spot to 0.5, so this makes the whole strict score
    fall back to un-weighted coverage x support and isolates what the weighting buys.
    """
    return {(sid, int(spid)): 0.5 for sid, spid in zip(frame["sid"], frame["spot_id"])}


# ----------------------------------------------------------------------------- inspection
def crossval_auroc(frame: pd.DataFrame, *, k: int = 5, seed: int = 0,
                   factors: list[str] | None = None) -> dict:
    """Individual-split CV AUROC of the learned score vs the human labels, plus the hand-weighted
    ``strict_match`` score as a training-free reference. Split on individual so a spot's animal is
    never in both train and test."""
    lab = frame[frame["labeled"]].reset_index(drop=True)
    labels = np.array(["_".join(s.split("_")[:2]) for s in lab["sid"]])   # individual id
    uniq = np.array(sorted(set(labels)))
    rng = np.random.default_rng(seed)
    uniq = uniq[rng.permutation(len(uniq))]
    folds = np.array_split(uniq, k)

    y = lab["y"].to_numpy(float)
    X = lab[factors or FACTOR_NAMES].to_numpy(float)
    hand = lab["distinctiveness"].to_numpy(float)

    learned_scores = np.full(len(lab), np.nan)
    aurocs = []
    for fold in folds:
        te = np.isin(labels, fold)
        tr = ~te
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        model, scaler = train_aggregator(X[tr], y[tr], hidden=0, seed=seed)
        s = _prob(model, scaler, X[te])
        learned_scores[te] = s
        aurocs.append(auroc(s, y[te]))
    ok = ~np.isnan(learned_scores)
    return dict(
        learned_auroc_mean=float(np.mean(aurocs)), learned_auroc_std=float(np.std(aurocs)),
        learned_auroc_pooled=float(auroc(learned_scores[ok], y[ok])),
        hand_auroc=float(auroc(hand, y)),
        n=int(ok.sum()), n_pos=int(y[ok].sum()),
    )


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    logger.info(" " + rl.review_summary(d.dataset_name))
    logger.info("loading sets + factors ...")
    sets = d.get_image_sets(d.get_spot_embeddings())
    # The ONLY caller that asks for LEGACY_WEIGHTS: it forces the rarity neighbour search so the
    # "does dropping rarity cost anything" A/B below has a real column to compare, at the cost of
    # the ~240 s every other caller now skips. The blended column is immediately replaced by the
    # shipped blend, so `hand_auroc` below reports what the matcher actually uses, not the legacy.
    factors = load_spot_factors(sets, weights=sm.LEGACY_WEIGHTS)
    factors["distinctiveness"] = blend(factors, sm.DEFAULT_WEIGHTS)
    frame = attach_labels(factors, load_interesting())
    n_lab = int(frame["labeled"].sum())
    logger.info(f" spots={len(frame)}  labeled={n_lab}  interesting={int(frame['y'].sum())} "
          f"({frame[frame['labeled']]['y'].mean():.3f} of labeled)")

    logger.info("=== Is 'special' learnable from the factors? (individual-split CV) ===")
    cv = crossval_auroc(frame, k=5, seed=0)
    logger.info(f" learned logreg AUROC  {cv['learned_auroc_mean']:.3f} ± {cv['learned_auroc_std']:.3f}"
          f"  (pooled {cv['learned_auroc_pooled']:.3f})")
    logger.info(f" hand strict_match     AUROC {cv['hand_auroc']:.3f}   (training-free reference)")
    logger.info(f" n={cv['n']}  positives={cv['n_pos']}  (0.5 = chance)")

    # --- the A/B that justifies removing rarity, run every time so the claim stays checked ---
    logger.info("=== DROPPING RARITY — does it cost anything? ===")
    cv_all = crossval_auroc(frame, k=5, seed=0, factors=ALL_FACTOR_NAMES)
    logger.info(f" learned, WITH rarity ({len(ALL_FACTOR_NAMES)} factors):  "
          f"{cv_all['learned_auroc_mean']:.3f} ± {cv_all['learned_auroc_std']:.3f}")
    logger.info(f" learned, WITHOUT     ({len(FACTOR_NAMES)} factors):  "
          f"{cv['learned_auroc_mean']:.3f} ± {cv['learned_auroc_std']:.3f}"
          f"   -> {cv['learned_auroc_mean'] - cv_all['learned_auroc_mean']:+.3f}")
    legacy_hand = auroc(blend(frame[frame["labeled"]], sm.LEGACY_WEIGHTS),
                        frame[frame["labeled"]]["y"].to_numpy(float))
    new_hand = auroc(blend(frame[frame["labeled"]], sm.DEFAULT_WEIGHTS),
                     frame[frame["labeled"]]["y"].to_numpy(float))
    logger.info(f" hand blend, WITH rarity (legacy, 1.5):  {legacy_hand:.3f}")
    logger.info(f" hand blend, WITHOUT     (shipped, 0.0): {new_hand:.3f}"
          f"   -> {new_hand - legacy_hand:+.3f}")
    if new_hand + 1e-9 >= legacy_hand and cv["learned_auroc_mean"] + 0.005 >= cv_all["learned_auroc_mean"]:
        logger.info(" => removing rarity costs nothing on either path. Keep it out.")
    else:
        logger.warning(" => removing rarity COSTS something here — revisit before shipping the change.")

    # The label refresh is itself an experiment: if the extra 92 images do not move held-out AUROC,
    # that is results.md #34 ("more of the same does not help") showing up at the label level, and
    # it is worth knowing before anyone collects more clicks.
    legacy = INTERESTING_JSON
    cv_old = old = None
    if legacy.is_file():
        old = attach_labels(factors, load_interesting(path=legacy))
        cv_old = crossval_auroc(old, k=5, seed=0)
        logger.info(f" legacy export ({int(old['labeled'].sum())} labeled spots, "
              f"{old['sid'][old['labeled']].nunique()} images):"
              f"  AUROC {cv_old['learned_auroc_mean']:.3f} ± {cv_old['learned_auroc_std']:.3f}")
        logger.info(f" review store  ({n_lab} labeled spots, "
              f"{frame['sid'][frame['labeled']].nunique()} images):"
              f"  AUROC {cv['learned_auroc_mean']:.3f} ± {cv['learned_auroc_std']:.3f}"
              f"   -> {cv['learned_auroc_mean'] - cv_old['learned_auroc_mean']:+.3f} from the refresh")

    logger.info("=== What makes a spot 'special'? (logreg weights, standardized) ===")
    model, scaler = fit_distinctiveness(frame)
    w = model.net.weight.detach().numpy().ravel()
    for nm, wt in sorted(zip(FACTOR_NAMES, w), key=lambda t: -abs(t[1])):
        logger.info(f"   {nm:16s} {wt:+.3f}")

    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "distinctiveness"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "RESULTS_distinctiveness.md").write_text("\n".join([
        "# Learned distinctiveness — is 'special' learnable from the six factors?", "",
        f"- dataset `{d.dataset_name}` · {rl.review_summary(d.dataset_name)}",
        f"- {n_lab} labeled spots ({int(frame['y'].sum())} interesting) over "
        f"{frame['sid'][frame['labeled']].nunique()} images", "",
        "| score | held-out AUROC |", "|---|---|",
        f"| learned logreg (individual-split CV) | **{cv['learned_auroc_mean']:.3f} ± "
        f"{cv['learned_auroc_std']:.3f}** |",
        f"| hand `strict_match.DEFAULT_WEIGHTS` (shipped, rarity 0) | {cv['hand_auroc']:.3f} |", "",
        "**Dropping `rarity`** — re-checked on every run, because this is the claim the shipped",
        "`DEFAULT_WEIGHTS` rests on:", "",
        "| path | with rarity | without | delta |", "|---|---|---|---|",
        f"| learned logreg | {cv_all['learned_auroc_mean']:.3f} ± {cv_all['learned_auroc_std']:.3f}"
        f" | {cv['learned_auroc_mean']:.3f} ± {cv['learned_auroc_std']:.3f} | "
        f"{cv['learned_auroc_mean'] - cv_all['learned_auroc_mean']:+.3f} |",
        f"| hand blend | {legacy_hand:.3f} (weight 1.5) | {new_hand:.3f} (weight 0) | "
        f"{new_hand - legacy_hand:+.3f} |", "",
        *([
            "**Does the label refresh help?** — the export was 92 images stale, so this is the",
            "experiment on whether collecting more clicks is worth a session:", "",
            "| labels | images | labeled spots | held-out AUROC |", "|---|---|---|---|",
            f"| legacy export | {old['sid'][old['labeled']].nunique()} | "
            f"{int(old['labeled'].sum())} | {cv_old['learned_auroc_mean']:.3f} ± "
            f"{cv_old['learned_auroc_std']:.3f} |",
            f"| review store (live) | {frame['sid'][frame['labeled']].nunique()} | {n_lab} | "
            f"{cv['learned_auroc_mean']:.3f} ± {cv['learned_auroc_std']:.3f} |", "",
            f"Delta from the refresh: **{cv['learned_auroc_mean'] - cv_old['learned_auroc_mean']:+.3f}**.",
            "", ] if cv_old is not None else []),
        "**Learned factor weights** (standardized). The shipped hand blend weights `size` 2.0 and",
        "`elongation` / `noncircularity` / `irregularity` / `isolation` 1.0 each; compare those",
        "against what the labels actually support:", "",
        *[f"- `{nm}` {wt:+.3f}" for nm, wt in sorted(zip(FACTOR_NAMES, w), key=lambda t: -abs(t[1]))],
    ]) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / 'RESULTS_distinctiveness.md'}")


if __name__ == "__main__":
    main()
