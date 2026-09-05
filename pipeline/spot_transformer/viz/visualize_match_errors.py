"""One SVG grid of spot matches — top / middle / bottom by score, for EVERY match algorithm.

Each real photo of an individual that has another real photo is a query; it is matched against
every other real photo and its best candidate is its top-1. Every algorithm's queries are ranked
by that top-1 score and cut into three bands:

    TOP     most confident matches      (mostly correct — what success looks like)
    MID     the decision boundary       (correct and wrong mixed — where the errors live)
    BOT     least confident matches     (misses and hard cases)

so the failure modes are comparable across methods. Any ``--methods`` from two groups:

  PER-SPOT (rank by soft-chamfer AND draw their OWN correspondence arcs):
    raw          the 62-d hand-engineered embedding
    strict       raw correspondences scored by the STRICT rule (distinctiveness-weighted
                 coverage x match support -- penalizes unmatched characteristic spots, round
                 spots, few matches, mismatches; see core/strict_match.py)
    ssl_simclr   the SimCLR-pretrained per-spot embedding
    ssl_corr     the correspondence-mined SSL embedding

  LEARNED MATCHERS (trained on CV fold-0, rank held-out queries; NO spot-level output, so their
  arcs fall back to the raw 62-d correspondences the model SCORED -- an honest 'why' view, not a
  per-spot decision the model made):
    logreg  mlp_deep                       (summary-feature classifiers)
    axial_cnn  set_transformer  deepsets_cons   (match-set networks)
    e2e_pretrained  e2e_transformer  pretrained_cnn   (encoder + voting)

Each cell is three panels — match arcs on top, both individuals' spot-size maps below:

    [ spot-match arcs: query vs matched ]   (spans the cell top)
    [ query spots by size ] [ matched spots by size ]

    MIN_QUALITY=0.4 pixi run python pipeline/spot_transformer/visualize_match_errors.py
    MIN_QUALITY=0.4 pixi run python pipeline/spot_transformer/visualize_match_errors.py \
        --methods raw,logreg,axial_cnn,e2e_pretrained --n 5

Learned matchers train on fold-0 first (VIZ_EP_SET / VIZ_EP_E2E cap the epochs -- lighter than
the sweep, since this ranks rather than chases a leaderboard number). WARNING: cells = methods x 3
x n; the full 10-method x 5 grid is 150 cells (~40 MB) and 30+ min. It is meant to be zoomed, not
printed -- cut --methods or --n for a lighter file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embeddings as E                                          # noqa: E402
import matplotlib                                              # noqa: E402

matplotlib.use("Agg")                                          # headless: write files, no window
import matplotlib.pyplot as plt                               # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)


def derive_label(sid: str) -> str:
    return "_".join(sid.split("_")[:2])


def is_synth(sid: str) -> bool:
    return sid.split("_")[-1].startswith("g")


def load_spots():
    """Spots with the SAME small-spot filter used to build the embeddings, plus the embedding
    column, so the ranking and the drawn arcs come from one representation."""
    spots = E.filter_small_spots(E.get_spots(), min_pct=10)
    spots = E._ensure_embeddings(spots)

    # optional image-quality gate (same env vars as the sweep), applied to REAL photos only
    min_q = os.environ.get("MIN_QUALITY")
    if min_q:
        import duckdb
        con = duckdb.connect(str(E.DB_PATH), read_only=True)
        ok = {r[0] for r in con.execute(
            "SELECT salamander_id FROM image_quality WHERE overall_quality >= ?",
            [float(min_q)]).fetchall()}
        con.close()
        keep = spots.salamander_id.map(lambda s: is_synth(s) or s in ok)
        spots = spots[keep].reset_index(drop=True)
        logger.info(f"  MIN_QUALITY={min_q}: kept {spots.salamander_id.nunique()} images")
    return spots


def rank_embedding_rows(spots, emb_col):
    """Per-query top-1 by soft-chamfer over any PER-SPOT embedding column. When ``emb_col`` is
    also the arc source, the ranking and the drawn correspondences use one representation. Used
    for raw (62-d) and each SSL embedding (64-d) alike -- they differ only in the column."""
    by_img: dict[str, np.ndarray] = {}
    for sid, sub in spots.groupby("salamander_id"):
        if is_synth(sid):
            continue
        e = np.vstack(sub[emb_col].to_numpy())
        by_img[sid] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    ids = [i for i in by_img if len(by_img[i]) >= 3]
    by_label: dict[str, list[str]] = {}
    for i in ids:
        by_label.setdefault(derive_label(i), []).append(i)
    queries = [i for i in ids if len(by_label[derive_label(i)]) >= 2]
    rows = []
    for q in queries:
        eq = by_img[q]
        best_s, best_j = -1.0, None
        for g in ids:
            if g == q:
                continue
            s = float((eq @ by_img[g].T).max(1).mean())        # soft-chamfer (mean best-cosine)
            if s > best_s:
                best_s, best_j = s, g
        rows.append((q, best_j, best_s, derive_label(best_j) == derive_label(q)))
    r1 = sum(r[3] for r in rows) / max(len(rows), 1)
    logger.info(f"    {len(queries)} queries, R@1 = {r1:.3f}")
    return rows


def rank_strict(spots):
    """Per-query top-1 by the STRICT score (distinctiveness-weighted coverage x match support).

    Unlike soft-chamfer's mean best-cosine, this rewards matching the *distinctive* spots and
    penalizes the six manual-review failure modes (unmatched characteristic spots, round spots,
    few matches, mismatches, position, shape). Arcs still come from the raw 62-d correspondences
    the score is built on, so ``arc_col='embedding'``. See ``core/strict_match.py``."""
    import strict_match as SM                                     # noqa: PLC0415
    e_all = np.vstack(spots["embedding"].to_numpy())
    e_all = e_all / (np.linalg.norm(e_all, axis=1, keepdims=True) + 1e-12)
    w_all = SM.distinctiveness(spots, e_all)                      # dataset-wide distinctiveness

    by_img: dict[str, np.ndarray] = {}
    w_img: dict[str, np.ndarray] = {}
    sal = spots.salamander_id.to_numpy()
    for sid in spots.salamander_id.unique():
        if is_synth(sid):
            continue
        m = sal == sid
        by_img[sid] = e_all[m]
        w_img[sid] = w_all[m]
    ids = [i for i in by_img if len(by_img[i]) >= 3]
    by_label: dict[str, list[str]] = {}
    for i in ids:
        by_label.setdefault(derive_label(i), []).append(i)
    queries = [i for i in ids if len(by_label[derive_label(i)]) >= 2]
    rows = []
    for q in queries:
        best_s, best_j = -1.0, None
        for g in ids:
            if g == q:
                continue
            s = SM.strict_pair_score(by_img[q], by_img[g], w_img[q], w_img[g])
            if s > best_s:
                best_s, best_j = s, g
        rows.append((q, best_j, best_s, derive_label(best_j) == derive_label(q)))
    r1 = sum(r[3] for r in rows) / max(len(rows), 1)
    logger.info(f"    {len(queries)} queries, R@1 = {r1:.3f}  (strict)")
    return rows


def attach_arc_embedding(spots, source, col=None):
    """Add a column of PER-SPOT vectors for a learned representation so arcs (and soft-chamfer
    ranking) can use it. ``source`` in {raw, ssl_simclr, ssl_corr}; raw is a no-op (the default
    62-d ``embedding``). Any per-spot method plugs in here -- exactly the class of learned model
    whose spot-to-spot correspondences can be drawn. ``col`` names the target column so several
    representations can coexist on one frame."""
    if source == "raw":
        return spots, "embedding"
    import pickle
    col = col or f"arc_{source}"
    if col in spots.columns:
        return spots, col
    tag = {"ssl_simclr": "simclr", "ssl_corr": "corr"}[source]
    blob = pickle.loads((E.REPO_ROOT / "artifacts" / "spot_transformer" / "ssl" /
                         f"{E.dataset_name}_{tag}_64d.pkl").read_bytes())
    by_spot = blob["by_spot"]                                  # (salamander_id, spot_id) -> vec
    dim = blob["dim"]
    vecs, miss = [], 0
    for sid, spid in zip(spots.salamander_id.to_numpy(), spots.spot_id.to_numpy()):
        v = by_spot.get((sid, int(spid)))
        if v is None:
            v = np.zeros(dim, np.float32); miss += 1
        vecs.append(v)
    spots[col] = vecs
    if miss:
        logger.warning(f"  arc-source {source}: {miss} spot(s) had no SSL vector (zero-filled)")
    return spots, col


# Learned matchers, keyed by name. These RANK (produce a top-1 per query) but emit no spot-level
# pairs, so their arcs fall back to the raw 62-d correspondences the model scored. Params mirror
# sweep_all9.MODELS so the ranking matches the sweep's models.
MODEL_CONFIGS = {
    "logreg":          dict(family="feat", hidden=0),
    "mlp_deep":        dict(family="feat", hidden=64, n_layers=3, dropout=0.2, weight_decay=1e-3),
    "axial_cnn":       dict(family="set", arch="axialcnn", h=32, dropout=0.2, weight_decay=1e-3,
                            n_layers=2, kernel=3, n_bands=16),
    "set_transformer": dict(family="set", arch="transformer", h=32, dropout=0.3,
                            weight_decay=1e-2, n_layers=1, n_heads=2),
    "deepsets_cons":   dict(family="set", arch="deepsets", h=32, dropout=0.3, weight_decay=1e-2,
                            ensemble=5),
    "e2e_transformer": dict(family="e2e", arch="transformer", depth=3, dropout=0.2, n_heads=2),
    "e2e_pretrained":  dict(family="e2e", arch="frozen", dropout=0.2),
    "pretrained_cnn":  dict(family="e2e", arch="mlp", depth=1, dropout=0.2,
                            feature_attr="cnn_feat", out_dim=64, residual=False),
}
EP_SET = int(os.environ.get("VIZ_EP_SET", 120))                # lighter than the sweep's 250/30:
EP_E2E = int(os.environ.get("VIZ_EP_E2E", 20))                 # ranking, not a leaderboard result
_SETS = {}                                                     # cached ImageSet build (+cnn once)


def _get_sets(need_cnn=False):
    import data as D
    from aggregator import attach_centroids                      # noqa: PLC0415
    key = "cnn" if need_cnn else "base"
    if key not in _SETS:
        sets = _SETS.get("base")
        if sets is None:
            sets = attach_centroids(D.get_image_sets(D.get_spot_embeddings()))
            _SETS["base"] = sets
        if need_cnn:
            from pretrained_cnn import attach_cnn_features           # noqa: PLC0415
            attach_cnn_features(sets)
            _SETS["cnn"] = sets
    return _SETS[key]


def rank_learned(spots, method):
    """Per-query top-1 by a trained matcher (feat / set / e2e family). Trains on CV fold-0's
    train individuals, ranks the held-out queries image-to-image -- the same protocol as the
    sweep, so these are the model's genuine errors. Arcs are raw (the model has no spot output)."""
    import numpy as np
    cfg = MODEL_CONFIGS[method]
    fam = cfg["family"]
    import data as D
    sets = _get_sets(need_cnn=cfg.get("feature_attr") == "cnn_feat")
    keep = set(spots.salamander_id.unique())
    real = lambda i: not sets[i].is_synth and sets[i].sid in keep
    tr, ev = D.get_cv_folds(sets, k=5, seed=0)[0]
    tr = [i for i in tr if real(i)]
    ev = [i for i in ev if real(i) and len(sets[i].spots) >= 3]
    by_label: dict = {}
    for i in ev:
        by_label.setdefault(sets[i].label, []).append(i)
    queries = [i for i in ev if len(by_label[sets[i].label]) >= 2]
    logger.info(f"    training on {len(tr)} imgs, ranking {len(queries)} held-out queries")
    nrm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    if fam == "feat":
        from aggregator import build_pairs, match_features, train_aggregator, _prob
        Xtr, ytr, _, _ = build_pairs(sets, tr, neg_per_query=30, seed=0)
        model, scaler = train_aggregator(Xtr, ytr, hidden=cfg.get("hidden", 0),
                                         n_layers=cfg.get("n_layers", 1),
                                         dropout=cfg.get("dropout", 0.1),
                                         weight_decay=cfg.get("weight_decay", 1e-4), seed=0)

        def score(q, cand):
            X = np.array([match_features(nrm(sets[q].spots), nrm(sets[g].spots),
                                         np.asarray(sets[q].centroids),
                                         np.asarray(sets[g].centroids)) for g in cand])
            return _prob(model, scaler, X)

    elif fam == "e2e":
        from aggregator_e2e import build_e2e_pairs, score_e2e, train_e2e
        fattr = cfg.get("feature_attr", "spots")
        pairs, y, _, _ = build_e2e_pairs(sets, tr, neg_per_query=30, seed=0)
        kw = {k: cfg[k] for k in ("arch", "out_dim", "depth", "dropout", "n_heads", "residual")
              if k in cfg}
        model = train_e2e(sets, pairs, y, epochs=EP_E2E, seed=0, verbose=False,
                          feature_attr=fattr, **kw)

        def score(q, cand):
            return score_e2e(model, sets, [(q, [g]) for g in cand], feature_attr=fattr)

    else:                                                       # set family
        from aggregator_set import (axial_bands, build_record_pairs, match_records,
                                     score_pairs_set, train_set)
        nb = cfg.get("n_bands", 0)
        recs_tr, ytr, _, _ = build_record_pairs(sets, tr, neg_per_query=30, seed=0, n_bands=nb)
        mk = {k: cfg[k] for k in ("arch", "h", "dropout", "weight_decay", "n_layers", "n_heads",
                                  "kernel") if k in cfg}
        nseed = cfg.get("ensemble", 1)
        Ms, Ss = [], []
        for si in range(nseed):
            m, s, _ = train_set(recs_tr, ytr, epochs=EP_SET, seed=si, verbose=False, **mk)
            Ms.append(m); Ss.append(s)
        M = Ms if nseed > 1 else Ms[0]
        S = Ss if nseed > 1 else Ss[0]

        def score(q, cand):
            recs = []
            for g in cand:
                r = match_records(nrm(sets[q].spots), nrm(sets[g].spots),
                                  np.asarray(sets[q].centroids), np.asarray(sets[g].centroids))
                if nb:
                    r = axial_bands(r, sets[q].axis_t, nb)
                recs.append(r)
            z = np.zeros(len(cand))
            return score_pairs_set(M, S, recs, z, z)

    rows = []
    for qi, q in enumerate(queries):
        cand = [g for g in ev if g != q]
        sc = np.asarray(score(q, cand))
        j = int(np.argmax(sc))
        gsid = sets[cand[j]].sid
        rows.append((sets[q].sid, gsid, float(sc[j]),
                     derive_label(gsid) == derive_label(sets[q].sid)))
        if (qi + 1) % 50 == 0:
            logger.info(f"    scored {qi + 1}/{len(queries)}")
    r1 = sum(r[3] for r in rows) / max(len(rows), 1)
    logger.info(f"    R@1 = {r1:.3f}")
    return rows


def rank_for(spots, method):
    """(sorted rows desc, arc_col) for one match algorithm.

    Per-spot embeddings (raw, ssl_*) rank by soft-chamfer AND draw their own correspondences.
    The learned matchers (MODEL_CONFIGS) rank with their trained model but have no spot-level
    output, so their arcs fall back to the raw 62-d correspondences the model scored.
    """
    logger.info(f"  [{method}]")
    if method == "strict":
        rows, arc_col = rank_strict(spots), "embedding"
    elif method in MODEL_CONFIGS:
        rows, arc_col = rank_learned(spots, method), "embedding"
    else:
        spots, arc_col = attach_arc_embedding(spots, method, col=(None if method != "raw"
                                                                  else "embedding"))
        rows = rank_embedding_rows(spots, arc_col)
    rows.sort(key=lambda r: -r[2])
    return rows, arc_col


def draw_pair(subfig, spots, q, m, score, correct, tag, cutoff, method, emb_col="embedding"):
    """One cell: match arcs on top, the two individuals' size maps below."""
    gs = subfig.add_gridspec(2, 2, height_ratios=[2.0, 1.3], hspace=0.05, wspace=0.05)
    ax_match = subfig.add_subplot(gs[0, :])
    ax_q = subfig.add_subplot(gs[1, 0])
    ax_m = subfig.add_subplot(gs[1, 1])

    mark = "OK" if correct else "WRONG"
    subfig.suptitle(f"{tag}  ·  {q}  ->  {m}   score={score:.2f}   [{mark}]",
                    fontsize=12, fontweight="bold",
                    color=("tab:green" if correct else "tab:red"))
    try:
        E.visualize_spot_matches(spots, q, m, cutoff=cutoff, method=method, ax=ax_match,
                                 emb_col=emb_col)
    except Exception as exc:                                   # never let one pair kill the grid
        ax_match.text(0.5, 0.5, f"match failed:\n{exc}", ha="center", va="center",
                      fontsize=8, transform=ax_match.transAxes); ax_match.axis("off")
    for ax, sid, side in ((ax_q, q, "query"), (ax_m, m, "matched")):
        try:
            E.visualize_spot_sizes(spots, sid, ax=ax, colorbar=False)
            ax.set_title(f"{side}: {sid}", fontsize=9)
        except Exception as exc:
            ax.text(0.5, 0.5, f"{sid}\n{exc}", ha="center", va="center", fontsize=8,
                    transform=ax.transAxes); ax.axis("off")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5, help="matches per slice (top-N, middle-N, bottom-N)")
    p.add_argument("--cutoff", type=float, default=0.5, help="min cosine for a drawn match arc")
    p.add_argument("--method", default="mutual", choices=("mutual", "hungarian", "threshold"))
    p.add_argument("--methods", default="raw,ssl_simclr,ssl_corr,logreg",
                   help="comma list of match algorithms to include, each a block of top/mid/bottom")
    p.add_argument("--out", default=None, help="output .svg (default artifacts/match_grid_<ds>.svg)")
    args = p.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    logger.info(f"dataset: {E.dataset_name}   methods: {methods}   n/slice: {args.n}")
    spots = load_spots()

    # rank every method, then cut each into three score bands: the most confident matches, the
    # decision-boundary middle (where correct and wrong mix), and the least confident. Showing all
    # three per algorithm is what makes the failure modes comparable across methods.
    def slices(rows, n):
        n = min(n, len(rows))
        mid = (len(rows) - n) // 2
        return [("TOP", rows[:n]), ("MID", rows[mid:mid + n]), ("BOT", rows[-n:] if n else [])]

    blocks = []                                                # (method, arc_col, [(band, rows)])
    for method in methods:
        try:
            rows, arc_col = rank_for(spots, method)
        except Exception as exc:
            logger.error(f"  [{method}] SKIPPED: {type(exc).__name__}: {exc}")
            continue
        if rows:
            blocks.append((method, arc_col, slices(rows, args.n)))
    if not blocks:
        logger.warning("nothing to plot"); return 1

    n = args.n
    n_rows = len(blocks) * 3                                    # 3 score bands per method
    fig = plt.figure(figsize=(5.6 * n, 5.0 * n_rows + 1.5), constrained_layout=True)
    fig.suptitle(f"Spot matches — top / middle / bottom by score, per algorithm\n"
                 f"{E.dataset_name}   ·   each cell: match arcs + both individuals' spot sizes",
                 fontsize=17, fontweight="bold")

    # one subfigure-row per (method, band); a bold band header on the leftmost cell
    rowfigs = fig.subfigures(n_rows, 1, hspace=0.02)
    rowfigs = np.atleast_1d(rowfigs)
    ri = 0
    for method, arc_col, bands in blocks:
        for band, rows in bands:
            rowfig = rowfigs[ri]; ri += 1
            rowfig.suptitle(f"{method}  —  {band} {len(rows)}", fontsize=14, fontweight="bold",
                            x=0.01, ha="left",
                            color={"TOP": "tab:green", "MID": "tab:orange",
                                   "BOT": "tab:red"}[band])
            cells = np.atleast_1d(rowfig.subfigures(1, max(n, 1), wspace=0.02))
            for ci in range(n):
                if ci < len(rows):
                    q, m, s, c = rows[ci]
                    draw_pair(cells[ci], spots, q, m, s, c, f"{band}{ci + 1}", args.cutoff,
                              args.method, arc_col)
                else:
                    cells[ci].add_subplot(111).axis("off")

    out = Path(args.out) if args.out else (E.REPO_ROOT / "artifacts" /
                                           f"match_grid_{E.dataset_name}.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    sz = out.stat().st_size / 1e6
    logger.info(f"wrote {out}  ({sz:.1f} MB, {len(blocks)} methods x 3 bands x {n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
