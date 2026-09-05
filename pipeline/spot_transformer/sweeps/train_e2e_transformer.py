"""Train ONE e2e_transformer (strict, special-spot-aware) with per-epoch train/test logging.

Input = the fixed 62-dim per-spot embeddings (``ImageSet.spots``); a transformer re-encodes each
spot in context; the only loss is the matching label. The vote prefers special spots, penalizes
unmatched special spots / too-few matches / far-apart matches, and defaults to no-match at the
census threshold (see ``aggregator_e2e_strict``).

Every epoch prints train & test **loss** and train & test **evals** (census F0.5, ident R@1,
open-set AUROC), plus the gate AUROC (does the learned per-spot gate recover the human
interesting-spot labels). Train evals come from an open-set split carved out of the TRAIN
individuals (subsampled for speed); test evals from the held-out fold.

    pixi run python pipeline/spot_transformer/sweeps/train_e2e_transformer.py
    QUICK=1 ... | EPOCHS=60 E2E_NEG=30 FOLD=2 GATE_LAMBDA=0.5 SIGMA_POS=0.12 MIN_QUALITY=0.4 ...
    ARCH=frozen ...   # ablation: gate+strict vote on the fixed embedding, no re-encoding
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval", "sweeps"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

import data as d                                            # noqa: E402
import census as cen                                        # noqa: E402
import compare_strict as cs                                 # noqa: E402  (_census_row helper)
import aggregator_e2e_strict as e2s                         # noqa: E402
from aggregator import attach_centroids                     # noqa: E402
from aggregator_e2e import build_e2e_pairs, build_e2e_openset_pairs  # noqa: E402

QUICK = bool(os.environ.get("QUICK"))
EPOCHS = 4 if QUICK else int(os.environ.get("EPOCHS", "40"))
E2E_NEG = 8 if QUICK else int(os.environ.get("E2E_NEG", "25"))
FOLD = int(os.environ.get("FOLD", "0"))
ARCH = os.environ.get("ARCH", "transformer")
GATE_LAMBDA = float(os.environ.get("GATE_LAMBDA", "0.5"))
SIGMA_POS = float(os.environ.get("SIGMA_POS", "0.12"))
LR = float(os.environ.get("LR", "1e-3"))
COSINE = bool(os.environ.get("COSINE"))                      # decay LR to ~0 over EPOCHS (long runs)
SEED = 0
NOVEL_FRAC = 0.35
TRAIN_EVAL_INDIV = 30                                        # cap train-eval size for per-epoch speed


def build_pos_lookup_light(db_path=None):
    """``(sid, spot_id) -> (axis_t, axis_offset/length_px)`` straight from the DB (no factor calc)."""
    import duckdb
    con = duckdb.connect(str(db_path or d.DB_PATH), read_only=True)
    try:
        length = {r[0]: r[1] for r in
                  con.execute("SELECT salamander_id, length_px FROM body_axis").fetchall()}
        rows = con.execute("SELECT salamander_id, spot_id, axis_t, axis_offset FROM spots").fetchall()
    finally:
        con.close()
    out = {}
    for sid, spid, t, off in rows:
        L = length.get(sid)
        lat = (off / L) if (L and off is not None) else np.nan
        out[(sid, int(spid))] = (float(t) if t is not None else np.nan, float(lat))
    return out


def subsample_by_individual(sets, images, n_indiv, seed=0):
    by = {}
    for i in images:
        by.setdefault(sets[i].label, []).append(i)
    labs = list(by); rng = np.random.default_rng(seed)
    keep = rng.permutation(len(labs))[:n_indiv]
    return [i for k in keep for i in by[labs[k]]]


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    logger.info("loading pre-embedded spots + interesting labels + body positions ...")
    sets = attach_centroids(d.get_image_sets(d.get_spot_embeddings()))
    e2s.attach_interesting(sets)
    e2s.attach_positions(sets, build_pos_lookup_light())

    eval_mask = d.quality_keep_mask(sets)
    folds = d.get_cv_folds(sets, k=5, seed=SEED, eval_mask=eval_mask)
    tr, ev = folds[FOLD]
    train_imgs = [i for i in tr if not sets[i].is_synth]

    # matching pairs for the loss curves
    pairs_tr, y_tr, _, _ = build_e2e_pairs(sets, train_imgs, neg_per_query=E2E_NEG, seed=SEED)
    pairs_te, y_te, _, _ = build_e2e_pairs(sets, ev, neg_per_query=E2E_NEG, seed=SEED)
    npos = max(float(y_tr.sum()), 1.0); PW = (len(y_tr) - npos) / npos     # shared loss scale

    # open-set eval splits (train subsampled for per-epoch speed)
    tr_eval_imgs = subsample_by_individual(sets, train_imgs, TRAIN_EVAL_INDIV, seed=SEED)
    g_tr, q_tr = cen.make_openset_split(sets, tr_eval_imgs, NOVEL_FRAC, SEED)
    g_te, q_te = cen.make_openset_split(sets, ev, NOVEL_FRAC, SEED)
    os_tr = build_e2e_openset_pairs(sets, g_tr, q_tr); true_tr = {q: sets[q].label for q in q_tr}
    os_te = build_e2e_openset_pairs(sets, g_te, q_te); true_te = {q: sets[q].label for q in q_te}

    logger.info(f" fold {FOLD}  arch={ARCH}  train_imgs={len(train_imgs)}  test_imgs={len(ev)}  "
          f"train_pairs={len(pairs_tr)}  test_pairs={len(pairs_te)}  "
          f"quality={d.quality_tag() or 'none'}")
    logger.info(f" rules: prefer special (gate, λ={GATE_LAMBDA}) · position gate σ={SIGMA_POS} · "
          f"support (few-match penalty) · unexplained-special penalty · default-no-match@threshold")
    logger.info(f" epochs={EPOCHS}  lr={LR}  cosine_schedule={COSINE}")
    logger.info("=" * 116)
    logger.info(f" {'ep':>3} | {'loss tr/te':>12} | {'F0.5 te':>7} | {'R@1':>6}{'R@5':>6}{'R@10':>6} te"
          f" | {'AUROC te':>8} | {'cov@P90':>7} | {'gate':>5}")
    logger.info(" (cov@P90 = %queries answered while staying >=90% precise; the rest it abstains on;"
          " '*' = new best, checkpoint saved)")
    logger.info("-" * 116)

    # best-checkpoint bookkeeping: keep the epoch with the best (census F0.5, then AUROC) on test,
    # so a long run yields a reusable model, not just a log. F0.5 is the deployment headline; AUROC
    # breaks the (frequent) F0.5 ties toward the more-converged epoch.
    outdir = d.REPO_ROOT / "artifacts" / "spot_transformer" / "sweeps" / "strict"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / f"e2e_ckpt_fold{FOLD}_{ARCH}{d.quality_tag()}.pt"
    model_config = dict(in_dim=int(sets[train_imgs[0]].spots.shape[1]), out_dim=None, arch=ARCH,
                        depth=3, dropout=0.1, n_heads=2, tau=0.05, hidden=32, residual=True,
                        sigma_pos=SIGMA_POS)
    best_state = {"key": (-1.0, -1.0), "ep": -1}

    log = []

    def on_epoch(ep, model, train_loss):
        te_loss = e2s.pair_loss(model, sets, pairs_te, y_te, pos_weight=PW, gate_lambda=0.0)
        sc_tr = e2s.score_strict_e2e(model, sets, os_tr[0])
        sc_te = e2s.score_strict_e2e(model, sets, os_te[0])
        m_tr = cs._census_row(sc_tr, os_tr[1], os_tr[2], true_tr)
        m_te = cs._census_row(sc_te, os_te[1], os_te[2], true_te)
        ga = e2s.gate_auroc(model, sets, q_te)
        row = dict(ep=ep, train_loss=train_loss, test_loss=te_loss,
                   f_tr=m_tr["census_f"], f_te=m_te["census_f"],
                   r1_tr=m_tr["ident_r1"], r1_te=m_te["ident_r1"],
                   r5_te=m_te["r5"], r10_te=m_te["r10"],
                   auc_tr=m_tr["auroc_top1"], auc_te=m_te["auroc_top1"],
                   cov_te=m_te["cov_at_p"], gate=ga)
        log.append(row)
        key = (m_te["census_f"], m_te["auroc_top1"])
        saved = key > best_state["key"]
        if saved:
            best_state["key"] = key; best_state["ep"] = ep
            e2s.save_checkpoint(ckpt_path, model, model_config, epoch=ep, metrics=row,
                                dataset=d.dataset_name, fold=FOLD,
                                quality=d.quality_tag() or "none")
        logger.info(f" {ep:>3} | {train_loss:5.3f}/{te_loss:5.3f} | {m_te['census_f']:7.3f} | "
              f"{m_te['ident_r1']:6.3f}{m_te['r5']:6.3f}{m_te['r10']:6.3f}    | "
              f"{m_te['auroc_top1']:8.3f} | {m_te['cov_at_p']:7.3f} | {ga:5.3f}"
              f"{'  *' if saved else ''}")

    t0 = time.time()
    e2s.train_strict_e2e(sets, pairs_tr, y_tr, arch=ARCH, epochs=EPOCHS, lr=LR,
                         gate_lambda=GATE_LAMBDA, sigma_pos=SIGMA_POS, pos_weight=None,
                         cosine=COSINE, seed=SEED, on_epoch=on_epoch)
    logger.info("-" * 108)
    best = max(log, key=lambda r: (r["f_te"], r["auc_te"]))
    logger.info(f" best test F0.5 = {best['f_te']:.3f} (AUROC {best['auc_te']:.3f}) at epoch {best['ep']}"
          f"  ·  total {time.time()-t0:.0f}s")
    logger.info(f" best checkpoint -> {ckpt_path}")
    logger.info(f"   load with: aggregator_e2e_strict.load_strict_e2e(r'{ckpt_path}')")

    # write per-epoch curve
    md = [f"# e2e_transformer (strict, special-spot-aware) — fold {FOLD}, arch {ARCH}", "",
          f"- dataset `{d.dataset_name}` · {EPOCHS} epochs · neg {E2E_NEG} · gate λ {GATE_LAMBDA}"
          f" · σ_pos {SIGMA_POS} · quality `{d.quality_tag() or 'none'}`"
          + ("  **[QUICK]**" if QUICK else ""),
          f"- best test F0.5 **{best['f_te']:.3f}** @ epoch {best['ep']}",
          "- **cov@P90** = abstain operating point: fraction of test photos answered while emitted",
          "  matches stay >=90% precise (the rest abstained). R@k = correct animal in top-k shortlist.",
          "",
          "| ep | train loss | test loss | F0.5 te | R@1 te | R@5 te | R@10 te | AUROC te | cov@P90 te | gate |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in log:
        md.append(f"| {r['ep']} | {r['train_loss']:.3f} | {r['test_loss']:.3f} | {r['f_te']:.3f} "
                  f"| {r['r1_te']:.3f} | {r['r5_te']:.3f} | {r['r10_te']:.3f} | {r['auc_te']:.3f} "
                  f"| {r['cov_te']:.3f} | {r['gate']:.3f} |")
    fname = f"RESULTS_e2e_train_fold{FOLD}_{ARCH}{d.quality_tag()}" + ("_quick" if QUICK else "") + ".md"
    (outdir / fname).write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info(f"wrote {outdir / fname}")


if __name__ == "__main__":
    main()
