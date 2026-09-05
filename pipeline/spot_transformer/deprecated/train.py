from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
    from transformer import SpotSetTransformer
    from losses import supcon_loss
    from eval import (encode_model, encode_baseline, retrieval_metrics,
                      match_precision_at_thresholds, print_match_table, match_auc)
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d
    from pipeline.spot_transformer.transformer import SpotSetTransformer
    from pipeline.spot_transformer.losses import supcon_loss
    from pipeline.spot_transformer.eval import (encode_model, encode_baseline, retrieval_metrics,
                                                match_precision_at_thresholds, print_match_table, match_auc)


def train_one_fold(
    sets,
    train_idx,
    eval_idx,
    *,
    # batch / schedule
    P: int = 16,
    K: int = 4,
    num_batches: int = 50,
    epochs: int = 25,
    # model
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    out_dim: int = 128,
    model_dropout: float = 0.1,
    # data aug
    dropout_p: float = 0.2,
    jitter_std: float = 0.0,
    # optim
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    # bookkeeping
    patience: int = 8,
    seed: int = 0,
    device: str | None = None,
    ckpt_path: str | None = None,
    verbose: bool = True,
):
    """Train the SpotSetTransformer on one CV fold with SupCon, logging train vs eval
    retrieval every epoch.

    Returns ``(best_model, report)`` where ``report`` is a dict::

        {config, baseline, pretrain, best, best_epoch, beat_baseline_at, history}

    ``baseline`` = mean-pool (no training), ``pretrain`` = the untrained model *before*
    the run, ``best`` = the best-epoch eval metrics *after*. ``history`` has one dict per
    epoch with ``train``/``eval`` metric sub-dicts for the learning curves.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    log = print if verbose else (lambda *a, **k: None)

    # real-only views for the diagnostics (eval_idx is already real; train has synthetics)
    real_train = [i for i in train_idx if not sets[i].is_synth]
    n_real = len(real_train)
    n_synth = len(train_idx) - n_real

    # Gallery-matched TRAIN diagnostic: retrieval R@1 depends on how many distractor
    # identities are in the gallery, so to compare in-sample vs held-out fairly we score
    # train on a subset the same SIZE as the eval fold (same #individuals, real images,
    # >=2 each). Without this, train R@1 looks worse than eval only because its gallery
    # (272 individuals) is denser than eval's (18) -- not because of over/under-fitting.
    real_by_label: dict[str, list[int]] = {}
    for i in real_train:
        real_by_label.setdefault(sets[i].label, []).append(i)
    multi = [lbl for lbl, ii in real_by_label.items() if len(ii) >= 2]
    n_eval_ind = len({sets[i].label for i in eval_idx})
    chosen = np.random.default_rng(seed).choice(multi, size=min(n_eval_ind, len(multi)), replace=False)
    train_diag = [i for lbl in chosen for i in real_by_label[lbl]]

    # data
    train_ds = d.SpotSetDataset(sets, train_idx, train=True, dropout_p=dropout_p,
                                jitter_std=jitter_std, seed=seed)
    sampler = d.PKSampler(train_ds, P=P, K=K, num_batches=num_batches, seed=seed)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=d.collate_sets)

    # model / optim
    model = SpotSetTransformer(in_dim=sets[0].spots.shape[1], d_model=d_model, n_heads=n_heads,
                               n_layers=n_layers, out_dim=out_dim, dropout=model_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters())

    # the bar to beat: zero-training mean-pool baseline on the held-out eval fold
    Zb, yb = encode_baseline(sets, eval_idx)
    base = retrieval_metrics(Zb, yb, ks=(1, 5, 10))
    # BEFORE: the untrained (random-init) model on the same eval fold
    Zp, yp = encode_model(model, sets, eval_idx, device=device)
    pretrain = retrieval_metrics(Zp, yp, ks=(1, 5, 10))

    log("=" * 78)
    log(f" SpotSetTransformer - SupCon training")
    log("=" * 78)
    log(f" device       : {device}")
    log(f" train images : {len(train_idx)}  (real {n_real} / synth {n_synth})   "
        f"individuals: {len({sets[i].label for i in train_idx})}")
    log(f" eval  images : {len(eval_idx)}   individuals: {len({sets[i].label for i in eval_idx})}")
    log(f" train diag   : {len(train_diag)} imgs / {len(chosen)} individuals (gallery-matched to eval)")
    log(f" batch        : P={P} x K={K} = {P*K}  x {num_batches} batches/epoch")
    log(f" optim        : AdamW lr={lr:g} wd={weight_decay:g}  cosine->0   temp={temperature}")
    log(f" aug          : spot_dropout={dropout_p}  jitter={jitter_std}   model_dropout={model_dropout}")
    log(f" model params : {n_params:,}")
    log(f" BEFORE  eval : R@1={pretrain['recall@1']:.3f}  R@5={pretrain['recall@5']:.3f}  "
        f"mAP={pretrain['mAP']:.3f}   (untrained model)")
    log(f" BASELINE eval: R@1={base['recall@1']:.3f}  R@5={base['recall@5']:.3f}  "
        f"R@10={base['recall@10']:.3f}  mAP={base['mAP']:.3f}   <-- beat this")
    log("=" * 78)
    log(" ep | loss  |   lr    | TRAIN R@1 mAP | EVAL R@1  R@5  R@10 mAP  med | gap  |")
    log("-" * 78)

    best_score, best_epoch, best_state, no_improve = -1.0, 0, None, 0
    best_eval = pretrain
    beat_baseline_at = None
    history = []

    for epoch in range(1, epochs + 1):
        lr_now = sched.get_last_lr()[0]
        model.train()
        t0 = time.time()
        losses = []
        for X, mask, y in loader:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            z = model(X, mask)
            loss = supcon_loss(z, y, temperature=temperature)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        train_loss = float(np.mean(losses))

        # diagnostics: in-sample (train, gallery-matched to eval) vs held-out (eval) retrieval
        Zt, yt = encode_model(model, sets, train_diag, device=device)
        tm = retrieval_metrics(Zt, yt, ks=(1,))
        Ze, ye = encode_model(model, sets, eval_idx, device=device)
        em = retrieval_metrics(Ze, ye, ks=(1, 5, 10))

        score = em["recall@1"]
        is_best = score > best_score
        if is_best:
            best_score, best_epoch, best_eval = score, epoch, em
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if beat_baseline_at is None and score > base["recall@1"]:
            beat_baseline_at = epoch

        gap = tm["recall@1"] - em["recall@1"]                    # overfitting gap
        note = ""
        if is_best:
            note += " *best"
        if beat_baseline_at == epoch:
            note += "  <<< beats baseline"
        log(f" {epoch:2d} | {train_loss:5.3f} | {lr_now:.1e} |"
            f"   {tm['recall@1']:.3f} {tm['mAP']:.3f} |"
            f"  {em['recall@1']:.3f} {em['recall@5']:.3f} {em['recall@10']:.3f} "
            f"{em['mAP']:.3f} {em['median_rank']:>3.0f} |"
            f" {gap:+.2f} |{note}", flush=True)

        history.append(dict(epoch=epoch, loss=train_loss, lr=lr_now,
                            train=tm, eval=em, gap=gap))

        if no_improve >= patience:
            log(f"-- early stop: no eval R@1 improvement for {patience} epochs --")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # match-precision-at-threshold: image-level all-to-all matching on the held-out eval fold,
    # BEFORE (mean-pool) vs AFTER (trained). Mirrors the spot-level naive_match_all analysis.
    Zev, yev = encode_model(model, sets, eval_idx, device=device)
    match_after = match_precision_at_thresholds(Zev, yev)
    match_before = match_precision_at_thresholds(Zb, yb)   # Zb,yb = mean-pool eval (computed above)
    match_auc_after = match_auc(Zev, yev)                  # threshold-free, cross-space comparable
    match_auc_before = match_auc(Zb, yb)
    if verbose:
        print_match_table(match_before, "\n match BEFORE (mean-pool, eval fold):")
        print_match_table(match_after, " match AFTER  (trained, eval fold):")

    log("-" * 78)
    log(f" best epoch   : {best_epoch}   eval R@1={best_score:.3f}  "
        f"(baseline {base['recall@1']:.3f}, {'+' if best_score>=base['recall@1'] else ''}"
        f"{best_score-base['recall@1']:+.3f})")
    if beat_baseline_at:
        log(f" beat baseline: first at epoch {beat_baseline_at}")
    else:
        log(f" beat baseline: NEVER -- model did not surpass mean-pool")
    log("=" * 78)

    if ckpt_path:
        torch.save(best_state, ckpt_path)
        log(f" saved best checkpoint -> {ckpt_path}")

    report = dict(
        config=dict(P=P, K=K, num_batches=num_batches, epochs=epochs, d_model=d_model,
                    n_layers=n_layers, n_heads=n_heads, out_dim=out_dim,
                    model_dropout=model_dropout, dropout_p=dropout_p, jitter_std=jitter_std,
                    lr=lr, weight_decay=weight_decay, temperature=temperature,
                    patience=patience, seed=seed),
        baseline=base, pretrain=pretrain, best=best_eval, best_epoch=best_epoch,
        beat_baseline_at=beat_baseline_at, history=history,
        match_before=match_before, match_after=match_after,
        match_auc_before=match_auc_before, match_auc_after=match_auc_after,
    )
    return model, report


if __name__ == "__main__":
    sets = d.get_image_sets(d.get_spot_embeddings())
    train_idx, eval_idx = d.get_cv_folds(sets, k=5, seed=0)[0]

    model, report = train_one_fold(
        sets, train_idx, eval_idx,
        P=16, K=4, num_batches=50, epochs=25,
        dropout_p=0.2, jitter_std=0.0,
        lr=1e-3, temperature=0.1, patience=8, seed=0,
    )

    # to persist the final fingerprints, encode ALL images with the best model and write them:
    #   from eval import encode_model
    #   Z, labels = encode_model(model, sets, list(range(len(sets))))
    #   d.write_image_embeddings([s.sid for s in sets], Z, table="image_embeddings",
    #                            labels=[s.label for s in sets], is_synth=[s.is_synth for s in sets])

    breakpoint()   # live: model, report
