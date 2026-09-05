from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

try:  # run as script (path[0] = this dir) OR imported as a package
    import data as d
    from transformer import SpotSetTransformer, SpotMLP
    from losses import supcon_loss
    from eval_spot import (encode_spots_raw, encode_spots_model,
                           spot_vote_retrieval, spot_match_precision, spot_match_auc)
except ModuleNotFoundError:
    from pipeline.spot_transformer import data as d
    from pipeline.spot_transformer.transformer import SpotSetTransformer, SpotMLP
    from pipeline.spot_transformer.losses import supcon_loss
    from pipeline.spot_transformer.eval_spot import (encode_spots_raw, encode_spots_model,
                                                     spot_vote_retrieval, spot_match_precision,
                                                     spot_match_auc)


def train_one_fold_spot(
    sets, train_idx, eval_idx, *,
    P: int = 16, K: int = 4, num_batches: int = 50, epochs: int = 25,
    model_type: str = "transformer",   # "transformer" (contextual) | "mlp" (context-free control)
    d_model: int = 128, n_layers: int = 2, n_heads: int = 4, out_dim: int = 128,
    model_dropout: float = 0.1, dropout_p: float = 0.2, jitter_std: float = 0.0,
    lr: float = 1e-3, weight_decay: float = 1e-4, temperature: float = 0.1,
    patience: int = 8, seed: int = 0, device: str | None = None,
    ckpt_path: str | None = None, verbose: bool = True,
):
    """Spot-level track: train ``SpotSetTransformer.forward_spots`` with spot-level SupCon
    (same-image pairs excluded), evaluated by spot-VOTING identification.

    The metric that matters (and that early-stopping watches) is held-out **voting recall@1**;
    the BEFORE bar is raw-spot voting (no training). Returns ``(best_model, report)``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    log = print if verbose else (lambda *a, **k: None)

    # gallery-matched train diagnostic (same #individuals as eval; real images, >=2 each)
    real_train = [i for i in train_idx if not sets[i].is_synth]
    real_by_label: dict[str, list[int]] = {}
    for i in real_train:
        real_by_label.setdefault(sets[i].label, []).append(i)
    multi = [lbl for lbl, ii in real_by_label.items() if len(ii) >= 2]
    n_eval_ind = len({sets[i].label for i in eval_idx})
    chosen = np.random.default_rng(seed).choice(multi, size=min(n_eval_ind, len(multi)), replace=False)
    train_diag = [i for lbl in chosen for i in real_by_label[lbl]]

    train_ds = d.SpotSetDataset(sets, train_idx, train=True, dropout_p=dropout_p,
                                jitter_std=jitter_std, seed=seed)
    sampler = d.PKSampler(train_ds, P=P, K=K, num_batches=num_batches, seed=seed)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=d.collate_sets)

    in_dim = sets[0].spots.shape[1]
    if model_type == "mlp":
        model = SpotMLP(in_dim=in_dim, hidden=d_model, out_dim=out_dim, dropout=model_dropout).to(device)
    else:
        model = SpotSetTransformer(in_dim=in_dim, d_model=d_model, n_heads=n_heads,
                                   n_layers=n_layers, out_dim=out_dim, dropout=model_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters())

    # BEFORE: raw-spot voting on the held-out fold (the bar to beat), and the untrained model
    Zr, lr_, ir, _ = encode_spots_raw(sets, eval_idx)
    base = spot_vote_retrieval(Zr, lr_, ir, ks=(1, 5, 10))
    base_auc = spot_match_auc(Zr, lr_, ir)
    match_before = spot_match_precision(Zr, lr_, ir)
    Zp, lp, ip, _ = encode_spots_model(model, sets, eval_idx, device=device)
    pretrain = spot_vote_retrieval(Zp, lp, ip, ks=(1, 5, 10))

    log("=" * 82)
    log(" SpotSetTransformer - SPOT-LEVEL SupCon (voting identification)")
    log("=" * 82)
    log(f" device       : {device}   model: {model_type}   params: {n_params:,}")
    log(f" train images : {len(train_idx)}  ({len(real_train)} real)   "
        f"individuals: {len({sets[i].label for i in train_idx})}")
    log(f" eval  images : {len(eval_idx)}   individuals: {n_eval_ind}   ({len(Zr)} eval spots)")
    log(f" batch        : P={P} x K={K} images  x {num_batches}/epoch   temp={temperature}   "
        f"aug drop={dropout_p} jit={jitter_std}")
    log(f" BEFORE  vote : R@1={pretrain['recall@1']:.3f}  (untrained model)")
    log(f" BASELINE vote: R@1={base['recall@1']:.3f}  R@5={base['recall@5']:.3f}  "
        f"R@10={base['recall@10']:.3f}   <-- raw-spot voting, beat this")
    log("=" * 82)
    log(" ep | loss  |   lr    | TRAIN R@1 | EVAL R@1  R@5  R@10 | match-AUC | gap  |")
    log("-" * 82)

    best_score, best_epoch, best_state, no_improve = -1.0, 0, None, 0
    best_eval, best_after_auc, match_after = pretrain, base_auc, match_before
    beat_baseline_at = None
    history = []

    for epoch in range(1, epochs + 1):
        lr_now = sched.get_last_lr()[0]
        model.train()
        losses = []
        for X, mask, y in loader:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            spot_emb = model.forward_spots(X, mask)                    # (B, N, out)
            Z, sl, grp = d.flatten_spots(spot_emb, mask, y)            # (M,out),(M,),(M,)
            loss = supcon_loss(Z, sl, temperature=temperature, groups=grp)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        train_loss = float(np.mean(losses))

        Zt, lt, it, _ = encode_spots_model(model, sets, train_diag, device=device)
        tm = spot_vote_retrieval(Zt, lt, it, ks=(1,))
        Ze, le, ie, _ = encode_spots_model(model, sets, eval_idx, device=device)
        em = spot_vote_retrieval(Ze, le, ie, ks=(1, 5, 10))
        em_auc = spot_match_auc(Ze, le, ie)

        score = em["recall@1"]
        is_best = score > best_score
        if is_best:
            best_score, best_epoch, best_eval, best_after_auc = score, epoch, em, em_auc
            best_state = copy.deepcopy(model.state_dict())
            match_after = spot_match_precision(Ze, le, ie)
            no_improve = 0
        else:
            no_improve += 1
        if beat_baseline_at is None and score > base["recall@1"]:
            beat_baseline_at = epoch

        gap = tm["recall@1"] - em["recall@1"]
        note = " *best" if is_best else ""
        if beat_baseline_at == epoch:
            note += "  <<< beats raw-spot baseline"
        log(f" {epoch:2d} | {train_loss:5.3f} | {lr_now:.1e} |   {tm['recall@1']:.3f}   |"
            f"  {em['recall@1']:.3f} {em['recall@5']:.3f} {em['recall@10']:.3f} |"
            f"   {em_auc:.3f}   | {gap:+.2f} |{note}", flush=True)
        history.append(dict(epoch=epoch, loss=train_loss, lr=lr_now, train=tm, eval=em, auc=em_auc, gap=gap))

        if no_improve >= patience:
            log(f"-- early stop: no eval R@1 improvement for {patience} epochs --")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    log("-" * 82)
    log(f" best epoch   : {best_epoch}   eval vote R@1={best_score:.3f}  "
        f"(raw-spot {base['recall@1']:.3f}, {best_score-base['recall@1']:+.3f})")
    log(f" beat baseline: {'epoch '+str(beat_baseline_at) if beat_baseline_at else 'NEVER'}")
    log("=" * 82)

    if ckpt_path:
        torch.save(best_state, ckpt_path)
        log(f" saved best checkpoint -> {ckpt_path}")

    report = dict(
        track="spot",
        config=dict(P=P, K=K, num_batches=num_batches, epochs=epochs, model_type=model_type,
                    d_model=d_model, n_layers=n_layers, n_heads=n_heads, out_dim=out_dim, model_dropout=model_dropout,
                    dropout_p=dropout_p, jitter_std=jitter_std, lr=lr, weight_decay=weight_decay,
                    temperature=temperature, patience=patience, seed=seed),
        baseline=base, baseline_auc=base_auc, pretrain=pretrain,
        best=best_eval, best_epoch=best_epoch, best_auc=best_after_auc,
        beat_baseline_at=beat_baseline_at, history=history,
        match_before=match_before, match_after=match_after,
    )
    return model, report


if __name__ == "__main__":
    sets = d.get_image_sets(d.get_spot_embeddings())
    train_idx, eval_idx = d.get_cv_folds(sets, k=5, seed=0)[0]
    model, report = train_one_fold_spot(
        sets, train_idx, eval_idx,
        P=16, K=4, num_batches=50, epochs=25,
        dropout_p=0.2, jitter_std=0.0, lr=1e-3, temperature=0.1, patience=8, seed=0,
    )
    breakpoint()   # live: model, report
