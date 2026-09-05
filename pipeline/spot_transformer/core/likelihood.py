"""Partial-match scoring: weigh each spot by how much it is actually worth as evidence.

At 24.7% spot survival, "most of the pattern is missing" is the NORMAL case for a true pair, so a
score built around explaining the whole pattern is measuring the wrong thing. The existing rules
handle absence in two ways, and both are miscalibrated in the same direction:

* ``strict_pair_score`` divides explained mass by the **total** distinctiveness of both animals, so
  a perfect true match is structurally capped near the survival rate and most of the denominator is
  pattern nobody expected to survive;
* ``combine_residual`` goes further and charges unmatched pattern as *contradiction* (at λ=1, a unit
  of missing pattern costs as much as a unit of shared pattern) plus a single-spot ``veto``. At a
  76% miss rate, that fires on true pairs constantly.

The calibrated alternative asks what a match and a miss are each **worth**, measured rather than
assumed. Over real photo pairs of this dataset:

    P(a spot finds a match | SAME animal)      = 0.247
    P(a spot finds a match | DIFFERENT animal) = 0.159

so, as log-likelihood ratios,

    a match  ->  log(0.247/0.159) = **+0.44**   (1.6x evidence FOR)
    a miss   ->  log(0.753/0.841) = **-0.11**   (0.9x -- nearly nothing)

**One match is worth about four misses.** That single ratio is what "partial matching" means
concretely: a true pair sharing only a handful of spots is not penalised for the rest, because the
rest going missing is exactly what a true pair looks like.

Two things make this better than hard-coding 4:1.

1. **The ratio is conditional.** A big spot on a clearly-visible flank survives far more often than
   a small one (Part A: size +0.52, observability +0.26), so its absence is genuinely surprising
   while a small spot's is not. Both probabilities are therefore fitted as functions of the spot's
   own properties, and the evidence each spot contributes follows.
2. **The score is calibrated log-odds**, not an arbitrary [0, 1]. That matters for the deliverable —
   an abstention threshold on log-odds means something, and can be set from a target precision
   rather than tuned.

Fitted on TRAINING individuals only; a spot's animal is never seen when its own weight is used.

--------------------------------------------------------------------------------------------------
**MEASURED OUTCOME (2026-08-16): this loses to counting matched spots, and the module is kept as the
record of that.** On 120 queries each ranked against its true partner plus 20 random others:

    raw matched-spot count .......................... AUROC 0.674   <- the thing to beat
    llr, query-intrinsic conditioning (size) ........ AUROC 0.569
    llr + observability ............................. AUROC 0.562
    llr + observability + candidate spot count ...... AUROC 0.524   (as first built)

Two separate lessons, both worth more than the scorer would have been.

1. **Only query-intrinsic properties may condition a ranking score.** The first build conditioned on
   the candidate's spot count and on observability (derived from the candidate's axial span). The
   result correlated **-0.40 with the candidate's spot count** and only **+0.09 with the number of
   matched spots**: it was ranking on how many spots the other animal happened to have. Anything
   candidate-dependent varies across the very comparison being ranked.

2. **The arithmetic explains why even the clean version loses.** With constant probabilities the
   score reduces to ``n_matched * (A - B) + nq * B`` — for a fixed query, exactly a monotone
   function of the match count, i.e. identical to the baseline. Every deviation from the baseline is
   therefore the *conditioning*, and here the conditioning costs more in variance than it earns in
   signal. That is results.md #24 ("simplest wins; capacity is not the constraint") arriving again.

**What survives, and it is the actionable part.** The measured base rates are real and they say a
miss is worth about a quarter of a match. That does not need a new scorer — it needs the existing
one's penalty re-scaled: ``strict_match.combine_residual`` charges contradiction at ``lam=1.0``,
i.e. a unit of missing pattern costing as much as a unit of shared pattern, when the data says
~0.25. Sweeping ``lam`` is a one-parameter change to a rule already in use, and it is where
"partial-match scoring" should actually be spent.
--------------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

MATCH_THR = 0.4

# ONLY query-spot-intrinsic properties may condition the ratio. This is the hard-won constraint of
# this module, and violating it destroyed the first version.
#
# For ranking, a query photo is scored against many candidates. Anything in the conditioning that
# depends on the CANDIDATE therefore varies across the very comparison being ranked, and injects an
# ordering signal that has nothing to do with identity. The first build conditioned on
# ``log_n_other`` (the candidate's spot count) and ``observable`` (derived from the candidate's
# axial span). Measured, the resulting score correlated **-0.40 with the candidate's spot count**
# and only **+0.09 with the number of matched spots** -- it was ranking on how many spots the other
# animal happened to have. AUROC 0.524, against **0.674 for simply counting matched spots**.
#
# ``observable`` is the seductive one, because "a miss in a body region the other photo never showed
# is not evidence" is *true*. But misses carry negative weight, so excusing them RAISES the score,
# and a candidate photo showing less of the animal would win by revealing less. Correct Bayesian
# bookkeeping and a usable ranking score are not the same object here. Left out until that is
# resolved; see the note in ``spot_rows``.
FEATURES = ["size_pct"]
ALL_FEATURES = ["size_pct", "observable", "log_n_other"]


# ----------------------------------------------------------------------------- observations
def _match_mask(A, B, thr: float = MATCH_THR) -> np.ndarray:
    """Per-spot: did this spot of ``A`` find a mutual nearest neighbour in ``B`` clearing ``thr``."""
    from aggregator import _norm                                          # noqa: PLC0415
    S = _norm(A.spots) @ _norm(B.spots).T
    ab, bb = S.argmax(1), S.argmax(0)
    return np.array([bb[ab[i]] == i and float(S[i, ab[i]]) >= thr for i in range(len(S))])


def spot_rows(sets, images, size, tpos, *, same: bool, n_pairs: int, seed: int = 0):
    """``(X, y)`` observations of "did this spot survive", drawn from same- or different-animal pairs.

    Both classes must be sampled from the SAME photo pool, otherwise the two fitted probabilities
    differ because of what was photographed rather than because of identity, and their ratio stops
    meaning anything.
    """
    import strict_match as sm                                             # noqa: PLC0415
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = {}
    for i in images:
        if len(sets[i].spots) >= 4:
            by.setdefault(sets[i].label, []).append(i)
    pool = [i for v in by.values() for i in v]

    if same:
        pairs = [(a, b) for v in by.values() if len(v) > 1 for a in v for b in v if a != b]
        if len(pairs) > n_pairs:
            pairs = [pairs[k] for k in rng.choice(len(pairs), n_pairs, replace=False)]
    else:
        pairs, guard = [], 0
        while len(pairs) < n_pairs and guard < n_pairs * 50:
            guard += 1
            a, b = int(rng.choice(pool)), int(rng.choice(pool))
            if sets[a].label != sets[b].label:
                pairs.append((a, b))

    X, y = [], []
    for a, b in pairs:
        A, B = sets[a], sets[b]
        surv = _match_mask(A, B)
        tb = np.array([tpos.get((B.sid, int(j)), np.nan) for j in B.spot_ids], float)
        ta = np.array([tpos.get((A.sid, int(j)), np.nan) for j in A.spot_ids], float)
        obs = sm.observability(ta, sm.axial_span(tb))
        ln = float(np.log1p(len(B.spots)))
        for k, spid in enumerate(A.spot_ids):
            # Built as a dict and then SELECTED by FEATURES, so which properties condition the ratio
            # is one list at the top of the module rather than three parallel edits that can drift.
            row = {"size_pct": size.get((A.sid, int(spid)), 0.5),
                   "observable": float(obs[k]),
                   "log_n_other": ln}
            X.append([row[f] for f in FEATURES])
            y.append(int(surv[k]))
    return np.asarray(X, float), np.asarray(y, float)


# ----------------------------------------------------------------------------- model
class LLRModel:
    """Per-spot log-likelihood ratio: how much this spot's fate argues for the pair being one animal.

    Holds two fitted probabilities — of a spot matching given the pair IS the same animal, and given
    it is NOT — as functions of the spot's size, how visible its body region was in the other photo,
    and how many spots that other photo has. Everything else follows from those two.
    """

    def __init__(self, m_same, s_same, m_diff, s_diff, base_same: float, base_diff: float):
        self.m_same, self.s_same = m_same, s_same
        self.m_diff, self.s_diff = m_diff, s_diff
        self.base_same, self.base_diff = base_same, base_diff

    @staticmethod
    def fit(sets, train_images, size, tpos, *, n_pairs=1500, seed=0) -> "LLRModel":
        from aggregator import train_aggregator, _prob                    # noqa: PLC0415
        Xs, ys = spot_rows(sets, train_images, size, tpos, same=True, n_pairs=n_pairs, seed=seed)
        Xd, yd = spot_rows(sets, train_images, size, tpos, same=False, n_pairs=n_pairs, seed=seed)
        ms, ss = train_aggregator(Xs, ys, hidden=0, seed=seed)
        md, sd = train_aggregator(Xd, yd, hidden=0, seed=seed)
        return LLRModel(ms, ss, md, sd, float(ys.mean()), float(yd.mean()))

    def _p(self, model, scaler, X):
        from aggregator import _prob                                      # noqa: PLC0415
        # Clipped away from 0/1: an unclipped probability makes a single confident spot able to
        # dominate the whole sum through a log blowing up, which is how one extraction artefact
        # would decide a pair.
        return np.clip(_prob(model, scaler, X), 1e-3, 1 - 1e-3)

    def spot_llr(self, X: np.ndarray, matched: np.ndarray) -> np.ndarray:
        """Log-likelihood ratio contributed by each spot, given whether it matched."""
        ps = self._p(self.m_same, self.s_same, X)
        pd = self._p(self.m_diff, self.s_diff, X)
        matched = np.asarray(matched, bool)
        return np.where(matched, np.log(ps / pd), np.log((1.0 - ps) / (1.0 - pd)))

    def pair_score(self, A, B, size, tpos) -> float:
        """Total evidence, in log-odds, that these two photos show one animal.

        Summed over the QUERY's spots only. For one query ranked against many candidates the number
        of terms is then constant, so candidates are compared on evidence rather than on how many
        spots they happen to have — the same size-normalisation the learned aggregator discovered
        for itself (results.md #25, ``log_nq`` -0.64).
        """
        import strict_match as sm                                         # noqa: PLC0415
        if len(A.spots) == 0 or len(B.spots) == 0:
            return 0.0
        surv = _match_mask(A, B)
        ta = np.array([tpos.get((A.sid, int(j)), np.nan) for j in A.spot_ids], float)
        tb = np.array([tpos.get((B.sid, int(j)), np.nan) for j in B.spot_ids], float)
        obs = sm.observability(ta, sm.axial_span(tb))
        ln = float(np.log1p(len(B.spots)))
        rows = [{"size_pct": size.get((A.sid, int(j)), 0.5), "observable": float(o),
                 "log_n_other": ln} for j, o in zip(A.spot_ids, obs)]
        X = np.array([[r[f] for f in FEATURES] for r in rows], float)
        return float(self.spot_llr(X, surv).sum())

    def describe(self) -> str:
        w_s = self.m_same.net.weight.detach().numpy().ravel()
        w_d = self.m_diff.net.weight.detach().numpy().ravel()
        out = [f"  base rates: P(match|same) {self.base_same:.3f}   "
               f"P(match|diff) {self.base_diff:.3f}   "
               f"-> a match is worth {np.log(self.base_same / max(self.base_diff, 1e-9)):+.2f} "
               f"log-odds, a miss "
               f"{np.log((1 - self.base_same) / max(1 - self.base_diff, 1e-9)):+.2f}",
               f"  {'feature':<14}{'weight|same':>13}{'weight|diff':>13}"]
        for i, f in enumerate(FEATURES):
            out.append(f"  {f:<14}{w_s[i]:>13.3f}{w_d[i]:>13.3f}")
        return "\n".join(out)


# ----------------------------------------------------------------------------- open-set builder
def build_openset_llr(model: LLRModel, sets, gallery_imgs, query_imgs, size, tpos):
    """``(scores, qids, clab)`` — one log-odds score per (query photo, candidate individual).

    A candidate individual is scored by its BEST photo rather than by its photos concatenated: at
    this survival rate, concatenating gives a candidate with three photos three chances to explain
    every query spot, so the animals we know best would score highest regardless of identity. "Does
    any single photo of this animal agree with the query" is also the question a person answers.
    """
    by: dict[str, list[int]] = {}
    for i in gallery_imgs:
        by.setdefault(sets[i].label, []).append(i)
    sc, qids, clab = [], [], []
    for q in query_imgs:
        for c, gal_all in by.items():
            gal = [g for g in gal_all if g != q]
            if not gal:
                continue
            sc.append(max(model.pair_score(sets[q], sets[g], size, tpos) for g in gal))
            qids.append(q); clab.append(c)
    return np.array(sc, float), np.array(qids), np.array(clab, dtype=object)
