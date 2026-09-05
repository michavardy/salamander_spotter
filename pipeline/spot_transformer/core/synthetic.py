"""Synthetic salamander populations, in the matcher's own feature space.

WHY THIS EXISTS. Every number in ``sweeps/strict/RESULTS_*.md`` rests on ~28 test queries per
fold, so an F0.5 difference of 0.02 is one photo. Two questions cannot be answered at that size:

  1. **Does scale help?** — which is really *two* questions that real data cannot separate,
     because collecting more animals moves both at once:
         train scale    more individuals to fit on          -> expected to HELP
         gallery scale  more candidates to be confused by   -> expected to HURT
     Here they are independent knobs (``n_train`` vs ``n_gallery`` in ``sweep_synthetic.py``).
  2. **Below what score is a photo hopeless?** — needs many queries spanning a controlled
     quality range, with the ground truth known for every one.

WHAT IS AND IS NOT SIMULATED. This generates spots directly as ``(N, 62)`` embeddings in the
SAME space ``embeddings.get_spot_embeddings`` produces — ``[37 shape | 25 position]``, each block
L2-normalized — using the *same* ``_sinusoidal`` position featurizer, so `strict_match`,
`strict_voter` and the census eval run on it unmodified. Pixels, segmentation and spot extraction
are NOT simulated; their failure modes enter as the detection/noise model below. That is the
intended scope: this measures the MATCHER, holding the detector's behaviour as a parameter.

THE GENERATIVE MODEL.

  identity   individual j owns ``M_j ~ Poisson`` true spots. Each spot is a point
             ``(t, u)`` on the body (t: 0=head..1=tail, u: signed midline offset in half-width
             units) carrying a shape code on the unit sphere in R^37. Spot positions are drawn
             with a hard-core radius so spots do not overlap, as real ones do not.

  photo      each photo of j is a NOISY, PARTIAL view of that identity:
                 detection   each true spot survives w.p. p_detect      (missed extraction)
                 spurious    Poisson(lam) extra spots not in the identity (false extraction)
                 shape noise shape code += N(0, sigma_shape), renormalized
                 position    (t, u) += N(0, sigma_pos_*)                (alignment error)
                 axis jitter t -> a*t + b per photo                     (body-axis misfit)
             All five are tied to one per-photo scalar ``quality`` in (0, 1]: quality 1 is a
             perfect view, quality 0 is unusable. That scalar is the ground truth the
             "below which score can my model not cope" question is measured against.

WHAT YOU GET BACK. :class:`Population` carries the ``ImageSet`` list the pipeline consumes, the
``pos_lookup`` the position gate needs, an ORACLE ``weight_lookup`` (true per-spot
distinctiveness — the thing ``distinctiveness.py`` tries to learn, blended to mirror the SHIPPED
``strict_match.DEFAULT_WEIGHTS``; see :func:`_oracle_distinctiveness`), and a ``truth`` frame with
one row per photo (its quality and how many spots were kept/invented). Because spot ids are
stable across photos of one individual, you also get ground-truth spot CORRESPONDENCE, which
real photos never give you.

    from synthetic import SynthConfig, generate
    pop = generate(SynthConfig(n_individuals=2000, seed=0))
    pop.sets, pop.pos_lookup, pop.weight_lookup, pop.truth

CALIBRATION CAVEAT. Defaults are plausible, not fitted: real spot counts, axial density and
shape-code covariance are not measured here. Conclusions are therefore about *the method under
this generator*. Before trusting an absolute number, check a summary statistic you did NOT tune
(nearest-neighbour spacing, spots-per-animal) against the real DB. Relative statements — "R@1
falls X% per doubling of the gallery" — are far more robust than absolute ones.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_ST = Path(__file__).resolve().parents[1]
for _sub in ("core", "models", "eval"):
    p = str(_ST / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from data import ImageSet                                    # noqa: E402
from embeddings import _sinusoidal, POS_BANDS, SHAPE_HARMONICS   # noqa: E402  (share, never re-derive)
from strict_match import _rank01                             # noqa: E402

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

SHAPE_DIM = SHAPE_HARMONICS * 4 - 3        # 37 — matches _efd's output width
POS_DIM = 4 * POS_BANDS + 1                # 25 — sin/cos(t) + sin/cos(u) + side
EMB_DIM = SHAPE_DIM + POS_DIM              # 62


@dataclass
class SynthConfig:
    """Every knob. Defaults give a ~2000-animal population that runs in well under a minute."""

    # ---- population size -------------------------------------------------------------
    n_individuals: int = 2000
    photos_per_individual: int = 3          # >=2 so a gallery individual can also be queried
    seed: int = 0

    # ---- pattern: how many spots, and where ------------------------------------------
    mean_spots: float = 25.0                # Poisson mean of TRUE spots per individual
    min_spots: int = 5
    t_density: str = "dorsal"               # "uniform" or "dorsal" (denser mid-body, as observed)
    hard_core: float = 0.02                 # min spot separation in body-length units (no overlap)
    width_frac: float = 0.15                # body width / body length — converts u to body units

    # ---- pattern: how DISTINGUISHABLE shapes are (the entropy knob) -------------------
    # Shape codes live on the unit sphere in R^37. Sampling isotropically makes every spot
    # near-orthogonal to every other (cos ~ 1/sqrt(37)) — far more distinctive than reality.
    # `shape_rank` restricts them to a k-dim subspace: LOWER rank = shapes crowd together =
    # more collisions = a harder, more realistic population. This is the single most important
    # knob for "how many animals can be told apart".
    shape_rank: int = 8
    shape_concentration: float = 0.0        # >0 pulls codes toward a few archetypes (mixture)
    n_archetypes: int = 12                  # only used when shape_concentration > 0

    # ---- photo degradation: all scale with (1 - quality) ------------------------------
    quality_a: float = 5.0                  # per-photo quality ~ Beta(a, b); (5,2) is mostly-good
    quality_b: float = 2.0
    min_detect: float = 0.35                # p_detect at quality=0 (at quality=1 it is 1.0)
    # Descriptor noise is specified as the COSINE between two views of the SAME spot at quality=0,
    # not as a raw sigma. A sigma is unreadable at 37 dimensions: sigma=1.2 sounds moderate but
    # gives noise 3.6x the signal, i.e. cos 0.27, at which mutual-nearest-neighbour matching
    # cannot work at all and `raw_voting` sits at exactly chance. That is a regime the real data
    # is nowhere near, and calibrating into it silently invalidates every conclusion.
    min_shape_cos: float = 0.60             # same-spot shape cosine at quality=0 (1.0 at quality=1)
    max_pos_noise_t: float = 0.06           # sigma on t (body-length units) at quality=0
    max_pos_noise_u: float = 0.35           # sigma on u (half-width units) at quality=0
    max_spurious: float = 6.0               # Poisson mean of invented spots at quality=0
    max_axis_jitter: float = 0.08           # per-photo AFFINE drift of the body axis at quality=0
    # A purely affine axis error is too kind: the spot constellation stays rigid, so the
    # geometric-consistency and RANSAC features stay near-perfect for true matches and `logreg`
    # scores far above the other matchers -- a gap the real data does not show. Real animals bend.
    # This adds a smooth nonrigid warp of the axial coordinate, which is what actually destroys
    # constellation geometry in the field.
    max_warp: float = 0.10                  # amplitude of the nonrigid axial warp at quality=0

    # ---- oracle distinctiveness ------------------------------------------------------
    # ``shipped`` mirrors ``strict_match.DEFAULT_WEIGHTS``: size 2.0, isolation 1.0, rarity 0.
    # ``legacy`` is the pre-2026-08 oracle (rarity + isolation, equal halves) and exists only to
    # reproduce synthetic numbers recorded before the rarity strip. See _oracle_distinctiveness.
    oracle_blend: str = "shipped"
    size_sigma: float = 0.6                 # log-normal spread of per-spot size (identity-level)
    k_rarity: int = 10                      # legacy blend only
    k_isolation: int = 3
    rarity_ref_cap: int = 50_000            # legacy blend only: subsample the kNN reference cloud

    label_prefix: str = "sy"
    preset_name: str = "realistic"

    def tag(self) -> str:
        """Filename-safe description of the knobs a result depends on.

        ``oracle_blend`` is appended only when it is NOT the shipped one, so switching back to the
        legacy rarity oracle writes to its own artifacts instead of overwriting a current result
        with a number produced under a different definition of "distinctive".
        """
        suffix = "" if self.oracle_blend == "shipped" else f"_oracle{self.oracle_blend}"
        return (f"{self.preset_name}_n{self.n_individuals}_p{self.photos_per_individual}"
                f"_s{self.mean_spots:g}_r{self.shape_rank}_seed{self.seed}{suffix}")

    @classmethod
    def preset(cls, name: str = "realistic", **overrides) -> "SynthConfig":
        """A difficulty preset, plus any explicit overrides.

        THE DEFAULTS ARE NOT NEUTRAL. An unconstrained population is trivially separable — 22
        spots whose positions are known to within a percent of body length is a near-unique
        fingerprint, and the first calibration run scored R@1 = 1.000 at a 50-animal gallery
        against ~0.4 on the real data. A scaling curve measured there would describe a task
        nobody has. The knobs that actually control difficulty, in order of leverage:

            shape_rank        how crowded the shape vocabulary is (low = confusable)
            max_axis_jitter   body-axis misfit — breaks the positional fingerprint, and is the
                              single biggest reason real matching is hard
            max_pos_noise_*   residual alignment error
            min_detect        how many spots survive extraction at low quality
            max_spurious      invented spots that must be reasoned past

        ``easy`` is the ceiling (near-noiseless — use it to check a change *can* help at all),
        ``realistic`` is tuned so R@1 at a real-sized gallery lands in the neighbourhood of the
        measured 0.37-0.49, and ``hard`` brackets it from below. Always re-run ``MODE=calibrate``
        after changing the real dataset — these are matched to `all_sasa_norm_2026_23_07`.
        """
        base: dict = {}
        if name == "easy":
            base = dict(shape_rank=8, min_shape_cos=0.90, max_pos_noise_t=0.05,
                        max_pos_noise_u=0.30, max_axis_jitter=0.05, max_warp=0.03,
                        min_detect=0.45, max_spurious=4.0, quality_a=5.0, quality_b=2.0)
        elif name == "realistic":
            base = dict(shape_rank=2, min_shape_cos=0.55, max_pos_noise_t=0.13,
                        max_pos_noise_u=0.75, max_axis_jitter=0.17, max_warp=0.13,
                        min_detect=0.26, max_spurious=10.0, quality_a=2.2, quality_b=2.0)
        elif name == "hard":
            base = dict(shape_rank=2, min_shape_cos=0.45, max_pos_noise_t=0.20,
                        max_pos_noise_u=1.0, max_axis_jitter=0.24, max_warp=0.20,
                        min_detect=0.20, max_spurious=14.0, quality_a=1.8, quality_b=2.5)
        else:
            raise ValueError(f"unknown preset {name!r} (easy | realistic | hard)")
        return cls(preset_name=name, **{**base, **overrides})


@dataclass
class Population:
    sets: list                              # list[ImageSet] — feed straight to the matcher
    pos_lookup: dict                        # (sid, spot_id) -> (axis_t, offset/length)
    weight_lookup: dict                     # (sid, spot_id) -> ORACLE distinctiveness in [0,1]
    truth: pd.DataFrame                     # one row per photo: quality, n_true/kept/spurious
    identities: dict                        # label -> dict(t, u, shape, spot_ids) noise-free
    cfg: SynthConfig
    labels: list = field(default_factory=list)

    def index_by_label(self) -> dict:
        out: dict = {}
        for i, s in enumerate(self.sets):
            out.setdefault(s.label, []).append(i)
        return out


# ----------------------------------------------------------------------------- pattern


def _sample_positions(n: int, cfg: SynthConfig, rng) -> tuple[np.ndarray, np.ndarray]:
    """``n`` spot positions ``(t, u)`` with a hard-core minimum separation.

    Dart-throwing rejection: propose, keep only if no accepted spot is within ``hard_core``
    (measured in body-length units, so u is converted before the distance test). Gives the
    repulsion real pigment patterns show; a pure Poisson process clumps unrealistically.

    Proposals come in batches into preallocated arrays — this is called once per individual AND
    once per photo (for spurious spots), so per-proposal allocation dominated generation time.
    """
    if n <= 0:
        return np.empty(0), np.empty(0)
    half_w = cfg.width_frac / 2.0
    hc2 = cfg.hard_core * cfg.hard_core
    ts = np.empty(n, float); us = np.empty(n, float); k = 0
    batch = max(4 * n, 32)
    for _ in range(30):
        tp = rng.beta(2.0, 2.0, size=batch) if cfg.t_density == "dorsal" \
            else rng.uniform(0.0, 1.0, size=batch)
        up = np.clip(rng.normal(0.0, 0.55, size=batch), -1.0, 1.0)
        for i in range(batch):
            if k:
                dt = ts[:k] - tp[i]
                du = (us[:k] - up[i]) * half_w                      # u -> body-length units
                if np.min(dt * dt + du * du) < hc2:
                    continue
            ts[k] = tp[i]; us[k] = up[i]; k += 1
            if k == n:
                return ts, us
    return ts[:k], us[:k]


def _sample_shapes(n: int, cfg: SynthConfig, rng, basis: np.ndarray,
                   archetypes: np.ndarray | None) -> np.ndarray:
    """``(n, 37)`` unit-norm shape codes confined to a ``shape_rank``-dimensional subspace.

    The subspace is what makes the population realistically confusable: at full rank two random
    spots are near-orthogonal, so telling animals apart is trivially easy and every scaling
    result would be optimistic.
    """
    k = max(1, min(cfg.shape_rank, SHAPE_DIM))
    z = rng.normal(size=(n, k))
    if archetypes is not None and cfg.shape_concentration > 0:
        pick = rng.integers(0, len(archetypes), size=n)
        z = archetypes[pick] + z / max(cfg.shape_concentration, 1e-6)
    codes = z @ basis[:k]                                          # (n, 37)
    return codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)


def _embed(shape: np.ndarray, t: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``[37 shape | 25 position]`` with each block L2-normalized — byte-for-byte the layout
    ``embeddings.get_spot_embeddings`` emits, so the matcher cannot tell these apart by format."""
    sh = shape / (np.linalg.norm(shape, axis=1, keepdims=True) + 1e-8)
    side = np.where(u >= 0, 1.0, -1.0).reshape(-1, 1)
    pos = np.concatenate([_sinusoidal(t, POS_BANDS), _sinusoidal(u, POS_BANDS), side], axis=1)
    pos /= np.linalg.norm(pos, axis=1, keepdims=True) + 1e-8
    return np.concatenate([sh, pos], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- photos


def _sigma_for_cos(min_cos: float, quality: float) -> float:
    """Per-component noise sigma that yields a given same-spot cosine, at this photo's quality.

    For a unit signal in R^D plus iid N(0, sigma^2) noise, ``cos ~ 1/sqrt(1 + D*sigma^2)``, so
    ``sigma = sqrt(1/cos^2 - 1)/sqrt(D)``. Target cosine interpolates from ``min_cos`` at
    quality 0 to 1 (noiseless) at quality 1.
    """
    c = float(np.clip(min_cos + (1.0 - min_cos) * quality, 1e-3, 0.999999))
    return float(np.sqrt(1.0 / (c * c) - 1.0) / np.sqrt(SHAPE_DIM))


def _degrade(ident: dict, quality: float, cfg: SynthConfig, rng) -> dict:
    """One noisy, partial view of an identity. Returns the observed spots + a bookkeeping row.

    Every degradation scales with ``1 - quality``, so a single scalar orders photos from perfect
    to unusable and the sweep can ask "below which quality does the matcher fail?".
    """
    q = float(np.clip(quality, 0.0, 1.0))
    bad = 1.0 - q
    n_true = len(ident["t"])

    p_detect = cfg.min_detect + (1.0 - cfg.min_detect) * q
    keep = rng.random(n_true) < p_detect
    if not keep.any():                                   # a photo with no spots is not a photo
        keep[rng.integers(n_true)] = True

    t = ident["t"][keep].copy()
    u = ident["u"][keep].copy()
    shape = ident["shape"][keep].copy()
    ids = ident["spot_ids"][keep].copy()
    size = ident["size"][keep].copy()                    # a spot's size is the spot's, not the photo's

    # body-axis misfit: a per-photo affine drift of the axial coordinate ...
    a = 1.0 + rng.normal(0.0, cfg.max_axis_jitter * bad)
    b = rng.normal(0.0, cfg.max_axis_jitter * bad)
    t = a * t + b
    # ... plus a smooth nonrigid bend, which is what really breaks constellation geometry
    if cfg.max_warp > 0:
        f = rng.uniform(1.0, 2.5)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        t = t + cfg.max_warp * bad * np.sin(2.0 * np.pi * f * t + phi)

    t = t + rng.normal(0.0, cfg.max_pos_noise_t * bad, size=t.shape)
    u = u + rng.normal(0.0, cfg.max_pos_noise_u * bad, size=u.shape)
    shape = shape + rng.normal(0.0, _sigma_for_cos(cfg.min_shape_cos, q), size=shape.shape)

    # spurious detections: spots that belong to no identity (debris, shadow, merged blobs)
    n_sp = int(rng.poisson(cfg.max_spurious * bad))
    if n_sp:
        st, su = _sample_positions(n_sp, cfg, rng)
        n_sp = len(st)
    if n_sp:
        ssh = rng.normal(size=(n_sp, SHAPE_DIM))
        t = np.concatenate([t, st]); u = np.concatenate([u, su])
        shape = np.vstack([shape, ssh])
        ids = np.concatenate([ids, -np.arange(1, n_sp + 1)])       # negative id = not a real spot
        # Debris and merged blobs are drawn from the same size law as real spots — deliberately.
        # Giving spurious spots systematically small sizes would let the size-weighted oracle
        # discount them for free, i.e. hand the matcher a detector for the noise it is supposed to
        # have to reason past.
        size = np.concatenate([size, rng.lognormal(0.0, cfg.size_sigma, size=n_sp)])

    t = np.clip(t, 0.0, 1.0)
    u = np.clip(u, -1.0, 1.0)
    return dict(t=t, u=u, shape=shape, spot_ids=ids.astype(int), size=size,
                n_true=n_true, n_kept=int(keep.sum()), n_spurious=int(n_sp), quality=q)


# ----------------------------------------------------------------------------- oracle weights


def _oracle_distinctiveness(all_emb: np.ndarray, group_sizes: np.ndarray, tu: np.ndarray,
                            size: np.ndarray, cfg: SynthConfig, rng) -> np.ndarray:
    """True per-spot distinctiveness in [0, 1] — what ``distinctiveness.py`` is trying to learn.

    The oracle has to weight the same things the REAL matcher weights, or a synthetic sweep
    calibrates ``strict_hand_pos`` against a salience no deployed code computes. Under the shipped
    ``strict_match.DEFAULT_WEIGHTS`` two factors have a synthetic analogue:

        size       equivalent-circle size percentile — hand weight **2.0**, and the factor the
                   human interesting-spot clicks fit at **+0.707** (results.md #46). Generated per
                   spot at identity level (:func:`generate`), so it is a property of the spot and
                   survives across photos of the same animal, as real spot size does.
        isolation  mean distance to the k nearest spots on the SAME animal in (t, u) — hand weight
                   1.0 ("not many other spots around").

    Percentile-normalized and blended in the shipped 2:1 ratio, exactly as
    ``strict_match.distinctiveness`` combines its factors. ``elongation`` / ``noncircularity`` /
    ``irregularity`` have no synthetic analogue (there is no contour to measure), so an
    oracle-vs-learned comparison here remains a LOWER bound on what the learned head can reach —
    and irregularity is the strongest real factor (+0.824), so the bound is a loose one.

    ``rarity`` is **out**, matching ``DEFAULT_WEIGHTS`` since 2026-08-15: it fits at −0.070 against
    the human clicks and correlates with spots that do NOT survive to a second photo, so weighting
    it steered evidence toward extraction artefacts. It was half of this oracle until now, which
    made every synthetic conclusion about distinctiveness-weighted matching a conclusion about a
    blend the matcher had already discarded. ``cfg.oracle_blend='legacy'`` restores it (and the
    population-wide kNN it needs) for reproducing pre-change numbers only.
    """
    from scipy.spatial import cKDTree

    n = len(all_emb)

    # Spots arrive as one contiguous block per image, so walk offsets rather than testing
    # `owner == i` per image — that scan is O(n_images * n_spots) and was 94% of generation.
    isolation = np.zeros(n)
    off = 0
    for m in group_sizes:
        sl = slice(off, off + m)
        off += m
        if m <= 1:
            isolation[sl] = 1.0
            continue
        kk = min(cfg.k_isolation, m - 1)
        d, _ = cKDTree(tu[sl]).query(tu[sl], k=kk + 1)
        isolation[sl] = d[:, 1:].mean(1)

    if cfg.oracle_blend == "legacy":
        ref = all_emb
        if n > cfg.rarity_ref_cap:                   # kNN against a subsample: same statistic, cheaper
            ref = all_emb[rng.choice(n, cfg.rarity_ref_cap, replace=False)]
        dist, _ = cKDTree(ref).query(all_emb, k=min(cfg.k_rarity + 1, len(ref)))
        rarity = dist[:, 1:].mean(1)
        return 0.5 * (_rank01(rarity) + _rank01(isolation))
    if cfg.oracle_blend != "shipped":
        raise ValueError(f"oracle_blend must be 'shipped' or 'legacy', got {cfg.oracle_blend!r}")

    w_size, w_iso = 2.0, 1.0                         # == strict_match.DEFAULT_WEIGHTS
    return (w_size * _rank01(size) + w_iso * _rank01(isolation)) / (w_size + w_iso)


# ----------------------------------------------------------------------------- driver


def generate(cfg: SynthConfig | None = None, *, verbose: bool = True) -> Population:
    """Build a whole synthetic population. O(n_individuals) — 2000 animals takes seconds."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)

    # a fixed random orthonormal basis: shape codes occupy its first `shape_rank` directions
    basis = np.linalg.qr(rng.normal(size=(SHAPE_DIM, SHAPE_DIM)))[0]
    archetypes = None
    if cfg.shape_concentration > 0:
        archetypes = rng.normal(size=(cfg.n_archetypes, max(1, min(cfg.shape_rank, SHAPE_DIM))))

    identities: dict = {}
    labels: list[str] = []
    for j in range(cfg.n_individuals):
        label = f"{cfg.label_prefix}_{j:05d}"
        m = max(cfg.min_spots, int(rng.poisson(cfg.mean_spots)))
        t, u = _sample_positions(m, cfg, rng)
        shape = _sample_shapes(len(t), cfg, rng, basis, archetypes)
        # Per-spot size, log-normal and fixed at IDENTITY level: real spot area is a property of
        # the animal's pigment, so the same spot is big in every photo of it. The oracle weight
        # (:func:`_oracle_distinctiveness`) reads it; nothing in the embedding or the matching does,
        # so it changes which spots count as evidence without changing the evidence itself.
        identities[label] = dict(t=t, u=u, shape=shape,
                                 size=rng.lognormal(0.0, cfg.size_sigma, size=len(t)),
                                 spot_ids=np.arange(1, len(t) + 1, dtype=int))
        labels.append(label)

    sets, pos_lookup, rows = [], {}, []
    half_w = cfg.width_frac / 2.0
    for label in labels:
        ident = identities[label]
        for p in range(cfg.photos_per_individual):
            q = float(rng.beta(cfg.quality_a, cfg.quality_b))
            obs = _degrade(ident, q, cfg, rng)
            sid = f"{label}_{p}"
            emb = _embed(obs["shape"], obs["t"], obs["u"])
            s = ImageSet(sid=sid, label=label, is_synth=False, spots=emb,
                         spot_ids=obs["spot_ids"])
            # centroids: the geometric features want image pixels; body-frame * a nominal body
            # length is the right shape and scale-free, and nothing downstream assumes real px.
            s.centroids = np.column_stack([obs["t"] * 1000.0, obs["u"] * half_w * 1000.0])
            s.axis_t = obs["t"].copy()
            s.spot_size = obs["size"].copy()           # oracle-weight input; see _oracle_distinctiveness
            sets.append(s)
            for spid, tt, uu in zip(obs["spot_ids"], obs["t"], obs["u"]):
                # the position gate's convention: (axis_t, offset / body LENGTH) — see
                # compare_strict.build_pos_lookup, which divides by length_px, not half-width
                pos_lookup[(sid, int(spid))] = (float(tt), float(uu * half_w))
            rows.append(dict(sid=sid, label=label, quality=obs["quality"],
                             n_true=obs["n_true"], n_kept=obs["n_kept"],
                             n_spurious=obs["n_spurious"], n_spots=len(obs["t"])))

    if verbose:
        logger.info(f" synthetic population: {cfg.n_individuals} individuals x "
              f"{cfg.photos_per_individual} photos = {len(sets)} images, "
              f"{sum(len(s.spots) for s in sets)} spots  (shape_rank={cfg.shape_rank})")

    # ---- oracle distinctiveness over every OBSERVED spot ----
    all_emb = np.vstack([s.spots for s in sets]).astype(np.float64)
    all_emb /= np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-12
    group_sizes = np.array([len(s.spots) for s in sets])
    tu = np.vstack([np.column_stack([s.axis_t, s.centroids[:, 1] / 1000.0]) for s in sets])
    size = np.concatenate([s.spot_size for s in sets])
    w = _oracle_distinctiveness(all_emb, group_sizes, tu, size, cfg, rng)

    weight_lookup = {}
    off = 0
    for s in sets:
        for k, spid in enumerate(s.spot_ids):
            weight_lookup[(s.sid, int(spid))] = float(w[off + k])
        off += len(s.spot_ids)

    return Population(sets=sets, pos_lookup=pos_lookup, weight_lookup=weight_lookup,
                      truth=pd.DataFrame(rows), identities=identities, cfg=cfg, labels=labels)


# ----------------------------------------------------------------------------- calibration


def compare_to_real(pop: Population, db_path=None) -> pd.DataFrame:
    """Summary statistics of the synthetic population beside the REAL one, for the caveat above.

    Compares spots-per-image and the within-image nearest-neighbour spacing distribution — two
    statistics the generator is not fitted to. Large divergence means the absolute numbers from a
    sweep should not be quoted as if they were about real salamanders.
    """
    import duckdb
    from data import DB_PATH

    con = duckdb.connect(str(db_path or DB_PATH), read_only=True)
    try:
        real = con.execute("SELECT salamander_id, spot_id, axis_t, axis_offset FROM spots").df()
        length = {r[0]: r[1] for r in
                  con.execute("SELECT salamander_id, length_px FROM body_axis").fetchall()}
    finally:
        con.close()

    def _spacing(t, u):
        if len(t) < 2:
            return []
        pts = np.column_stack([t, u])
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(d, np.inf)
        return list(d.min(1))

    r_counts, r_space = [], []
    for sid, g in real.groupby("salamander_id"):
        L = length.get(sid)
        if not L:
            continue
        r_counts.append(len(g))
        r_space += _spacing(g["axis_t"].to_numpy(float), g["axis_offset"].to_numpy(float) / L)

    s_counts = [len(s.spots) for s in pop.sets]
    s_space = []
    for s in pop.sets:
        s_space += _spacing(s.axis_t, s.centroids[:, 1] / 1000.0)

    def _row(name, real_v, syn_v):
        return dict(statistic=name,
                    real=float(np.nanmedian(real_v)) if len(real_v) else float("nan"),
                    synthetic=float(np.nanmedian(syn_v)) if len(syn_v) else float("nan"))

    return pd.DataFrame([
        _row("spots per image (median)", r_counts, s_counts),
        _row("nearest-spot spacing (median, body-length units)", r_space, s_space),
    ])


if __name__ == "__main__":
    pop = generate(SynthConfig(n_individuals=200, seed=0))
    logger.info(pop.truth.describe()[["quality", "n_spots", "n_kept", "n_spurious"]].round(3))
    logger.info(f" embedding dim = {pop.sets[0].spots.shape[1]} (expect {EMB_DIM})")
    logger.info(f" oracle weights: {len(pop.weight_lookup)} spots, "
          f"mean {np.mean(list(pop.weight_lookup.values())):.3f}")
