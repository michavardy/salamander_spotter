from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, Sampler

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

dataset_name = "all_sasa_norm_2026_23_07"
dataset_path = f"datasets/{dataset_name}"
REPO_ROOT = Path(__file__).resolve().parents[3]   # core/ -> spot_transformer/ -> pipeline/ -> repo root
DB_PATH = REPO_ROOT / dataset_path / "db" / "contours.db"

# Which embedding FLOW everything downstream reads. The flows are separate tables built by
# `embeddings.build_flow`, so switching is a table swap with nothing to rebuild:
#   spot_embeddings         62-dim  [shape|position]  -> cosine = (cos_shape + cos_pos)/2  (additive)
#   spot_embeddings_tensor  988-dim shape (x) position -> cosine =  cos_shape * cos_pos    (conjunctive)
EMB_TABLE = os.environ.get("EMB_TABLE", "spot_embeddings")


def emb_tag() -> str:
    """Filename-safe suffix naming the active embedding flow (empty for the default), so a
    tensor run writes its results next to the concat run instead of overwriting it."""
    return "" if EMB_TABLE == "spot_embeddings" else "_" + EMB_TABLE.replace("spot_embeddings_", "")


def quality_tag() -> str:
    """Filename-safe suffix describing the active image_quality filter (empty when off), so a
    filtered run writes to its own results dir instead of clobbering the all-images run."""
    parts = []
    if os.environ.get("MIN_QUALITY"):       parts.append(f"q{os.environ['MIN_QUALITY']}")
    if os.environ.get("MAX_SPOTS_OUTSIDE"): parts.append(f"so{os.environ['MAX_SPOTS_OUTSIDE']}")
    return ("_" + "_".join(parts)) if parts else ""


# Which COLLECTION the individuals come from. ``all_sasa_norm`` is two populations that were merged
# by ``scripts/dataset/merge_haifa.py``, and they are not the same problem:
#
#   sasa   290 individuals, 443 real photos,  87 eval-eligible  -- the original study population
#   kf     461 individuals, 822 real photos, 237 eval-eligible  -- the Haifa/KF field export
#
# The merge nearly quadrupled the gallery, and every number in ``results.md`` predates it (that
# table's "~83 eligible individuals" is the sasa-only figure). Scoring against 750 individuals when
# only 290 are in the population of interest inflates the difficulty for free, so this gate exists
# to ask the narrower question honestly.
#
# There is no source column in the DB -- ``merge_haifa`` records KF individuals by writing
# ``KF-<series>-<number>`` into ``label_map.csv``'s ``hebrew_name`` field, where every sasa
# individual carries an actual Hebrew name. That file is therefore the authority.
SOURCE = os.environ.get("SOURCE", "all")             # all | sasa | kf
LABEL_MAP = os.environ.get("LABEL_MAP", "")          # override the label_map.csv location


def source_tag() -> str:
    """Filename-safe suffix naming the active source gate (empty when off), so a sasa-only run
    writes its results beside the all-images run instead of overwriting it."""
    return "" if SOURCE == "all" else "_" + SOURCE


def label_map_path() -> Path:
    """``images/<collection>/label_map.csv`` for the active dataset.

    ``dataset_name`` is the image directory plus a build date (``all_sasa_norm`` + ``_2026_23_07``),
    so the directory is recovered by dropping the three trailing date fields. Env ``LABEL_MAP``
    overrides it outright for datasets that do not follow the convention.
    """
    if LABEL_MAP:
        return Path(LABEL_MAP)
    return REPO_ROOT / "images" / dataset_name.rsplit("_", 3)[0] / "label_map.csv"


def label_sources() -> dict[str, str]:
    """``{label: 'kf' | 'sasa'}`` from ``label_map.csv``. Empty dict when the file is absent."""
    import csv                                                            # noqa: PLC0415
    p = label_map_path()
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return {r["label"]: ("kf" if str(r.get("hebrew_name", "")).startswith("KF-") else "sasa")
                for r in csv.DictReader(fh) if r.get("label")}


def source_keep_mask(sets, source: str | None = None, *, verbose: bool = True):
    """Boolean mask over ``sets``: True where the individual belongs to ``source``.

    Same mask contract as :func:`quality_keep_mask` and :func:`review_keep_mask`, and it exists in
    mask form for the same reason: gating the SCORED side (queries + gallery) is a different
    decision from gating TRAINING. Restricting the gallery to sasa is what makes the metric measure
    the population of interest; dropping the KF photos from training as well throws away 822 photos
    and 237 multi-photo individuals, and the aggregator is individual-agnostic by construction
    (``results.md`` #3), so it may well be learning something from them that transfers. Which of
    those is better is a measurement, not a preference -- keep them separable so it can be run.

    Synthetic views follow their individual, and an individual missing from ``label_map.csv`` is
    KEPT with a warning rather than silently dropped: an unreadable map should not quietly shrink
    the dataset.
    """
    source = SOURCE if source is None else source
    if source == "all":
        return np.ones(len(sets), dtype=bool)
    if source not in ("sasa", "kf"):
        raise ValueError(f"SOURCE must be all|sasa|kf, got {source!r}")

    src = label_sources()
    if not src:
        logger.warning(f" source gate: no label_map at {label_map_path()} -> keeping everything")
        return np.ones(len(sets), dtype=bool)

    unknown = {s.label for s in sets if s.label not in src}
    mask = np.array([src.get(s.label, source) == source for s in sets], dtype=bool)
    if verbose:
        real = np.array([not s.is_synth for s in sets])
        logger.info(f" source gate  SOURCE={source}  -> kept {int(mask.sum())}/{len(sets)} images "
              f"({int((mask & real).sum())}/{int(real.sum())} real, "
              f"{len({s.label for s, k in zip(sets, mask) if k})} individuals)")
        if unknown:
            logger.warning(f"   ! {len(unknown)} labels absent from label_map.csv, kept: "
                  f"{sorted(unknown)[:5]}")
    return mask


def quality_keep_mask(sets, db_path: Path | str = DB_PATH, *, verbose: bool = True):
    """Boolean mask over ``sets``: True where the photo passes the image_quality gate.

    The mask form exists so a caller can apply the gate to the EVAL side only. Filtering the list
    (``apply_quality_filter``) necessarily gates training too, and at MIN_QUALITY=0.4 that costs
    ~3x the training images -- measurably worse on R@1, AUROC, census F0.5 and balanced accuracy
    than keeping the dropped photos as training data and gating only what is scored. See
    ``sweep_quality_control.py`` and its ``TRAIN_POOL`` knob for the measurement.

    Env vars, unchanged: MIN_QUALITY (overall_quality >=), MAX_SPOTS_OUTSIDE (spots_outside_frac
    <=). All-True when neither is set. Synthetic (`_g`) views and photos with no image_quality
    row are always kept -- the former are training-only, the latter are never guessed at.
    """
    min_q = os.environ.get("MIN_QUALITY")
    max_out = os.environ.get("MAX_SPOTS_OUTSIDE")
    if not min_q and not max_out:
        return np.ones(len(sets), dtype=bool)
    min_q = float(min_q) if min_q else None
    max_out = float(max_out) if max_out else None
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute("SELECT salamander_id, overall_quality, spots_outside_frac "
                       "FROM image_quality").fetchall()
    con.close()
    qual = {sid: (oq, so) for sid, oq, so in rows}

    def keep(s):
        if s.is_synth or s.sid not in qual:
            return True
        oq, so = qual[s.sid]
        if min_q is not None and (oq is None or oq < min_q):
            return False
        if max_out is not None and so is not None and so > max_out:
            return False
        return True

    mask = np.array([keep(s) for s in sets], dtype=bool)
    if verbose:
        real = np.array([not s.is_synth for s in sets])
        logger.info(f" quality filter  MIN_QUALITY={min_q} MAX_SPOTS_OUTSIDE={max_out}  -> "
              f"kept {int(mask.sum())}/{len(sets)} images "
              f"({int((mask & real).sum())}/{int(real.sum())} real)")
    return mask


def review_keep_mask(sets, *, dataset: str | None = None, unreviewed: str = "keep",
                     verbose: bool = True):
    """Boolean mask over ``sets``: True where a HUMAN accepted the photo.

    The counterpart to :func:`quality_keep_mask` — same mask contract, human judgement instead of
    computed metrics. It exists to answer whether ``MIN_QUALITY=0.4`` (the single largest effect in
    results.md, census F0.5 0.347 -> 0.642) is picking the photos a person would pick, since that
    cutoff was never validated against one.

    ``unreviewed`` is the load-bearing choice, because only ~8% of the dataset was reviewed:

        keep    (default) unreviewed photos pass. The gate can then only REMOVE the 37 photos a
                human rejected, so it is a clean subtraction from the all-images baseline.
        drop    only human-accepted photos pass. This is the honest "trust the labels" gate, but it
                shrinks eval to ~117 photos and confounds the human's opinion with the review
                ORDER — individuals were served lowest-quality-first, so the reviewed subset is not
                a random sample and its numbers are not comparable to a full-dataset run.

    Synthetic views always pass: they are training-only (results.md #11) and the reviewer judged
    them as generator output, not as photographs.
    """
    keep_unreviewed = {"keep": True, "drop": False}[unreviewed]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import review_labels as rl                                          # noqa: PLC0415

    dataset = dataset or dataset_name
    accepted = rl.accepted_images(dataset)
    rejected = rl.rejected_images(dataset)
    if not accepted and not rejected:
        if verbose:
            logger.warning(f" review gate: no review store for {dataset} -> keeping everything")
        return np.ones(len(sets), dtype=bool)

    def keep(s):
        if s.is_synth or s.sid in accepted:
            return True
        if s.sid in rejected:
            return False
        return keep_unreviewed

    mask = np.array([keep(s) for s in sets], dtype=bool)
    if verbose:
        real = np.array([not s.is_synth for s in sets])
        n_rev = sum(1 for s in sets if s.sid in accepted or s.sid in rejected)
        logger.info(f" review gate  unreviewed={unreviewed}  -> kept {int(mask.sum())}/{len(sets)} images "
              f"({int((mask & real).sum())}/{int(real.sum())} real; {n_rev} were reviewed at all)")
    return mask


def apply_quality_filter(sets, db_path: Path | str = DB_PATH):
    """Drop image sets whose photo fails the image_quality gate. See :func:`quality_keep_mask`.

    Gates TRAINING as well as eval, because the dropped images leave the list entirely. Prefer
    ``quality_keep_mask`` + ``get_cv_folds(eval_mask=...)`` when you want the gate on the scored
    side only; that measured better on every metric.
    """
    mask = quality_keep_mask(sets, db_path)
    return [s for s, k in zip(sets, mask) if k]


class SpotSetDataset(Dataset):
    def __init__(
        self,
        sets: list[ImageSet],
        indices: list[int],
        *,
        label_to_id: dict[str, int] | None = None,   # None on train → build it here; pass train's on eval
        train: bool = False,
        dropout_p: float = 0.2,
        jitter_std: float = 0.0,                      # feature-space aug; 0.0 = off (default)
        seed: int = 0,
    ) -> None:
        self.sets = sets
        self.indices = list(indices)
        self.train = train
        self.dropout_p = dropout_p
        self.jitter_std = jitter_std
        self.rng = np.random.default_rng(seed)
        self.label_to_id = self.get_label_to_id(label_to_id) 
        self.n_classes = len(self.label_to_id)

    def get_label_to_id(self,label_to_id: dict[str, int] | None) -> dict[str, int]:
        if label_to_id is None:                                   # build from THIS split's labels
            labels = sorted({self.sets[j].label for j in self.indices})
            label_to_id = {lbl: i for i, lbl in enumerate(labels)}
        return label_to_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, bool]:
        s = self.sets[self.indices[i]]
        spots = s.spots
        if self.train:
            spots = spot_dropout(spots, self.dropout_p, self.rng)     # fresh every fetch
            spots = feature_jitter(spots, self.jitter_std, self.rng)  # no-op when jitter_std == 0
        return torch.from_numpy(spots), self.label_to_id[s.label], s.is_synth

class PKSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: SpotSetDataset,
        P: int,
        K: int,
        num_batches: int,
        seed: int = 0,
    ) -> None:
        self.P, self.K, self.num_batches = P, K, num_batches
        self.rng = np.random.default_rng(seed)
        # label -> dataset POSITIONS (0..len(dataset)-1) — what __getitem__ indexes
        self.label_to_pos: dict[str, list[int]] = {}
        for pos, gidx in enumerate(dataset.indices):
            self.label_to_pos.setdefault(dataset.sets[gidx].label, []).append(pos)
        self.labels = list(self.label_to_pos)
        if len(self.labels) < P:
            raise ValueError(f"P={P} but split has only {len(self.labels)} individuals")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for li in self.rng.choice(len(self.labels), size=self.P, replace=False):
                pool = self.label_to_pos[self.labels[li]]
                picks = self.rng.choice(pool, size=self.K, replace=len(pool) < self.K)
                batch.extend(int(p) for p in picks)
            yield batch                       # list of P*K positions

    def __len__(self):
        return self.num_batches

@dataclass
class ImageSet:
    sid: str            # per-photo key, "aa_1_g1"
    label: str          # identity, "aa_1"
    is_synth: bool
    spots: np.ndarray   # (N, 62) float32 — the tokens
    spot_ids: np.ndarray = None   # (N,) int spot_id per token (aligned to spots), for spot-level analysis
    centroids: np.ndarray = None  # (N, 2) float image-px centroid per spot, for geometric-consistency
    axis_t: np.ndarray = None     # (N,) float position along the body line (0=head..1=tail), for
                                  # axis-ordered models; NaN where the spot has no axis
    cnn_feat: np.ndarray = None   # (N, 512) frozen ImageNet-ResNet18 features of each spot's mask
                                  # crop — the "pretrained network" input (see pretrained_cnn.py)
    ssl_feat: np.ndarray = None   # (N, 64) self-supervised (SimCLR) spot embedding, pretrained on
                                  # unlabeled crops — see ssl_pretrain.py
    ssl_rand_feat: np.ndarray = None  # (N, 64) SAME encoder, never trained — the ablation control
                                      # that separates pretraining's benefit from the architecture's
    ssl_corr_feat: np.ndarray = None  # (N, 64) SimCLR trained on REAL cross-photo spot
                                      # correspondences instead of synthetic augmentations

def group_by_label(sets: list[ImageSet]) -> dict[str, list[int]]:
    """label -> indices into `sets` (all photos of one individual)."""
    label_to_indices = {}
    for i, s in enumerate(sets):
        label_to_indices.setdefault(s.label, []).append(i)
    return label_to_indices

def get_spot_embeddings(
    db_path: Path | str = DB_PATH,
    *,
    table: str | None = None,
) -> pd.DataFrame:
    """Read the persisted per-spot embeddings back from the DB.

    Mirrors :func:`embeddings.get_spots`: one row per spot, ordered by
    ``(salamander_id, spot_id)`` so a photo's spots are contiguous and
    ``df.groupby("salamander_id")`` yields per-image token sequences. The ``embedding``
    column is decoded from DuckDB's ``DOUBLE[]`` into an ``(d,)`` float array.

    The table is written by :func:`embeddings.write_spot_embeddings`; it defaults to the flow
    named by :data:`EMB_TABLE` (env ``EMB_TABLE``), so a whole run switches embedding with one
    variable instead of a signature change in every caller.
    """
    table = table or EMB_TABLE
    con = duckdb.connect(database=str(db_path), read_only=True)
    try:
        df = con.execute(
            f"SELECT salamander_id, spot_id, embedding FROM {table} "
            "ORDER BY salamander_id, spot_id"
        ).df()
    finally:
        con.close()
    df["embedding"] = df["embedding"].map(lambda e: np.asarray(e, dtype=np.float64))
    return df

_CORRECTIONS_ANNOUNCED = False


def get_image_sets(embeddings: pd.DataFrame) -> list[ImageSet]:
    """Photos -> ``ImageSet``s, with confirmed data corrections applied.

    This is the single point every model, sweep and metric passes through, which is why the
    corrections are applied HERE rather than in each of them: a merge or an exclusion that has to
    be remembered by twenty callers is one that will be forgotten by nineteen. See
    ``corrections.py``; ``CORRECTIONS=0`` turns it off for an A/B.
    """
    global _CORRECTIONS_ANNOUNCED
    import corrections as corr                                          # noqa: PLC0415

    aliases = corr.alias_map(dataset_name)
    ex_sids = corr.excluded_sids(dataset_name)
    ex_inds = corr.excluded_individuals(dataset_name)

    image_sets, n_merged, n_dropped = [], 0, 0
    for sid, sub_group in embeddings.groupby('salamander_id'):
        raw_label = "_".join(sid.split('_')[:2])
        label = aliases.get(raw_label, raw_label)
        n_merged += int(label != raw_label)
        if sid in ex_sids or label in ex_inds or raw_label in ex_inds:
            n_dropped += 1
            continue
        is_synth = sid.split('_')[-1].startswith('g')
        spots = np.stack(sub_group['embedding'].values).astype(np.float32)
        spot_ids = sub_group['spot_id'].to_numpy()
        image_sets.append(ImageSet(sid=sid, label=label, is_synth=is_synth,
                                   spots=spots, spot_ids=spot_ids))

    if not _CORRECTIONS_ANNOUNCED and (n_merged or n_dropped or not corr.ENABLED):
        _CORRECTIONS_ANNOUNCED = True
        logger.info(f"{corr.describe(dataset_name)}  ->  {n_merged} photos relabelled, "
              f"{n_dropped} photos dropped")
    return image_sets


def flatten_spots(spot_emb, mask, image_labels):
    """Batch of per-image spot tokens -> a flat bag of real spots, for spot-level SupCon.

    ``spot_emb`` (B, N, d), ``mask`` (B, N) bool [True == PAD], ``image_labels`` (B,) ->
    ``(Z (M, d), labels (M,), groups (M,))`` where M = number of real spots, ``labels`` is
    each spot's individual (its image's label), and ``groups`` is the image index within the
    batch (so the loss can exclude same-image spot pairs from the positives).
    """
    import torch
    real = ~mask                                                   # (B, N) True where real
    Z = spot_emb[real]                                             # (M, d)
    B, N = mask.shape
    groups = torch.arange(B, device=mask.device).unsqueeze(1).expand(B, N)[real]   # (M,)
    labels = image_labels.unsqueeze(1).expand(B, N)[real]                          # (M,)
    return Z, labels, groups

def get_cv_folds(sets, k=5, seed=0, eval_mask=None) -> list[tuple[list[int], list[int]]]:
    """k folds split on INDIVIDUALS (leakage-guarded).

    Eval folds are drawn only from individuals with >=2 real images (so every
    eval query has >=1 real gallery match). An eval individual is fully held out:
    train sees none of its images; eval uses its REAL images only (synthetic dropped).
    Everything else — synthetic-only and single-real individuals — is train-only.

    ``eval_mask`` (optional, boolean over ``sets``) restricts what may appear on the EVAL side
    without removing anything from training -- the quality gate applied to the scored side only.
    Eligibility is then judged on masked-in images too, so an individual needs >=2 images that
    pass the gate to be evaluable. Leave it None for the original behaviour.
    """
    by_label = group_by_label(sets)
    ok = (lambda i: True) if eval_mask is None else (lambda i: bool(eval_mask[i]))
    n_real = lambda lbl: sum(not sets[i].is_synth and ok(i) for i in by_label[lbl])
    eligible = sorted(lbl for lbl in by_label if n_real(lbl) >= 2)   # the ~83

    rng = np.random.default_rng(seed)
    eligible = [eligible[i] for i in rng.permutation(len(eligible))]
    fold_labels = np.array_split(eligible, k)

    all_labels = set(by_label)
    folds = []
    for evl in fold_labels:
        evl = set(evl)
        train_idx = [i for lbl in (all_labels - evl) for i in by_label[lbl]]
        eval_idx  = [i for lbl in evl for i in by_label[lbl]
                     if not sets[i].is_synth and ok(i)]
        folds.append((sorted(train_idx), sorted(eval_idx)))
    return folds

def assert_evaluable(folds, *, min_eval: int = 5, what: str = "the eval gate"):
    """Abort if a gate has left folds too small to score, instead of returning NaNs.

    An over-tight eval mask does not fail — it produces folds with one or zero queries, and every
    downstream metric comes back NaN or as a mean over two photos, which reads like a result. On
    this dataset ``review_keep_mask(unreviewed="drop")`` leaves **3** eligible individuals and two
    empty folds, because only ~8% of photos were reviewed. Fail loudly, name the cause, and say what
    the alternative is: a wrong number that looks plausible is more expensive than a crash.
    """
    sizes = [len(ev) for _, ev in folds]
    if min(sizes, default=0) >= min_eval:
        return folds
    raise SystemExit(
        f"{what} left folds with eval sizes {sizes} (need >= {min_eval} each).\n"
        f"  Too few images survive to measure anything - any number produced here would be an\n"
        f"  average over a handful of photos, not a result.\n"
        f"  If this is the review gate: only a small fraction of the dataset was reviewed, and\n"
        f"  individuals were served lowest-quality-first, so the reviewed subset is neither large\n"
        f"  enough nor a random sample. Use UNREVIEWED=keep (a clean subtraction of the photos a\n"
        f"  human rejected) rather than UNREVIEWED=drop.")


def spot_dropout(spots: np.ndarray, p: float, rng) -> np.ndarray:
    """Drop each spot row independently w.p. `p` (regularize + survive missing spots).
    Never returns 0 rows — a set needs >=1 token."""
    N = len(spots)
    if N <= 1 or p <= 0:
        return spots
    keep = rng.random(N) >= p
    if not keep.any():
        keep[rng.integers(N)] = True
    return spots[keep]

def feature_jitter(spots: np.ndarray, std: float, rng) -> np.ndarray:
    """Add small Gaussian noise to each spot's embedding (feature-space aug).

    ``std=0`` -> identity (the default in SpotSetDataset, so this is off unless opted in).
    Simulates descriptor measurement noise the EFD/position invariances don't cover, so the
    model learns embeddings robust to small perturbations. Blocks are L2-normalized upstream,
    so a sane ``std`` is small relative to that — try 0.01-0.03. Returns a fresh array."""
    if std <= 0:
        return spots
    return spots + rng.normal(0.0, std, size=spots.shape).astype(spots.dtype)

def collate_sets(batch):
    spots, label_ids, is_synth = zip(*batch)
    B, D = len(spots), spots[0].shape[1]
    lengths = [s.shape[0] for s in spots]
    N_max = max(lengths)
    X = torch.zeros(B, N_max, D, dtype=torch.float32)
    mask = torch.ones(B, N_max, dtype=torch.bool)     # start all-PAD (True)
    for i, (s, n) in enumerate(zip(spots, lengths)):
        X[i, :n] = s
        mask[i, :n] = False                            # real rows -> False
    labels = torch.tensor(label_ids, dtype=torch.long)
    return X, mask, labels                             # (+ torch.tensor(is_synth) if you keep it)


def write_image_embeddings(sids, Z, db_path=DB_PATH, *, table="image_embeddings",
                           labels=None, is_synth=None):
    df = pd.DataFrame({
        "salamander_id": list(sids),
        "embedding": [np.asarray(z, dtype=np.float64).tolist() for z in Z],
    })
    if labels is not None:   df["label"] = list(labels)
    if is_synth is not None: df["is_synth"] = list(is_synth)
    con = duckdb.connect(database=str(db_path))
    try:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.register("_img_df", df)
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM _img_df")
        con.unregister("_img_df")
    finally:
        con.close()
    logger.info(f"wrote {len(df)} image embeddings (dim {len(df['embedding'].iloc[0])}) -> {table}")


if __name__ == "__main__":
    from collections import Counter
    from torch.utils.data import DataLoader

    # 1) DB -> image sets -> fold[0] -> train dataset
    embeddings = get_spot_embeddings()
    image_sets = get_image_sets(embeddings)
    folds = get_cv_folds(image_sets, k=5, seed=0)
    train_idx, eval_idx = folds[0]
    dataset = SpotSetDataset(image_sets, train_idx, train=True, dropout_p=0.2, seed=0)
    logger.info(f"images={len(image_sets)}  train={len(dataset)}  "
          f"eval={len(eval_idx)}  classes={dataset.n_classes}")

    # 2) PKSampler -> collate_sets: pull ONE batch and look at the padded tensor + mask
    P, K = 4, 2
    sampler = PKSampler(dataset, P=P, K=K, num_batches=3, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_sets)  # num_workers=0
    X, mask, labels = next(iter(loader))
    real_per_row = (~mask).sum(1)                      # spots kept per image (post-dropout)
    logger.info(f"batch  X={tuple(X.shape)}  mask={tuple(mask.shape)}  N_max={X.shape[1]}  "
          f"real_per_row={real_per_row.tolist()}")
    logger.info(f"labels={labels.tolist()}  -> {len(set(labels.tolist()))} distinct (expect P={P}), "
          f"counts={dict(Counter(labels.tolist()))} (expect all K={K})")
    n0 = int(real_per_row[0])                           # verify the mask convention on row 0
    assert not mask[0, :n0].any(), "real rows must be False"
    assert mask[0, n0:].all(),     "pad rows must be True"

    # 3) write_image_embeddings: no transformer yet -> mean-pool baseline as the fingerprint
    #    (mean over each image's spots -> one (62,) vector; the baseline eval.py compares to)
    Z = np.stack([img.spots.mean(0) for img in image_sets])          # (M, 62)
    write_image_embeddings(
        [img.sid for img in image_sets], Z,
        table="image_embeddings_smoke",                              # scratch table, safe to drop
        labels=[img.label for img in image_sets],
        is_synth=[img.is_synth for img in image_sets],
    )
    con = duckdb.connect(str(DB_PATH), read_only=True)
    readback = con.execute(
        "SELECT salamander_id, label, is_synth, len(embedding) AS d "
        "FROM image_embeddings_smoke LIMIT 5"
    ).df()
    con.close()
    logger.info(readback)

    breakpoint()   # live: X, mask, labels, real_per_row, Z, readback, dataset, sampler