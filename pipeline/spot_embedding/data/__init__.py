"""data — read contours.db into SpotSet objects, score quality, build leakage-safe CV splits."""
from .quality import QualityConfig, assess, filter_ids
from .spot_store import SpotSet, dataset_summary, load_mask, load_spotsets, reconcile_raw
from .splits import EvalFold, check_leakage, make_folds

__all__ = [
    "SpotSet",
    "load_spotsets",
    "load_mask",
    "dataset_summary",
    "reconcile_raw",
    "EvalFold",
    "make_folds",
    "check_leakage",
    "QualityConfig",
    "assess",
    "filter_ids",
]
