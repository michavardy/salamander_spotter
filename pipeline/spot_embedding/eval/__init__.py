"""eval — retrieval / verification / open-set metrics, the harness, and reports."""
from .harness import evaluate, evaluate_per_fold, score_fold
from .report import write_report

__all__ = ["evaluate", "evaluate_per_fold", "score_fold", "write_report"]
