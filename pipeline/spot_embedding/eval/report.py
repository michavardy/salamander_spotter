"""Write a harness result to ``artifacts/spot_embedding/runs/<id>/`` as JSON + Markdown + CSV."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .._common import runs_dir, save_json


def _fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.4f}"
    return str(x)


def write_report(result: dict, *, dataset: str, config: dict | None = None,
                 run_id: str | None = None) -> Path:
    """Persist ``result`` (from :func:`harness.evaluate`) and return the run directory."""
    run_id = run_id or f"{datetime.now():%Y%m%d_%H%M%S}_{result['matcher']}"
    run_dir = runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "metrics.json",
              {"dataset": dataset, "config": config or {}, **result})

    # risk_coverage.csv
    rc = result.get("risk_coverage", {})
    with open(run_dir / "risk_coverage.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["coverage", "risk"])
        for cov, risk in zip(rc.get("coverage", []), rc.get("risk", [])):
            w.writerow([f"{cov:.6f}", f"{risk:.6f}"])

    # report.md
    agg = result["aggregate"]
    lines = [
        f"# Eval report — {result['matcher']}",
        "",
        f"- dataset: `{dataset}`",
        f"- folds: {result['n_folds']}",
        "",
        "## Aggregate (mean over folds)",
        "",
        "| metric | value |",
        "|--------|-------|",
        f"| rank-1 | {_fmt(agg.get('rank1'))} |",
        f"| rank-5 | {_fmt(agg.get('rank5'))} |",
        f"| mAP | {_fmt(agg.get('mAP'))} |",
        f"| verification AUC | {_fmt(agg.get('verify_auc'))} |",
        f"| verification TPR@1%FPR | {_fmt(agg.get('verify_tpr@fpr'))} |",
        f"| open-set AUROC | {_fmt(agg.get('openset_auroc'))} |",
        f"| AURC (risk-coverage, lower=better) | {_fmt(agg.get('aurc'))} |",
        "",
        "## Per fold",
        "",
        "| fold | gallery | closed | open | rank-1 | mAP | verify AUC | open AUROC |",
        "|------|---------|--------|------|--------|-----|------------|------------|",
    ]
    for r in result["folds"]:
        lines.append(
            f"| {r['fold']} | {r['n_gallery']} | {r['n_closed']} | {r['n_open']} | "
            f"{_fmt(r.get('rank1'))} | {_fmt(r.get('mAP'))} | "
            f"{_fmt(r.get('verify_auc'))} | {_fmt(r.get('openset_auroc'))} |"
        )
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    return run_dir
