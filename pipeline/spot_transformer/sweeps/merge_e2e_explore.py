"""Merge every ``e2e_explore_*.json`` shard/study into one ranked table.

    python pipeline/spot_transformer/sweeps/merge_e2e_explore.py
    -> artifacts/spot_transformer/sweeps/e2e_explore/RESULTS_e2e_explore_MERGED.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIR = REPO / "artifacts" / "spot_transformer" / "sweeps" / "e2e_explore"
REF = dict(ident_r1=(0.717, 0.062), census_f=(0.807, 0.0), bal_acc_cal=(0.771, 0.0),
           review_at90=(0.85, 0.0), auroc_base=(0.731, 0.0))

# study membership by config-name prefix (the JSON files don't carry it per-row)
def _study(name: str) -> str:
    if name == "baseline":
        return "baseline"
    return {"id_": "identr", "gate": "gate", "hn_": "hardneg", "gen_": "general"}.get(
        next((p for p in ("id_", "gate", "hn_", "gen_") if name.startswith(p)), ""), "other")


def _g(v):
    return v[0] if isinstance(v, (list, tuple)) else v


def main() -> None:
    files = sorted(DIR.glob("e2e_explore_*.json"))
    files = [f for f in files if "_MERGED" not in f.name]
    if not files:
        sys.exit(f"no e2e_explore_*.json in {DIR}")

    rows: dict[str, dict] = {}                          # name -> row (dedupe baseline across shards)
    for f in files:
        blob = json.loads(f.read_text())
        for r in blob["results"]:
            if r["name"] in rows and r["name"] != "baseline":
                continue
            rows.setdefault(r["name"], r)

    by_study: dict[str, list] = {}
    for r in rows.values():
        by_study.setdefault(_study(r["name"]), []).append(r)

    base = rows.get("baseline")
    md = ["# e2e_transformer exploration — MERGED", "",
          f"- {len(files)} shard/study files · {len(rows)} distinct configs",
          "- Columns: identR@1 (identification) · censusF · balAcc + review@90 (novelty gate).",
          "  `Δ` is vs the local `baseline` row.", ""]
    if base:
        b = base
        md += [f"**baseline**  identR@1 {_g(b['ident_r1']):.3f} ± {b['ident_r1'][1]:.3f} · "
               f"censusF {_g(b['census_f']):.3f} · balAcc {_g(b['bal_acc_cal']):.3f} · "
               f"review@90 {_g(b['review_at90']):.0%}   "
               f"(all9_q0.4 ref: identR@1 {REF['ident_r1'][0]:.3f} ± {REF['ident_r1'][1]:.3f})", ""]

    order = ["identr", "gate", "hardneg", "general", "other"]
    for study in [s for s in order if s in by_study]:
        rs = by_study[study]
        key = "bal_acc_cal" if study == "gate" else "ident_r1"
        rs.sort(key=lambda r: -_g(r[key]))
        md += ["", f"## {study}  (ranked by {'balAcc' if key == 'bal_acc_cal' else 'identR@1'})", "",
               "| config | identR@1 | Δ | censusF | balAcc | b′ | review@90 | t |",
               "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            d1 = _g(r["ident_r1"]) - (_g(base["ident_r1"]) if base else REF["ident_r1"][0])
            md.append(
                f"| {r['name']} | {_g(r['ident_r1']):.3f} ± {r['ident_r1'][1]:.3f} | {d1:+.3f} "
                f"| {_g(r['census_f']):.3f} | {_g(r['bal_acc_cal']):.3f} "
                f"| {_g(r['auroc_bprime']):.3f} | {_g(r['review_at90']):.0%} | {_g(r['secs']):.0f}s |")

    # overall bests
    allr = [r for r in rows.values() if r["name"] != "baseline"]
    best_id = max(allr, key=lambda r: _g(r["ident_r1"]))
    best_gate = max(allr, key=lambda r: _g(r["bal_acc_cal"]))
    least_review = min(allr, key=lambda r: _g(r["review_at90"]))
    md += ["", "## Overall", "",
           f"- best **identR@1**: `{best_id['name']}`  {_g(best_id['ident_r1']):.3f} ± "
           f"{best_id['ident_r1'][1]:.3f}",
           f"- best **balAcc gate**: `{best_gate['name']}`  {_g(best_gate['bal_acc_cal']):.3f}",
           f"- least **human review**: `{least_review['name']}`  {_g(least_review['review_at90']):.0%} "
           f"(identR@1 {_g(least_review['ident_r1']):.3f})", ""]

    out = DIR / "RESULTS_e2e_explore_MERGED.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(rows)} configs from {len(files)} files)")


if __name__ == "__main__":
    main()
