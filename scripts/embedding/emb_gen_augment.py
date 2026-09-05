#!/usr/bin/env python3
"""emb-gen-augment — Gemini whole-image augmentation (opt-in, BILLED, offline).

Generates new photorealistic views of the SAME individuals (different lighting / background /
angle, spot pattern held fixed) to manufacture extra training positives. Refuses to run without
``--limit`` (a hard cap on API calls); skips already-generated files; needs GEMINI_API_KEY::

    pixi run emb-gen-augment --limit 20                      # ~20 views, singletons first
    pixi run emb-gen-augment --limit 40 --which singletons --n-per 2
    pixi run emb-gen-augment --limit 10 --which multi

``--which real-singletons`` counts only the REAL photos, so an individual that already has
synthetic views folded in still counts as a singleton and gets topped up. Combined with
``--top-up`` (``--n-per`` becomes a TARGET TOTAL rather than "generate this many"), it is the way
to bring every one-photo individual to the same number of views without re-paying for the ones
already done:

    pixi run emb-gen-augment --dataset all_sasa_norm_2026_19_07 \\
        --which real-singletons --n-per 2 --top-up --limit 0 --dry-run

Outputs land in ``images/synth_<dataset>/<label>_g<k>.png``. To USE them for training you must
then extract + validate spots (see docs/running.md → "Gemini augmentation workflow"):

    pixi run extract-spot-labels all --input synth_<dataset>
    pixi run package-dataset --input synth_<dataset> --name synth_<dataset>
    # then fold the synth individuals into the training pool (self-consistency filtered).

Argument parsing only; logic lives in ``spot_embedding.augment.gemini_view``.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from spot_embedding import reconfigure_utf8  # noqa: E402
from spot_embedding._common import DEFAULT_DATASET  # noqa: E402
from spot_embedding.augment.gemini_view import GeminiViewGenerator  # noqa: E402
from spot_embedding.data import load_spotsets  # noqa: E402


G_RE = re.compile(r"_g\d+$")


def _is_synthetic(salamander_id: str) -> bool:
    """A generated view — the last delimited section is ``g<k>`` (``aa_1_g0``)."""
    return bool(G_RE.search(salamander_id))


def _select_ids(dataset: str, which: str) -> list[str]:
    spotsets = [ss for ss in load_spotsets(dataset) if not ss.is_empty]
    per = Counter(ss.label for ss in spotsets)
    by_label: dict[str, list[str]] = {}
    for ss in spotsets:
        by_label.setdefault(ss.label, []).append(ss.salamander_id)
    if which == "all":
        return sorted(ss.salamander_id for ss in spotsets)
    if which == "multi":
        return sorted(ids[0] for lbl, ids in by_label.items() if per[lbl] >= 2)
    if which == "real-singletons":
        # Count REAL photos only, so an individual whose synthetic views are already folded into
        # the dataset still counts as a singleton. The source is always its real photo.
        real: dict[str, list[str]] = {}
        for ss in spotsets:
            if not _is_synthetic(ss.salamander_id):
                real.setdefault(ss.label, []).append(ss.salamander_id)
        return sorted(ids[0] for ids in real.values() if len(ids) == 1)
    return sorted(ids[0] for lbl, ids in by_label.items() if per[lbl] == 1)  # singletons


def _existing_synth(dataset: str) -> Counter:
    """label -> how many synthetic views the dataset ALREADY carries (i.e. already paid for)."""
    return Counter(ss.label for ss in load_spotsets(dataset) if _is_synthetic(ss.salamander_id))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emb-gen-augment", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--limit", type=int, required=True,
                   help="hard cap on Gemini API calls (billed). Use --limit 0 for NO cap "
                        "(generate everything remaining — resumable, skips existing files)")
    p.add_argument("--only", default=None, metavar="IDS",
                   help="comma-separated source salamander_ids to generate for, e.g. "
                        "'je_3_1,ac_3_2'. Overrides --which and skips the spotset scan; this is "
                        "what the regenerate button in `pixi run preprocess-review` calls")
    p.add_argument("--which", choices=("singletons", "real-singletons", "multi", "all"),
                   default="singletons",
                   help="which individuals to augment (default singletons — they gain the most). "
                        "'real-singletons' counts REAL photos only, so individuals that already "
                        "have synthetic views still qualify")
    p.add_argument("--n-per", type=int, default=2, help="views to generate per source photo")
    p.add_argument("--top-up", action="store_true",
                   help="treat --n-per as the TARGET TOTAL of synthetic views per individual and "
                        "subtract the ones the dataset already has, instead of generating --n-per "
                        "more for everyone")
    p.add_argument("--dry-run", action="store_true",
                   help="print the per-individual plan and the billed-call estimate, then stop")
    p.add_argument("--min-source-spots", type=int, default=0, metavar="N",
                   help="skip individuals whose source photo shows fewer than N spots. An "
                        "occluded source gives Gemini almost no pattern to preserve, so it "
                        "invents one (ca_21_1 shows 10 spots through grass; its views came back "
                        "as a different animal with 34). Cheaper to skip than to filter later")
    p.add_argument("--out-dir", default=None, metavar="NAME",
                   help="write the views into images/<NAME> instead of images/synth_<dataset>. "
                        "Point it at the master dir to generate IN PLACE — existing _g views are "
                        "counted there and the numbering continues past them, so no fold step is "
                        "needed and nothing already generated is re-billed")
    p.add_argument("--model", default=None, help="override GEMINI_MODEL (image model)")
    return p


def main(argv: list[str] | None = None) -> int:
    reconfigure_utf8()
    args = build_parser().parse_args(argv)

    from spot_embedding._common import derive_label

    if args.only:
        ids = [s.strip() for s in args.only.split(",") if s.strip()]
        print(f"--only: {len(ids)} source photo(s): {', '.join(ids)}")
    else:
        ids = _select_ids(args.dataset, args.which)
    if args.min_source_spots > 0:
        n_spots = {ss.salamander_id: ss.n_spots for ss in load_spotsets(args.dataset)}
        kept = [i for i in ids if n_spots.get(i, 0) >= args.min_source_spots]
        print(f"--min-source-spots {args.min_source_spots}: skipping {len(ids) - len(kept)} "
              f"individual(s) whose source photo is too occluded to preserve a pattern")
        ids = kept
    gen = GeminiViewGenerator(args.dataset, limit=args.limit, out_dir=args.out_dir,
                              model=args.model)

    # How many views each selected individual still needs, under --top-up (--n-per is a target
    # total). Which dir is the source of truth for "already have" depends on where we write:
    #   --out-dir  : the output dir itself — the views live there, so counting them there and
    #                continuing the numbering past them makes a re-run a no-op (idempotent).
    #   default    : the packaged dataset — the output dir is a fresh synth_* dir holding only
    #                this run's progress, which generate()'s own `dst.exists()` skip handles.
    in_place = args.out_dir is not None
    start = gen.next_free_index() if in_place else {}
    if not args.top_up:
        have_synth = Counter()
    elif in_place:
        have_synth = gen.existing_views()
    else:
        have_synth = _existing_synth(args.dataset)
    need = {i: max(0, args.n_per - have_synth[derive_label(i)]) for i in ids}

    # cost preview: outputs are keyed by individual label; count what's missing (resumable)
    done = remaining = 0
    for i, n in need.items():
        lbl = derive_label(i)
        on_disk = 0 if in_place else len(list(gen.out.glob(f"{lbl}_g*.png"))) if gen.out.exists() else 0
        done += min(on_disk, n)
        remaining += max(0, n - on_disk)
    budget = "no cap" if args.limit <= 0 else f"{args.limit} calls"
    per_need = Counter(need.values())
    target = f"target {args.n_per} total" if args.top_up else f"{args.n_per}/individual"
    print(f"{args.which}: {len(ids)} individual(s), {target} → "
          f"{sum(have_synth.get(derive_label(i), 0) for i in ids)} already on disk, "
          f"~{remaining} to generate. budget: {budget} (BILLED).")
    for n in sorted(per_need):
        print(f"    {per_need[n]:4d} individual(s) need {n} more view(s)")
    if in_place:
        ex = [f"{derive_label(i)}_g{start.get(derive_label(i), 0)}"
              for i in ids if need[i]][:3]
        print(f"    IN PLACE -> {gen.out.name}/ ; numbering continues: {', '.join(ex)} ...")

    if args.dry_run:
        print(f"\nDRY RUN — nothing generated, nothing billed. Output would go to {gen.out}")
        return 0
    if remaining == 0:
        print("nothing to do — all selected views already generated.")
        return 0

    # Group by need so each generate() call stays uniform; ascending, so under a partial budget
    # the cheap top-ups finish before the from-scratch individuals consume it.
    written = []
    for n in sorted({v for v in need.values() if v > 0}):
        written += gen.generate(sorted(i for i, v in need.items() if v == n), n_per=n, start=start)
    print(f"done: wrote {len(written)} new synthetic view(s) -> {gen.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
