#!/usr/bin/env python3
"""Review spot-match pairs one-at-a-time, full screen, with a comment per pair.

The match-error *grid* (``pipeline/spot_transformer/visualize_match_errors.py``) packs every
pair into one huge SVG where each salamander is tiny and the size-percentile numbers cover its
back. This tool is the opposite: it renders each pair as its **own big PNG** and gives you a
little local web app to page through them, jotting a comment on the ones that show a problem.

Two steps (a subcommand each):

    pixi run pair-review-gen              # rank -> render one big PNG per unique pair + manifest
    pixi run pair-review                  # open the browser app: ←/→ to page, type a comment

``generate`` reuses the exact ranking from ``visualize_match_errors`` (so the pairs are the same
ones the grid would show), then draws every unique (query, matched) pair once as a full-figure
PNG: match arcs on top, both individuals' spot-size maps below. ``--methods`` picks which match
algorithms contribute their top/mid/bottom pairs (default ``raw`` — fast, no training; add
``logreg`` etc. to include learned matchers, which train first and are slower).

``serve`` starts a stdlib HTTP server (no new deps, same shape as the interesting-spot selector).
Two things are saved *per pair*: a **verdict** (the label) and a **comment** (the colour). Page
back and both are still there; page to a pair you have never seen and both are blank. Everything
lands in::

    artifacts/pair_review/<dataset>/
        images/<query>__<matched>.png     one big PNG per pair
        manifest.json                     the pairs + where each came from (method/band/score)
        comments.json                     { "<query>__<matched>": "your note", ... }
        verdicts.json                     { "<query>__<matched>": {verdict, reasons, at}, ... }
        review.json                       manifest + verdict + comment merged — the deliverable

THE VERDICT IS THE POINT, and it is a deliberate change of altitude. results.md #44 measured the
per-EDGE label (is this spot the same spot) and found the human's accept/reject uncorrelated with
anything the 62-dim embedding encodes — AUROC 0.484 — because they are judging the *pair* on global
structure, not the edge on appearance. #46 measured the per-SPOT click and found it saturated
(-0.002 AUROC for a quarter more labels). So this tool asks the one question the human answers
reliably and cheaply, once per pair: **match / different / can't tell**, plus why.

Three things come out of it that the free-text comment could not feed:

* a calibration set for the ``match / new / abstain`` head (results.md open thread #6) — a verdict
  per pair, in seconds, where the old loop cost a click per spot;
* **label errors, for free**: a pair the filenames call the same animal that the reviewer calls
  different (or the reverse) is flagged ``disagrees_with_label``, which is the duplicate-identity
  adjudication of open thread #4 arriving as a side effect of ordinary review;
* a ``multi-animal`` reason chip, which accumulates the frames that
  ``data_hygiene.multi_animal_candidates`` can only *guess* at (next_steps_2.md §4).

``review.json`` is the deliverable: every pair with its metadata, verdict and comment inline. The
modelling side reads it through ``review_labels.pair_verdicts``. See docs/labeling_priorities.md
for why annotation effort moved here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
# spot_transformer is split into subpackages (core / models / eval / viz / sweeps). Its modules
# still import each other by bare name (viz: `import embeddings`, `from aggregator import …`), so
# each subpackage dir goes on the path; the repo root covers the modules already migrated to
# absolute `pipeline.spot_transformer.*` imports (e.g. models/aggregator.py).
_ST = REPO_ROOT / "pipeline" / "spot_transformer"
sys.path.insert(0, str(REPO_ROOT))
for _sub in ("core", "models", "eval", "viz"):
    sys.path.insert(0, str(_ST / _sub))


# --------------------------------------------------------------------------- verdict vocabulary
# Defined once, in Python, and shipped to the page through /api/manifest — a second copy in the
# JS is how a slug ends up written two ways in one store.
#
# THREE verdicts, not two. "unsure" is not a cop-out to be discouraged: the deliverable is a
# calibration set for a match/new/**abstain** head, so a pair the human cannot call is a labelled
# example of the abstain class, and forcing it into match/different would poison both.
VERDICTS = [
    ("match",     "Same animal",   "1"),
    ("different", "Different",     "2"),
    ("unsure",    "Can't tell",    "3"),
]

# Why, in the reviewer's own terms. The first six are the recurring rejection reasons the manual
# review surfaced (see strict_match's module docstring); `multi-animal` and `quality` are frame
# problems rather than match problems and route to the pre-filter work instead of the matcher.
REASONS = [
    ("posture",      "Posture / curl differs"),
    ("layout",       "Global spot layout disagrees"),
    ("missing-spot", "Characteristic spot has no counterpart"),
    ("weak-spots",   "Spots too round to decide"),
    ("few-matches",  "Too few corresponding spots"),
    ("quality",      "Photo too poor to judge"),
    ("multi-animal", "More than one animal in frame"),
]
_VERDICT_SLUGS = {v for v, _, _ in VERDICTS}
_REASON_SLUGS = {r for r, _ in REASONS}


def _reconfigure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def out_dir_for(dataset: str) -> Path:
    return REPO_ROOT / "artifacts" / "pair_review" / dataset


# --------------------------------------------------------------------------- generate

def _slices(rows: list, n: int) -> list[tuple[str, list]]:
    """Top-N / middle-N / bottom-N of a score-sorted list — the three bands the grid shows."""
    n = min(n, len(rows))
    mid = (len(rows) - n) // 2
    return [("TOP", rows[:n]), ("MID", rows[mid:mid + n]), ("BOT", rows[-n:] if n else [])]


def _slice_random(rows: list, n: int, seed: int) -> list[tuple[str, list]]:
    """A uniform random sample of pairs, as the single band ``RANDOM``.

    The other two samplers are stratified by score, which is right for error analysis and **wrong
    for measuring anything about the score itself**: results.md #50 got AUROC 0.412 / 0.342 / 0.129
    on three bands and could not pool them, because the strata were chosen by the very quantity
    under test. Conditioning on score and then asking how score behaves answers a different
    question than it appears to.

    Deterministic given ``seed``, so a pass is reproducible and can be extended without
    re-rendering (the same seed and a larger ``n`` is NOT nested — record the seed AND the n).
    """
    import random                                                  # noqa: PLC0415
    n = min(n, len(rows))
    return [("RANDOM", random.Random(seed).sample(list(rows), n))]


def _slices_by_correctness(rows: list, n_correct: int, n_wrong: int, n_mid: int
                           ) -> list[tuple[str, list]]:
    """Error-analysis banding: most-confident RIGHT, most-confident WRONG, and a middle sample.

    ``rows`` is ``(query, matched, score, correct)`` sorted by score descending. Splitting on
    correctness is the point: the plain TOP/MID/BOT banding buries the failures that matter,
    because a confident WRONG match is the one that silently merges two animals in a census,
    while a confident right one only tells you the easy cases are easy. Asking for more wrong
    than right is the correct asymmetry for reviewing a matcher.
    """
    good = [r for r in rows if r[3]]
    bad = [r for r in rows if not r[3]]
    mid_start = max(0, (len(rows) - n_mid) // 2)
    return [("CONFIDENT-CORRECT", good[:n_correct]),
            ("CONFIDENT-WRONG", bad[:n_wrong]),
            ("MID", rows[mid_start:mid_start + n_mid])]


class Residuals:
    """Per-spot "what did the score charge for", for the pair panels.

    A match panel can only draw what DID match, which is the half of the evidence that looks the
    same on a true pair and on a plausible-looking wrong one. What separates them is the pattern
    with no counterpart — so this computes ``strict_match.residual_detail`` for a pair and hands
    the panel (a) the scorer's own one-to-one assignment to draw as arcs and (b) the contradicted
    spots to fill in magenta. Weights are the training-free six-factor distinctiveness over the
    whole loaded population, so no model or human labels are needed to render a review.
    """

    def __init__(self, spots, *, sigma_pos: float = 0.12, match_thr: float = 0.4):
        import numpy as np                                        # noqa: PLC0415
        import strict_match as sm                                 # noqa: PLC0415
        self.sm, self.np = sm, np
        self.sigma_pos, self.match_thr = sigma_pos, match_thr
        emb = np.vstack(spots["embedding"].to_numpy())
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        w = sm.distinctiveness(spots, emb)
        self.w = {(s, int(i)): float(x)
                  for s, i, x in zip(spots["salamander_id"], spots["spot_id"], w)}
        self.w_ref = float(np.percentile(w, 90))

    def _side(self, sub):
        np = self.np
        e = np.vstack(sub["embedding"].to_numpy())
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        L = sub["length_px"].to_numpy(float)
        t = np.nan_to_num(sub["axis_t"].to_numpy(float), nan=0.5)
        off = np.nan_to_num(sub["axis_offset"].to_numpy(float), nan=0.0)
        lat = np.divide(off, L, out=np.zeros(len(sub)), where=np.isfinite(L) & (L > 0))
        w = np.array([self.w.get((s, int(i)), 0.5)
                      for s, i in zip(sub["salamander_id"], sub["spot_id"])], float)
        return e, w, np.column_stack([t, lat]), np.column_stack(
            [sub["global_centroid_x"].to_numpy(float), sub["global_centroid_y"].to_numpy(float)])

    def for_pair(self, spots, q: str, m: str):
        """``(pairs, mark, note)`` — arcs to draw, spots to flag, and a one-line breakdown."""
        s1 = spots[spots.salamander_id == q].reset_index(drop=True)
        s2 = spots[spots.salamander_id == m].reset_index(drop=True)
        if not len(s1) or not len(s2):
            return None, None, ""
        e1, w1, xy1, c1 = self._side(s1)
        e2, w2, xy2, c2 = self._side(s2)
        det = self.sm.residual_detail(e1, e2, w1, w2, xyq=xy1, xyc=xy2,
                                      sigma_pos=self.sigma_pos, match_thr=self.match_thr,
                                      geom_xyq=c1, geom_xyc=c2)
        if det is None:
            return None, None, ""
        pairs = [(int(i), int(j), float(s))
                 for i, j, s in zip(det["assign_q"], det["assign_c"], det["assign_quality"])]
        mark = {q: {int(sid): float(r) / max(self.w_ref, 1e-9)
                    for sid, r in zip(s1["spot_id"], det["res_q"])},
                m: {int(sid): float(r) / max(self.w_ref, 1e-9)
                    for sid, r in zip(s2["spot_id"], det["res_c"])}}
        E_mass = float((det["wq"] * det["qual_q"]).sum() + (det["wc"] * det["qual_c"]).sum())
        C_mass = float(det["res_q"].sum() + det["res_c"].sum())
        worst = max(float(det["res_q"].max()), float(det["res_c"].max()))
        note = (f"explained {E_mass:.1f}  vs  contradicted {C_mass:.1f}   →   "
                f"evidence {E_mass / (E_mass + C_mass + 1e-9):.2f}    ·    "
                f"{det['n_good']} good matches    ·    worst unanswered spot "
                f"{worst / max(self.w_ref, 1e-9):.2f}")
        return pairs, mark, note


def _render_pair_png(E, spots, q: str, m: str, score: float, correct: bool,
                     out_path: Path, *, cutoff: float, method: str,
                     figw: float, figh: float, dpi: int, residuals: "Residuals | None" = None
                     ) -> None:
    """One full-figure PNG: match arcs across the top, both size maps below. Never raises for a
    single bad panel — it draws an error note in that panel and still writes the file, so the
    manifest and the images stay in lock-step."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(figw, figh))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.4], hspace=0.07, wspace=0.04,
                          left=0.02, right=0.98, top=0.89, bottom=0.03)
    ax_match = fig.add_subplot(gs[0, :])
    ax_q = fig.add_subplot(gs[1, 0])
    ax_m = fig.add_subplot(gs[1, 1])

    pairs = marks = None
    note = ""
    if residuals is not None:
        try:
            pairs, marks, note = residuals.for_pair(spots, q, m)
        except Exception as exc:                                 # diagnostics must never block a render
            note = f"(residuals unavailable: {type(exc).__name__}: {exc})"

    mark = "OK  (same individual)" if correct else "WRONG  (different individuals)"
    fig.suptitle(f"{q}   →   {m}      score = {score:.2f}      [{mark}]",
                 fontsize=18, fontweight="bold", y=0.985,
                 color=("tab:green" if correct else "tab:red"))
    if note:                                  # between the suptitle and the match panel's own title
        fig.text(0.5, 0.955, note, ha="center", va="top", fontsize=11, color="0.3")

    try:
        E.visualize_spot_matches(spots, q, m, cutoff=cutoff, method=method, ax=ax_match,
                                 emb_col="embedding", pairs=pairs, mark=marks)
    except Exception as exc:                                     # never let one panel kill the file
        ax_match.text(0.5, 0.5, f"match panel failed:\n{exc}", ha="center", va="center",
                      fontsize=11, transform=ax_match.transAxes)
        ax_match.axis("off")
    for ax, sid, side in ((ax_q, q, "query"), (ax_m, m, "matched")):
        try:
            E.visualize_spot_sizes(spots, sid, ax=ax, colorbar=False)
            ax.set_title(f"{side}: {sid}", fontsize=13)
        except Exception as exc:
            ax.text(0.5, 0.5, f"{sid}\n{exc}", ha="center", va="center", fontsize=11,
                    transform=ax.transAxes)
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _bakeoff_sources(run_dir: Path) -> list[tuple[str, list[tuple]]]:
    """Load a ``sweep_bakeoff`` run's ranked pairs -> ``[(method_name, rows), ...]``.

    ``rows`` matches what ``visualize_match_errors.rank_for`` returns — ``(query_sid,
    matched_sid, score, correct)``, sorted by score descending — so the banding, dedupe and
    rendering below are shared with the normal path. Method names are ``<flow>__<matcher>``
    (e.g. ``tensor__logreg``), and because the same pair can come from several of them, the
    review UI shows every matcher that produced it in the pair's ``sources``.
    """
    pair_dir = run_dir / "ranked_pairs"
    if not pair_dir.is_dir():
        raise FileNotFoundError(f"no ranked_pairs/ under {run_dir} — run the bakeoff first")
    out = []
    for path in sorted(pair_dir.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        rows = [(r["query"], r["matched"], float(r["score"]), bool(r["correct"]))
                for r in blob.get("rows", [])]
        rows.sort(key=lambda t: -t[2])
        if rows:
            out.append((path.stem, rows))
            print(f"  {path.stem:28s} {len(rows):>4} pairs   R@1 {blob.get('r_at_1', float('nan')):.3f}")
    if not out:
        raise ValueError(f"no ranked pairs found in {pair_dir}")
    return out


def cmd_generate(args: argparse.Namespace) -> int:
    import embeddings as E                                       # noqa: PLC0415
    import visualize_match_errors as V                           # noqa: PLC0415

    dataset = E.dataset_name
    out_dir = out_dir_for(dataset)
    img_dir = out_dir / "images"

    if args.from_bakeoff:
        sources = _bakeoff_sources(args.from_bakeoff)
        methods = [name for name, _ in sources]
        print(f"bakeoff: {args.from_bakeoff}")
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        sources = None
    by_correctness = any(v is not None for v in (args.n_correct, args.n_wrong, args.n_mid))
    if args.random and by_correctness:
        print("error: --random and --n-correct/--n-wrong/--n-mid are different sampling designs; "
              "pick one", file=sys.stderr)
        return 2
    if args.random:
        print(f"dataset: {dataset}   methods: {methods}   "
              f"UNSTRATIFIED random sample: {args.random} pairs (seed {args.seed})")
    elif by_correctness:
        args.n_correct = 5 if args.n_correct is None else args.n_correct
        args.n_wrong = 20 if args.n_wrong is None else args.n_wrong
        args.n_mid = 10 if args.n_mid is None else args.n_mid
        print(f"dataset: {dataset}   methods: {methods}   bands: "
              f"{args.n_correct} correct / {args.n_wrong} wrong / {args.n_mid} mid")
    else:
        print(f"dataset: {dataset}   methods: {methods}   n/band: {args.n}")
    print(f"output : {out_dir}")

    spots = V.load_spots()

    seen: dict[tuple[str, str], int] = {}
    items: list[dict] = []
    for mi, method in enumerate(methods):
        if sources is not None:
            rows = sources[mi][1]
        else:
            try:
                rows, _arc_col = V.rank_for(spots, method)
            except Exception as exc:                             # a slow/learned method may fail
                print(f"  [{method}] SKIPPED: {type(exc).__name__}: {exc}")
                continue
        if args.random:
            bands = _slice_random(rows, args.random, args.seed)
        elif by_correctness:
            bands = _slices_by_correctness(rows, args.n_correct, args.n_wrong, args.n_mid)
        else:
            bands = _slices(rows, args.n)
        for band, band_rows in bands:
            for q, m, s, c in band_rows:
                key = (q, m)
                src = {"method": method, "band": band, "score": round(float(s), 4),
                       "correct": bool(c)}
                if key in seen:
                    items[seen[key]]["sources"].append(src)
                    continue
                seen[key] = len(items)
                items.append({"id": f"{q}__{m}", "query": q, "matched": m,
                              "correct": bool(c), "score": round(float(s), 4),
                              "sources": [src], "image": f"{q}__{m}.png"})

    if not items:
        print("no pairs to render — nothing ranked.")
        return 1

    residuals = None
    if not args.no_residual:
        print("computing per-spot distinctiveness for the residual overlay ...", flush=True)
        residuals = Residuals(spots, sigma_pos=args.sigma_pos, match_thr=args.match_thr)

    print(f"\nrendering {len(items)} unique pairs -> {img_dir}")
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items, 1):
        _render_pair_png(E, spots, it["query"], it["matched"], it["score"], it["correct"],
                         img_dir / it["image"], cutoff=args.cutoff, method=args.match_method,
                         figw=args.figw, figh=args.figh, dpi=args.dpi, residuals=residuals)
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    # A regenerate REPLACES manifest.json and review.json, and those two carry the sampling design
    # (`sources[].band`) that the verdicts can only be interpreted against — verdicts.json alone
    # cannot tell you a pair came from CONFIDENT-WRONG. Switching designs (banded -> --random) would
    # therefore silently strand an already-finished pass, so the previous pair of files is archived
    # first. Cheap: they are small JSON, and the alternative is unrecoverable.
    prev = out_dir / "review.json"
    if prev.is_file():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        arch = out_dir / "archive" / stamp
        arch.mkdir(parents=True, exist_ok=True)
        for name in ("review.json", "manifest.json"):
            if (out_dir / name).is_file():
                (arch / name).write_bytes((out_dir / name).read_bytes())
        print(f"  archived the previous pass -> {arch}")

    manifest = {"dataset": dataset, "methods": methods, "n_per_band": args.n,
                "sampling": ("random" if args.random else
                             "by_correctness" if by_correctness else "score_bands"),
                "random_n": args.random, "seed": args.seed,
                "cutoff": args.cutoff, "match_method": args.match_method,
                "bakeoff": str(args.from_bakeoff) if args.from_bakeoff else None,
                "items": items}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Regenerating re-renders the PNGs and rewrites review.json; it must NOT drop hand labels, so
    # both stores are read back in. Pairs that survive the new ranking keep their verdict (the id
    # is <query>__<matched>, stable across runs); pairs that fall out of it keep their entry in
    # verdicts.json and simply stop appearing in review.json.
    comments_path = out_dir / "comments.json"
    if not comments_path.exists():
        comments_path.write_text("{}", encoding="utf-8")
    verdicts_path = out_dir / "verdicts.json"
    verdicts = (json.loads(verdicts_path.read_text(encoding="utf-8"))
                if verdicts_path.is_file() else {})
    _write_review(out_dir, items, json.loads(comments_path.read_text(encoding="utf-8")), verdicts)

    kept = sum(1 for it in items if it["id"] in verdicts)
    print(f"\nwrote manifest.json ({len(items)} pairs"
          + (f", {kept}/{len(verdicts)} existing verdicts still in range" if verdicts else "")
          + ").  Now:  pixi run pair-review")
    return 0


# --------------------------------------------------------------------------- serve

def _write_review(out_dir: Path, items: list[dict], comments: dict[str, str],
                  verdicts: dict[str, dict] | None = None) -> None:
    """review.json = every pair with its verdict and comment inline (the file you actually review).

    ``disagrees_with_label`` is computed here rather than left to every consumer: ``correct`` comes
    from the filenames and ``verdict`` from a person looking at the two animals, so the rows where
    they conflict are exactly the suspected label errors (open thread #4). It is ``None`` — not
    ``False`` — when there is no verdict or the verdict is ``unsure``, because "nobody looked" and
    "the human agrees" must not collapse into the same value.
    """
    verdicts = verdicts or {}
    review = []
    for it in items:
        v = verdicts.get(it["id"]) or {}
        slug = v.get("verdict")
        disagrees = None
        if slug in ("match", "different"):
            disagrees = (slug == "match") != bool(it["correct"])
        review.append({
            **{k: it[k] for k in ("id", "query", "matched", "correct", "score", "sources")},
            "verdict": slug, "reasons": v.get("reasons") or [], "verdict_at": v.get("at"),
            "disagrees_with_label": disagrees,
            "comment": comments.get(it["id"], ""),
        })
    _atomic_write(out_dir / "review.json", json.dumps(review, indent=2))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ReviewApp:
    """Loads the generated pairs + any saved comments; persists a comment per pair."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.img_dir = out_dir / "images"
        manifest_path = out_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest at {manifest_path} — run:  pixi run pair-review-gen")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.items: list[dict] = self.manifest["items"]
        self.by_id = {it["id"]: it for it in self.items}
        self.comments_path = out_dir / "comments.json"
        self.comments: dict[str, str] = (
            json.loads(self.comments_path.read_text(encoding="utf-8"))
            if self.comments_path.is_file() else {})
        # Verdicts live in their OWN file rather than beside the comments: comments.json predates
        # them and other things read it, so widening its value type from str to dict would break
        # every existing reader for no gain.
        self.verdicts_path = out_dir / "verdicts.json"
        self.verdicts: dict[str, dict] = (
            json.loads(self.verdicts_path.read_text(encoding="utf-8"))
            if self.verdicts_path.is_file() else {})
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return len(self.items)

    def manifest_payload(self) -> dict:
        return {"dataset": self.manifest.get("dataset"), "items": self.items,
                "comments": self.comments, "verdicts": self.verdicts,
                # the page renders its buttons and chips from these, so the slugs it POSTs back
                # cannot drift from the ones validated in set_verdict
                "vocab": {"verdicts": [{"slug": s, "label": l, "key": k} for s, l, k in VERDICTS],
                          "reasons": [{"slug": s, "label": l} for s, l in REASONS]}}

    def image_path(self, i: int) -> Path | None:
        if 0 <= i < self.total:
            return self.img_dir / self.items[i]["image"]
        return None

    def set_comment(self, cid: str, text: str) -> dict:
        if cid not in self.by_id:
            raise KeyError(cid)
        text = (text or "").strip()
        with self._lock:
            if text:
                self.comments[cid] = text
            else:
                self.comments.pop(cid, None)
            _atomic_write(self.comments_path, json.dumps(self.comments, indent=2))
            _write_review(self.out_dir, self.items, self.comments, self.verdicts)
            return {"ok": True, "id": cid, "n_commented": len(self.comments)}

    def set_verdict(self, cid: str, verdict: str | None, reasons: list | None = None) -> dict:
        """Record (or clear, with ``verdict=None``) one pair's verdict + reasons.

        Unknown slugs are **rejected**, not coerced or stored: this file is training data, and a
        typo'd verdict that silently persists is a mislabelled example nobody will ever find.
        """
        if cid not in self.by_id:
            raise KeyError(cid)
        slugs = [str(r) for r in (reasons or [])]
        bad = sorted(set(slugs) - _REASON_SLUGS)
        if bad:
            raise ValueError(f"unknown reason(s) {bad}; expected one of {sorted(_REASON_SLUGS)}")
        if verdict is not None and verdict not in _VERDICT_SLUGS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {sorted(_VERDICT_SLUGS)}")
        with self._lock:
            if verdict is None:
                self.verdicts.pop(cid, None)
            else:
                self.verdicts[cid] = {
                    "verdict": verdict,
                    "reasons": [r for r, _ in REASONS if r in set(slugs)],   # canonical order
                    "at": datetime.now().isoformat(timespec="seconds"),
                }
            _atomic_write(self.verdicts_path, json.dumps(self.verdicts, indent=2))
            _write_review(self.out_dir, self.items, self.comments, self.verdicts)
            return {"ok": True, "id": cid, "n_verdicts": len(self.verdicts)}


# The whole UI is one page — kept inline so the tool is a single file, like a "small simple app".
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pair review</title>
<style>
  :root { --bg:#111; --bar:#1c1c1f; --fg:#eee; --dim:#9aa; --ok:#3fbf6f; --wrong:#e0554c; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
               font-family: system-ui, sans-serif; }
  #app { display:flex; flex-direction:column; height:100vh; }
  #stage { flex:1 1 auto; min-height:0; display:flex; align-items:center; justify-content:center;
           overflow:hidden; }
  #img { max-width:100%; max-height:100%; object-fit:contain; }
  #bar { flex:0 0 auto; background:var(--bar); border-top:1px solid #333; padding:10px 14px;
         display:flex; gap:14px; align-items:stretch; }
  #meta { flex:0 0 260px; font-size:13px; line-height:1.5; color:var(--dim); overflow:hidden; }
  #meta .pair { color:var(--fg); font-weight:700; font-size:15px; word-break:break-all; }
  #meta .verdict.ok { color:var(--ok); } #meta .verdict.wrong { color:var(--wrong); }
  #meta .src { font-size:11px; color:#889; }
  #comment { flex:1 1 auto; resize:none; background:#0d0d0f; color:var(--fg);
             border:1px solid #3a3a40; border-radius:6px; padding:10px 12px; font-size:15px;
             font-family:inherit; }
  #comment:focus { outline:2px solid #4a7; border-color:#4a7; }
  #label { flex:0 0 300px; display:flex; flex-direction:column; gap:6px; min-width:0; }
  #verdicts { display:flex; gap:6px; }
  #verdicts button { flex:1 1 0; padding:8px 4px; font-size:13px; }
  /* the selected verdict is filled, not merely outlined: at a glance across a long pass, "did I
     already call this one" has to be answerable without reading */
  #verdicts button.on[data-v="match"]     { background:var(--ok);    border-color:var(--ok);
                                            color:#07210f; font-weight:700; }
  #verdicts button.on[data-v="different"] { background:var(--wrong); border-color:var(--wrong);
                                            color:#2a0906; font-weight:700; }
  #verdicts button.on[data-v="unsure"]    { background:var(--dim);   border-color:var(--dim);
                                            color:#111; font-weight:700; }
  #reasons { display:flex; flex-wrap:wrap; gap:4px; overflow:auto; }
  .chip { background:#26262c; color:var(--dim); border:1px solid #3a3a40; border-radius:11px;
          padding:2px 9px; font-size:11px; cursor:pointer; white-space:nowrap; }
  .chip:hover { border-color:#666; }
  .chip.on { background:#3a3a46; color:var(--fg); border-color:#7a7a8a; }
  #side { flex:0 0 auto; display:flex; flex-direction:column; justify-content:space-between;
          gap:8px; }
  #nav { display:flex; gap:8px; }
  button { background:#2a2a30; color:var(--fg); border:1px solid #444; border-radius:6px;
           padding:8px 16px; font-size:15px; cursor:pointer; }
  button:hover { background:#35353d; }
  #counter { font-size:12px; color:var(--dim); text-align:right; }
  #counter a { color:#6bf; }
  #saved { font-size:11px; color:var(--ok); height:14px; text-align:right; }
  /* the filenames say one thing and the reviewer said the other — a suspected label error, which
     is a finding in its own right (open thread #4), so it gets its own loud line */
  #m-conflict { color:#ffb454; font-size:12px; font-weight:700; }
</style>
</head>
<body>
<div id="app">
  <div id="stage"><img id="img" alt="pair"></div>
  <div id="bar">
    <div id="meta">
      <div class="pair" id="m-pair">—</div>
      <div class="verdict" id="m-verdict"></div>
      <div id="m-conflict"></div>
      <div class="src" id="m-src"></div>
    </div>
    <div id="label">
      <div id="verdicts"></div>
      <div id="reasons"></div>
    </div>
    <textarea id="comment" placeholder="Optional note (what's the problem / how to fix it)…
1/2/3 = verdict. ←/→ to move. U = next unlabelled."></textarea>
    <div id="side">
      <div id="nav">
        <button id="prev" title="Previous (←)">← Prev</button>
        <button id="next" title="Next (→)">Next →</button>
      </div>
      <div id="counter"></div>
      <div id="saved"></div>
    </div>
  </div>
</div>
<script>
let items = [], comments = {}, verdicts = {}, vocab = { verdicts: [], reasons: [] };
let i = 0, saveTimer = null;

async function boot() {
  const r = await fetch('/api/manifest');
  const d = await r.json();
  items = d.items || [];
  comments = d.comments || {};
  verdicts = d.verdicts || {};
  vocab = d.vocab || vocab;
  buildLabelUI();
  if (!items.length) { document.getElementById('m-pair').textContent = 'no pairs — run pair-review-gen'; return; }
  render();
}

/** Buttons and chips are built from the server's vocabulary, never hardcoded here. */
function buildLabelUI() {
  const vb = document.getElementById('verdicts');
  vb.innerHTML = '';
  for (const v of vocab.verdicts) {
    const b = document.createElement('button');
    b.textContent = `${v.key}  ${v.label}`;
    b.dataset.v = v.slug;
    b.title = `${v.label} (key ${v.key}) — press again to clear`;
    b.onclick = () => setVerdict(v.slug);
    vb.appendChild(b);
  }
  const rb = document.getElementById('reasons');
  rb.innerHTML = '';
  for (const r of vocab.reasons) {
    const c = document.createElement('span');
    c.className = 'chip';
    c.textContent = r.label;
    c.dataset.r = r.slug;
    c.onclick = () => toggleReason(r.slug);
    rb.appendChild(c);
  }
}

function render() {
  const it = items[i];
  document.getElementById('img').src = '/img?i=' + i + '&t=' + Date.now();
  document.getElementById('m-pair').textContent = it.query + '  →  ' + it.matched;
  const v = document.getElementById('m-verdict');
  v.textContent = it.correct ? 'labelled: same individual' : 'labelled: different individuals';
  v.className = 'verdict ' + (it.correct ? 'ok' : 'wrong');
  document.getElementById('m-src').textContent =
    (it.sources || []).map(s => `${s.method}/${s.band} ${Number(s.score).toFixed(2)}`).join('  ·  ');
  const box = document.getElementById('comment');
  box.value = comments[it.id] || '';
  renderLabel();
  updateCounter();
  document.getElementById('saved').textContent = '';
}

function renderLabel() {
  const it = items[i];
  const cur = verdicts[it.id] || {};
  for (const b of document.querySelectorAll('#verdicts button'))
    b.classList.toggle('on', b.dataset.v === cur.verdict);
  const rs = new Set(cur.reasons || []);
  for (const c of document.querySelectorAll('#reasons .chip'))
    c.classList.toggle('on', rs.has(c.dataset.r));
  // The conflict line is the whole reason the ground-truth label stays on screen: the reviewer is
  // judging the pixels, and where they disagree with the filenames the FILENAMES are the suspect.
  const conflict = document.getElementById('m-conflict');
  const disagrees = (cur.verdict === 'match' || cur.verdict === 'different')
    && ((cur.verdict === 'match') !== !!it.correct);
  conflict.textContent = disagrees ? '⚠ disagrees with the filename label — suspected label error' : '';
}

function updateCounter() {
  const n = Object.keys(verdicts).length, c = Object.keys(comments).length;
  document.getElementById('counter').innerHTML =
    `#${i + 1} / ${items.length} &nbsp;·&nbsp; <b>${n} judged</b> &nbsp;·&nbsp; ${c} noted ` +
    `&nbsp;·&nbsp; <a href="/review.json" target="_blank">review.json</a>`;
}

/** Pressing the verdict already set clears it — the same key is set and undo. */
async function setVerdict(slug) {
  const it = items[i];
  const cur = verdicts[it.id] || {};
  const next = cur.verdict === slug ? null : slug;
  if (next === null) delete verdicts[it.id];
  else verdicts[it.id] = { verdict: next, reasons: cur.reasons || [] };
  renderLabel(); updateCounter();
  await postVerdict(it.id, next, (verdicts[it.id] || {}).reasons || []);
}

async function toggleReason(slug) {
  const it = items[i];
  const cur = verdicts[it.id];
  // A reason with no verdict is unanchored ("posture differs" — and so?), so it needs the call
  // first. Flash the buttons rather than silently doing nothing.
  if (!cur) {
    const vb = document.getElementById('verdicts');
    vb.style.outline = '2px solid #ffb454';
    setTimeout(() => { vb.style.outline = ''; }, 500);
    return;
  }
  const rs = new Set(cur.reasons || []);
  rs.has(slug) ? rs.delete(slug) : rs.add(slug);
  cur.reasons = [...rs];
  renderLabel();
  await postVerdict(it.id, cur.verdict, cur.reasons);
}

async function postVerdict(id, verdict, reasons) {
  try {
    const r = await fetch('/api/verdict', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, verdict, reasons }) });
    if (!r.ok) throw new Error((await r.json()).error || r.status);
    const s = document.getElementById('saved');
    s.textContent = 'saved ✓'; setTimeout(() => { s.textContent = ''; }, 1200);
  } catch (e) { document.getElementById('saved').textContent = 'save failed: ' + e.message; }
}

async function save(showFlag) {
  const it = items[i];
  const text = document.getElementById('comment').value;
  const had = comments[it.id] || '';
  if (text.trim() === had.trim()) return;
  if (text.trim()) comments[it.id] = text.trim(); else delete comments[it.id];
  updateCounter();
  try {
    await fetch('/api/comment', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: it.id, comment: text }) });
    if (showFlag) { const s = document.getElementById('saved'); s.textContent = 'saved ✓';
      setTimeout(() => { s.textContent = ''; }, 1200); }
  } catch (e) { document.getElementById('saved').textContent = 'save failed!'; }
}

async function go(delta) {
  await save(false);
  i = (i + delta + items.length) % items.length;
  render();
}

/** Next pair with no VERDICT — the verdict is the label now, so it is what "unlabelled" means. */
function nextUnjudged() {
  for (let k = 1; k <= items.length; k++) {
    const j = (i + k) % items.length;
    if (!verdicts[items[j].id]) { save(false).then(() => { i = j; render(); }); return; }
  }
  document.getElementById('saved').textContent = 'all judged ✓';
}

document.getElementById('prev').onclick = () => go(-1);
document.getElementById('next').onclick = () => go(1);

document.getElementById('comment').addEventListener('input', () => {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => save(true), 600);
});

document.addEventListener('keydown', (e) => {
  const typing = document.activeElement === document.getElementById('comment');
  if (typing && !(e.key === 'Enter' && (e.ctrlKey || e.metaKey))) return;
  if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
  else if (e.key === 'u' || e.key === 'U') { e.preventDefault(); nextUnjudged(); }
  else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); save(true).then(() => go(1)); }
  else {
    // A verdict key judges AND advances: the pass is hundreds of pairs long and the second
    // keystroke to move on is most of the cost per label.
    const v = vocab.verdicts.find(x => x.key === e.key);
    if (v) { e.preventDefault(); setVerdict(v.slug).then(() => { if (verdicts[items[i].id]) go(1); }); }
  }
});

window.addEventListener('beforeunload', () => {
  const it = items[i]; if (!it) return;
  const text = document.getElementById('comment').value;
  if ((comments[it.id] || '').trim() === text.trim()) return;
  navigator.sendBeacon('/api/comment',
    new Blob([JSON.stringify({ id: it.id, comment: text })], { type: 'application/json' }));
});

boot();
</script>
</body>
</html>
"""


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, app: ReviewApp):
        super().__init__(addr, handler)
        self.app = app


class _Handler(BaseHTTPRequestHandler):
    server_version = "PairReview/1.0"

    def log_message(self, fmt, *args):  # noqa: A003 — keep the console quiet
        pass

    @property
    def app(self) -> ReviewApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _err(self, code: int, msg: str):
        self._json({"error": msg}, code)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif path == "/api/manifest":
                self._json(self.app.manifest_payload())
            elif path == "/img":
                self._img(q)
            elif path in ("/review.json", "/comments.json", "/verdicts.json"):
                fp = self.app.out_dir / path.lstrip("/")
                if fp.is_file():
                    self._send(200, fp.read_bytes(), "application/json")
                else:
                    self._err(404, "not generated yet")
            else:
                self._err(404, "not found")
        except Exception as exc:                                 # one bad request never stops serving
            self._err(500, str(exc))

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path not in ("/api/comment", "/api/verdict"):
            self._err(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/comment":
                self._json(self.app.set_comment(str(body["id"]), body.get("comment", "")))
            else:
                v = body.get("verdict")
                self._json(self.app.set_verdict(str(body["id"]),
                                                None if v is None else str(v),
                                                body.get("reasons") or []))
        except KeyError as exc:
            self._err(404, f"unknown pair {exc}")
        except Exception as exc:
            self._err(400, str(exc))

    def _img(self, q):
        try:
            i = int(q.get("i", ["-1"])[0])
        except ValueError:
            self._err(400, "bad index")
            return
        path = self.app.image_path(i)
        if path is None or not path.is_file():
            self._err(404, f"no image for index {i}")
            return
        self._send(200, path.read_bytes(), "image/png", cache=False)


def cmd_serve(args: argparse.Namespace) -> int:
    import embeddings as E                                       # noqa: PLC0415 — for dataset_name
    out_dir = out_dir_for(args.dataset or E.dataset_name)
    try:
        app = ReviewApp(out_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    httpd = _Server((args.host, args.port), _Handler, app)
    url = f"http://{args.host}:{args.port}/"
    print(f"pair-review  ->  {url}")
    print(f"  pairs    : {app.total}  ({len(app.verdicts)} judged, {len(app.comments)} noted)")
    print(f"  verdicts : {app.verdicts_path}")
    print(f"  comments : {app.comments_path}")
    print(f"  review   : {out_dir / 'review.json'}")
    print("  keys     : " + " · ".join(f"{k} {lbl.lower()}" for _, lbl, k in VERDICTS)
          + " (judges + advances) · ←/→ prev/next · U next-unjudged · Ctrl+Enter save note+next.")
    print("             Ctrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        httpd.shutdown()
    return 0


# --------------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    _reconfigure_utf8()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", aliases=["gen"], help="render one big PNG per pair + manifest")
    g.add_argument("--methods", default="raw",
                   help="comma list of match algorithms whose top/mid/bottom pairs to include "
                        "(default: raw). Learned matchers (logreg, mlp_deep, axial_cnn, "
                        "set_transformer, deepsets_cons, e2e_pretrained, e2e_transformer, "
                        "pretrained_cnn) train first and are slower.")
    g.add_argument("--from-bakeoff", type=Path, default=None, metavar="RUN_DIR",
                   help="review a sweep_bakeoff run instead of re-ranking here: reads "
                        "RUN_DIR/ranked_pairs/*.json (one per embedding x matcher) and treats "
                        "each file as a method, so concat vs tensor sit side by side on the "
                        "same pair. --methods is ignored.")
    g.add_argument("--n", type=int, default=8, help="pairs per band (top-N, mid-N, bottom-N)")
    g.add_argument("--random", type=int, default=None, metavar="N",
                   help="UNSTRATIFIED: sample N pairs uniformly instead of banding by score. "
                        "The only design that can measure the score itself (results.md #50) — "
                        "every other mode picks pairs BY score. Mutually exclusive with "
                        "--n-correct/--n-wrong/--n-mid.")
    g.add_argument("--seed", type=int, default=0,
                   help="seed for --random; record it with the n, the samples are not nested")
    # Error-analysis banding. Setting ANY of these switches from score bands (top/mid/bottom) to
    # correctness bands (most-confident right / most-confident WRONG / a middle sample), which is
    # what you want when reviewing a matcher: a confident wrong match is the costly failure.
    g.add_argument("--n-correct", type=int, default=None, metavar="N",
                   help="most-confident CORRECT matches to include (default 5 when any "
                        "--n-correct/--n-wrong/--n-mid is given)")
    g.add_argument("--n-wrong", type=int, default=None, metavar="N",
                   help="most-confident INCORRECT matches — the dangerous errors (default 20)")
    g.add_argument("--n-mid", type=int, default=None, metavar="N",
                   help="mid-confidence matches, the decision boundary (default 10)")
    g.add_argument("--cutoff", type=float, default=0.5, help="min cosine for a drawn match arc")
    g.add_argument("--match-method", default="mutual",
                   choices=("mutual", "hungarian", "threshold"))
    g.add_argument("--figw", type=float, default=16.0, help="figure width in inches")
    g.add_argument("--figh", type=float, default=11.0, help="figure height in inches")
    g.add_argument("--dpi", type=int, default=110, help="PNG resolution (110 -> ~1760x1210)")
    # Residual overlay: which spots the strict scorer PENALIZED (magenta), and the arcs it
    # actually used. On by default — the unmatched pattern is the half of the evidence the arcs
    # cannot show, and it is what a reviewer is being asked to judge.
    g.add_argument("--no-residual", action="store_true",
                   help="draw only the match arcs (no magenta unmatched-spot overlay)")
    g.add_argument("--sigma-pos", type=float, default=0.12,
                   help="residual overlay: body-frame position-gate width (body lengths)")
    g.add_argument("--match-thr", type=float, default=0.4,
                   help="residual overlay: min gated similarity for an assigned match")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("serve", help="open the review web app")
    s.add_argument("--dataset", default=None, help="override dataset folder (default: current)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
