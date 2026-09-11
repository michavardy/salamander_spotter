"""Launch the e2e_transformer exploration as a RunPod batch (one pod per study slice).

Ships ``pipeline/`` as a module, mounts the ``contours.db`` network volume, uploads the
two small data files, and runs ``e2e_transform_explore.py`` on each pod with a different
``STUDY`` / ``CONFIG_SHARD``. Results are pulled back to
``artifacts/spot_transformer/sweeps/e2e_explore/``.

PREREQUISITES
  - a network volume holding ``contours.db`` at its root — create + fill with:
        cd ../runpod_runner
        .venv/Scripts/python -m runpod_runner.cli volume create --name salamander-spotter --size 10 --datacenter EU-RO-1
        .venv/Scripts/python -m runpod_runner.cli volume put <repo>/datasets/all_sasa_norm_2026_23_07/db/contours.db:contours.db --volume salamander-spotter
  - RunPod creds in ``runpod_runner/.env`` (RUNPOD_API_KEY, RUNPOD_SSH_KEY)

RUN  (use the runpod_runner venv, which has the `runpod` SDK):
    ../runpod_runner/.venv/Scripts/python pipeline/spot_transformer/sweeps/run_e2e_runpod.py \
        --volume salamander-spotter --max-concurrency 3

    --dry-run            print the plan, create nothing
    --studies a,b        only these study pods (default: all)
    --k-folds N          folds per config (default 5)
    --epoch-scale F      shrink every epoch budget (0.4 ~ a fast first pass)
    --rm-volume-after    delete the network volume once every pod finishes (stops billing)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]              # sweeps -> spot_transformer -> pipeline -> repo
_RUNPOD_SRC = REPO.parent / "runpod_runner" / "src"
if str(_RUNPOD_SRC) not in sys.path:
    sys.path.insert(0, str(_RUNPOD_SRC))

from runpod_runner import RunSpec, load_config                       # noqa: E402
from runpod_runner import registry                                   # noqa: E402
from runpod_runner.batch import run_batch                            # noqa: E402
from runpod_runner.volumes import delete_volume, find_volume         # noqa: E402
from runpod_runner.web_server import WebServer                       # noqa: E402

DATASET = "all_sasa_norm_2026_23_07"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# small, cheap cards first — the sweep model is tiny; create_pod cycles this list on capacity.
GPU = "NVIDIA RTX 2000 Ada Generation"
GPU_FALLBACKS = [
    "NVIDIA RTX A4000", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX A5000",
    "NVIDIA A40", "NVIDIA L40S", "NVIDIA GeForce RTX 4090",
]

# one pod per entry. Each re-runs `baseline` (5 fold-trains, cheap) as a self-check and to
# anchor its own RESULTS table; the identr study (24 configs) is split 3 ways.
STUDY_PODS: list[tuple[str, dict[str, str]]] = [
    ("identr-0", {"STUDY": "baseline,identr", "CONFIG_SHARD": "0/3"}),
    ("identr-1", {"STUDY": "baseline,identr", "CONFIG_SHARD": "1/3"}),
    ("identr-2", {"STUDY": "baseline,identr", "CONFIG_SHARD": "2/3"}),
    ("gate",     {"STUDY": "baseline,gate"}),
    ("hardneg",  {"STUDY": "baseline,hardneg"}),
    ("general",  {"STUDY": "baseline,general"}),
]

REMOTE_OUT = "/workspace/job/bundle/artifacts/spot_transformer/sweeps/e2e_explore"
LOCAL_OUT = REPO / "artifacts" / "spot_transformer" / "sweeps" / "e2e_explore"


def build_specs(args, experiment: str) -> list[RunSpec]:
    uploads = [
        (REPO / "datasets" / DATASET / "corrections.json",
         f"/workspace/job/bundle/datasets/{DATASET}/corrections.json"),
        (REPO / "images" / "all_sasa_norm" / "label_map.csv",
         "/workspace/job/bundle/images/all_sasa_norm/label_map.csv"),
    ]
    for local, _ in uploads:
        if not local.is_file():
            raise SystemExit(f"missing required upload: {local}")

    common = dict(
        requirements=REPO / "pipeline" / "spot_transformer" / "sweeps" / "runpod_reqs.txt",
        modules=[REPO / "pipeline"],
        startup_script=REPO / "pipeline" / "spot_transformer" / "sweeps" / "runpod_startup.sh",
        uploads=uploads,
        pulls=[(REMOTE_OUT, LOCAL_OUT)],
        experiment_id=experiment,
        budget_usd=args.budget,
    )
    base_env = {
        "MIN_QUALITY": "0.4", "SOURCE": "sasa", "DEVICE": "cuda",
        "K_FOLDS": str(args.k_folds),
    }
    if args.epoch_scale != 1.0:
        base_env["EPOCH_SCALE"] = str(args.epoch_scale)

    want = set(args.studies.split(",")) if args.studies else None
    specs = []
    for tag, env in STUDY_PODS:
        if want and tag not in want:
            continue
        specs.append(RunSpec(
            cmd="python bundle/pipeline/spot_transformer/sweeps/e2e_transform_explore.py",
            name=f"{experiment}-{tag}",
            env={**base_env, **env},
            metadata={"slice": tag, **env},
            **common,
        ))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", default="salamander-spotter", help="network volume id or name")
    ap.add_argument("--max-concurrency", type=int, default=3)
    ap.add_argument("--studies", default="", help="comma list of study-pod tags (default: all)")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--epoch-scale", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=3.0, help="USD cap per pod")
    ap.add_argument("--experiment", default="e2e-explore")
    ap.add_argument("--rm-volume-after", action="store_true",
                    help="delete the network volume when all pods finish (stops storage billing)")
    ap.add_argument("--web-ui", dest="web_ui", action="store_true", default=True)
    ap.add_argument("--no-web-ui", dest="web_ui", action="store_false",
                    help="disable the live dashboard (default: on)")
    ap.add_argument("--web-ui-port", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runpod_env = _RUNPOD_SRC.parent / ".env"
    cfg = load_config(
        image=IMAGE, gpu_type=GPU, gpu_type_fallbacks=GPU_FALLBACKS,
        dotenv_path=runpod_env if runpod_env.is_file() else None,
    )
    vol = find_volume(args.volume, api_key=cfg.api_key)
    if vol is None:
        raise SystemExit(f"network volume {args.volume!r} not found — create it and 'volume put' "
                         f"contours.db first (see this file's docstring)")
    cfg.network_volume_id = vol["id"]
    cfg.data_center_id = vol["dataCenterId"]

    specs = build_specs(args, args.experiment)

    print(f"experiment  {args.experiment}")
    print(f"volume      {vol['id']} ({vol['name']}, {vol['size']}GB in {vol['dataCenterId']})")
    print(f"image       {IMAGE}")
    print(f"gpu         {GPU}  (fallbacks: {', '.join(GPU_FALLBACKS)})")
    print(f"pods        {len(specs)}  ·  max-concurrency {args.max_concurrency}")
    for s in specs:
        print(f"  - {s.name:<22} {s.env.get('STUDY')}"
              + (f"  shard {s.env['CONFIG_SHARD']}" if 'CONFIG_SHARD' in s.env else ""))
    print(f"results ->  {LOCAL_OUT}")
    if args.dry_run:
        print("\n--dry-run: nothing launched")
        return

    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    web = None
    on_event = None
    if args.web_ui:
        registry.clear()
        web = WebServer(host="127.0.0.1", port=args.web_ui_port)
        web.start()
        on_event = web.publish
        print(f"\nweb UI      {web.url}\n")

    results = run_batch(specs, config=cfg, max_concurrency=args.max_concurrency,
                        on_event=on_event)

    print("\n=== summary ===")
    ok = 0
    for r in results:
        print(f"  {r.name:<24} {r.status:<8} rc={r.exit_code} {r.duration_s/60:.1f}m ${r.total_cost_usd:.2f}")
        ok += r.status == "success"
    print(f"  {ok}/{len(results)} pods succeeded  ·  ${sum(r.total_cost_usd for r in results):.2f} total")

    if args.rm_volume_after:
        if ok == len(results):
            delete_volume(vol["id"], api_key=cfg.api_key)
            print(f"deleted network volume {vol['id']} — storage billing stopped")
        else:
            print(f"NOT deleting volume {vol['id']}: {len(results)-ok} pod(s) did not succeed. "
                  f"Delete manually when done:  runpod-runner volume rm {vol['id']}")

    if web is not None:
        print(f"\nweb UI still serving at {web.url}  (Ctrl-C to stop)")
        try:
            while True:
                __import__("time").sleep(2)
        except KeyboardInterrupt:
            web.stop()


if __name__ == "__main__":
    main()
