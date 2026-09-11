
Where to resume
worktree	C:\Users\michav\projects\salamander_spotter\.claude\worktrees\pipeline-refactor
branch	pipeline-refactor (off master @ bbbad58)
HEAD	367fcde
master	untouched — app still ships from the primary checkout

cd C:\Users\michav\projects\salamander_spotter\.claude\worktrees\pipeline-refactor
git log --oneline master..HEAD
pixi run pytest -q          # 182 passing

Done & committed (each commit pytest-green)
Stage 0 7e29d59 — housekeeping
9a08610 — committed app/db/migrations/*.sql (were untracked on master; every DB test errored without them)
Stage 1 64b07e9 — import surgery: pipeline is a real package. All 75 sys.path.insert hacks gone, sibling imports → absolute across 42 files, pyproject.toml + pixi activation PYTHONPATH, a task per sweep. 0 import failures, full pytest = baseline, all 3 app seams verified.
Stage 3 d2e7e9b — ExperimentConfig: frozen picklable dataclass, from_env() reproduces defaults exactly; sweep_all9 has no os.environ left. +4 unit tests.
Stage 5 (partial) 47fa13e — Splitter/Fold (adds repeated-CV / E2) + CensusProtocol (census scoring lifted out of sweep_all9). +3 unit tests.

Left
Numeric regression sweep — a QUICK sweep-all9 is running in the background; it's now through fold 0 for several models with sane numbers (the 10‑hr job that was choking the machine has eased). Folded into the §11 adoption gate; Stages 1/3/5 are behaviour-preserving + pytest-verified.
Stage 2 (archive) and Stage 4 (contract package layout) — need your decisions (the spot_embedding triage; whether pipeline/ moves under src/).
Stages 5-finish, 6–12 — not started. Resume order in the doc: decisions → 10 → 6 (run the sweep regression here) → 7 → 8 → 11 → 9 → 12.

decisions needed

Stage 2 (archive):

D1 — spot_embedding/: it's not dead (backs the emb-* tasks; spot_transformer borrows its crop/encoder code). → recommend leave alone.
D2 — the "strict matching" cluster: 10 interlinked files (strict_voter, aggregator_e2e_strict, distinctiveness, gate_calibration, compare_strict, compare_e2e_strict, train_e2e_transformer, sweep_synthetic, scaling_set, the run_*.sh runners). Nothing outside the cluster imports them except sweep_bakeoff. Is that investigation finished? → archive whole / keep live / partial.
D3 — pretrained_cnn.py: its own comment says "DROPPED"; no MODELS row uses it. → recommend archive.
D4 — the run_strict_experiments.sh / scripts/experiments/*.sh runners: archive with D2 or keep.


Stage 4 (contract package):

D5 — layout: pipeline/ stays at repo root importing salamander_spotter (rec) vs. everything moves under src/salamander_spotter/.
D6 — package location: salamander_spotter/ at repo root (rec) vs. src/ layout.
D7 — does correspondences (seam S1) move into the contract package, so the app stops importing pipeline.spot_transformer.* at all? → recommend yes



# Pipeline Refactor — from research scripts to an experiment engine

A staged plan to reshape `pipeline/` (primarily `pipeline/spot_transformer/`) so that it is
easier to read, has clear contracts between layers, and **natively supports the experiments
planned next**: parallel folds, fold reshuffling, pickling work onto RunPod GPUs, a shared
data model with the `salamander_spotter` app, and standard experiment tracking.

- **How to read this:** each stage has `- [ ]` sub-tasks, a **Why**, a **Definition of done
  (DoD)**, and a **Check** you can run. Stages are ordered; each one leaves every sweep
  runnable.
- **Where this happens:** every stage (0–12) runs on a long-lived `pipeline-refactor` branch
  in a **separate `git worktree`**, never on `master`. `master` keeps shipping the app
  untouched until the whole refactor passes its adoption gate
  ([§11](#11-development-model--the-isolated-worktree)).
- **Naming is a proposal** — rename freely, but keep the layer boundaries.
- **Scope:** `pipeline/spot_transformer/` is the target. `spot_embedding/`,
  `generate_spot_labels/`, `preprocessing/`, `correspondence/`, `interesting_spots/` are
  touched only where they block a stage (see [§10](#10-out-of-scope-and-open-questions)).
- **The app must not break:** the app touches the pipeline at exactly three seams
  ([§2](#the-app-integration-surface-the-seams-to-preserve)); the adoption gate
  ([§11](#11-development-model--the-isolated-worktree)) is what proves they still hold.

---

## 1. Goals

### Experimentation goals (the reason for this refactor)

| # | Goal | What the architecture must provide |
|---|---|---|
| E1 | Run all folds in parallel | `run_fold` is a **pure, picklable function**; folds are an independent task list |
| E2 | Reshuffle fold assignments easily | fold generation is one primitive with a seed and `n_repeats`; the realized plan is an artifact |
| E3 | Pickle + run on RunPod GPUs; split light-ML from heavy-DL | serializable task/result types; a per-model `compute` tag; an `Executor` abstraction with a RunPod backend |
| E4 | Data structures fit the app natively | shared contract types in one importable package both `app/` and `pipeline/` depend on |
| E5 | Use standard ML experiment-tracking tools | a pluggable `Tracker`; config is structured data, not env vars; metrics land in a standard file layout |
| E6 | De-cluttered, intuitive `pixi run` covering every lifecycle stage | one `ss` CLI with lifecycle command groups; `pixi run <group> <subcommand>`; ~12 pass-through tasks instead of ~50 flat ones |

### Code-quality goals (stated in the refactor discussion)

- Remove or archive unused code (extend the existing `deprecated/` pattern).
- Introduce object primitives that hold logic naturally instead of `dict` + `if family ==`.
- Shared code between pipelines with explicit, documented contracts.
- High-level orchestration (sweeps) holds no low-level logic.
- Fewer inline comments; self-documenting names; a `README.md` per sub-package.
- Research → engineering: every runnable thing is a `pixi` task; a smoke test in CI.
- `pixi run` is a discoverable hierarchy keyed to lifecycle stages (`data`, `labels`,
  `review`, `features`, `experiment`, `audit`, `viz`, `app`, `ui`, `qa`, `release`), not a
  flat list of ~50 script names with three naming conventions (see [§8](#8-command-hierarchy--the-ss-cli--pixi)).

### The unifying principle

> **A pure-functional task graph with serializable boundaries.**

Every stage below moves the code toward this shape. Once `run_task(config, fold, spec) ->
result` is pure and picklable, E1–E3 are "which executor", and E5 is "log the config and
the results". E4 is the type layer that the same boundary reuses.

---

## 2. Current state

### Two eras stacked

| Area | What it is | Status |
|---|---|---|
| `spot_transformer/` | current champion track; already split into `core/ models/ eval/ sweeps/ viz/ deprecated/` | active; this refactor's target |
| `spot_embedding/` | earlier "Phase 0 harness" framework; backs the `emb-*` pixi tasks; `spot_transformer` borrows its SSL crop code | semi-active — [triage in §10](#10-out-of-scope-and-open-questions) |
| `generate_spot_labels/`, `preprocessing/`, `correspondence/`, `interesting_spots/` | data-prep + Gradio review apps (`app.py`/`page.py`/`server.py`) | active; out of scope |
| `app/pipeline_bridge/` | the app's only consumer of `pipeline/`; "thin adapters, no logic" | active; the E4 contract point |

### What makes it hard to use today

1. **Import system.** ~60 files do `sys.path.insert(...)` then `import data as d` /
   `import census as cen`. `spot_transformer` cannot be imported as a package; every script
   re-bootstraps. `deprecated/README.md` is entirely about this breaking. Sweep docstrings
   say `pixi run python pipeline/.../sweep_all9.py` while `scripts/README.md` says "never
   bare `python`".
2. **Orchestration holds low-level logic.** `sweep_all9.py::run_fold` is a ~100-line
   `if fam == "raw" / "feat" / "set" / "e2e"` dispatch mixing data-view construction, training,
   scoring, and metric computation. This pattern is copied across 11 sweep drivers.
3. **Config is env vars.** `QUICK`, `ONLY`, `TRAIN_POOL`, `SOURCE`, `SASA_TRAIN_ONLY`,
   `MIN_QUALITY`, `SSL_*` — read at module load in several files. Cannot be pickled to a
   worker or logged as structured params.
4. **Research narrative lives in source.** Multi-paragraph lab-notebook comments inside the
   `MODELS` list and inline in `_novelty_block`.
5. **One README** in the whole of `pipeline/` (`deprecated/`).
6. **Uncommitted WIP is broken.** `config/__init__.py` uses `os` without importing it;
   `models/__init__.py` references `N_BANDS`/`QUICK` it does not import. Resolve before Stage 1.

### The app integration surface (the seams to preserve)

The app couples to `pipeline/` at **exactly three points**. Everything else in
`app/pipeline_bridge/` is either a `Fake*` (tests) or a `NotImplementedError` stub. These
three are what the refactor must not break, and what the adoption gate ([§11](#11-development-model--the-isolated-worktree)) verifies:

| # | Seam | Where | Contract to freeze | Refactor stages that touch it |
|---|---|---|---|---|
| S1 | `correspondences` import | `app/pipeline_bridge/matching.py` — `from pipeline.spot_transformer.core.strict_match import correspondences` | the callable's signature + return shape | Stage 1 (package path), Stage 4 (moves to `salamander_spotter`), Stage 6 |
| S2 | training subprocess | `app/pipeline_bridge/training.py` — spawns `sweeps/train_all13.py`, greps stdout for `^wrote (.*manifest\.json)$`, parses `manifest.json` = `[{name, kind, metrics, weights_path}, ...]` | **the `manifest.json` schema** + a stable entrypoint + the `wrote …` stdout line | Stage 6, Stage 11 (entrypoint → `experiment train --preset deploy`) |
| S3 | `spot_embeddings` DB table | `app/pipeline_bridge/matching.py::_spot_embeddings` reads `contours.db` directly via duckdb; the additive-cosine formula mirrors `core/embeddings.py` | **`contours.db` schema** + the 2×31-dim L2-normalized-halves layout | none — materialization writes *new* artifacts, never rewrites the source DB |

Plus two soft couplings: the shared ID convention (`app/ids.py` ↔ pipeline, spec §6.1) and
`settings_store.py`'s reference to `pipeline/utils/logger_utils.py`.

Existing coverage: `tests/integration/test_pipeline_matching_bridge.py` (S1, S3) and
`tests/unit/test_pipeline_training_bridge.py` (S2). Running these green **against the
refactored pipeline** is the core of the adoption gate.

---

## 3. Target architecture — the spine

```
ExperimentConfig                 frozen dataclass · one from_env()/from_cli() at the entrypoint
      │                          (replaces the QUICK/ONLY/TRAIN_POOL/SOURCE/SSL_* env sprawl)
      ▼
Splitter(seed, n_repeats)  ──▶  list[Fold]        Fold(fold_id, repeat_id, train_ids, eval_ids)  frozen
      │
      ▼
prepare(config, dataset)  ──▶  DatasetArtifact + [ViewArtifact ...]     content-addressed files on disk
      │                                                                 (see §5 Materialization)
      ▼
tasks = [FoldTask(fold, spec, artifact_refs, config)   for fold in folds  for spec in specs]
      │                                                  picklable · artifact_refs are hashes+paths
      ▼
Executor.map(run_task, tasks)          backend per spec.compute:  local · process-pool · runpod-gpu
      │
      ▼
run_task(task) ──▶ FoldResult(spec_name, fold_id, repeat_id, metrics, checkpoint_path)   pure · picklable
      │
      ▼
Aggregator.reduce(results) ──▶ RunSummary          mean/std over folds × repeats, per model
      │
      ▼
Tracker.log(config, results, summary)   +   ResultsTable.write(run_dir)      both always run
```

### The primitives

Each is a small module with one job and a `README`-level docstring.

#### `ExperimentConfig` — `spot_transformer/config.py`

A frozen dataclass. Everything currently read from `os.environ` becomes a field with a
default. One constructor `from_env()` (back-compat) and one `from_cli(argv)`. Passed
explicitly into every function that needs it — **no module-level env reads anywhere else**.

```python
@dataclass(frozen=True)
class ExperimentConfig:
    quick: bool = False
    only: tuple[str, ...] = ()
    train_pool: Literal["filtered", "full"] = "filtered"
    source: Literal["all", "sasa", "kf"] = "all"
    sasa_train_only: bool = False
    min_quality: float = 0.0
    k_folds: int = 5
    n_repeats: int = 1            # E2: repeated CV with reshuffled folds
    seed: int = 0
    neg_per_query: int = 60
    n_bands: int = 16
    novel_frac: float = 0.35
    ep_set: int = 250
    ep_e2e: int = 30
    embedding_table: str = "default"
```

**DoD:** `grep -rn "os.environ" pipeline/spot_transformer` returns only `config.py`.

#### `Fold` + `Splitter` — `spot_transformer/eval/splits.py`

`Fold` is a frozen dataclass of index arrays. `Splitter` owns **all** the leakage-safe rules
(split by individual, `>= 2` real images per individual, synthetic images train-only) that
are currently spread through `data.get_cv_folds`.

```python
class Splitter:
    def __init__(self, config: ExperimentConfig): ...
    def plan(self, dataset) -> list[Fold]:
        """k_folds × n_repeats folds. repeat r uses seed = hash(config.seed, r)."""
```

- **E2:** `n_repeats > 1` gives repeated CV; each repeat reshuffles. `RunSummary` aggregates
  over the flattened `folds × repeats`, so a seed-sensitive metric surfaces as variance.
- The realized `list[Fold]` is written to `run_dir/fold_plan.json` — a run is reproducible
  and re-attachable from that file alone.

**DoD:** `Splitter(cfg).plan(ds)` is deterministic given `cfg`; `fold_plan.json` round-trips.

#### `ModelSpec` — `spot_transformer/models/registry.py`

Replaces `dict(name=..., family=..., hidden=...)`. A dataclass with typed, family-specific
hyperparameters and a `compute` tag. The `MODELS` list becomes a clean registry; the
lab-notebook prose moves to `models/README.md`.

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str
    formulation: str                       # "feat" | "set" | "e2e" | "raw"
    compute: Literal["cpu_light", "cpu_heavy", "gpu"] = "cpu_light"
    hparams: Mapping[str, object] = field(default_factory=dict)
    feature_attr: str = "spots"
    ssl_tag: str | None = None
```

`compute` routing default: `feat`/`raw` → `cpu_light`; `set` → `cpu_heavy`; `e2e`/`ssl` → `gpu`.

#### `Formulation` — `spot_transformer/models/formulations/`

The polymorphism that removes `run_fold`'s `if fam ==` ladder. One class per formulation,
each implementing:

```python
class Formulation(Protocol):
    compute: str
    def build_views(self, dataset, split: Fold, config) -> Views: ...
    def train(self, views: Views, spec: ModelSpec, config) -> Fitted: ...
    def score(self, fitted: Fitted, views: Views) -> np.ndarray: ...   # per-candidate match score
    def save(self, fitted: Fitted, path: Path) -> None: ...            # state_dict / joblib
    def load(self, path: Path) -> Fitted: ...
```

Implementations: `FeatFormulation`, `SetFormulation`, `E2EFormulation`, `RawBaseline`.
`E2EFormulation` is the **only** module that imports `torch`, and it does so lazily so the
light-ML path never pays for it.

#### `run_task` — `spot_transformer/experiment/task.py`

Today's `run_fold` inner body, made pure:

```python
def run_task(task: FoldTask) -> FoldResult:
    seed_everything(hash(task.config.seed, task.fold.repeat_id, task.fold.fold_id, task.spec.name))
    form  = FORMULATIONS[task.spec.formulation]
    views = formulation_views(form, task)          # loads ViewArtifact from disk, slices to the fold
    fitted = form.train(views, task.spec, task.config)
    scores = form.score(fitted, views)
    metrics = CensusProtocol(task.config).evaluate(scores, views.split)
    ckpt = task.run_dir / "checkpoints" / f"{task.spec.name}_f{task.fold.fold_id}.pt"
    form.save(fitted, ckpt)
    return FoldResult(task.spec.name, task.fold.fold_id, task.fold.repeat_id, metrics, ckpt)
```

- **No** logger side effects mid-computation (return structured progress instead).
- **No** module globals; `config` is an argument.
- Everything it touches (`FoldTask`, `FoldResult`, `ViewArtifact` refs) is a dataclass of
  primitives / paths — picklable.

**DoD:** `pickle.dumps(FoldTask(...))` succeeds; `run_task` produces identical `FoldResult`
in-process and via `ProcessPoolExecutor`.

#### `CensusProtocol` — `spot_transformer/eval/census_protocol.py`

Owns the open-set split + `_census_metrics` + `_novelty_block`, all currently inlined in
every sweep. `evaluate(scores, split) -> dict[str, float]`.

#### `Executor` — `spot_transformer/experiment/executors/`

```python
class Executor(Protocol):
    def map(self, fn, tasks: list[FoldTask]) -> list[FoldResult]: ...
```

| Impl | Backend | For |
|---|---|---|
| `SerialExecutor` | in-process loop | debugging; the default |
| `ProcessPoolExecutor` wrapper | `concurrent.futures` / joblib | E1: local parallel folds |
| `RunpodExecutor` | RunPod jobs (see §6) | E3: `gpu` tasks |

A `RoutingExecutor` dispatches per `task.spec.compute` — one sweep fans `cpu_light` rows
across local cores and `gpu` rows to RunPod in the same run.

#### `Tracker` — `spot_transformer/experiment/tracking.py`

See [§7](#7-tracker). `FileTracker` (default, always on) + optional `MlflowTracker`.

#### `ResultsTable` — `spot_transformer/experiment/results_table.py`

`.add_row(...)` / `.write(run_dir)`. Emits the human-readable `summary.md` the sweeps write
by hand today, plus a provenance header (git sha, `config.json` digest, timestamp).

#### `Experiment` — `spot_transformer/experiment/runner.py`

The driver that every sweep currently reimplements: parse config → `Splitter.plan` →
`prepare` → build tasks → `Executor.map` → `Aggregator.reduce` → `Tracker.log` +
`ResultsTable.write`. Also owns the cp1255 stdout reconfigure.

```python
def run_experiment(config: ExperimentConfig, specs: list[ModelSpec],
                   executor: Executor, tracker: Tracker) -> RunSummary: ...
```

A sweep becomes ~30 lines: pick `specs`, pick an `executor`, call `run_experiment`.

---

## 4. The `src/salamander_spotter/` contract layer (E4)

A new **importable, dependency-light** package (no `torch`, no `opencv`) holding the types
that cross every boundary — the app's DB, the pipeline, and the RunPod workers:

```
src/salamander_spotter/
├── __init__.py
├── types.py          Frame / ImageSet (spots + embeddings), Individual, Spot,
│                     Correspondence, MatchResult, MatchCandidate
├── serialize.py      to_npz / from_npz, to_dict / from_dict  (the only (de)serialization)
└── ids.py            stable id derivation  (merge with app/ids.py)
```

- `app/` maps its ORM rows to/from these types instead of hand-reaching into
  `pipeline.spot_transformer.core.*` (as `app/pipeline_bridge/matching.py` does today).
- `pipeline/` consumes and produces these types at its edges; `core/data.py` returns
  `list[ImageSet]` built from them.
- RunPod workers `pip install salamander_spotter` (cheap) + `torch`, nothing else.
- Same types are what `serialize.py` writes into the materialized artifacts (§5), so the
  contract layer and the cloud boundary share one implementation.

**Layout decision:** adopt a `src/` layout, editable-installed via `pixi` /
`pyproject.toml`. **Open:** does `pipeline/` also move under
`src/salamander_spotter/pipeline/` (one package, larger move) or stay at repo root and
import the contract package? Recommendation: **stay put now**, revisit after Stage 6.

---

## 5. Materialization (E3, and faster local reruns)

### The problem

Every sweep, every fold, every model rebuilds model input from `contours.db`:

```
contours.db ──load──▶ list[ImageSet] ──per (fold, family)──▶ pair tables
                      (live objects)   build_pairs / build_record_pairs / build_e2e_pairs
                                       RANSAC per candidate pair · soft-chamfer · crop-resize
```

The expensive part is the transform, not the DB read, and it is recomputed constantly.

### The design

**Materialization = run the transform once, write the result to disk as plain arrays, have
every task read the file.** It is a *computation cache*, not a storage format swap.

| | `contours.db` | materialized artifact |
|---|---|---|
| holds | source of truth: spot polygons, labels, provenance | derived: pair feature matrices, aligned records, spot-crop tensors, split indices |
| regenerable | no | yes — pure function of (db + config + code version) |
| lifetime | evolves (Haifa merge, repurple) | frozen; filename = hash of its inputs |
| access | row-oriented SQL, "spots for image X" | one `float32[N, D]` block, `mmap`'d, zero-copy into a tensor |
| worker needs | DB + full `pipeline/` + opencv/skimage | numpy/torch + `salamander_spotter` |

### Two layers

1. **`DatasetArtifact`** — `list[ImageSet]` (spots, embeddings, quality, labels, source)
   serialized compactly to one `.npz`. Cheap; shared by everything; changes rarely.
2. **`ViewArtifact`** — the per-formulation pair tables for a given
   `(dataset_hash, split_hash, config_subset)`. Expensive; family-specific.
   **Opt-in per formulation**: `feat`/`raw` build inline (trivial); `set` (RANSAC) and `e2e`
   (spot tensors) materialize.

### Content addressing

```
artifacts/materialized/
├── dataset_<hash>.npz                 hash = digest(contours.db mtime+size, embedding_table, source, quality)
├── views_set_<hash>.npz               hash = digest(dataset_hash, fold_plan_hash, neg_per_query, n_bands, seed)
└── views_e2e_<hash>/                   dir: spots.npy (memmap) + pairs_<fold>.npz
```

A run records the hashes it used in `run_dir/config.json`. Same inputs → same file → skip
recompute. Different config → different file → no silent staleness.

### New workflow

```
pixi run prepare-views        # dataset → DatasetArtifact + ViewArtifacts   (skippable if cached)
pixi run sweep-all9           # reads artifacts by hash; never opens contours.db directly
```

`prepare-views` is itself an `Executor.map` over `(fold, formulation)` — it can be
parallelized and farmed out like everything else. Light-ML-only sweeps can pass
`--no-materialize` and build views inline as today.

**DoD:** `prepare-views` twice in a row does no work the second time; deleting
`artifacts/materialized/` and re-running produces byte-identical artifacts.

---

## 6. RunpodExecutor (E3)

The `Executor` contract is unchanged; only the mechanics differ.

```
run_experiment (local)
  │  build gpu FoldTasks
  ▼
RunpodExecutor.map:
  1. ensure ViewArtifacts for gpu tasks exist locally  (prepare-views)
  2. push artifacts to RunPod volume / S3        runpodctl send  (once per artifact hash, cached)
  3. submit one job per task (or one batched job): image = salamander_spotter + torch,
     entrypoint = python -m salamander_spotter.worker run_task <task.json>
  4. poll; pull back FoldResult json + checkpoint files
  ▼
merge FoldResults with the local cpu results → Aggregator.reduce
```

- The worker image is thin: `salamander_spotter` (contract layer) + `torch` +
  `spot_transformer.models.formulations.e2e`. No opencv, no DB.
- `FoldTask` / `FoldResult` serialize to JSON (primitives + artifact hashes + paths).
- Checkpoints come back as files into `run_dir/checkpoints/`.
- **Open:** RunPod serverless endpoint vs. on-demand pod + `runpodctl exec`. Serverless is
  cleaner for many short `feat`-sized jobs; a pod is better for one long `train_all13`.
  Start with a pod.

---

## 7. Tracker

Minimal, file-based, no server — chosen to not interfere with the existing workflow.

### `FileTracker` (default, always on)

```
artifacts/runs/2026-09-05T14-22_a4bc3f1_all9/
├── config.json         the frozen ExperimentConfig
├── fold_plan.json      realized folds × repeats — reproducible
├── metrics.jsonl       one line per FoldResult   {spec, fold, repeat, census_f, ident_r1, ...}
├── summary.md          the human table (ResultsTable)
└── checkpoints/
```

`metrics.jsonl` + `config.json` is the whole contract — greppable, diffable, loads into
pandas/DuckDB in one line, and can be replayed into any tracker later.

### `MlflowTracker` (optional, behind `config` flag or env)

Mirrors the same events to MLflow in local file mode (`file:./mlruns`) — zero infra, gives
the run-comparison UI if wanted. Sweep = parent run; folds = nested runs (or `step=fold_id`).
`pip install mlflow` is an optional extra; nothing imports it unless enabled.

**DoD:** a sweep with no flags writes only the `artifacts/runs/<id>/` tree and needs no
non-stdlib tracking dependency.

---

## 8. Command hierarchy — the `ss` CLI + pixi (E6)

### The problem

~50 flat `pixi` tasks with three naming conventions in one namespace (`app:serve`,
`emb-prepare`, `merge-haifa`, `data-hygiene`), most of them one-off research scripts promoted
to top-level tasks, each documented by a multi-line block comment in `pixi.toml`. `pixi run
<TAB>` is a wall of names with no grouping and no way to ask "what are the data-prep steps".

### The shape

One console CLI — `ss` (Typer) — with command groups mirroring the project lifecycle. `pixi`
collapses to ~12 thin pass-through tasks, one per group, that forward trailing args:

```toml
[tasks]
data       = "python -m salamander_spotter.cli data"
labels     = "python -m salamander_spotter.cli labels"
review     = "python -m salamander_spotter.cli review"
features   = "python -m salamander_spotter.cli features"
experiment = "python -m salamander_spotter.cli experiment"
audit      = "python -m salamander_spotter.cli audit"
viz        = "python -m salamander_spotter.cli viz"
app        = "python -m salamander_spotter.cli app"
ui         = { cmd = "python -m salamander_spotter.cli ui", cwd = "web" }
qa         = "python -m salamander_spotter.cli qa"
release    = "python -m salamander_spotter.cli release"
```

`pixi` appends everything after the task name to the command, so the two-token form works
directly:

```
pixi run labels body-extraction --input all_sasa_norm --dry-run
pixi run data merge-haifa --dry-run
pixi run experiment sweep --preset all9 --parallel
pixi run experiment sweep --preset all9 --gpu-backend runpod
pixi run audit hygiene
pixi run app serve
```

`ss <group> --help` lists that group's subcommands with one-line descriptions — the
discoverability a flat task list never had.

### The groups (project lifecycle)

| group | lifecycle stage | absorbs (current tasks) |
|---|---|---|
| `data` | dataset assembly & naming | normalize, rename-amir, merge-haifa, correct-axis, build-dataset, package-dataset, fold-synth |
| `labels` | spot-label generation | extract-spot-labels {count,segment,anatomy,contours}, repurple, compute-quality |
| `review` | human-in-the-loop tools | correspondence(-analyze), pair-review(-gen), preprocess-review/export/interest, interesting-spot-selector, review-labels, corrections |
| `features` | representations | build-embeddings, emb-prepare, prepare-views (materialization) |
| `experiment` | model bake-offs | the sweeps (as `--preset`), train-all13, bakeoff |
| `audit` | dataset & model checks | label-consistency, data-hygiene, feasibility, geo-check, mining-audit, repr-check, confidence-bands, constellation-check, count-animals, distinctiveness, gate-calibration |
| `viz` | failure inspection | failing-photos, unmatched-examples, missing-spots |
| `app` | the FastAPI app | app:serve / dev / migrate / import-dataset / backup |
| `ui` | the web frontend | ui:install / dev / build / bake |
| `qa` | tests & lint | app:test{,-unit,-integration}, import-lint, smoke |
| `release` | build & deploy | docker:build, deploy |

### Rules

- The `ss` CLI is a **thin dispatcher** — argument parsing only, delegating to existing
  `scripts/*` / `pipeline/*` mains. Same rule as `scripts/README.md` today; no logic moves in.
- Every subcommand carries a one-line `help=` string — this replaces the `pixi.toml` block
  comments.
- One-off investigations that are genuinely finished are **not** exposed; they move to
  `deprecated/` and are run with bare `python` if ever revisited.
- The `emb-*` tasks fold into `features` / `experiment` / `audit` per the `spot_embedding`
  triage ([open question 1](#10-out-of-scope-and-open-questions)).

**DoD:** `pixi run <TAB>` shows ~12 names; `ss --help` shows the lifecycle; every retained
script is reachable as `pixi run <group> <subcommand>`.

---

## 9. Directory layout

### Before (spot_transformer/, abridged)

```
spot_transformer/
├── config/__init__.py        (broken WIP)
├── core/          data.py  embeddings.py  strict_match.py  synthetic.py  ...
├── models/        aggregator.py  aggregator_set.py  aggregator_e2e.py  aggregator_e2e_strict.py
│                  ssl_pretrain.py  pretrained_cnn.py  distinctiveness.py  strict_voter.py  ...
├── eval/          census.py  novelty.py  + 9 one-shot check scripts
├── sweeps/        sweep_all9.py  train_all13.py  + 9 more drivers, each ~100–400 lines
├── viz/
└── deprecated/
```

### After

```
spot_transformer/
├── config.py                     ExperimentConfig
├── core/                         data access + feature extraction (no torch training)
│   ├── README.md
│   ├── data.py                   contours.db → list[ImageSet]  (via salamander_spotter.types)
│   ├── embeddings.py
│   └── strict_match.py
├── models/
│   ├── README.md                 the MODELS narrative lives here, not in the source
│   ├── registry.py               ModelSpec + the MODELS list
│   └── formulations/
│       ├── feat.py  set.py  e2e.py  raw.py      each: build_views/train/score/save/load
├── eval/
│   ├── README.md
│   ├── splits.py                 Fold + Splitter
│   ├── census_protocol.py        CensusProtocol (was _census_metrics + _novelty_block)
│   └── metrics.py
├── experiment/
│   ├── README.md
│   ├── config_env.py             from_env / from_cli
│   ├── task.py                   FoldTask, FoldResult, run_task
│   ├── runner.py                 run_experiment  (the shared driver)
│   ├── materialize.py            DatasetArtifact, ViewArtifact, content hashing
│   ├── executors/                serial.py  processpool.py  runpod.py  routing.py
│   ├── tracking.py               FileTracker, MlflowTracker
│   └── results_table.py
├── sweeps/                       ORCHESTRATION ONLY — target < 40 lines each
│   ├── README.md
│   ├── all9.py                   pick specs → run_experiment
│   └── ...
├── viz/
└── deprecated/                   + archived sweeps/models, each with a "superseded by" header
```

---

## 10. Out of scope, and open questions

### Out of scope

- `generate_spot_labels/`, `preprocessing/`, `correspondence/`, `interesting_spots/` —
  except the shared `salamander_spotter.types` adoption at their read edges (Stage 4).
- The Gradio review apps.
- Retraining or changing any model's numbers. **Every stage must reproduce current
  `results.md` metrics** (within seed variance) as its regression check.

### Open questions

1. **`spot_embedding/` triage.** It backs the `emb-*` tasks and lends SSL crop code to
   `spot_transformer`. Options: (a) leave as-is, extract only `train/crops.py` +
   `encoders/spot_encoder.py` into a shared spot; (b) fold the live parts into
   `spot_transformer/` and `deprecated/` the rest; (c) leave entirely alone.
   *Decision needed before Stage 2.*
2. **`pipeline/` under `src/`** or stay at repo root importing `salamander_spotter`?
   Recommendation: stay put until Stage 6.
3. **RunPod:** serverless endpoint vs. on-demand pod. Recommendation: pod first.
4. **Archive candidates** — confirm which are done investigations:
   `models/pretrained_cnn.py` (its own comment says "DROPPED"), `aggregator_e2e_strict.py`,
   `strict_voter.py`, `distinctiveness.py`, `gate_calibration.py`;
   `sweeps/compare_strict.py`, `compare_e2e_strict.py`, `sweep_synthetic.py`,
   `scaling_set.py`; root `run_strict_experiments.sh`, `nohup.out`, duplicate `results.md`.

---

## 11. Development model — the isolated worktree

### Why

The refactor spans ~60 files and 12 stages. `master` must keep shipping the app the entire
time. So the work happens on a long-lived `pipeline-refactor` branch checked out in a
**separate `git worktree`** — a second working directory backed by the same `.git`, with its
own files and its own `pixi` env. `master` is never in a half-refactored state; the app runs
from the primary checkout throughout.

> A worktree, not a fork or a separate repo: one history, one set of branches, no remote to
> keep in sync. If `pipeline/` later becomes its own installable package (plausible for the
> RunPod worker — see §4), that is a *post-adoption* move, not the vehicle for the refactor.

### Setup

```bash
# from the primary checkout — stays on master, app keeps working
git worktree add ../salamander_spotter.refactor -b pipeline-refactor
cd ../salamander_spotter.refactor
pixi install                       # its own env; installs the src/ layout editable
```

Two directories, one history:

| dir | branch | role |
|---|---|---|
| `salamander_spotter/` | `master` | the app runs here; **untouched** by the refactor |
| `salamander_spotter.refactor/` | `pipeline-refactor` | all 12 stages land here as commits |

### Branch hygiene

- Each stage = one or more self-contained commits on `pipeline-refactor`, tests green at
  every commit.
- **Rebase `pipeline-refactor` onto `master` frequently** — weekly, and after any `master`
  change under `pipeline/`, `app/pipeline_bridge/`, `app/ids.py`, or `scripts/`. Many small
  rebases, never a big-bang merge at the end.
- `master` changes outside `app/pipeline_bridge/` never conflict — the refactor touches app
  code only at the three seam stages (4, 6, 11), and only the files in
  [§2](#the-app-integration-surface-the-seams-to-preserve).
- When a seam stage needs an app-side edit, keep the **old call path working as a shim** too
  where cheap (e.g. `sweeps/train_all13.py` stays as a 3-line forwarder to
  `experiment train --preset deploy`). That way an accidental early merge, or a partial
  cherry-pick, cannot strand the app.

### The adoption gate (whole-refactor DoD)

`pipeline-refactor` is **not merged** until, run inside the worktree:

- [ ] every stage's **Check** passes
- [ ] `pixi run qa test` — the **full app suite** — green, with
      `tests/integration/test_pipeline_matching_bridge.py` and
      `tests/unit/test_pipeline_training_bridge.py` exercised against the refactored pipeline
- [ ] the three seams verified end-to-end
      ([§2](#the-app-integration-surface-the-seams-to-preserve)): S1 `correspondences`
      import resolves from wherever it now lives; S2 `experiment train` subprocess emits a
      schema-compliant `manifest.json` and the `wrote …` line; S3 `contours.db`
      `spot_embeddings` read is byte-unchanged
- [ ] `results.md` numbers reproduced within seed variance (regression check)
- [ ] a RunPod `gpu` sweep reproduces the all-local numbers
- [ ] every retained script reachable via `pixi run <group> <sub>`; CI smoke test green
- [ ] this document updated to match what was actually built

### Adoption

1. Final rebase onto `master`; re-run the full gate.
2. Tag the merge-base: `git tag pre-pipeline-refactor` — the old tree is then one
   `git checkout` away.
3. One PR `pipeline-refactor → master`, reviewed **stage by stage** (every commit was green).
4. Merge. `git worktree remove ../salamander_spotter.refactor`.
5. Watch the app in production for one cycle. If a regression the gate missed appears,
   `git revert` the merge commit — `master` is restored in one commit — fix on a fresh
   worktree off the branch, re-gate.

---

## 12. Staged checklist

Each stage is a separate commit set on `pipeline-refactor` ([§11](#11-development-model--the-isolated-worktree)),
leaves all sweeps runnable, and reproduces `results.md`.

### Stage 0 — housekeeping

- [ ] Finish or stash the `config/` + `models/__init__.py` WIP so `master` imports cleanly.
- [ ] Remove `nohup.out`; resolve the root vs `spot_transformer/results.md` duplication; add
      `nohup.out`, `artifacts/materialized/`, `artifacts/runs/` to `.gitignore`.
- **Check:** `QUICK=1 pixi run python pipeline/spot_transformer/sweeps/sweep_all9.py` still
  runs (package import doesn't work yet — that's Stage 1).

### Stage 1 — import surgery *(mechanical, ~60 files, no logic change)*

- [ ] `pyproject.toml` / `pixi` editable install so `pipeline` is importable.
- [ ] Delete every `sys.path.insert` and run-as-script fallback.
- [ ] `import data as d` → `from pipeline.spot_transformer.core import data as d` (etc.).
- [ ] A `pixi` task per sweep (`sweep-all9`, `train-all13`, …); update docstrings.
- [ ] `conftest.py` at repo root so tests resolve identically.
- **Check:** `python -c "import pipeline.spot_transformer.sweeps.all9"`; `QUICK=1 pixi run
  sweep-all9` matches pre-refactor numbers.

### Stage 2 — archive pass

- [ ] Resolve open question 1 (`spot_embedding`).
- [ ] Move confirmed-done sweeps/models to `deprecated/` with a `superseded by … (DATE)` header.
- [ ] `deprecated/` code may be import-broken; everything else must import.
- **Check:** `import`-lint — no non-`deprecated` module imports a `deprecated` one.

### Stage 3 — `ExperimentConfig`

- [ ] Dataclass + `from_env()` + `from_cli()`.
- [ ] Replace every `os.environ.get` outside `config.py` with a passed `config` arg.
- **Check:** `grep -rn "os.environ" pipeline/spot_transformer` → only `config.py`.

### Stage 4 — `salamander_spotter` contract package

- [ ] `src/salamander_spotter/` with `types.py`, `serialize.py`, `ids.py`; editable install.
- [ ] `core/data.py` returns objects built from `salamander_spotter.types`.
- [ ] `app/pipeline_bridge/matching.py` + `app/ids.py` use the shared types (seam **S1**).
- **Check:** full app `pytest` suite green — especially
  `tests/integration/test_pipeline_matching_bridge.py`; `import salamander_spotter` pulls in
  no `torch`/`cv2`.

### Stage 5 — `Fold` / `Splitter` / `CensusProtocol` / `ResultsTable`

- [ ] Extract `Splitter` from `data.get_cv_folds`; add `n_repeats` (E2).
- [ ] Extract `CensusProtocol` from the inlined `_census_metrics` / `_novelty_block`.
- [ ] `ResultsTable` writes `summary.md` + provenance header.
- **Check:** `sweep-all9` output diff vs. Stage 1 is whitespace only.

### Stage 6 — `Formulation` + `ModelSpec` + `run_task` + `Experiment`

- [ ] Four `Formulation` classes; `run_fold`'s `if fam ==` ladder deleted.
- [ ] `ModelSpec` dataclass; `MODELS` narrative → `models/README.md`.
- [ ] `run_task` pure + picklable; `run_experiment` shared driver.
- [ ] Every sweep in `sweeps/` rewritten to < 40 lines.
- [ ] `train_all13.py` kept as a thin forwarder so seam **S2** still resolves this stage.
- **Check:** `pickle.dumps(FoldTask(...))`; `run_task` identical in-process vs
  `ProcessPoolExecutor`; `sweep-all9` numbers unchanged; `test_pipeline_training_bridge.py`
  green.

### Stage 7 — `Executor` + parallel folds (E1)

- [ ] `SerialExecutor`, `ProcessPoolExecutor` wrapper, `RoutingExecutor`.
- [ ] `sweep-all9 --parallel` runs folds concurrently.
- **Check:** parallel vs serial `summary.md` identical; wall-clock drops ~`k_folds`×.

### Stage 8 — materialization (E3 prerequisite)

- [ ] `materialize.py`: `DatasetArtifact`, `ViewArtifact`, content hashing.
- [ ] `pixi run prepare-views`; sweeps read artifacts by hash.
- [ ] `set` + `e2e` formulations load from `ViewArtifact`; `feat`/`raw` stay inline.
- **Check:** `prepare-views` is idempotent; artifacts are byte-reproducible; `sweep-all9`
  numbers unchanged.

### Stage 9 — `RunpodExecutor` (E3)

- [ ] Thin worker image + `python -m salamander_spotter.worker`.
- [ ] Artifact push (cached by hash) + job submit + result pull.
- [ ] `RoutingExecutor` sends `gpu` specs to RunPod, `cpu_*` to the local pool.
- **Check:** `sweep-all9 --gpu-backend runpod` reproduces the all-local numbers.

### Stage 10 — `Tracker` + tracking (E5)

- [ ] `FileTracker` writes the `artifacts/runs/<id>/` tree for every run.
- [ ] Optional `MlflowTracker` behind a flag.
- **Check:** default run needs no non-stdlib tracking dep; `metrics.jsonl` loads in pandas.

### Stage 11 — command hierarchy (E6)

- [ ] `salamander_spotter.cli` — Typer app with the 11 lifecycle groups; `python -m` entrypoint.
- [ ] Each group dispatches to existing `scripts/*` / `pipeline/*` mains — no logic moves in.
- [ ] Collapse `pixi.toml` to ~12 pass-through tasks; delete the block comments (now `help=`).
- [ ] Sweeps become `experiment sweep --preset <name>`; deploy training becomes
  `experiment train --preset deploy` — point `app/pipeline_bridge/training.py::TRAIN_SCRIPT`
  at it and drop the `train_all13.py` shim (seam **S2** final form).
- **Why:** ~50 flat tasks with 3 naming conventions → a discoverable lifecycle hierarchy.
- **Check:** `pixi run <TAB>` ≈ 12 entries; every retained script reachable as
  `pixi run <group> <sub>`; `ss <group> --help` lists subcommands;
  `test_pipeline_training_bridge.py` green against the new entrypoint.
- *The stable groups (`data`, `labels`, `review`, `app`, `ui`, `qa`) can land right after
  Stage 1; `experiment` lands with Stage 6, `features` with Stage 8.*

### Stage 12 — docs + polish

- [ ] `README.md` in `core/`, `models/`, `eval/`, `experiment/`, `sweeps/`, and `pipeline/`.
- [ ] Inline comments cut to mechanism-only; `_prob`/`_r1`/`cen`/`d`/`nov` → real names.
- [ ] CI smoke test: import every non-`deprecated` module + `QUICK=1` on the live sweeps.
- **Check:** CI green on a clean clone.

---

## 13. What this enables, mapped back

| Goal | Delivered by |
|---|---|
| E1 parallel folds | Stage 6 (pure `run_task`) + Stage 7 (`Executor`) |
| E2 fold reshuffling | Stage 3 (`n_repeats`) + Stage 5 (`Splitter`, `fold_plan.json`) |
| E3 pickle + RunPod, light/heavy split | Stage 6 (picklable tasks) + Stage 8 (materialization) + Stage 9 (`RunpodExecutor`) + `ModelSpec.compute` |
| E4 app-native data structures | Stage 4 (`salamander_spotter` contract package) |
| E5 experiment tracking | Stage 3 (structured config) + Stage 10 (`Tracker`) |
| E6 intuitive command hierarchy | Stage 11 (`ss` CLI + pixi collapse) |
| readability / contracts / altitude | Stages 1, 2, 5, 6, 11, 12 |
