# `app/` — Salamander Spotter backend

Implements `docs/salamander_spotter_spec.md`. A full first pass of **M0–M7** is in place
(see spec §17.0 for the status table and the intentional deviations).

## Layout

```
app/
├── config.py            §4.3  env-driven settings + data-dir resolution
├── ids.py               §6.1  label ↔ display, code allocator, up_ provisional
├── settings_store.py    §11   typed settings (DuckDB) + secrets.json (0600)
├── db/                  §8    schema (0001, 0002), migrations, single-writer handle
├── worker/              §5.3  ThreadPool job queue + jobs table + SSE event bus
├── bridges.py / job_runners.py   assembly of pipeline seams + background jobs
├── pipeline_bridge/     §7    ExtractionBridge · MatchingBridge · CorrespondenceBridge
│                              · EditorBridge · TrainingBridge  (Fake* tested, Pipeline* = TODO)
├── services/
│   ├── ingest.py        §7.9  dataset transfer (A) + incremental ingest (B)
│   ├── quality_ladder.py §7.2 composites → tier
│   ├── extraction.py    §7.1  cost estimate, daily LLM budget, low-token warn
│   ├── matching.py      §7.3  run match, calibrator, Top-N by coverage
│   ├── decision.py      §9.1  decision engine, auto-approve, override guard, enrollment
│   ├── batches.py       §9.2  batches, publish/unpublish, published census
│   ├── extraction_editor.py §10.6  join/split/head-tail, re-bin/re-score, revert, corr. labels
│   ├── models.py        §7.4  registry, Score = Σ coef·metric, promote/rollback
│   ├── training.py      §7.5  snapshot → train → eval → best → auto-promote, scheduler
│   ├── census_report.py §9.4  XLSX + PDF + CSV into exports/
│   ├── notify.py        §9.3  notifications + SMTP
│   └── dashboard.py     §10.1 overview aggregations + map data
├── api/                 §4,§10  FastAPI factory + routers (routes/*)
├── backup/              §2.2,§2.3  backup · restore · export --full · import --full
└── __main__.py          §5.4  `app serve|migrate|import-dataset|backup|restore|export|import`
```

## Run

```bash
pixi run app:migrate
pixi run app:import-dataset datasets/all_sasa_norm_2026_23_07   # 0 LLM calls, idempotent
pixi run ui:build && pixi run ui:bake                           # bake web/out into app/resources/web
pixi run app:serve                                              # 127.0.0.1:8756
# or: pixi run app:dev  (uvicorn --reload)  +  pixi run ui:dev  (Next on :3000)
```

## Test — `pixi run app:test` (132 tests)

Unit + integration. Integration tests build a real tiny DuckDB `contours.db` (the §18.3
schema) under a temp dir — no pipeline, no network, no billing. `Bridges.fakes()` supplies
deterministic ML.

## Still open

Wire the `Pipeline*` bridges to the billed/GPU pipeline code (inside M2–M5); frontend
Vitest/Playwright suites (spec §16); MapLibre map rendering polish.
