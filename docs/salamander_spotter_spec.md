# Salamander Spotter — Application Specification & Execution Plan

**Status:** Draft for review · **Date:** 2026-09-04 · **Owner:** M. Vardy

This document fully specifies the **Salamander Spotter** application — a small self‑hosted web
application for the Kibbutz Sasa fire‑salamander census — and lays out the plan to build it.

**Deployment stance (see §2):** the app is delivered **remote‑first** — one server instance,
run and maintained by the project maintainer (M. Vardy), used through a browser by a handful of
reviewers. The architecture is deployment‑agnostic (§2.4) so relocating the app (even onto the
reviewer's own machine) stays a configuration change, not a rewrite.

It is the contract for implementation: nothing in `docs/salamander_spotter_ui/` (the design
canvas) or `pipeline/` should be re‑interpreted; where this document and the canvas disagree,
this document wins, and disagreements are logged in *Open questions* (§18).

The research/modeling side of the project is already built in `pipeline/` and documented in
`docs/project_goal.md`, `docs/project_proposal.md`, `docs/modeling_strategy.md`,
`docs/next_steps_2.md` and `results.md`. This spec covers **the application that wraps it**.

---

## 1. Purpose & scope

### 1.1 What the app is

A small self‑hosted web application, used by a review team of 2–5 people, that turns a stream of
community salamander photos into a **maintained population census**. It does four things:

1. **Ingest & extract** — take field photos (including WhatsApp exports), run the existing
   extraction pipeline (body mask, head→tail axis, per‑spot contours, quality scoring), and
   grade each photo on a quality ladder.
2. **Match & decide** — for every new sighting, search the enrolled individuals with the active
   matcher, produce a ranked candidate list with calibrated confidence, auto‑confirm the easy
   ones, and route the rest to a human review screen with a 4‑way decision.
3. **Curate the roster** — browse, rename, merge, split, flag and export individuals and their
   photo sets; correct extractions in place.
4. **Manage models** — retrain the matcher on newly reviewed data on a schedule, evaluate
   candidates against the production model, and promote/rollback with an auditable log.

The roster is **seeded once** by transferring the dataset the pipeline has already produced
(`datasets/all_sasa_norm_*` — raw images + `contours.db`), not by re‑running extraction; from then on
every new photo is extracted individually and **appended** to the database (§7.9).

### 1.2 What it is not

- Not a public SaaS. It is **one instance for one project**, on infrastructure the maintainer
  controls, reachable only through the maintainer's tunnel/edge. No public sign‑up, no multi‑tenant
  separation, no billing.
- Not a training framework. It *orchestrates* `pipeline/` code; the ML lives there.
- Not a species classifier (every animal is *Salamandra salamandra* — see `docs/project_goal.md`).
- Not offline‑capable, and not required to be (§18). The instance assumes the server has
  connectivity; working without internet is out of scope.
- Not multi‑site in v1. The census covers the **single Sasa site** and nothing in v1 assumes more
  than one; the data model keeps `site_id` so a **version N** *could* add sites, but that is not an
  assumption v1 builds on (§18).

### 1.3 Users & roles

**v1 has no in‑app authentication** (§18). Anyone who can reach the URL has the full app; access is
restricted at the network edge (the Cloudflare/Tailscale tunnel or Caddy — §2.1), not by a login
screen. There is one effective role in v1: a full user who can upload, review, curate the roster,
correct extractions, change settings, and promote models.

Role separation (admin / reviewer / viewer) is kept as a **later** option (§18) but is not built in
v1. "Contributor" (who took a photo) remains a **data label**, not an account. Mutating actions are
still attributed where an actor is known — the UI can carry an optional "who's reviewing" name (picked
once, sent as a header) so `review_decisions.reviewer_id`, `audit.actor` and the activity feed stay
meaningful. It is a label, not a credential.

---

## 2. Deployment model

> The brief's concern: the primary reviewer is non‑technical and 2.5 h away by car; if the app
> lives only on his machine, every "check the logs / rotate the API key / push a fix" becomes a
> screen‑share or a trip. The answer is to **host it where the maintainer already has shell
> access**, and only move it onto his machine later, if ever.

### 2.1 v1 — self‑hosted single instance (the way it ships first)

One long‑running server, run and owned by the maintainer. Reviewers open a URL and use it. The maintainer has shell access to the box at all times, so software updates, log
inspection, key rotation, DB fixes and restarts never require the primary reviewer to do
anything.

**Where the server runs — pick one (see §18):**

| Option | Cost | Notes |
|---|---|---|
| **Maintainer's own hardware at home** (mini‑PC / spare desktop) + a tunnel (Cloudflare Tunnel or Tailscale Funnel) | ~€0/mo (power) | Full control, data stays with the maintainer, no VPS to patch. Depends on home power/uptime. **Recommended starting point.** |
| **Small cloud VPS** (2 vCPU / 4 GB — Hetzner, DigitalOcean, …) | ~€6–15/mo | Better uptime, off‑site by default. Maintainer patches the OS. **Chosen host: a DigitalOcean droplet.** |
| **Maintainer's laptop, on demand** | €0 | Only viable if reviewers work in scheduled sessions; not for a standing queue. |

CPU‑only training (every ~60 days) is slow but acceptable on any of these; if a run gets
painful it is offloaded to the maintainer's workstation and weights are pushed to the server
(§7.5).

**How it's exposed:**
- The app listens on `127.0.0.1:8756` (plain HTTP) inside the host.
- A **reverse proxy / tunnel** terminates TLS and is the only thing on the public interface
  (§18):
  - **Cloudflare Tunnel** or **Tailscale Funnel** — no open inbound ports, HTTPS handled for
    you, optional identity gating at the edge (Cloudflare Access email OTP). Preferred.
  - or **Caddy** on the host — automatic Let's Encrypt cert for `spotter.<domain>`,
    `:443 → 127.0.0.1:8756`.
- Reviewers bookmark a stable HTTPS URL.

**Access control:** none inside the app in v1 (§18). The tunnel/proxy is the only gate — e.g.
Cloudflare Access email OTP or Tailscale ACLs at the edge. No login screen, no accounts, no sessions.

**Ships as:** a **Docker image** + `deploy/docker-compose.yml` (app + Caddy, or app alone behind a
tunnel) + a `.env` file. `docker compose up -d`. **v1 updates are manual**, done by the maintainer:
`git pull && docker compose up -d --build` (or the one‑line `deploy/deploy.sh`).

**Planned (post‑v1, §18):** CI builds the image and pushes it to **GitHub Container Registry**
(`ghcr.io`; public image pulls are free and need no login), and the remote **updates itself** (a small
watcher pulls the new image and runs `docker compose up -d`). Not required for v1 — dockerising and
updating by hand is enough to start. A non‑Docker **systemd unit** is also provided for a bare install.

### 2.2 Backups (v1 — maintainer's responsibility, automated)

The census is irreplaceable, so the server must not be the only copy.

- A nightly job (`app backup`, run from cron or a compose sidecar):
  1. makes a consistent copy of `app.duckdb` and `contours.db`,
  2. `tar`s those copies + `config.toml` + `images/` (only files that changed since the last run) +
     `models/registry.json` into one dated archive,
  3. copies that archive to **a second location off the server** — the simplest being another disk or
     the maintainer's own machine (a cloud bucket like Backblaze B2 is optional). Keep ~30 daily and
     ~6 monthly archives, deleting older ones.
- Encryption of the archive is **optional** and off by default; turn it on only if the off‑box target
  is untrusted (e.g. a public cloud bucket). If on, the key lives in `secrets.json` on the maintainer's
  own machine, not on the server (§18).
- `app restore <archive>` rebuilds a fresh instance from one archive (tested in CI — §16).
- The Dashboard shows "last successful backup" and warns if it is more than 48 h old.

### 2.3 Local single‑machine build — out of scope

An earlier draft proposed a PyInstaller desktop bundle for the primary reviewer's own PC. **This is
dropped** (§18): there is no PyInstaller build. If the app ever needs to run on the reviewer's machine,
it runs there the same way it runs on the server — as the **same Docker image** under Docker Desktop —
so there is nothing extra to build. Deployment stays a configuration change, not a repackage.

If server ⇄ machine migration is ever wanted, `app export --full` / `app import --full` move a full
data dir between hosts; config is env‑driven and the data layout is identical.

### 2.4 Deployment‑agnostic requirements (must hold from M0)

So moving the app between hosts (or onto the reviewer's machine) stays a config change, every
milestone keeps these true:

- Bind address, external base URL, DB path and data dir are **all environment / config driven** — no
  hard‑coded `localhost`, no compiled‑in URLs.
- The frontend never assumes its own origin: it calls `/api` relative and reads any absolute base from
  a runtime `/runtime-config.json` the backend serves (§4.2).
- The backend runs correctly **behind a reverse proxy on a subdomain** (honours `X‑Forwarded‑Proto` /
  `Host`, emits no absolute redirects).
- No feature depends on the browser and the server being the same machine ("reveal in file manager"
  degrades to "show path" when the browser is remote).
- Packaging produces **one artefact — the Docker image** — from the `app/` + `web/` code (§5).

### 2.5 From source (contributors / the ML pipeline)

```
git clone https://github.com/<org>/salamander_spotter
cd salamander_spotter
pixi install          # provisions Python + Node + all deps from pixi.toml
pixi run app:dev      # backend :8756 (--reload), Next.js dev server :3000 (proxied), no auth
```

---

## 3. Technology stack

| Layer | Choice | Rationale / notes |
|---|---|---|
| **Package / env manager** | **Pixi** (`pixi.toml`, already in repo) | One tool provisions the conda‑forge Python stack *and* `nodejs` + `pnpm` for the frontend. New task namespaces: `app:*`, `ui:*`, `docker:*`. |
| **Backend** | **Python 3.10+, FastAPI**, **uvicorn** bound to `127.0.0.1:8756`, public via a TLS‑terminating reverse proxy / tunnel (§2.1) | Async, typed (Pydantic v2), auto OpenAPI at `/api/docs` (exposure toggled in Settings, default off). Wraps `pipeline/` directly; heavy jobs go to a background worker (§5.3). |
| **Access control** | none in‑app (v1); gating is at the tunnel/proxy edge (§2.1) | No accounts, sessions or password hashing in v1. A per‑session CSRF token cookie is still set (§15). |
| **Public edge** | Cloudflare Tunnel **or** Tailscale Funnel **or** Caddy (auto‑HTTPS) | Only the edge is internet‑facing; the app never binds a public interface. |
| **Containerisation** | Docker (multi‑stage: `pnpm build` → Python runtime) + `docker compose` | One image, `linux/amd64` (+ `arm64` for a Pi / Apple‑silicon mini). |
| **Frontend** | **Next.js (App Router), TypeScript**, `output: 'export'` (fully static) | No SSR/Node at runtime. Client‑side data via **TanStack Query** against `/api`. Static bundle is served by the backend. |
| **UI toolkit** | React 18, **Tailwind CSS** + a small headless component set (Radix), **lucide-react** icons | Reproduces the canvas design system (§13) exactly: warm cream ground, dark sidebar, IBM Plex Mono for IDs, Hanken Grotesk headings. |
| **Client image/canvas work** | HTML `<canvas>` / SVG overlays; **Konva** (or plain canvas) for the extraction editor | Spot polygons, spine, head/tail anchors, match lines. |
| **App database** | **DuckDB** (`app.duckdb`), single writer serialized through the worker; **SQLite** fallback (§18) | Multi‑user now means concurrent reads + occasional writes — all writes funnel through one connection/queue; reads use short‑lived connections. |
| **Extraction artifacts** | Per‑dataset **`contours.db`** (DuckDB, existing schema) + raw images on disk | Unchanged from `datasets/*/`. The app owns an `images/` tree and a working `contours.db`. |
| **Delivery** | **Docker image** + `docker compose` (app + Caddy). **GitHub Actions** builds and (post‑v1) ships it to the remote for self‑update (§2.1). | Frontend is built first (`pnpm build`) and baked into the image. No PyInstaller. |
| **Maps** | **MapLibre GL JS** with online raster tiles (OpenFreeMap / MapTiler key) | Requires connectivity; on tile failure it shows a plain marker layer, not an offline mode (§9.6). |
| **LLM providers** | Google Gemini (default), OpenAI, Anthropic, Azure, local — via the existing pipeline abstraction | Provider + model *ladder* configured in Settings (§10.11). |

### 3.1 Repository layout (added by this project)

```
salamander_spotter/
├── pipeline/                     # EXISTING — extraction, matchers, eval (unchanged API surface)
├── app/                          # NEW — the application backend
│   ├── main.py                   # FastAPI app factory, static mount, SPA fallback, port pick
│   ├── config.py                 # settings model, data-dir resolution, secrets
│   ├── db/                       # DuckDB schema, migrations, repositories
│   ├── api/                      # routers: dashboard, upload, review, individuals, images,
│   │                             #          models, settings, export, notifications
│   ├── services/                 # ingest, extraction, quality-ladder, matching, decision,
│   │                             #          calibration, training-scheduler, census, notify
│   ├── worker/                   # background job queue + runners (extract, match, train)
│   ├── security/                 # CSRF token, edge-forwarded headers, rate limits
│   ├── backup/                   # backup / restore / export / import commands
│   ├── pipeline_bridge/          # thin adapters onto pipeline/* (no logic, just wiring)
│   └── resources/                # web build baked in at image build time
├── web/                          # NEW — Next.js frontend (TypeScript)
│   ├── app/                      # routes (see §10 for the map to screens)
│   ├── components/               # design-system components
│   ├── lib/                      # api client, query hooks, types (generated from OpenAPI)
│   └── next.config.mjs           # output: 'export'
├── deploy/
│   ├── Dockerfile                # multi-stage: web build → python runtime
│   ├── docker-compose.yml        # app + Caddy (or app alone behind a tunnel) + backup sidecar
│   ├── Caddyfile                 # spotter.<domain> → app:8756, auto-HTTPS
│   ├── salamander-spotter.service# systemd unit for a bare install
│   ├── .env.example              # every SPOTTER_* var, documented
│   └── deploy.sh                 # git pull + compose up + migrate + health-check + rollback
├── .github/workflows/
│   └── release.yml               # GitHub Actions: build + push the Docker image (+ post-v1 self-update)
├── docs/
│   ├── salamander_spotter_spec.md           # THIS FILE
│   └── salamander_spotter_ui/               # EXISTING design canvas
└── pixi.toml                     # EXISTING — gains app:*, ui:*, docker:* tasks
```

---

## 4. Runtime architecture

### 4.1 Process model (v1 — remote)

```
                        internet
                           │  HTTPS
              ┌────────────▼─────────────┐
              │  edge: Cloudflare Tunnel │   only public component;
              │  / Tailscale Funnel /    │   terminates TLS, optional
              │  Caddy (auto-HTTPS)      │   edge identity check
              └────────────┬─────────────┘
                           │  http 127.0.0.1:8756
        ┌──────────────────▼───────────────────────────────────┐
        │  container / host  (maintainer-controlled)            │
        │  uvicorn (FastAPI)                                    │
        │   ├── /              static SPA (web build)           │
        │   ├── /runtime-config.json  runtime config for the SPA│
        │   ├── /api/*         REST + SSE   (CSRF-guarded)      │
        │   ├── /media/*       images, overlays, thumbs         │
        │   ├── (no in-app auth — edge gates access)            │
        │   └── background worker  (extract · match · train)    │
        │                                                      │
        │   /data:  app.duckdb · contours.db · images/ ·        │
        │           models/ · logs/   (mounted volume)          │
        │   nightly backup ──► off-box target                   │
        └──────────────── outbound ────────────────────────────┘
                 LLM API · map tiles · SMTP · GitHub (updates)
```

Reviewers: browser → bookmarked HTTPS URL → the same SPA + API the canvas describes.

### 4.1a Deployment modes

| Mode | Bind | Used by |
|---|---|---|
| **Remote** (v1) | `127.0.0.1:8756` behind the edge | the hosted instance |
| **Dev** | `127.0.0.1:8756` + Next dev proxy | `pixi run app:dev` |

There is no separate build of the backend logic — modes differ only by configuration. There is no auth
mode: v1 has no in‑app auth (§1.3, §15).

### 4.1b Access control & attribution

- **No login in v1** (§18). `/api/*` and `/media/*` are reachable by anyone who reaches the origin; the
  tunnel/proxy edge is the gate (§2.1).
- A random **CSRF token** is still set in a cookie and required on mutating (`POST/PATCH/DELETE`)
  requests, to block drive‑by requests from other browser tabs (§15).
- **Actor attribution is optional and cheap:** the UI can carry a "who's reviewing" name (picked once,
  stored client‑side and sent as a header) so `review_decisions.reviewer_id`, `audit.actor` and the
  activity feed stay meaningful. It is a label, not a credential.
- Accounts, passwords, sessions, roles and rate‑limited login are **deferred** (§18); if LAN/multi‑user
  access is ever needed they become a separate milestone.

### 4.1c Single‑instance & concurrency

- One server process per data dir (advisory lock file); horizontal scaling is out of scope.
- All DB **writes** go through the worker's single writer connection (a queue); mutating request
  handlers enqueue a unit of work and await it. Reads open short‑lived connections.
- Optimistic concurrency on the roster: `individuals` / `images` carry a `rev`; a stale `PATCH`
  gets `409` and the client refetches. A review decision takes a row lock on the sighting, so a
  second reviewer sees "already decided by G. Esh‑Am" instead of double‑deciding.
- **SSE** (`/api/events`) pushes job progress, new notifications, and "queue
  changed" so every open tab stays live without polling.

### 4.2 Frontend routing under static export

`output: 'export'` cannot pre‑render `/names/[id]` etc. Handling:

- The SPA fetches `/runtime-config.json` on boot (external base URL, feature flags, optional actor
  label). There is no login screen in v1.
- All unknown paths under `/` are served `index.html` by a FastAPI catch‑all (SPA fallback);
  `/api/*`, `/media/*` and `/runtime-config.json` are matched *before* it.
- Deep links (a bookmarked `/review/s_1_2`) work — the client router restores them directly.

### 4.3 Data directory

Set by `SPOTTER_DATA_DIR`; in Docker it is the **mounted `/data` volume**. For a bare source run it
defaults to `~/.local/share/salamander-spotter` (Linux) / `%APPDATA%\SalamanderSpotter` (Win) /
`~/Library/Application Support/SalamanderSpotter` (mac). Contains:

```
<data-dir>/
├── config.toml            # non-secret settings (thresholds, ladders, notifications, sites)
├── secrets.json           # API tokens — 0600, or OS keychain if available (§16.2)
├── app.duckdb             # application database (§8)
├── contours.db            # extraction artifacts (pipeline schema)
├── images/
│   ├── raw/<image_id>.<ext>
│   ├── purple/<image_id>.png        # intermediate magenta-spot render
│   └── thumb/<image_id>.webp
├── models/
│   ├── <model_name>/weights.pt
│   ├── <model_name>/calibration.json
│   └── registry.json                # mirror of the models table for offline inspection
├── exports/               # generated CSV/XLSX/PDF census reports
└── logs/
    ├── app.log
    ├── worker.log
    └── model_stats/<run_id>.log     # WF4 artifact
```

---

## 5. Packaging & build

### 5.1 Pixi tasks (added to `pixi.toml`)

| Task | Does |
|---|---|
| `pixi run ui:install` | `pnpm install` in `web/` |
| `pixi run ui:dev` | `next dev` on `:3000` (proxies `/api` → `:8756`) |
| `pixi run ui:build` | `next build` → static site in `web/out/` |
| `pixi run app:dev` | `uvicorn app.main:app --reload --port 8756` (serves `web/out` if present, else expects `ui:dev`) |
| `pixi run app:serve` | production‑style local run, no reload |
| `pixi run gen:types` | regenerate `web/lib/api-types.ts` from the live OpenAPI schema |
| `pixi run db:migrate` | apply DuckDB migrations to the configured data dir |
| `pixi run docker:build` | `ui:build` → copy `web/out` into `app/resources/web` → `docker build` the app image |
| `pixi run docker:release` | `docker:build` + tag + push to the registry (CI), named per version |

Pixi gains a `[dependencies]` addition for `nodejs`, `pnpm` (or corepack), `fastapi`, `uvicorn`,
`pydantic`, `httpx`, `openpyxl` (XLSX), `weasyprint` or `reportlab` (PDF), `python-multipart`
(uploads), `pillow`, `pillow-heif` (HEIC/HEIF decode — §10.2). No `pyinstaller`.

### 5.2 Docker image

- **Multi‑stage** build: stage 1 runs `pnpm build` to produce `web/out`; stage 2 is a slim Python
  runtime that copies the static site into `app/resources/web` and serves it with
  `StaticFiles(directory=resource_path('web'))`.
- Base image ships CPU‑only Torch (matches the pipeline's stated CPU baseline); GPU users run from
  source. `duckdb`, `cv2`, `torch`, `google.genai` are ordinary wheels — no hidden‑import juggling.
- `/data` (app.duckdb, contours.db, images/, models/, logs/) is a **mounted volume**, never baked into
  the image, so updates never touch the census.
- Images: `linux/amd64` (+ `arm64` for a Pi / Apple‑silicon mini). Target compressed size **< 2 GB**
  with CPU Torch; weights are pulled to the volume separately (§7.6).

### 5.3 Background worker

Long jobs (extraction batch, full match search, training) run on an in‑process **thread pool +
job table** in DuckDB, not Celery/Redis (single machine, keep it simple). Each job:

- has `id, type, status(queued|running|done|failed|cancelled), progress, created_at, params, result, error`,
- streams progress over SSE,
- is resumable where the underlying pipeline stage is (extraction already is — see the
  `extract-spot-labels` task notes in `pixi.toml`).

Training runs as a **separate subprocess** (its own Python process) so a crash or OOM can't take
the server down, and its stdout is tee'd to `logs/model_stats/<run_id>.log`.

Training never touches the serving path. It runs in its own subprocess against a dataset snapshot and
writes only to the Trained table + `models/`. The **active (serving) model does not change** until a
candidate is explicitly **promoted** (manually, or by the auto‑promotion rule when it beats Best on the
selected metric). Only then do new inferences use it; the previous model is kept for one‑click rollback
(§18).

### 5.4 CLI (`app`)

`app` is the project's single console entrypoint (a `[project.scripts]` script / `python -m app`) built
on the same FastAPI app factory and config. Subcommands:

- `app serve` — run the server (what the container runs).
- `app migrate` — apply DuckDB migrations to the data dir.
- `app backup` / `app restore <archive>` — the backup/restore jobs (§2.2).
- `app export --full` / `app import --full` — move a full data dir between hosts (§2.3).
- `app import-dataset <path-to-datasets/all_sasa_norm_*>` — the one‑time / delta dataset transfer of
  §7.9 A (same work as the first‑run "Import existing data" step); idempotent, safe to re‑run.

---

## 6. Domain model

### 6.1 Identifiers (must match the pipeline convention)

- **Image ID** = raw file stem = `<code>_<individual>_<instance>`, e.g. `aj_1_2`. Synthetic
  views are `<label>_g<k>` (e.g. `aj_1_g0`). This is `salamander_id` in `contours.db`.
- **Individual ID (label)** = image ID with the trailing `_<instance>` removed: `aj_1_2 → aj_1`.
  `aj_1` and `aj_2` are **different animals**.
- **Display ID** = the label upper‑cased with a hyphen for the UI: `aj_1 → AJ-1` (canvas shows
  `AC-3`, `EN-1`). Round‑trips losslessly.
- **New individuals** are allocated in a **single, site‑less namespace** — all animals are from Sasa, so
  the leading `<code>` is **not** a site code (§18). Each new individual gets its own **two‑letter code**
  — the next free two‑letter combination not already used by imported data (`aa`, `ab`, … skipping taken
  ones) — so its label is `<code>_1` and its first photo `<code>_1_1` (e.g. a provisional upload `x1_3_2`
  becomes `aa_1_1`, display `AA-1`). When all 26×26 two‑letter codes are used, the allocator **expands to
  three‑or‑more letters** (`aaa`, `aab`, …); the width is not fixed. Prefixes already present in imported
  data (`aj_`, `ca_`, `en_`, …, historical source‑name codes) are **preserved as‑is** and skipped by the
  allocator so new IDs never collide with them.
- **Provisional (upload) ID** — a freshly uploaded photo, before any re‑ID decision, gets a
  reserved‑prefix id `up_<shortid>` (a short random / ULID token, e.g. `up_7f3a2b9c`). This is its
  `salamander_id` in `contours.db` through extraction and matching. On a decision it is renamed via a
  resumable mutation (§9.7): **Confirm re‑ID `X`** → `<X>_<next free instance>` (e.g. `aa_1_3`); **New
  individual** → `<newcode>_1_1`. The `up_` prefix is reserved and never allocated to an individual.
- **SSID** (spot ID shown in the extraction/image screens) = `<DISPLAY_ID>-<instance>-<spot_no>`,
  e.g. `EN-1-3-06`. Internally `(salamander_id, spot_id)` from `contours.db`.

### 6.2 Entities

| Entity | Key fields | Source of truth |
|---|---|---|
| **Contributor** | `id, name, short_name, contact?` | `app.duckdb` |
| **Site** | `id, name, code, parent_site?, lat, lon, notes` | `app.duckdb` / `config.toml` seed |
| **Image** (a.k.a. sighting) | `image_id, individual_id?, site_id, contributor_id, photographed_at, uploaded_at, source_filename, is_synthetic, status, ladder_tier, quality{overall,blur,lighting,spot,body}, n_spots, review_batch_id?` | `app.duckdb`; pixels/extraction in `contours.db` + `images/` |
| **Individual** | `individual_id, display_id, nickname?, site_id, status(provisional\|confirmed\|published\|merged_into), first_seen, last_seen, contributor_ids[], reference_image_id` | `app.duckdb` |
| **Spot** | `image_id, spot_id, centroid, area, local_contour[], mask, axial_bin, lateral_bin, axis_t, axis_offset, interest_score, ssid` | `contours.db` (+ interest cache) |
| **BodyAxis** | `image_id, head_xy, tail_tip_xy, midline[], left[], right[], source, judged_ok, judge_feedback` | `contours.db` |
| **MatchResult** | `id, query_image_id, model_name, created_at, candidates[]` where each candidate = `{individual_id, best_photo_id, rank, similarity, calibrated_confidence, spot_correspondences[]}` | `app.duckdb` |
| **ReviewDecision** | `id, image_id, batch_id, verdict(confirm\|new\|uncertain\|disqualify), chosen_individual_id?, new_individual_id?, reason_chips[], reviewer_id, decided_at, model_suggestion{individual_id,confidence}, was_override, note?` | `app.duckdb` |
| **ReviewBatch / Survey** | `id, name, created_at, status(open\|published), image_ids[], reviewed_count, published_at?` | `app.duckdb` |
| **Extraction correction** | `id, image_id, editor_id, edited_at, diff{spots_joined,spots_split,spine,head_tail,outline}, prev_snapshot` | `app.duckdb` + `contours.db` rows |
| **Model** | `name, kind, trained_at, dataset_snapshot_id, metrics{r1,r5,r10,bal_acc,novelty_auroc,review_at_90,score}, status(active\|candidate\|best\|archived), weights_path, calibration_path` | `app.duckdb` + `models/` |
| **TrainingRun** | `id, trigger(schedule\|manual\|images_threshold), started_at, finished_at, models_evaluated[], promotion{from,to,metric,auto}, log_path, emailed_to[]` | `app.duckdb` + `logs/model_stats/` |
| **Notification** | `id, type, severity, title, body, copy_text?, created_at, read_at?, dismissed_at?, action{label,href}?` | `app.duckdb` |
| **Job** | see §5.3 | `app.duckdb` |
| **Setting** | typed key/value; see §15 | `config.toml` / `secrets.json` |

### 6.3 Image lifecycle (status machine)

```
uploaded ─► extracting ─► extracted ──► matched ──► (auto_approved | in_review)
   │            │             │                          │
   │            └─► failed_extraction ─(fix / use anyway)─┘
   │                                                      ▼
   └─────────────────────────────────────────►  decided: confirmed / enrolled_new /
                                                           flagged_uncertain / disqualified
                                                              │
                                                (re-opened by "Flag for re-review")
```

`ladder_tier ∈ {auto_accept, needs_a_look, hand_correction, failed}` is derived from the quality
composites at extraction time (§9.2) and shown throughout.

---

## 7. ML pipeline integration

The app never re‑implements ML. It calls `pipeline/` through `app/pipeline_bridge/`.

### 7.1 Extraction (per uploaded photo)

Mirrors `pixi run extract-spot-labels all`:

1. **Animal count / screen** (`pipeline/generate_spot_labels/llm_animal_count.py`, `animal_screen.py`)
   — **multi‑animal frames are rejected automatically** (§18): extraction stops, the photo is marked
   `status=disqualified` with reason `multi‑animal frame`, and it never enters matching or the queue.
   Single‑animal frames continue.
2. **Spot segmentation** — LLM "magenta spot" repaint (`llm_spot_segmentation.py`), model
   *ladder* from Settings; OpenCV keys the colour and traces contours
   (`extract_spot_contours.py`).
3. **Body mask + axis** (`body_mask.py`, `llm_body_fill.py`, `llm_anatomy.py`) — whole‑body
   mask, head/tail tips, bisecting midline; geometric re‑tip when needed.
4. **Binning** (`binning.py`) — `axial_bin` 1–4, `lateral_bin` left/right/overlap.
5. **Quality** (`quality.py`) — the RAW markers + the four 0–1 composites + `overall_quality`
   (schema in `datasets/*/README.md`).
6. Write rows to `contours.db`; write `images/purple/`, `images/thumb/`.

Every stage is **resumable** and each LLM call is **billed** — the app must show a cost estimate
before a batch (like the pipeline's `--dry-run`) and enforce a per‑day call budget with the
low‑token notification (§10.2, §10.11).

### 7.2 Quality ladder → tier

| Composite gate (defaults, tunable in Settings) | Tier |
|---|---|
| `overall_quality ≥ 0.65` **and** extraction produced a usable axis + ≥ N spots (Settings, `min_spots_auto_accept`, default `3`) | **Auto‑accept** |
| `0.40 ≤ overall_quality < 0.65` | **Needs a look** |
| `overall_quality < 0.40` but extraction returned geometry | **Hand‑correction** |
| extraction failed (no mask / no spots / axis rejected + no fallback) | **Failed extraction** |

The **Quality cut‑off** slider (Settings, default `0.40`) is a *separate* control: photos below
it are kept as **training data only**, never used as review queries, unless the user picks "Use
anyway" per photo in Upload.

### 7.3 Matching (per sighting that reaches review)

Mirrors `pipeline/spot_transformer` matchers (`strict_match.py`, `models/aggregator*.py`,
`models/strict_voter.py`) and `spot_embedding`:

- **One active model does all inference** (§18). The model chosen on the Models tab / Settings is the
  **sole** matcher: it scores the query image against every enrolled individual's photos → per‑pair
  **similarity**. No other model contributes to the ranking, the suggestion, or auto‑approve.
- Similarity → **calibrated confidence** via that model's calibrator (`models/gate_calibration.py`).
- Candidates are ranked; **Top‑N** is the smallest N whose confidence mass covers the **coverage
  target** (default 90%), capped at **max candidates** (default 6). Canvas: "Top 3 of 289 — covers 91%".
- **Spot correspondences** for the Match‑lines view are produced by an always‑available **geometric
  matcher** (`strict_match.py` / constellation check), run purely for visualisation, so Match‑lines work
  regardless of whether the active model is embedding‑based and produces no per‑spot assignment itself.
- **Auto‑approve**: if the active model's top calibrated confidence ≥ **auto‑approve threshold**
  (default `0.95`), the sighting is confirmed without reaching the queue; it is still logged and
  reversible until the batch is published.

### 7.4 Calibration & metrics

Evaluation metrics come straight from `pipeline/spot_transformer/eval/` (`metrics.py`, `novelty.py`,
`census.py`, `confidence_bands.py`). Each metric gets a **fixed letter** so the composite Score is a
plain linear combination, not a parsed expression (§18):

| letter | metric | higher better |
|---|---|---|
| `a` | R@1 | ✓ |
| `b` | R@5 | ✓ |
| `c` | R@10 | ✓ |
| `d` | Balanced accuracy | ✓ |
| `e` | Novelty AUROC | ✓ |
| `f` | Review@90 (share auto‑handled at 90% precision) | ✓ |

**Score** = a weighted sum of these letters. The formula is stored as a **coefficient map**, e.g.
`{ "a": 0.5, "e": 0.5 }` (the default, = `0.5·R@1 + 0.5·Novelty AUROC`) or
`{ "a": 0.3, "c": 0.6, "e": 0.1 }`. The backend computes `Score = Σ coef·metric` directly — there is
**no `eval`/expression parser**, so a malformed entry is just a rejected coefficient, never executable
input (§15). Coefficients need not sum to 1; they are used as given.

### 7.5 Training run

Mirrors `pixi run build-dataset` (rebuild path, no API) + the sweep/eval code:

1. Snapshot the reviewed dataset (all confirmed images + any synthetic views the user has generated and
   kept — §7.7; none are generated automatically).
2. Retrain every registered *trainable* model on the **full** snapshot (canvas: "Every retrain
   uses the full dataset, not just what's new").
3. Evaluate each on the held **train/eval split** (kept stable unless the user re‑splits).
4. Write metrics to the **Trained** table; recompute **Best** by the selected metric/formula.
5. If a candidate beats the current **Best** *and* auto‑promotion is enabled, promote it to
   **active** (previous kept for one‑click rollback). Otherwise leave production alone.
6. Emit `model_stats.log`; raise the "training complete" notification; optionally email the
   research team (§10.12 WF4).

**Model retention & status tags.** Past models are kept only if tagged; the rest are discarded to save
disk. The tags (`models.status`) are: `active` (the single serving/inference model — shown as **Current**
in the UI), `candidate` (a freshly trained model from the latest run, awaiting promotion), `best` (the
best past version **of each model type**, by the selected Score), and `archived` (explicitly kept). Each
run **saves and tags**: (a) every model from the **latest full training batch**, (b) the **newly
trained** models (candidates), and (c) the **best past version of every model type**. Any model not
covered by a tag is not retained.

### 7.6 Model weights (local)

Model weights + calibrators live in `<data-dir>/models/` on the mounted volume. In v1 they are placed
there directly by the maintainer (copied in, or produced by a training run) — there is **no automatic
download** from a GitHub Release (§18). `config.toml` records which model is active and where its
weights/calibration sit; `registry.json` mirrors the table for offline inspection.

### 7.7 Synthetic views (on demand, opt‑in)

Synthetic `<label>_g<k>` views (§6.1) exist to give **individuals that have no positive pair** (a
singleton — one real photo) something to train against. In the app they are **never generated
automatically** (§18):

- When the roster contains singletons, the app **suggests** generating synthetic views for them (a
  Dashboard / Models hint listing how many singletons would benefit).
- The user chooses **whether** to run it and **for which** individuals, and picks the **attributes** of
  each synthetic view — how many views (`k`), and which variations (lighting / background / pose / angle)
  — before any billed generation happens (a cost estimate is shown, like extraction — §7.1).
- Generated views are tagged `is_synthetic=true`, kept **training‑only**, never shown as candidates or
  used as queries (§18), and only enter the next snapshot if the user keeps them.

### 7.8 Train/eval split — rules

The split must be **stable** so Score is comparable across retrains (§18):

- At first import a **base eval set** is carved out once — a stratified, held‑out slice of individuals
  that have ≥2 real photos (so each has a queryable positive). It is **frozen**: the same individuals stay
  in eval across every retrain unless the user explicitly **Re‑splits** (Models tab).
- **Synthetic views never go in eval** (they'd measure the generator, §7.7) — eval is real photos only.
- **New individuals default to train.** They do not enter the frozen eval set automatically, so adding
  animals cannot silently move the goalposts.
- **The tradeoff (advice):** more reviewed matches → better training, but a flood of brand‑new singletons
  mostly adds negatives and can dilute the signal without improving eval. Recommendation: let the roster
  grow freely (train benefits), keep the eval set frozen for comparability, and only **Re‑split** on
  purpose — roughly once the eval set becomes small relative to the roster (e.g. < ~15% of individuals with
  ≥2 photos), then re‑freeze. Surface this as a hint rather than doing it automatically.

**Training‑value & eval hints (image‑quality based).** The app scores each individual's *training value*
and *eval suitability* from the `image_quality` composites already in `contours.db` (§18.3) — no new
model — and surfaces them as **suggestions only** (nothing is auto‑applied). They show mainly as a
colour‑coded **health dot / icon** in the Names table (§10.7), with roll‑up alerts on the Dashboard:

| Health | Rule (defaults, tunable) | Suggestion |
|---|---|---|
| 🟢 **strong** | ≥2 real photos, median `overall_quality` ≥ 0.65, enough spots, axis `judged_ok` | good positive pair for training **and** a candidate for the frozen eval set |
| 🟡 **thin** | a singleton, **or** only middling photos (`overall_quality` 0.40–0.65) | capture/keep a better photo, or generate synthetic views (§7.7); stays in train, never eval |
| 🔴 **weak** | only photos below the quality cut‑off, very few spots, or a rejected axis | hard to match — flag "needs a better photo"; excluded from eval |

Dashboard/Names roll‑ups: "N individuals would benefit from synthetic views", "N individuals have only
low‑quality photos", "eval set is thin — consider Re‑split". The same `overall_quality` / `blur_quality` /
`spot_extraction_quality` fields drive both the per‑photo quality scores and these per‑individual hints, so
the reviewer sees *why* an animal is amber/red by opening it.

---

### 7.9 Dataset transfer & incremental ingest — the pipeline is never re‑run wholesale

The extraction pipeline (`pipeline/generate_spot_labels/*`) and the dataset builder
(`scripts/dataset/build_dataset.py`) are a **one‑time or per‑photo** cost. The app **never** re‑runs
them across the whole corpus, and there is **no "rebuild the dataset" action** anywhere in the UI or
CLI. Data reaches `/data` by exactly two paths, and both are **additive**.

#### A. One‑time transfer of the existing Sasa dataset (first‑run import, §14.1 step 5)

The existing `datasets/all_sasa_norm_<date>/` (currently `2026_23_07`: 1,869 images, 751 individuals —
§18.3, dataset README) is **moved in, not recomputed**:

| Source artefact | Goes to | How |
|---|---|---|
| `raw/<stem>.<ext>` | `<data-dir>/images/raw/<image_id>.<ext>` | file copy (or hardlink) — **no re‑encode** |
| `db/contours.db` | `<data-dir>/contours.db` | copied **verbatim** — schema unchanged (§18.3); no spot re‑extraction, no body‑mask/axis/bin/quality recompute |
| `image_quality` rows | `images.q_*`, `n_spots`, derived `ladder_tier` (§7.2) | read from `contours.db`, no recompute |
| filename stems | `images` (one row per `salamander_id`), `individuals` (one row per label = stem minus `_<instance>`) | derived from names + `contours.db` joins |
| `corrections.json` | `extraction_corrections` rows | so the human‑correction history survives the transfer |
| `images.purple_image` (if present) | `<data-dir>/images/purple/<image_id>.png` | copied if the dataset shipped it; otherwise left null — **never re‑generated with an LLM** |
| — | `<data-dir>/images/thumb/<image_id>.webp` | the **only** thing computed on import: a cheap local resize, no model, no billing |

- **Zero LLM calls, zero billing.** Import is a disk copy + a batch of DuckDB inserts. Target: the full
  23_07 dataset in a few minutes on the VPS.
- Imported prefixes (`aj_`, `ca_`, `en_`, …) are added to the ID allocator's **reserved set** so new
  two‑letter codes never collide (§6.1).
- Imported individuals land `status=published` (they are the standing census); their sightings are
  `status=confirmed` and attached to a synthetic **"Imported (pre‑app)"** batch that is already
  `published`.
- Synthetic `<label>_g<k>` views already in the source dataset are imported `is_synthetic=true`,
  **training‑only** (§7.7), and are **not regenerated**.
- **Idempotent & resumable.** Re‑running the import against the same folder is a no‑op for rows that
  already exist (keyed on `image_id` + a content checksum). Pointing it at a **newer** `datasets/`
  snapshot imports only the images that snapshot adds — the delta, via path B.

#### B. Incremental ingest — new datapoints append, they do not rebuild

Every photo added after the transfer — via the Upload screen (§10.2), a dropped folder, or a newer
`datasets/` snapshot — is processed **one photo at a time**:

- extraction (§7.1) runs **only for that photo**, appending new rows to `contours.db` and writing that
  photo's `images/*` renders;
- **one** new `images` row is inserted into `app.duckdb` (plus **one** `individuals` row only on a
  "New individual" decision, §9.1);
- **nothing already in either database is touched** — existing extractions, quality scores, bins,
  individuals, and decisions are left exactly as they are.

`contours.db` therefore only ever **grows by append**. The app mutates an existing `contours.db` row
**only** through the extraction editor (§10.6) and `POST /api/images/{id}/re-extract` (§10.2) — both
operate on a **single photo**, never a batch.

#### C. Training reads a snapshot of the live DB — it does not drive the pipeline

A training run (§7.5) takes a `dataset_snapshots` cut (row counts + checksum) of the **current**
`app.duckdb` + `contours.db` and trains on that. It never calls extraction or synthetic‑view
generation. Adding animals between runs simply means the next snapshot has more rows — importing data
or uploading photos has **no coupling to training** and triggers no retrain on its own.

#### D. Host migration moves the computed data dir, never recomputes

`app export --full` / `app import --full` (§2.3) move the already‑computed `/data` directory
(`app.duckdb` + `contours.db` + `images/` + `models/`) between hosts wholesale. No extraction, no
rebuild.

#### Summary — what each operation costs

| Operation | Runs extraction? | LLM cost | Existing rows |
|---|---|---|---|
| First‑run import of `datasets/all_sasa_norm_*` | no | €0 | insert‑only |
| Import a newer `datasets/` snapshot | new images only | new images only | insert‑only |
| Upload one new photo | that photo only | ~1 photo | untouched |
| "New individual" decision | no | €0 | relabels that one photo |
| Extraction editor save | re‑bin + re‑quality that photo | €0 (geometry) or ~1 photo (re‑purple) | that one photo |
| Training run | no | €0 | untouched (reads a snapshot) |
| `app import --full` | no | €0 | replaces the data dir |

---

## 8. Application database (`app.duckdb`)

Tables (DuckDB; migrations in `app/db/migrations/`). Extraction pixel data stays in
`contours.db`; this DB holds roster/workflow state and joins to it on `image_id`.

```
contributors(id, name, short_name, contact, created_at)
sites(id, name, code, parent_id, lat, lon, notes)
images(image_id PK, individual_id, site_id, contributor_id, photographed_at, uploaded_at,
       source_filename, is_synthetic, status, ladder_tier, q_overall, q_blur, q_lighting,
       q_spot, q_body, n_spots, review_batch_id, use_anyway, created_at)
individuals(individual_id PK, display_id, nickname, site_id, status, first_seen, last_seen,
            reference_image_id, merged_into, created_at)
individual_contributors(individual_id, contributor_id)          -- m:n
match_results(id PK, query_image_id, model_name, created_at, coverage_target, params_json)
match_candidates(match_result_id, rank, individual_id, best_photo_id, similarity,
                 calibrated_confidence)
spot_correspondences(match_result_id, candidate_individual_id, query_spot_id,
                     cand_image_id, cand_spot_id, score)          -- MACHINE-proposed (matcher output)
correspondence_labels(id PK, query_image_id, query_spot_id, other_image_id, other_spot_id,
                      verdict, editor, created_at)                -- HUMAN-verified spot pairs (§10.6)
review_batches(id PK, name, status, created_at, published_at)
review_decisions(id PK, image_id, batch_id, verdict, chosen_individual_id, new_individual_id,
                 reviewer_id, decided_at, model_suggestion_individual_id,
                 model_suggestion_confidence, was_override, note)
review_decision_reasons(decision_id, chip)                      -- reason chips
extraction_corrections(id PK, image_id, editor_id, edited_at, diff_json, prev_snapshot_json)
models(name PK, kind, trained_at, dataset_snapshot_id, r1, r5, r10, bal_acc, novelty_auroc,
       review_at_90, score, status, weights_path, calibration_path, notes)
training_runs(id PK, trigger, started_at, finished_at, promotion_from, promotion_to,
              promotion_metric, promotion_auto, log_path)
training_run_models(run_id, model_name, r1, r5, r10, bal_acc, novelty_auroc, review_at_90, score)
dataset_snapshots(id PK, created_at, n_images, n_individuals, checksum)
notifications(id PK, type, severity, title, body, copy_text, action_label, action_href,
              created_at, read_at, dismissed_at)
jobs(id PK, type, status, progress, params_json, result_json, error, created_at, updated_at)
activity(id PK, kind, summary, detail, actor, ref_type, ref_id, created_at)   -- dashboard feed
mutations(id PK, kind, state, plan_json, cursor_json, undo_json, actor,       -- resumable rename/
          created_at, updated_at)                                            --   merge/split (§9.7)
audit(id PK, at, actor, action, entity, entity_id, before_json, after_json)   -- everything mutating
```

All destructive/mutating actions write an `audit` row. Merges, renames, disqualifications and
promotions are reversible until a batch/run is published/confirmed.

---

## 9. Cross‑cutting behaviour

### 9.1 The decision engine

Given a sighting `S` with match result `M` and settings `T`:

| Condition | Outcome |
|---|---|
| `M.top.confidence ≥ T.auto_approve` (0.95) | **auto‑confirm** to `M.top.individual`; `status=confirmed`; `activity` + optional notification; **not** queued |
| ladder tier `failed` | queued as **hand‑correction**; no candidates shown until fixed |
| otherwise | queued; suggested = `M.top` if `M.top.confidence ≥ T.match_threshold` (0.55), else no suggestion → "likely new" |

Reviewer verdicts (canvas decision bar, keys `1–4`, `S`):

- **Confirm re‑ID `X`** → `images.individual_id = X`, `status=confirmed`, update individual
  `last_seen`/contributors, recompute reference photo.
- **New individual** → allocate a new two‑letter label (§6.1), rename the photo, create an
  `individuals` row (`status=provisional`), enroll.
- **Uncertain — flag** → `status=flagged_uncertain`; stays out of the census; the sighting waits in
  an **Uncertain** category until resolved. It can be resolved from the **Review** queue *or* from the
  **Names** table (open the sighting and re‑decide) — it does not have to be handled in the review flow.
- **Disqualify** → `status=disqualified` (bad photo / not a salamander / duplicate); excluded
  everywhere; reason chip required.
- **Skip** → no decision; next sighting.

**Override guard**: picking an individual other than the suggested one, or "New" when a
suggestion exists above threshold, shows the confirm toast ("You picked `ca_47` (0.12) over the
suggested `en_1` (0.68). Sure?") when *Warn before assigning against the model's suggestion* is
on (Settings, default on). The decision records `was_override=true`.

**Reason chips** (multi‑select, from the review_approval canvas): `distinct spot layout`,
`no candidate above threshold`, `good photo quality`, `few corresponding spots`,
`partial match only`, `pose/curl differs`, `possible duplicate frame`. Editable list in Settings
(§18). (`multi‑animal frame` is dropped — those frames are auto‑rejected before review, §7.1; a
site chip is omitted in single‑site v1.)

### 9.2 Batches / surveys

- There is exactly one **open** batch at a time; every upload attaches to it (default name e.g.
  "Autumn 2026 survey"; the user can rename it or open a new one, which closes the previous).
- A sighting **stays in its batch regardless of decision state** — undecided/unclassified sightings
  remain listed in the batch (and its counts) until they are decided.
- Decisions are **reversible until the batch is published**. Publishing:
  - freezes decisions,
  - promotes `provisional` individuals to `published` (any sighting still `flagged_uncertain` is held
    back with its individual until it is resolved),
  - stamps the census numbers,
  - does **not** require every sighting to be decided — undecided ones simply aren't counted,
  - is itself undoable only by an explicit "unpublish" with an audit entry.
- The **published census** = confirmed sightings in published batches, counting `published`
  individuals. Provisional/flagged/disqualified/undecided never count.

### 9.3 Notifications & alerts

Raised by services, surfaced on the **bell** (top‑right, badge count) and the Dashboard alert
band. Types + toggles in Settings (§10.11):

| Type | Trigger | Payload extras |
|---|---|---|
| `training_complete` | a training run finished | which model, deltas vs production, "Review in Models →" |
| `model_auto_promoted` | promotion rule changed the active model on its own | from→to, metric |
| `extraction_llm_low` | remaining daily call budget < threshold | **copy‑pastable** text block for the research team (canvas Upload toast) |
| `judge_llm_low` | same, judge model | same |
| `hand_correction_spike` | a batch's failure rate ≫ recent average | batch id, rate |
| `update_available` *(post‑v1)* | newer image published to GHCR (§2.1) | version, link |

Copy‑pastable text format (canvas): `Salamander Spotter: extraction LLM (<model>) low on tokens
— ~<n> calls left, resets <time>. Reported by <reviewer>, <timestamp>.`

### 9.4 Exports & the season census report

From Names and the Dashboard:

- **CSV / XLSX** of the roster (individuals × sightings, sites, dates, contributors) — canvas
  Names toolbar.
- **Sightings CSV** for one individual — canvas Individual‑image‑set.
- **Season census report** (PDF + XLSX) — the artefact for the biology team (WF3):
  cover (site, date range, batch), headline counts (individuals, new this season, images,
  contributors), per‑site breakdown, the map as a static image, the roster table, methods note
  (which model, thresholds, that counts are provisional vs published), and an appendix of
  flagged/uncertain items. Generated into `<data-dir>/exports/`.

### 9.5 Keyboard shortcuts (review)

`1` confirm · `2` new · `3` uncertain · `4` disqualify · `S` skip · `←/→` prev/next sighting ·
`E` open extraction editor for the suggested candidate · `L` toggle Cards/Match‑lines · `?` help.

### 9.6 Connectivity

Offline operation is **out of scope** (§1.2, §18). The server is assumed online. If the LLM provider
or map‑tile provider is unreachable, extraction/tiles fail with a clear error and retry; there is no
offline queue and no bundled‑tile fallback.

### 9.7 Resumable roster mutations (rename / merge / split)

A rename or merge touches three stores that cannot be updated in one transaction — `app.duckdb`,
`contours.db`, and image files on disk (`images/raw|thumb|purple/<image_id>.<ext>`). To survive a crash
mid‑way (§18) each such mutation is a **journaled, resumable job**:

1. The operation is planned up front and written to a `mutations` row (`state=planned`, `plan_json`
   listing every old→new id and every file move).
2. It then advances through explicit, **idempotent** states, persisting a `cursor` after each step:
   `renaming_files → updating_contours → updating_appdb → reindexing → done`.
3. Files are renamed via a **two‑phase, collision‑free scheme**: each target is first written as
   `<newid>.partial`, then atomically renamed to `<newid>` once its DB rows are updated; a half‑done id
   is always recognisable by the `.partial` suffix.
4. On startup the worker scans for any `mutations` row not in `done`/`failed` and **resumes from its
   cursor** — every step re‑checks "is this already applied?" so re‑running is safe.
5. `undo_json` captures the inverse plan, so a mutation stays reversible until the batch is published
   (§9.2).

The same journal backs promote/rollback and disqualify, so any half‑applied change is either completed
or cleanly rolled back on the next start.

---

## 10. Screen specifications

Route ⇄ screen map (canvas artboards in parentheses):

| Route | Screen | Canvas file |
|---|---|---|
| `/` | Overview / Dashboard | `Main.dc.html` |
| `/upload` | Upload images | `Upload.dc.html` |
| `/review` | Review queue | `Review.dc.html` |
| `/review/[imageId]` | Top match comparison | `TopMatchComparison.dc.html` |
| `/review/[imageId]/match-lines` | Match lines | `MatchLines.dc.html` |
| `/review/[imageId]/extraction/[candidateId]` | Extraction editor | `Extraction.dc.html` |
| `/names` | Names (roster table) | `Names.dc.html` |
| `/names/[individualId]` | Individual image set | `IndividualImageSet.dc.html` |
| `/names/[individualId]/[imageId]` | Image (annotated photo) | `Image.dc.html` |
| `/models` | Models tab | `Models.dc.html` |
| `/models/settings` | Models → Settings tab | `Settings.dc.html` |

Global chrome: 220px dark sidebar with 5 items — **Overview, Upload, Review, Names, Models** —
plus a nested sub‑nav under Review/Names that mirrors the current drill‑down (canvas shows
`Review › Top match comparison › Match lines`). Settings is the **second tab under Models**, not
a sidebar item. Top bar carries page title + context subtitle + the notification bell. Drill‑down
screens additionally show a URL pill, a breadcrumb, and a page‑type kicker.

---

### 10.1 Overview / Dashboard — `/`

**Purpose:** at‑a‑glance census state + what needs attention.

**Layout**

1. **Alert band** (dismissible) — the most important open notification (e.g. "Training complete
   — `set_transformer_v4` beats production on Novelty AUROC 0.71 vs 0.67 → Review in Models").
2. **Stat cards** (3): **Individuals**, **Images**, **Contributors**, each with a value, a
   "+N this month" delta, and a sparkline (Individuals card only, last ~7 periods).
3. **Two‑column lower area**:
   - **Recent activity** — feed of `activity` rows (training complete, new sighting from X,
     batch reviewed by Y, enrollment, merge…), each with icon, one‑line summary, sub‑line,
     relative time; "View all →" opens a full activity page.
   - **Sightings map** — MapLibre centred on the single Sasa site; marker kinds: sighting (blue dot),
     notable pattern (amber star), new‑in‑last‑Nd (red, pulsing). Legend bottom‑left. A "latest
     sighting" peek card (top‑right) with the newest unconfirmed sighting and "Review now →".
     "Open full map →" for a bigger view.
     If tiles fail to load, markers are shown on a plain background (no offline/schematic mode — §9.6).

**Data:** `GET /api/dashboard` → `{ stats, deltas, sparklines, activity[], map:{sites[],markers[]}, alert }`.

**Interactions:** dismiss alert (`POST /api/notifications/{id}/dismiss`); every card/panel links
into its section.

---

### 10.2 Upload images — `/upload`

**Purpose:** get photos in, extract them, grade them, and hand off to review in one sitting.

**Layout**

- **Dropzone** — drag‑and‑drop or click to browse; accepts JPG/PNG (and HEIC → converted);
  "WhatsApp exports welcome" (handles `IMG-*.jpg`, zipped chat exports → extracts images +
  best‑effort date from filename/EXIF).
- On drop: each file becomes a **card** in a grid, gets a provisional `image_id`, and an
  extraction job is queued. Cards show live stage progress: **Upload → Body → Spots** tracks.
- **Filter chips**: All / Auto‑accept / Needs a look / Hand‑correction / Failed — counts per
  tier.
- **Card** (per photo): thumbnail, provisional id, tier badge, quality scores
  (**Qual, Sharp, Light, Spots, Body** — `overall_quality` plus the four composites, incl.
  `lighting_quality`), pass/fail vs the quality cut‑off, and contextual actions:
  - Auto‑accept → "Will auto‑approve in Review — no action needed".
  - Needs a look / Hand‑correction / Failed → **Fix extraction** (opens the extraction editor
    for this image) and, below the cut‑off, **Use anyway** (with a "not recommended" tooltip;
    sets `use_anyway=true`).
- Before a large batch, a **cost estimate** modal (billed LLM calls, like `--dry-run`) with a
  confirm.
- **Metadata step** — per card or bulk: **contributor**, **site**, **photographed date**
  (defaults from EXIF), **batch**. Required before the sighting can enter review.
- **CTA bar**: "*N photos need a decision.* Review this batch now →" → `/review` filtered to
  this batch.
- **Toast**: extraction/judge LLM low on tokens, with the copy‑pastable block and
  "Copy for research team" / "Dismiss".

**Endpoints**

```
POST /api/uploads                 multipart; returns [{image_id, job_id}]
POST /api/uploads/estimate        {count} → {billed_calls_estimate, budget_remaining}
PATCH /api/images/{id}            {contributor_id, site_id, photographed_at, batch_id, use_anyway}
POST /api/images/{id}/re-extract  {overwrite?, model_ladder_override?}
GET  /api/uploads/batch/{id}      cards + tier counts (SSE mirrors progress)
```

---

### 10.3 Review queue — `/review`

**Purpose:** the worklist of sightings needing a human decision.

**Layout**

- **Auto‑approve band** (green): "N more sightings auto‑approved at ≥0.95 confidence this batch —
  they never landed here." Links to Settings.
- **Toolbar**: search (by ID or contributor), filter chips (All / Needs a look /
  Hand‑correction / Flagged), batch selector.
- **Table**: Incoming sighting (thumb + id + contributor), Site, Uploaded, Ladder tier,
  Top suggestion (`individual` + confidence), Quick actions (icons → Top match comparison /
  Match lines / Extraction).
- Row click → `/review/[imageId]` (Top match comparison). Pager, count ("Showing 5 of 18").

**Endpoints**

```
GET /api/review?batch=&filter=&q=&page=      → rows + auto_approved_count + totals
```

---

### 10.4 Top match comparison — `/review/[imageId]`

**Purpose:** decide one sighting against the Top‑N candidates.

**Layout**

- **Header**: breadcrumb `Review`, kicker "Top match comparison", the mono `image_id`, progress
  ("18 / 74 reviewed" + bar), prev/next, help.
- **Left rail**:
  - **View** — Layout: *Cards* ⇄ *Match lines* (→ `/match-lines`); Image ⇄ *Normalized*
    (body‑intrinsic straightened view); Match arcs: *All* ⇄ *Significant*; Overlays toggles:
    *spots, body, spine, anchors*.
  - **Matcher** — the active (serving) model + meta (in production, R@10, trained date). There is
    **no "compare all models"** view in v1 — a single model does all inference (§7.3).
- **Auto‑approve band** (as queue).
- **Stage**:
  - **Target card** (incoming): figure with overlays, quality scores + cut‑off pass, meta
    (contributor, photographed, site, spots extracted).
  - **Candidates** (Top‑N grid, default 3 wide): heading "Top 3 of 289 — covers 91% confidence"
    + "coverage target 90%, ranked by `<model>`". Each candidate card:
    - individual id + "suggested" tag (green ring) on the top pick, else `#rank`,
    - "K photos enrolled · since YYYY",
    - figure (best matching enrolled photo) with hover **"Extraction →"** (opens the editor for
      that candidate photo) and a confidence panel: **similarity** bar + **calibrated
      confidence** bar,
    - quality scores + cut‑off,
    - a pencil to the extraction editor.
  - **More bar**: "+2 more candidates below 0.12 confidence complete the 91% coverage.
    Show all 5 · or search all 289 individuals →" (opens a searchable all‑individuals picker).
- **Suggestion info band** (amber): "Suggested re‑ID `en_1` — confidence 0.68, above the 0.55
  threshold. Still worth a look: open‑set calibration on this dataset is weak (AUROC 0.59), and
  ~a fifth of spots vanish between sightings."
- **Decision bar** (sticky footer): **Confirm re‑ID `X`** (`1`), **New individual** (`2`),
  **Uncertain — flag** (`3`), **Disqualify** (`4`), **Skip** (`S`). "auto‑saves to review
  queue".
- **Override toast** as §9.1.
- After **New individual**: the enrollment result view (canvas `NewIndividual.dc.html`) —
  "Enrolled as a new individual", assigned id, the photo relabeled (`x1_3_2 → now aa_1_1`),
  reason chips in the rail, Top matches kept dimmed "for audit", "Next sighting →",
  "Undo — back to review", "Re‑open candidate search".

**Endpoints**

```
GET  /api/review/{imageId}                 → target, candidates[], suggestion, progress, model
POST /api/review/{imageId}/decision        {verdict, chosen_individual_id?, reason_chips[], note?, override_ack?}
GET  /api/individuals/search?q=             all-individuals picker
```

---

### 10.5 Match lines — `/review/[imageId]/match-lines`

**Purpose:** show *why* the matcher paired these two — the machine‑proposed spot correspondence.

**Layout:** two figures side by side (incoming vs suggested match photo), dashed red connectors
between corresponding spots labelled with per‑pair scores. Left rail: View (layout back to
*Cards*, overlays spots/body), and a **Spot matches** list (`SD‑1‑2‑01 → EN‑1‑3‑01  score 0.84`,
…). Caption explains red = machine‑proposed. Toggle *All* vs *Significant* arcs from the rail.

**Endpoints:** `GET /api/review/{imageId}/match-lines?candidate=<individualId>` →
`{ query_spots[], cand_spots[], correspondences[] }`.

---

### 10.6 Extraction editor — `/review/[imageId]/extraction/[candidateId]` and `/names/[id]/[imageId]` (edit mode)

**Purpose:** fix a bad extraction (spots, spine, head/tail, outline) without touching the re‑ID
decision.

**Layout**

- Header: breadcrumb + kicker "Extraction" + the mono photo id + close.
- Left rail:
  - **View** — Image ⇄ Normalized; Overlays: spots, body, spine, anchors.
  - **Edit tools** — *Select*, *Join spots* (merge two fragments into one), *Split spot*,
    *Draw spine* (redraw the midline polyline), *Select head / tail* (move the anchors),
    *Draw outline* (redraw the body mask boundary), *Delete spot*, *Add spot* (§18 —
    freehand vs LLM‑assisted).
- Canvas: the photo with editable overlays; a selected spot shows its callout (SSID, interest,
  head distance, side) and a colour swatch.
- Right column: **Quality assessment** (recomputed live after edits) + a context hint
  ("Spot 06 is selected — click a spot on another photo in this family to join them…").
- Footer: **Save extraction** / **Cancel**; note: "Editing does not change the re‑ID decision on
  `en_1`".

**Behaviour**

- Edits are applied to `contours.db` rows (spots re‑binned, quality recomputed) and recorded as
  an `extraction_corrections` row with a `prev_snapshot` for undo.
- Saving from the review flow returns to Top match comparison; the affected candidate's
  similarity/confidence is **re‑scored** and the card updates.
- Corrections feed the next training snapshot (they are higher‑value labels — see
  `next_steps_2.md` §6).
- Cross‑photo **spot correspondence** linking (join a spot in photo A to the same physical spot
  in photo B) reuses the `pipeline/correspondence` app's logic and **persists the human verdict** to
  `app.duckdb` `correspondence_labels` (§8). Together with the re‑ID decision (`review_decisions`) and the
  corrected geometry (`extraction_corrections` + the updated `contours.db` rows), this means **every
  human‑produced signal — matcher pick, chosen label, corrected extraction, and spot‑pair verdicts — is
  stored in the database** (§18).

**Endpoints**

```
GET  /api/images/{id}/extraction            spots, axis, mask, quality, overlays
POST /api/images/{id}/extraction            {ops:[{join|split|spine|head_tail|outline|add|delete ...}]}
POST /api/images/{id}/extraction/revert     to a prior snapshot
POST /api/correspondence                    {query_image, query_spot, other_image, other_spot}
```

---

### 10.7 Names — `/names`

**Purpose:** the roster; entry point for search & curation (WF2) and export (WF3).

**Layout**

- Toolbar: search (ID / nickname), export buttons **CSV** / **Excel**. (v1 is single‑site — §1.2 — so
  there is no site filter; the control returns when a second site exists.)
- Table: **Individual** (avatar + display id + nickname), **Health** (🟢/🟡/🔴 training‑value dot —
  §7.8, hover for the reason), **Site**, **Sightings** (count), **First seen**, **Last seen**,
  **Contributors**, chevron. A "new" tag on individuals enrolled this batch. Sort by any column
  (incl. Health). Pager ("Showing 5 of 68").
- Row click → `/names/[individualId]`.
- Status facets (§18): show/hide provisional, flagged (**Uncertain**), merged. An **Uncertain** sighting
  can be re‑decided here without going through the review flow (§9.1).
- A dismissible **hints band** rolls up the §7.8 suggestions ("12 individuals would benefit from
  synthetic views", "eval set is thin — consider Re‑split").

**Endpoints:** `GET /api/individuals?site=&q=&status=&sort=&page=`;
`GET /api/export/individuals.csv` / `.xlsx`.

---

### 10.8 Individual image set — `/names/[individualId]`

**Purpose:** everything about one confirmed animal.

**Layout**

- Header: breadcrumb `Names ›`, kicker "Individual image set", display id + **editable
  nickname** (pencil), sub ("4 sightings · Sasa, north slope"), prev/next through the roster.
- **Reference card** (left): the highest‑quality confirmed photo, quality scores, meta (first
  seen, last seen, contributors, spots tracked). "Open image →".
- **All photos** grid: every confirmed photo + synthetic views (`_g<k>`, tagged "synthetic",
  no quality scores, `q` shows `✓ᵍ`), sorted most‑recent‑first; each with date bar and "Open
  image →". An **"Link another sighting"** add‑card.
- Footer actions: **Rename** (change display id / label — cascades to all photo ids and
  `contours.db`; audited), **Merge with…** (pick another individual → combine photo sets, keep
  the surviving id, `merged_into` on the other; reversible pre‑publish), **Flag for re‑review**
  (sends every photo back to the review queue), **Export sightings** (CSV).
- Split (§18): "Move photo to…" on a photo → detach into its own / another individual.

**Endpoints**

```
GET   /api/individuals/{id}                 profile + photos[]
PATCH /api/individuals/{id}                 {nickname?, display_id?, site_id?}
POST  /api/individuals/{id}/merge           {into_id}
POST  /api/individuals/{id}/flag            → re-queues photos
POST  /api/individuals/{id}/photos          {image_id}   link a sighting
DELETE/api/individuals/{id}/photos/{imageId}             unlink / move
GET   /api/export/individuals/{id}/sightings.csv
```

---

### 10.9 Image — `/names/[individualId]/[imageId]`

**Purpose:** one fully annotated photo, read‑only, with a route into correction.

**Layout**

- Header: breadcrumb `Names › AC‑3`, kicker "Image", mono photo id, close.
- Left rail: **View** (Image ⇄ Normalized), **Overlays** (spots, body, spine, anchors),
  **This individual‑image‑set** filmstrip (all photos of the set, current highlighted).
- Canvas: the photo with overlays; clicking a spot shows its callout (SSID e.g. `AC‑3‑1‑06`,
  Interest 0.44, Head 0.47 (312px), Side right) and swatch; legend (spot / spine / head / tail).
- Right column: **Quality assessment** (Passed/Failed + the five scores incl. **Light**) and **Details** (contributor,
  photographed, site, spots extracted, reviewed by).
- Footer: **Suggest a correction** (→ extraction editor in edit mode), **Download photo**.
  "Read‑only — corrections open this image in edit mode."

**Endpoints:** `GET /api/images/{id}` (full detail); `GET /media/raw/{id}`,
`GET /media/overlay/{id}?layers=spots,spine`, `GET /media/normalized/{id}`.

---

### 10.10 Models — `/models`

**Purpose:** see which matcher is live, how candidates compare, retrain, promote/rollback.

**Layout**

- Header actions: **Download `model_stats.log`**, **Retrain all now**.
- Tab bar: **Models** (this) · **Settings**.
- Info chips: Last full run · Images used (`1,421 · full dataset`) · Next scheduled run
  (`in 12 days, or +49 images`).
- **Model table** with sub‑tabs **Current / Trained / Best**, columns: Model (+ status badge
  `active` / `candidate`), Trained, **R@1, R@5, R@10, Bal. acc, Novelty AUROC, Review@90,
  Score**; ranked by Score; "Top 4 of 13 · 9 more incl. `ensemble_vote` (training now)". Row
  click selects a model for the promotion panel.
- **"Best fit to your corrections"** card — surfaces when the reviewer's recent hand‑picks
  disagree with the active model ("6 of your last 8 picks favoured a different model · avg rank
  #5.4 vs the model's #1.8"). It can **pick the model that best fits your hand‑picks** — either the
  registered model whose ranking best matches them, or a **simple linear regression over your recent
  inputs** — and offers to make it active (or adjust the Score formula). This is the only "multi‑model"
  surface; live inference still runs a single model (§7.3).
- **Training** panel: "Every retrain uses the full dataset." Controls: *Retrain every N days*,
  *or every M new reviewed images*, *Train/eval split* (Keep current / Re‑split).
- **Promotion** panel (the consequential one): **1 Select model** (dropdown or table row) →
  **2 Promote** ("Make active" — replaces the current active immediately) → **3 Rollback**
  ("Restore previous best" — back to the prior active's run). Every promotion/rollback is
  audited and notified.

**Endpoints**

```
GET  /api/models?view=current|trained|best
POST /api/models/retrain                    {which:"all"|name} → job_id
POST /api/models/{name}/promote
POST /api/models/rollback
GET  /api/models/fit-to-corrections
GET  /api/models/stats-log                  latest model_stats.log
```

---

### 10.11 Models → Settings — `/models/settings`

Sections (canvas `Settings.dc.html`):

**Pipeline**
- **Extraction model** — Provider (Gemini / OpenAI / Anthropic / Azure / Local); API token
  (masked, "Update"); **Model ladder** — ordered list tried in sequence on error/timeout/budget
  ("gemini‑2.5‑pro → gemini‑2.5‑flash → gemini‑2.0‑flash"), add/remove steps.
- **Judge model** — same, with a mandatory final **rule‑based (no LLM)** step so scoring never
  fully stops.
- **Quality cut‑off** slider (default `0.40`) — below = training‑data‑only, not a review query
  (per‑photo "Use anyway" still available).
- **Min spots for auto‑accept** (`min_spots_auto_accept`, default `3`) — a photo needs at least this
  many extracted spots (with a usable axis) to qualify for the Auto‑accept tier (§7.2).

**Notifications** — toggles: Training complete · Extraction LLM low on tokens · Judge LLM low on
tokens · Unusual spike in hand‑correction‑needed photos · A model is auto‑promoted to active.

**Review & automation**
- **Auto‑approve threshold** slider (default `0.95`).
- **Candidate coverage** (default `90%`) + **Max candidates shown** (default `6`).
- **Active matching model** — *Manual pick* ⇄ *Optimize by metric* ⇄ *Best fit to my corrections*.
  When optimizing, a metric dropdown (Novelty AUROC / Review@90 / Balanced accuracy / R@10 / **Custom
  formula**); **Custom formula** is coefficients on the lettered metrics `a`–`f` (§7.4), e.g.
  `0.3a + 0.6c + 0.1e`, parsed into a coefficient map — no free‑form expressions. *Best fit to my
  corrections* selects the model (or a simple linear fit) that best matches your recent hand‑picks
  (§10.10). Whatever is chosen, exactly one model serves inference. Synced with the Models tab.
- **Warn before assigning against the model's suggestion** toggle (default on).

**Also on this screen (not in the canvas, needed for the app — §18):**
- **Data directory** (read‑only display + "Reveal in file manager").
- **Sites** editor (name, code, lat/lon) — feeds the map. Sasa is the only site in v1; the census code
  (§6.1) is site‑less.
- **Contributors** editor.
- **Map tiles** — provider + key.
- **Email** (optional) — SMTP host/port/user/pass + recipient list for `model_stats.log` (WF4).
- **Daily LLM call budget** + low‑budget warning threshold.
- **Reason chips** list editor.
- **Expose API docs** toggle (default off) — serves `/api/docs` when on (§11).

**Endpoints:** `GET /api/settings`, `PATCH /api/settings`, `POST /api/settings/token`
(provider, value), `POST /api/settings/test-llm` (provider ladder reachability),
`POST /api/settings/test-email`.

---

### 10.12 Workflows (end‑to‑end)

**WF1 — Image upload, review & approval** (canvas `WorkflowUpload`)
WhatsApp photo → dropped into Upload (one sitting) → pipeline extracts body/spots, judge scores →
**ladder gate**: "needs a look" or better? → if not, guided hand‑correction (fix, or "Use
anyway") → auto‑matched against enrolled → candidates ranked, Top‑N to coverage → **auto‑approve
gate**: ≥ threshold → auto‑confirmed to Dashboard, else Top match comparison → reviewer decision
(confirm / new / uncertain; override warns first) → Individual set + Dashboard + map refresh.

**WF2 — Salamander search & review** (canvas `WorkflowSearch`)
Names → filter (ID/site/date/contributor) → row → Individual image set → photo → Image screen →
issue? → *no issue* keep browsing · *wrong spot/axis* edit & save in place · *wrong individual*
flag → returns to Review.

**WF3 — Statistics & knowledge gathering** (canvas `WorkflowStats`)
Dashboard (counts / activity / map) → Names roster or a map marker → filtered sightings →
Export CSV/Excel → **season census report** shared with the biology team.

**WF4 — Model update & analysis** (canvas `WorkflowModel`)
Trigger (N days elapsed **or** M new reviewed images) → retrain matcher on the full reviewed
snapshot → evaluate (R@1/5/10, bal. acc, novelty AUROC, review@90) → write to Candidate/Trained
table → beats Best on the selected metric? → **promote** (previous kept for rollback) or keep
current → generate `model_stats.log` → email it to the research team → remote reviewer:
**approve** (confirmed in Models on next visit) / **rollback** (one click) / **hold** (no action).
Training never self‑promotes silently: the log is always written and the notification always
raised.

---

## 11. API surface (summary)

Base `/api`. JSON, Pydantic‑typed, OpenAPI at `/api/docs` (exposure toggled in Settings, default
off). SSE at `/api/events`.

```
Dashboard     GET  /dashboard
Uploads       POST /uploads · POST /uploads/estimate · GET /uploads/batch/{id}
Images        GET/PATCH /images/{id} · GET /images/{id}/extraction
              POST /images/{id}/extraction · POST /images/{id}/re-extract
Review        GET  /review · GET /review/{id}
              GET  /review/{id}/match-lines · POST /review/{id}/decision
Batches       GET/POST /batches · POST /batches/{id}/publish · POST /batches/{id}/unpublish
Individuals   GET /individuals · GET/PATCH /individuals/{id} · POST /individuals/{id}/merge
              POST /individuals/{id}/flag · .../photos · GET /individuals/search
Models        GET /models · POST /models/retrain · POST /models/{name}/promote
              POST /models/rollback · GET /models/fit-to-corrections · GET /models/stats-log
Training      GET /training-runs · GET /training-runs/{id}
Settings      GET/PATCH /settings · POST /settings/token · POST /settings/test-llm · /test-email
Sites/Contrib GET/POST/PATCH /sites · /contributors
Notifications GET /notifications · POST /notifications/{id}/read · /dismiss
Jobs          GET /jobs · GET /jobs/{id} · POST /jobs/{id}/cancel
Export        GET /export/individuals.{csv,xlsx} · /export/individuals/{id}/sightings.csv
              POST /export/census-report → {job_id} → GET /exports/{file}
Media         GET /media/raw/{id} · /media/thumb/{id} · /media/purple/{id}
              /media/overlay/{id}?layers= · /media/normalized/{id}
Meta          GET /health · GET /version    (self-update / update-check: post-v1, §2.1)
```

---

## 12. Configuration reference

`config.toml` (non‑secret) — thresholds, ladders (model names only), notification toggles,
sites, contributors, map provider, email host, budgets, data‑dir marker.
`secrets.json` (0600) or OS keychain — provider API tokens, SMTP password, map‑tiles key.
Env overrides (for source runs / CI): `SPOTTER_DATA_DIR`, `SPOTTER_PORT`, `SPOTTER_BASE_URL`,
`SPOTTER_GEMINI_API_KEY`, `SPOTTER_LOG_LEVEL`.

Defaults: port `8756`, auto‑approve `0.95`, match threshold `0.55`, quality cut‑off `0.40`, min spots for
auto‑accept `3`, coverage `0.90`, max candidates `6`, retrain `every 60 days or 100 new reviewed images`,
score formula `{a: 0.5, e: 0.5}` (= `0.5·R@1 + 0.5·Novelty AUROC`, §7.4), warn‑on‑override `on`, daily LLM
budget unset (warn at 40 remaining). **All timestamps are stored UTC and displayed in the browser's local
time** (§18); budget/backup "resets/staleness" are computed in UTC.

---

## 13. Design system (from the canvas)

- **Palette:** ground `#faf8f5`, ink `#33302c`, panel `#fff`, sidebar `#2b2825`, accent
  (terracotta) `#b9782a`; semantic green `#2f7d4f` (confirm/promote), amber `#8a5a1e`
  (new/needs‑a‑look), blue `#3f5f9c` (flag/info), red `#a5443b` (disqualify/rollback). Full
  token set is in every `*.dc.html` `:root`.
- **Type:** Hanken Grotesk (headings/UI), IBM Plex Mono (IDs, scores, SSIDs, URLs), system sans
  (body).
- **Motion:** hover lift on every card (`translateY(-2px)` + soft shadow), 120–160ms eases.
- **Icons:** lucide (line, ~1.7 stroke).
- **Decision colour law:** green confirms/promotes, amber is new/needs‑a‑look, blue/red is
  flagged/reverted, dashed grey is terminal/deferred — applied consistently in flows and badges.
- **Theme:** light only in v1 (the canvas commits to one look). Respect `prefers-reduced-motion`.
- Build these as reusable React components: `Sidebar`, `TopBar`, `UrlPill`, `Breadcrumb`,
  `StatCard`, `Panel`, `SalamanderFigure` (SVG overlay renderer), `QualityScores` (overall + 4 composites incl. lighting), `TierBadge`,
  `CandidateCard`, `DecisionBar`, `ConfidenceBars`, `MatchLinesOverlay`, `ExtractionCanvas`,
  `ModelTable`, `PromotionPanel`, `Toast`, `AlertBand`, `SightingsMap`.

---

## 14. First‑run & help

### 14.1 First‑run setup

Runs as a **web flow** the first time the app starts against an empty `/data` volume (no desktop
wizard, no data‑dir picker — the volume is fixed by the container, §5.2). Steps:

1. Welcome.
2. **API keys** — extraction + judge provider and token; "Test" button; skippable (extraction disabled
   until set).
3. **Contributors** — seed from `config.toml` defaults or add (site is fixed to Sasa, §6.1).
4. **Model weights** — confirm the model(s) already present in `/data/models/`, and pick the active one.
   (No download step — weights are placed on the volume by the maintainer, §7.6.)
5. **Import existing data** (optional) — point at a mounted `datasets/all_sasa_norm_*/` folder → a
   **one‑time transfer** (§7.9 A): raw images copied, `contours.db` copied verbatim, `app.duckdb` rows
   derived from filenames + joins, `corrections.json` replayed. **No extraction, no LLM calls.**
   Re‑runnable; a newer snapshot later imports only its delta.
6. Done → Dashboard.

### 14.2 In‑app help
A `?` on drill‑down screens opens a slide‑over explaining that screen (what the scores mean,
what each decision does, why calibration is weak on this dataset). Content authored from
`docs/project_goal.md` + `next_steps_2.md`.

---

## 15. Security & privacy

- **The app binds `127.0.0.1:8756`** and is only reachable through the tunnel/proxy edge, which is the
  sole access gate in v1 (§1.3, §2.1). There is **no in‑app authentication** (§18); if a network surface
  beyond the edge is ever needed, accounts become a separate milestone.
- **CORS**: same‑origin only. A random **CSRF token** in a cookie is required on mutating requests, to
  block drive‑by requests from other browser tabs.
- **Secrets** (LLM tokens, SMTP password, map key) never logged, never sent to the frontend (only
  "set / not set" + last 4 chars). Stored in `secrets.json` (0600) on the `/data` volume, or the OS
  keychain (`keyring`) when available.
- **Outbound traffic** is limited to: the configured LLM provider(s), the map‑tile provider, the Docker
  registry / GitHub (update check + image pull), and optional SMTP. All are disclosed in Settings.
- **Photo data** stays on the server except the pixels sent to the LLM for extraction — this is
  disclosed in first‑run setup and Settings, and a fully local extraction model can be used instead.
- The **Docker image** is built in CI; releases are tagged and, post‑v1, pulled by the remote for
  self‑update (§2.1).

---

## 16. Testing & QA

| Layer | Approach |
|---|---|
| Backend unit | `pytest` on services (decision engine, ladder→tier, Top‑N/coverage, ID allocation, merge/rename cascades, census counting). |
| Pipeline bridge | contract tests against a tiny fixture dataset (5 images) checked into the repo. |
| API | `pytest` + `httpx` against the app with a temp data dir; golden OpenAPI schema. |
| Frontend unit | Vitest + React Testing Library on components (QualityScores, DecisionBar, ConfidenceBars, ModelTable). |
| E2E | Playwright against `pixi run app:serve` + fixture data: full WF1 (upload → extract stub → review → confirm), WF2 (search → correct), WF4 (retrain stub → promote → rollback). |
| Packaging | CI job runs the built **Docker image** on a clean runner, hits `/api/health` and loads `/`. |
| Performance targets | roster of 1,000 individuals / 5,000 images: Names loads < 400 ms; Top match comparison < 800 ms after match job; extraction of one photo < 25 s (LLM‑bound); UI stays at 60 fps on the extraction canvas with ~40 spots. |

LLM calls are **mocked by default** in tests via a recorded‑cassette fixture; a nightly job runs
a handful against the real provider.

---

## 17. Execution plan

Sequenced milestones. Each ends with a **demoable build** and its acceptance check. Rough
size only — adjust in planning.

### 17.0 Implementation status (2026-09-04)

A full first pass of **M0–M7** landed in one sweep. What exists:

| Area | State |
|---|---|
| `app/` backend — config, IDs, DuckDB schema (`0001`+`0002`), single-writer handle | done |
| §7.9 dataset transfer + incremental ingest | done, 0-LLM, idempotent, delta-capable |
| Background worker + `jobs` table + SSE `/api/events` (§5.3) | done |
| Matching (`run_match`, Top-N/coverage, calibrator), decision engine, auto-approve, override guard, batches, census (§7.3, §9.1, §9.2) | done |
| Extraction editor (join/split/head-tail/re-bin/re-score/revert) + `correspondence_labels` (§10.6) | done |
| Model registry, Score formula (linear combo of `a`–`f`), training runs, promote/rollback, scheduler (§7.4, §7.5) | done |
| Register an already-trained checkpoint from disk (`app import-model` / `POST /api/models/import` / Models-page panel) — copies weights onto the volume, catalogues metrics, no retrain (§7.6) | done |
| Cost estimate + daily LLM budget + low-token notification (§7.1, §10.2) | done |
| Season census report — XLSX + PDF + CSV (§9.4) | done |
| Notifications + SMTP (§9.3) | done |
| `app` CLI: `serve migrate import-dataset backup restore export import` (§5.4) | done |
| Backup / restore / full export-import (§2.2, §2.3) | done |
| `deploy/` — Dockerfile (multi-stage), compose (+ Caddy + backup sidecar), Caddyfile, systemd unit, `.env.example`, `deploy.sh`; `.github/workflows/release.yml` (test → image smoke → GHCR push) | done |
| `web/` — Next.js `output:'export'`, Tailwind design tokens (§13), TanStack Query; screens: Dashboard, Upload, Review (queue + top-match + match-lines + extraction editor), Names (roster + individual + image), Models, Models→Settings | done (builds to `web/out/`) |
| **Tests: 165 (`pixi run app:test`)** — unit + integration, real tiny `contours.db` fixture, no network/billing | done |

**ML bridges are seams (§7):** `app/pipeline_bridge/*` defines `ExtractionBridge`,
`MatchingBridge`, `CorrespondenceBridge`, `EditorBridge`, `TrainingBridge`. The `Fake*`
implementations are the tested path. `PipelineExtractionBridge`/`PipelineTrainingBridge`/
`PipelineEditorBridge` remain stubs (`NotImplementedError`) — real wiring is M2/M5 work,
gated on billed LLM calls / GPU time this pass didn't spend.

**Matching (§7.3) was attempted and deliberately rolled back to the stub, on evidence, not
by default.** `PipelineMatchingBridge` / `PipelineCorrespondenceBridge` (in
`app/pipeline_bridge/matching.py`) are real, tested, working code: soft-chamfer
(mean-best-cosine) over the per-spot embeddings already sitting in an imported
`contours.db`'s `spot_embeddings` table (written once by `pipeline/spot_transformer/
core/embeddings.py`; verified the additive `(cos_shape+cos_pos)/2` reproduction against
the table's own vector norms) — no GPU, no LLM call, no re-embedding at request time.
**Validated against the live 743-individual dataset it ranks the true match at population
chance** — median rank ≈ `gallery_size/2` at gallery sizes 5, 20 and 60 alike (a real
discriminator would show a rank advantage that *shrinks* relative to chance as the gallery
grows, not track it exactly) — consistent with the pipeline's own docs calling plain
soft-chamfer "the permissive score" a stricter scorer exists to improve on, not a
deployable matcher. Serving it as the active matcher would produce confident-looking but
essentially random review suggestions, which is worse than the honest 501 it replaced, so
`Bridges.production()` (`app/bridges.py`) keeps the new `StubMatchingBridge` /
`StubCorrespondenceBridge` as default; the real implementation is reachable only via
`Bridges.experimental_matcher()`, clearly labelled not to be trusted for identification.
**What a trustworthy matcher needs next** (none attempted, all bigger than this session):
`strict_pair_score` (`pipeline/spot_transformer/core/strict_match.py`) with real
distinctiveness weighting + the position gate (needs a `body_axis`/`image_quality` join
for body-width normalisation this pass didn't build), evaluated the way this repo's own
harness insists on — proper CV folds, a frozen eval split, calibration — not an ad-hoc
gallery sample; or the fold-trained Set Transformer, whose deployment path
(`scripts/embedding/emb_identify.py`) is explicitly unbuilt in the pipeline itself
("planned for Phase 4").

**Deviations from this document (intentional):**
- Non-secret settings live in a `settings` DuckDB table, not `config.toml` (one transaction, visible over SSE). Secrets still in `secrets.json` (0600).
- `images.source_checksum` column added as the ingest idempotency key.
- Frontend drill-downs use query strings (`/review?id=…`, `/names?id=…&img=…`) instead of path params, because `output:'export'` cannot pre-render unknown dynamic segments; a FastAPI catch-all still serves the SPA for any path.
- `corrections.json` merges/exclusions are applied to `individuals`/`images` + `audit` (exclusions also write an `extraction_corrections` row).
- Frontend Vitest/Playwright suites (§16) not yet written.

### M0 — Skeleton & packaging proof  *(foundational)*
- Repo layout (§3.1); `pixi.toml` gains `app:*`, `ui:*`, `docker:*`.
- FastAPI app factory, `127.0.0.1` + port pick + lockfile; SPA fallback; SSE stub; CSRF token.
- Next.js `output: 'export'` scaffold with the design system shell (sidebar, top bar, tokens,
  fonts) and a **static Dashboard** wired to `GET /api/dashboard` (mock data).
- Dockerfile (multi‑stage) building a runnable image that serves the Dashboard from the bundled backend.
- GitHub Actions: build + push the Docker image on tag.
- **Accept:** run the image on a clean box (`docker compose up`), Dashboard renders from the bundled
  backend.

### M1 — Data layer & roster (read‑only)
- `app.duckdb` schema + migrations; data‑dir resolution + first‑run setup (API keys, contributors, import).
- **Transfer** an existing `datasets/all_sasa_norm_*` per §7.9 A — copy raw images, copy `contours.db`
  verbatim, derive `app.duckdb` rows, replay `corrections.json`; zero LLM calls; idempotent + delta‑capable.
- **Names**, **Individual image set**, **Image** screens (read‑only), with real overlays
  (`SalamanderFigure` rendering spots/body/spine/anchors from `contours.db`).
- CSV/XLSX roster export.
- **Accept:** import the 23_07 dataset; browse 751 individuals; open any photo with correct
  overlays; export matches a hand‑checked sample.

### M2 — Upload & extraction
- Dropzone, WhatsApp export handling, batch model, metadata step.
- `pipeline_bridge` for the full extraction pipeline; background worker + job table + SSE
  progress; cost estimate/`--dry-run`.
- Quality ladder → tier; Upload cards with the quality scores, filters, "Fix extraction" / "Use anyway".
- Settings: Pipeline section (providers, tokens, ladders, quality cut‑off) + `test-llm`.
- Low‑token notification + copy‑pastable toast; daily budget.
- **Accept:** drop 10 real photos, watch them extract, land in the right tiers; a forced
  provider error falls through the ladder.

### M3 — Matching, review queue, decisions
- `pipeline_bridge` for the active matcher + calibration; match job; Top‑N/coverage.
- **Review queue**, **Top match comparison**, decision engine, auto‑approve, override guard,
  reason chips, **New individual** enrollment + relabel.
- Batches: open/switch, reversibility, publish/unpublish, census counting.
- Settings: Review & automation section.
- **Accept:** WF1 end‑to‑end on a held‑out set; auto‑approve rate and queue counts match a
  manual calc; enrolling a new individual relabels the photo and updates counts.

### M4 — Match lines & extraction editor
- Match‑lines overlay from the matcher's spot assignment.
- Extraction editor: select / join / split / spine / head‑tail / outline; live re‑bin + quality
  recompute; `extraction_corrections` + revert; re‑score the affected candidate on save.
- Cross‑photo correspondence linking (reuse `pipeline/correspondence`).
- **Accept:** fix a fragmented‑spot photo (canvas #2202‑style), save, see its tier and a
  candidate's confidence change.

### M5 — Models & training
- Models table (Current/Trained/Best), info chips, **Best fit to corrections**.
- Training run as a subprocess: snapshot → retrain all → eval → Trained table → Best recompute.
- Promotion / rollback (audited, notified); scheduler (N days / M images); "Retrain all now".
- `model_stats.log` generation + download; optional SMTP email; WF4 remote approve/rollback/hold.
- Settings: Notifications section; Email.
- **Accept:** trigger a retrain on a small snapshot; a synthetic "better" candidate auto‑promotes
  with a notification; rollback restores in one click; `model_stats.log` is well‑formed.

### M6 — Dashboard, map, census report
- Activity feed from `activity`; full alert band wiring.
- Sightings map (MapLibre; plain marker layer on tile failure); marker kinds; marker → filtered Names.
- **Season census report** (PDF + XLSX) generator.
- **Accept:** WF3 produces a report the biology team can read without the app.

### M7 — Release hardening
- Auto‑update banner; first‑run web setup complete; model weights confirmed on the volume.
- Full Playwright E2E; packaging smoke test in CI (run the Docker image); performance pass.
- `arm64` image variant (§5.2).
- User guide (`docs/running.md` refresh) + in‑app help content.
- **Accept:** a fresh box pulls the image, runs the four workflows unaided from the in‑app help.

### Dependency order

```
M0 ─► M1 ─► M2 ─► M3 ─► M4
                    └─► M5 ─► M6 ─► M7
```

M4 and M5 can overlap once M3 lands. M6 needs M3 (counts) + M5 (which model produced them).

---

## 18. Decisions & open questions

### 18.1 Resolved this round

| # | Decision |
|---|---|
| D1 | **No in‑app auth in v1.** Access is gated only at the tunnel/proxy edge; no accounts/sessions/roles. Optional actor label for attribution. Roles are a later option. |
| D2 | **Single‑site (Sasa) ID scheme.** The leading `<code>` is *not* a site code. New individuals get their own **two‑letter code** (next free combo, e.g. `aa_1`), expanding to three‑or‑more letters when all 26×26 two‑letter codes are used. Imported prefixes preserved & skipped (§6.1). |
| D3 | **No offline mode** and **no PyInstaller/local desktop build.** Server is assumed online; if it ever runs on the reviewer's PC it runs the same Docker image. |
| D4 | **Docker is the only artefact.** Host is a **DigitalOcean VPS**. v1 = build image, deploy + update **manually**. Post‑v1: CI pushes the image to **GHCR** (public pulls are free & anonymous) and the remote **self‑updates** by pulling it. |
| D5 | **Resumable roster mutations** (rename/merge/split) via a journaled `mutations` table + `.partial` file scheme; resume‑from‑cursor on restart (§9.7). |
| D6 | **One active model does all inference** (picked on Models or Settings). Match‑lines correspondences come from an always‑available geometric matcher for visualisation only. *(Superseded re: agreement dots — removed in v1, see D21.)* (§7.3). |
| D7 | **Score = linear combination of lettered metrics** `a`–`f` (§7.4), stored as a coefficient map (e.g. `0.3a + 0.6c + 0.1e`). No expression parser. |
| D8 | **Training is fully separated from serving:** runs in a subprocess against a snapshot; the serving model only changes on explicit **promote** (§7.5). |
| D9 | **`min_spots_auto_accept`** added to Settings (default `3`) for the Auto‑accept tier (§7.2). |
| D10 | **Multi‑animal frames are auto‑rejected** (`disqualified`, reason `multi‑animal frame`) — never queued (§7.1). |
| D11 | **Synthetic views are opt‑in and on demand** — suggested for singletons only; the user picks which individuals and the view attributes; training‑only, never a candidate/query (§7.7). |
| D12 | **Train/eval:** a frozen, real‑only base eval set; new individuals default to train; re‑split only on purpose (§7.8). |
| D13 | **Backups simplified:** nightly tar to a second location; encryption **optional/off by default** (key on the maintainer's machine, not the server) (§2.2). |
| D14 | **No model download from GitHub** — weights are placed on the `/data` volume by the maintainer (§7.6). |
| D15 | **HEIC deps added** (`pillow-heif`); `pyinstaller` removed from deps (§5.1). |
| D16 | **Timezone:** all timestamps stored **UTC**, displayed in the browser's local time (§12). |
| D17 | **Runtime SPA config renamed** `/config.json` → `/runtime-config.json` (§4.2). |
| D18 | `contours.db` schema **unchanged** (see 18.3 for a reminder of the current schema). Admin‑lockout recovery dropped (no auth). |
| D19 | **Single site (Sasa)** in v1 and the foreseeable future; `site_id` stays in the schema for a possible version‑N extension, but v1 assumes one site (no site filter, no site‑marker filtering) (§1.2, §10.7). |
| D20 | **Model status tags** `active\|candidate\|best\|archived`; each run saves/tags the latest full batch, the new candidates, and the best past version per model type — untagged models are discarded (§7.5). |
| D21 | **No "compare all models" and no agreement dots** in v1 — a single model serves inference. Model choice can still be driven by **best fit to corrections** (matching model, or a linear fit over hand‑picks) (§7.3, §10.10, §10.11). |
| D22 | **Provisional upload IDs** `up_<shortid>`, renamed on decision to `<label>_<instance>` (§6.1). |
| D23 | **No second/naming review gates** (`second_review_ok`/`naming_pass_ok` dropped). Uncertain sightings wait in an **Uncertain** category, resolvable from Review or Names (§9.1, §9.2, §10.7). |
| D24 | **All human signals stored in the DB:** re‑ID decision, chosen label, corrected extraction, and spot‑correspondence verdicts (`correspondence_labels`) (§8, §10.6). |
| D25 | **Batches keep undecided sightings** — they stay in the batch list/counts; publishing doesn't require every sighting decided (§9.2). |
| D26 | **`lighting_quality` is shown** — the quality display is `Qual, Sharp, Light, Spots, Body` (§10.2). |
| D27 | **Update‑check / self‑update deferred to post‑v1** (§9.3, §11). |
| D28 | **`app` is the single console entrypoint** (`app serve\|migrate\|backup\|restore\|export\|import\|import-dataset`) on the FastAPI factory (§5.4). |
| D29 | **API‑docs exposure is a Settings toggle**, default off (§3, §10.11, §11). |
| D30 | **The pipeline is never re‑run wholesale (§7.9).** The existing Sasa dataset is *transferred* into `/data` — raw images copied, `contours.db` copied verbatim, `app.duckdb` rows derived from filenames + joins, `corrections.json` replayed — with **zero LLM calls**. New photos are extracted **one at a time and appended**; `contours.db` only grows by append (edited only per‑photo via the extraction editor). There is **no dataset‑rebuild action** in the app; `scripts/dataset/build_dataset.py`'s full rebuild stays a pipeline‑dev tool. Training runs read a snapshot of the live DB and never invoke extraction. The import is idempotent and imports only the delta from a newer `datasets/` snapshot. |
| D31 | **Synthetic `_g<k>` views in the source dataset are imported training‑only and never regenerated** on transfer; new synthetic views remain opt‑in per §7.7 (§7.9 A). |

### 18.2 Still fuzzy — please confirm

| # | Question | Proposed default |
|---|---|---|
| Q1 | *(resolved — two‑letter codes, expand to n letters; §6.1)* | — |
| Q2 | *(resolved — agreement dots **removed** in v1; single inference model, see D21/§7.3)* | — |
| Q3 | *(resolved — re‑split is suggested via a hint, never auto‑applied; §7.8)* | — |
| Q4 | *(resolved — GHCR public image pulls are free/anonymous, so post‑v1 self‑update pulls from GHCR; §2.1, D4)* | — |
| Q5 | Reason‑chip list, split/facets, "Add spot" tool, email delivery, notable‑pattern star — carry over the earlier defaults? | Yes (as previously drafted). |

### 18.3 Reminder — current `contours.db` schema (unchanged)

Per‑dataset DuckDB written by the extraction pipeline; the app joins to it on `salamander_id`
(= `image_id`). Tables:

- **`images`** — `salamander_id` (PK, = raw stem `<code>_<individual>_<instance>`), `width`, `height`,
  `n_spots`, `source_image`, `purple_image`, `body_mask_png` (BLOB, whole‑body mask), `created_at`,
  `is_synthetic`.
- **`spots`** — PK `(salamander_id, spot_id)`; `global_centroid_x/y`, `area_pixels`,
  `local_contour DOUBLE[][]`, `mask_png` (BLOB), `axial_bin` 1–4, `lateral_bin` left/right/overlap,
  `bin` 1–8 (NULL when overlap), `axis_t` 0→1, `axis_side`, `axis_offset` (signed px from midline).
- **`body_axis`** — PK `salamander_id`; `head_x/y`, `tail_tip_x/y`, `length_px`,
  `midline_x/y[]`, `left_x/y[]`, `right_x/y[]`, `source` (`mask`|`mask_corrected`|`none`),
  `judged_ok`, `judge_feedback`.
- **`body_bins`** — PK `bin_id` (`<salamander_id>_b<bin>`); `salamander_id`, `bin`, `quartile`, `side`,
  `t_lo`, `t_hi`, `polygon_x/y[]`.
- **`image_quality`** — one row per photo; RAW markers (`blur_score`, `mean_brightness`,
  `under/overexposed_frac`, `glare_frac`, `pattern_contrast`, `spots_outside_frac`, `solidity`,
  `border_frac`, `curl_deg`, `aspect_ratio`, `body_area_frac`, `line_inside_frac`, …) plus the four
  0–1 composites `blur_quality`, `lighting_quality`, `spot_extraction_quality`,
  `body_extraction_quality`, and `overall_quality` (their geometric mean).

The app writes roster/workflow state to `app.duckdb` (§8) and never mutates `contours.db` except through
the extraction editor (spots re‑binned, quality recomputed) (§10.6).

---

*End of spec. Please annotate inline or in §18 and return for a second pass before M0 begins.*
