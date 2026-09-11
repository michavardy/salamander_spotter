#!/usr/bin/env bash
# Runs on the pod after the bundle is uploaded, before the sweep command.
# Links the big contours.db from the mounted network volume into the path the
# pipeline expects (data.REPO_ROOT == /workspace/job/bundle on the pod).
set -euo pipefail

BUNDLE=/workspace/job/bundle
DATASET=all_sasa_norm_2026_23_07
VOL_DB=${RUNPOD_VOLUME_MOUNT_PATH:-/runpod-volume}/contours.db
DEST_DB="$BUNDLE/datasets/$DATASET/db/contours.db"

if [[ ! -f "$VOL_DB" ]]; then
  echo "[startup] FATAL: $VOL_DB not found — is the network volume attached?" >&2
  ls -la "$(dirname "$VOL_DB")" >&2 || true
  exit 1
fi

mkdir -p "$(dirname "$DEST_DB")"
ln -sf "$VOL_DB" "$DEST_DB"
echo "[startup] linked $(du -h "$VOL_DB" | cut -f1) contours.db -> $DEST_DB"

# sanity: the two small data files should have been uploaded next to it
for f in "datasets/$DATASET/corrections.json" "images/all_sasa_norm/label_map.csv"; do
  [[ -f "$BUNDLE/$f" ]] && echo "[startup] ok  $f" || echo "[startup] WARN missing $BUNDLE/$f" >&2
done

python -c "import duckdb,scipy; print('[startup] duckdb', duckdb.__version__, 'scipy', scipy.__version__)"
