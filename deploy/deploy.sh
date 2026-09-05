#!/usr/bin/env bash
# spec §2.1 — one-line manual update for the maintainer.
#   git pull && deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> pulling latest"
git pull --ff-only

echo "==> building + starting"
docker compose -f deploy/docker-compose.yml up -d --build

echo "==> applying migrations"
docker compose -f deploy/docker-compose.yml exec -T app python -m app migrate

echo "==> health check"
for i in $(seq 1 30); do
  if docker compose -f deploy/docker-compose.yml exec -T app \
       python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8756/api/health')" 2>/dev/null; then
    echo "OK"; exit 0
  fi
  sleep 2
done

echo "!! health check failed — rolling back" >&2
docker compose -f deploy/docker-compose.yml rollback 2>/dev/null || \
  git reset --hard HEAD@{1} && docker compose -f deploy/docker-compose.yml up -d --build
exit 1
