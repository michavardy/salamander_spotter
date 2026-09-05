# Salamander Spotter — multi-stage image (spec §5.2).
# Stage 1 builds the static web bundle; stage 2 is a slim Python runtime that
# serves it from the backend. /data is a mounted volume, never baked in.

# ---- stage 1: web build ----------------------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/pnpm-lock.yaml* ./
RUN corepack enable && (pnpm install --frozen-lockfile || pnpm install)
COPY web/ ./
RUN pnpm build            # -> /web/out  (next.config.mjs: output: 'export')

# ---- stage 2: python runtime --------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SPOTTER_DATA_DIR=/data SPOTTER_HOST=0.0.0.0 SPOTTER_PORT=8756

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY --from=web /web/out ./app/resources/web

VOLUME ["/data"]
EXPOSE 8756
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8756/api/health').status==200 else 1)"

ENTRYPOINT ["python", "-m", "app"]
CMD ["serve"]
