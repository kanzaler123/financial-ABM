# Financial ABM Local Web UI (Stage 1–3 Single-Asset Market)

Local React 19 + TypeScript + Vite + ECharts frontend for the **Stage 1–3 single-asset artificial market** backend (`src/abm/web.py`).

> **Scope note:** This application runs, monitors, compares, and replays local **single-asset** market runs (Stages 1–3) via FastAPI. For the **multi-asset ESG / MAS disclosure policy desk** (Stages F & §§29–32, Panels ①–④), use the static offline exhibit in [`../showcase/`](../showcase/README.md) or visit [https://kanzaler123.github.io/financial-ABM/](https://kanzaler123.github.io/financial-ABM/).

## Prerequisites

- Node.js `>= 22.12`
- Python `>= 3.12` with `pip install -c requirements-lock.txt -e ".[plot,web]"` in the repository root

## Build and Run

```sh
# From repository root:
cd web
npm ci
npm run build
cd ..
python -m abm.web --open
```

The FastAPI server serves `web/dist` at `http://127.0.0.1:8765` (with Chinese / English toggle in the top bar).

## Development & Checks

```sh
cd web
npm run dev      # Vite dev server (proxies API requests when backend runs)
npm run lint     # Oxlint static analysis
npm test         # Node unit tests (api.test.mjs, comparison.test.mjs)
npm run build    # TypeScript check + production bundle into web/dist
```

For full operating instructions, Stage 2 configuration options, and replay workflows, see [`../OPERATIONS.md`](../OPERATIONS.md).
