# LLM Inference Lab — visual companion

A React exhibit for tracing prompt prefill, transformer attention, KV state reuse, token decoding, and local model serving.

## Run locally

```sh
npm install
npm run dev
```

For the integrated local lab, run FastAPI from the repository root with `python -m uvicorn backend.app.main:app --reload --port 8000`. Vite proxies same-origin `/api/*` calls to it. When the API is offline, recorded visualizations continue using the committed generated data; the compact status pill reports `RECORDED DATA`.

The build copies the committed benchmark exports from the sibling `llm-inference-lab` project into `public/data/benchmarks.json`, then type-checks and builds the frontend:

```sh
npm run sync:data
npm run typecheck
npm run lint
npm run build
```

Set `BENCHMARK_SOURCE` to another checkout of `llm-inference-lab` before running `npm run sync:data` or `npm run build` when the sibling directory is unavailable.

`npm run qa:capture` uses the installed Chrome executable to capture the desktop and mobile screenshots in `docs/screenshots/`. The capture script also reports browser errors, horizontal overflow, the KV output-length control, and reduced-motion chapter visibility.

## Structure

- `src/components/` contains the inference core and reusable technical readouts.
- `src/data/` defines the typed benchmark and hardware records.
- `scripts/sync-benchmarks.mjs` reads the authoritative JSON and hardware metadata from the benchmark project.
- `src/data/benchmarks.generated.json` is generated build input; the source artifacts remain authoritative.
- `scripts/capture-qa.mjs` creates visual QA captures and a machine-readable report.

The canonical Vite project is this existing `llm-inference-visualizer/frontend/` directory; it is retained in place.
