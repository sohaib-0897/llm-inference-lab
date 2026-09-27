from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.routes import benchmarks, health, models, system

app = FastAPI(title="LLM Inference Lab", version="0.1.0")
app.include_router(health.router)
app.include_router(system.router)
app.include_router(benchmarks.router)
app.include_router(models.router)

DIST = Path(__file__).resolve().parents[2] / "llm-inference-visualizer" / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        candidate = (DIST / path).resolve()
        if candidate.is_relative_to(DIST.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
