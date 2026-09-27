from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.schemas import BenchmarkArtifactResponse, BenchmarkFamily, BenchmarkIndexResponse
from backend.app.services.benchmark_service import (
    ARTIFACTS,
    ArtifactMissingError,
    benchmark_index,
    read_artifact,
)

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


@router.get("", response_model=BenchmarkIndexResponse)
def index() -> BenchmarkIndexResponse:
    return BenchmarkIndexResponse(families=[BenchmarkFamily(**family) for family in benchmark_index()])


def _artifact(family: str) -> BenchmarkArtifactResponse:
    try:
        rows: list[dict[str, Any]] = read_artifact(family)
    except ArtifactMissingError as exc:
        raise HTTPException(status_code=404, detail=f"Benchmark artifact is missing: {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return BenchmarkArtifactResponse(family=family, source=ARTIFACTS[family][0], records=rows)


@router.get("/custom", response_model=BenchmarkArtifactResponse)
def custom() -> BenchmarkArtifactResponse:
    return _artifact("custom")


@router.get("/ollama/baseline", response_model=BenchmarkArtifactResponse)
def ollama_baseline() -> BenchmarkArtifactResponse:
    return _artifact("ollama_baseline")


@router.get("/ollama/context", response_model=BenchmarkArtifactResponse)
def ollama_context() -> BenchmarkArtifactResponse:
    return _artifact("ollama_context")


@router.get("/ollama/output", response_model=BenchmarkArtifactResponse)
def ollama_output() -> BenchmarkArtifactResponse:
    return _artifact("ollama_output")


@router.get("/ollama/models", response_model=BenchmarkArtifactResponse)
def ollama_models() -> BenchmarkArtifactResponse:
    return _artifact("ollama_models")


@router.get("/ollama/concurrency", response_model=BenchmarkArtifactResponse)
def ollama_concurrency() -> BenchmarkArtifactResponse:
    return _artifact("ollama_concurrency")
