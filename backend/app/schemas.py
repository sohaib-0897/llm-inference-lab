from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_available: bool


class SystemInfoResponse(BaseModel):
    python_version: str
    project_version: str
    cpu: str
    ram_total_mb: int | None
    gpu: str | None
    ollama_available: bool
    ollama_version: str | None
    installed_backends: list[str]


class BenchmarkFamily(BaseModel):
    name: str
    description: str
    artifact: str
    available: bool


class BenchmarkIndexResponse(BaseModel):
    families: list[BenchmarkFamily]


class BenchmarkArtifactResponse(BaseModel):
    family: str
    source: str
    records: list[dict[str, Any]]


class ModelInfo(BaseModel):
    name: str
    kind: str
    available: bool


class ModelsResponse(BaseModel):
    custom_presets: list[ModelInfo]
    ollama_models: list[ModelInfo]
