from fastapi.testclient import TestClient

from backend.app.dependencies import get_ollama_backend
from backend.app.main import app
from llm_lab.backends.ollama_backend import OllamaConnectionError


class OfflineOllama:
    def is_available(self) -> bool:
        return False

    def list_models(self) -> list[dict[str, str]]:
        raise OllamaConnectionError("offline")


app.dependency_overrides[get_ollama_backend] = OfflineOllama
client = TestClient(app)


def test_health_reports_ollama_unavailable() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ollama_available"] is False


def test_system_has_safe_runtime_metadata() -> None:
    response = client.get("/api/system")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["ollama_available"], bool)
    assert "username" not in payload
    assert "home" not in payload


def test_custom_benchmark_artifact_is_exposed() -> None:
    response = client.get("/api/benchmarks/custom")
    assert response.status_code == 200
    assert response.json()["records"]


def test_benchmark_index_lists_authoritative_families() -> None:
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    assert {row["name"] for row in response.json()["families"]} >= {"custom", "ollama_baseline"}


def test_missing_artifact_returns_explicit_404(monkeypatch) -> None:
    from backend.app.routes import benchmarks

    def missing(_family: str) -> list[dict[str, object]]:
        from backend.app.services.benchmark_service import ArtifactMissingError

        raise ArtifactMissingError("missing.json")

    monkeypatch.setattr(benchmarks, "read_artifact", missing)
    response = client.get("/api/benchmarks/custom")
    assert response.status_code == 404


def test_models_separate_custom_and_ollama() -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    assert {row["name"] for row in response.json()["custom_presets"]} == {"nano", "micro", "small"}
    assert response.json()["ollama_models"] == []


def teardown_module() -> None:
    app.dependency_overrides.clear()
