from fastapi import APIRouter, Depends

from backend.app.dependencies import get_ollama_backend
from backend.app.schemas import HealthResponse
from llm_lab import __version__
from llm_lab.backends.ollama_backend import OllamaBackend

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(ollama: OllamaBackend = Depends(get_ollama_backend)) -> HealthResponse:  # noqa: B008
    return HealthResponse(status="ok", version=__version__, ollama_available=ollama.is_available())
