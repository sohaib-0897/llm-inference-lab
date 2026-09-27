from fastapi import APIRouter, Depends

from backend.app.dependencies import get_ollama_backend
from backend.app.schemas import ModelInfo, ModelsResponse
from llm_lab.backends.ollama_backend import OllamaBackend, OllamaError

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def models(ollama: OllamaBackend = Depends(get_ollama_backend)) -> ModelsResponse:  # noqa: B008
    installed: list[ModelInfo] = []
    try:
        installed = [ModelInfo(name=str(item["name"]), kind="ollama", available=True)
                     for item in ollama.list_models() if isinstance(item.get("name"), str)]
    except OllamaError:
        pass
    return ModelsResponse(
        custom_presets=[ModelInfo(name=name, kind="custom_pytorch", available=True)
                        for name in ("nano", "micro", "small")],
        ollama_models=installed,
    )
