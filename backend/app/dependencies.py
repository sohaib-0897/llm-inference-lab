from llm_lab.backends.ollama_backend import OllamaBackend


def get_ollama_backend() -> OllamaBackend:
    return OllamaBackend(timeout=3.0)
