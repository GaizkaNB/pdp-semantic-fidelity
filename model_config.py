from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from extractor import OllamaExtractor


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str
    timeout_seconds: int
    temperature: float
    num_ctx: int
    num_predict: int
    think: bool


def load_model_config(path: Path, model_override: str | None = None) -> ModelConfig:
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)

    provider = str(raw.get("provider", "ollama")).strip().lower()
    if provider != "ollama":
        raise ValueError(
            f"Unsupported model provider '{provider}'. Only 'ollama' is implemented."
        )

    # Command-line model overrides let you test another Ollama model without
    # editing the shared Data/AI Config.json file.
    model = model_override or str(raw.get("model", "gemma4:e2b"))

    return ModelConfig(
        provider=provider,
        model=model,
        base_url=str(raw.get("base_url", "http://localhost:11434")),
        timeout_seconds=int(raw.get("timeout_seconds", 180)),
        temperature=float(raw.get("temperature", 0)),
        num_ctx=int(raw.get("num_ctx", 8192)),
        num_predict=int(raw.get("num_predict", 800)),
        think=bool(raw.get("think", False)),
    )


def build_extractor(model_config: ModelConfig) -> OllamaExtractor:
    if model_config.provider == "ollama":
        return OllamaExtractor(
            model=model_config.model,
            base_url=model_config.base_url,
            timeout_seconds=model_config.timeout_seconds,
            temperature=model_config.temperature,
            num_ctx=model_config.num_ctx,
            num_predict=model_config.num_predict,
            think=model_config.think,
        )

    raise ValueError(f"Unsupported model provider '{model_config.provider}'.")

