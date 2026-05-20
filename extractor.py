from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from requests import Response

from prompts import SYSTEM_PROMPT, build_extraction_prompt
from schemas import CategorySchema
from utils import extract_json_object


@dataclass(frozen=True)
class ExtractionResult:
    data: dict[str, Any]
    raw_response: str
    model: str


class ModelOutputError(ValueError):
    def __init__(self, message: str, raw_responses: list[str]) -> None:
        super().__init__(message)
        self.raw_responses = raw_responses


class OllamaExtractor:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 180,
        temperature: float = 0,
        num_ctx: int = 8192,
        num_predict: int = 800,
        think: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.think = think

    def extract(self, category_schema: CategorySchema, page_text: str) -> ExtractionResult:
        prompt = build_extraction_prompt(category_schema, page_text)
        raw_responses: list[str] = []

        # Prefer the exact schema first. If a local model struggles with schema
        # decoding, retry with Ollama's simpler JSON mode before failing.
        for format_value in (category_schema.as_ollama_json_schema(), "json"):
            content = self._chat(prompt=prompt, format_value=format_value)
            raw_responses.append(content)
            try:
                data = extract_json_object(content)
            except (ValueError, TypeError) as error:
                last_error = error
                continue

            return ExtractionResult(data=data, raw_response=content, model=self.model)

        raise ModelOutputError(
            f"Ollama did not return parseable JSON: {last_error}",
            raw_responses=raw_responses,
        )

    def _chat(self, prompt: str, format_value: Any) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # gemma4:e2b may otherwise place reasoning in `thinking` and leave
            # `message.content` empty, which breaks the JSON-only pipeline.
            "think": self.think,
            "format": format_value,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        _raise_for_ollama_error(response)
        return response.json().get("message", {}).get("content", "")


def _raise_for_ollama_error(response: Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise requests.HTTPError(f"{error}; body={response.text[:1000]}") from error
