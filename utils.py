from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def normalize_category(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_")


def normalize_field_name(value: Any) -> str:
    return str(value).strip().lower()


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort parser for model responses that wrap JSON in extra text."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from the model.")
    return parsed

