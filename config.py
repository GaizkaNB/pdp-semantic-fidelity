from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    products_csv: Path = BASE_DIR / "Data" / "URLs-demo.csv"
    schemas_csv: Path = BASE_DIR / "Data" / "Category Specs.csv"
    model_config_path: Path = BASE_DIR / "Data" / "AI Config.json"
    output_dir: Path = BASE_DIR / "output"
    request_timeout_seconds: int = int(os.getenv("PDP_REQUEST_TIMEOUT", "30"))
    max_content_chars: int = int(os.getenv("PDP_MAX_CONTENT_CHARS", "8000"))
    hallucination_penalty: float = float(os.getenv("HALLUCINATION_PENALTY", "2.5"))
    required_missing_multiplier: float = float(os.getenv("REQUIRED_MISSING_MULTIPLIER", "1.5"))
