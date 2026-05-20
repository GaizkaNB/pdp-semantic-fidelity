from __future__ import annotations

import argparse
import json
import csv
import re
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from comparator import compare_extraction
from config import AppConfig
from evidence import extract_page_evidence
from extractor import ModelOutputError
from fetcher import PageFetcher
from model_config import build_extractor, load_model_config
from schemas import load_category_schemas
from scorer import score_report
from utils import ensure_dir, normalize_category


def parse_args() -> argparse.Namespace:
    config = AppConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate PDP semantic reconstructability with category-aware schemas."
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=None,
        help="Full or relative path to a products CSV. Overrides --data-file.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Products CSV filename inside the Data folder, for example URLs-demo.csv.",
    )
    parser.add_argument("--schemas", type=Path, default=config.schemas_csv)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional base filename for both output files, without extension.",
    )
    parser.add_argument("--model-config", type=Path, default=config.model_config_path)
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the selected provider config.",
    )
    parser.add_argument("--max-content-chars", type=int, default=config.max_content_chars)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV inputs and schemas without fetching pages or calling Ollama.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products_csv = resolve_products_csv(args.products, args.data_file)
    config = AppConfig(
        products_csv=products_csv,
        schemas_csv=args.schemas,
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        max_content_chars=args.max_content_chars,
    )
    model_config = load_model_config(config.model_config_path, model_override=args.model)

    schemas = load_category_schemas(config.schemas_csv)
    products = load_products(config.products_csv)
    if args.limit is not None:
        products = products[: args.limit]

    unknown_categories = sorted(
        {product["category"] for product in products}.difference(schemas)
    )
    if unknown_categories:
        raise ValueError(
            f"Products CSV contains categories without schemas: {unknown_categories}"
        )

    print(
        f"Loaded {len(products)} products and {len(schemas)} schemas "
        f"from {config.products_csv.name}."
    )
    print(f"Using {model_config.provider} model {model_config.model}.")

    if args.dry_run:
        print("Dry run complete. Inputs are valid.")
        return

    ensure_dir(config.output_dir)
    fetcher = PageFetcher(
        timeout_seconds=config.request_timeout_seconds,
        max_content_chars=config.max_content_chars,
    )
    extractor = build_extractor(model_config)

    results: list[dict[str, Any]] = []
    for index, product in enumerate(products, start=1):
        print(f"[{index}/{len(products)}] Evaluating {product['url']}")
        try:
            result = evaluate_product(
                product,
                schemas[product["category"]],
                fetcher,
                extractor,
                config,
            )
        except Exception as error:
            # Keep batch runs resilient: one bad URL/model response should be
            # recorded in the output instead of stopping the whole demo.
            result = failed_result(product, error)
            print(f"  Failed: {error}")
        results.append(result)

    json_path, csv_path = write_outputs(results, config.output_dir, args.output_name)
    print(f"Done. Wrote results to {json_path} and {csv_path}")


def resolve_products_csv(products_path: Path | None, data_file: str | None) -> Path:
    if products_path is not None:
        return products_path
    if data_file:
        return AppConfig().products_csv.parent / data_file
    return AppConfig().products_csv


def load_products(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
        columns = set(rows[0]) if rows else set()

    required_columns = {"url", "category"}
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        raise ValueError(f"Products CSV is missing columns: {sorted(missing_columns)}")

    products: list[dict[str, str]] = []
    for row in rows:
        url = str(row["url"]).strip()
        category = normalize_category(row["category"])
        if not url:
            raise ValueError("Products CSV contains an empty URL.")
        if not category:
            raise ValueError(f"Products CSV contains an empty category for URL: {url}")
        products.append({"url": url, "category": category})
    return products


def evaluate_product(
    product: dict[str, str],
    category_schema,
    fetcher: PageFetcher,
    extractor: Extractor,
    config: AppConfig,
) -> dict[str, Any]:
    page = fetcher.fetch(product["url"])
    evidence = extract_page_evidence(page.html)
    extraction = extractor.extract(category_schema, page.visible_text)
    comparison = compare_extraction(category_schema, extraction.data, evidence=evidence)
    score = score_report(
        comparison,
        hallucination_penalty_per_field=config.hallucination_penalty,
        required_missing_multiplier=config.required_missing_multiplier,
    )

    return {
        "url": product["url"],
        "category": product["category"],
        "fetched_status_code": page.status_code,
        "model": extraction.model,
        "extracted": extraction.data,
        "score": asdict(score),
        "missing_semantic_fields": comparison.missing_fields,
        "degraded_fields": comparison.degraded_fields,
        "unsupported_fields": comparison.unsupported_fields,
        "hallucinated_fields": list(comparison.hallucinated_fields),
        "hallucination_count": len(comparison.hallucinated_fields),
        "page_evidence": {
            "json_ld_count": evidence.json_ld_count,
            "text_length": len(evidence.text),
        },
        "comparison": [asdict(field) for field in comparison.fields],
        "raw_model_response": extraction.raw_response,
    }


def failed_result(product: dict[str, str], error: Exception) -> dict[str, Any]:
    raw_model_response = ""
    if isinstance(error, ModelOutputError):
        # Model output failures are the most useful to inspect later, so keep
        # the attempted responses in the detailed JSON report.
        raw_model_response = "\n\n--- RETRY ---\n\n".join(error.raw_responses)

    return {
        "url": product["url"],
        "category": product["category"],
        "fetched_status_code": None,
        "model": None,
        "extracted": {},
        "score": {
            "score": 0.0,
            "max_score": 0.0,
            "earned_score": 0.0,
            "missing_penalty": 0.0,
            "hallucination_penalty": 0.0,
        },
        "missing_semantic_fields": [],
        "degraded_fields": [],
        "unsupported_fields": [],
        "hallucinated_fields": [],
        "hallucination_count": 0,
        "comparison": [],
        "raw_model_response": raw_model_response,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


def write_outputs(
    results: list[dict[str, Any]],
    output_dir: Path,
    output_name: str | None = None,
) -> tuple[Path, Path]:
    if output_name:
        stem = safe_output_stem(output_name)
        json_path = output_dir / f"{stem}.json"
        csv_path = output_dir / f"{stem}.csv"
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        json_path = output_dir / f"semantic_fidelity_results_{timestamp}.json"
        csv_path = output_dir / f"semantic_fidelity_summary_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    summary = [
        {
            "url": result["url"],
            "category": result["category"],
            "score": result["score"]["score"],
            "missing_semantic_fields": ", ".join(result["missing_semantic_fields"]),
            "degraded_fields": ", ".join(result["degraded_fields"]),
            "unsupported_fields": ", ".join(result.get("unsupported_fields", [])),
            "hallucination_count": result["hallucination_count"],
            "hallucinated_fields": ", ".join(result["hallucinated_fields"]),
        }
        for result in results
    ]
    if summary:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

    return json_path, csv_path


def safe_output_stem(output_name: str) -> str:
    stem = Path(output_name).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    if not stem:
        raise ValueError("--output-name must contain at least one filename character.")
    return stem


class Extractor(Protocol):
    def extract(self, category_schema, page_text: str):
        ...


if __name__ == "__main__":
    main()
