from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import coerce_bool, normalize_category, normalize_field_name


@dataclass(frozen=True)
class FieldSpec:
    category: str
    field: str
    weight: float
    required: bool
    field_type: str


@dataclass(frozen=True)
class CategorySchema:
    category: str
    fields: tuple[FieldSpec, ...]

    @property
    def field_names(self) -> set[str]:
        return {field.field for field in self.fields}

    def as_prompt_table(self) -> str:
        rows = ["field | weight | required | field_type"]
        rows.extend(
            f"{field.field} | {field.weight:g} | {field.required} | {field.field_type}"
            for field in self.fields
        )
        return "\n".join(rows)

    def as_ollama_json_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required = []

        # Ollama's structured output needs a concrete JSON schema. We derive it
        # from the CSV so category fields stay data-driven instead of hardcoded.
        for field in self.fields:
            required.append(field.field)
            properties[field.field] = _json_schema_for_field(field)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


def load_category_schemas(csv_path: Path) -> dict[str, CategorySchema]:
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
        columns = set(rows[0]) if rows else set()

    required_columns = {"category", "field", "weight", "required", "field_type"}
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        raise ValueError(f"Schema CSV is missing columns: {sorted(missing_columns)}")

    fields_by_category: dict[str, list[FieldSpec]] = {}
    for row in rows:
        category = normalize_category(row["category"])
        field = FieldSpec(
            category=category,
            field=normalize_field_name(row["field"]),
            weight=float(row["weight"]),
            required=coerce_bool(row["required"]),
            field_type=str(row["field_type"]).strip().lower(),
        )
        fields_by_category.setdefault(category, []).append(field)

    return {
        category: CategorySchema(category=category, fields=tuple(fields))
        for category, fields in fields_by_category.items()
    }


def _json_schema_for_field(field: FieldSpec) -> dict[str, Any]:
    description = (
        f"{field.field_type} value for {field.field}. Use null when the PDP does not "
        "explicitly state the value. Preserve units and exact wording."
    )

    # Multi-value fields are arrays because preserving distinct values matters:
    # ["VRR", "ALLM"] is more useful than a single vague joined string.
    if field.field_type == "multi_value":
        return {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": description,
        }

    if field.field_type == "boolean":
        return {
            "type": ["boolean", "string", "null"],
            "description": description,
        }

    # Most product attributes are accepted as string or number because PDPs may
    # express values as "65 inch", "65", "120Hz", or similar mixed forms.
    return {
        "type": ["string", "number", "null"],
        "description": description,
    }
