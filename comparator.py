from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evidence import PageEvidence, assess_value_support
from schemas import CategorySchema, FieldSpec


@dataclass(frozen=True)
class FieldComparison:
    field: str
    field_type: str
    weight: float
    required: bool
    status: str
    value: Any
    note: str
    credit: float
    evidence_status: str
    evidence_snippet: str | None
    unsupported_values: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonReport:
    category: str
    fields: tuple[FieldComparison, ...]
    hallucinated_fields: tuple[str, ...]

    @property
    def missing_fields(self) -> list[str]:
        return [field.field for field in self.fields if field.status == "missing"]

    @property
    def degraded_fields(self) -> list[str]:
        return [field.field for field in self.fields if field.status == "degraded"]

    @property
    def unsupported_fields(self) -> list[str]:
        return [
            field.field
            for field in self.fields
            if field.evidence_status in {"partial", "unsupported"}
        ]


def compare_extraction(
    category_schema: CategorySchema,
    extraction: dict[str, Any],
    evidence: PageEvidence | None = None,
) -> ComparisonReport:
    schema_fields = category_schema.field_names
    # Anything outside the category schema is treated as hallucinated because
    # the prompt explicitly asks for schema fields only.
    hallucinated = tuple(sorted(set(extraction).difference(schema_fields)))

    comparisons = tuple(
        _compare_field(field, extraction.get(field.field), evidence)
        for field in category_schema.fields
    )

    return ComparisonReport(
        category=category_schema.category,
        fields=comparisons,
        hallucinated_fields=hallucinated,
    )


def _compare_field(
    field: FieldSpec,
    value: Any,
    evidence: PageEvidence | None,
) -> FieldComparison:
    if _is_missing(value):
        return FieldComparison(
            field=field.field,
            field_type=field.field_type,
            weight=field.weight,
            required=field.required,
            status="missing",
            value=None,
            note="No explicit value extracted.",
            credit=0.0,
            evidence_status="not_checked",
            evidence_snippet=None,
            unsupported_values=(),
        )

    if field.field_type == "multi_value":
        if not isinstance(value, list):
            return _degraded(field, value, "Expected an array of explicit values.", evidence)
        # Drop null-like array members so ["VRR", null] still earns credit for
        # the explicit value without pretending the null is useful information.
        cleaned_values = [item for item in value if not _is_missing(item)]
        if not cleaned_values:
            return _degraded(field, value, "Array did not contain usable values.", evidence)
        return _present(field, cleaned_values, evidence=evidence)

    if field.field_type == "boolean" and not isinstance(value, (bool, str)):
        return _degraded(field, value, "Expected boolean, string, or null.", evidence)

    if field.field_type in {"numeric", "categorical", "text", "boolean"}:
        if isinstance(value, (list, dict)):
            return _degraded(field, value, "Expected a scalar value.", evidence)
        return _present(field, value, evidence=evidence)

    return _present(
        field,
        value,
        note=f"Unknown field_type '{field.field_type}' accepted.",
        evidence=evidence,
    )


def _present(
    field: FieldSpec,
    value: Any,
    note: str = "Value extracted.",
    evidence: PageEvidence | None = None,
) -> FieldComparison:
    evidence_status, evidence_snippet, unsupported_values, evidence_factor = _evidence_result(
        evidence,
        value,
    )
    return FieldComparison(
        field=field.field,
        field_type=field.field_type,
        weight=field.weight,
        required=field.required,
        status="present",
        value=value,
        note=note,
        credit=1.0 * evidence_factor,
        evidence_status=evidence_status,
        evidence_snippet=evidence_snippet,
        unsupported_values=unsupported_values,
    )


def _degraded(
    field: FieldSpec,
    value: Any,
    note: str,
    evidence: PageEvidence | None = None,
) -> FieldComparison:
    evidence_status, evidence_snippet, unsupported_values, evidence_factor = _evidence_result(
        evidence,
        value,
    )
    return FieldComparison(
        field=field.field,
        field_type=field.field_type,
        weight=field.weight,
        required=field.required,
        status="degraded",
        value=value,
        note=note,
        credit=0.5 * evidence_factor,
        evidence_status=evidence_status,
        evidence_snippet=evidence_snippet,
        unsupported_values=unsupported_values,
    )


def _evidence_result(
    evidence: PageEvidence | None,
    value: Any,
) -> tuple[str, str | None, tuple[str, ...], float]:
    if evidence is None:
        return "not_checked", None, (), 1.0

    match = assess_value_support(evidence, value)
    if match.status == "supported":
        return match.status, match.snippet, match.missing_values, 1.0
    if match.status == "partial":
        return match.status, match.snippet, match.missing_values, 0.75
    if match.status == "unsupported":
        return match.status, match.snippet, match.missing_values, 0.25
    return match.status, match.snippet, match.missing_values, 1.0


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        # Models sometimes return natural-language placeholders instead of JSON
        # null. Treat those as missing rather than semantically extracted.
        return value.strip().lower() in {"", "null", "none", "not specified", "unknown", "n/a"}
    if isinstance(value, list):
        return len(value) == 0
    return False
