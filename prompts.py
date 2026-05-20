from __future__ import annotations

from schemas import CategorySchema


SYSTEM_PROMPT = """You extract product attributes from ecommerce PDP text.
Return JSON only.
Do not guess.
Do not hallucinate.
Use null when the PDP text does not explicitly contain the value.
Preserve original semantic meaning, constraints, units, and specific terms.
Do not replace precise values with vague summaries.
"""


def build_extraction_prompt(category_schema: CategorySchema, page_text: str) -> str:
    return f"""Category: {category_schema.category}

Extract only the fields in this schema:
{category_schema.as_prompt_table()}

Rules:
- Return one JSON object only.
- Include every schema field.
- Do not add fields outside the schema.
- If a field is absent or ambiguous, return null.
- Preserve exact constraints, measurements, units, standards, and product terms.
- For multi_value fields, return an array of explicit values or null.

PDP visible text:
{page_text}
"""

