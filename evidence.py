from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from utils import compact_whitespace


@dataclass(frozen=True)
class EvidenceMatch:
    status: str
    matched_values: tuple[str, ...]
    missing_values: tuple[str, ...]
    snippet: str | None = None


@dataclass(frozen=True)
class PageEvidence:
    text: str
    json_ld_count: int


def extract_page_evidence(html: str) -> PageEvidence:
    soup = BeautifulSoup(html, "html.parser")
    parts: list[str] = []
    json_ld_count = 0

    if soup.title and soup.title.string:
        parts.append(soup.title.string)

    for meta in soup.find_all("meta"):
        content = meta.get("content")
        if content:
            parts.append(str(content))

    # JSON-LD is often the most machine-readable page evidence, but retailers
    # vary widely in how much product detail they expose there.
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        parsed = _parse_json(script.string or script.get_text())
        if parsed is not None:
            json_ld_count += 1
            parts.append(_flatten_json(parsed))

    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()

    main = soup.find("main") or soup.body or soup
    parts.append(main.get_text(separator=" ", strip=True))

    return PageEvidence(text=compact_whitespace(" ".join(parts)), json_ld_count=json_ld_count)


def assess_value_support(evidence: PageEvidence, value: Any) -> EvidenceMatch:
    values = _value_parts(value)
    if not values:
        return EvidenceMatch(status="not_checked", matched_values=(), missing_values=())

    normalized_evidence = _normalize_for_match(evidence.text)
    matched: list[str] = []
    missing: list[str] = []
    snippet: str | None = None

    for item in values:
        if _is_supported(normalized_evidence, item):
            matched.append(item)
            if snippet is None:
                snippet = _find_snippet(evidence.text, item)
        else:
            missing.append(item)

    if not missing:
        status = "supported"
    elif matched:
        status = "partial"
    else:
        status = "unsupported"

    return EvidenceMatch(
        status=status,
        matched_values=tuple(matched),
        missing_values=tuple(missing),
        snippet=snippet,
    )


def _value_parts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, bool):
        return [str(value).lower()]
    return [str(value).strip()] if str(value).strip() else []


def _is_supported(normalized_evidence: str, value: str) -> bool:
    normalized_value = _normalize_for_match(value)
    if not normalized_value:
        return False

    if normalized_value in normalized_evidence:
        return True

    # Product values often differ only in punctuation or unit spelling:
    # "65 inch", "65-inch", and "65\"" should be treated as the same evidence.
    for alias in _aliases(normalized_value):
        if alias in normalized_evidence:
            return True

    tokens = [token for token in normalized_value.split() if len(token) > 1]
    if len(tokens) >= 2 and all(token in normalized_evidence for token in tokens):
        return True

    return False


def _aliases(value: str) -> set[str]:
    aliases = {value}

    inch_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in)", value)
    if inch_match:
        size = inch_match.group(1)
        aliases.update({f"{size} inch", f"{size} inches", f"{size} in", f"{size}\""})

    hz_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)", value)
    if hz_match:
        rate = hz_match.group(1)
        aliases.update({f"{rate} hz", f"{rate}hz", f"{rate} hertz"})

    if value == "4k":
        aliases.add("4k ultra hd")
    if value == "4k ultra hd":
        aliases.add("4k")

    return aliases


def _find_snippet(text: str, value: str, radius: int = 80) -> str | None:
    match = re.search(re.escape(value), text, flags=re.IGNORECASE)
    if not match:
        return None
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return compact_whitespace(text[start:end])


def _normalize_for_match(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    lowered = ascii_text.lower()
    lowered = lowered.replace('"', " inch ")
    lowered = lowered.replace("″", " inch ")
    lowered = re.sub(r"(\d)\s*-\s*(inch|inches|in)\b", r"\1 \2", lowered)
    lowered = re.sub(r"(\d)\s*hz\b", r"\1 hz", lowered)
    lowered = re.sub(r"[^a-z0-9.%]+", " ", lowered)
    return compact_whitespace(lowered)


def _parse_json(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _flatten_json(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_json(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_json(item) for item in value)
    return str(value)

