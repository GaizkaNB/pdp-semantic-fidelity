from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from utils import compact_whitespace


@dataclass(frozen=True)
class FetchedPage:
    url: str
    status_code: int
    html: str
    visible_text: str


class PageFetcher:
    def __init__(self, timeout_seconds: int = 30, max_content_chars: int = 18000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_content_chars = max_content_chars
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0 Safari/537.36"
                ),
                "Accept-Language": "en-GB,en;q=0.9",
            }
        )

    def fetch(self, url: str) -> FetchedPage:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        html = response.text
        visible_text = extract_visible_text(html, max_chars=self.max_content_chars)
        return FetchedPage(
            url=url,
            status_code=response.status_code,
            html=html,
            visible_text=visible_text,
        )


def extract_visible_text(html: str, max_chars: int = 18000) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove content that is usually invisible or noisy for semantic extraction.
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()

    parts: list[str] = []
    if soup.title and soup.title.string:
        parts.append(soup.title.string)

    # Metadata often contains compact product summaries that are useful when the
    # visible PDP body is long, repetitive, or loaded with merchandising text.
    for meta_name in ("description", "og:title", "og:description"):
        selector = (
            f'meta[property="{meta_name}"]'
            if meta_name.startswith("og:")
            else f'meta[name="{meta_name}"]'
        )
        meta = soup.select_one(selector)
        if meta and meta.get("content"):
            parts.append(str(meta["content"]))

    main = soup.find("main") or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    parts.append(text)

    cleaned = compact_whitespace(" ".join(parts))
    # Keep prompts bounded so local Gemma remains responsive and reproducible.
    return cleaned[:max_chars]
