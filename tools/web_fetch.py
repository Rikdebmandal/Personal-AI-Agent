"""Fetch a URL and extract its main readable text content."""
from __future__ import annotations

import httpx
import trafilatura

MAX_CHARS = 15_000
TIMEOUT_SECONDS = 20


def fetch_url(url: str) -> dict:
    try:
        resp = httpx.get(
            url,
            timeout=TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (personal-ai-agent)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"failed to fetch {url}: {e}"}

    text = trafilatura.extract(resp.text, url=url) or ""
    truncated = len(text) > MAX_CHARS
    return {"url": url, "content": text[:MAX_CHARS], "truncated": truncated}


TOOL_SPECS = [
    {
        "name": "fetch_url",
        "description": "Fetch a web page by URL and return its main text content (HTML stripped).",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch"}},
            "required": ["url"],
        },
        "requires_confirmation": False,
    },
]

FUNCTIONS = {"fetch_url": fetch_url}
