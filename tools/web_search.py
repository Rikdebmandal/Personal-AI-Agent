"""Web search tool: ddgs (no API key) first, optional Tavily key as a
fallback for when ddgs gets rate-limited (it does, in practice)."""
from __future__ import annotations

import time

import httpx

from ddgs import DDGS
from ddgs.exceptions import DDGSException

RETRIES = 2
BACKOFF_SECONDS = 2.0

_tavily_api_key = ""


def configure(tavily_api_key: str = "") -> None:
    global _tavily_api_key
    _tavily_api_key = tavily_api_key or ""


def _search_ddgs(query: str, max_results: int) -> list[dict]:
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
            return [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in results
            ]
        except DDGSException as e:
            last_error = e
            if attempt < RETRIES:
                time.sleep(BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"ddgs failed after {RETRIES + 1} attempts: {last_error}")


def _search_tavily(query: str, max_results: int) -> list[dict]:
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": _tavily_api_key, "query": query, "max_results": max_results},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]


def web_search(query: str, max_results: int = 5) -> dict:
    try:
        results = _search_ddgs(query, max_results)
        return {"query": query, "source": "ddgs", "results": results}
    except Exception as ddgs_error:
        if _tavily_api_key:
            try:
                results = _search_tavily(query, max_results)
                return {"query": query, "source": "tavily", "results": results}
            except Exception as tavily_error:
                return {"error": f"ddgs failed ({ddgs_error}); tavily fallback also failed ({tavily_error})"}
        return {"error": f"ddgs failed and no Tavily fallback key is configured: {ddgs_error}"}


TOOL_SPECS = [
    {
        "name": "web_search",
        "description": "Search the web and return a list of {title, url, snippet} results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Number of results to return (default 5)"},
            },
            "required": ["query"],
        },
        "requires_confirmation": False,
    },
]

FUNCTIONS = {"web_search": web_search}
