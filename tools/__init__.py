"""Aggregates every tool module's specs and callable functions into one registry."""
from __future__ import annotations

from . import fs, shell, web_fetch, web_search

_MODULES = [fs, shell, web_search, web_fetch]

TOOL_SPECS: list[dict] = [spec for m in _MODULES for spec in m.TOOL_SPECS]
FUNCTIONS: dict = {name: fn for m in _MODULES for name, fn in m.FUNCTIONS.items()}
REQUIRES_CONFIRMATION: dict[str, bool] = {
    spec["name"]: spec["requires_confirmation"] for spec in TOOL_SPECS
}


def as_ollama_tools() -> list[dict]:
    """Convert TOOL_SPECS into Ollama's OpenAI-style function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in TOOL_SPECS
    ]
