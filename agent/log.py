"""Append-only JSONL log of every message and tool call the agent makes."""
from __future__ import annotations

import json
import time
from pathlib import Path

_log_path: Path | None = None


def configure(path: str) -> None:
    global _log_path
    _log_path = Path(path)
    _log_path.parent.mkdir(parents=True, exist_ok=True)


def record(event: str, **fields) -> None:
    if _log_path is None:
        return
    entry = {"ts": time.time(), "event": event, **fields}
    with _log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
