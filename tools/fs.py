"""Filesystem tool: unrestricted read/write/list/search across the machine.

Write and delete operations are marked `requires_confirmation` so the agent
loop can pause for a human approval click before touching disk.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path

MAX_READ_BYTES = 500_000
MAX_LIST_ENTRIES = 500


def _resolve(path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    if expanded.exists():
        return expanded
    # Small local models sometimes rewrite "C:\Users\name\..." as
    # "~\Users\name\...". expanduser then turns that into a duplicated
    # "C:\Users\name\Users\name\...". If that happened, retry treating the
    # leading "~" as the drive root instead of the home directory.
    if path.startswith("~") and len(path) > 1 and path[1] in ("\\", "/"):
        home = Path.home()
        if home.drive:
            alt = Path(home.drive + os.sep + path[2:]).resolve()
            if alt.exists():
                return alt
    return expanded


def read_file(path: str) -> dict:
    p = _resolve(path)
    if not p.is_file():
        return {"error": f"not a file: {p}"}
    data = p.read_bytes()
    truncated = len(data) > MAX_READ_BYTES
    text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
    return {"path": str(p), "content": text, "truncated": truncated}


def list_dir(path: str) -> dict:
    p = _resolve(path)
    if not p.is_dir():
        return {"error": f"not a directory: {p}"}
    entries = []
    for i, child in enumerate(sorted(p.iterdir())):
        if i >= MAX_LIST_ENTRIES:
            entries.append("... (truncated)")
            break
        kind = "dir" if child.is_dir() else "file"
        entries.append(f"{kind}\t{child.name}")
    return {"path": str(p), "entries": entries}


def search_files(root: str, pattern: str, max_results: int = 100) -> dict:
    p = _resolve(root)
    if not p.is_dir():
        return {"error": f"not a directory: {p}"}
    matches = []
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__", ".venv")]
        for name in filenames:
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                matches.append(str(Path(dirpath) / name))
                if len(matches) >= max_results:
                    return {"matches": matches, "truncated": True}
    return {"matches": matches, "truncated": False}


def write_file(path: str, content: str) -> dict:
    """Requires confirmation — overwrites or creates a file."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    return {"path": str(p), "overwritten": existed}


def delete_path(path: str) -> dict:
    """Requires confirmation — deletes a file or empty directory."""
    p = _resolve(path)
    if p.is_dir():
        p.rmdir()
    elif p.exists():
        p.unlink()
    else:
        return {"error": f"path does not exist: {p}"}
    return {"deleted": str(p)}


TOOL_SPECS = [
    {
        "name": "read_file",
        "description": "Read a text file from the filesystem. Returns file content (truncated if very large).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or ~-relative path to the file"}},
            "required": ["path"],
        },
        "requires_confirmation": False,
    },
    {
        "name": "list_dir",
        "description": "List entries (files and subdirectories) inside a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or ~-relative path to the directory"}},
            "required": ["path"],
        },
        "requires_confirmation": False,
    },
    {
        "name": "search_files",
        "description": "Recursively search for files under a root directory matching a glob pattern (e.g. '*.py').",
        "parameters": {
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "Directory to search under"},
                "pattern": {"type": "string", "description": "Glob pattern to match filenames against, e.g. '*.pdf'"},
            },
            "required": ["root", "pattern"],
        },
        "requires_confirmation": False,
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~-relative path to the file"},
                "content": {"type": "string", "description": "Full text content to write"},
            },
            "required": ["path", "content"],
        },
        "requires_confirmation": True,
    },
    {
        "name": "delete_path",
        "description": "Permanently delete a file or empty directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or ~-relative path to delete"}},
            "required": ["path"],
        },
        "requires_confirmation": True,
    },
]

FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "search_files": search_files,
    "write_file": write_file,
    "delete_path": delete_path,
}
