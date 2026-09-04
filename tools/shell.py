"""Shell command execution tool. Always requires confirmation — this is the
single most powerful (and dangerous) tool available to the agent."""
from __future__ import annotations

import subprocess

TIMEOUT_SECONDS = 60
MAX_OUTPUT_CHARS = 20_000


def run_command(command: str, cwd: str | None = None) -> dict:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"command timed out after {TIMEOUT_SECONDS}s"}

    stdout = result.stdout[:MAX_OUTPUT_CHARS]
    stderr = result.stderr[:MAX_OUTPUT_CHARS]
    return {"exit_code": result.returncode, "stdout": stdout, "stderr": stderr}


TOOL_SPECS = [
    {
        "name": "run_command",
        "description": "Execute a shell command on the user's machine (PowerShell on Windows) and return stdout/stderr/exit code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The full shell command to execute"},
                "cwd": {"type": "string", "description": "Optional working directory to run the command in"},
            },
            "required": ["command"],
        },
        "requires_confirmation": True,
    },
]

FUNCTIONS = {"run_command": run_command}
