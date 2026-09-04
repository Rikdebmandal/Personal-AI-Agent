"""Tool-calling agent loop against a local Ollama model."""
from __future__ import annotations

import json
import re
from typing import AsyncIterator

import httpx

import tools

from . import confirmation, log

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
MAX_TOOL_ITERATIONS = 12

SYSTEM_PROMPT = (
    "You are a helpful personal AI agent running locally on the user's own computer (Windows). "
    "You have tools to read/write files, run shell commands, and search/fetch the web. "
    "Use tools when they help answer the request. Be direct and concise.\n\n"
    "Critical rules:\n"
    "- Never invent, guess, or hallucinate file contents, directory listings, or command output. "
    "Only report what a tool call actually returned.\n"
    "- If a tool call returns an error, say so plainly. You may retry once with a corrected "
    "argument (e.g. a fixed path) if the error suggests an obvious fix, but do not fabricate "
    "a result instead of retrying or reporting the failure.\n"
    "- When the user gives you a file path, pass it to tools exactly as given, character for "
    "character. Do not abbreviate it with '~' or otherwise rewrite it."
)


class Agent:
    def __init__(self, ollama_host: str, model: str, require_confirmation: bool = True):
        self.ollama_host = ollama_host.rstrip("/")
        self.model = model
        self.require_confirmation = require_confirmation
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def _call_ollama(self, client: httpx.AsyncClient) -> dict:
        resp = await client.post(
            f"{self.ollama_host}/api/chat",
            json={
                "model": self.model,
                "messages": self.history,
                "tools": tools.as_ollama_tools(),
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()

    async def run(self, user_message: str) -> AsyncIterator[dict]:
        self.history.append({"role": "user", "content": user_message})
        log.record("user_message", content=user_message)

        async with httpx.AsyncClient() as client:
            for _ in range(MAX_TOOL_ITERATIONS):
                data = await self._call_ollama(client)
                message = data.get("message", {})
                tool_calls = message.get("tool_calls") or []

                if not tool_calls:
                    raw_content = message.get("content", "")
                    text = THINK_TAG_RE.sub("", raw_content).strip()
                    self.history.append({"role": "assistant", "content": raw_content})
                    log.record("assistant_message", content=text)
                    yield {"type": "text", "content": text}
                    yield {"type": "done"}
                    return

                self.history.append(message)

                for call in tool_calls:
                    name = call["function"]["name"]
                    raw_args = call["function"].get("arguments", {})
                    args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")

                    yield {"type": "tool_call", "tool": name, "arguments": args}

                    if self.require_confirmation and tools.REQUIRES_CONFIRMATION.get(name, True):
                        cid, future = confirmation.create_pending()
                        yield {"type": "confirmation_required", "id": cid, "tool": name, "arguments": args}
                        approved = await confirmation.wait_for(cid, future)
                        if not approved:
                            result = {"error": "user denied this action"}
                            yield {"type": "tool_result", "tool": name, "result": result, "denied": True}
                            log.record("tool_denied", tool=name, arguments=args)
                            self.history.append({"role": "tool", "content": json.dumps(result)})
                            continue

                    fn = tools.FUNCTIONS.get(name)
                    if fn is None:
                        result = {"error": f"unknown tool: {name}"}
                    else:
                        try:
                            result = fn(**args)
                        except Exception as e:
                            result = {"error": str(e)}

                    log.record("tool_call", tool=name, arguments=args, result=result)
                    yield {"type": "tool_result", "tool": name, "result": result}
                    self.history.append({"role": "tool", "content": json.dumps(result, default=str)})

            yield {"type": "text", "content": "Stopped after too many tool-call iterations without a final answer."}
            yield {"type": "done"}
