"""Personal AI agent — FastAPI server + chat UI, localhost only."""
from __future__ import annotations

import json

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import tools
from agent import confirmation, log
from agent.core import Agent

with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

tools.web_search.configure(CONFIG.get("web_search", {}).get("tavily_api_key", ""))
log.configure(CONFIG.get("safety", {}).get("action_log_path", "logs/actions.jsonl"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Single shared conversation for this personal, single-user local app.
agent = Agent(
    ollama_host=CONFIG["ollama"]["host"],
    model=CONFIG["ollama"]["model"],
    require_confirmation=CONFIG.get("safety", {}).get("require_confirmation", True),
)


class ChatRequest(BaseModel):
    message: str


class ConfirmRequest(BaseModel):
    id: str
    approved: bool


@app.get("/")
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return StreamingResponse(iter([f.read()]), media_type="text/html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        async for event in agent.run(req.message):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/confirm")
async def confirm(req: ConfirmRequest):
    ok = confirmation.resolve(req.id, req.approved)
    return {"resolved": ok}


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG["server"]["host"], port=CONFIG["server"]["port"])
