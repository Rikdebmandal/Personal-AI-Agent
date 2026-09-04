"""In-process human-in-the-loop confirmation gate for risky tool calls.

The agent loop creates a pending confirmation and awaits it; the /api/confirm
HTTP endpoint resolves it once the user clicks approve/deny in the UI.
"""
from __future__ import annotations

import asyncio
import uuid

TIMEOUT_SECONDS = 300
_pending: dict[str, asyncio.Future] = {}


def create_pending() -> tuple[str, asyncio.Future]:
    cid = str(uuid.uuid4())
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _pending[cid] = future
    return cid, future


async def wait_for(cid: str, future: asyncio.Future) -> bool:
    """Awaits approval, denying automatically if nobody responds in time."""
    try:
        return await asyncio.wait_for(future, timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return False
    finally:
        _pending.pop(cid, None)


def resolve(confirmation_id: str, approved: bool) -> bool:
    future = _pending.get(confirmation_id)
    if future is None or future.done():
        return False
    future.set_result(approved)
    return True
