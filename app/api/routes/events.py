from __future__ import annotations

import asyncio
import json
import queue

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from . import _deps

router = APIRouter()


@router.get("/events")
async def events(request: Request, once: bool = False) -> StreamingResponse:
    """Server-Sent Events: job progress, notifications, "queue changed" (spec §4.1c).

    ``?once=1`` drains the pending events and closes — used by tests and simple pollers.
    """
    bus = _deps.worker(request).bus
    q = bus.subscribe()

    async def stream():
        try:
            yield "event: hello\ndata: {}\n\n"
            if once:
                while True:
                    try:
                        payload = q.get_nowait()
                    except queue.Empty:
                        break
                    yield f"event: {payload['event']}\ndata: {json.dumps(payload['data'], default=str)}\n\n"
                return
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = q.get_nowait()
                    yield f"event: {payload['event']}\ndata: {json.dumps(payload['data'], default=str)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    await asyncio.sleep(1.0)
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream")
