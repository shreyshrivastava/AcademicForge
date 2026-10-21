"""NDJSON stream with explicit completion/error and no replay after partial output."""

import json
import asyncio
import logging

from starlette.concurrency import run_in_threadpool

from backend.errors import ServiceError, StreamInterrupted
from backend.llm import generation_trace

logger = logging.getLogger(__name__)
_END = object()


def _next(iterator):
    return next(iterator, _END)


async def events(iterator, lease, request_id):
    pending = None
    trace = {}
    generation_trace.set(trace)

    def cleanup(future=None):
        try:
            if future is not None and not future.cancelled():
                future.exception()
            iterator.close()
        finally:
            lease.release()

    try:
        yield json.dumps({"type": "start", "request_id": request_id}) + "\n"
        while True:
            pending = asyncio.create_task(run_in_threadpool(_next, iterator))
            chunk = await asyncio.shield(pending)
            if chunk is _END:
                break
            yield json.dumps({"type": "delta", "text": chunk}) + "\n"
        yield json.dumps({"type": "done", "generation": trace}) + "\n"
    except Exception as exc:
        logger.exception("Generation stream failed request_id=%s", request_id)
        error = exc if isinstance(exc, ServiceError) else StreamInterrupted()
        yield json.dumps({"type": "error", "code": error.code, "message": error.public_message,
                          "request_id": request_id}) + "\n"
    finally:
        if pending is not None and not pending.done():
            pending.add_done_callback(cleanup)
        else:
            await run_in_threadpool(cleanup)
