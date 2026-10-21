"""Bounded admission before moving blocking work off the API event loop."""

import os
import asyncio
from threading import BoundedSemaphore, Lock

from starlette.concurrency import run_in_threadpool

from backend.errors import CapacityError


def positive_int(name, default):
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class Capacity:
    def __init__(self, limit):
        self._slots = BoundedSemaphore(limit)

    def reserve(self):
        if not self._slots.acquire(blocking=False):
            raise CapacityError()
        return Lease(self._slots)


class Lease:
    def __init__(self, slots):
        self._slots = slots
        self._lock = Lock()
        self._released = False

    def release(self):
        with self._lock:
            if not self._released:
                self._slots.release()
                self._released = True


SEARCH_CAPACITY = Capacity(positive_int("ACADEMICFORGE_SEARCH_CONCURRENCY", 2))
GENERATION_CAPACITY = Capacity(positive_int("ACADEMICFORGE_GENERATION_CONCURRENCY", 1))


async def execute(capacity, operation, *args, **kwargs):
    lease = capacity.reserve()
    task = asyncio.create_task(run_in_threadpool(operation, *args, **kwargs))

    def completed(future):
        lease.release()
        if not future.cancelled():
            future.exception()  # Consume errors when the client already disconnected.

    task.add_done_callback(completed)
    # Disconnects stop waiting; they do not free GPU capacity while inference still runs.
    return await asyncio.shield(task)
