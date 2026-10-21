"""Request body bounds and a per-process admission rate for a small deployment."""

from collections import OrderedDict, deque
import os
import time

from starlette.responses import JSONResponse


class RequestBoundary:
    def __init__(self, app):
        self.app = app
        self.max_bytes = int(os.getenv("ACADEMICFORGE_MAX_REQUEST_BYTES", "1048576"))
        self.per_minute = int(os.getenv("ACADEMICFORGE_REQUESTS_PER_MINUTE", "60"))
        if self.max_bytes < 1 or self.per_minute < 1:
            raise ValueError("Request size and rate limits must be positive")
        self.clients = OrderedDict()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {"/health", "/live"}:
            return await self.app(scope, receive, send)
        key = (scope.get("client") or ("unknown",))[0]
        now = time.monotonic()
        window = self.clients.setdefault(key, deque())
        self.clients.move_to_end(key)
        while window and window[0] < now - 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return await JSONResponse({"detail": "Request limit reached.", "error": {"code": "rate_limited"}},
                                      status_code=429, headers={"Retry-After": "60"})(scope, receive, send)
        window.append(now)
        while len(self.clients) > 4096:
            self.clients.popitem(last=False)
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = self.max_bytes + 1
        if length < 0 or length > self.max_bytes:
            return await self.too_large(scope, receive, send)
        # Bound chunked bodies as well as Content-Length. Buffer at most the configured limit.
        if scope["method"] in {"POST", "PUT", "PATCH"}:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.max_bytes:
                    return await self.too_large(scope, receive, send)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def buffered_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            return await self.app(scope, buffered_receive, send)
        return await self.app(scope, receive, send)

    async def too_large(self, scope, receive, send):
        await JSONResponse({"detail": "Request body is too large.", "error": {"code": "request_too_large"}},
                           status_code=413)(scope, receive, send)
