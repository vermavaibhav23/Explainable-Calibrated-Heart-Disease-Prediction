"""Bound requests before parsing; never echo patient payloads in errors."""
import json
import secrets
import threading
import uuid


class RequestBoundary:
    def __init__(self, app, owner, max_bytes=16384, max_inflight=4):
        self.app, self.owner, self.max_bytes = app, owner, max_bytes
        self.slots = threading.BoundedSemaphore(max_inflight)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex

        async def secure_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [
                    (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                    (b"x-request-id", request_id.encode())]
            await send(message)

        async def reject(status, detail):
            body = json.dumps({"detail": detail, "request_id": request_id}).encode()
            await secure_send({"type": "http.response.start", "status": status,
                               "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})

        if scope["method"] != "POST":
            return await self.app(scope, receive, secure_send)
        headers = dict(scope.get("headers", []))
        key = headers.get(b"x-api-key", b"")
        expected = getattr(self.owner.state, "token", "").encode()
        if not key or not expected or not secrets.compare_digest(key, expected):
            return await reject(401, "Invalid API key")
        if headers.get(b"content-type", b"").split(b";")[0].lower() != b"application/json":
            return await reject(415, "Content-Type must be application/json")
        if not self.slots.acquire(blocking=False):
            return await reject(503, "Inference capacity reached; retry later")
        try:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.max_bytes:
                    return await reject(413, "Request body exceeds 16 KiB")
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, secure_send)
        finally:
            self.slots.release()
