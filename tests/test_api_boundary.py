import asyncio
from types import SimpleNamespace

from heart_research.api_boundary import RequestBoundary


def test_overload_rejected_and_capacity_released_after_exception():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def downstream(scope, receive, send):
            entered.set()
            await release.wait()
            raise RuntimeError("inference failed")

        owner = SimpleNamespace(state=SimpleNamespace(token="x" * 32))
        boundary = RequestBoundary(downstream, owner, max_inflight=1)
        scope = {"type": "http", "method": "POST", "headers": [
            (b"x-api-key", b"x" * 32), (b"content-type", b"application/json")]}
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"{}"}

        async def send(message):
            sent.append(message)

        task = asyncio.create_task(boundary(scope, receive, send))
        await entered.wait()
        await boundary(scope, receive, send)
        assert sent[0]["status"] == 503
        release.set()
        try:
            await task
        except RuntimeError:
            pass
        assert boundary.slots.acquire(blocking=False)
        boundary.slots.release()

    asyncio.run(scenario())
