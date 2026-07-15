# -*- coding: utf-8 -*-

"""Network-isolated route tests for ``POST /v1/responses``."""

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ASGITransport, AsyncClient

from kiro.config import PROXY_API_KEY
from kiro.routes_responses import router


@pytest.fixture
def responses_app() -> FastAPI:
    """Create a minimal application containing only the Responses route."""

    app = FastAPI()
    app.include_router(router)
    return app


def _transport(app: FastAPI) -> ASGITransport:
    """Create an in-process HTTP transport with no network access."""

    return ASGITransport(app=app)


@pytest.mark.asyncio
async def test_responses_route_requires_bearer_api_key(responses_app: FastAPI) -> None:
    """Responses uses the same Authorization header as the existing routes."""

    async with AsyncClient(
        transport=_transport(responses_app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"


@pytest.mark.asyncio
async def test_non_streaming_route_reuses_chat_and_returns_response_object(
    responses_app: FastAPI,
) -> None:
    """The adapter delegates backend work to the existing Chat route."""

    chat_result = JSONResponse(
        content={
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello from Kiro"},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )
    mocked_chat = AsyncMock(return_value=chat_result)

    with patch("kiro.routes_responses.chat_completions", mocked_chat):
        async with AsyncClient(
            transport=_transport(responses_app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={
                    "model": "test-model",
                    "instructions": "Be concise",
                    "input": "Hello",
                    "reasoning": {"effort": "high"},
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["output"][0]["content"][0]["text"] == "Hello from Kiro"
    delegated_request = mocked_chat.await_args.args[1]
    assert [message.role for message in delegated_request.messages] == ["system", "user"]
    assert delegated_request.reasoning_effort == "high"


async def _chat_stream() -> AsyncIterator[str]:
    """Yield a deterministic Chat completion SSE stream."""

    chunks = [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_streaming_route_returns_responses_sse(responses_app: FastAPI) -> None:
    """A delegated Chat stream is exposed as Responses typed events."""

    mocked_chat = AsyncMock(
        return_value=StreamingResponse(_chat_stream(), media_type="text/event-stream")
    )
    with patch("kiro.routes_responses.chat_completions", mocked_chat):
        async with AsyncClient(
            transport=_transport(responses_app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={"model": "test-model", "input": "Hello", "stream": True},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hi"


@pytest.mark.asyncio
async def test_previous_response_id_returns_clear_400(responses_app: FastAPI) -> None:
    """Unsupported server-side state is rejected before backend delegation."""

    mocked_chat = AsyncMock()
    with patch("kiro.routes_responses.chat_completions", mocked_chat):
        async with AsyncClient(
            transport=_transport(responses_app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={
                    "model": "test-model",
                    "input": "Continue",
                    "previous_response_id": "resp_old",
                },
            )

    assert response.status_code == 400
    assert "stateless gateway" in response.json()["detail"]
    mocked_chat.assert_not_awaited()
