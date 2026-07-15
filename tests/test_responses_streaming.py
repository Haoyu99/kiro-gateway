# -*- coding: utf-8 -*-

"""Tests for Chat SSE to OpenAI Responses SSE translation."""

import json
from typing import Any, AsyncIterator, Dict, List

import pytest

from kiro.converters_responses import responses_request_to_chat
from kiro.models_responses import ResponseCreateRequest
from kiro.streaming_responses import stream_chat_to_responses


async def _source(payloads: List[Any]) -> AsyncIterator[str]:
    """Create a Chat Completions SSE source."""

    for payload in payloads:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        yield f"data: {data}\n\n"


async def _collect_events(
    payloads: List[Any],
    request: ResponseCreateRequest,
) -> List[Dict[str, Any]]:
    """Run the stream adapter and decode its emitted event objects."""

    _, context = responses_request_to_chat(request)
    events = []
    async for chunk in stream_chat_to_responses(_source(payloads), context):
        assert chunk.startswith("data: ")
        events.append(json.loads(chunk[6:].strip()))
    return events


@pytest.mark.asyncio
async def test_text_stream_has_responses_lifecycle_and_usage() -> None:
    """Text deltas are wrapped in the required typed Item lifecycle."""

    request = ResponseCreateRequest(model="test-model", input="Hello", stream=True)
    events = await _collect_events(
        [
            {
                "choices": [{"delta": {"role": "assistant", "content": "Hel"}}],
            },
            {
                "choices": [{"delta": {"content": "lo"}}],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
            "[DONE]",
        ],
        request,
    )

    types = [event["type"] for event in events]
    assert types[:2] == ["response.created", "response.in_progress"]
    assert types.count("response.output_text.delta") == 2
    assert types[-1] == "response.completed"
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    completed = events[-1]["response"]
    assert completed["output"][0]["content"][0]["text"] == "Hello"
    assert completed["usage"]["input_tokens"] == 3
    assert completed["usage"]["output_tokens"] == 2
    assert "[DONE]" not in json.dumps(events)


@pytest.mark.asyncio
async def test_function_tool_stream_uses_function_argument_events() -> None:
    """A Chat function call becomes a Responses function_call Item."""

    request = ResponseCreateRequest(
        model="test-model",
        input="Run pwd",
        stream=True,
        tools=[
            {
                "type": "function",
                "name": "shell",
                "parameters": {"type": "object"},
            }
        ],
    )
    events = await _collect_events(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_shell",
                                    "type": "function",
                                    "function": {
                                        "name": "shell",
                                        "arguments": '{"cmd":"pwd"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {}},
            "[DONE]",
        ],
        request,
    )

    types = [event["type"] for event in events]
    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" in types
    completed = events[-1]["response"]
    assert completed["output"][0] == {
        "id": completed["output"][0]["id"],
        "type": "function_call",
        "status": "completed",
        "call_id": "call_shell",
        "name": "shell",
        "arguments": '{"cmd":"pwd"}',
    }


@pytest.mark.asyncio
async def test_custom_tool_stream_exposes_raw_input() -> None:
    """The custom-tool JSON wrapper is removed from streamed Codex Items."""

    request = ResponseCreateRequest(
        model="test-model",
        input="Patch it",
        stream=True,
        tools=[{"type": "custom", "name": "apply_patch"}],
    )
    events = await _collect_events(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_patch",
                                    "function": {
                                        "name": "apply_patch",
                                        "arguments": '{"input":"*** ',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'Begin Patch"}'},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        ],
        request,
    )

    types = [event["type"] for event in events]
    assert "response.custom_tool_call_input.delta" in types
    assert "response.custom_tool_call_input.done" in types
    completed_item = events[-1]["response"]["output"][0]
    assert completed_item["type"] == "custom_tool_call"
    assert completed_item["call_id"] == "call_patch"
    assert completed_item["input"] == "*** Begin Patch"
    assert "_arguments" not in completed_item


@pytest.mark.asyncio
async def test_reasoning_stream_creates_summary_events() -> None:
    """Kiro reasoning deltas are exposed as a Responses reasoning summary."""

    request = ResponseCreateRequest(model="test-model", input="Think", stream=True)
    events = await _collect_events(
        [
            {"choices": [{"delta": {"reasoning_content": "Inspecting "}}]},
            {"choices": [{"delta": {"reasoning_content": "the code"}}]},
            {"choices": [{"delta": {"content": "Done"}, "finish_reason": "stop"}]},
            "[DONE]",
        ],
        request,
    )

    types = [event["type"] for event in events]
    assert types.count("response.reasoning_summary_text.delta") == 2
    assert "response.reasoning_summary_text.done" in types
    completed = events[-1]["response"]
    assert completed["output"][0]["summary"][0]["text"] == "Inspecting the code"
    assert completed["output"][1]["content"][0]["text"] == "Done"


@pytest.mark.asyncio
async def test_length_stream_ends_with_response_incomplete() -> None:
    """Streaming length termination uses response.incomplete."""

    request = ResponseCreateRequest(model="test-model", input="Long answer", stream=True)
    events = await _collect_events(
        [
            {"choices": [{"delta": {"content": "partial"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
            "[DONE]",
        ],
        request,
    )

    assert events[-1]["type"] == "response.incomplete"
    assert events[-1]["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
