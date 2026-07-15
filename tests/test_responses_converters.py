# -*- coding: utf-8 -*-

"""Unit tests for the OpenAI Responses compatibility converters."""

import json

import pytest

from kiro.converters_responses import (
    chat_response_to_responses,
    responses_request_to_chat,
)
from kiro.models_responses import ResponseCreateRequest


def test_current_codex_request_shape_converts_to_chat() -> None:
    """Current Codex request fields and client-side tools are accepted."""

    request = ResponseCreateRequest(
        model="gpt-5.6-sol",
        instructions="You are a coding agent.",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect the project"}],
            }
        ],
        tools=[
            {
                "type": "function",
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
                "strict": False,
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a source patch",
            },
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        reasoning={"effort": "xhigh"},
        text={"verbosity": "low"},
        include=["reasoning.encrypted_content"],
        store=False,
        stream=True,
        client_metadata={"originator": "codex_cli_rs"},
    )

    chat, context = responses_request_to_chat(request)

    assert chat.model == "gpt-5.6-sol"
    assert chat.stream is True
    assert chat.reasoning_effort == "xhigh"
    assert [message.role for message in chat.messages] == ["system", "user"]
    assert chat.messages[1].content == "Inspect the project"
    assert chat.tools is not None
    assert chat.tools[0].function.name == "shell"
    assert chat.tools[1].function.name == "apply_patch"
    assert chat.tools[1].function.parameters["required"] == ["input"]
    assert context.tool_kinds == {"shell": "function", "apply_patch": "custom"}


def test_replayed_function_call_and_output_convert_to_chat_messages() -> None:
    """Stateless Responses Item replay preserves the call ID and arguments."""

    request = ResponseCreateRequest(
        model="test-model",
        input=[
            {"type": "message", "role": "user", "content": "List files"},
            {
                "type": "function_call",
                "id": "fc_internal",
                "call_id": "call_123",
                "name": "shell",
                "arguments": '{"cmd":"ls"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": [{"type": "input_text", "text": "README.md"}],
            },
        ],
    )

    chat, _ = responses_request_to_chat(request)

    assert [message.role for message in chat.messages] == ["user", "assistant", "tool"]
    assert chat.messages[1].tool_calls[0]["id"] == "call_123"
    assert chat.messages[1].tool_calls[0]["function"]["arguments"] == '{"cmd":"ls"}'
    assert chat.messages[2].tool_call_id == "call_123"
    assert chat.messages[2].content == "README.md"


def test_custom_tool_replay_uses_json_wrapper_for_kiro() -> None:
    """Raw custom tool input is represented by a one-field JSON schema."""

    request = ResponseCreateRequest(
        model="test-model",
        input=[
            {
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": "*** Begin Patch",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_patch",
                "output": "Done!",
            },
        ],
    )

    chat, _ = responses_request_to_chat(request)

    arguments = json.loads(chat.messages[0].tool_calls[0]["function"]["arguments"])
    assert arguments == {"input": "*** Begin Patch"}
    assert chat.messages[1].content == "Done!"


def test_chat_response_converts_text_reasoning_tools_and_usage() -> None:
    """A non-streaming Chat response becomes typed Responses output Items."""

    request = ResponseCreateRequest(
        model="test-model",
        input="Fix it",
        tools=[
            {"type": "function", "name": "shell", "parameters": {"type": "object"}},
            {"type": "custom", "name": "apply_patch"},
        ],
        reasoning={"effort": "high"},
    )
    _, context = responses_request_to_chat(request)
    chat_response = {
        "model": "test-model",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "I found the issue.",
                    "reasoning_content": "Checked the relevant paths.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "apply_patch",
                                "arguments": '{"input":"*** Begin Patch"}',
                            },
                        },
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }

    response = chat_response_to_responses(chat_response, context)

    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert [item["type"] for item in response["output"]] == [
        "reasoning",
        "message",
        "function_call",
        "custom_tool_call",
    ]
    assert response["output"][2]["call_id"] == "call_1"
    assert response["output"][3]["input"] == "*** Begin Patch"
    assert response["usage"] == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 14,
    }


def test_length_finish_reason_creates_incomplete_response() -> None:
    """Chat length termination maps to Responses incomplete details."""

    request = ResponseCreateRequest(model="test-model", input="Write more")
    _, context = responses_request_to_chat(request)
    response = chat_response_to_responses(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "partial"},
                }
            ],
            "usage": {},
        },
        context,
    )

    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}


def test_previous_response_id_is_rejected_with_replay_guidance() -> None:
    """The adapter does not pretend to provide server-side response storage."""

    request = ResponseCreateRequest(
        model="test-model",
        input="Continue",
        previous_response_id="resp_previous",
    )

    with pytest.raises(ValueError, match="replay prior response output Items"):
        responses_request_to_chat(request)


def test_unsupported_hosted_tool_is_rejected() -> None:
    """Hosted OpenAI tools cannot be silently represented as Kiro functions."""

    request = ResponseCreateRequest(
        model="test-model",
        input="Search",
        tools=[{"type": "web_search_preview"}],
    )

    with pytest.raises(ValueError, match="not supported"):
        responses_request_to_chat(request)


def test_data_url_image_converts_to_chat_image_part() -> None:
    """Responses data URL images use the existing Chat multimodal path."""

    request = ResponseCreateRequest(
        model="test-model",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    chat, _ = responses_request_to_chat(request)

    assert chat.messages[0].content[0] == {"type": "text", "text": "Describe this"}
    assert chat.messages[0].content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"},
    }


def test_namespace_tool_round_trip_preserves_separate_namespace() -> None:
    """Namespace tools use a Kiro-safe wire name and typed response field."""

    request = ResponseCreateRequest(
        model="test-model",
        input=[
            {"type": "message", "role": "user", "content": "Find the customer"},
            {
                "type": "function_call",
                "call_id": "call_lookup",
                "namespace": "crm",
                "name": "lookup",
                "arguments": '{"id":"123"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_lookup",
                "output": "Alice",
            },
        ],
        tools=[
            {
                "type": "namespace",
                "name": "crm",
                "description": "Customer relationship tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        ],
    )

    chat, context = responses_request_to_chat(request)

    assert chat.tools[0].function.name == "crm__lookup"
    assert chat.messages[1].tool_calls[0]["function"]["name"] == "crm__lookup"
    response = chat_response_to_responses(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_next",
                                "function": {
                                    "name": "crm__lookup",
                                    "arguments": '{"id":"456"}',
                                },
                            }
                        ],
                    },
                }
            ]
        },
        context,
    )
    item = response["output"][0]
    assert item["name"] == "lookup"
    assert item["namespace"] == "crm"
