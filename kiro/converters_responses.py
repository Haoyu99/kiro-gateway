# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Conversions between OpenAI Responses and Chat Completions payloads."""

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from kiro.models_openai import ChatCompletionRequest, ChatMessage, Tool, ToolFunction
from kiro.models_responses import ResponseCreateRequest


@dataclass
class ResponsesConversionContext:
    """State needed to translate a Chat response back to Responses Items."""

    request: ResponseCreateRequest
    response_id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    created_at: int = field(default_factory=lambda: int(time.time()))
    tool_kinds: Dict[str, str] = field(default_factory=dict)
    tool_response_names: Dict[str, str] = field(default_factory=dict)
    tool_namespaces: Dict[str, Optional[str]] = field(default_factory=dict)
    tool_wire_names: Dict[Tuple[Optional[str], str], str] = field(default_factory=dict)


def _as_dict(value: Any) -> Dict[str, Any]:
    """Return a plain dictionary for a mapping or Pydantic model."""

    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    raise ValueError(f"Expected an object, got {type(value).__name__}")


def _text_from_parts(content: Any, field_name: str) -> str:
    """Convert textual Responses content into a single string.

    Args:
        content: A string, content object, or list of content objects.
        field_name: Field name used in validation errors.

    Returns:
        Concatenated text.

    Raises:
        ValueError: If the content contains a non-text part.
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        part_type = content.get("type")
        if part_type in {"input_text", "output_text", "text", "summary_text"}:
            return str(content.get("text", ""))
        raise ValueError(f"{field_name} contains unsupported content type '{part_type}'")
    if isinstance(content, list):
        return "".join(_text_from_parts(part, field_name) for part in content)
    return json.dumps(content, ensure_ascii=False)


def _message_content_to_chat(content: Any) -> Union[str, List[Dict[str, Any]]]:
    """Convert Responses message content to Chat Completions content."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content)

    chat_parts: List[Dict[str, Any]] = []
    has_image = False
    for raw_part in content:
        if isinstance(raw_part, str):
            chat_parts.append({"type": "text", "text": raw_part})
            continue

        part = _as_dict(raw_part)
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            chat_parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif part_type in {"input_image", "image_url"}:
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                detail = image_url.get("detail")
            else:
                url = image_url
                detail = part.get("detail")
            if not url:
                if part.get("file_id"):
                    raise ValueError(
                        "Responses input_image.file_id is not supported; send a data URL in image_url"
                    )
                raise ValueError("Responses input_image requires image_url")
            image_payload: Dict[str, Any] = {"url": url}
            if detail:
                image_payload["detail"] = detail
            chat_parts.append({"type": "image_url", "image_url": image_payload})
            has_image = True
        elif part_type == "input_file":
            raise ValueError("Responses input_file is not supported by the Kiro backend")
        else:
            raise ValueError(f"Unsupported Responses message content type '{part_type}'")

    if not has_image:
        return "".join(str(part.get("text", "")) for part in chat_parts)
    return chat_parts


def _append_tool_call(
    messages: List[ChatMessage],
    call_id: str,
    name: str,
    arguments: str,
) -> None:
    """Append a tool call, merging adjacent assistant calls into one message."""

    tool_call = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    if messages and messages[-1].role == "assistant":
        calls = list(messages[-1].tool_calls or [])
        calls.append(tool_call)
        messages[-1] = messages[-1].model_copy(update={"tool_calls": calls})
    else:
        messages.append(ChatMessage(role="assistant", content=None, tool_calls=[tool_call]))


def _namespace_wire_name(namespace: str, name: str) -> str:
    """Create a Kiro-safe function name for a namespaced Responses tool."""

    original = f"{namespace}__{name}"
    candidate = re.sub(r"[^a-zA-Z0-9_-]", "_", original)
    if len(candidate) <= 64:
        return candidate
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[:55]}_{digest}"


def _convert_input_items(
    request: ResponseCreateRequest,
    context: ResponsesConversionContext,
) -> List[ChatMessage]:
    """Convert Responses input Items into Chat Completions messages."""

    messages: List[ChatMessage] = []
    if request.instructions is not None:
        instructions = _text_from_parts(request.instructions, "instructions")
        messages.append(ChatMessage(role="system", content=instructions))

    if isinstance(request.input, str):
        messages.append(ChatMessage(role="user", content=request.input))
        return messages

    for raw_item in request.input:
        if isinstance(raw_item, str):
            messages.append(ChatMessage(role="user", content=raw_item))
            continue

        item = _as_dict(raw_item)
        item_type = item.get("type")

        if item_type == "reasoning":
            logger.debug("Ignoring replayed Responses reasoning item for stateless Kiro conversion")
            continue

        if item_type in {"function_call", "custom_tool_call"}:
            call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}")
            name = item.get("name")
            if not name:
                raise ValueError(f"Responses {item_type} item requires name")
            namespace = item.get("namespace")
            wire_name = context.tool_wire_names.get(
                (str(namespace) if namespace else None, str(name)),
                _namespace_wire_name(str(namespace), str(name)) if namespace else str(name),
            )
            if item_type == "custom_tool_call":
                arguments = json.dumps(
                    {"input": str(item.get("input", ""))},
                    ensure_ascii=False,
                )
            else:
                arguments = item.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
            _append_tool_call(messages, call_id, wire_name, arguments)
            continue

        if item_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = item.get("call_id")
            if not call_id:
                raise ValueError(f"Responses {item_type} item requires call_id")
            output = _text_from_parts(item.get("output", ""), f"{item_type}.output")
            messages.append(
                ChatMessage(role="tool", tool_call_id=str(call_id), content=output)
            )
            continue

        if item_type == "item_reference":
            raise ValueError(
                "Responses item_reference is not supported; replay the referenced Items in input"
            )

        if item_type in {"input_text", "output_text"}:
            messages.append(ChatMessage(role="user", content=str(item.get("text", ""))))
            continue

        if item_type == "message" or "role" in item:
            role = str(item.get("role", "user"))
            if role in {"developer", "system"}:
                role = "system"
            messages.append(
                ChatMessage(role=role, content=_message_content_to_chat(item.get("content")))
            )
            continue

        raise ValueError(f"Unsupported Responses input item type '{item_type}'")

    if not messages:
        messages.append(ChatMessage(role="user", content=""))
    return messages


def _convert_tools(
    tools: Optional[List[Dict[str, Any]]],
    context: ResponsesConversionContext,
) -> Optional[List[Tool]]:
    """Convert Responses function, custom, and namespace tools to Chat tools."""

    if not tools:
        return None

    converted: List[Tool] = []

    def add_tool(tool: Dict[str, Any], namespace: Optional[str] = None) -> None:
        tool_type = tool.get("type")
        if tool_type not in {"function", "custom"}:
            raise ValueError(
                f"Responses tool type '{tool_type}' is not supported by the Kiro backend; "
                "use function or custom tools"
            )
        name = tool.get("name")
        if not name:
            raise ValueError(f"Responses {tool_type or 'unknown'} tool requires name")
        wire_name = _namespace_wire_name(namespace, str(name)) if namespace else str(name)
        if wire_name in context.tool_kinds:
            raise ValueError(f"Duplicate Responses tool name '{wire_name}'")

        description = tool.get("description")
        if namespace and not description:
            description = f"Tool {name} in the {namespace} namespace"

        if tool_type == "function":
            parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        elif tool_type == "custom":
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Raw input for this custom tool",
                    }
                },
                "required": ["input"],
                "additionalProperties": False,
            }
        context.tool_kinds[wire_name] = str(tool_type)
        context.tool_response_names[wire_name] = str(name)
        context.tool_namespaces[wire_name] = namespace
        context.tool_wire_names[(namespace, str(name))] = wire_name
        converted.append(
            Tool(
                type="function",
                function=ToolFunction(
                    name=wire_name,
                    description=description,
                    parameters=parameters,
                ),
            )
        )

    for raw_tool in tools:
        tool = _as_dict(raw_tool)
        if tool.get("type") == "namespace":
            namespace = tool.get("name")
            if not namespace:
                raise ValueError("Responses namespace tool requires name")
            nested_tools = tool.get("tools") or []
            if not nested_tools:
                raise ValueError(f"Responses namespace tool '{namespace}' has no tools")
            for nested in nested_tools:
                add_tool(_as_dict(nested), namespace=str(namespace))
        else:
            add_tool(tool)
    return converted


def _convert_tool_choice(
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    context: ResponsesConversionContext,
) -> Optional[Union[str, Dict[str, Any]]]:
    """Convert the Responses flat named-tool choice to Chat format."""

    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    choice = _as_dict(tool_choice)
    choice_type = choice.get("type")
    if choice_type in {"function", "custom"}:
        name = choice.get("name")
        if not name:
            raise ValueError(f"Responses {choice_type} tool_choice requires name")
        namespace = choice.get("namespace")
        wire_name = context.tool_wire_names.get(
            (str(namespace) if namespace else None, str(name)),
            _namespace_wire_name(str(namespace), str(name)) if namespace else str(name),
        )
        return {"type": "function", "function": {"name": wire_name}}
    if choice_type == "allowed_tools":
        return "auto"
    return choice


def responses_request_to_chat(
    request: ResponseCreateRequest,
) -> Tuple[ChatCompletionRequest, ResponsesConversionContext]:
    """Translate a Responses create request to the existing Chat route model."""

    if request.previous_response_id:
        raise ValueError(
            "previous_response_id is not supported by this stateless gateway; "
            "replay prior response output Items in input instead"
        )

    context = ResponsesConversionContext(request=request)
    converted_tools = _convert_tools(request.tools, context)
    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=_convert_input_items(request, context),
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        max_completion_tokens=request.max_output_tokens,
        reasoning_effort=(request.reasoning or {}).get("effort"),
        tools=converted_tools,
        tool_choice=_convert_tool_choice(request.tool_choice, context),
        parallel_tool_calls=request.parallel_tool_calls,
        user=request.user,
    )
    return chat_request, context


def _new_item_id(prefix: str) -> str:
    """Create an opaque Responses output Item ID."""

    return f"{prefix}_{uuid.uuid4().hex}"


def response_usage(chat_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Translate Chat token usage to Responses token usage."""

    if chat_usage is None:
        return None
    input_tokens = int(chat_usage.get("prompt_tokens", chat_usage.get("input_tokens", 0)) or 0)
    output_tokens = int(
        chat_usage.get("completion_tokens", chat_usage.get("output_tokens", 0)) or 0
    )
    total_tokens = int(chat_usage.get("total_tokens", input_tokens + output_tokens) or 0)
    prompt_details = chat_usage.get("prompt_tokens_details") or {}
    completion_details = chat_usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": int(prompt_details.get("cached_tokens", 0) or 0)
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0)
        },
        "total_tokens": total_tokens,
    }


def custom_tool_input(arguments: str) -> str:
    """Extract raw custom-tool input from the compatibility JSON wrapper."""

    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments
    if isinstance(parsed, dict) and "input" in parsed:
        value = parsed["input"]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return arguments


def tool_call_to_response_item(
    tool_call: Dict[str, Any],
    context: ResponsesConversionContext,
) -> Dict[str, Any]:
    """Convert one Chat tool call into a Responses output Item."""

    function = tool_call.get("function") or {}
    wire_name = str(function.get("name", ""))
    name = context.tool_response_names.get(wire_name, wire_name)
    namespace = context.tool_namespaces.get(wire_name)
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex}")
    if context.tool_kinds.get(wire_name) == "custom":
        item = {
            "id": _new_item_id("ctc"),
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": call_id,
            "name": name,
            "input": custom_tool_input(arguments),
        }
        if namespace:
            item["namespace"] = namespace
        return item
    item = {
        "id": _new_item_id("fc"),
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    if namespace:
        item["namespace"] = namespace
    return item


def chat_message_to_response_items(
    message: Dict[str, Any],
    context: ResponsesConversionContext,
) -> List[Dict[str, Any]]:
    """Convert a Chat assistant message into typed Responses output Items."""

    output: List[Dict[str, Any]] = []
    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        output.append(
            {
                "id": _new_item_id("rs"),
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": str(reasoning_content)}],
            }
        )

    content = message.get("content")
    if content is not None and (content != "" or not message.get("tool_calls")):
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        output.append(
            {
                "id": _new_item_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": content,
                    }
                ],
            }
        )

    for tool_call in message.get("tool_calls") or []:
        output.append(tool_call_to_response_item(tool_call, context))
    return output


def response_status(finish_reason: Optional[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Map a Chat finish reason to Responses status and incomplete details."""

    if finish_reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if finish_reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    return "completed", None


def build_response_object(
    context: ResponsesConversionContext,
    output: List[Dict[str, Any]],
    status: str,
    usage: Optional[Dict[str, Any]] = None,
    incomplete_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete OpenAI Responses response object."""

    request = context.request
    reasoning = dict(request.reasoning or {})
    reasoning.setdefault("summary", None)
    text_config = dict(request.text or {})
    text_config.setdefault("format", {"type": "text"})
    return {
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "status": status,
        "background": bool(getattr(request, "background", False)),
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "max_tool_calls": getattr(request, "max_tool_calls", None),
        "model": request.model,
        "output": output,
        "parallel_tool_calls": bool(request.parallel_tool_calls),
        "previous_response_id": request.previous_response_id,
        "reasoning": reasoning,
        "service_tier": request.service_tier,
        "store": bool(request.store),
        "temperature": request.temperature,
        "text": text_config,
        "tool_choice": request.tool_choice or "auto",
        "tools": request.tools or [],
        "top_logprobs": getattr(request, "top_logprobs", 0) or 0,
        "top_p": request.top_p,
        "truncation": request.truncation or "disabled",
        "usage": usage,
        "user": request.user,
        "metadata": request.metadata or {},
    }


def chat_response_to_responses(
    chat_response: Dict[str, Any],
    context: ResponsesConversionContext,
) -> Dict[str, Any]:
    """Translate a non-streaming Chat completion to a Responses object."""

    choices = chat_response.get("choices") or []
    if not choices:
        raise ValueError("Chat compatibility response did not contain any choices")
    choice = choices[0]
    message = choice.get("message") or {}
    output = chat_message_to_response_items(message, context)
    status, incomplete_details = response_status(choice.get("finish_reason"))
    return build_response_object(
        context=context,
        output=output,
        status=status,
        usage=response_usage(chat_response.get("usage")),
        incomplete_details=incomplete_details,
    )
