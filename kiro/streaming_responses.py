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

"""Translate Chat Completions SSE chunks into Responses API events."""

import json
import uuid
from typing import Any, AsyncIterable, AsyncIterator, Dict, List, Optional

from loguru import logger

from kiro.converters_responses import (
    ResponsesConversionContext,
    build_response_object,
    custom_tool_input,
    response_status,
    response_usage,
)


async def _iter_sse_data(chunks: AsyncIterable[Any]) -> AsyncIterator[str]:
    """Yield data payloads from an arbitrary stream of SSE byte chunks."""

    buffer = ""
    async for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        buffer += str(chunk).replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            data_lines = [line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")]
            if data_lines:
                yield "\n".join(data_lines)
    if buffer.strip():
        data_lines = [line[5:].lstrip() for line in buffer.splitlines() if line.startswith("data:")]
        if data_lines:
            yield "\n".join(data_lines)


class ResponsesStreamState:
    """Accumulate Chat deltas and create ordered Responses stream events."""

    def __init__(self, context: ResponsesConversionContext):
        """Initialize stream state for one response."""

        self.context = context
        self.sequence_number = 0
        self.output: List[Dict[str, Any]] = []
        self.text_item: Optional[Dict[str, Any]] = None
        self.text_output_index: Optional[int] = None
        self.text = ""
        self.reasoning_item: Optional[Dict[str, Any]] = None
        self.reasoning_output_index: Optional[int] = None
        self.reasoning = ""
        self.tool_items: Dict[int, Dict[str, Any]] = {}
        self.tool_output_indexes: Dict[int, int] = {}
        self.finish_reason: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None

    def event(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        """Create an event with a monotonically increasing sequence number."""

        event = {
            "type": event_type,
            "sequence_number": self.sequence_number,
            **fields,
        }
        self.sequence_number += 1
        return event

    def initial_events(self) -> List[Dict[str, Any]]:
        """Create the response.created and response.in_progress events."""

        response = build_response_object(
            context=self.context,
            output=[],
            status="in_progress",
        )
        return [
            self.event("response.created", response=response),
            self.event("response.in_progress", response=response),
        ]

    def _start_reasoning(self) -> List[Dict[str, Any]]:
        """Start a reasoning summary output Item."""

        item_id = f"rs_{uuid.uuid4().hex}"
        item = {
            "id": item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
        }
        self.reasoning_item = item
        self.reasoning_output_index = len(self.output)
        self.output.append(item)
        part = {"type": "summary_text", "text": ""}
        item["summary"] = [part]
        return [
            self.event(
                "response.output_item.added",
                output_index=self.reasoning_output_index,
                item={**item, "summary": []},
            ),
            self.event(
                "response.reasoning_summary_part.added",
                item_id=item_id,
                output_index=self.reasoning_output_index,
                summary_index=0,
                part=dict(part),
            ),
        ]

    def add_reasoning(self, delta: str) -> List[Dict[str, Any]]:
        """Append one reasoning summary delta."""

        events = self._start_reasoning() if self.reasoning_item is None else []
        self.reasoning += delta
        if self.reasoning_item:
            self.reasoning_item["summary"][0]["text"] = self.reasoning
            events.append(
                self.event(
                    "response.reasoning_summary_text.delta",
                    item_id=self.reasoning_item["id"],
                    output_index=self.reasoning_output_index,
                    summary_index=0,
                    delta=delta,
                )
            )
        return events

    def _start_text(self) -> List[Dict[str, Any]]:
        """Start an assistant message output Item and output_text part."""

        item_id = f"msg_{uuid.uuid4().hex}"
        item = {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        self.text_item = item
        self.text_output_index = len(self.output)
        self.output.append(item)
        part = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
        item["content"] = [part]
        return [
            self.event(
                "response.output_item.added",
                output_index=self.text_output_index,
                item={**item, "content": []},
            ),
            self.event(
                "response.content_part.added",
                item_id=item_id,
                output_index=self.text_output_index,
                content_index=0,
                part=dict(part),
            ),
        ]

    def add_text(self, delta: str) -> List[Dict[str, Any]]:
        """Append one output text delta."""

        events = self._start_text() if self.text_item is None else []
        self.text += delta
        if self.text_item:
            self.text_item["content"][0]["text"] = self.text
            events.append(
                self.event(
                    "response.output_text.delta",
                    item_id=self.text_item["id"],
                    output_index=self.text_output_index,
                    content_index=0,
                    delta=delta,
                    logprobs=[],
                )
            )
        return events

    def add_tool_call(self, raw_call: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Append a Chat tool-call delta and emit the matching typed event."""

        index = int(raw_call.get("index", len(self.tool_items)))
        function = raw_call.get("function") or {}
        name = str(function.get("name") or "")
        argument_delta = function.get("arguments") or ""
        if not isinstance(argument_delta, str):
            argument_delta = json.dumps(argument_delta, ensure_ascii=False)
        events: List[Dict[str, Any]] = []

        if index not in self.tool_items:
            call_id = str(raw_call.get("id") or f"call_{uuid.uuid4().hex}")
            is_custom = self.context.tool_kinds.get(name) == "custom"
            response_name = self.context.tool_response_names.get(name, name)
            namespace = self.context.tool_namespaces.get(name)
            item = {
                "id": f"{'ctc' if is_custom else 'fc'}_{uuid.uuid4().hex}",
                "type": "custom_tool_call" if is_custom else "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": response_name,
            }
            if namespace:
                item["namespace"] = namespace
            item["input" if is_custom else "arguments"] = ""
            self.tool_items[index] = item
            self.tool_output_indexes[index] = len(self.output)
            self.output.append(item)
            events.append(
                self.event(
                    "response.output_item.added",
                    output_index=self.tool_output_indexes[index],
                    item=dict(item),
                )
            )

        item = self.tool_items[index]
        if name:
            item["name"] = self.context.tool_response_names.get(name, name)
            namespace = self.context.tool_namespaces.get(name)
            if namespace:
                item["namespace"] = namespace
        value_field = "input" if item["type"] == "custom_tool_call" else "arguments"
        if item["type"] == "custom_tool_call":
            previous_arguments = item.pop("_arguments", "")
            combined_arguments = previous_arguments + argument_delta
            item["_arguments"] = combined_arguments
            try:
                parsed_arguments = json.loads(combined_arguments)
            except json.JSONDecodeError:
                parsed_arguments = None
            if isinstance(parsed_arguments, dict) and "input" in parsed_arguments:
                parsed_input = parsed_arguments["input"]
                custom_input = (
                    parsed_input
                    if isinstance(parsed_input, str)
                    else json.dumps(parsed_input, ensure_ascii=False)
                )
                previous_input = item.get("input", "")
                item["input"] = custom_input
                emitted_delta = (
                    custom_input[len(previous_input):]
                    if custom_input.startswith(previous_input)
                    else custom_input
                )
            else:
                emitted_delta = ""
            if emitted_delta:
                events.append(
                    self.event(
                        "response.custom_tool_call_input.delta",
                        item_id=item["id"],
                        output_index=self.tool_output_indexes[index],
                        delta=emitted_delta,
                    )
                )
        else:
            item[value_field] += argument_delta
            if argument_delta:
                events.append(
                    self.event(
                        "response.function_call_arguments.delta",
                        item_id=item["id"],
                        output_index=self.tool_output_indexes[index],
                        delta=argument_delta,
                    )
                )
        return events

    def consume_chat_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consume one Chat completion chunk."""

        events: List[Dict[str, Any]] = []
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                events.extend(self.add_reasoning(str(delta["reasoning_content"])))
            if delta.get("content") is not None and delta.get("content") != "":
                events.extend(self.add_text(str(delta["content"])))
            for tool_call in delta.get("tool_calls") or []:
                events.extend(self.add_tool_call(tool_call))
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if chunk.get("usage") is not None:
            self.usage = response_usage(chunk["usage"])
        return events

    def _finish_reasoning(self) -> List[Dict[str, Any]]:
        """Complete the reasoning summary Item."""

        if not self.reasoning_item:
            return []
        item = self.reasoning_item
        item["status"] = "completed"
        part = item["summary"][0]
        return [
            self.event(
                "response.reasoning_summary_text.done",
                item_id=item["id"],
                output_index=self.reasoning_output_index,
                summary_index=0,
                text=self.reasoning,
            ),
            self.event(
                "response.reasoning_summary_part.done",
                item_id=item["id"],
                output_index=self.reasoning_output_index,
                summary_index=0,
                part=part,
            ),
            self.event(
                "response.output_item.done",
                output_index=self.reasoning_output_index,
                item=item,
            ),
        ]

    def _finish_text(self) -> List[Dict[str, Any]]:
        """Complete the assistant message Item."""

        if not self.text_item:
            return []
        item = self.text_item
        item["status"] = "completed"
        part = item["content"][0]
        return [
            self.event(
                "response.output_text.done",
                item_id=item["id"],
                output_index=self.text_output_index,
                content_index=0,
                text=self.text,
                logprobs=[],
            ),
            self.event(
                "response.content_part.done",
                item_id=item["id"],
                output_index=self.text_output_index,
                content_index=0,
                part=part,
            ),
            self.event(
                "response.output_item.done",
                output_index=self.text_output_index,
                item=item,
            ),
        ]

    def _finish_tools(self) -> List[Dict[str, Any]]:
        """Complete all tool-call Items."""

        events: List[Dict[str, Any]] = []
        for index, item in self.tool_items.items():
            output_index = self.tool_output_indexes[index]
            item["status"] = "completed"
            if item["type"] == "custom_tool_call":
                raw_arguments = item.pop("_arguments", "")
                final_input = custom_tool_input(raw_arguments)
                previous_input = item.get("input", "")
                if final_input != previous_input:
                    item["input"] = final_input
                    events.append(
                        self.event(
                            "response.custom_tool_call_input.delta",
                            item_id=item["id"],
                            output_index=output_index,
                            delta=final_input,
                        )
                    )
                events.append(
                    self.event(
                        "response.custom_tool_call_input.done",
                        item_id=item["id"],
                        output_index=output_index,
                        input=item["input"],
                    )
                )
            else:
                events.append(
                    self.event(
                        "response.function_call_arguments.done",
                        item_id=item["id"],
                        output_index=output_index,
                        arguments=item["arguments"],
                    )
                )
            events.append(
                self.event(
                    "response.output_item.done",
                    output_index=output_index,
                    item=item,
                )
            )
        return events

    def final_events(self) -> List[Dict[str, Any]]:
        """Complete all Items and emit the terminal response event."""

        events = self._finish_reasoning() + self._finish_text() + self._finish_tools()
        status, incomplete_details = response_status(self.finish_reason)
        response = build_response_object(
            context=self.context,
            output=self.output,
            status=status,
            usage=self.usage,
            incomplete_details=incomplete_details,
        )
        terminal_type = "response.completed" if status == "completed" else "response.incomplete"
        events.append(self.event(terminal_type, response=response))
        return events


async def stream_chat_to_responses(
    chunks: AsyncIterable[Any],
    context: ResponsesConversionContext,
) -> AsyncIterator[str]:
    """Convert an OpenAI Chat SSE stream into Responses typed SSE events."""

    state = ResponsesStreamState(context)
    for event in state.initial_events():
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    async for data in _iter_sse_data(chunks):
        if not data or data == "[DONE]":
            continue
        try:
            chat_chunk = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed Chat SSE data while adapting Responses stream")
            continue
        for event in state.consume_chat_chunk(chat_chunk):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    for event in state.final_events():
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
