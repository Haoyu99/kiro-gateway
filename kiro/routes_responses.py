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

"""OpenAI Responses API route used by current Codex clients."""

import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from kiro.converters_responses import (
    chat_response_to_responses,
    responses_request_to_chat,
)
from kiro.models_responses import ResponseCreateRequest
from kiro.routes_openai import chat_completions, verify_api_key
from kiro.streaming_responses import stream_chat_to_responses


router = APIRouter()


def _json_response_body(response: Response) -> Dict[str, Any]:
    """Decode the JSON body of a Starlette response."""

    try:
        value = json.loads(response.body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("Chat compatibility route returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Chat compatibility route returned a non-object JSON response")
    return value


@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def create_response(
    request: Request,
    request_data: ResponseCreateRequest,
) -> Response:
    """Create a response through the existing Kiro Chat compatibility path.

    Args:
        request: FastAPI request carrying initialized account state.
        request_data: OpenAI Responses create request.

    Returns:
        A Responses object or typed Responses SSE stream.

    Raises:
        HTTPException: If the request uses unsupported state or Item types.
    """

    logger.info(
        f"Request to /v1/responses (model={request_data.model}, stream={request_data.stream})"
    )
    try:
        chat_request, context = responses_request_to_chat(request_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chat_result = await chat_completions(request, chat_request)
    if isinstance(chat_result, StreamingResponse):
        passthrough_headers = {
            name: value
            for name, value in chat_result.headers.items()
            if name.lower() not in {"content-length", "content-type"}
        }
        return StreamingResponse(
            stream_chat_to_responses(chat_result.body_iterator, context),
            status_code=chat_result.status_code,
            headers=passthrough_headers,
            media_type="text/event-stream",
            background=chat_result.background,
        )

    if isinstance(chat_result, Response):
        if chat_result.status_code >= 400:
            return chat_result
        try:
            chat_payload = _json_response_body(chat_result)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    elif isinstance(chat_result, dict):
        chat_payload = chat_result
    else:
        raise HTTPException(
            status_code=502,
            detail="Chat compatibility route returned an unsupported response type",
        )

    try:
        response_payload = chat_response_to_responses(chat_payload, context)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(content=response_payload)
