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

"""Pydantic request models for the OpenAI Responses API adapter."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class ResponseCreateRequest(BaseModel):
    """Request accepted by the ``POST /v1/responses`` endpoint.

    The Responses API evolves frequently, so unknown fields are accepted and
    forwarded only when the Chat Completions compatibility layer understands
    them. The declared fields cover requests emitted by current Codex clients.
    """

    model: str
    input: Union[str, List[Any]]
    instructions: Optional[Union[str, List[Any]]] = None
    stream: bool = False

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = True

    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning: Optional[Dict[str, Any]] = None
    text: Optional[Dict[str, Any]] = None

    include: Optional[List[str]] = None
    store: Optional[bool] = False
    previous_response_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    truncation: Optional[Literal["auto", "disabled"]] = None
    user: Optional[str] = None
    service_tier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    client_metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}
