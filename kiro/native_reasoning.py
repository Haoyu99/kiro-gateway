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

"""
Native reasoning (effort) support for Kiro API.

Newer Kiro models advertise an `additionalModelRequestFieldsSchema` in
/ListAvailableModels. A request may carry a top-level `additionalModelRequestFields`
object that Kiro validates strictly against that schema (unknown properties or
enum values are rejected with REQUEST_BODY_INVALID, and models without a schema
reject the field entirely).

Two schema families exist today:

- Claude style::

    {"thinking": {"type": "adaptive" | "disabled" | ..., "display": "summarized" | "omitted"},
     "output_config": {"effort": "low" | "medium" | "high" | "xhigh" | "max"}}

- OpenAI style::

    {"reasoning": {"effort": "none" | "low" | "medium" | "high" | "xhigh" | "max"}}

This module is the single place that turns a client's reasoning intent
(OpenAI `reasoning_effort`, Anthropic `thinking` / `output_config.effort`,
Responses `reasoning.effort`) into fields that are valid for the target model.
It reads the allowed values from the model's schema instead of hardcoding
per-model tables, so new models work without code changes.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

# Effort levels ordered from least to most reasoning.
# Used to map a requested level to the nearest level a model supports.
EFFORT_ORDER: List[str] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

# Thinking type values in the Claude-style schema
THINKING_TYPE_ADAPTIVE: str = "adaptive"
THINKING_TYPE_DISABLED: str = "disabled"

# Preferred thinking display: "summarized" returns readable reasoning text,
# "omitted" returns none. We prefer readable text so clients can show it.
PREFERRED_THINKING_DISPLAY: str = "summarized"


@dataclass
class ReasoningRequest:
    """
    API-agnostic reasoning intent extracted from a client request.

    Attributes:
        effort: Requested effort level (any value from EFFORT_ORDER), or None
            to let the model use its default effort
        disabled: True if the client explicitly turned reasoning off
        enabled: True if the client explicitly turned reasoning on
            (e.g. Anthropic thinking.type "enabled" / "adaptive")
        display: Requested thinking display ("summarized" / "omitted"), or None
    """
    effort: Optional[str] = None
    disabled: bool = False
    enabled: bool = False
    display: Optional[str] = None

    def is_empty(self) -> bool:
        """
        Check whether the client expressed no reasoning preference at all.

        Returns:
            True if nothing was requested (the model default should apply)
        """
        return not (self.effort or self.disabled or self.enabled or self.display)


def _get_schema_properties(model_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return the top-level properties of a model's additional fields schema.

    Args:
        model_info: Model metadata from ModelInfoCache (may be None)

    Returns:
        Properties dictionary, or an empty dict if the model has no schema
    """
    if not isinstance(model_info, dict):
        return {}
    schema = model_info.get("additionalModelRequestFieldsSchema")
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _get_enum(properties: Dict[str, Any], parent: str, child: str) -> List[str]:
    """
    Read the enum list of a nested string property, e.g. output_config.effort.

    Args:
        properties: Top-level schema properties
        parent: Name of the object property (e.g. "output_config")
        child: Name of the nested string property (e.g. "effort")

    Returns:
        Allowed values, or an empty list if the property or enum is missing
    """
    parent_schema = properties.get(parent)
    if not isinstance(parent_schema, dict):
        return []
    child_schema = (parent_schema.get("properties") or {}).get(child)
    if not isinstance(child_schema, dict):
        return []
    enum = child_schema.get("enum")
    if not isinstance(enum, list):
        return []
    return [value for value in enum if isinstance(value, str)]


def supports_native_reasoning(model_info: Optional[Dict[str, Any]]) -> bool:
    """
    Check whether a model accepts native reasoning fields.

    Args:
        model_info: Model metadata from ModelInfoCache (may be None)

    Returns:
        True if the model schema exposes a Claude-style or OpenAI-style effort field
    """
    properties = _get_schema_properties(model_info)
    return bool(
        _get_enum(properties, "output_config", "effort")
        or _get_enum(properties, "reasoning", "effort")
    )


def resolve_effort(requested: str, allowed: List[str]) -> Optional[str]:
    """
    Map a requested effort level to the closest level the model allows.

    Exact matches are returned as-is. Otherwise the nearest level by position in
    EFFORT_ORDER is chosen; on a tie the lower (cheaper) level wins.
    "none" is only chosen when it was requested, so a low effort never silently
    turns reasoning off. Unknown requested values return None so the model
    default is used.

    Args:
        requested: Effort level requested by the client
        allowed: Effort levels allowed by the model schema

    Returns:
        Effort level to send, or None if nothing sensible can be chosen

    Examples:
        >>> resolve_effort("xhigh", ["low", "medium", "high", "max"])
        'high'
        >>> resolve_effort("minimal", ["low", "medium", "high", "xhigh", "max"])
        'low'
        >>> resolve_effort("minimal", ["none", "low", "medium", "high"])
        'low'
    """
    if requested in allowed:
        return requested
    if requested not in EFFORT_ORDER:
        return None

    ordered_allowed = [level for level in EFFORT_ORDER if level in allowed and level != "none"]
    if not ordered_allowed:
        return None

    requested_index = EFFORT_ORDER.index(requested)
    # min() keeps the first candidate on ties; ordered_allowed is ascending,
    # so ties resolve to the lower level.
    return min(ordered_allowed, key=lambda level: abs(EFFORT_ORDER.index(level) - requested_index))


def _build_claude_style_fields(
    properties: Dict[str, Any], reasoning: ReasoningRequest, model_id: str
) -> Dict[str, Any]:
    """
    Build Claude-style fields (thinking + output_config.effort).

    Args:
        properties: Top-level schema properties
        reasoning: Client reasoning intent
        model_id: Model ID (for logging)

    Returns:
        Fields dictionary (may be empty)
    """
    thinking_types = _get_enum(properties, "thinking", "type")
    display_values = _get_enum(properties, "thinking", "display")
    effort_values = _get_enum(properties, "output_config", "effort")

    effort = reasoning.effort
    wants_disabled = reasoning.disabled or effort == "none"

    if wants_disabled:
        if THINKING_TYPE_DISABLED in thinking_types:
            return {"thinking": {"type": THINKING_TYPE_DISABLED}}
        # Some models (e.g. claude-opus-5.5) cannot turn thinking off.
        # Use the lowest effort they allow instead of failing the request.
        effort = next((level for level in EFFORT_ORDER if level in effort_values), None)
        logger.info(
            f"Model {model_id} does not allow disabling thinking; using lowest effort '{effort}'"
        )

    fields: Dict[str, Any] = {}

    # Turn on adaptive thinking explicitly so the requested effort is applied
    # to reasoning regardless of the model's default thinking mode.
    if THINKING_TYPE_ADAPTIVE in thinking_types:
        thinking: Dict[str, Any] = {"type": THINKING_TYPE_ADAPTIVE}
        if reasoning.display in display_values:
            thinking["display"] = reasoning.display
        elif PREFERRED_THINKING_DISPLAY in display_values:
            # Readable reasoning text so clients can show it
            thinking["display"] = PREFERRED_THINKING_DISPLAY
        fields["thinking"] = thinking

    if effort and effort_values:
        resolved = resolve_effort(effort, effort_values)
        if resolved:
            if resolved != effort:
                logger.info(f"Effort '{effort}' not supported by {model_id}; using '{resolved}'")
            fields["output_config"] = {"effort": resolved}
        else:
            logger.warning(f"Ignoring unknown reasoning effort '{effort}' for {model_id}")

    return fields


def _build_openai_style_fields(
    properties: Dict[str, Any], reasoning: ReasoningRequest, model_id: str
) -> Dict[str, Any]:
    """
    Build OpenAI-style fields (reasoning.effort).

    Args:
        properties: Top-level schema properties
        reasoning: Client reasoning intent
        model_id: Model ID (for logging)

    Returns:
        Fields dictionary (may be empty)
    """
    effort_values = _get_enum(properties, "reasoning", "effort")
    effort = "none" if reasoning.disabled else reasoning.effort
    if not effort:
        return {}

    resolved = resolve_effort(effort, effort_values)
    if not resolved:
        logger.warning(f"Ignoring unknown reasoning effort '{effort}' for {model_id}")
        return {}
    if resolved != effort:
        logger.info(f"Effort '{effort}' not supported by {model_id}; using '{resolved}'")
    return {"reasoning": {"effort": resolved}}


def build_additional_model_request_fields(
    model_info: Optional[Dict[str, Any]],
    reasoning: ReasoningRequest,
    model_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Build the `additionalModelRequestFields` object for a Kiro request.

    Only fields present in the model's own schema are produced, so the result
    always passes Kiro's validation. Models without a schema get None, and the
    caller must not send the field (Kiro rejects it for those models).
    When the client expressed no preference, nothing is sent and the model
    uses its own defaults (transparent pass-through).

    Args:
        model_info: Model metadata from ModelInfoCache (may be None)
        reasoning: Client reasoning intent
        model_id: Model ID (for logging)

    Returns:
        Fields to send, or None if nothing should be sent

    Examples:
        >>> build_additional_model_request_fields(haiku_5_5_info, ReasoningRequest(effort="max"), "claude-haiku-5.5")
        {'thinking': {'type': 'adaptive', 'display': 'summarized'}, 'output_config': {'effort': 'max'}}
        >>> build_additional_model_request_fields(gpt_info, ReasoningRequest(effort="low"), "gpt-5.6-luna")
        {'reasoning': {'effort': 'low'}}
        >>> build_additional_model_request_fields(None, ReasoningRequest(effort="low"), "claude-sonnet-4.5")
        None
    """
    properties = _get_schema_properties(model_info)

    if reasoning.is_empty():
        return None

    if _get_enum(properties, "output_config", "effort"):
        fields = _build_claude_style_fields(properties, reasoning, model_id)
    elif _get_enum(properties, "reasoning", "effort"):
        fields = _build_openai_style_fields(properties, reasoning, model_id)
    else:
        if reasoning.effort and not reasoning.disabled:
            logger.debug(
                f"Model {model_id} has no native reasoning schema; effort '{reasoning.effort}' "
                f"handled by fake reasoning only"
            )
        return None

    if fields:
        logger.debug(f"Native reasoning fields for {model_id}: {fields}")
    return fields or None
