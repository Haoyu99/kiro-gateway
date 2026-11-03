# -*- coding: utf-8 -*-

"""
Unit tests for kiro/native_reasoning.py (native reasoning effort support).

Schemas below are copied from real /ListAvailableModels responses
(q.us-east-1.amazonaws.com, 2026-11-03).
"""

import copy

import pytest

from kiro.native_reasoning import (
    EFFORT_ORDER,
    ReasoningRequest,
    build_additional_model_request_fields,
    resolve_effort,
    supports_native_reasoning,
)


def _claude_schema(thinking_types, efforts, default="high"):
    """Build a Claude-style model info dict like Kiro returns."""
    return {
        "modelId": "claude-test",
        "additionalModelRequestFieldsSchema": {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": list(thinking_types)},
                        "display": {"type": "string", "enum": ["summarized", "omitted"]},
                    },
                    "required": ["type"],
                },
                "output_config": {
                    "type": "object",
                    "properties": {
                        "effort": {"type": "string", "enum": list(efforts), "default": default}
                    },
                },
                "max_tokens": {"type": "integer", "minimum": 1024, "maximum": 64000},
            },
            "additionalProperties": False,
        },
    }


FIVE_LEVELS = ["low", "medium", "high", "xhigh", "max"]

# claude-haiku-5.5: thinking can be disabled, 5 effort levels
HAIKU_5_5 = _claude_schema(["adaptive", "disabled"], FIVE_LEVELS, default="medium")
# claude-opus-5.5: thinking can NOT be disabled
OPUS_5_5 = _claude_schema(["adaptive"], FIVE_LEVELS, default="medium")
# claude-opus-4.6: no xhigh level
OPUS_4_6 = _claude_schema(["adaptive", "disabled"], ["low", "medium", "high", "max"])

GPT_5_6 = {
    "modelId": "gpt-5.6-luna",
    "additionalModelRequestFieldsSchema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "object",
                "properties": {
                    "effort": {
                        "type": "string",
                        "enum": ["none", "low", "medium", "high", "xhigh", "max"],
                        "default": "high",
                    }
                },
            }
        },
        "additionalProperties": False,
    },
}

# claude-sonnet-4.5 and friends: Kiro rejects additionalModelRequestFields entirely
NO_SCHEMA = {"modelId": "claude-sonnet-4.5", "additionalModelRequestFieldsSchema": None}


class TestSupportsNativeReasoning:
    """Tests for supports_native_reasoning()."""

    @pytest.mark.parametrize("info", [HAIKU_5_5, OPUS_5_5, OPUS_4_6, GPT_5_6])
    def test_models_with_effort_schema_are_supported(self, info):
        """
        What it does: Detects Claude-style and OpenAI-style effort schemas.
        Purpose: Only these models may receive additionalModelRequestFields.
        """
        assert supports_native_reasoning(info) is True

    @pytest.mark.parametrize(
        "info",
        [
            None,
            {},
            NO_SCHEMA,
            {"modelId": "x"},
            {"additionalModelRequestFieldsSchema": "not-a-dict"},
            {"additionalModelRequestFieldsSchema": {"properties": "broken"}},
            {"additionalModelRequestFieldsSchema": {"properties": {"max_tokens": {"type": "integer"}}}},
            {"additionalModelRequestFieldsSchema": {"properties": {"output_config": {"properties": {}}}}},
            {"additionalModelRequestFieldsSchema": {"properties": {"output_config": {"properties": {"effort": {"enum": "low"}}}}}},
            "not-a-dict",
        ],
    )
    def test_missing_or_malformed_schema_is_not_supported(self, info):
        """
        What it does: Returns False for absent, partial or malformed schemas.
        Purpose: Never crash on unexpected upstream metadata; fall back to fake reasoning.
        """
        assert supports_native_reasoning(info) is False


class TestResolveEffort:
    """Tests for resolve_effort()."""

    @pytest.mark.parametrize("level", FIVE_LEVELS)
    def test_exact_match_is_kept(self, level):
        """
        What it does: Keeps each of the five upstream levels unchanged.
        Purpose: Low/Medium/High/Xhigh/Max must reach Kiro as requested.
        """
        assert resolve_effort(level, FIVE_LEVELS) == level

    def test_xhigh_maps_down_to_high_when_missing(self):
        """
        What it does: xhigh is equidistant from high and max; the lower one wins.
        Purpose: claude-opus-4.6 / sonnet-4.6 have no xhigh; ties pick the cheaper level.
        """
        assert resolve_effort("xhigh", ["low", "medium", "high", "max"]) == "high"

    def test_minimal_maps_to_low(self):
        """
        What it does: OpenAI "minimal" maps to the closest allowed level.
        Purpose: Clients sending minimal do not get a 400 from Kiro.
        """
        assert resolve_effort("minimal", FIVE_LEVELS) == "low"

    def test_minimal_never_maps_to_none(self):
        """
        What it does: Even when "none" is allowed, a non-none request never resolves to it.
        Purpose: A low effort must not silently disable reasoning.
        """
        assert resolve_effort("minimal", ["none", "low", "medium"]) == "low"

    def test_none_is_kept_when_requested_and_allowed(self):
        """
        What it does: Explicit "none" passes through for GPT models.
        Purpose: GPT 5.6 supports reasoning effort "none".
        """
        assert resolve_effort("none", ["none", "low"]) == "none"

    @pytest.mark.parametrize("bad", ["bogus", "", "HIGH", "ultra"])
    def test_unknown_level_returns_none(self, bad):
        """
        What it does: Unknown levels resolve to None (model default).
        Purpose: Never forward a value Kiro would reject.
        """
        assert resolve_effort(bad, FIVE_LEVELS) is None

    def test_empty_allowed_returns_none(self):
        """
        What it does: No allowed levels means nothing can be chosen.
        Purpose: Guard against empty enums in the schema.
        """
        assert resolve_effort("high", []) is None

    def test_only_none_allowed_returns_none_for_other_levels(self):
        """
        What it does: If only "none" is allowed, a real level cannot be satisfied.
        Purpose: Avoid turning reasoning off when the client asked for it.
        """
        assert resolve_effort("high", ["none"]) is None

    def test_effort_order_is_ascending(self):
        """
        What it does: Checks the ordering constant.
        Purpose: Nearest-level mapping depends on this order.
        """
        assert EFFORT_ORDER == ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class TestReasoningRequest:
    """Tests for ReasoningRequest.is_empty()."""

    def test_default_is_empty(self):
        """
        What it does: No fields set means no preference.
        Purpose: Empty requests must not add fields (pass-through).
        """
        assert ReasoningRequest().is_empty() is True

    @pytest.mark.parametrize(
        "kwargs",
        [{"effort": "low"}, {"disabled": True}, {"enabled": True}, {"display": "omitted"}],
    )
    def test_any_field_makes_it_non_empty(self, kwargs):
        """
        What it does: Any single preference is enough.
        Purpose: Each client signal must be honored.
        """
        assert ReasoningRequest(**kwargs).is_empty() is False


class TestBuildClaudeStyleFields:
    """Tests for build_additional_model_request_fields() with Claude-style schemas."""

    @pytest.mark.parametrize("level", FIVE_LEVELS)
    def test_each_level_produces_adaptive_thinking_and_effort(self, level):
        """
        What it does: Builds fields for each of the five levels.
        Purpose: Exact shape verified against the live API (top-level object).
        """
        fields = build_additional_model_request_fields(
            HAIKU_5_5, ReasoningRequest(effort=level), "claude-haiku-5.5"
        )
        assert fields == {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": level},
        }

    def test_no_preference_sends_nothing(self):
        """
        What it does: Empty request returns None even for capable models.
        Purpose: Model defaults apply; gateway stays transparent.
        """
        assert build_additional_model_request_fields(HAIKU_5_5, ReasoningRequest(), "m") is None

    def test_disabled_uses_thinking_disabled(self):
        """
        What it does: Explicit disable maps to thinking.type=disabled.
        Purpose: Honors Anthropic thinking.type=disabled and OpenAI reasoning_effort=none.
        """
        fields = build_additional_model_request_fields(
            HAIKU_5_5, ReasoningRequest(disabled=True), "claude-haiku-5.5"
        )
        assert fields == {"thinking": {"type": "disabled"}}

    def test_effort_none_is_treated_as_disabled(self):
        """
        What it does: effort="none" on a Claude model disables thinking.
        Purpose: Claude schemas have no "none" effort; sending it would be rejected.
        """
        fields = build_additional_model_request_fields(
            HAIKU_5_5, ReasoningRequest(effort="none"), "claude-haiku-5.5"
        )
        assert fields == {"thinking": {"type": "disabled"}}

    def test_disable_on_model_that_cannot_disable_uses_lowest_effort(self):
        """
        What it does: claude-opus-5.5 only allows adaptive; disable falls back to low effort.
        Purpose: Avoid a 400 for a request the client reasonably made.
        """
        fields = build_additional_model_request_fields(
            OPUS_5_5, ReasoningRequest(disabled=True), "claude-opus-5.5"
        )
        assert fields == {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "low"},
        }

    def test_xhigh_on_model_without_xhigh_maps_to_high(self):
        """
        What it does: Maps unsupported xhigh to high for claude-opus-4.6.
        Purpose: Per-model enums differ; never send an invalid enum value.
        """
        fields = build_additional_model_request_fields(
            OPUS_4_6, ReasoningRequest(effort="xhigh"), "claude-opus-4.6"
        )
        assert fields["output_config"] == {"effort": "high"}

    def test_unknown_effort_keeps_thinking_but_drops_effort(self):
        """
        What it does: Unknown effort is dropped; adaptive thinking is still enabled.
        Purpose: Invalid client values never reach Kiro.
        """
        fields = build_additional_model_request_fields(
            HAIKU_5_5, ReasoningRequest(effort="bogus"), "claude-haiku-5.5"
        )
        assert fields == {"thinking": {"type": "adaptive", "display": "summarized"}}

    def test_enabled_without_effort_only_turns_on_thinking(self):
        """
        What it does: thinking enabled with no effort sends adaptive thinking only.
        Purpose: Model default effort applies.
        """
        fields = build_additional_model_request_fields(
            HAIKU_5_5, ReasoningRequest(enabled=True), "claude-haiku-5.5"
        )
        assert fields == {"thinking": {"type": "adaptive", "display": "summarized"}}

    def test_client_display_is_respected(self):
        """
        What it does: Client-provided display=omitted overrides the summarized default.
        Purpose: Clients that do not want reasoning text can opt out.
        """
        fields = build_additional_model_request_fields(
            HAIKU_5_5, ReasoningRequest(effort="low", display="omitted"), "claude-haiku-5.5"
        )
        assert fields["thinking"] == {"type": "adaptive", "display": "omitted"}

    def test_invalid_client_display_falls_back_to_summarized(self):
        """
        What it does: Unknown display values are replaced by the preferred one.
        Purpose: Never forward a display value outside the schema enum.
        """
        fields = build_additional_model_request_fields(
            HAIKU_5_5, ReasoningRequest(effort="low", display="verbose"), "claude-haiku-5.5"
        )
        assert fields["thinking"]["display"] == "summarized"

    def test_schema_without_display_omits_display(self):
        """
        What it does: No display enum in schema means no display field.
        Purpose: additionalProperties=false would reject an unknown property.
        """
        info = copy.deepcopy(HAIKU_5_5)
        del info["additionalModelRequestFieldsSchema"]["properties"]["thinking"]["properties"]["display"]
        fields = build_additional_model_request_fields(info, ReasoningRequest(effort="max"), "m")
        assert fields == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}

    def test_schema_without_thinking_sends_only_effort(self):
        """
        What it does: Effort-only schemas get only output_config.
        Purpose: Fields are derived from the schema, not hardcoded.
        """
        info = copy.deepcopy(HAIKU_5_5)
        del info["additionalModelRequestFieldsSchema"]["properties"]["thinking"]
        fields = build_additional_model_request_fields(info, ReasoningRequest(effort="max"), "m")
        assert fields == {"output_config": {"effort": "max"}}

    def test_disable_with_no_way_to_disable_and_no_thinking_returns_lowest_effort(self):
        """
        What it does: Schema with effort only, disable requested.
        Purpose: Lowest effort is the closest valid approximation.
        """
        info = copy.deepcopy(HAIKU_5_5)
        del info["additionalModelRequestFieldsSchema"]["properties"]["thinking"]
        fields = build_additional_model_request_fields(info, ReasoningRequest(disabled=True), "m")
        assert fields == {"output_config": {"effort": "low"}}

    def test_does_not_mutate_model_info(self):
        """
        What it does: Builds fields twice and checks the cached schema is unchanged.
        Purpose: Model cache entries are shared across requests.
        """
        info = copy.deepcopy(HAIKU_5_5)
        snapshot = copy.deepcopy(info)
        build_additional_model_request_fields(info, ReasoningRequest(effort="max"), "m")
        build_additional_model_request_fields(info, ReasoningRequest(disabled=True), "m")
        assert info == snapshot


class TestBuildOpenAIStyleFields:
    """Tests for build_additional_model_request_fields() with OpenAI-style schemas."""

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
    def test_levels_map_to_reasoning_effort(self, level):
        """
        What it does: Builds reasoning.effort for GPT models.
        Purpose: GPT 5.6 uses a different schema family than Claude.
        """
        fields = build_additional_model_request_fields(
            GPT_5_6, ReasoningRequest(effort=level), "gpt-5.6-luna"
        )
        assert fields == {"reasoning": {"effort": level}}

    def test_disabled_maps_to_none(self):
        """
        What it does: Disable becomes reasoning.effort=none.
        Purpose: GPT schema expresses "off" via the effort enum.
        """
        fields = build_additional_model_request_fields(
            GPT_5_6, ReasoningRequest(disabled=True), "gpt-5.6-luna"
        )
        assert fields == {"reasoning": {"effort": "none"}}

    def test_enabled_without_effort_sends_nothing(self):
        """
        What it does: No effort and not disabled returns None.
        Purpose: GPT schema has no thinking toggle; default applies.
        """
        assert build_additional_model_request_fields(
            GPT_5_6, ReasoningRequest(enabled=True), "gpt-5.6-luna"
        ) is None

    def test_unknown_effort_sends_nothing(self):
        """
        What it does: Invalid effort is dropped.
        Purpose: Kiro returns 400 for values outside the enum (verified live).
        """
        assert build_additional_model_request_fields(
            GPT_5_6, ReasoningRequest(effort="bogus"), "gpt-5.6-luna"
        ) is None

    def test_never_sends_claude_fields(self):
        """
        What it does: Claude-only properties are never produced for GPT.
        Purpose: Kiro rejects properties not defined in the schema (verified live).
        """
        fields = build_additional_model_request_fields(
            GPT_5_6, ReasoningRequest(effort="high", enabled=True, display="summarized"), "gpt"
        )
        assert set(fields) == {"reasoning"}


class TestBuildFieldsForUnsupportedModels:
    """Tests for models without a native reasoning schema."""

    @pytest.mark.parametrize("info", [None, NO_SCHEMA, {}])
    @pytest.mark.parametrize(
        "reasoning",
        [ReasoningRequest(effort="max"), ReasoningRequest(disabled=True), ReasoningRequest(enabled=True)],
    )
    def test_returns_none(self, info, reasoning):
        """
        What it does: Unsupported models never get additionalModelRequestFields.
        Purpose: Kiro answers 400 "not supported for this model" (verified live).
        """
        assert build_additional_model_request_fields(info, reasoning, "claude-sonnet-4.5") is None
