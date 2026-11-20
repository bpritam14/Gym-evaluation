# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the Gym/OpenCode boundary native and detect incomplete model replies."""

from __future__ import annotations

import json
import re


MIXED_PLANNING_MARKER = "<mixed_planning_output>"
_PLANNING_FIELDS = {"evaluation_previous_goal", "memory", "next_goal", "current_plan_item", "plan_update"}


class ModelProtocolError(ValueError):
    """The model intended an action but did not return an executable tool call."""


def requires_mixed_planning(messages: list[dict], *, tools_available: bool) -> bool:
    """Recognize mixed actor turns without changing title or compaction requests."""
    return tools_available and any(
        message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and MIXED_PLANNING_MARKER in message["content"]
        for message in messages
    )


def _unfence_planning(content: str) -> str:
    """Remove bounded whole-content presentation wrappers, never extract actions.

    Accept JSON-string encoding, paired cosmetic quotes, inline backticks and
    complete JSON/untagged Markdown fences. The entire resulting text must still
    validate as the planning object. Prose, sibling blocks and textual tool calls
    are never promoted to executable content or native tool_calls.
    """
    value = content.strip()
    for _ in range(4):
        match = re.fullmatch(r"(`{3,})(?:json)?[ \t]*\r?\n(.*?)\r?\n\1", value, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(2).strip()
            continue
        if value.startswith('"'):
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                pass
            else:
                if isinstance(decoded, str):
                    value = decoded.strip()
                    continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"', "`"):
            inner = value[1:-1].strip()
            if inner.startswith(("{", "```")):
                value = inner
                continue
        break
    return value


def planning_fields(content: str, *, final: bool) -> dict:
    """Validate visible mixed planning without manufacturing missing fields."""
    try:
        fields = json.loads(_unfence_planning(content) if isinstance(content, str) else content)
    except (ValueError, TypeError) as exc:
        raise ModelProtocolError("Mixed planning content must be one JSON object, alongside MCP tool_calls") from exc
    if not isinstance(fields, dict) or not _PLANNING_FIELDS.issubset(fields):
        raise ModelProtocolError("Mixed planning content must contain all five planning fields")
    allowed = _PLANNING_FIELDS | ({"final_response"} if final else set())
    if set(fields) - allowed:
        raise ModelProtocolError("Mixed planning content contains unsupported fields; actions belong in tool_calls")
    for name in ("evaluation_previous_goal", "memory", "next_goal"):
        if not isinstance(fields[name], str) or not fields[name].strip():
            raise ModelProtocolError(f"Mixed planning field {name} must be a nonempty string")
    item = fields["current_plan_item"]
    if item is not None and (type(item) is not int or item < 0):
        raise ModelProtocolError("Mixed current_plan_item must be null or a nonnegative integer")
    plan = fields["plan_update"]
    if plan is not None and (
        not isinstance(plan, list) or any(not isinstance(value, str) or not value.strip() for value in plan)
    ):
        raise ModelProtocolError("Mixed plan_update must be null or a list of nonempty plan-item strings")
    if final and (not isinstance(fields.get("final_response"), str) or not fields["final_response"].strip()):
        raise ModelProtocolError("Mixed final response requires a nonempty final_response string")
    return fields


def native_message(
    message: dict, *, tools_available: bool, finish_reason: str | None = None, planning_required: bool = False
) -> dict:
    """Undo Gym's reasoning wrapper without executing tool-like reasoning text."""
    result = dict(message)
    if len(result.get("tool_calls") or []) > 32:
        raise ModelProtocolError(
            "Model returned more than 32 tool calls in one turn; retry with a smaller batch. "
            "No calls from this response were executed."
        )
    content = result.get("content") or ""
    if not isinstance(content, str):
        if planning_required and finish_reason != "length":
            raise ModelProtocolError("Mixed planning content must be a JSON text object")
        return result
    match = re.match(r"^<think>\n?(.*?)</think>(.*)$", content, re.DOTALL)
    if match:
        reasoning, content = match.groups()
        result["reasoning_content"] = reasoning
        result["reasoning"] = reasoning
        result["content"] = content or None
    if planning_required and finish_reason != "length":
        content = _unfence_planning(content)
        planning_fields(content, final=not bool(result.get("tool_calls")))
        result["content"] = content
    if tools_available and not result.get("tool_calls"):
        if finish_reason == "length":
            # Preserve the provider's output limit for native OpenCode. An
            # exhausted generation is not malformed, and reasoning text is
            # still never interpreted as an executable action.
            return result
        if re.search(
            r"<tool_call>.*</tool_call>\s*(?:</think>)?\s*$",
            message.get("content") or "",
            re.DOTALL,
        ):
            raise ModelProtocolError(
                "Model returned tool-call markup as text instead of structured tool_calls; retry generation. "
                "Do not execute actions embedded in reasoning."
            )
        if not content.strip():
            raise ModelProtocolError(
                "Model returned only reasoning, with no answer or structured action; retry generation"
            )
    return result


def has_model_turn(events: list[dict]) -> bool:
    """Require generated content/action/token evidence, not process startup."""
    for event in events:
        if not isinstance(event, dict):
            continue
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        if event.get("type") == "text" and str(part.get("text") or "").strip():
            return True
        if event.get("type") == "tool_use" and part.get("tool"):
            return True
        if event.get("type") == "step_finish":
            tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
            if any(type(tokens.get(k)) in (int, float) and tokens[k] > 0 for k in ("output", "reasoning")):
                return True
    return False


def execution_health(events: list[dict], *, timed_out: bool = False) -> dict:
    """Report native execution independently from task score and process exit."""
    from saas_bench.opencode.core import context_overflow_recovered

    errors = [event for event in events if event.get("type") == "error"]
    recovered = context_overflow_recovered(events) if errors else False
    tools = [e.get("part", {}) for e in events if e.get("type") == "tool_use"]
    failed = [t for t in tools if t.get("state", {}).get("status") == "error"]
    final = next(
        (e.get("part", {}).get("text", "") for e in reversed(events) if e.get("type") == "text"),
        "",
    )
    protocol_errors = any(
        "Model returned tool-call markup as text" in str(e)
        or "Model returned only reasoning" in str(e)
        or "Model returned more than 32 tool calls" in str(e)
        or "Model violated the native tool protocol" in str(e)
        for e in events
        if e.get("type") == "error"
    )
    malformed = bool(re.search(r"<tool_call>.*</tool_call>", final, re.DOTALL))
    last_finish = next(
        (e.get("part", {}) for e in reversed(events) if e.get("type") == "step_finish"),
        {},
    )
    output_limit = last_finish.get("reason") == "length"
    termination = (
        "timeout"
        if timed_out
        else "model_protocol_error"
        if protocol_errors
        else "context_limit"
        if not recovered and any(e.get("error", {}).get("name") == "ContextOverflowError" for e in errors)
        else "native_error"
        if errors and not recovered
        else "output_token_limit"
        if output_limit
        else "malformed_tool_call"
        if malformed
        else "final_response"
    )
    return {
        "tool_calls": len(tools),
        "tool_errors": len(failed),
        "termination": termination,
        "execution_healthy": not failed and termination == "final_response",
        "native_error_events": len(errors),
        "recovered_context_overflow": recovered,
        "output_token_limit": output_limit,
        "task_success": None,
    }
