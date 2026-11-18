# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use actual Gym response schemas at the OpenCode-specific adapter boundary."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from responses_api_agents.saas_bench_agent.runtime import BENCHMARK_ROOT
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


sys.path.insert(0, str(BENCHMARK_ROOT))
from responses_api_models.vllm_model import saas_opencode as module


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_real_gym_schema_preserves_reasoning_and_rejects_unexecuted_action(monkeypatch, malformed):
    async def upstream(self, request, body):
        return NeMoGymChatCompletion.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "super",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "<think>Next<tool_call><function=saas_browser_interact></function></tool_call></think>"
                            if malformed
                            else "<think>Checked UI</think>Saved document",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    monkeypatch.setattr("nemo_gym.config.global_config.get_global_config_dict", dict)
    config = VLLMModelConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="saas_opencode.py",
        name="policy_model",
        base_url="http://localhost:9999/v1",
        api_key="EMPTY",
        model="super",
        return_token_id_information=False,
        uses_reasoning_parser=True,
    )
    adapter = module.SaasOpenCodeModel(config=config, server_client=MagicMock(spec=ServerClient))
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "super",
            "messages": [{"role": "user", "content": "Complete task"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "saas_browser_interact",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    if malformed:
        with pytest.raises(HTTPException) as error:
            await adapter.chat_completions(None, body)
        assert error.value.status_code == 422
        assert "after 3 provider attempts" in error.value.detail
        with pytest.raises(HTTPException) as retry:
            await adapter.chat_completions(None, body)
        assert retry.value.status_code == 422
    else:
        response = await adapter.chat_completions(None, body)
        msg = response.model_dump()["choices"][0]["message"]
        assert msg["content"] == "Saved document"
        assert msg["reasoning_content"] == "Checked UI"


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("frequency_penalty,client_penalty", [(0.0, 0.3), (0.3, 0.0)])
async def test_effective_outbound_super_sampling_overrides_client_defaults(
    monkeypatch, thinking, frequency_penalty, client_penalty
):
    from starlette.requests import Request

    monkeypatch.setattr("nemo_gym.config.global_config.get_global_config_dict", dict)
    config = VLLMModelConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="saas_opencode.py",
        name="policy_model",
        base_url="http://127.0.0.1:9/v1",
        api_key="EMPTY",
        model="super",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        sampling_overrides={
            "temperature": 0.6,
            "top_p": 0.95,
            "frequency_penalty": frequency_penalty,
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )
    client = MagicMock()
    client.create_chat_completion = AsyncMock(
        return_value={
            "id": "response",
            "object": "chat.completion",
            "created": 1,
            "model": "super",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Saved"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
    )
    monkeypatch.setattr(VLLMModel, "_resolve_client", lambda self, request: client)
    adapter = module.SaasOpenCodeModel(config=config, server_client=MagicMock(spec=ServerClient))
    request = Request({"type": "http", "headers": [], "session": {}})
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "super",
            "messages": [{"role": "user", "content": "Use the app"}],
            "temperature": 1.0,
            "top_p": 1.0,
            "frequency_penalty": client_penalty,
            "max_tokens": 16384,
        }
    )
    await adapter.chat_completions(request, body)
    sent = client.create_chat_completion.call_args.kwargs
    assert sent["temperature"] == 0.6 and sent["top_p"] == 0.95
    assert sent["frequency_penalty"] == frequency_penalty
    assert sent["max_tokens"] == 16384
    assert sent["chat_template_kwargs"]["enable_thinking"] is thinking


@pytest.mark.asyncio
async def test_oversized_batches_have_bounded_retries_and_retained_provider_evidence(monkeypatch, tmp_path):
    calls = []

    async def upstream(self, request, body):
        calls.append(body)
        return NeMoGymChatCompletion.model_validate(
            {
                "id": "oversized-response",
                "object": "chat.completion",
                "created": 1,
                "model": "super",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{i}",
                                    "type": "function",
                                    "function": {
                                        "name": "saas_browser_observe",
                                        "arguments": "{}",
                                    },
                                }
                                for i in range(33)
                            ],
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    monkeypatch.setattr(
        "nemo_gym.config.global_config.get_global_config_dict",
        lambda: {"model_call_capture_dir": str(tmp_path)},
    )
    config = VLLMModelConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="saas_opencode.py",
        name="policy_model",
        base_url="http://127.0.0.1:9/v1",
        api_key="EMPTY",
        model="super",
        return_token_id_information=False,
        uses_reasoning_parser=True,
    )
    adapter = module.SaasOpenCodeModel(config=config, server_client=MagicMock(spec=ServerClient))
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "super",
            "messages": [{"role": "user", "content": "Use tools"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "saas_browser_observe",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    for status in [422, 422]:
        with pytest.raises(HTTPException) as error:
            await adapter.chat_completions(None, body)
        assert error.value.status_code == status
    assert len(calls) == 3
    records = [
        json.loads(line) for path in tmp_path.glob("saas_protocol_*.jsonl") for line in path.read_text().splitlines()
    ]
    assert len(records) == 3
    assert [record["provider_attempt"] for record in records] == [1, 2, 3]
    assert all(len(r["provider_response"]["choices"][0]["message"]["tool_calls"]) == 33 for r in records)


def mixed_plan():
    return {
        "evaluation_previous_goal": "Initial observation; no prior action.",
        "memory": "The login page is open and the task is unfinished.",
        "next_goal": "Enter the access code.",
        "current_plan_item": 0,
        "plan_update": ["Log in", "Create the requested document", "Verify the saved result"],
    }


def test_mixed_planning_preserves_tool_arguments_reasoning_and_next_turn_history():
    from saas_bench.opencode.protocol import native_message, requires_mixed_planning

    planning = json.dumps(mixed_plan())
    tools = [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "saas_browser_input",
                "arguments": '{"index":4,"text":"seed","observation_id":"00004"}',
            },
        }
    ]
    message = native_message(
        {"role": "assistant", "content": "<think>Provider reasoning</think>" + planning, "tool_calls": tools},
        tools_available=True,
        finish_reason="tool_calls",
        planning_required=True,
    )
    assert message["content"] == planning
    assert message["tool_calls"] == tools
    assert message["reasoning_content"] == "Provider reasoning"
    history = [
        {"role": "system", "content": "<mixed_planning_output>Use planning JSON.</mixed_planning_output>"},
        {"role": "user", "content": "Create a document."},
        message,
        {"role": "tool", "tool_call_id": "call-1", "content": "Input succeeded; fresh observation 00005"},
    ]
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "super",
            "messages": history,
            "tools": [
                {"type": "function", "function": {"name": "saas_browser_input", "parameters": {"type": "object"}}}
            ],
        }
    )
    actual = body.model_dump(exclude_unset=True)
    assert json.loads(actual["messages"][2]["content"]) == mixed_plan()
    assert actual["messages"][2]["tool_calls"] == tools
    assert requires_mixed_planning(actual["messages"], tools_available=True)
    assert not requires_mixed_planning(actual["messages"], tools_available=False)
    assert not requires_mixed_planning([{"role": "user", "content": "<mixed_planning_output>"}], tools_available=True)
    assert not requires_mixed_planning([{"role": "system", "content": "Standard OpenCode"}], tools_available=True)


@pytest.mark.parametrize(
    "change",
    [
        {"memory": ""},
        {"current_plan_item": True},
        {"current_plan_item": -1},
        {"current_plan_item": 1.5},
        {"plan_update": "Log in"},
        {"plan_update": [None]},
        {"thinking": "extra"},
        {"think": "extra"},
        {"action": [{"click": {"index": 9}}]},
    ],
)
def test_invalid_mixed_planning_is_not_an_executable_model_message(change):
    from saas_bench.opencode.protocol import ModelProtocolError, native_message

    with pytest.raises(ModelProtocolError):
        native_message(
            {
                "role": "assistant",
                "content": json.dumps(mixed_plan() | change),
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "saas_browser_observe", "arguments": "{}"},
                    }
                ],
            },
            tools_available=True,
            planning_required=True,
            finish_reason="tool_calls",
        )


def test_mixed_final_report_and_output_cap_preserve_native_endings():
    from saas_bench.opencode.protocol import ModelProtocolError, native_message

    value = mixed_plan() | {
        "current_plan_item": None,
        "plan_update": None,
        "final_response": "Document saved and verified.",
    }
    message = {"role": "assistant", "content": json.dumps(value)}
    assert native_message(message, tools_available=True, planning_required=True, finish_reason="stop") == message
    with pytest.raises(ModelProtocolError, match="final_response"):
        native_message(
            {"role": "assistant", "content": json.dumps(mixed_plan())},
            tools_available=True,
            planning_required=True,
            finish_reason="stop",
        )
    partial = {"role": "assistant", "content": '{"evaluation_previous_goal":'}
    assert native_message(partial, tools_available=True, planning_required=True, finish_reason="length") == partial
    assert native_message({"role": "assistant", "content": "Standard answer"}, tools_available=True) == {
        "role": "assistant",
        "content": "Standard answer",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_mixed_planning_has_bounded_rejection_receipts_before_returning_any_action(
    monkeypatch, tmp_path, recover
):
    from saas_bench.opencode.prompts import BROWSER_USE_SYSTEM_PROMPT

    attempts = []

    async def upstream(self, request, body):
        attempts.append(body.model_dump(exclude_unset=True))
        return NeMoGymChatCompletion.model_validate(
            {
                "id": "mixed-response",
                "object": "chat.completion",
                "created": 1,
                "model": "super",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(mixed_plan()) if recover and len(attempts) == 2 else None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "saas_browser_input",
                                        "arguments": '{"index":4,"text":"seed","observation_id":"00004"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    monkeypatch.setattr(
        "nemo_gym.config.global_config.get_global_config_dict", lambda: {"model_call_capture_dir": str(tmp_path)}
    )
    config = VLLMModelConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="saas_opencode.py",
        name="policy_model",
        base_url="http://127.0.0.1:9/v1",
        api_key="EMPTY",
        model="super",
        return_token_id_information=False,
        uses_reasoning_parser=True,
    )
    adapter = module.SaasOpenCodeModel(config=config, server_client=MagicMock(spec=ServerClient))
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "super",
            "messages": [
                {"role": "system", "content": BROWSER_USE_SYSTEM_PROMPT},
                {"role": "user", "content": "Create a document"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "saas_browser_input", "parameters": {"type": "object"}}}
            ],
        }
    )
    if recover:
        response = await adapter.chat_completions(None, body)
        returned = response.model_dump()["choices"][0]["message"]
        assert json.loads(returned["content"]) == mixed_plan()
        assert (
            returned["tool_calls"][0]["function"]["arguments"] == '{"index":4,"text":"seed","observation_id":"00004"}'
        )
        assert len(attempts) == 2
    else:
        for _ in range(2):
            with pytest.raises(HTTPException) as error:
                await adapter.chat_completions(None, body)
            assert error.value.status_code == 422
        assert len(attempts) == 3
    records = [
        json.loads(line) for path in tmp_path.glob("saas_protocol_*.jsonl") for line in path.read_text().splitlines()
    ]
    assert len(records) == (1 if recover else 3)
    assert all(record["provider_response"]["choices"][0]["message"]["content"] is None for record in records)
    assert all(attempt == body.model_dump(exclude_unset=True) for attempt in attempts)
