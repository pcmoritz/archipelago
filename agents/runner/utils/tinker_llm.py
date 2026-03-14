"""
Custom LiteLLM provider that uses tinker-cookbook to render chat messages into
tokens (to measure prompt length), then forwards the request to OpenRouter
via litellm, constraining max_tokens based on the rendered token count.

Requires OPENROUTER_API_KEY env var.

Usage:
    from runner.utils.tinker_llm import register_tinker_provider
    register_tinker_provider()

    response = await litellm.acompletion(
        model="tinker/moonshotai/Kimi-K2.5",
        messages=[{"role": "user", "content": "Hello"}],
        base_model="moonshotai/Kimi-K2.5",
    )
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional, Union

import httpx
import litellm
from litellm.exceptions import ContextWindowExceededError
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse
from tinker_cookbook.renderers import Message as TinkerMessage
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.tokenizer_utils import get_tokenizer

MAX_TOTAL_TOKENS = 65536


@lru_cache(maxsize=4)
def _get_tokenizer(base_model: str):
    return get_tokenizer(base_model)


@lru_cache(maxsize=4)
def _get_renderer(base_model: str, renderer_name: str):
    return get_renderer(renderer_name, _get_tokenizer(base_model))


def _to_tinker_messages(messages: list[dict[str, Any]]) -> list[TinkerMessage]:
    """Convert litellm message dicts to tinker-cookbook Messages."""
    out: list[TinkerMessage] = []
    for msg in messages:
        tinker_msg = TinkerMessage(role=msg["role"], content=msg.get("content") or "")
        if "name" in msg:
            tinker_msg["name"] = msg["name"]
        if "tool_call_id" in msg:
            tinker_msg["tool_call_id"] = msg["tool_call_id"]
        if "tool_calls" in msg:
            allowed = set(ToolCall.model_fields)
            tinker_msg["tool_calls"] = [
                ToolCall.model_validate({k: v for k, v in tc.items() if k in allowed})
                for tc in msg["tool_calls"]
            ]
        out.append(tinker_msg)
    return out


class TinkerCookbookLLM(CustomLLM):
    """Renders messages to tokens to measure prompt length, then delegates to OpenRouter via litellm."""

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ModelResponse:
        base_model: str = litellm_params["base_model"]

        renderer = _get_renderer(base_model, "qwen3_5")

        # Convert messages to tinker format to measure token count
        tinker_msgs = _to_tinker_messages(messages)

        # Inject tool declarations into the prompt
        tools = optional_params.get("tools")
        if tools:
            tool_specs = [t["function"] for t in tools if "function" in t]
            match tinker_msgs:
                case [{"role": "system", "content": system_prompt}, *rest]:
                    pass
                case rest:
                    system_prompt = ""
            tinker_msgs = renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt) + rest

        model_input = renderer.build_generation_prompt(tinker_msgs)
        input_token_count = len(model_input.to_ints())

        # Constrain max_tokens so input + output <= MAX_TOTAL_TOKENS
        max_output_tokens = MAX_TOTAL_TOKENS - input_token_count
        if max_output_tokens <= 0:
            raise ContextWindowExceededError(
                f"Input tokens ({input_token_count}) exceed MAX_TOTAL_TOKENS ({MAX_TOTAL_TOKENS})",
                model=model,
                llm_provider="openrouter",
            )
        optional_params["max_tokens"] = max_output_tokens

        # Forward to OpenRouter via litellm
        # model comes in as the part after "tinker/", e.g. "moonshotai/Kimi-K2.5"
        openrouter_model = f"openrouter/{model}"

        response = await litellm.acompletion(
            model=openrouter_model,
            messages=messages,
            timeout=timeout,
            **optional_params,
        )

        return response


tinker_llm_instance = TinkerCookbookLLM()


def register_tinker_provider() -> None:
    """Register the tinker provider with litellm. Call once at startup."""
    litellm.custom_provider_map.append(
        {"provider": "tinker", "custom_handler": tinker_llm_instance}
    )
