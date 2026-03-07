"""
Tinker CustomLLM adapter for LiteLLM.

Register with register_tinker_provider(), then use model "tinker/<base_model>"
in acompletion calls. Requires TINKER_API_KEY env var.
"""

import json
import uuid
from typing import Any

import litellm
import tinker
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)
from loguru import logger
from tinker.types import SamplingParams
from tinker_cookbook.renderers import (
    Message as TinkerMessage,
    ToolCall as TinkerToolCall,
    get_renderer,
)

_sampling_client: tinker.SamplingClient | None = None
_renderer: Any = None


def _get_client_and_renderer(model: str) -> tuple[tinker.SamplingClient, Any]:
    global _sampling_client, _renderer
    if _sampling_client is None:
        base_model = model.removeprefix("tinker/")
        logger.info(f"Creating Tinker sampling client for {base_model}")
        sc = tinker.ServiceClient()
        _sampling_client = sc.create_sampling_client(base_model=base_model)
        tokenizer = _sampling_client.get_tokenizer()
        _renderer = get_renderer("qwen3_5", tokenizer)
        logger.info("Tinker sampling client ready")
    return _sampling_client, _renderer


def _convert_messages(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
    renderer: Any = None,
) -> list[TinkerMessage]:
    """Convert LiteLLM messages to Tinker renderer format."""
    result: list[TinkerMessage] = []

    # Use the renderer's native tool prefix to inject tool definitions
    # in the format the model was trained on.
    if tools and renderer and hasattr(renderer, "create_conversation_prefix_with_tools"):
        tool_specs = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "parameters": t["function"].get("parameters", {}),
            }
            for t in tools
        ]
        # Extract system prompt from messages so we can merge it
        system_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break
        prefix_msgs = renderer.create_conversation_prefix_with_tools(
            tool_specs, system_prompt=system_prompt,
        )
        result.extend(prefix_msgs)
        # Skip the original system message since we merged it
        messages = [m for m in messages if m.get("role") != "system"]
    elif tools:
        # Fallback: inject tool definitions as JSON
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"].get("parameters", {}),
                },
            }
            for t in tools
        ]
        result.append(TinkerMessage(
            role="system",
            content=(
                "# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                f"<tools>\n{json.dumps(tool_defs, indent=2)}\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "args": {<args-json-object>}}\n'
                "</tool_call>"
            ),
        ))

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""

        if role == "tool":
            result.append(TinkerMessage(
                role="tool", content=content,
                name=msg.get("name", ""), tool_call_id=msg.get("tool_call_id"),
            ))
        elif role == "assistant" and msg.get("tool_calls"):
            tc_list = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args)
                tc_list.append(TinkerToolCall(
                    type="function", id=tc.get("id"),
                    function=TinkerToolCall.FunctionBody(name=fn.get("name", ""), arguments=args),
                ))
            result.append(TinkerMessage(role="assistant", content=content, tool_calls=tc_list))
        else:
            result.append(TinkerMessage(role=role, content=content))

    return result


def _extract_content(raw_content: Any) -> tuple[str | None, str | None]:
    """Extract text content and thinking from renderer output.

    The qwen3_5 renderer may return content as a list of blocks
    (e.g. [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}])
    instead of a plain string.

    Returns (text_content, reasoning_content).
    """
    if raw_content is None:
        return None, None
    if isinstance(raw_content, str):
        return raw_content or None, None

    # List of content blocks
    text_parts = []
    thinking_parts = []
    for block in raw_content:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            else:
                text_parts.append(str(block))
        else:
            text_parts.append(str(block))

    text = "".join(text_parts) or None
    thinking = "".join(thinking_parts) or None
    return text, thinking


def _to_model_response(
    parsed: TinkerMessage, model: str, prompt_tokens: int, completion_tokens: int,
) -> ModelResponse:
    """Convert parsed Tinker response to LiteLLM ModelResponse."""
    content, reasoning_content = _extract_content(parsed.get("content"))
    tool_calls_raw = parsed.get("tool_calls", [])
    finish_reason = "tool_calls" if tool_calls_raw else "stop"

    litellm_tool_calls = None
    if tool_calls_raw:
        litellm_tool_calls = []
        for tc in tool_calls_raw:
            fn = tc.function
            args = fn.arguments if isinstance(fn.arguments, str) else json.dumps(fn.arguments)
            litellm_tool_calls.append(ChatCompletionMessageToolCall(
                id=getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=Function(name=fn.name, arguments=args),
            ))

    return ModelResponse(
        id=f"tinker-{uuid.uuid4().hex[:12]}",
        model=model,
        choices=[Choices(
            finish_reason=finish_reason, index=0,
            message=Message(
                role="assistant",
                content=content,
                tool_calls=litellm_tool_calls,
                reasoning_content=reasoning_content,
            ),
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class TinkerLLM(CustomLLM):
    async def acompletion(self, model, messages, api_base, custom_prompt_dict,
                          model_response, print_verbose, encoding, api_key,
                          logging_obj, optional_params, acompletion=None,
                          litellm_params=None, logger_fn=None, headers={},
                          timeout=None, client=None) -> ModelResponse:
        sampling_client, renderer = _get_client_and_renderer(model)

        tinker_messages = _convert_messages(messages, optional_params.get("tools"), renderer)
        model_input = renderer.build_generation_prompt(tinker_messages)
        prompt_tokens = model_input.length

        params = SamplingParams(
            max_tokens=optional_params.get("max_tokens", 16384),
            temperature=optional_params.get("temperature", 0.6),
            top_p=optional_params.get("top_p", 0.95),
            stop=renderer.get_stop_sequences(),
        )

        response = await sampling_client.sample_async(
            prompt=model_input, sampling_params=params, num_samples=1,
        )

        parsed, success = renderer.parse_response(response.sequences[0].tokens)
        if not success:
            logger.warning("Tinker response parse failed, returning raw content")

        return _to_model_response(
            parsed, model, prompt_tokens, len(response.sequences[0].tokens),
        )


def register_tinker_provider() -> None:
    """Register Tinker as a LiteLLM custom provider. Call once at startup."""
    litellm.custom_provider_map.append(
        {"provider": "tinker", "custom_handler": TinkerLLM()}
    )
    logger.info("Registered Tinker LLM provider with LiteLLM")
