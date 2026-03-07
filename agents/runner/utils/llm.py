"""LLM utilities for agents using LiteLLM."""

from typing import Any

import litellm
from litellm import acompletion, aresponses
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.files.main import ModelResponse
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from runner.agents.models import LitellmAnyMessage
from runner.utils.decorators import with_retry
from runner.utils.settings import get_settings

settings = get_settings()

# Configure LiteLLM proxy routing if configured
if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
    litellm.use_litellm_proxy = True


def _is_context_window_error(e: Exception) -> bool:
    """
    Detect context window exceeded errors that LiteLLM doesn't properly classify.

    Some providers (notably Gemini) return context window errors as BadRequestError
    instead of ContextWindowExceededError. This predicate catches those cases
    by checking the error message content.

    Known error patterns:
    - Gemini: "input token count exceeds the maximum number of tokens allowed"
    - OpenAI: "context_length_exceeded" (usually caught as ContextWindowExceededError)
    - Anthropic: "prompt is too long" (usually caught as ContextWindowExceededError)
    """
    error_str = str(e).lower()

    # Common patterns indicating context/token limit exceeded
    context_patterns = [
        "token count exceeds",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum number of tokens",
        "prompt is too long",
        "input too long",
        "exceeds the model's maximum context",
    ]

    return any(pattern in error_str for pattern in context_patterns)


def _is_non_retriable_bad_request(e: Exception) -> bool:
    """
    Detect BadRequestErrors that are deterministic and should NOT be retried.

    These are configuration/validation errors that will always fail regardless
    of retry attempts. Retrying wastes time and resources.

    Note: Patterns must be specific enough to avoid matching transient errors
    like rate limits (e.g., "maximum of 100 requests" should NOT match).
    """
    error_str = str(e).lower()

    non_retriable_patterns = [
        # Tool count errors - be specific to avoid matching rate limits
        "tools are supported",  # "Maximum of 128 tools are supported"
        "too many tools",
        # Model/auth errors
        "model not found",
        "does not exist",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "unsupported parameter",
        "unsupported value",
        "unknown parameter",
    ]

    return any(pattern in error_str for pattern in non_retriable_patterns)


def _should_skip_retry(e: Exception) -> bool:
    """Combined check for all non-retriable errors."""
    return _is_context_window_error(e) or _is_non_retriable_bad_request(e)


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def generate_response(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[ChatCompletionToolParam],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> ModelResponse:
    """
    Generate a response from the LLM with retry logic.

    Args:
        model: The model identifier to use
        messages: The conversation messages (input AllMessageValues or output Message)
        tools: Available tools for the model to call
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments to pass to the completion call
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The model response
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "timeout": llm_response_timeout,
        **extra_args,
    }

    # If LiteLLM proxy is configured, add tracking tags
    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        tags = ["service:trajectory"]
        if trajectory_id:
            tags.append(f"trajectory_id:{trajectory_id}")
        kwargs["extra_body"] = {"tags": tags}

    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        stream_iter: Any = await acompletion(**kwargs)
        chunks: list[ModelResponse] = []
        async for chunk in stream_iter:
            chunks.append(chunk)
        rebuilt = litellm.stream_chunk_builder(chunks, messages=messages)
        if rebuilt is None:
            raise RuntimeError("stream_chunk_builder returned None — empty stream")
        return ModelResponse.model_validate(rebuilt)

    response = await acompletion(**kwargs)
    return ModelResponse.model_validate(response)


def _chat_messages_to_responses_input(
    messages: list[LitellmAnyMessage],
) -> list[dict[str, Any]]:
    """Convert chat completion messages to Responses API input format.

    Chat completions format:
      - {"role": "system/user", "content": "..."}
      - {"role": "assistant", "content": "...", "tool_calls": [...]}
      - {"role": "tool", "tool_call_id": "...", "content": "..."}

    Responses API format:
      - {"type": "message", "role": "system/user/developer", "content": "..."}
      - {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}
      - {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}
      - {"type": "function_call_output", "call_id": "...", "output": "..."}
    """
    from runner.agents.models import LitellmOutputMessage

    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, LitellmOutputMessage):
            role = msg.role
            content = msg.content
            tool_calls = getattr(msg, "tool_calls", None)
            tool_call_id = getattr(msg, "tool_call_id", None)
            name = getattr(msg, "name", None)
        else:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")

        if role in ("system", "user", "developer"):
            item: dict[str, Any] = {"type": "message", "role": role, "content": content or ""}
            result.append(item)

        elif role == "assistant":
            # Add text content as a message if present
            if content:
                result.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
            # Add tool calls as separate function_call items
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        result.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", ""),
                        })
                    else:
                        result.append({
                            "type": "function_call",
                            "call_id": getattr(tc, "id", ""),
                            "name": getattr(tc.function, "name", ""),
                            "arguments": getattr(tc.function, "arguments", ""),
                        })

        elif role == "tool":
            call_id = tool_call_id or ""
            output = content if isinstance(content, str) else str(content) if content else ""
            result.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })

    return result


def _chat_tools_to_responses_tools(
    tools: list[ChatCompletionToolParam],
) -> list[dict[str, Any]]:
    """Convert chat completion tool format to Responses API function tool format."""
    result = []
    for tool in tools:
        func = tool.get("function", {})
        result.append({
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        })
    return result


def _responses_api_to_model_response(response: Any) -> ModelResponse:
    """Convert a Responses API response to a ModelResponse for agent compatibility."""
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_content: str | None = None

    for item in getattr(response, "output", []):
        item_type = getattr(item, "type", None)

        if item_type == "message":
            for block in getattr(item, "content", []):
                if getattr(block, "type", None) == "output_text":
                    content_parts.append(getattr(block, "text", ""))

        elif item_type == "function_call":
            tool_calls.append({
                "id": getattr(item, "call_id", ""),
                "type": "function",
                "function": {
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", ""),
                },
            })

        elif item_type == "reasoning":
            summaries = getattr(item, "summary", [])
            if summaries:
                reasoning_content = "\n".join(
                    getattr(s, "text", "") for s in summaries
                )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(content_parts) if content_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": getattr(usage_obj, "input_tokens", 0) if usage_obj else 0,
        "completion_tokens": getattr(usage_obj, "output_tokens", 0) if usage_obj else 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
    }

    return ModelResponse.model_validate({
        "id": getattr(response, "id", ""),
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "model": getattr(response, "model", ""),
        "usage": usage,
    })


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def generate_response_via_responses_api(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[ChatCompletionToolParam],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> ModelResponse:
    """Generate a response using the Responses API, returning a ModelResponse for compatibility."""
    responses_tools = _chat_tools_to_responses_tools(tools)

    # Separate responses-api-specific args from extra_args
    responses_extra = {k: v for k, v in extra_args.items() if k != "use_responses_api"}

    responses_input = _chat_messages_to_responses_input(messages)

    kwargs: dict[str, Any] = {
        "model": model,
        "input": responses_input,
        "tools": responses_tools,
        "timeout": llm_response_timeout,
        **responses_extra,
    }

    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        kwargs["api_base"] = settings.LITELLM_PROXY_API_BASE
        kwargs["api_key"] = settings.LITELLM_PROXY_API_KEY
        tags = ["service:trajectory"]
        if trajectory_id:
            tags.append(f"trajectory_id:{trajectory_id}")
        kwargs["extra_body"] = {"tags": tags}

    response = await aresponses(**kwargs)
    return _responses_api_to_model_response(response)


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def call_responses_api(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[dict[str, Any]],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> Any:
    """
    Generate a response using a provider's Responses API (e.g., web search) with retry logic.

    Uses litellm.aresponses() which is the native async version.

    Args:
        model: The model identifier to use (e.g., 'openai/gpt-4o')
        messages: The conversation messages
        tools: Tools for web search (e.g., [{"type": "web_search"}])
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments (reasoning, etc.)
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The OpenAI responses API response object
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "input": messages,
        "tools": tools,
        "timeout": llm_response_timeout,
        **extra_args,
    }

    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        kwargs["api_base"] = settings.LITELLM_PROXY_API_BASE
        kwargs["api_key"] = settings.LITELLM_PROXY_API_KEY
        tags = ["service:trajectory"]
        if trajectory_id:
            tags.append(f"trajectory_id:{trajectory_id}")
        kwargs["extra_body"] = {"tags": tags}

    if stream:
        kwargs["stream"] = True
        stream_iter: Any = await aresponses(**kwargs)
        completed_response = None
        async for event in stream_iter:
            if getattr(event, "type", None) == "response.completed":
                completed_response = getattr(event, "response", None)
        if completed_response is None:
            raise RuntimeError(
                "No response.completed event received from Responses API stream"
            )
        return completed_response

    response = await aresponses(**kwargs)
    return response
