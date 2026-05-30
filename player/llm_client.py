"""LLM client module for ChatNVIDIA integration.

Provides functions to initialize a ChatNVIDIA client with structured output
bound to ActionModel, and to invoke the LLM with system prompt, game state,
and optional behavior override. Includes a hard timeout mechanism.
"""

import concurrent.futures
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import ActionModel, LLM_TIMEOUT


# Type alias for the structured LLM client
StructuredLLM = Any


def create_llm_client(model: str = "meta/llama-3.1-8b-instruct") -> StructuredLLM:
    """Initialize ChatNVIDIA with structured output bound to ActionModel.

    Creates a ChatNVIDIA instance and binds ActionModel via
    `.with_structured_output()` so that all invocations return
    a validated ActionModel instance.

    Args:
        model: The NVIDIA NIM model identifier to use.

    Returns:
        A structured LLM client that returns ActionModel on invocation.
    """
    llm = ChatNVIDIA(model=model)
    structured_llm = llm.with_structured_output(ActionModel)
    return structured_llm


def assemble_messages(
    system_prompt: str,
    game_state_json: str,
    behavior_override: str = "",
) -> list:
    """Assemble the message list for LLM invocation.

    Constructs a system message from the system_prompt and a user message
    from the game_state_json. If behavior_override is non-empty, it is
    appended to the user message separated by a newline.

    Args:
        system_prompt: The system prompt instructing the LLM on strategy.
        game_state_json: The current game state as a JSON string.
        behavior_override: Optional tactical override text to append.

    Returns:
        A list of LangChain message objects [SystemMessage, HumanMessage].
    """
    user_content = game_state_json
    if behavior_override:
        user_content = game_state_json + "\n" + behavior_override

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    return messages


def invoke_llm(
    client: StructuredLLM,
    system_prompt: str,
    game_state_json: str,
    behavior_override: str = "",
    timeout: float = LLM_TIMEOUT,
) -> ActionModel:
    """Invoke the LLM and return an ActionModel. Raises on timeout/error.

    Assembles messages from the system prompt, game state JSON, and optional
    behavior override, then invokes the structured LLM client. Enforces a
    hard timeout using a thread pool executor.

    Args:
        client: The structured LLM client (from create_llm_client).
        system_prompt: The system prompt for the LLM.
        game_state_json: The current game state as a JSON string.
        behavior_override: Optional behavior override text.
        timeout: Maximum seconds to wait for LLM response (default 10s).

    Returns:
        An ActionModel instance with the LLM's decision.

    Raises:
        TimeoutError: If the LLM does not respond within the timeout.
        Exception: Any exception raised during LLM invocation is propagated.
    """
    messages = assemble_messages(system_prompt, game_state_json, behavior_override)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.invoke, messages)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"LLM invocation timed out after {timeout} seconds"
            )

    if result is None:
        raise ValueError("LLM returned None response")

    return result
