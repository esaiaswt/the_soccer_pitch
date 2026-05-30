"""Property-based tests for LLM message assembly (Property 6)."""

from hypothesis import given, settings
import hypothesis.strategies as st
from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import assemble_messages


# Feature: agent-control-panel, Property 6: LLM message assembly with optional behavior override
class TestMessageAssemblyWithOptionalBehaviorOverride:
    """Property 6: LLM message assembly with optional behavior override.

    For any non-empty system prompt, any game state JSON string, and any
    optional behavior override string, the message assembly function SHALL
    produce a system message containing exactly the system prompt, and a user
    message containing the game state JSON followed by the behavior override
    (if non-empty) separated by a newline. When the override is empty, the
    user message SHALL contain only the game state JSON.

    **Validates: Requirements 4.4, 4.6**
    """

    @given(
        system_prompt=st.text(min_size=1, max_size=200),
        game_state_json=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=100)
    def test_empty_override_produces_user_message_with_only_game_state(
        self, system_prompt: str, game_state_json: str
    ):
        """When override is empty, user message contains only game state JSON."""
        messages = assemble_messages(system_prompt, game_state_json, "")
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == system_prompt
        assert messages[1].content == game_state_json

    @given(
        system_prompt=st.text(min_size=1, max_size=200),
        game_state_json=st.text(min_size=1, max_size=500),
        behavior_override=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=100)
    def test_non_empty_override_appended_with_newline_separator(
        self, system_prompt: str, game_state_json: str, behavior_override: str
    ):
        """When override is non-empty, user message is game state + newline + override."""
        messages = assemble_messages(system_prompt, game_state_json, behavior_override)
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == system_prompt
        expected_user_content = game_state_json + "\n" + behavior_override
        assert messages[1].content == expected_user_content

    @given(
        system_prompt=st.text(min_size=1, max_size=200),
        game_state_json=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=100)
    def test_default_override_parameter_produces_game_state_only(
        self, system_prompt: str, game_state_json: str
    ):
        """When override parameter is omitted (default), user message is game state only."""
        messages = assemble_messages(system_prompt, game_state_json)
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == system_prompt
        assert messages[1].content == game_state_json

    @given(
        system_prompt=st.text(min_size=1, max_size=200),
        game_state_json=st.text(min_size=1, max_size=500),
        behavior_override=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=100)
    def test_system_message_always_contains_exact_system_prompt(
        self, system_prompt: str, game_state_json: str, behavior_override: str
    ):
        """System message always contains exactly the system prompt regardless of override."""
        messages = assemble_messages(system_prompt, game_state_json, behavior_override)
        assert messages[0].content == system_prompt
