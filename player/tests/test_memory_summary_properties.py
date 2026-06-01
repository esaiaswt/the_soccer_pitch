"""Property-based tests for the Memory Summarizer module (Properties 3, 4, and 5).

# Feature: full-agentic-upgrade, Property 3: Memory summary episode count limit
# Feature: full-agentic-upgrade, Property 4: Memory summary line format
# Feature: full-agentic-upgrade, Property 5: Memory summary truncation preserves most recent
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from episodic_memory import Episode, EpisodicMemory
from memory_summary import summarize_memory, _format_episode


# --- Strategies ---

action_strategy = st.fixed_dictionaries(
    {
        "dx": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "dy": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "kick": st.booleans(),
    }
)

game_state_strategy = st.fixed_dictionaries(
    {
        "ball_x": st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
        "ball_y": st.floats(min_value=-350, max_value=350, allow_nan=False, allow_infinity=False),
        "player_x": st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
        "player_y": st.floats(min_value=-350, max_value=350, allow_nan=False, allow_infinity=False),
    }
)

next_state_delta_strategy = st.fixed_dictionaries(
    {
        "ball_dx": st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        "ball_dy": st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        "possession_changed": st.booleans(),
    }
)

effectiveness_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


def episode_strategy(cycle: int | None = None):
    """Generate a valid Episode with an optional fixed cycle number."""
    cycle_st = st.just(cycle) if cycle is not None else st.integers(min_value=0, max_value=10000)
    return st.builds(
        Episode,
        cycle=cycle_st,
        game_state=game_state_strategy,
        action=action_strategy,
        next_state_delta=next_state_delta_strategy,
        effectiveness=effectiveness_strategy,
    )


# Feature: full-agentic-upgrade, Property 3: Memory summary episode count limit
class TestMemorySummaryEpisodeCountLimit:
    """Property 3: Memory summary episode count limit.

    For any EpisodicMemory containing any number of episodes (0 to max_capacity),
    the Memory_Summary SHALL contain at most 5 episode lines.

    **Validates: Requirements 2.1**
    """

    @given(
        num_episodes=st.integers(min_value=0, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_summary_contains_at_most_5_lines(self, num_episodes, data):
        """Memory summary contains at most 5 episode lines regardless of memory size."""
        memory = EpisodicMemory(max_capacity=100)

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)

        summary = summarize_memory(memory)

        if num_episodes == 0:
            assert summary == "", f"Expected empty summary for empty memory, got: {summary!r}"
        else:
            lines = summary.split("\n")
            assert len(lines) <= 5, (
                f"Summary has {len(lines)} lines but should have at most 5. "
                f"Memory had {num_episodes} episodes."
            )

    @given(
        num_episodes=st.integers(min_value=6, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_summary_limited_even_with_many_episodes(self, num_episodes, data):
        """Even with many episodes in memory, summary never exceeds 5 lines."""
        memory = EpisodicMemory(max_capacity=100)

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)

        summary = summarize_memory(memory)
        lines = summary.split("\n")

        assert len(lines) <= 5, (
            f"Summary has {len(lines)} lines with {num_episodes} episodes in memory."
        )


# Feature: full-agentic-upgrade, Property 4: Memory summary line format
class TestMemorySummaryLineFormat:
    """Property 4: Memory summary line format.

    For any episode in the EpisodicMemory, its formatted summary line SHALL
    contain the cycle number, the action taken, and an outcome classification
    that is one of "positive", "neutral", or "negative".

    **Validates: Requirements 2.2**
    """

    @given(episode=episode_strategy())
    @settings(max_examples=100)
    def test_formatted_line_contains_cycle_number(self, episode):
        """Each formatted episode line contains the cycle number."""
        line = _format_episode(episode)
        assert f"Cycle {episode.cycle}" in line, (
            f"Line does not contain cycle number {episode.cycle}: {line!r}"
        )

    @given(episode=episode_strategy())
    @settings(max_examples=100)
    def test_formatted_line_contains_action_verb(self, episode):
        """Each formatted episode line contains an action verb derived from the action."""
        line = _format_episode(episode)

        # Determine expected action verb
        if episode.action.get("kick", False):
            expected_verb = "kicked"
        elif episode.action.get("dx", 0) != 0 or episode.action.get("dy", 0) != 0:
            expected_verb = "moved"
        else:
            expected_verb = "waited"

        assert expected_verb in line, (
            f"Line does not contain expected action verb '{expected_verb}': {line!r}"
        )

    @given(episode=episode_strategy())
    @settings(max_examples=100)
    def test_formatted_line_contains_valid_outcome_classification(self, episode):
        """Each formatted episode line contains one of the valid outcome classifications."""
        line = _format_episode(episode)
        valid_outcomes = ["positive", "neutral", "negative"]

        has_outcome = any(outcome in line for outcome in valid_outcomes)
        assert has_outcome, (
            f"Line does not contain a valid outcome classification "
            f"(positive/neutral/negative): {line!r}"
        )

    @given(
        num_episodes=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_all_summary_lines_have_correct_format(self, num_episodes, data):
        """All lines in the summary follow the expected format pattern."""
        memory = EpisodicMemory(max_capacity=100)

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)

        summary = summarize_memory(memory)
        lines = summary.split("\n")

        valid_outcomes = ["positive", "neutral", "negative"]
        valid_verbs = ["kicked", "moved", "waited"]

        for line in lines:
            # Each line should start with "Cycle {n}:"
            assert line.startswith("Cycle "), (
                f"Line does not start with 'Cycle ': {line!r}"
            )
            # Each line should contain an arrow separator
            assert "\u2192" in line, (
                f"Line does not contain arrow separator: {line!r}"
            )
            # Each line should contain a valid action verb
            assert any(verb in line for verb in valid_verbs), (
                f"Line does not contain a valid action verb: {line!r}"
            )
            # Each line should end with a valid outcome
            assert any(line.endswith(outcome) for outcome in valid_outcomes), (
                f"Line does not end with a valid outcome: {line!r}"
            )


# Feature: full-agentic-upgrade, Property 5: Memory summary truncation preserves most recent
class TestMemorySummaryTruncationPreservesMostRecent:
    """Property 5: Memory summary truncation preserves most recent.

    For any EpisodicMemory state, the Memory_Summary SHALL not exceed 500
    characters, and when truncation is necessary, the most recent episode
    SHALL always be preserved in the output while older episodes are removed first.

    **Validates: Requirements 2.3, 2.4**
    """

    @given(
        num_episodes=st.integers(min_value=0, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_summary_does_not_exceed_500_characters(self, num_episodes, data):
        """Memory summary never exceeds 500 characters."""
        memory = EpisodicMemory(max_capacity=100)

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)

        summary = summarize_memory(memory)

        # The summary should not exceed 500 characters
        # Note: a single line for the most recent episode is always preserved
        # even if it alone exceeds 500 chars, per the implementation
        if num_episodes == 0:
            assert summary == ""
        else:
            # When there's more than one episode, truncation removes older ones
            # to stay within limit. A single episode line is always preserved.
            most_recent = memory.get_recent(1)[0]
            single_line = _format_episode(most_recent)
            if len(single_line) <= 500:
                assert len(summary) <= 500, (
                    f"Summary is {len(summary)} chars, exceeds 500 char limit."
                )

    @given(
        num_episodes=st.integers(min_value=1, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_most_recent_episode_always_preserved(self, num_episodes, data):
        """The most recent episode is always preserved in the summary output."""
        memory = EpisodicMemory(max_capacity=100)

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)

        summary = summarize_memory(memory)
        lines = summary.split("\n")

        # The most recent episode should be the last line in the summary
        most_recent = memory.get_recent(1)[0]
        expected_last_line = _format_episode(most_recent)

        assert lines[-1] == expected_last_line, (
            f"Most recent episode not preserved as last line.\n"
            f"Expected: {expected_last_line!r}\n"
            f"Got last line: {lines[-1]!r}"
        )

    @given(
        num_episodes=st.integers(min_value=2, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_truncation_removes_older_episodes_first(self, num_episodes, data):
        """When truncation occurs, older episodes are removed before newer ones."""
        memory = EpisodicMemory(max_capacity=100)

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)

        summary = summarize_memory(memory)
        lines = summary.split("\n")

        # Get the episodes that should be in the summary (most recent 5)
        recent_episodes = memory.get_recent(5)
        recent_lines = [_format_episode(ep) for ep in recent_episodes]

        # The lines in the summary should be a suffix of the recent_lines
        # (older ones removed first means the remaining are the most recent)
        assert len(lines) <= len(recent_lines), (
            f"Summary has {len(lines)} lines but only {len(recent_lines)} recent episodes."
        )

        # Verify the summary lines match the tail of the recent episode lines
        expected_suffix = recent_lines[-len(lines):]
        assert lines == expected_suffix, (
            f"Summary lines don't match the most recent episodes.\n"
            f"Summary: {lines}\n"
            f"Expected (suffix of recent): {expected_suffix}"
        )
