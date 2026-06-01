"""Property-based tests for the team/ Memory Summarizer module.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from team.episodic_memory import Episode, EpisodicMemory
from team.memory_summary import summarize_memory


# --- Strategies for generating valid episodes ---

action_strategy = st.fixed_dictionaries({
    "dx": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "dy": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "kick": st.booleans(),
})

effectiveness_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


def episode_strategy(cycle: int | None = None):
    """Generate a valid Episode with an optional fixed cycle number."""
    cycle_st = st.just(cycle) if cycle is not None else st.integers(min_value=0, max_value=100_000)
    return st.builds(
        Episode,
        cycle=cycle_st,
        game_state=st.just({}),
        action=action_strategy,
        next_state_delta=st.just({}),
        effectiveness=effectiveness_strategy,
    )


def episodes_list_strategy(min_size: int = 0, max_size: int = 200):
    """Generate a list of episodes with strictly increasing cycle numbers."""
    return st.lists(
        st.integers(min_value=0, max_value=100_000),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    ).flatmap(
        lambda cycles: st.tuples(
            *[episode_strategy(c) for c in sorted(cycles)]
        ).map(list)
        if cycles
        else st.just([])
    )


# Feature: full-agentic-upgrade, Property 3: Memory summary episode count limit
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4**


class TestMemorySummaryEpisodeCountLimit:
    """Property 3: Memory summary episode count limit.

    For any EpisodicMemory containing any number of episodes (0 to max_capacity),
    the Memory_Summary SHALL contain at most 5 episode lines.
    """

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=0, max_size=150))
    def test_summary_contains_at_most_5_lines(self, episodes):
        """The Memory_Summary SHALL contain at most 5 episode lines.

        **Validates: Requirements 2.1**
        """
        memory = EpisodicMemory(max_capacity=200)
        for ep in episodes:
            memory.add(ep)

        result = summarize_memory(memory)

        if result == "":
            # Empty memory produces empty summary
            assert len(episodes) == 0 or len(memory) == 0
        else:
            lines = result.strip().split("\n")
            assert len(lines) <= 5, (
                f"Summary has {len(lines)} lines but should have at most 5. "
                f"Memory had {len(memory)} episodes."
            )

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=0, max_size=150))
    def test_empty_memory_produces_empty_summary(self, episodes):
        """An empty EpisodicMemory SHALL produce an empty summary string.

        **Validates: Requirements 2.1**
        """
        memory = EpisodicMemory(max_capacity=200)
        # Don't add any episodes
        result = summarize_memory(memory)
        assert result == ""


# Feature: full-agentic-upgrade, Property 4: Memory summary line format
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4**


class TestMemorySummaryLineFormat:
    """Property 4: Memory summary line format.

    For any episode in the EpisodicMemory, its formatted summary line SHALL
    contain the cycle number, the action taken, and an outcome classification
    that is one of "positive", "neutral", or "negative".
    """

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=5))
    def test_each_line_contains_cycle_number(self, episodes):
        """Each summary line SHALL contain the cycle number.

        **Validates: Requirements 2.2**
        """
        memory = EpisodicMemory(max_capacity=200)
        for ep in episodes:
            memory.add(ep)

        result = summarize_memory(memory)
        lines = result.strip().split("\n")

        # Each line should start with "Cycle {n}:"
        for line in lines:
            assert line.startswith("Cycle "), (
                f"Line does not start with 'Cycle ': {line}"
            )
            # Extract cycle number - should be an integer after "Cycle "
            parts = line.split(":")
            cycle_part = parts[0].replace("Cycle ", "")
            assert cycle_part.strip().isdigit() or cycle_part.strip().lstrip("-").isdigit(), (
                f"Could not extract cycle number from line: {line}"
            )

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=5))
    def test_each_line_contains_action_verb(self, episodes):
        """Each summary line SHALL contain the action taken.

        **Validates: Requirements 2.2**
        """
        memory = EpisodicMemory(max_capacity=200)
        for ep in episodes:
            memory.add(ep)

        result = summarize_memory(memory)
        lines = result.strip().split("\n")

        valid_verbs = {"kicked", "moved", "waited"}
        for line in lines:
            found_verb = any(verb in line for verb in valid_verbs)
            assert found_verb, (
                f"Line does not contain a valid action verb (kicked/moved/waited): {line}"
            )

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=5))
    def test_each_line_contains_outcome_classification(self, episodes):
        """Each summary line SHALL contain an outcome classification that is one of
        "positive", "neutral", or "negative".

        **Validates: Requirements 2.2**
        """
        memory = EpisodicMemory(max_capacity=200)
        for ep in episodes:
            memory.add(ep)

        result = summarize_memory(memory)
        lines = result.strip().split("\n")

        valid_outcomes = {"positive", "neutral", "negative"}
        for line in lines:
            found_outcome = any(outcome in line for outcome in valid_outcomes)
            assert found_outcome, (
                f"Line does not contain a valid outcome classification "
                f"(positive/neutral/negative): {line}"
            )

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=5))
    def test_line_format_matches_expected_pattern(self, episodes):
        """Each summary line SHALL follow the format 'Cycle {n}: {verb} → {outcome}'.

        **Validates: Requirements 2.2**
        """
        memory = EpisodicMemory(max_capacity=200)
        for ep in episodes:
            memory.add(ep)

        result = summarize_memory(memory)
        lines = result.strip().split("\n")

        valid_verbs = {"kicked", "moved", "waited"}
        valid_outcomes = {"positive", "neutral", "negative"}

        for line in lines:
            # Check the arrow separator is present
            assert "\u2192" in line, f"Line missing '\u2192' separator: {line}"

            # Split on arrow
            before_arrow, after_arrow = line.split("\u2192")
            before_arrow = before_arrow.strip()
            after_arrow = after_arrow.strip()

            # Before arrow: "Cycle {n}: {verb}"
            assert before_arrow.startswith("Cycle "), (
                f"Before arrow does not start with 'Cycle ': {before_arrow}"
            )

            # After arrow: outcome classification
            assert after_arrow in valid_outcomes, (
                f"After arrow '{after_arrow}' is not a valid outcome classification"
            )

            # Check verb is present before the arrow
            found_verb = any(verb in before_arrow for verb in valid_verbs)
            assert found_verb, (
                f"No valid verb found before arrow in: {before_arrow}"
            )


# Feature: full-agentic-upgrade, Property 5: Memory summary truncation preserves most recent
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4**


class TestMemorySummaryTruncationPreservesMostRecent:
    """Property 5: Memory summary truncation preserves most recent.

    For any EpisodicMemory state, the Memory_Summary SHALL not exceed 500
    characters, and when truncation is necessary, the most recent episode SHALL
    always be preserved in the output while older episodes are removed first.
    """

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=150))
    def test_summary_does_not_exceed_500_characters(self, episodes):
        """The Memory_Summary SHALL not exceed 500 characters.

        **Validates: Requirements 2.3**
        """
        memory = EpisodicMemory(max_capacity=200)
        for ep in episodes:
            memory.add(ep)

        result = summarize_memory(memory)
        assert len(result) <= 500, (
            f"Summary is {len(result)} characters but should not exceed 500."
        )

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=150))
    def test_most_recent_episode_always_preserved(self, episodes):
        """When truncation is necessary, the most recent episode SHALL always be
        preserved in the output.

        **Validates: Requirements 2.4**
        """
        memory = EpisodicMemory(max_capacity=200)
        for ep in episodes:
            memory.add(ep)

        result = summarize_memory(memory)

        if result == "":
            return

        # The most recent episode is the last one added
        most_recent = episodes[-1]
        most_recent_cycle = most_recent.cycle

        # The summary should contain the most recent episode's cycle number
        lines = result.strip().split("\n")
        last_line = lines[-1]
        assert f"Cycle {most_recent_cycle}:" in last_line, (
            f"Most recent episode (cycle {most_recent_cycle}) not found in "
            f"last line of summary. Last line: '{last_line}'"
        )

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=2, max_size=150))
    def test_older_episodes_removed_first_during_truncation(self, episodes):
        """When truncation is necessary, older episodes SHALL be removed first.

        **Validates: Requirements 2.3, 2.4**
        """
        memory = EpisodicMemory(max_capacity=200)
        for ep in episodes:
            memory.add(ep)

        result = summarize_memory(memory)

        if result == "":
            return

        lines = result.strip().split("\n")

        # Extract cycle numbers from the summary lines
        summary_cycles = []
        for line in lines:
            # Extract cycle number from "Cycle {n}: ..."
            cycle_str = line.split(":")[0].replace("Cycle ", "").strip()
            summary_cycles.append(int(cycle_str))

        # The cycles in the summary should be in ascending order (chronological)
        assert summary_cycles == sorted(summary_cycles), (
            f"Summary cycles are not in chronological order: {summary_cycles}"
        )

        # The most recent episode's cycle should be the last in the summary
        most_recent_cycle = episodes[-1].cycle
        assert summary_cycles[-1] == most_recent_cycle, (
            f"Most recent cycle {most_recent_cycle} is not the last in summary. "
            f"Summary cycles: {summary_cycles}"
        )

        # If fewer lines than available episodes (within the 5-episode limit),
        # the omitted ones should be older
        recent_5 = episodes[-5:]
        recent_5_cycles = [ep.cycle for ep in recent_5]
        for cycle in summary_cycles:
            assert cycle in recent_5_cycles, (
                f"Summary contains cycle {cycle} which is not among the 5 most "
                f"recent episodes: {recent_5_cycles}"
            )
