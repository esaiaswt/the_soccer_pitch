"""Property-based tests for Coach Agent integration with player adaptation data.

Uses Hypothesis to verify correctness properties 25 and 26 from the design document.
Tests the CoachAgent._build_adaptation_section() method which aggregates player
adaptation data into the coaching prompt.
"""

from __future__ import annotations

from threading import Event
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from team.coach_agent import CoachAgent
from team.config import TeamConfig
from team.instruction_store import InstructionStore
from team.shared_state import SharedState
from team.strategy_tracker import AdaptationRecord, StrategyTracker


# --- Helper to create a CoachAgent with mocked dependencies ---


def _make_coach_agent(
    player_trackers: dict[str, StrategyTracker] | None = None,
) -> CoachAgent:
    """Create a CoachAgent with mocked LLM and minimal config for testing."""
    config = TeamConfig(
        pitch_host="localhost",
        pitch_port=8000,
        nvidia_api_key="test-key",
        coach_model="test-model",
        player_model="test-model",
        coaching_frequency=7.0,
        poll_interval=1.0,
        streamlit_port=None,
        team_color="Red",
        coach_memory_size=50,
        agent_name="TestBot",
    )
    shared_state = SharedState()
    instruction_store = InstructionStore()
    stop_event = Event()

    with patch("team.coach_agent.ChatNVIDIA"):
        agent = CoachAgent(
            config=config,
            shared_state=shared_state,
            instruction_store=instruction_store,
            stop_event=stop_event,
            player_trackers=player_trackers,
        )

    return agent


# --- Strategies for generating valid inputs ---

PLAYER_POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Striker"]

# Strategy for generating observed_pattern strings (short, realistic)
observed_pattern_strategy = st.sampled_from([
    "opponent_favors_left_flank",
    "opponent_favors_right_flank",
    "opponent_favors_center",
    "opponent_pushes_forward",
    "opponent_sits_back",
    "opponent_high_press",
    "opponent_counter_attack",
    "opponent_wide_play",
])

# Strategy for generating counter_strategy strings
counter_strategy_strategy = st.sampled_from([
    "shift_defensive_coverage_left",
    "shift_defensive_coverage_right",
    "compact_defensive_center",
    "drop_defensive_line_deeper",
    "push_higher_press",
    "hold_possession",
    "play_through_center",
    "use_flanks",
])

# Strategy for generating a single AdaptationRecord
adaptation_record_strategy = st.builds(
    AdaptationRecord,
    observed_pattern=observed_pattern_strategy,
    counter_strategy=counter_strategy_strategy,
    confidence=st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False),
)


def _create_tracker_with_adaptations(adaptations: list[AdaptationRecord]) -> StrategyTracker:
    """Create a StrategyTracker pre-populated with the given AdaptationRecords."""
    tracker = StrategyTracker()
    for adaptation in adaptations:
        tracker._adaptations.append(adaptation)
    return tracker


# Feature: full-agentic-upgrade, Property 25: Coach adaptation summary within token limit
# **Validates: Requirements 11.1, 11.4**


class TestCoachAdaptationSummaryWithinTokenLimit:
    """Property 25: Coach adaptation summary within token limit.

    For any set of player AdaptationRecords (from up to 4 players), the
    Coach_Agent's adaptation summary SHALL contain at most 1 sentence per
    player and SHALL not exceed 200 tokens total.
    """

    @settings(max_examples=100)
    @given(
        num_players=st.integers(min_value=1, max_value=4),
        data=st.data(),
    )
    def test_adaptation_summary_within_200_token_limit(self, num_players, data):
        """The adaptation summary SHALL not exceed 200 tokens total.

        Token estimation: ~4 characters per token, so 200 tokens ≈ 800 characters.

        **Validates: Requirements 11.4**
        """
        positions = PLAYER_POSITIONS[:num_players]
        player_trackers: dict[str, StrategyTracker] = {}

        for position in positions:
            num_adaptations = data.draw(
                st.integers(min_value=1, max_value=2),
                label=f"num_adaptations_{position}",
            )
            adaptations = data.draw(
                st.lists(adaptation_record_strategy, min_size=num_adaptations, max_size=num_adaptations),
                label=f"adaptations_{position}",
            )
            player_trackers[position] = _create_tracker_with_adaptations(adaptations)

        agent = _make_coach_agent(player_trackers=player_trackers)
        section = agent._build_adaptation_section()

        # 200 tokens ≈ 800 characters (using the 4 chars/token estimate from design)
        max_chars = 800
        assert len(section) <= max_chars, (
            f"Adaptation section exceeds 200 token limit ({max_chars} chars). "
            f"Got {len(section)} chars:\n{section}"
        )

    @settings(max_examples=100)
    @given(
        num_players=st.integers(min_value=1, max_value=4),
        data=st.data(),
    )
    def test_adaptation_summary_at_most_one_sentence_per_player(self, num_players, data):
        """The adaptation summary SHALL contain at most 1 sentence per player.

        **Validates: Requirements 11.1**
        """
        positions = PLAYER_POSITIONS[:num_players]
        player_trackers: dict[str, StrategyTracker] = {}

        for position in positions:
            num_adaptations = data.draw(
                st.integers(min_value=1, max_value=2),
                label=f"num_adaptations_{position}",
            )
            adaptations = data.draw(
                st.lists(adaptation_record_strategy, min_size=num_adaptations, max_size=num_adaptations),
                label=f"adaptations_{position}",
            )
            player_trackers[position] = _create_tracker_with_adaptations(adaptations)

        agent = _make_coach_agent(player_trackers=player_trackers)
        section = agent._build_adaptation_section()

        if not section:
            return  # No data, nothing to check

        # Count how many lines reference each player position
        lines = section.split("\n")
        for position in positions:
            # Count lines that start with the player position (the per-player summary lines)
            player_lines = [
                line for line in lines
                if line.startswith(f"{position} reports:")
            ]
            assert len(player_lines) <= 1, (
                f"Expected at most 1 sentence for {position}, "
                f"got {len(player_lines)} lines: {player_lines}"
            )

    @settings(max_examples=100)
    @given(data=st.data())
    def test_empty_trackers_produce_empty_section(self, data):
        """When no player has adaptation data, the section SHALL be empty.

        **Validates: Requirements 11.1**
        """
        num_players = data.draw(st.integers(min_value=1, max_value=4))
        positions = PLAYER_POSITIONS[:num_players]
        player_trackers: dict[str, StrategyTracker] = {}

        for position in positions:
            # Create trackers with no adaptations
            player_trackers[position] = StrategyTracker()

        agent = _make_coach_agent(player_trackers=player_trackers)
        section = agent._build_adaptation_section()

        assert section == "", (
            f"Expected empty section when no adaptations exist, got: '{section}'"
        )

    def test_no_trackers_produces_empty_section(self):
        """When player_trackers is None, the section SHALL be empty.

        **Validates: Requirements 11.1**
        """
        agent = _make_coach_agent(player_trackers=None)
        section = agent._build_adaptation_section()
        assert section == ""

    def test_empty_dict_trackers_produces_empty_section(self):
        """When player_trackers is an empty dict, the section SHALL be empty.

        **Validates: Requirements 11.1**
        """
        agent = _make_coach_agent(player_trackers={})
        section = agent._build_adaptation_section()
        assert section == ""


# Feature: full-agentic-upgrade, Property 26: Coach coordinated instructions on shared tendency
# **Validates: Requirements 11.2**


class TestCoachCoordinatedInstructionsOnSharedTendency:
    """Property 26: Coach coordinated instructions on shared tendency.

    For any scenario where 2 or more Player agents report the same opponent
    tendency (matching observed_pattern), the Coach_Agent SHALL produce
    coordinated tactical instructions that reference that shared tendency.
    """

    @settings(max_examples=100)
    @given(
        shared_pattern=observed_pattern_strategy,
        num_sharing_players=st.integers(min_value=2, max_value=4),
        data=st.data(),
    )
    def test_shared_tendency_produces_coordinated_instructions(
        self, shared_pattern, num_sharing_players, data
    ):
        """When 2+ players report the same pattern, coordinated instructions SHALL
        reference that shared tendency.

        **Validates: Requirements 11.2**
        """
        positions = PLAYER_POSITIONS[:num_sharing_players]
        player_trackers: dict[str, StrategyTracker] = {}

        # Give each player the same observed_pattern as their top adaptation
        for position in positions:
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False),
                label=f"confidence_{position}",
            )
            counter = data.draw(counter_strategy_strategy, label=f"counter_{position}")
            adaptation = AdaptationRecord(
                observed_pattern=shared_pattern,
                counter_strategy=counter,
                confidence=confidence,
            )
            player_trackers[position] = _create_tracker_with_adaptations([adaptation])

        agent = _make_coach_agent(player_trackers=player_trackers)
        section = agent._build_adaptation_section()

        # The section should contain coordinated instructions referencing the shared pattern
        assert "COORDINATED" in section, (
            f"Expected 'COORDINATED' keyword in section when {num_sharing_players} "
            f"players share pattern '{shared_pattern}'. Got:\n{section}"
        )
        assert shared_pattern in section, (
            f"Expected shared pattern '{shared_pattern}' to be referenced in "
            f"coordinated instructions. Got:\n{section}"
        )

    @settings(max_examples=100)
    @given(
        shared_pattern=observed_pattern_strategy,
        data=st.data(),
    )
    def test_shared_tendency_references_reporting_positions(
        self, shared_pattern, data
    ):
        """Coordinated instructions SHALL reference the positions that report
        the shared tendency.

        **Validates: Requirements 11.2**
        """
        # Use exactly 2 players for clear position verification
        num_sharing = data.draw(st.integers(min_value=2, max_value=4), label="num_sharing")
        positions = PLAYER_POSITIONS[:num_sharing]
        player_trackers: dict[str, StrategyTracker] = {}

        for position in positions:
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False),
                label=f"confidence_{position}",
            )
            adaptation = AdaptationRecord(
                observed_pattern=shared_pattern,
                counter_strategy="shift_defensive_coverage_left",
                confidence=confidence,
            )
            player_trackers[position] = _create_tracker_with_adaptations([adaptation])

        agent = _make_coach_agent(player_trackers=player_trackers)
        section = agent._build_adaptation_section()

        # Find the coordinated instruction line(s)
        coordinated_lines = [
            line for line in section.split("\n")
            if "COORDINATED" in line
        ]
        assert len(coordinated_lines) >= 1, (
            f"Expected at least 1 coordinated instruction line. Got:\n{section}"
        )

        # At least one coordinated line should reference the positions
        coordinated_text = " ".join(coordinated_lines)
        for position in positions:
            assert position in coordinated_text, (
                f"Expected position '{position}' to be referenced in coordinated "
                f"instructions. Got: {coordinated_text}"
            )

    @settings(max_examples=100)
    @given(data=st.data())
    def test_no_shared_tendency_no_coordinated_instructions(self, data):
        """When no 2 players share the same pattern, no coordinated instructions
        SHALL be produced.

        **Validates: Requirements 11.2**
        """
        # Give each player a unique pattern
        unique_patterns = [
            "opponent_favors_left_flank",
            "opponent_favors_right_flank",
            "opponent_favors_center",
            "opponent_pushes_forward",
        ]
        num_players = data.draw(st.integers(min_value=2, max_value=4), label="num_players")
        positions = PLAYER_POSITIONS[:num_players]
        player_trackers: dict[str, StrategyTracker] = {}

        for i, position in enumerate(positions):
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False),
                label=f"confidence_{position}",
            )
            adaptation = AdaptationRecord(
                observed_pattern=unique_patterns[i],
                counter_strategy=f"counter_{i}",
                confidence=confidence,
            )
            player_trackers[position] = _create_tracker_with_adaptations([adaptation])

        agent = _make_coach_agent(player_trackers=player_trackers)
        section = agent._build_adaptation_section()

        # No coordinated instructions should appear
        assert "COORDINATED" not in section, (
            f"Expected no 'COORDINATED' keyword when all patterns are unique. "
            f"Got:\n{section}"
        )

    @settings(max_examples=100)
    @given(
        shared_pattern=observed_pattern_strategy,
        extra_pattern=observed_pattern_strategy,
        data=st.data(),
    )
    def test_mixed_shared_and_unique_patterns(self, shared_pattern, extra_pattern, data):
        """When some players share a pattern and others don't, coordinated
        instructions SHALL only reference the shared pattern.

        **Validates: Requirements 11.2**
        """
        # Ensure the extra pattern is different from the shared one
        assume(extra_pattern != shared_pattern)

        player_trackers: dict[str, StrategyTracker] = {}

        # Two players share the same pattern
        for position in ["Defender", "Midfielder"]:
            confidence = data.draw(
                st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False),
                label=f"confidence_{position}",
            )
            adaptation = AdaptationRecord(
                observed_pattern=shared_pattern,
                counter_strategy="shift_defensive_coverage_left",
                confidence=confidence,
            )
            player_trackers[position] = _create_tracker_with_adaptations([adaptation])

        # One player has a unique pattern
        confidence = data.draw(
            st.floats(min_value=0.71, max_value=1.0, allow_nan=False, allow_infinity=False),
            label="confidence_Striker",
        )
        adaptation = AdaptationRecord(
            observed_pattern=extra_pattern,
            counter_strategy="hold_possession",
            confidence=confidence,
        )
        player_trackers["Striker"] = _create_tracker_with_adaptations([adaptation])

        agent = _make_coach_agent(player_trackers=player_trackers)
        section = agent._build_adaptation_section()

        # Should have coordinated instructions for the shared pattern
        assert "COORDINATED" in section, (
            f"Expected 'COORDINATED' for shared pattern '{shared_pattern}'. Got:\n{section}"
        )
        assert shared_pattern in section, (
            f"Expected shared pattern '{shared_pattern}' in section. Got:\n{section}"
        )
