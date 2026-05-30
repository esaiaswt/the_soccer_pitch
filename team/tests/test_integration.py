"""Integration tests for end-to-end data flow.

Tests cover:
- State Poller → SharedState → Player Agent flow with mock Pitch server (Req 1.2)
- Coach Agent → InstructionStore → Player Agent instruction delivery (Req 3.1, 4.4)
- Multi-thread concurrent access to SharedState and InstructionStore (Req 8.1)
- Two-instance isolation (no shared mutable state) (Req 8.1)

Requirements: 1.2, 3.1, 4.4, 8.1
"""

from __future__ import annotations

import threading
import time
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from team.config import TeamConfig
from team.debug_store import DebugStore
from team.instruction_store import CoachInstruction, InstructionStore
from team.orchestrator import TeamOrchestrator
from team.player_agent import PlayerAgent
from team.shared_state import SharedState
from team.state_poller import StatePoller


def _make_config(**overrides) -> TeamConfig:
    """Create a TeamConfig with sensible defaults for integration testing."""
    defaults = {
        "pitch_host": "localhost",
        "pitch_port": 8000,
        "nvidia_api_key": "test",
        "coach_model": "m",
        "player_model": "m",
        "coaching_frequency": 2.0,
        "poll_interval": 0.05,
        "streamlit_port": None,
        "team_color": "Red",
        "coach_memory_size": 50, "agent_name": "TeamBot",
    }
    defaults.update(overrides)
    return TeamConfig(**defaults)


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def sample_snapshot():
    """A valid game state snapshot from the Pitch server."""
    return {
        "match_state": "Playing",
        "time_left": 60.0,
        "score": {"Red": 1, "Blue": 0},
        "ball": {"x": 500.0, "y": 300.0},
        "players": {
            "Red_Goalkeeper": {"x": 50.0, "y": 425.0},
            "Red_Defender": {"x": 200.0, "y": 350.0},
            "Red_Midfielder": {"x": 400.0, "y": 400.0},
            "Red_Striker": {"x": 600.0, "y": 300.0},
            "Blue_Goalkeeper": {"x": 1150.0, "y": 425.0},
            "Blue_Defender": {"x": 900.0, "y": 350.0},
            "Blue_Midfielder": {"x": 700.0, "y": 400.0},
            "Blue_Striker": {"x": 500.0, "y": 500.0},
        },
    }


class TestStatePollerToPlayerAgent:
    """Integration: State Poller → SharedState → Player Agent.

    Validates Requirement 1.2: State Poller makes snapshot available to all agents.
    """

    @patch("team.state_poller.requests.get")
    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_poller_updates_shared_state_player_reads_it(
        self, mock_chat, mock_post, mock_get, config, sample_snapshot
    ):
        """State Poller polls Pitch, updates SharedState, Player reads the snapshot."""
        # Set up mock Pitch server response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_snapshot
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Set up mock LLM response
        llm_response = MagicMock()
        llm_response.content = "dx=0.3 dy=-0.1 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = llm_response
        mock_chat.return_value = mock_chat_instance

        # Set up mock action POST
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        # Create shared components
        shared_state = SharedState()
        instruction_store = InstructionStore()
        debug_store = DebugStore()
        stop_event = Event()

        # Start State Poller in a thread
        poller = StatePoller(config=config, shared_state=shared_state, stop_event=stop_event)
        poller_thread = threading.Thread(target=poller.run, daemon=True)
        poller_thread.start()

        # Wait for at least one poll cycle
        time.sleep(0.15)

        # Verify SharedState was updated by the poller
        snapshot = shared_state.get_snapshot()
        assert snapshot is not None
        assert snapshot["match_state"] == "Playing"
        assert snapshot["ball"]["x"] == 500.0

        # Now run a Player Agent iteration using the same SharedState
        player = PlayerAgent(
            config=config,
            position="Striker",
            shared_state=shared_state,
            instruction_store=instruction_store,
            stop_event=stop_event,
            debug_store=debug_store,
        )
        player._loop_iteration()

        # Verify the player invoked LLM with game state context
        mock_chat_instance.invoke.assert_called_once()
        messages = mock_chat_instance.invoke.call_args[0][0]
        human_msg_content = messages[1].content
        assert "x=500" in human_msg_content  # ball x from snapshot
        assert "Playing" in human_msg_content  # match state

        # Verify action was posted to Pitch server
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["team"] == "Red"
        assert payload["position"] == "Striker"
        assert payload["vector"]["dx"] == 0.3
        assert payload["vector"]["dy"] == -0.1

        # Clean up
        stop_event.set()
        poller_thread.join(timeout=1.0)

    @patch("team.state_poller.requests.get")
    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_poller_updates_propagate_to_multiple_players(
        self, mock_chat, mock_post, mock_get, config, sample_snapshot
    ):
        """Multiple Player Agents all read the same snapshot from SharedState."""
        # Set up mock Pitch server
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_snapshot
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Set up mock LLM
        llm_response = MagicMock()
        llm_response.content = "dx=0.1 dy=0.2 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.return_value = llm_response
        mock_chat.return_value = mock_chat_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        # Shared components
        shared_state = SharedState()
        instruction_store = InstructionStore()
        debug_store = DebugStore()
        stop_event = Event()

        # Start poller
        poller = StatePoller(config=config, shared_state=shared_state, stop_event=stop_event)
        poller_thread = threading.Thread(target=poller.run, daemon=True)
        poller_thread.start()
        time.sleep(0.15)

        # Create multiple players and run one iteration each
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]
        for position in positions:
            player = PlayerAgent(
                config=config,
                position=position,
                shared_state=shared_state,
                instruction_store=instruction_store,
                stop_event=stop_event,
                debug_store=debug_store,
            )
            player._loop_iteration()

        # All 4 players should have posted actions
        assert mock_post.call_count == 4

        # Each player should have read the same snapshot
        for call in mock_post.call_args_list:
            payload = call[1]["json"]
            assert payload["team"] == "Red"
            assert payload["position"] in positions

        # Clean up
        stop_event.set()
        poller_thread.join(timeout=1.0)


class TestCoachToPlayerInstructionDelivery:
    """Integration: Coach Agent → InstructionStore → Player Agent.

    Validates Requirements 3.1 and 4.4:
    - Coach issues instructions to each player per coaching frequency
    - Player includes Coach instruction in LLM context
    """

    @patch("team.coach_agent.ChatNVIDIA")
    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_coach_instructions_reach_player_context(
        self, mock_player_chat, mock_post, mock_coach_chat, config, sample_snapshot
    ):
        """Coach generates instructions, Player includes them in LLM messages."""
        from team.coach_agent import CoachAgent

        # Set up mock Coach LLM response
        coach_response = MagicMock()
        coach_response.content = (
            "Goalkeeper: Stay on your line and watch the ball\n"
            "Defender: Mark the opposing striker closely\n"
            "Midfielder: Control the midfield and distribute\n"
            "Striker: Make runs behind the defense"
        )
        coach_response.usage_metadata = None
        coach_response.response_metadata = None
        mock_coach_instance = MagicMock()
        mock_coach_instance.invoke.return_value = coach_response
        mock_coach_chat.return_value = mock_coach_instance

        # Set up mock Player LLM response
        player_response = MagicMock()
        player_response.content = "dx=0.5 dy=0.0 kick=false"
        player_response.usage_metadata = None
        player_response.response_metadata = None
        mock_player_instance = MagicMock()
        mock_player_instance.invoke.return_value = player_response
        mock_player_chat.return_value = mock_player_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        # Shared components
        shared_state = SharedState()
        instruction_store = InstructionStore()
        debug_store = DebugStore()
        stop_event = Event()

        # Pre-populate shared state (simulating State Poller)
        shared_state.set_snapshot(sample_snapshot)

        # Create and run one coaching cycle
        coach = CoachAgent(
            config=config,
            shared_state=shared_state,
            instruction_store=instruction_store,
            stop_event=stop_event,
            debug_store=debug_store,
        )
        coach._coaching_cycle()

        # Verify instructions were stored for all positions
        for position in ["Goalkeeper", "Defender", "Midfielder", "Striker"]:
            instr = instruction_store.get_instruction(position)
            assert instr is not None, f"No instruction for {position}"
            assert len(instr.content) > 0

        # Now create a Player Agent and verify it includes the instruction
        player = PlayerAgent(
            config=config,
            position="Striker",
            shared_state=shared_state,
            instruction_store=instruction_store,
            stop_event=stop_event,
            debug_store=debug_store,
        )
        player._loop_iteration()

        # Verify the player's LLM was called with coach instruction in context
        mock_player_instance.invoke.assert_called_once()
        messages = mock_player_instance.invoke.call_args[0][0]
        human_msg_content = messages[1].content
        assert "Coach Advisory" in human_msg_content
        assert "Make runs behind the defense" in human_msg_content

    @patch("team.coach_agent.ChatNVIDIA")
    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_coach_instruction_updates_overwrite_previous(
        self, mock_player_chat, mock_post, mock_coach_chat, config, sample_snapshot
    ):
        """New coach instructions overwrite old ones in InstructionStore."""
        from team.coach_agent import CoachAgent

        # First coaching cycle response
        first_response = MagicMock()
        first_response.content = (
            "Goalkeeper: Stay back\n"
            "Defender: Hold the line\n"
            "Midfielder: Pass forward\n"
            "Striker: Wait for the ball"
        )
        first_response.usage_metadata = None
        first_response.response_metadata = None

        # Second coaching cycle response
        second_response = MagicMock()
        second_response.content = (
            "Goalkeeper: Come out for crosses\n"
            "Defender: Push up high\n"
            "Midfielder: Press aggressively\n"
            "Striker: Drop deep to receive"
        )
        second_response.usage_metadata = None
        second_response.response_metadata = None

        mock_coach_instance = MagicMock()
        mock_coach_instance.invoke.side_effect = [first_response, second_response]
        mock_coach_chat.return_value = mock_coach_instance

        # Player LLM setup
        player_response = MagicMock()
        player_response.content = "dx=0.0 dy=0.5 kick=false"
        player_response.usage_metadata = None
        player_response.response_metadata = None
        mock_player_instance = MagicMock()
        mock_player_instance.invoke.return_value = player_response
        mock_player_chat.return_value = mock_player_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        # Shared components
        shared_state = SharedState()
        instruction_store = InstructionStore()
        debug_store = DebugStore()
        stop_event = Event()
        shared_state.set_snapshot(sample_snapshot)

        # Run two coaching cycles
        coach = CoachAgent(
            config=config,
            shared_state=shared_state,
            instruction_store=instruction_store,
            stop_event=stop_event,
            debug_store=debug_store,
        )
        coach._coaching_cycle()
        coach._coaching_cycle()

        # Verify the latest instruction is from the second cycle
        striker_instr = instruction_store.get_instruction("Striker")
        assert striker_instr is not None
        assert "Drop deep to receive" in striker_instr.content

        # Player should see the updated instruction
        player = PlayerAgent(
            config=config,
            position="Striker",
            shared_state=shared_state,
            instruction_store=instruction_store,
            stop_event=stop_event,
            debug_store=debug_store,
        )
        player._loop_iteration()

        messages = mock_player_instance.invoke.call_args[0][0]
        human_msg_content = messages[1].content
        assert "Drop deep to receive" in human_msg_content


class TestMultiThreadConcurrentAccess:
    """Integration: Multi-thread concurrent access to SharedState and InstructionStore.

    Validates Requirement 8.1: No shared mutable state corruption under concurrency.
    """

    def test_concurrent_shared_state_reads_and_writes(self, config, sample_snapshot):
        """Multiple threads reading/writing SharedState simultaneously without corruption."""
        shared_state = SharedState()
        errors: list[str] = []
        iterations = 100

        def writer():
            """Write snapshots with incrementing time_left values."""
            for i in range(iterations):
                snapshot = {**sample_snapshot, "time_left": float(i)}
                shared_state.set_snapshot(snapshot)
                time.sleep(0.001)

        def reader():
            """Read snapshots and verify they are internally consistent."""
            for _ in range(iterations):
                snapshot = shared_state.get_snapshot()
                if snapshot is not None:
                    # Verify the snapshot is a valid dict (not corrupted)
                    if not isinstance(snapshot, dict):
                        errors.append(f"Snapshot is not a dict: {type(snapshot)}")
                    if "time_left" not in snapshot:
                        errors.append("Snapshot missing time_left field")
                    if "ball" not in snapshot:
                        errors.append("Snapshot missing ball field")
                time.sleep(0.001)

        # Start one writer and multiple readers
        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(5)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join(timeout=5.0)
        for t in reader_threads:
            t.join(timeout=5.0)

        assert errors == [], f"Concurrent access errors: {errors}"

        # Final snapshot should be the last one written
        final = shared_state.get_snapshot()
        assert final is not None
        assert final["time_left"] == float(iterations - 1)

    def test_concurrent_instruction_store_reads_and_writes(self, config):
        """Multiple threads reading/writing InstructionStore simultaneously."""
        instruction_store = InstructionStore()
        errors: list[str] = []
        iterations = 100
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]

        def writer(position: str):
            """Write instructions for a specific position."""
            for i in range(iterations):
                instr = CoachInstruction(
                    content=f"Instruction {i} for {position}",
                    timestamp=time.time(),
                    target_position=position,
                )
                instruction_store.set_instruction(position, instr)
                time.sleep(0.001)

        def reader():
            """Read instructions and verify they are valid."""
            for _ in range(iterations):
                for position in positions:
                    instr = instruction_store.get_instruction(position)
                    if instr is not None:
                        if not isinstance(instr.content, str):
                            errors.append(f"Instruction content is not str: {type(instr.content)}")
                        if instr.target_position != position:
                            errors.append(
                                f"Position mismatch: expected {position}, "
                                f"got {instr.target_position}"
                            )
                time.sleep(0.001)

        # Start one writer per position and multiple readers
        writer_threads = [threading.Thread(target=writer, args=(pos,)) for pos in positions]
        reader_threads = [threading.Thread(target=reader) for _ in range(4)]

        for t in writer_threads + reader_threads:
            t.start()

        for t in writer_threads + reader_threads:
            t.join(timeout=10.0)

        assert errors == [], f"Concurrent access errors: {errors}"

        # Each position should have the last instruction written
        for position in positions:
            instr = instruction_store.get_instruction(position)
            assert instr is not None
            assert instr.target_position == position

    def test_concurrent_shared_state_and_instruction_store_together(
        self, config, sample_snapshot
    ):
        """SharedState and InstructionStore accessed concurrently by mixed threads."""
        shared_state = SharedState()
        instruction_store = InstructionStore()
        errors: list[str] = []
        iterations = 50

        def state_writer():
            for i in range(iterations):
                snapshot = {**sample_snapshot, "time_left": float(90 - i)}
                shared_state.set_snapshot(snapshot)
                time.sleep(0.002)

        def instruction_writer():
            positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]
            for i in range(iterations):
                for pos in positions:
                    instr = CoachInstruction(
                        content=f"Cycle {i}: instruction for {pos}",
                        timestamp=time.time(),
                        target_position=pos,
                    )
                    instruction_store.set_instruction(pos, instr)
                time.sleep(0.002)

        def mixed_reader():
            """Simulates a Player Agent reading both state and instructions."""
            for _ in range(iterations):
                snapshot = shared_state.get_snapshot()
                if snapshot is not None and not isinstance(snapshot, dict):
                    errors.append("SharedState returned non-dict")
                instr = instruction_store.get_instruction("Striker")
                if instr is not None and not isinstance(instr.content, str):
                    errors.append("InstructionStore returned non-str content")
                time.sleep(0.002)

        threads = [
            threading.Thread(target=state_writer),
            threading.Thread(target=instruction_writer),
            threading.Thread(target=mixed_reader),
            threading.Thread(target=mixed_reader),
            threading.Thread(target=mixed_reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Concurrent access errors: {errors}"


class TestTwoInstanceIsolation:
    """Integration: Two TeamOrchestrator instances have no shared mutable state.

    Validates Requirement 8.1: Two instances run concurrently without shared
    mutable state.
    """

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_two_instances_have_separate_containers(
        self, mock_player_cls, mock_coach_cls, mock_poller_cls
    ):
        """Red and Blue orchestrators create independent state containers."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config_red = _make_config(team_color="Red")
        config_blue = _make_config(team_color="Blue")

        orch_red = TeamOrchestrator(config_red)
        orch_blue = TeamOrchestrator(config_blue)

        orch_red.start()
        orch_blue.start()

        # Verify completely separate state containers
        assert orch_red.shared_state is not orch_blue.shared_state
        assert orch_red.instruction_store is not orch_blue.instruction_store
        assert orch_red.debug_store is not orch_blue.debug_store
        assert orch_red._stop_event is not orch_blue._stop_event

        # Verify separate thread lists
        assert orch_red._threads is not orch_blue._threads
        assert len(orch_red._threads) == 6
        assert len(orch_blue._threads) == 6

        # Clean up
        orch_red.stop(timeout=2.0)
        orch_blue.stop(timeout=2.0)

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_writing_to_one_instance_does_not_affect_other(
        self, mock_player_cls, mock_coach_cls, mock_poller_cls, sample_snapshot
    ):
        """Writing state to Red instance does not appear in Blue instance."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config_red = _make_config(team_color="Red")
        config_blue = _make_config(team_color="Blue")

        orch_red = TeamOrchestrator(config_red)
        orch_blue = TeamOrchestrator(config_blue)

        orch_red.start()
        orch_blue.start()

        # Write to Red's SharedState
        orch_red.shared_state.set_snapshot(sample_snapshot)

        # Blue's SharedState should still be None
        assert orch_blue.shared_state.get_snapshot() is None

        # Write instruction to Red's InstructionStore
        instr = CoachInstruction(
            content="Red team instruction",
            timestamp=time.time(),
            target_position="Striker",
        )
        orch_red.instruction_store.set_instruction("Striker", instr)

        # Blue's InstructionStore should have no instruction
        assert orch_blue.instruction_store.get_instruction("Striker") is None

        # Clean up
        orch_red.stop(timeout=2.0)
        orch_blue.stop(timeout=2.0)

    @patch("team.orchestrator.StatePoller")
    @patch("team.orchestrator.CoachAgent")
    @patch("team.orchestrator.PlayerAgent")
    def test_stopping_one_instance_does_not_affect_other(
        self, mock_player_cls, mock_coach_cls, mock_poller_cls
    ):
        """Stopping Red does not stop Blue."""
        mock_poller_cls.return_value.run = MagicMock()
        mock_coach_cls.return_value.run = MagicMock()
        mock_player_cls.return_value.run = MagicMock()

        config_red = _make_config(team_color="Red")
        config_blue = _make_config(team_color="Blue")

        orch_red = TeamOrchestrator(config_red)
        orch_blue = TeamOrchestrator(config_blue)

        orch_red.start()
        orch_blue.start()

        # Stop Red
        orch_red.stop(timeout=2.0)

        # Red's stop event should be set
        assert orch_red._stop_event.is_set()

        # Blue's stop event should NOT be set
        assert not orch_blue._stop_event.is_set()

        # Clean up
        orch_blue.stop(timeout=2.0)
