"""Integration tests for agentic modules in the team player agent.

Tests verify:
1. Exactly one LLM call per PlayerAgent _loop_iteration() cycle (Req 9.2)
2. No LLM calls from agentic modules (Req 9.1)
3. Signal Bus thread safety with concurrent readers/writers (Req 7.5)
4. Application independence — no cross-package imports (Req 10.1, 10.2)

Requirements: 7.5, 9.1, 9.2, 10.1, 10.2
"""

from __future__ import annotations

import ast
import os
import threading
import time
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from team.config import TeamConfig
from team.debug_store import DebugStore
from team.episodic_memory import Episode, EpisodicMemory
from team.instruction_store import InstructionStore
from team.player_agent import PlayerAgent
from team.shared_state import SharedState
from team.signal_bus import Signal, SignalBus


# --- Fixtures and helpers ---


def _make_config(**overrides) -> TeamConfig:
    """Create a TeamConfig with sensible defaults for integration testing."""
    defaults = {
        "pitch_host": "localhost",
        "pitch_port": 8000,
        "nvidia_api_key": "test-key",
        "coach_model": "test-model",
        "player_model": "test-model",
        "coaching_frequency": 2.0,
        "poll_interval": 0.05,
        "streamlit_port": None,
        "team_color": "Red",
        "coach_memory_size": 50,
        "agent_name": "TestBot",
    }
    defaults.update(overrides)
    return TeamConfig(**defaults)


def _make_game_state() -> dict:
    """Create a realistic game state dict for testing."""
    return {
        "ball": {"x": 600.0, "y": 400.0},
        "players": {
            "Red_Goalkeeper": {"x": 50.0, "y": 425.0},
            "Red_Defender": {"x": 200.0, "y": 350.0},
            "Red_Midfielder": {"x": 400.0, "y": 400.0},
            "Red_Striker": {"x": 500.0, "y": 400.0},
            "Blue_Goalkeeper": {"x": 1150.0, "y": 425.0},
            "Blue_Defender": {"x": 900.0, "y": 350.0},
            "Blue_Midfielder": {"x": 700.0, "y": 400.0},
            "Blue_Striker": {"x": 600.0, "y": 500.0},
        },
        "match_state": "Playing",
        "score": {"Red": 0, "Blue": 0},
        "time_left": 60.0,
    }


def _make_player_agent(signal_bus=None) -> PlayerAgent:
    """Create a PlayerAgent with mocked LLM for integration testing.

    The ChatNVIDIA constructor is patched at the module level in each test.
    """
    config = _make_config()
    shared_state = SharedState()
    shared_state.set_snapshot(_make_game_state())
    instruction_store = InstructionStore()
    debug_store = DebugStore()
    stop_event = Event()

    player = PlayerAgent(
        config=config,
        position="Striker",
        shared_state=shared_state,
        instruction_store=instruction_store,
        stop_event=stop_event,
        debug_store=debug_store,
        signal_bus=signal_bus,
    )
    return player


# --- Test Classes ---


class TestSingleLLMCallPerPlayerCycle:
    """Verify exactly one LLM call per PlayerAgent _loop_iteration() cycle.

    Validates: Requirement 9.2 - The Player_Agent SHALL continue to make
    exactly one LLM API call per Look-Think-Act_Cycle.
    """

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_single_llm_call_per_iteration(self, mock_chat_cls, mock_post):
        """A single _loop_iteration() makes exactly 1 LLM call."""
        # Set up mock LLM
        llm_response = MagicMock()
        llm_response.content = "dx=0.5 dy=-0.3 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        # Set up mock action POST
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        player = _make_player_agent()

        player._loop_iteration()

        assert mock_llm_instance.invoke.call_count == 1

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_single_llm_call_across_multiple_iterations(self, mock_chat_cls, mock_post):
        """Multiple iterations each make exactly 1 LLM call (N iterations = N calls)."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.1 dy=0.2 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        player = _make_player_agent()

        for _ in range(3):
            player._loop_iteration()

        assert mock_llm_instance.invoke.call_count == 3

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_single_llm_call_with_active_plan(self, mock_chat_cls, mock_post):
        """Even with an active plan, only 1 LLM call is made per cycle."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.8 dy=0.0 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        player = _make_player_agent()

        # Modify game state to trigger a plan (ball in attacking half, near player)
        game_state = _make_game_state()
        game_state["ball"] = {"x": 700.0, "y": 400.0}
        game_state["players"]["Red_Striker"] = {"x": 680.0, "y": 400.0}
        player._shared_state.set_snapshot(game_state)

        # Run multiple iterations to allow plan creation and evaluation
        for _ in range(5):
            player._loop_iteration()

        # Each iteration should have exactly 1 LLM call
        assert mock_llm_instance.invoke.call_count == 5

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_single_llm_call_with_signal_bus(self, mock_chat_cls, mock_post):
        """With SignalBus active, still only 1 LLM call per cycle."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.3 dy=-0.1 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        signal_bus = SignalBus()
        # Pre-populate with a signal from a teammate
        signal_bus.publish(Signal(
            sender_position="Midfielder",
            signal_type="making_run",
            payload="left flank",
            timestamp=time.time(),
        ))

        player = _make_player_agent(signal_bus=signal_bus)

        player._loop_iteration()

        assert mock_llm_instance.invoke.call_count == 1

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_single_llm_call_with_reflection_running(self, mock_chat_cls, mock_post):
        """Reflection engine running does not add extra LLM calls."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.3 dy=-0.1 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        player = _make_player_agent()

        # First iteration sets up previous state
        player._loop_iteration()
        # Second iteration triggers reflection on previous action
        player._loop_iteration()

        # Still exactly 1 call per iteration
        assert mock_llm_instance.invoke.call_count == 2


class TestNoLLMCallsFromAgenticModules:
    """Verify agentic modules (EpisodicMemory, Planner, ReflectionEngine,
    StrategyTracker, SignalBus, SignalGenerator) do not make any LLM calls.

    Validates: Requirement 9.1 - The Episodic_Memory, Plan evaluation,
    Reflection_Engine, Strategy_Tracker, and Signal_Bus SHALL execute
    entirely in Python without making LLM API calls.
    """

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_post_look_agentic_makes_no_llm_calls(self, mock_chat_cls, mock_post):
        """_post_look_agentic() does not trigger any LLM calls."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.1 dy=0.1 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        signal_bus = SignalBus()
        player = _make_player_agent(signal_bus=signal_bus)

        # Set up previous state so reflection runs
        player._previous_state = _make_game_state()
        player._previous_action = {"dx": 0.5, "dy": 0.0, "kick": False}
        player._cycle_counter = 1

        # Reset mock to track only post-look calls
        mock_llm_instance.invoke.reset_mock()

        game_state = _make_game_state()
        player._post_look_agentic(game_state)

        # No LLM calls should have been made
        mock_llm_instance.invoke.assert_not_called()

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_assemble_agentic_context_makes_no_llm_calls(self, mock_chat_cls, mock_post):
        """_assemble_agentic_context() does not trigger any LLM calls."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.1 dy=0.1 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        signal_bus = SignalBus()
        player = _make_player_agent(signal_bus=signal_bus)

        # Add some episodes to memory so summarization runs
        for i in range(5):
            episode = Episode(
                cycle=i,
                game_state=_make_game_state(),
                action={"dx": 0.1, "dy": 0.2, "kick": False},
                next_state_delta={},
                effectiveness=0.6,
            )
            player._memory.add(episode)

        # Reset mock to track only context assembly calls
        mock_llm_instance.invoke.reset_mock()

        context = player._assemble_agentic_context()

        # No LLM calls should have been made
        mock_llm_instance.invoke.assert_not_called()
        # Context should be a string (possibly empty)
        assert isinstance(context, str)

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_post_act_agentic_makes_no_llm_calls(self, mock_chat_cls, mock_post):
        """_post_act_agentic() does not trigger any LLM calls."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.1 dy=0.1 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        signal_bus = SignalBus()
        player = _make_player_agent(signal_bus=signal_bus)
        player._cycle_counter = 1

        # Reset mock to track only post-act calls
        mock_llm_instance.invoke.reset_mock()

        game_state = _make_game_state()
        action_dict = {"dx": 0.5, "dy": -0.3, "kick": True}
        player._post_act_agentic(game_state, action_dict)

        # No LLM calls should have been made
        mock_llm_instance.invoke.assert_not_called()

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_strategy_tracker_analysis_makes_no_llm_calls(self, mock_chat_cls, mock_post):
        """Strategy tracker analysis (triggered periodically) makes no LLM calls."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.1 dy=0.1 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        player = _make_player_agent()

        # Run enough iterations to trigger strategy analysis (every 10 cycles)
        for _ in range(11):
            player._loop_iteration()

        # Should be exactly 11 LLM calls (1 per iteration), no extras from analysis
        assert mock_llm_instance.invoke.call_count == 11

    @patch("team.player_agent.requests.post")
    @patch("team.player_agent.ChatNVIDIA")
    def test_signal_generator_makes_no_llm_calls(self, mock_chat_cls, mock_post):
        """Signal generation during post-act does not make LLM calls."""
        llm_response = MagicMock()
        llm_response.content = "dx=0.5 dy=0.0 kick=false"
        llm_response.usage_metadata = None
        llm_response.response_metadata = None
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = llm_response
        mock_chat_cls.return_value = mock_llm_instance

        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        signal_bus = SignalBus()
        player = _make_player_agent(signal_bus=signal_bus)

        # Run a full iteration (signal generation happens in post-act)
        player._loop_iteration()

        # Exactly 1 LLM call from the think step, none from signal generation
        assert mock_llm_instance.invoke.call_count == 1


class TestSignalBusThreadSafety:
    """Verify Signal Bus thread safety with concurrent readers and writers.

    Validates: Requirement 7.5 - THE Signal_Bus SHALL be thread-safe,
    supporting concurrent reads from 4 Player agents and concurrent writes
    from 4 Player agents.
    """

    def test_concurrent_writers_no_data_corruption(self):
        """4 concurrent writer threads publishing signals without data corruption."""
        signal_bus = SignalBus()
        errors: list[str] = []
        iterations = 100
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]

        def writer(position: str):
            """Write signals for a specific position."""
            for i in range(iterations):
                signal = Signal(
                    sender_position=position,
                    signal_type="making_run",
                    payload=f"iter_{i}",
                    timestamp=time.time(),
                )
                signal_bus.publish(signal)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=writer, args=(pos,))
            for pos in positions
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Verify no corruption: each position should have exactly one signal
        # (the most recent one)
        signals = signal_bus.read_all()
        assert len(signals) == 4

        seen_positions = {s.sender_position for s in signals}
        assert seen_positions == set(positions)

        # Each signal should have valid data
        for signal in signals:
            assert signal.signal_type == "making_run"
            assert signal.payload.startswith("iter_")
            assert signal.sender_position in positions

    def test_concurrent_readers_no_data_corruption(self):
        """4 concurrent reader threads reading signals without data corruption."""
        signal_bus = SignalBus()
        errors: list[str] = []
        iterations = 100

        # Pre-populate with signals
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]
        for pos in positions:
            signal_bus.publish(Signal(
                sender_position=pos,
                signal_type="covering_zone",
                payload=f"zone_{pos}",
                timestamp=time.time(),
            ))

        def reader(reader_id: int):
            """Read signals and verify consistency."""
            for _ in range(iterations):
                signals = signal_bus.read_all()
                if not isinstance(signals, list):
                    errors.append(f"Reader {reader_id}: signals is not a list")
                    continue
                for s in signals:
                    if not isinstance(s, Signal):
                        errors.append(f"Reader {reader_id}: item is not a Signal")
                    elif s.sender_position not in positions:
                        errors.append(
                            f"Reader {reader_id}: invalid position {s.sender_position}"
                        )
                    elif s.signal_type != "covering_zone":
                        errors.append(
                            f"Reader {reader_id}: unexpected type {s.signal_type}"
                        )
                time.sleep(0.001)

        threads = [
            threading.Thread(target=reader, args=(i,))
            for i in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Concurrent read errors: {errors}"

    def test_concurrent_readers_and_writers_no_corruption(self):
        """4 writer threads and 4 reader threads operating simultaneously."""
        signal_bus = SignalBus()
        errors: list[str] = []
        iterations = 100
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]

        def writer(position: str):
            """Write signals for a specific position."""
            for i in range(iterations):
                signal = Signal(
                    sender_position=position,
                    signal_type="requesting_pass",
                    payload=f"w{i}",
                    timestamp=time.time(),
                )
                signal_bus.publish(signal)
                time.sleep(0.001)

        def reader(reader_id: int):
            """Read signals and verify they are well-formed."""
            for _ in range(iterations):
                signals = signal_bus.read_all()
                if not isinstance(signals, list):
                    errors.append(f"Reader {reader_id}: signals is not a list")
                    continue
                for s in signals:
                    if not isinstance(s, Signal):
                        errors.append(f"Reader {reader_id}: item is not a Signal")
                    elif s.sender_position not in positions:
                        errors.append(
                            f"Reader {reader_id}: invalid position {s.sender_position}"
                        )
                    elif not s.payload.startswith("w"):
                        errors.append(
                            f"Reader {reader_id}: corrupted payload {s.payload}"
                        )
                time.sleep(0.001)

        writer_threads = [
            threading.Thread(target=writer, args=(pos,))
            for pos in positions
        ]
        reader_threads = [
            threading.Thread(target=reader, args=(i,))
            for i in range(4)
        ]

        all_threads = writer_threads + reader_threads
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join(timeout=15.0)

        assert errors == [], f"Concurrent access errors: {errors}"

        # Final state: each position should have exactly one signal
        final_signals = signal_bus.read_all()
        assert len(final_signals) == 4
        final_positions = {s.sender_position for s in final_signals}
        assert final_positions == set(positions)

    def test_concurrent_writers_with_clear_no_crash(self):
        """Writers and clear() operating concurrently do not crash or corrupt."""
        signal_bus = SignalBus()
        errors: list[str] = []
        iterations = 50
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]

        def writer(position: str):
            for i in range(iterations):
                signal = Signal(
                    sender_position=position,
                    signal_type="supporting",
                    payload=f"c{i}",
                    timestamp=time.time(),
                )
                signal_bus.publish(signal)
                time.sleep(0.002)

        def clearer():
            """Periodically clear the bus (simulating dead ball detection)."""
            for _ in range(10):
                time.sleep(0.01)
                signal_bus.clear()

        def reader(reader_id: int):
            for _ in range(iterations):
                try:
                    signals = signal_bus.read_all()
                    if not isinstance(signals, list):
                        errors.append(f"Reader {reader_id}: not a list")
                except Exception as e:
                    errors.append(f"Reader {reader_id}: exception {e}")
                time.sleep(0.002)

        writer_threads = [
            threading.Thread(target=writer, args=(pos,))
            for pos in positions
        ]
        reader_threads = [
            threading.Thread(target=reader, args=(i,))
            for i in range(4)
        ]
        clear_thread = threading.Thread(target=clearer)

        all_threads = writer_threads + reader_threads + [clear_thread]
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join(timeout=15.0)

        assert errors == [], f"Concurrent access with clear errors: {errors}"


class TestApplicationIndependence:
    """Verify application independence: no cross-package imports.

    Validates: Requirements 10.1, 10.2 -
    - THE player/ application SHALL have no imports from the team/ package.
    - THE team/ application SHALL have no imports from the player/ package.

    Uses static analysis (AST parsing) to scan all .py files in team/ and
    verify no imports from the player/ package.
    """

    def _get_team_py_files(self) -> list[Path]:
        """Get all .py files in the team/ package (excluding tests)."""
        team_dir = Path(__file__).parent.parent
        py_files = []
        for f in team_dir.rglob("*.py"):
            # Skip __pycache__ directories
            if "__pycache__" in str(f):
                continue
            py_files.append(f)
        return py_files

    def _extract_imports(self, filepath: Path) -> list[str]:
        """Extract all import module names from a Python file using AST."""
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            return []

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def test_team_has_no_imports_from_player_package(self):
        """No .py file in team/ imports from the player/ package."""
        team_files = self._get_team_py_files()
        violations: list[str] = []

        for filepath in team_files:
            imports = self._extract_imports(filepath)
            for imp in imports:
                # Check if the import references the player package
                if imp == "player" or imp.startswith("player."):
                    rel_path = filepath.relative_to(filepath.parent.parent)
                    violations.append(f"{rel_path}: imports '{imp}'")

        assert violations == [], (
            f"team/ package has imports from player/ package:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_team_source_files_exist(self):
        """Verify team/ has the expected agentic module files."""
        team_dir = Path(__file__).parent.parent
        expected_modules = [
            "episodic_memory.py",
            "memory_summary.py",
            "planner.py",
            "reflection.py",
            "strategy_tracker.py",
            "context_assembler.py",
            "signal_bus.py",
            "signal_generator.py",
        ]
        for module in expected_modules:
            module_path = team_dir / module
            assert module_path.exists(), f"Expected module {module} not found in team/"

    def test_no_shared_library_imports(self):
        """No .py file in team/ imports from a 'shared' or 'common' library."""
        team_files = self._get_team_py_files()
        violations: list[str] = []

        for filepath in team_files:
            imports = self._extract_imports(filepath)
            for imp in imports:
                # Check for shared/common library patterns
                if imp in ("shared", "common") or imp.startswith(
                    ("shared.", "common.")
                ):
                    rel_path = filepath.relative_to(filepath.parent.parent)
                    violations.append(f"{rel_path}: imports '{imp}'")

        assert violations == [], (
            f"team/ package has imports from shared library:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
