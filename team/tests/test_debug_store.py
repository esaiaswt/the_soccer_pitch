"""Unit tests for the DebugStore thread-safe container."""

from __future__ import annotations

import threading
import time

from team.debug_store import DebugStore, PlayerDebugInfo


class TestPlayerDebugInfo:
    """Tests for the PlayerDebugInfo dataclass."""

    def test_create_with_all_fields(self) -> None:
        info = PlayerDebugInfo(
            latest_state={"ball": {"x": 100, "y": 200}},
            latest_action={"dx": 0.5, "dy": -0.3, "kick": True},
            latest_instruction="Push forward",
            last_update=1000.0,
        )
        assert info.latest_state == {"ball": {"x": 100, "y": 200}}
        assert info.latest_action == {"dx": 0.5, "dy": -0.3, "kick": True}
        assert info.latest_instruction == "Push forward"
        assert info.last_update == 1000.0

    def test_create_with_none_fields(self) -> None:
        info = PlayerDebugInfo(
            latest_state=None,
            latest_action=None,
            latest_instruction=None,
            last_update=0.0,
        )
        assert info.latest_state is None
        assert info.latest_action is None
        assert info.latest_instruction is None
        assert info.last_update == 0.0


class TestDebugStorePlayerOperations:
    """Tests for DebugStore player-related methods."""

    def test_get_player_returns_none_for_unknown_position(self) -> None:
        store = DebugStore()
        assert store.get_player("Goalkeeper") is None

    def test_update_and_get_player(self) -> None:
        store = DebugStore()
        info = PlayerDebugInfo(
            latest_state={"match_state": "Playing"},
            latest_action={"dx": 1.0, "dy": 0.0, "kick": False},
            latest_instruction="Hold position",
            last_update=time.time(),
        )
        store.update_player("Defender", info)
        result = store.get_player("Defender")
        assert result == info

    def test_update_player_overwrites_previous(self) -> None:
        store = DebugStore()
        info1 = PlayerDebugInfo(
            latest_state={"v": 1},
            latest_action=None,
            latest_instruction=None,
            last_update=1.0,
        )
        info2 = PlayerDebugInfo(
            latest_state={"v": 2},
            latest_action={"dx": 0, "dy": 0, "kick": False},
            latest_instruction="Attack",
            last_update=2.0,
        )
        store.update_player("Striker", info1)
        store.update_player("Striker", info2)
        assert store.get_player("Striker") == info2

    def test_multiple_positions_independent(self) -> None:
        store = DebugStore()
        gk_info = PlayerDebugInfo(
            latest_state={"pos": "gk"},
            latest_action=None,
            latest_instruction=None,
            last_update=1.0,
        )
        st_info = PlayerDebugInfo(
            latest_state={"pos": "st"},
            latest_action=None,
            latest_instruction=None,
            last_update=2.0,
        )
        store.update_player("Goalkeeper", gk_info)
        store.update_player("Striker", st_info)
        assert store.get_player("Goalkeeper") == gk_info
        assert store.get_player("Striker") == st_info


class TestDebugStoreCoachOperations:
    """Tests for DebugStore coach-related methods."""

    def test_get_coach_returns_empty_defaults(self) -> None:
        store = DebugStore()
        result = store.get_coach()
        assert result == {"observations": [], "instructions": {}}

    def test_update_and_get_coach(self) -> None:
        store = DebugStore()
        observations = [{"ball": {"x": 100, "y": 200}}, {"ball": {"x": 150, "y": 250}}]
        instructions = {"Goalkeeper": "Stay back", "Striker": "Push forward"}
        store.update_coach(observations, instructions)
        result = store.get_coach()
        assert result == {"observations": observations, "instructions": instructions}

    def test_update_coach_overwrites_previous(self) -> None:
        store = DebugStore()
        store.update_coach([{"v": 1}], {"Goalkeeper": "old"})
        store.update_coach([{"v": 2}], {"Goalkeeper": "new"})
        result = store.get_coach()
        assert result == {"observations": [{"v": 2}], "instructions": {"Goalkeeper": "new"}}

    def test_get_coach_returns_copy(self) -> None:
        """Modifying the returned dict should not affect the store."""
        store = DebugStore()
        store.update_coach([{"data": 1}], {"Striker": "go"})
        result = store.get_coach()
        result["observations"].append({"data": 2})
        result["instructions"]["Defender"] = "hold"
        # Original should be unchanged
        original = store.get_coach()
        assert len(original["observations"]) == 1
        assert "Defender" not in original["instructions"]


class TestDebugStoreConcurrency:
    """Thread-safety tests for DebugStore."""

    def test_concurrent_player_writes_and_reads(self) -> None:
        """Multiple writers and readers should not corrupt data."""
        store = DebugStore()
        errors: list[str] = []
        stop_event = threading.Event()

        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]

        def writer(position: str) -> None:
            for i in range(50):
                info = PlayerDebugInfo(
                    latest_state={"iteration": i},
                    latest_action={"dx": 0, "dy": 0, "kick": False},
                    latest_instruction=f"instruction_{i}",
                    last_update=float(i),
                )
                store.update_player(position, info)
            stop_event.set()

        def reader() -> None:
            while not stop_event.is_set():
                for pos in positions:
                    result = store.get_player(pos)
                    if result is not None and "iteration" not in (result.latest_state or {}):
                        errors.append(f"Unexpected state for {pos}: {result.latest_state}")

        writer_threads = [threading.Thread(target=writer, args=(p,)) for p in positions]
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        for t in reader_threads:
            t.start()
        for t in writer_threads:
            t.start()

        for t in writer_threads:
            t.join()
        stop_event.set()
        for t in reader_threads:
            t.join()

        assert errors == []

    def test_concurrent_coach_writes_and_reads(self) -> None:
        """Coach writer and dashboard reader should not corrupt data."""
        store = DebugStore()
        errors: list[str] = []
        stop_event = threading.Event()

        def writer() -> None:
            for i in range(50):
                store.update_coach(
                    [{"iteration": i}],
                    {"Striker": f"instruction_{i}"},
                )
            stop_event.set()

        def reader() -> None:
            while not stop_event.is_set():
                result = store.get_coach()
                if "observations" not in result or "instructions" not in result:
                    errors.append(f"Missing keys in coach data: {result}")

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        for t in reader_threads:
            t.start()
        writer_thread.start()

        writer_thread.join()
        stop_event.set()
        for t in reader_threads:
            t.join()

        assert errors == []
