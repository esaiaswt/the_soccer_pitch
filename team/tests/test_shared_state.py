"""Unit tests for the SharedState thread-safe container."""

from __future__ import annotations

import threading
import time

from team.shared_state import SharedState


class TestSharedStateBasics:
    """Basic functionality tests for SharedState."""

    def test_initial_snapshot_is_none(self) -> None:
        state = SharedState()
        assert state.get_snapshot() is None

    def test_initial_last_update_time_is_none(self) -> None:
        state = SharedState()
        assert state.get_last_update_time() is None

    def test_set_and_get_snapshot(self) -> None:
        state = SharedState()
        snapshot = {"ball": {"x": 100, "y": 200}, "score": {"Red": 1, "Blue": 0}}
        state.set_snapshot(snapshot)
        assert state.get_snapshot() == snapshot

    def test_set_snapshot_updates_timestamp(self) -> None:
        state = SharedState()
        before = time.time()
        state.set_snapshot({"ball": {"x": 0, "y": 0}})
        after = time.time()
        update_time = state.get_last_update_time()
        assert update_time is not None
        assert before <= update_time <= after

    def test_set_snapshot_overwrites_previous(self) -> None:
        state = SharedState()
        state.set_snapshot({"version": 1})
        state.set_snapshot({"version": 2})
        assert state.get_snapshot() == {"version": 2}

    def test_timestamp_updates_on_each_set(self) -> None:
        state = SharedState()
        state.set_snapshot({"v": 1})
        t1 = state.get_last_update_time()
        time.sleep(0.01)
        state.set_snapshot({"v": 2})
        t2 = state.get_last_update_time()
        assert t2 is not None and t1 is not None
        assert t2 > t1


class TestSharedStateConcurrency:
    """Thread-safety tests for SharedState."""

    def test_concurrent_reads_and_writes(self) -> None:
        """Multiple readers and one writer should not corrupt data."""
        state = SharedState()
        errors: list[str] = []
        stop_event = threading.Event()

        def writer() -> None:
            for i in range(100):
                state.set_snapshot({"iteration": i})
            stop_event.set()

        def reader() -> None:
            while not stop_event.is_set():
                snapshot = state.get_snapshot()
                if snapshot is not None and "iteration" not in snapshot:
                    errors.append(f"Unexpected snapshot: {snapshot}")

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(5)]

        for t in reader_threads:
            t.start()
        writer_thread.start()

        writer_thread.join()
        stop_event.set()
        for t in reader_threads:
            t.join()

        assert errors == []

    def test_snapshot_preserved_across_threads(self) -> None:
        """Snapshot set by writer is visible to all readers."""
        state = SharedState()
        expected = {"ball": {"x": 500, "y": 300}, "match_state": "Playing"}
        state.set_snapshot(expected)

        results: list[dict | None] = [None] * 5

        def reader(idx: int) -> None:
            results[idx] = state.get_snapshot()

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            assert r == expected
