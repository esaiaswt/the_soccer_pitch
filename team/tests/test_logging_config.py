"""Unit tests for team/logging_config.py.

Verifies structured logging format, thread safety, and helper functions.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading

import pytest

from team.logging_config import (
    TeamLogFormatter,
    get_logger,
    log_agent_error,
    log_coach_instruction,
    log_decision_latency,
    log_token_usage,
    log_token_usage_unavailable,
    setup_logging,
)


# ISO 8601 pattern with microseconds: 2024-01-15T10:30:00.123456
ISO8601_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}"
)


class TestSetupLogging:
    """Tests for setup_logging() function."""

    def test_returns_logger(self):
        logger = setup_logging("TestRed")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "team_testred"

    def test_creates_log_file(self):
        logger = setup_logging("TestCreate")
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "team_testcreate.log")
        # Logger is configured but file is only created on first write
        logger.info("test", extra={"agent_identity": "Test", "structured_context": ""})
        assert os.path.exists(log_path)
        # Close handlers before removing file (Windows file locking)
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)
        os.remove(log_path)

    def test_no_duplicate_handlers_on_repeated_calls(self):
        logger1 = setup_logging("TestDup")
        handler_count = len(logger1.handlers)
        logger2 = setup_logging("TestDup")
        assert len(logger2.handlers) == handler_count
        # Clean up
        for h in logger2.handlers[:]:
            logger2.removeHandler(h)
            h.close()

    def test_log_level_is_debug(self):
        logger = setup_logging("TestLevel")
        assert logger.level == logging.DEBUG
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()


class TestTeamLogFormatter:
    """Tests for the custom log formatter."""

    def test_format_contains_iso8601_timestamp(self):
        formatter = TeamLogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=None, exc_info=None,
        )
        record.agent_identity = "Coach"
        record.structured_context = "key=value"
        output = formatter.format(record)
        assert ISO8601_PATTERN.search(output) is not None

    def test_format_pipe_delimited(self):
        formatter = TeamLogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="Something happened", args=None, exc_info=None,
        )
        record.agent_identity = "Player_Striker"
        record.structured_context = "detail=info"
        output = formatter.format(record)
        parts = output.split(" | ", 4)
        assert len(parts) == 5
        assert parts[1] == "WARNING"
        assert parts[2] == "Player_Striker"
        assert parts[3] == "Something happened"
        assert parts[4] == "detail=info"

    def test_format_defaults_agent_identity_to_system(self):
        formatter = TeamLogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="No identity", args=None, exc_info=None,
        )
        output = formatter.format(record)
        assert "SYSTEM" in output


class TestHelperFunctions:
    """Tests for logging helper functions."""

    @pytest.fixture(autouse=True)
    def _setup_logger(self, tmp_path):
        """Set up a temporary logger for each test."""
        import team.logging_config as lc

        log_file = tmp_path / "test.log"
        logger = logging.getLogger(f"test_helper_{id(self)}")
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        handler.setFormatter(TeamLogFormatter())
        logger.addHandler(handler)

        # Patch the module-level logger
        original = lc._team_logger
        lc._team_logger = logger
        self.log_file = log_file
        yield
        lc._team_logger = original
        handler.close()
        logger.removeHandler(handler)

    def _read_log(self) -> str:
        return self.log_file.read_text(encoding="utf-8")

    def test_log_coach_instruction(self):
        log_coach_instruction("Striker", "Push forward aggressively")
        content = self._read_log()
        assert "Coach" in content
        assert "Coach instruction issued" in content
        assert "target=Striker" in content
        assert "content=Push forward aggressively" in content
        assert ISO8601_PATTERN.search(content) is not None

    def test_log_token_usage(self):
        log_token_usage("Player_Goalkeeper", 100, 50, 150)
        content = self._read_log()
        assert "Player_Goalkeeper" in content
        assert "Token usage" in content
        assert "prompt_tokens=100" in content
        assert "completion_tokens=50" in content
        assert "total_tokens=150" in content

    def test_log_token_usage_unavailable(self):
        log_token_usage_unavailable("Player_Midfielder")
        content = self._read_log()
        assert "Player_Midfielder" in content
        assert "WARNING" in content
        assert "Token usage metadata unavailable" in content
        assert "token_data=unavailable" in content

    def test_log_decision_latency(self):
        log_decision_latency("Coach", 256.78)
        content = self._read_log()
        assert "Coach" in content
        assert "Decision latency" in content
        assert "latency_ms=256.78" in content

    def test_log_agent_error(self):
        log_agent_error(
            agent_identity="Player_Defender",
            error_type="TimeoutError",
            match_state="Playing",
            attempted_action="llm_invoke",
            error_details="LLM did not respond within 10s",
        )
        content = self._read_log()
        assert "ERROR" in content
        assert "Player_Defender" in content
        assert "error_type=TimeoutError" in content
        assert "match_state=Playing" in content
        assert "attempted_action=llm_invoke" in content
        assert "LLM did not respond within 10s" in content


class TestThreadSafety:
    """Tests for concurrent logging thread safety."""

    def test_concurrent_writes_no_interleaving(self, tmp_path):
        """Verify that concurrent writes from multiple threads produce complete lines."""
        import team.logging_config as lc

        log_file = tmp_path / "concurrent.log"
        logger = logging.getLogger("test_concurrent")
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        handler.setFormatter(TeamLogFormatter())
        logger.addHandler(handler)

        original = lc._team_logger
        lc._team_logger = logger

        try:
            threads = []
            num_threads = 20

            for i in range(num_threads):
                t = threading.Thread(
                    target=log_coach_instruction,
                    args=(f"Player_{i}", f"Instruction number {i}"),
                )
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            handler.flush()
            lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == num_threads

            # Each line should have at least 4 pipe delimiters
            for line in lines:
                assert line.count("|") >= 4, f"Malformed line: {line}"
                assert ISO8601_PATTERN.search(line) is not None
        finally:
            lc._team_logger = original
            handler.close()
            logger.removeHandler(handler)
