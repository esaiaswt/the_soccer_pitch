# Feature: agent-control-panel, Property 11: Log entries contain ISO 8601 timestamps and required event data
"""Property-based tests for logging configuration.

Validates: Requirements 11.2, 11.3
"""

import io
import logging
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from logging_config import ISO8601Formatter, LOG_FORMAT, DATE_FORMAT


# ISO 8601 timestamp pattern (YYYY-MM-DDTHH:MM:SS) without microseconds
ISO8601_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"

# Valid log levels as defined in the requirements
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

# Full log entry pattern: {ISO_8601_TIMESTAMP} | {LEVEL} | {message}
LOG_ENTRY_PATTERN = re.compile(
    rf"^{ISO8601_PATTERN} \| (?:DEBUG|INFO|WARNING|ERROR) \| .+$"
)


@settings(max_examples=100)
@given(
    message=st.text(
        min_size=1,
        alphabet=st.characters(
            blacklist_categories=("Cc", "Cs"),
            blacklist_characters="\n\r",
        ),
    ).filter(lambda s: len(s.strip()) > 0),
    level=st.sampled_from(VALID_LOG_LEVELS),
)
def test_log_entries_contain_iso8601_timestamps_and_required_data(message: str, level: str):
    """For any logged event, the log entry SHALL contain an ISO 8601 formatted
    timestamp (YYYY-MM-DDTHH:MM:SS), a valid log level (DEBUG, INFO, WARNING,
    or ERROR), and event-specific details.

    **Validates: Requirements 11.2, 11.3**
    """
    # Set up a logger with a StringIO handler using the same formatter as logging_config
    test_logger = logging.getLogger(f"test_property11_{id(message)}")
    test_logger.setLevel(logging.DEBUG)
    test_logger.handlers.clear()

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    formatter = ISO8601Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    test_logger.addHandler(handler)

    # Log the generated message at the generated level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    test_logger.log(level_map[level], message)

    # Read the output - strip trailing newline added by StreamHandler
    output = stream.getvalue().rstrip("\n")

    # Property: log entry matches the expected format pattern
    assert LOG_ENTRY_PATTERN.match(output), (
        f"Log output did not match expected pattern.\n"
        f"Output: {output!r}\n"
        f"Expected pattern: {ISO8601_PATTERN} | {{LEVEL}} | {{message}}"
    )

    # Property: timestamp is valid ISO 8601 (YYYY-MM-DDTHH:MM:SS)
    timestamp_match = re.match(ISO8601_PATTERN, output)
    assert timestamp_match is not None, (
        f"No ISO 8601 timestamp found at start of log entry.\n"
        f"Output: {output!r}"
    )

    # Property: log level is one of the valid levels
    parts = output.split(" | ", 2)
    assert len(parts) == 3, (
        f"Log entry does not have 3 pipe-separated parts.\n"
        f"Output: {output!r}"
    )
    assert parts[1] in VALID_LOG_LEVELS, (
        f"Log level '{parts[1]}' is not a valid level.\n"
        f"Valid levels: {VALID_LOG_LEVELS}"
    )

    # Property: the logged level matches the requested level
    assert parts[1] == level, (
        f"Logged level '{parts[1]}' does not match requested level '{level}'.\n"
        f"Output: {output!r}"
    )

    # Property: event-specific details (message) are present in the log entry
    assert message in parts[2], (
        f"Message not found in log entry details.\n"
        f"Message: {message!r}\n"
        f"Details section: {parts[2]!r}"
    )

    # Clean up
    test_logger.removeHandler(handler)
    handler.close()
