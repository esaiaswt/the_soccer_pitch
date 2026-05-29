# Feature: the-pitch, Property 13: Log entry format
"""Property-based tests for logging configuration.

Validates: Requirements 10.1
"""

import io
import logging
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from pitch.logging_config import ISO8601Formatter, LOG_FORMAT


# ISO 8601 timestamp pattern with microseconds
ISO8601_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}"
LOG_ENTRY_PATTERN = re.compile(
    rf"^{ISO8601_PATTERN} (INFO|WARNING|ERROR) .+$"
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
    level=st.sampled_from(["INFO", "WARNING", "ERROR"]),
)
def test_log_entry_format_matches_pattern(message: str, level: str):
    """For any log message, the formatted output shall match
    {ISO8601_TIMESTAMP} {LEVEL} {message} pattern.

    **Validates: Requirements 10.1**
    """
    # Set up a logger with a StringIO handler using the same formatter
    test_logger = logging.getLogger(f"test_property13_{id(message)}")
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
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    test_logger.log(level_map[level], message)

    # Read the output - only strip the trailing newline added by StreamHandler
    output = stream.getvalue().rstrip("\n")

    # Verify it matches the expected pattern
    assert LOG_ENTRY_PATTERN.match(output), (
        f"Log output did not match expected pattern.\n"
        f"Output: {output!r}\n"
        f"Expected pattern: {LOG_ENTRY_PATTERN.pattern}"
    )

    # Verify the message content is present in the output
    assert message in output, (
        f"Message not found in log output.\n"
        f"Message: {message!r}\n"
        f"Output: {output!r}"
    )

    # Clean up
    test_logger.removeHandler(handler)
    handler.close()
