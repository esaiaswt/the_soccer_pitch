"""State Poller — dedicated thread that polls the Pitch server for game state.

The State Poller is the ONLY component that calls GET /api/state on the Pitch
server. It updates the SharedState container on each successful poll so that
all agents (Coach + 4 Players) can read the latest snapshot.

On HTTP errors or connection timeouts, the poller logs the error and preserves
the last good snapshot in SharedState. It uses stop_event.wait() instead of
time.sleep() so that shutdown is responsive within one polling interval.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

from __future__ import annotations

from threading import Event

import requests

from team.config import TeamConfig
from team.logging_config import get_logger, log_agent_error
from team.shared_state import SharedState


class StatePoller:
    """Polls GET /api/state from the Pitch server at a configurable interval.

    Designed to be used as a thread target via ``threading.Thread(target=poller.run)``.

    Parameters
    ----------
    config : TeamConfig
        Team configuration containing pitch_host, pitch_port, and poll_interval.
    shared_state : SharedState
        Thread-safe container to store the latest game state snapshot.
    stop_event : Event
        Threading event signaling the poller to stop. When set, the poller
        exits its loop within one polling interval.
    """

    def __init__(self, config: TeamConfig, shared_state: SharedState, stop_event: Event) -> None:
        self._config = config
        self._shared_state = shared_state
        self._stop_event = stop_event
        self._url = f"http://{config.pitch_host}:{config.pitch_port}/api/state"

    def run(self) -> None:
        """Thread target — poll the Pitch server until stop_event is set.

        On each iteration:
        1. Send GET /api/state with a 5-second timeout.
        2. On success, parse JSON and update SharedState.
        3. On failure (HTTP error, timeout, connection error), log the error
           and preserve the last good snapshot.
        4. Wait for poll_interval using stop_event.wait() for responsive shutdown.
        """
        logger = get_logger()
        logger.info(
            "State Poller started",
            extra={
                "agent_identity": "StatePoller",
                "structured_context": f"url={self._url} | interval={self._config.poll_interval}s",
            },
        )

        while not self._stop_event.is_set():
            self._poll_once(logger)

            # Wait for the configured interval, but exit early if stop is signaled
            if self._stop_event.wait(timeout=self._config.poll_interval):
                break

        logger.info(
            "State Poller stopped",
            extra={"agent_identity": "StatePoller", "structured_context": ""},
        )

    def _poll_once(self, logger) -> None:
        """Execute a single poll attempt against the Pitch server."""
        try:
            response = requests.get(self._url, timeout=5)
            response.raise_for_status()
            data = response.json()
            self._shared_state.set_snapshot(data)
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            log_agent_error(
                agent_identity="StatePoller",
                error_type="HTTPError",
                match_state="unknown",
                attempted_action="poll_state",
                error_details=f"HTTP {status_code} from {self._url}",
            )
        except requests.exceptions.Timeout:
            log_agent_error(
                agent_identity="StatePoller",
                error_type="Timeout",
                match_state="unknown",
                attempted_action="poll_state",
                error_details=f"Connection timeout (5s) polling {self._url}",
            )
        except requests.exceptions.ConnectionError as exc:
            log_agent_error(
                agent_identity="StatePoller",
                error_type="ConnectionError",
                match_state="unknown",
                attempted_action="poll_state",
                error_details=f"Connection error polling {self._url}: {exc}",
            )
        except requests.exceptions.JSONDecodeError:
            log_agent_error(
                agent_identity="StatePoller",
                error_type="JSONDecodeError",
                match_state="unknown",
                attempted_action="poll_state",
                error_details=f"Malformed JSON response from {self._url}",
            )
        except Exception as exc:
            log_agent_error(
                agent_identity="StatePoller",
                error_type=type(exc).__name__,
                match_state="unknown",
                attempted_action="poll_state",
                error_details=f"Unexpected error polling {self._url}: {exc}",
            )
