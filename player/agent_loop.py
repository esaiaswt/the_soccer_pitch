"""Agent loop module implementing the Look-Think-Act cycle.

Runs in a background thread, continuously polling game state from the Pitch
server, invoking the LLM for movement decisions, and submitting actions back
to the server. Any failure at any step results in the Brake_Action (safe stop).
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

from config import (
    BRAKE_ACTION,
    LOOP_DELAY,
    REQUEST_TIMEOUT,
    TEAMS,
    ActionModel,
    build_url,
)
from llm_client import StructuredLLM, invoke_llm
from logging_config import setup_logging
from spatial import analyze_game_state, format_spatial_summary


@dataclass
class IterationResult:
    """Data from one loop iteration, used to update the debug console."""

    game_state: Optional[dict] = None
    action: ActionModel = field(default_factory=lambda: BRAKE_ACTION)
    fallback_reason: Optional[str] = None
    error_details: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).astimezone().isoformat())


class AgentLoop:
    """Implements the continuous Look-Think-Act agent loop.

    Runs in a background thread, polling game state, invoking the LLM for
    decisions, and posting actions to the Pitch server. Stops when the
    stop_event is set.

    Args:
        server_ip: The Pitch server hostname or IP address.
        team: The team name ("Red" or "Blue").
        position: The player position (e.g., "Striker").
        llm_client: A structured LLM client that returns ActionModel.
        get_system_prompt: Callable returning the current system prompt text.
        get_behavior_override: Callable returning the current behavior override text.
        on_iteration: Callback invoked with each IterationResult.
        stop_event: Threading event; loop stops when this is set.

    Raises:
        ValueError: If server_ip is empty or team is not in TEAMS list.
    """

    def __init__(
        self,
        server_ip: str,
        team: str,
        position: str,
        llm_client: StructuredLLM,
        get_system_prompt: Callable[[], str],
        get_behavior_override: Callable[[], str],
        on_iteration: Callable[[IterationResult], None],
        stop_event: threading.Event,
        agent_name: str = "",
    ) -> None:
        # Validate configuration before allowing loop to start
        if not server_ip or not server_ip.strip():
            raise ValueError("server_ip must not be empty")
        if team not in TEAMS:
            raise ValueError(f"team must be one of {TEAMS}, got '{team}'")

        self.server_ip = server_ip
        self.team = team
        self.position = position
        self.llm_client = llm_client
        self.get_system_prompt = get_system_prompt
        self.get_behavior_override = get_behavior_override
        self.on_iteration = on_iteration
        self.stop_event = stop_event
        self.agent_name = agent_name
        self.logger = setup_logging()

    def run(self) -> None:
        """Main loop: look -> think -> act -> sleep 1.5s. Stops on stop_event."""
        self.logger.info("Agent loop started (team=%s, position=%s)", self.team, self.position)

        while not self.stop_event.is_set():
            result = self._run_iteration()
            self.on_iteration(result)

            # Interruptible sleep using stop_event.wait
            self.stop_event.wait(LOOP_DELAY)

        self.logger.info("Agent loop stopped")

    def _run_iteration(self) -> IterationResult:
        """Execute one Look-Think-Act cycle with comprehensive error handling."""
        # --- Look step ---
        game_state = self._look()
        if isinstance(game_state, IterationResult):
            # Look failed, return the error result
            return game_state

        # --- Think step ---
        action = self._think(game_state)
        if isinstance(action, IterationResult):
            # Think failed, return the error result
            return action

        # --- Act step ---
        self._act(action)

        return IterationResult(
            game_state=game_state,
            action=action,
            fallback_reason=None,
            error_details=None,
        )

    def _look(self) -> "dict | IterationResult":
        """GET /api/state with 5s timeout. Returns game state dict or IterationResult on failure."""
        state_url = build_url(self.server_ip, "state")
        try:
            response = requests.get(state_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                game_state = response.json()
                self.logger.debug("Game state retrieved successfully")
                return game_state
            else:
                reason = f"HTTP {response.status_code}"
                self.logger.warning("Look step failed: %s", reason)
                return IterationResult(
                    action=BRAKE_ACTION,
                    fallback_reason=reason,
                    error_details=reason,
                )
        except requests.Timeout:
            reason = "connection timeout"
            self.logger.warning("Look step failed: %s", reason)
            return IterationResult(
                action=BRAKE_ACTION,
                fallback_reason=reason,
                error_details=reason,
            )
        except requests.RequestException as e:
            reason = f"connection error: {e}"
            self.logger.warning("Look step failed: %s", reason)
            return IterationResult(
                action=BRAKE_ACTION,
                fallback_reason=reason,
                error_details=str(e),
            )

    def _think(self, game_state: dict) -> "ActionModel | IterationResult":
        """Invoke LLM with system prompt + game state. Returns ActionModel or IterationResult on failure."""
        system_prompt = self.get_system_prompt()

        # Check for empty/whitespace system prompt
        if not system_prompt or not system_prompt.strip():
            reason = "system prompt required"
            self.logger.warning("Think step skipped: %s", reason)
            return IterationResult(
                game_state=game_state,
                action=BRAKE_ACTION,
                fallback_reason=reason,
                error_details=reason,
            )

        behavior_override = self.get_behavior_override()

        try:
            # Compute spatial analysis and append to game state for the LLM
            spatial_data = analyze_game_state(game_state, self.team, self.position)
            spatial_summary = format_spatial_summary(spatial_data)
            enriched_state = json.dumps(game_state) + "\n\n" + spatial_summary

            action = invoke_llm(
                client=self.llm_client,
                system_prompt=system_prompt,
                game_state_json=enriched_state,
                behavior_override=behavior_override,
            )
            if action is None:
                reason = "empty response"
                self.logger.warning("Think step fallback: %s", reason)
                return IterationResult(
                    game_state=game_state,
                    action=BRAKE_ACTION,
                    fallback_reason=reason,
                    error_details=reason,
                )
            self.logger.debug("LLM decision: dx=%.2f, dy=%.2f, kick=%s", action.dx, action.dy, action.kick)
            return action
        except Exception as e:
            reason = f"LLM error: {type(e).__name__}"
            self.logger.error("Think step failed: %s - %s", reason, str(e))
            return IterationResult(
                game_state=game_state,
                action=BRAKE_ACTION,
                fallback_reason=reason,
                error_details=str(e),
            )

    def _act(self, action: ActionModel) -> None:
        """POST /api/action with JSON payload. Logs errors but continues loop."""
        action_url = build_url(self.server_ip, "action")
        payload = {
            "team": self.team,
            "position": self.position,
            "vector": {"dx": action.dx, "dy": action.dy},
            "kick": action.kick,
            "agent_name": self.agent_name,
        }

        try:
            response = requests.post(action_url, json=payload, timeout=REQUEST_TIMEOUT)
            if response.status_code >= 200 and response.status_code < 300:
                self.logger.debug("Action submitted successfully")
            else:
                self.logger.error(
                    "Action submission returned HTTP %d", response.status_code
                )
        except Exception as e:
            self.logger.error("Action submission failed: %s", str(e))
