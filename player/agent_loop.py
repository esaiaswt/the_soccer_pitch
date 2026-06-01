"""Agent loop module implementing the Look-Think-Act cycle.

Runs in a background thread, continuously polling game state from the Pitch
server, invoking the LLM for movement decisions, and submitting actions back
to the server. Any failure at any step results in the Brake_Action (safe stop).

Integrates agentic modules (episodic memory, planner, reflection engine,
strategy tracker, context assembler) into the cycle while maintaining exactly
one LLM call per cycle.
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
from context_assembler import assemble_agentic_context
from episodic_memory import Episode, EpisodicMemory
from memory_summary import summarize_memory
from planner import Plan, Planner
from reflection import ReflectionEngine
from strategy_tracker import PatternEntry, StrategyTracker
from spatial import analyze_game_state, format_spatial_summary


# Interval (in cycles) between strategy tracker analysis runs
_STRATEGY_ANALYSIS_INTERVAL = 10


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

    Integrates agentic modules:
    - EpisodicMemory: stores past game states, actions, and outcomes
    - Planner: evaluates and manages multi-step plans
    - ReflectionEngine: evaluates action effectiveness
    - StrategyTracker: tracks opponent patterns and produces adaptations
    - Context assembler: builds agentic context for the LLM prompt

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

        # Agentic module instances
        self.memory = EpisodicMemory()
        self.planner = Planner()
        self.reflection = ReflectionEngine()
        self.strategy_tracker = StrategyTracker()

        # Agentic state tracking
        self._cycle_counter: int = 0
        self._active_plan: Optional[Plan] = None
        self._previous_state: Optional[dict] = None
        self._previous_action: Optional[dict] = None

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
        """Execute one Look-Think-Act cycle with agentic module integration.

        Flow:
        1. Look: GET game state
        2. Post-look agentic processing: plan evaluation, reflection, memory
        3. Pre-think: assemble agentic context
        4. Think: single LLM call with enriched state + agentic context
        5. Act: POST action
        6. Post-act: record episode and pattern
        """
        # --- Look step ---
        game_state = self._look()
        if isinstance(game_state, IterationResult):
            return game_state

        # --- Post-look agentic processing ---
        self._cycle_counter += 1
        self._post_look_agentic(game_state)

        # --- Pre-think: assemble agentic context ---
        agentic_context = self._assemble_context()

        # --- Think step (single LLM call) ---
        action = self._think(game_state, agentic_context)
        if isinstance(action, IterationResult):
            return action

        # --- Act step ---
        self._act(action)

        # --- Post-act agentic processing ---
        self._post_act_agentic(game_state, action)

        return IterationResult(
            game_state=game_state,
            action=action,
            fallback_reason=None,
            error_details=None,
        )

    def _post_look_agentic(self, game_state: dict) -> None:
        """Run agentic processing after look step.

        - Advance active plan if one exists
        - Check plan abandonment conditions
        - Evaluate for new/replacement plan
        - Run reflection on previous action
        - Run strategy analysis periodically
        """
        # Plan advancement and evaluation
        if self._active_plan is not None and not self._active_plan.completed:
            self._active_plan = self.planner.advance(
                self._active_plan, game_state, self.team, self.position
            )

        # Check abandonment (including reflection-based abandonment)
        if self._active_plan is not None and not self._active_plan.completed:
            if self.planner.should_abandon(
                self._active_plan, game_state, self.team, self.position
            ):
                self._active_plan = None

        # Evaluate for new or replacement plan
        self._active_plan = self.planner.evaluate(
            game_state, self.team, self.position, self._active_plan
        )

        # Clear completed plans
        if self._active_plan is not None and self._active_plan.completed:
            self._active_plan = None

        # Reflection on previous action
        if self._previous_action is not None and self._previous_state is not None:
            reflection_result = self.reflection.evaluate(
                action=self._previous_action,
                expected_outcome={},
                actual_state=game_state,
                previous_state=self._previous_state,
            )

            if reflection_result is not None:
                # Update the most recent episode's effectiveness
                if len(self.memory) > 0:
                    recent = self.memory.get_recent(1)
                    if recent:
                        recent[0].effectiveness = reflection_result.effectiveness_score

                # Check reflection-based abandonment
                if (
                    reflection_result.should_abandon_plan
                    and self._active_plan is not None
                ):
                    self._active_plan = None

        # Run strategy analysis periodically
        if self._cycle_counter % _STRATEGY_ANALYSIS_INTERVAL == 0:
            self.strategy_tracker.analyze()

    def _assemble_context(self) -> str:
        """Assemble agentic context for the LLM prompt.

        Combines memory summary, current plan step, and adaptation hints
        into a single string respecting the 300-token budget.
        """
        # Memory summary
        memory_summary = summarize_memory(self.memory)

        # Current plan step (Req 3.9: include when plan active, Req 3.10: omit when no plan)
        plan_step: Optional[str] = None
        if self._active_plan is not None and not self._active_plan.completed:
            idx = self._active_plan.current_index
            if idx < len(self._active_plan.sub_goals):
                plan_step = self._active_plan.sub_goals[idx].description

        # Adaptation hints from strategy tracker
        adaptations = self.strategy_tracker.get_active_adaptations()
        adaptation_hints = [
            f"{a.counter_strategy} (confidence: {a.confidence:.0%})"
            for a in adaptations
        ]

        return assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
        )

    def _post_act_agentic(self, game_state: dict, action: ActionModel) -> None:
        """Run agentic processing after act step.

        - Store episode in memory
        - Record pattern in strategy tracker
        - Save state for next cycle's reflection
        """
        action_dict = {"dx": action.dx, "dy": action.dy, "kick": action.kick}

        # Store episode in memory
        episode = Episode(
            cycle=self._cycle_counter,
            game_state=game_state,
            action=action_dict,
            next_state_delta={},  # Will be computed by reflection in next cycle
            effectiveness=None,  # Will be filled by reflection in next cycle
        )
        self.memory.add(episode)

        # Record pattern in strategy tracker
        opponents = self._extract_opponent_positions(game_state)
        ball_pos = game_state.get("ball", {"x": 600.0, "y": 400.0})
        # Use previous reflection score if available, default to 0.5 (neutral)
        effectiveness = 0.5
        if len(self.memory) >= 2:
            recent = self.memory.get_recent(2)
            if recent[0].effectiveness is not None:
                effectiveness = recent[0].effectiveness

        pattern_entry = PatternEntry(
            opponent_positions=opponents,
            ball_position=ball_pos,
            effectiveness=effectiveness,
        )
        self.strategy_tracker.record(pattern_entry)

        # Save state for next cycle's reflection
        self._previous_state = game_state
        self._previous_action = action_dict

    def _extract_opponent_positions(self, game_state: dict) -> list[dict]:
        """Extract opponent player positions from game state."""
        players = game_state.get("players", {})
        opponent_team = "Blue" if self.team == "Red" else "Red"
        positions = []
        for key, pos in players.items():
            if key.startswith(f"{opponent_team}_"):
                positions.append({"x": pos.get("x", 0.0), "y": pos.get("y", 0.0)})
        return positions

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

    def _think(self, game_state: dict, agentic_context: str = "") -> "ActionModel | IterationResult":
        """Invoke LLM with system prompt + game state + agentic context.

        Maintains exactly one LLM call per cycle. Agentic context is appended
        after the spatial summary (Req 2.5).

        Returns ActionModel or IterationResult on failure.
        """
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

            # Append agentic context after spatial summary (Req 2.5)
            if agentic_context:
                enriched_state = enriched_state + "\n\n" + agentic_context

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
