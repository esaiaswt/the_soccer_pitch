"""Player Agent module for the multi-agent soccer team.

Contains the PlayerAgent class that runs an independent Look-Think-Act loop
for a single player position. Each player reads shared game state, optionally
incorporates Coach instructions, invokes a lightweight LLM for decision-making,
and posts movement/kick actions to the Pitch server.

Integrates agentic modules (episodic memory, planner, reflection engine,
strategy tracker, context assembler, signal bus, signal generator) into the
cycle while maintaining exactly one LLM call per cycle.

Requirements: 2.5, 3.2, 3.6, 3.9, 3.10, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6,
              5.1, 5.2, 5.3, 5.4, 5.5, 7.2, 7.3, 7.4, 7.5, 8.1, 9.1, 9.2, 10.2
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from threading import Event
from typing import Optional

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from team.config import TeamConfig
from team.context_assembler import assemble_agentic_context
from team.debug_store import DebugStore, PlayerDebugInfo
from team.episodic_memory import Episode, EpisodicMemory
from team.instruction_store import InstructionStore
from team.logging_config import (
    get_logger,
    log_agent_error,
    log_decision_latency,
    log_token_usage,
    log_token_usage_unavailable,
)
from team.memory_summary import summarize_memory
from team.planner import Plan, Planner, build_default_templates
from team.reflection import ReflectionEngine
from team.shared_state import SharedState
from team.signal_bus import Signal, SignalBus
from team.signal_generator import SignalGenerator
from team.strategy_tracker import PatternEntry, StrategyTracker

# LLM invocation timeout in seconds (Req 5.2)
_LLM_TIMEOUT_SECONDS = 10

# Player loop cycle time in seconds (Req 4.2)
# Increased from 1.5 to 3.0 to reduce API pressure and avoid 429 rate limits
# when running 4 players + 1 coach against NVIDIA NIM free tier.
_LOOP_CYCLE_SECONDS = 3.0

# Backoff time in seconds after a 429 rate limit error
_RATE_LIMIT_BACKOFF_SECONDS = 5.0

# Interval (in cycles) between strategy tracker analysis runs
_STRATEGY_ANALYSIS_INTERVAL = 10


def _build_system_prompt(position: str, team_color: str) -> str:
    """Build the system prompt for a Player Agent LLM.

    Parameters
    ----------
    position : str
        The player's position (e.g., "Striker", "Goalkeeper").
    team_color : str
        The team color (e.g., "Red", "Blue").

    Returns
    -------
    str
        The system prompt instructing the LLM how to respond.
    """
    # Red attacks right (toward x=1200), Blue attacks left (toward x=0)
    if team_color == "Red":
        attack_direction = "RIGHT (increasing x)"
        own_goal = "LEFT side (x=0)"
        opponent_goal = "RIGHT side (x=1200)"
    else:
        attack_direction = "LEFT (decreasing x)"
        own_goal = "RIGHT side (x=1200)"
        opponent_goal = "LEFT side (x=0)"

    # Position-specific strategy
    if position == "Goalkeeper":
        role_strategy = (
            f"GOALKEEPER STRATEGY:\n"
            f"- Stay near your goal line ({own_goal}) and between the goalposts (y=325 to y=525)\n"
            f"- Position yourself between the ball and your goal to block shots\n"
            f"- After catching the ball (ball stopped near you with 'In kick range: YES'):\n"
            f"  → Look at 'Nearest teammate direction' in the spatial analysis\n"
            f"  → Position yourself so the teammate is AHEAD of you (in your attack direction)\n"
            f"  → Then kick=true to distribute to your teammate\n"
            f"- NEVER kick toward your own goal — always distribute forward or sideways\n"
            f"- If no teammate direction is shown, kick toward the center of the pitch"
        )
    elif position == "Defender":
        role_strategy = (
            f"DEFENDER STRATEGY:\n"
            f"- Stay between the ball and your own goal\n"
            f"- Intercept opponents approaching your goal area\n"
            f"- When you have the ball, pass forward (kick toward attack direction)\n"
            f"- Don't venture too far from your defensive third"
        )
    elif position == "Midfielder":
        role_strategy = (
            f"MIDFIELDER STRATEGY:\n"
            f"- Control the center of the pitch\n"
            f"- Move toward the ball aggressively to win possession\n"
            f"- When you have the ball, advance toward the opponent's goal\n"
            f"- Support both attack and defense depending on ball position"
        )
    else:  # Striker
        role_strategy = (
            f"STRIKER STRATEGY:\n"
            f"- Stay in the attacking half, close to the opponent's goal\n"
            f"- Chase the ball aggressively when it's in the attacking third\n"
            f"- Position yourself behind the ball (between ball and your own goal) to shoot\n"
            f"- When 'Behind ball: YES' and 'In kick range: YES' → SHOOT (kick=true)"
        )

    return (
        f"You are a soccer player on the {team_color} team playing as {position}.\n"
        f"The pitch is 1200 pixels wide and 800 pixels tall.\n"
        f"Your team attacks {attack_direction}. Your goal is on the {own_goal}. "
        f"The opponent's goal is on the {opponent_goal}.\n\n"
        f"COORDINATE SYSTEM:\n"
        f"- x=0 is the LEFT edge, x=1200 is the RIGHT edge\n"
        f"- y=0 is the TOP edge, y=800 is the BOTTOM edge\n"
        f"- The center of the pitch is x=600, y=400\n\n"
        f"YOUR OUTPUT — respond with EXACTLY one line:\n"
        f"dx=<float> dy=<float> kick=<true|false>\n\n"
        f"MOVEMENT RULES:\n"
        f"- dx and dy are between -1.0 and 1.0 (use large values like 0.8 or 1.0 for decisive movement)\n"
        f"- dx=1.0 moves RIGHT, dx=-1.0 moves LEFT\n"
        f"- dy=1.0 moves DOWN, dy=-1.0 moves UP\n\n"
        f"SHOOTING/KICKING STRATEGY (CRITICAL):\n"
        f"- A kick pushes the ball AWAY from you in the direction you're facing\n"
        f"- To score, you MUST be BEHIND the ball (between ball and your own goal) before kicking\n"
        f"- Check 'Behind ball' in the spatial analysis: if YES and 'In kick range' is YES → kick=true\n"
        f"- If in kick range but NOT behind ball → do NOT kick, move behind the ball first\n"
        f"- Use 'Shoot direction' to understand where the ball will go if kicked\n\n"
        f"{role_strategy}\n\n"
        f"GENERAL:\n"
        f"- Follow the RECOMMENDATION line in the spatial analysis\n"
        f"- Use the 'Direction to ball' hint to choose dx/dy values\n"
        f"- Be DECISIVE — use values of 0.5 to 1.0, not tiny values like 0.1\n"
        f"- Respond with ONLY the dx, dy, kick line — nothing else"
    )


def _parse_llm_response(response_text: str) -> tuple[float, float, bool]:
    """Parse the LLM response to extract dx, dy, and kick values.

    Attempts to find dx=<val> dy=<val> kick=<val> patterns in the response.
    Values are clamped to [-1, 1] for dx and dy.

    Parameters
    ----------
    response_text : str
        The raw LLM response text.

    Returns
    -------
    tuple[float, float, bool]
        A tuple of (dx, dy, kick) with dx/dy clamped to [-1, 1].

    Raises
    ------
    ValueError
        If the response cannot be parsed into valid action values.
    """
    # Try to extract dx, dy, kick from the response
    dx_match = re.search(r"dx\s*=\s*([+-]?\d*\.?\d+)", response_text)
    dy_match = re.search(r"dy\s*=\s*([+-]?\d*\.?\d+)", response_text)
    kick_match = re.search(r"kick\s*=\s*<?(\btrue\b|\bfalse\b)>?", response_text, re.IGNORECASE)

    if not dx_match or not dy_match or not kick_match:
        raise ValueError(f"Cannot parse action from LLM response: {response_text!r}")

    dx = float(dx_match.group(1))
    dy = float(dy_match.group(1))
    kick = kick_match.group(1).lower() == "true"

    # Clamp dx and dy to [-1, 1]
    dx = max(-1.0, min(1.0, dx))
    dy = max(-1.0, min(1.0, dy))

    return dx, dy, kick


class PlayerAgent:
    """LLM-powered Player Agent running an independent Look-Think-Act loop.

    Designed to be used as a thread target via
    ``threading.Thread(target=player.run)``.

    Each cycle:
    1. LOOK: Read the current game state from SharedState.
    2. POST-LOOK: Evaluate plan, run reflection, store episode, clear signals on dead ball.
    3. THINK: Invoke the Player LLM with game state + Coach instruction + agentic context.
    4. ACT: Parse the LLM response and POST the action to the Pitch server.
    5. POST-ACT: Record episode, record pattern, generate and publish signal.

    On LLM timeout or error, submits a Brake_Action (dx=0, dy=0, kick=false).

    Integrates agentic modules:
    - EpisodicMemory: stores past game states, actions, and outcomes
    - Planner: evaluates and manages multi-step plans
    - ReflectionEngine: evaluates action effectiveness
    - StrategyTracker: tracks opponent patterns and produces adaptations
    - SignalBus: inter-player communication (shared across all PlayerAgents)
    - SignalGenerator: generates signals from plan and game state

    Parameters
    ----------
    config : TeamConfig
        The team configuration (model, API key, pitch server, etc.).
    position : str
        The player's position: "Goalkeeper", "Defender", "Midfielder", or "Striker".
    shared_state : SharedState
        Thread-safe container for the latest game state snapshot.
    instruction_store : InstructionStore
        Thread-safe store for Coach-to-Player instructions.
    stop_event : Event
        Threading event signaling the agent to shut down.
    debug_store : DebugStore
        Debug store for dashboard consumption.
    signal_bus : SignalBus | None
        Shared signal bus for inter-player communication. Optional for
        backward compatibility; when None, signal features are disabled.
    """

    def __init__(
        self,
        config: TeamConfig,
        position: str,
        shared_state: SharedState,
        instruction_store: InstructionStore,
        stop_event: Event,
        debug_store: DebugStore,
        signal_bus: Optional[SignalBus] = None,
    ) -> None:
        self._config = config
        self._position = position
        self._shared_state = shared_state
        self._instruction_store = instruction_store
        self._stop_event = stop_event
        self._debug_store = debug_store
        self._signal_bus = signal_bus
        self._agent_identity = f"Player_{position}"
        self._llm = ChatNVIDIA(
            model=config.player_model,
            api_key=config.nvidia_api_key,
        )
        self._action_url = (
            f"http://{config.pitch_host}:{config.pitch_port}/api/action"
        )
        self._system_prompt = _build_system_prompt(position, config.team_color)
        self._logger = get_logger()

        # Agentic module instances
        self._memory = EpisodicMemory()
        self._planner = Planner(build_default_templates())
        self._reflection = ReflectionEngine()
        self._strategy_tracker = StrategyTracker()
        self._signal_generator = SignalGenerator()

        # Agentic state tracking
        self._cycle_counter: int = 0
        self._active_plan: Optional[Plan] = None
        self._previous_state: Optional[dict] = None
        self._previous_action: Optional[dict] = None

    def run(self) -> None:
        """Thread target: continuous Look-Think-Act loop at 1.5s cycle.

        Reads SharedState, optionally includes Coach instruction, invokes
        the Player LLM, parses the response into an action, and POSTs it
        to the Pitch server. On failure, submits a Brake_Action.

        Uses stop_event.wait(1.5) for responsive shutdown between cycles.

        Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2,
                   5.3, 5.4, 5.5, 7.2, 7.3, 7.4, 7.5
        """
        self._logger.info(
            f"{self._agent_identity} started",
            extra={
                "agent_identity": self._agent_identity,
                "structured_context": (
                    f"position={self._position} | "
                    f"model={self._config.player_model}"
                ),
            },
        )

        while not self._stop_event.is_set():
            self._loop_iteration()
            # Wait 1.5s or until stop is signaled (Req 4.2)
            self._stop_event.wait(timeout=_LOOP_CYCLE_SECONDS)

        self._logger.info(
            f"{self._agent_identity} stopped",
            extra={
                "agent_identity": self._agent_identity,
                "structured_context": "",
            },
        )

    def _loop_iteration(self) -> None:
        """Execute a single Look-Think-Act iteration with agentic integration.

        Flow:
        1. Look: read game state from SharedState
        2. Post-look: increment cycle, evaluate plan, run reflection, detect dead ball
        3. Pre-think: read signals, assemble agentic context
        4. Think: single LLM call with enriched state + agentic context
        5. Act: POST action to Pitch server
        6. Post-act: record episode, record pattern, generate and publish signal
        """
        # --- LOOK: Read game state ---
        snapshot = self._shared_state.get_snapshot()
        if snapshot is None:
            # No state available yet, submit brake and continue
            self._submit_brake_action()
            self._update_debug(None, self._make_brake_dict(), None)
            return

        # --- Check for stale state (Req 5.4) ---
        last_update = self._shared_state.get_last_update_time()
        if last_update is not None:
            staleness = time.time() - last_update
            if staleness > 2 * self._config.poll_interval:
                self._logger.warning(
                    "SharedState stale, using last available snapshot",
                    extra={
                        "agent_identity": self._agent_identity,
                        "structured_context": (
                            f"staleness_s={staleness:.2f} | "
                            f"threshold={2 * self._config.poll_interval:.2f}"
                        ),
                    },
                )
        # Use whatever snapshot is available (Req 5.4: use last available)

        # --- POST-LOOK: Agentic processing (Req 3.2, 3.6, 5.1, 5.5) ---
        self._cycle_counter += 1
        self._post_look_agentic(snapshot)

        # --- PRE-THINK: Read signals and assemble agentic context ---
        agentic_context = self._assemble_agentic_context()

        # --- Read Coach Instruction (Req 4.4, 5.1, 5.5) ---
        instruction_text = self._get_valid_instruction()

        # --- THINK: Invoke LLM (exactly one call per cycle - Req 9.1, 9.2) ---
        messages = self._build_messages(snapshot, instruction_text, agentic_context)
        dx, dy, kick = self._invoke_llm(messages, snapshot)

        # --- ACT: Post action to Pitch server ---
        action_payload = {
            "team": self._config.team_color,
            "position": self._position,
            "vector": {"dx": dx, "dy": dy},
            "kick": kick,
            "agent_name": self._config.agent_name,
        }
        self._post_action(action_payload, snapshot)

        # --- POST-ACT: Agentic processing ---
        action_dict = {"dx": dx, "dy": dy, "kick": kick}
        self._post_act_agentic(snapshot, action_dict)

        # --- Update DebugStore ---
        self._update_debug(snapshot, action_dict, instruction_text)

    # ------------------------------------------------------------------
    # Agentic integration methods
    # ------------------------------------------------------------------

    def _post_look_agentic(self, snapshot: dict) -> None:
        """Run agentic processing after look step.

        - Advance active plan if one exists
        - Check plan abandonment conditions
        - Evaluate for new/replacement plan
        - Run reflection on previous action
        - Clear SignalBus on dead ball detection
        - Run strategy analysis periodically

        All logic executes in Python without LLM calls (Req 3.6, 5.6, 9.1).
        """
        # Plan advancement and evaluation (Req 3.2)
        if self._active_plan is not None and not self._active_plan.completed:
            self._active_plan = self._planner.advance(
                self._active_plan, snapshot, self._config.team_color, self._position
            )

        # Check abandonment (including reflection-based abandonment)
        if self._active_plan is not None and not self._active_plan.completed:
            if self._planner.should_abandon(
                self._active_plan, snapshot, self._config.team_color, self._position
            ):
                self._active_plan = None

        # Evaluate for new or replacement plan
        self._active_plan = self._planner.evaluate(
            snapshot, self._config.team_color, self._position, self._active_plan
        )

        # Clear completed plans
        if self._active_plan is not None and self._active_plan.completed:
            self._active_plan = None

        # Reflection on previous action (Req 5.1, 5.5)
        if self._previous_action is not None and self._previous_state is not None:
            reflection_result = self._reflection.evaluate(
                action=self._previous_action,
                expected_outcome={},
                actual_state=snapshot,
                previous_state=self._previous_state,
            )

            if reflection_result is not None:
                # Update the most recent episode's effectiveness
                if len(self._memory) > 0:
                    recent = self._memory.get_recent(1)
                    if recent:
                        recent[0].effectiveness = reflection_result.effectiveness_score

                # Check reflection-based abandonment
                if (
                    reflection_result.should_abandon_plan
                    and self._active_plan is not None
                ):
                    self._active_plan = None

        # Clear SignalBus on dead ball detection (Req 7.4, 8.5)
        if self._signal_bus is not None:
            is_dead_ball = snapshot.get("dead_ball", False) or snapshot.get(
                "is_dead_ball", False
            )
            if is_dead_ball:
                self._signal_bus.clear()

        # Run strategy analysis periodically
        if self._cycle_counter % _STRATEGY_ANALYSIS_INTERVAL == 0:
            self._strategy_tracker.analyze()

    def _assemble_agentic_context(self) -> str:
        """Assemble agentic context for the LLM prompt.

        Combines memory summary, current plan step, adaptation hints,
        and teammate signals into a single string respecting the 300-token budget.

        Returns
        -------
        str
            Assembled agentic context string, or empty string if nothing to add.
        """
        # Memory summary (Req 2.5)
        memory_summary = summarize_memory(self._memory)

        # Current plan step (Req 3.9: include when plan active, Req 3.10: omit when no plan)
        plan_step: Optional[str] = None
        if self._active_plan is not None and not self._active_plan.completed:
            idx = self._active_plan.current_index
            if idx < len(self._active_plan.sub_goals):
                plan_step = self._active_plan.sub_goals[idx].description

        # Adaptation hints from strategy tracker
        adaptations = self._strategy_tracker.get_active_adaptations()
        adaptation_hints = [
            f"{a.counter_strategy} (confidence: {a.confidence:.0%})"
            for a in adaptations
        ]

        # Read signals from SignalBus (exclude own position) (Req 7.2)
        signals: Optional[list[str]] = None
        if self._signal_bus is not None:
            raw_signals = self._signal_bus.read_all(exclude_position=self._position)
            if raw_signals:
                signals = [
                    f"{s.sender_position}: {s.signal_type} ({s.payload})"
                    for s in raw_signals
                ]

        return assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            signals=signals,
        )

    def _post_act_agentic(self, snapshot: dict, action_dict: dict) -> None:
        """Run agentic processing after act step.

        - Store episode in memory
        - Record pattern in strategy tracker
        - Generate and publish signal via SignalGenerator
        - Save state for next cycle's reflection

        All logic executes in Python without LLM calls (Req 9.1).
        """
        # Store episode in memory
        episode = Episode(
            cycle=self._cycle_counter,
            game_state=snapshot,
            action=action_dict,
            next_state_delta={},  # Will be computed by reflection in next cycle
            effectiveness=None,  # Will be filled by reflection in next cycle
        )
        self._memory.add(episode)

        # Record pattern in strategy tracker
        opponents = self._extract_opponent_positions(snapshot)
        ball_pos = snapshot.get("ball", {"x": 600.0, "y": 400.0})
        # Use previous reflection score if available, default to 0.5 (neutral)
        effectiveness = 0.5
        if len(self._memory) >= 2:
            recent = self._memory.get_recent(2)
            if recent[0].effectiveness is not None:
                effectiveness = recent[0].effectiveness

        pattern_entry = PatternEntry(
            opponent_positions=opponents,
            ball_position=ball_pos,
            effectiveness=effectiveness,
        )
        self._strategy_tracker.record(pattern_entry)

        # Generate and publish signal via SignalGenerator (Req 8.1)
        if self._signal_bus is not None:
            signal = self._signal_generator.generate(
                plan=self._active_plan,
                game_state=snapshot,
                team=self._config.team_color,
                position=self._position,
            )
            if signal is not None:
                self._signal_bus.publish(signal)

        # Save state for next cycle's reflection
        self._previous_state = snapshot
        self._previous_action = action_dict

    def _extract_opponent_positions(self, game_state: dict) -> list[dict]:
        """Extract opponent player positions from game state.

        Parameters
        ----------
        game_state : dict
            The current game state snapshot.

        Returns
        -------
        list[dict]
            List of opponent position dicts with 'x' and 'y' keys.
        """
        players = game_state.get("players", {})
        opponent_team = "Blue" if self._config.team_color == "Red" else "Red"
        positions = []
        for key, pos in players.items():
            if key.startswith(f"{opponent_team}_"):
                positions.append({"x": pos.get("x", 0.0), "y": pos.get("y", 0.0)})
        return positions

    def _get_valid_instruction(self) -> str | None:
        """Retrieve the latest Coach instruction if it's not stale.

        An instruction is considered stale if its timestamp is older than
        3 × coaching_frequency from the current time (Req 5.1).

        Returns
        -------
        str | None
            The instruction content if valid and fresh, or None if no
            instruction exists or it's stale.
        """
        instruction = self._instruction_store.get_instruction(self._position)
        if instruction is None:
            # Req 4.5: No instruction received yet, operate without
            return None

        # Req 5.1: Check staleness
        staleness_threshold = 3 * self._config.coaching_frequency
        age = time.time() - instruction.timestamp
        if age > staleness_threshold:
            # Instruction is stale, exclude from context
            self._logger.info(
                "Coach instruction stale, operating without",
                extra={
                    "agent_identity": self._agent_identity,
                    "structured_context": (
                        f"instruction_age_s={age:.2f} | "
                        f"threshold_s={staleness_threshold:.2f}"
                    ),
                },
            )
            return None

        # Req 5.5: Fresh instruction available, include it
        return instruction.content

    def _build_messages(
        self, snapshot: dict, instruction_text: str | None, agentic_context: str = ""
    ) -> list:
        """Build the LLM message list with game state and pre-computed spatial analysis.

        Includes directional hints, shooting alignment check, a recommendation,
        and agentic context (memory + plan + adaptations + signals) appended
        after the spatial analysis block (Req 2.5).

        Parameters
        ----------
        snapshot : dict
            The current game state snapshot.
        instruction_text : str | None
            The Coach instruction content, or None if unavailable/stale.
        agentic_context : str
            Assembled agentic context to append after spatial analysis.

        Returns
        -------
        list
            Messages list for the LLM (system + human message).
        """
        # Format game state for the player
        ball = snapshot.get("ball", {})
        score = snapshot.get("score", {})
        time_left = snapshot.get("time_left", "unknown")
        match_state = snapshot.get("match_state", "unknown")
        players = snapshot.get("players", {})

        team_color = self._config.team_color

        # Find our own position
        my_key = f"{team_color}_{self._position}"
        my_pos = players.get(my_key, {})
        my_x = my_pos.get("x", 600.0)
        my_y = my_pos.get("y", 400.0)
        ball_x = ball.get("x", 600.0)
        ball_y = ball.get("y", 400.0)

        # --- Spatial Analysis (pre-computed for the LLM) ---

        # Direction to ball
        diff_x = ball_x - my_x
        diff_y = ball_y - my_y
        ball_distance = (diff_x**2 + diff_y**2) ** 0.5

        if ball_distance > 0:
            ball_dx = round(diff_x / ball_distance, 2)
            ball_dy = round(diff_y / ball_distance, 2)
        else:
            ball_dx = 0.0
            ball_dy = 0.0

        in_kick_range = ball_distance <= 30

        # Goal positions
        if team_color == "Red":
            goal_x, goal_y = 1200.0, 400.0  # Attack Blue's goal (right)
        else:
            goal_x, goal_y = 0.0, 400.0  # Attack Red's goal (left)

        # Direction from ball to opponent goal (shoot direction)
        shoot_diff_x = goal_x - ball_x
        shoot_diff_y = goal_y - ball_y
        shoot_dist = (shoot_diff_x**2 + shoot_diff_y**2) ** 0.5
        if shoot_dist > 0:
            shoot_dx = round(shoot_diff_x / shoot_dist, 2)
            shoot_dy = round(shoot_diff_y / shoot_dist, 2)
        else:
            shoot_dx = 0.0
            shoot_dy = 0.0

        # Alignment check: is player behind the ball relative to the goal?
        if team_color == "Red":
            is_behind_ball = my_x < ball_x  # Player left of ball, kick goes right
        else:
            is_behind_ball = my_x > ball_x  # Player right of ball, kick goes left

        # Goal distance from player
        goal_distance = ((goal_x - my_x)**2 + (goal_y - my_y)**2) ** 0.5

        # Find nearest teammate (for goalkeeper distribution)
        nearest_teammate_name = None
        nearest_teammate_dist = float("inf")
        nearest_teammate_dx = 0.0
        nearest_teammate_dy = 0.0

        for player_name, pos in players.items():
            # Only consider teammates (same team prefix), skip self
            if not player_name.startswith(f"{team_color}_"):
                continue
            if player_name == my_key:
                continue
            px = pos.get("x", 0.0)
            py = pos.get("y", 0.0)
            tdx = px - my_x
            tdy = py - my_y
            tdist = (tdx**2 + tdy**2) ** 0.5
            if tdist < nearest_teammate_dist and tdist > 0:
                nearest_teammate_dist = tdist
                nearest_teammate_name = player_name.replace(f"{team_color}_", "")
                nearest_teammate_dx = round(tdx / tdist, 2)
                nearest_teammate_dy = round(tdy / tdist, 2)

        # Generate recommendation
        if in_kick_range and is_behind_ball:
            if self._position == "Goalkeeper" and nearest_teammate_name:
                recommendation = (
                    f"DISTRIBUTE! Kick toward {nearest_teammate_name} "
                    f"(direction: dx={nearest_teammate_dx}, dy={nearest_teammate_dy}). "
                    f"Position yourself so the teammate is ahead, then kick=true."
                )
            else:
                recommendation = "KICK NOW! You are behind the ball and in range. Set kick=true."
        elif in_kick_range and not is_behind_ball:
            recommendation = (
                f"DO NOT KICK. Move behind the ball first. "
                f"Use shoot direction dx={shoot_dx}, dy={shoot_dy} to reposition."
            )
        elif ball_distance < 100:
            recommendation = f"Close to ball! Move toward it: dx={ball_dx}, dy={ball_dy}"
        else:
            recommendation = f"Move toward ball: dx={ball_dx}, dy={ball_dy}"

        # Build the message
        state_text = (
            f"Game State:\n"
            f"- Match: {match_state}\n"
            f"- Time Left: {time_left}s\n"
            f"- Score: {score}\n\n"
            f"--- SPATIAL ANALYSIS (pre-computed) ---\n"
            f"Your position ({self._position}): x={my_x:.0f}, y={my_y:.0f}\n"
            f"Ball: x={ball_x:.0f}, y={ball_y:.0f}\n"
            f"Ball distance: {ball_distance:.0f}px\n"
            f"Direction to ball: dx={ball_dx}, dy={ball_dy}\n"
            f"In kick range (<30px): {'YES' if in_kick_range else 'NO'}\n"
            f"Behind ball (good shooting position): {'YES' if is_behind_ball else 'NO'}\n"
            f"Opponent goal distance: {goal_distance:.0f}px\n"
            f"Shoot direction (ball→goal): dx={shoot_dx}, dy={shoot_dy}\n"
            f"{f'Nearest teammate: {nearest_teammate_name} ({nearest_teammate_dist:.0f}px away, direction: dx={nearest_teammate_dx}, dy={nearest_teammate_dy})' if nearest_teammate_name else 'Nearest teammate: none visible'}\n"
            f"RECOMMENDATION: {recommendation}\n"
            f"---\n\n"
            f"All players:\n"
        )
        for player_name, pos in players.items():
            state_text += f"  - {player_name}: x={pos.get('x', '?'):.0f}, y={pos.get('y', '?'):.0f}\n"

        # Append agentic context after spatial analysis (Req 2.5)
        if agentic_context:
            state_text += f"\n--- AGENTIC CONTEXT ---\n{agentic_context}\n---\n"

        # Include Coach instruction if available (Req 4.4)
        if instruction_text is not None:
            state_text += (
                f"\nCoach Advisory (consider but make your own decision):\n"
                f"{instruction_text}\n"
            )

        return [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=state_text),
        ]

    def _invoke_llm(
        self, messages: list, snapshot: dict
    ) -> tuple[float, float, bool]:
        """Invoke the Player LLM with a 10-second timeout.

        On timeout or error, returns Brake_Action values (0, 0, False).

        Parameters
        ----------
        messages : list
            The LLM messages to send.
        snapshot : dict
            The current game state (for error logging context).

        Returns
        -------
        tuple[float, float, bool]
            A tuple of (dx, dy, kick). Returns (0, 0, False) on failure.
        """
        match_state = snapshot.get("match_state", "unknown") if snapshot else "unknown"

        try:
            start_time = time.time()

            # Use ThreadPoolExecutor for timeout (Req 5.2)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._llm.invoke, messages)
                try:
                    response = future.result(timeout=_LLM_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    # Req 5.2: LLM timeout, submit Brake_Action
                    log_agent_error(
                        agent_identity=self._agent_identity,
                        error_type="TimeoutError",
                        match_state=match_state,
                        attempted_action="llm_invoke",
                        error_details=f"LLM invocation timed out after {_LLM_TIMEOUT_SECONDS}s",
                    )
                    return 0.0, 0.0, False

            latency_ms = (time.time() - start_time) * 1000

            # Log decision latency (Req 7.4)
            log_decision_latency(self._agent_identity, latency_ms)

            # Log token usage (Req 7.2, 7.3)
            self._log_token_usage(response)

            # Parse response (Req 4.6: player makes final decision from LLM output)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            dx, dy, kick = _parse_llm_response(response_text)
            return dx, dy, kick

        except concurrent.futures.TimeoutError:
            # Redundant safety net for timeout
            log_agent_error(
                agent_identity=self._agent_identity,
                error_type="TimeoutError",
                match_state=match_state,
                attempted_action="llm_invoke",
                error_details=f"LLM invocation timed out after {_LLM_TIMEOUT_SECONDS}s",
            )
            return 0.0, 0.0, False

        except ValueError as exc:
            # Parse error — could not extract action from LLM response
            log_agent_error(
                agent_identity=self._agent_identity,
                error_type="ValueError",
                match_state=match_state,
                attempted_action="parse_llm_response",
                error_details=str(exc),
            )
            return 0.0, 0.0, False

        except Exception as exc:
            # Req 5.3: Any other LLM error, submit Brake_Action
            log_agent_error(
                agent_identity=self._agent_identity,
                error_type=type(exc).__name__,
                match_state=match_state,
                attempted_action="llm_invoke",
                error_details=str(exc),
            )
            # Back off on rate limit (429) to let the quota recover
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                self._stop_event.wait(timeout=_RATE_LIMIT_BACKOFF_SECONDS)
            return 0.0, 0.0, False

    def _post_action(self, action_payload: dict, snapshot: dict | None) -> None:
        """POST the action to the Pitch server.

        Parameters
        ----------
        action_payload : dict
            The action payload to send.
        snapshot : dict | None
            The current game state (for error logging context).
        """
        try:
            response = requests.post(
                self._action_url,
                json=action_payload,
                timeout=5,
            )
            response.raise_for_status()
        except Exception as exc:
            match_state = (
                snapshot.get("match_state", "unknown") if snapshot else "unknown"
            )
            log_agent_error(
                agent_identity=self._agent_identity,
                error_type=type(exc).__name__,
                match_state=match_state,
                attempted_action="post_action",
                error_details=f"Failed to POST action to {self._action_url}: {exc}",
            )

    def _submit_brake_action(self) -> None:
        """Submit a Brake_Action (dx=0, dy=0, kick=false) to the Pitch server."""
        brake_payload = {
            "team": self._config.team_color,
            "position": self._position,
            "vector": {"dx": 0, "dy": 0},
            "kick": False,
            "agent_name": self._config.agent_name,
        }
        self._post_action(brake_payload, None)

    def _make_brake_dict(self) -> dict:
        """Return a brake action dict for debug purposes."""
        return {"dx": 0, "dy": 0, "kick": False}

    def _update_debug(
        self,
        snapshot: dict | None,
        action: dict,
        instruction_text: str | None,
    ) -> None:
        """Update the DebugStore with the latest player state.

        Parameters
        ----------
        snapshot : dict | None
            The game state used for this iteration.
        action : dict
            The action submitted (dx, dy, kick).
        instruction_text : str | None
            The Coach instruction used (or None).
        """
        debug_info = PlayerDebugInfo(
            latest_state=snapshot,
            latest_action=action,
            latest_instruction=instruction_text,
            last_update=time.time(),
        )
        self._debug_store.update_player(self._position, debug_info)

    def _log_token_usage(self, response: object) -> None:
        """Extract and log token usage from the LLM response metadata.

        Checks response_metadata and usage_metadata attributes for token counts.
        Logs a warning if token usage metadata is unavailable.

        Parameters
        ----------
        response : object
            The LLM response object (AIMessage from LangChain).
        """
        token_data_found = False

        # Try usage_metadata first (newer LangChain pattern)
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata and isinstance(usage_metadata, dict):
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get(
                "total_tokens", prompt_tokens + completion_tokens
            )
            if prompt_tokens or completion_tokens:
                log_token_usage(
                    self._agent_identity, prompt_tokens, completion_tokens, total_tokens
                )
                token_data_found = True

        # Try response_metadata as fallback
        if not token_data_found:
            response_metadata = getattr(response, "response_metadata", None)
            if response_metadata and isinstance(response_metadata, dict):
                usage = response_metadata.get("token_usage") or response_metadata.get(
                    "usage", {}
                )
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get(
                        "total_tokens", prompt_tokens + completion_tokens
                    )
                    if prompt_tokens or completion_tokens:
                        log_token_usage(
                            self._agent_identity,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                        )
                        token_data_found = True

        # Req 7.3: Log warning when token usage metadata unavailable
        if not token_data_found:
            log_token_usage_unavailable(self._agent_identity)
