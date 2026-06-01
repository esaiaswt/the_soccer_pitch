"""Coach Agent module for the multi-agent soccer team.

Contains the CoachMemory class for maintaining a rolling buffer of game state
snapshots, and the CoachAgent class for LLM-powered tactical instruction
generation.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6,
              7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

import time
from collections import deque
from threading import Event

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from team.config import TeamConfig
from team.debug_store import DebugStore
from team.instruction_store import CoachInstruction, InstructionStore
from team.logging_config import (
    get_logger,
    log_coach_instruction,
    log_decision_latency,
    log_token_usage,
    log_token_usage_unavailable,
)
from team.shared_state import SharedState
from team.strategy_tracker import StrategyTracker

# Required fields that every snapshot must contain to be stored in memory
REQUIRED_SNAPSHOT_FIELDS = ("ball", "players", "score", "time_left", "match_state")


class CoachMemory:
    """Rolling buffer of past game state snapshots for pattern detection.

    Maintains snapshots in chronological insertion order up to a configurable
    maximum size. When the buffer is full, the oldest snapshot is automatically
    discarded. Snapshots missing required fields are rejected with a warning log.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of snapshots to retain. Defaults to 50.
    """

    def __init__(self, max_size: int = 50) -> None:
        self._buffer: deque[dict] = deque(maxlen=max_size)
        self._max_size = max_size

    def add_snapshot(self, snapshot: dict) -> None:
        """Validate and add a game state snapshot to the memory buffer.

        The snapshot must contain all required fields: ball, players, score,
        time_left, and match_state. If any field is missing, the snapshot is
        rejected and a warning is logged.

        A ``received_at`` timestamp is added to the snapshot before storage.

        Parameters
        ----------
        snapshot : dict
            The game state snapshot to store.
        """
        missing_fields = [
            field for field in REQUIRED_SNAPSHOT_FIELDS if field not in snapshot
        ]
        if missing_fields:
            logger = get_logger()
            logger.warning(
                f"Snapshot rejected: missing required fields {missing_fields}",
                extra={
                    "agent_identity": "Coach",
                    "structured_context": f"missing_fields={missing_fields}",
                },
            )
            return

        entry = {**snapshot, "received_at": time.time()}
        self._buffer.append(entry)

    def get_history(self) -> list[dict]:
        """Return all snapshots in chronological order (oldest first).

        Returns
        -------
        list[dict]
            All stored snapshots ordered from oldest to newest.
        """
        return list(self._buffer)

    def get_recent(self, n: int) -> list[dict]:
        """Return the n most recent snapshots in chronological order.

        If n exceeds the number of stored snapshots, all snapshots are returned.

        Parameters
        ----------
        n : int
            Number of most recent snapshots to retrieve.

        Returns
        -------
        list[dict]
            The n most recent snapshots, ordered oldest first within the
            returned list.
        """
        if n <= 0:
            return []
        items = list(self._buffer)
        return items[-n:]

# Player positions that the Coach issues instructions to
PLAYER_POSITIONS = ("Goalkeeper", "Defender", "Midfielder", "Striker")

# System prompt for the Coach LLM
_COACH_SYSTEM_PROMPT = (
    "You are a tactical soccer coach. Analyze the current game state and recent "
    "history, then generate specific tactical instructions for each of your 4 players. "
    "Your team has a Goalkeeper, Defender, Midfielder, and Striker.\n\n"
    "Respond EXACTLY in this format (one instruction per line, no extra text):\n"
    "Goalkeeper: <tactical instruction for the goalkeeper>\n"
    "Defender: <tactical instruction for the defender>\n"
    "Midfielder: <tactical instruction for the midfielder>\n"
    "Striker: <tactical instruction for the striker>\n\n"
    "Each instruction should reference the player's role and provide actionable "
    "guidance based on the current game situation (ball position, score, time, "
    "player positions)."
)


class CoachAgent:
    """LLM-powered Coach Agent that generates tactical instructions for players.

    Runs as a thread target, looping at the configured coaching frequency.
    Each cycle: reads the current game state from SharedState, adds it to
    CoachMemory, invokes the Coach LLM to generate per-player instructions,
    and stores them in the InstructionStore.

    On LLM failure, logs the error, skips the cycle, and continues.

    Parameters
    ----------
    config : TeamConfig
        The team configuration (contains model name, API key, frequency).
    shared_state : SharedState
        Thread-safe container for the latest game state snapshot.
    instruction_store : InstructionStore
        Thread-safe store for Coach-to-Player instructions.
    stop_event : Event
        Threading event signaling the agent to shut down.
    debug_store : DebugStore | None
        Optional debug store for dashboard consumption.
    player_trackers : dict[str, StrategyTracker] | None
        Optional mapping from player position (e.g., "Striker") to that
        player's StrategyTracker. Used to aggregate adaptation data into
        the coaching prompt.
    """

    def __init__(
        self,
        config: TeamConfig,
        shared_state: SharedState,
        instruction_store: InstructionStore,
        stop_event: Event,
        debug_store: DebugStore | None = None,
        player_trackers: dict[str, StrategyTracker] | None = None,
    ) -> None:
        self._config = config
        self._shared_state = shared_state
        self._instruction_store = instruction_store
        self._stop_event = stop_event
        self._debug_store = debug_store
        self._player_trackers = player_trackers
        self._memory = CoachMemory(max_size=config.coach_memory_size)
        self._llm = ChatNVIDIA(
            model=config.coach_model,
            api_key=config.nvidia_api_key,
        )
        self._logger = get_logger()

    def run(self) -> None:
        """Thread target: loop at coaching frequency, generating instructions.

        Reads SharedState, updates CoachMemory, invokes the Coach LLM,
        parses the response into per-player instructions, and stores them.
        Uses stop_event.wait() for responsive shutdown between cycles.

        Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 7.1, 7.2, 7.3, 7.4
        """
        self._logger.info(
            "Coach Agent started",
            extra={
                "agent_identity": "Coach",
                "structured_context": (
                    f"model={self._config.coach_model} | "
                    f"frequency={self._config.coaching_frequency}s"
                ),
            },
        )

        while not self._stop_event.is_set():
            self._coaching_cycle()
            # Wait for the coaching frequency interval or until stop is signaled
            self._stop_event.wait(timeout=self._config.coaching_frequency)

        self._logger.info(
            "Coach Agent stopped",
            extra={"agent_identity": "Coach", "structured_context": ""},
        )

    def _coaching_cycle(self) -> None:
        """Execute a single coaching cycle: read state, invoke LLM, store instructions."""
        # Read current game state
        snapshot = self._shared_state.get_snapshot()
        if snapshot is None:
            self._logger.warning(
                "No game state available, skipping coaching cycle",
                extra={
                    "agent_identity": "Coach",
                    "structured_context": "reason=no_snapshot",
                },
            )
            return

        # Add snapshot to memory for pattern detection
        self._memory.add_snapshot(snapshot)

        # Build the prompt with current state and recent history
        prompt = self._build_prompt(snapshot)

        # Invoke LLM with error handling
        try:
            start_time = time.time()
            response = self._llm.invoke(prompt)
            latency_ms = (time.time() - start_time) * 1000

            # Log decision latency (Req 7.4)
            log_decision_latency("Coach", latency_ms)

            # Log token usage (Req 7.2, 7.3)
            self._log_token_usage(response)

            # Parse response into per-player instructions
            response_text = response.content if hasattr(response, "content") else str(response)
            instructions = self._parse_instructions(response_text)

            # Store instructions and log them
            now = time.time()
            instructions_for_debug: dict[str, str] = {}

            for position in PLAYER_POSITIONS:
                content = instructions.get(position, "")
                if not content:
                    continue

                instruction = CoachInstruction(
                    content=content,
                    timestamp=now,
                    target_position=position,
                )
                self._instruction_store.set_instruction(position, instruction)

                # Log each instruction (Req 7.1)
                log_coach_instruction(position, content)
                instructions_for_debug[position] = content

            # Update DebugStore with coach observations and instructions
            if self._debug_store is not None:
                recent_snapshots = self._memory.get_recent(5)
                self._debug_store.update_coach(
                    observations=recent_snapshots,
                    instructions=instructions_for_debug,
                )

        except Exception as exc:
            # Req 3.6: On LLM failure, log error, skip cycle, continue
            self._logger.error(
                f"Coach LLM invocation failed: {exc}",
                extra={
                    "agent_identity": "Coach",
                    "structured_context": (
                        f"error_type={type(exc).__name__} | "
                        f"match_state={snapshot.get('match_state', 'unknown')} | "
                        f"attempted_action=generate_instruction"
                    ),
                },
            )

    def _build_prompt(self, snapshot: dict) -> list[dict]:
        """Build the LLM prompt with game state and memory context.

        Parameters
        ----------
        snapshot : dict
            The current game state snapshot.

        Returns
        -------
        list[dict]
            Messages list for the LLM (system + human message).
        """
        recent_history = self._memory.get_recent(5)

        # Format current state
        team_color = self._config.team_color
        ball = snapshot.get("ball", {})
        score = snapshot.get("score", {})
        time_left = snapshot.get("time_left", "unknown")
        match_state = snapshot.get("match_state", "unknown")
        players = snapshot.get("players", {})

        # Filter to our team's players
        team_players = {
            k: v for k, v in players.items() if k.startswith(f"{team_color}_")
        }

        state_text = (
            f"Current Game State:\n"
            f"- Match State: {match_state}\n"
            f"- Time Left: {time_left}s\n"
            f"- Score: {score}\n"
            f"- Ball Position: x={ball.get('x', '?')}, y={ball.get('y', '?')}\n"
            f"- Your Team ({team_color}) Players:\n"
        )
        for player_name, pos in team_players.items():
            position_label = player_name.replace(f"{team_color}_", "")
            state_text += f"  - {position_label}: x={pos.get('x', '?')}, y={pos.get('y', '?')}\n"

        # Add recent history summary
        if len(recent_history) > 1:
            state_text += f"\nRecent History ({len(recent_history)} snapshots available):\n"
            oldest = recent_history[0]
            state_text += (
                f"- Oldest snapshot: ball at "
                f"x={oldest.get('ball', {}).get('x', '?')}, "
                f"y={oldest.get('ball', {}).get('y', '?')}\n"
            )

        # Append player adaptation section (Req 11.1, 11.2, 11.4)
        adaptation_section = self._build_adaptation_section()
        if adaptation_section:
            state_text += "\n" + adaptation_section

        from langchain_core.messages import HumanMessage, SystemMessage

        return [
            SystemMessage(content=_COACH_SYSTEM_PROMPT),
            HumanMessage(content=state_text),
        ]

    def _build_adaptation_section(self) -> str:
        """Aggregate player adaptation data into a prompt section.

        Collects active AdaptationRecords from all player StrategyTrackers,
        formats at most 1 sentence per player, detects shared opponent
        tendencies across multiple players, and limits the section to
        200 tokens (~800 characters).

        Does NOT directly modify player memories, plans, or trackers.

        Returns
        -------
        str
            The adaptation context section, or empty string if no data.
        """
        if not self._player_trackers:
            return ""

        # Collect per-player adaptation summaries and track pattern frequencies
        player_lines: list[str] = []
        pattern_counts: dict[str, list[str]] = {}  # pattern -> list of positions

        for position, tracker in self._player_trackers.items():
            adaptations = tracker.get_active_adaptations(max_count=2)
            if not adaptations:
                continue

            # Use the highest-confidence adaptation for the 1-sentence summary
            top = adaptations[0]
            player_lines.append(f"{position} reports: {top.observed_pattern}")

            # Track which positions report each pattern
            for adaptation in adaptations:
                pattern = adaptation.observed_pattern
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = []
                pattern_counts[pattern].append(position)

        if not player_lines:
            return ""

        # Build the section
        lines: list[str] = ["Player Adaptation Insights:"]
        lines.extend(player_lines)

        # Detect shared tendencies (2+ players report the same pattern)
        shared_instructions: list[str] = []
        for pattern, positions in pattern_counts.items():
            if len(positions) >= 2:
                positions_str = " and ".join(positions)
                shared_instructions.append(
                    f"COORDINATED: {positions_str} both report {pattern} — "
                    f"issue team-wide adjustment"
                )

        if shared_instructions:
            lines.append("Coordinated Instructions:")
            lines.extend(shared_instructions)

        # Join and enforce 200 token limit (~800 characters)
        max_chars = 800
        section = "\n".join(lines)

        if len(section) > max_chars:
            # Truncate by removing player lines from the end (keep coordinated instructions)
            section = section[:max_chars].rsplit("\n", 1)[0]

        return section

    def _parse_instructions(self, response_text: str) -> dict[str, str]:
        """Parse the LLM response into per-player instructions.

        Expects format:
            Goalkeeper: <instruction>
            Defender: <instruction>
            Midfielder: <instruction>
            Striker: <instruction>

        Falls back to assigning the full response to all players if parsing fails.

        Parameters
        ----------
        response_text : str
            The raw LLM response text.

        Returns
        -------
        dict[str, str]
            Mapping from position to instruction content.
        """
        instructions: dict[str, str] = {}

        for line in response_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for position in PLAYER_POSITIONS:
                prefix = f"{position}:"
                if line.startswith(prefix):
                    content = line[len(prefix):].strip()
                    if content:
                        instructions[position] = content
                    break

        # If we couldn't parse instructions for any position, use full response
        # for all positions as fallback (Req 3.4: no truncation)
        if not instructions:
            for position in PLAYER_POSITIONS:
                instructions[position] = response_text

        return instructions

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
            total_tokens = usage_metadata.get("total_tokens", prompt_tokens + completion_tokens)
            if prompt_tokens or completion_tokens:
                log_token_usage("Coach", prompt_tokens, completion_tokens, total_tokens)
                token_data_found = True

        # Try response_metadata as fallback
        if not token_data_found:
            response_metadata = getattr(response, "response_metadata", None)
            if response_metadata and isinstance(response_metadata, dict):
                usage = response_metadata.get("token_usage") or response_metadata.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                    if prompt_tokens or completion_tokens:
                        log_token_usage("Coach", prompt_tokens, completion_tokens, total_tokens)
                        token_data_found = True

        # Req 7.3: Log warning when token usage metadata unavailable
        if not token_data_found:
            log_token_usage_unavailable("Coach")
