"""Team Orchestrator — manages lifecycle of all agent threads.

Creates, starts, monitors, and stops the State Poller, Coach Agent, and
four Player Agent threads. Each instance maintains its own state containers
(SharedState, InstructionStore, DebugStore) to support running two team
instances concurrently without shared mutable state.

Requirements: 4.1, 6.2, 8.1
"""

from __future__ import annotations

import threading

from team.coach_agent import CoachAgent
from team.config import TeamConfig
from team.debug_store import DebugStore
from team.instruction_store import InstructionStore
from team.logging_config import get_logger, setup_logging
from team.player_agent import PlayerAgent
from team.shared_state import SharedState
from team.signal_bus import SignalBus
from team.state_poller import StatePoller

# Player positions that the orchestrator launches threads for
PLAYER_POSITIONS = ("Goalkeeper", "Defender", "Midfielder", "Striker")


class TeamOrchestrator:
    """Manages the lifecycle of all agent threads for a single team instance.

    Each instance creates its own SharedState, InstructionStore, and DebugStore
    so that two TeamOrchestrator instances (Red and Blue) can run concurrently
    on the same machine without shared mutable state.

    Parameters
    ----------
    config : TeamConfig
        The team configuration (contains all settings for agents and polling).
    """

    def __init__(self, config: TeamConfig) -> None:
        self._config = config
        self._threads: list[threading.Thread] = []
        self._stop_event: threading.Event | None = None
        self._shared_state: SharedState | None = None
        self._instruction_store: InstructionStore | None = None
        self._debug_store: DebugStore | None = None
        self._signal_bus: SignalBus | None = None

    @property
    def debug_store(self) -> DebugStore | None:
        """Return the DebugStore for dashboard access."""
        return self._debug_store

    @property
    def instruction_store(self) -> InstructionStore | None:
        """Return the InstructionStore for dashboard access."""
        return self._instruction_store

    @property
    def shared_state(self) -> SharedState | None:
        """Return the SharedState for dashboard access."""
        return self._shared_state

    @property
    def signal_bus(self) -> SignalBus | None:
        """Return the SignalBus for inter-player communication."""
        return self._signal_bus

    def start(self) -> None:
        """Create and launch all agent threads.

        Creates fresh instances of SharedState, InstructionStore, and DebugStore
        (not shared between orchestrator instances). Then starts:
        1. State Poller thread (daemon)
        2. Coach Agent thread (daemon)
        3. Four Player Agent threads (daemon), one per position

        All threads are stored for monitoring and shutdown.

        Validates: Requirements 4.1, 6.2, 8.1
        """
        # Set up logging for this team instance
        setup_logging(self._config.team_color)
        logger = get_logger()

        # Create fresh per-instance state containers (Req 8.1: no shared mutable state)
        self._shared_state = SharedState()
        self._instruction_store = InstructionStore()
        self._debug_store = DebugStore()
        self._signal_bus = SignalBus()
        self._stop_event = threading.Event()
        self._threads = []

        # 1. Create and start State Poller thread
        state_poller = StatePoller(
            config=self._config,
            shared_state=self._shared_state,
            stop_event=self._stop_event,
        )
        poller_thread = threading.Thread(
            target=state_poller.run,
            name=f"StatePoller-{self._config.team_color}",
            daemon=True,
        )
        self._threads.append(poller_thread)

        # 2. Create and start Coach Agent thread
        coach_agent = CoachAgent(
            config=self._config,
            shared_state=self._shared_state,
            instruction_store=self._instruction_store,
            stop_event=self._stop_event,
            debug_store=self._debug_store,
        )
        coach_thread = threading.Thread(
            target=coach_agent.run,
            name=f"CoachAgent-{self._config.team_color}",
            daemon=True,
        )
        self._threads.append(coach_thread)

        # 3. Create and start four Player Agent threads (Req 4.1)
        for position in PLAYER_POSITIONS:
            player_agent = PlayerAgent(
                config=self._config,
                position=position,
                shared_state=self._shared_state,
                instruction_store=self._instruction_store,
                stop_event=self._stop_event,
                debug_store=self._debug_store,
                signal_bus=self._signal_bus,
            )
            player_thread = threading.Thread(
                target=player_agent.run,
                name=f"Player_{position}-{self._config.team_color}",
                daemon=True,
            )
            self._threads.append(player_thread)

        # Start all threads with staggered player starts to avoid
        # simultaneous API calls that trigger rate limiting
        import time as _time

        # Start poller and coach first
        self._threads[0].start()  # StatePoller
        self._threads[1].start()  # CoachAgent

        # Stagger player starts by 0.75s each to spread API load
        for i, thread in enumerate(self._threads[2:]):
            if i > 0:
                _time.sleep(0.75)
            thread.start()

        logger.info(
            "Team Orchestrator started all threads",
            extra={
                "agent_identity": "Orchestrator",
                "structured_context": (
                    f"team={self._config.team_color} | "
                    f"threads={len(self._threads)}"
                ),
            },
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Signal all threads to stop and wait for them to terminate.

        Sets the stop_event to signal all threads, then joins each thread
        with the specified timeout. Reports which threads stopped successfully
        and which are still alive after the timeout.

        Parameters
        ----------
        timeout : float
            Maximum time in seconds to wait for each thread to terminate.
            Defaults to 30.0 seconds (Req 6.2).

        Validates: Requirements 6.2
        """
        logger = get_logger()

        if self._stop_event is None:
            logger.warning(
                "Stop called but orchestrator was never started",
                extra={
                    "agent_identity": "Orchestrator",
                    "structured_context": "",
                },
            )
            return

        # Signal all threads to stop
        self._stop_event.set()

        # Calculate per-thread timeout to stay within total timeout
        num_threads = len(self._threads)
        per_thread_timeout = timeout / max(num_threads, 1)

        stopped: list[str] = []
        still_alive: list[str] = []

        # Join each thread with timeout
        for thread in self._threads:
            thread.join(timeout=per_thread_timeout)
            if thread.is_alive():
                still_alive.append(thread.name)
            else:
                stopped.append(thread.name)

        # Report status
        if still_alive:
            logger.warning(
                "Some threads did not stop within timeout",
                extra={
                    "agent_identity": "Orchestrator",
                    "structured_context": (
                        f"stopped={stopped} | "
                        f"still_alive={still_alive} | "
                        f"timeout={timeout}s"
                    ),
                },
            )
        else:
            logger.info(
                "All threads stopped successfully",
                extra={
                    "agent_identity": "Orchestrator",
                    "structured_context": (
                        f"team={self._config.team_color} | "
                        f"stopped={stopped}"
                    ),
                },
            )

    def is_running(self) -> bool:
        """Check if any agent thread is still alive.

        Returns
        -------
        bool
            True if at least one thread is alive, False otherwise.
        """
        if not self._threads:
            return False
        return any(thread.is_alive() for thread in self._threads)
