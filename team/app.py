"""Streamlit Team Dashboard for the multi-agent soccer team.

Provides team configuration, start/stop controls, per-player overrides,
and live debug information for the Coach and Player agents.

Run with: streamlit run team/app.py

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `team` package is importable
# regardless of the working directory when running `streamlit run app.py`
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from team.config import TeamConfig, load_config
from team.instruction_store import CoachInstruction
from team.orchestrator import PLAYER_POSITIONS, TeamOrchestrator


def _get_orchestrator() -> TeamOrchestrator | None:
    """Return the orchestrator from session state, or None if not created."""
    return st.session_state.get("orchestrator")


def _is_running() -> bool:
    """Check if the orchestrator is currently running."""
    orch = _get_orchestrator()
    return orch is not None and orch.is_running()


def _render_sidebar() -> None:
    """Render the sidebar with team selection, start/stop, and overrides."""
    with st.sidebar:
        st.header("Team Control")

        # --- Pitch Server IP (editable) ---
        pitch_ip = st.text_input(
            "Pitch Server IP",
            value=st.session_state.get("pitch_ip", "localhost"),
            disabled=_is_running(),
            help="IP address or hostname of the Pitch server (default: localhost)",
            key="pitch_ip_input",
        )
        st.session_state["pitch_ip"] = pitch_ip

        # --- Team Selection (Req 6.1) ---
        team_options = ["", "Red", "Blue"]
        selected_team = st.selectbox(
            "Select Team",
            options=team_options,
            index=team_options.index(st.session_state.get("selected_team", "")),
            format_func=lambda x: "-- Choose a team --" if x == "" else x,
            disabled=_is_running(),
            key="team_selector",
        )
        st.session_state["selected_team"] = selected_team

        # --- Team Name (agent display name) ---
        team_name = st.text_input(
            "Team Name",
            value=st.session_state.get("team_name", "TeamBot"),
            disabled=_is_running(),
            max_chars=50,
            help="Display name for your agents on the pitch (max 50 chars)",
            key="team_name_input",
        )
        st.session_state["team_name"] = team_name

        # --- Start/Stop Buttons (Req 6.2, 6.3) ---
        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶ Start", disabled=_is_running(), use_container_width=True):
                # Validate team selection (Req 6.3)
                if not st.session_state.get("selected_team"):
                    st.session_state["start_error"] = (
                        "Please select a team (Red or Blue) before starting."
                    )
                else:
                    st.session_state.pop("start_error", None)
                    _start_team()

        with col2:
            if st.button("⏹ Stop", disabled=not _is_running(), use_container_width=True):
                _stop_team()

        # Display validation error (Req 6.3)
        if st.session_state.get("start_error"):
            st.error(st.session_state["start_error"])

        # Show running status
        if _is_running():
            st.success(f"🟢 Team {st.session_state.get('selected_team', '')} is running")
        else:
            st.info("⚪ Team is stopped")

        st.divider()

        # --- Coach Tactical Override (Req 6.5) ---
        st.subheader("Coach Tactical Override")
        coach_override = st.text_area(
            "Team-level tactical instruction",
            max_chars=500,
            height=100,
            key="coach_override_input",
            placeholder="Enter tactical override for the Coach (max 500 chars)...",
        )

        if st.button("Apply Coach Override", use_container_width=True):
            if coach_override and _is_running():
                _apply_coach_override(coach_override)
                st.success("Coach override applied")
            elif not _is_running():
                st.warning("Start the team first")

        st.divider()

        # --- Per-Player Overrides (Req 6.4) ---
        st.subheader("Player Overrides")
        for position in PLAYER_POSITIONS:
            override_text = st.text_area(
                f"{position} Override",
                max_chars=500,
                height=68,
                key=f"override_{position}",
                placeholder=f"Override for {position} (max 500 chars)...",
            )
            if st.button(
                f"Apply {position}",
                key=f"apply_{position}",
                use_container_width=True,
            ):
                if override_text and _is_running():
                    _apply_player_override(position, override_text)
                    st.success(f"{position} override applied")
                elif not _is_running():
                    st.warning("Start the team first")

        st.divider()

        # --- Shutdown App Button ---
        st.subheader("Application")
        if st.button("⏹️ Shutdown App", type="secondary", use_container_width=True):
            _shutdown_app()


def _start_team() -> None:
    """Create and start the TeamOrchestrator."""
    try:
        config = load_config()
        # Override with user's selections from the dashboard
        pitch_ip = st.session_state.get("pitch_ip", "localhost").strip() or "localhost"
        team_name = st.session_state.get("team_name", "TeamBot").strip() or "TeamBot"

        config = TeamConfig(
            pitch_host=pitch_ip,
            pitch_port=config.pitch_port,
            nvidia_api_key=config.nvidia_api_key,
            coach_model=config.coach_model,
            player_model=config.player_model,
            coaching_frequency=config.coaching_frequency,
            poll_interval=config.poll_interval,
            streamlit_port=config.streamlit_port,
            team_color=st.session_state["selected_team"],
            coach_memory_size=config.coach_memory_size,
            agent_name=team_name,
        )
        orchestrator = TeamOrchestrator(config)
        orchestrator.start()
        st.session_state["orchestrator"] = orchestrator
        st.session_state["agent_name"] = team_name
    except SystemExit as e:
        st.error(f"Configuration error: {e}")


def _stop_team() -> None:
    """Stop the TeamOrchestrator (waits up to 30s per Req 6.2)."""
    orch = _get_orchestrator()
    if orch is not None:
        orch.stop(timeout=30.0)
        st.session_state.pop("orchestrator", None)


def _shutdown_app() -> None:
    """Shut down the entire Team Dashboard application.

    Stops the orchestrator if running, then terminates the Streamlit process.
    Mirrors the shutdown behavior from the player/ app.
    """
    # Stop agents first if running
    orch = _get_orchestrator()
    if orch is not None:
        orch.stop(timeout=10.0)
        st.session_state.pop("orchestrator", None)

    st.warning("Shutting down Team Dashboard...")
    logging.shutdown()
    time.sleep(0.5)

    # Terminate the process
    try:
        import psutil
        pid = os.getpid()
        process = psutil.Process(pid)
        process.terminate()
    except ImportError:
        # psutil not available, fall back to os._exit
        os._exit(0)


def _apply_coach_override(content: str) -> None:
    """Inject a tactical override into the InstructionStore for all players."""
    orch = _get_orchestrator()
    if orch is None or orch.instruction_store is None:
        return
    for position in PLAYER_POSITIONS:
        instruction = CoachInstruction(
            content=content,
            timestamp=time.time(),
            target_position=position,
        )
        orch.instruction_store.set_instruction(position, instruction)


def _apply_player_override(position: str, content: str) -> None:
    """Inject a per-player override into the InstructionStore."""
    orch = _get_orchestrator()
    if orch is None or orch.instruction_store is None:
        return
    instruction = CoachInstruction(
        content=content,
        timestamp=time.time(),
        target_position=position,
    )
    orch.instruction_store.set_instruction(position, instruction)


def _render_player_debug_panels() -> None:
    """Render debug panels for each player (Req 6.6)."""
    st.header("Player Debug Panels")

    orch = _get_orchestrator()
    if orch is None or orch.debug_store is None:
        st.info("Start the team to see debug information.")
        return

    tabs = st.tabs(list(PLAYER_POSITIONS))

    for tab, position in zip(tabs, PLAYER_POSITIONS):
        with tab:
            player_info = orch.debug_store.get_player(position)
            if player_info is None:
                st.write(f"No data yet for {position}")
                continue

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Latest State")
                if player_info.latest_state is not None:
                    st.json(player_info.latest_state)
                else:
                    st.write("No state received yet")

            with col2:
                st.subheader("Latest Action")
                if player_info.latest_action is not None:
                    st.json(player_info.latest_action)
                else:
                    st.write("No action submitted yet")

            st.subheader("Coach Instruction")
            if player_info.latest_instruction is not None:
                st.write(player_info.latest_instruction)
            else:
                st.write("No instruction received yet")

            st.caption(
                f"Last updated: {time.strftime('%H:%M:%S', time.localtime(player_info.last_update))}"
            )


def _render_coach_memory() -> None:
    """Render the Coach memory/history view (Req 6.7)."""
    st.header("Coach Memory & History")

    orch = _get_orchestrator()
    if orch is None or orch.debug_store is None:
        st.info("Start the team to see coach history.")
        return

    coach_data = orch.debug_store.get_coach()
    observations = coach_data.get("observations", [])
    instructions = coach_data.get("instructions", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recent Observations (last 10)")
        # Show the 10 most recent observations
        recent_observations = observations[-10:] if observations else []
        if recent_observations:
            for i, obs in enumerate(reversed(recent_observations), 1):
                with st.expander(f"Observation {i}", expanded=(i == 1)):
                    st.json(obs)
        else:
            st.write("No observations recorded yet")

    with col2:
        st.subheader("Recent Instructions")
        if instructions:
            for position, instruction in instructions.items():
                with st.expander(f"→ {position}", expanded=True):
                    st.write(instruction)
        else:
            st.write("No instructions issued yet")


def main() -> None:
    """Main entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="Team Dashboard",
        page_icon="⚽",
        layout="wide",
    )

    st.title("⚽ Multi-Agent Team Dashboard")

    # Initialize session state defaults
    if "selected_team" not in st.session_state:
        st.session_state["selected_team"] = ""

    # Render layout
    _render_sidebar()
    _render_player_debug_panels()
    st.divider()
    _render_coach_memory()

    # Auto-refresh when running (rerun every 2 seconds for live updates)
    if _is_running():
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
