"""Streamlit dashboard entry point for the Agent Control Panel.

Provides the UI layout, session state management, and thread lifecycle
for the autonomous soccer agent. Loads environment variables at startup
and validates the NVIDIA API key before rendering the dashboard.
"""

import sys
import logging
import threading

import streamlit as st
from dotenv import load_dotenv
import os

from agent_loop import AgentLoop
from config import (
    DEFAULT_SERVER_IP,
    DEFAULT_SYSTEM_PROMPT,
    MAX_AGENT_NAME_LENGTH,
    MAX_BEHAVIOR_OVERRIDE_LENGTH,
    MAX_SYSTEM_PROMPT_LENGTH,
    POSITIONS,
    TEAMS,
    validate_api_key,
)
from llm_client import create_llm_client

# Load environment variables from .env file
load_dotenv()

# Validate NVIDIA_API_KEY at startup
_api_key = os.environ.get("NVIDIA_API_KEY")
if not validate_api_key(_api_key):
    st.error("NVIDIA_API_KEY is not configured. Please set it in the .env file.")
    sys.exit(1)

# --- Shared state dict for thread-safe communication ---
# st.session_state cannot be accessed from background threads, so we use a
# plain dict stored in session_state as a bridge. The background thread reads
# from / writes to this dict; the main thread syncs it with widgets each rerun.
if "shared" not in st.session_state:
    st.session_state.shared = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "behavior_override": "",
        "latest_iteration": None,
    }

# --- Session State Initialization ---
if "agent_thread" not in st.session_state:
    st.session_state.agent_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# --- Page Configuration ---
st.set_page_config(page_title="Agent Control Panel", layout="wide")
st.title("⚽ Agent Control Panel")

# --- Sidebar: Connection & Identity Configuration ---
with st.sidebar:
    st.header("Configuration")

    server_ip = st.text_input(
        "Server IP",
        value=DEFAULT_SERVER_IP,
        help="The Pitch server hostname or IP address",
    )

    team = st.selectbox(
        "Team",
        options=TEAMS,
        help="Select your team",
    )

    position = st.selectbox(
        "Position",
        options=POSITIONS,
        help="Select your player position",
    )

    agent_name = st.text_input(
        "Agent Name",
        max_chars=MAX_AGENT_NAME_LENGTH,
        help="Give your agent a name (max 50 characters)",
    )

# --- Main Area Top: Start/Stop Toggle & Status ---
col_toggle, col_status = st.columns([1, 2])

with col_toggle:
    toggle_label = "Stop Auto-Play" if st.session_state.is_running else "Start Auto-Play"
    toggle_active = st.button(toggle_label)

with col_status:
    if st.session_state.is_running:
        st.success("🟢 Agent is running")
    else:
        st.info("⚪ Agent is stopped")

# --- Toggle Logic: Start/Stop Agent Loop ---
if toggle_active:
    if not st.session_state.is_running:
        # --- Starting the agent ---
        # Validate required configuration
        if not server_ip or not server_ip.strip():
            st.error("Server IP is required. Please configure it in the sidebar.")
        elif not team:
            st.error("Team selection is required. Please configure it in the sidebar.")
        else:
            # Create stop event for thread signaling
            stop_event = threading.Event()
            st.session_state.stop_event = stop_event

            # Create LLM client (use a model that supports structured output)
            llm_client = create_llm_client(model="meta/llama-3.1-8b-instruct")

            # Reference the shared dict for thread-safe access
            shared = st.session_state.shared

            # Define callables that read from the shared dict (not st.session_state)
            def get_system_prompt():
                return shared["system_prompt"]

            def get_behavior_override():
                return shared["behavior_override"]

            # Define on_iteration callback to update the shared dict
            def on_iteration(result):
                shared["latest_iteration"] = result

            # Instantiate AgentLoop
            agent_loop = AgentLoop(
                server_ip=server_ip,
                team=team,
                position=position,
                llm_client=llm_client,
                get_system_prompt=get_system_prompt,
                get_behavior_override=get_behavior_override,
                on_iteration=on_iteration,
                stop_event=stop_event,
                agent_name=agent_name,
            )

            # Start background thread
            agent_thread = threading.Thread(
                target=agent_loop.run,
                daemon=True,
                name="agent-loop-thread",
            )
            agent_thread.start()

            # Store thread reference and update state
            st.session_state.agent_thread = agent_thread
            st.session_state.is_running = True
            st.rerun()
    else:
        # --- Stopping the agent ---
        stop_event = st.session_state.stop_event
        agent_thread = st.session_state.agent_thread

        if stop_event is not None:
            stop_event.set()

        if agent_thread is not None:
            agent_thread.join(timeout=30)

        # Clear state
        st.session_state.is_running = False
        st.session_state.agent_thread = None
        st.session_state.stop_event = None
        st.rerun()

# --- Main Area Middle: Prompt Engineering ---
st.subheader("Prompt Engineering")

system_prompt = st.text_area(
    "System Prompt",
    value=st.session_state.shared["system_prompt"],
    height=200,
    max_chars=MAX_SYSTEM_PROMPT_LENGTH,
    help="The system prompt instructs the LLM on gameplay strategy (max 2000 chars, min 6 rows)",
    key="system_prompt_widget",
)
# Sync widget value to shared dict so background thread picks it up next iteration
st.session_state.shared["system_prompt"] = system_prompt

behavior_override = st.text_input(
    "Current Behavior Override",
    value=st.session_state.shared["behavior_override"],
    max_chars=MAX_BEHAVIOR_OVERRIDE_LENGTH,
    help="Inject tactical commands during live gameplay (max 500 chars)",
    key="behavior_override_widget",
)
# Sync widget value to shared dict so background thread picks it up next iteration
st.session_state.shared["behavior_override"] = behavior_override

# --- Main Area Bottom: Debug Console ---
st.subheader("Debug Console")

iteration = st.session_state.shared["latest_iteration"]

if iteration is not None:
    col_state, col_action = st.columns(2)

    with col_state:
        st.markdown("**Game State (Latest)**")
        if iteration.game_state is not None:
            st.json(iteration.game_state)
        else:
            st.write("No game state available")

    with col_action:
        st.markdown("**LLM Response / Action**")
        action_data = {
            "dx": iteration.action.dx,
            "dy": iteration.action.dy,
            "kick": iteration.action.kick,
        }
        st.json(action_data)

        if iteration.fallback_reason:
            st.warning(f"⚠️ Brake Action used — Reason: {iteration.fallback_reason}")
else:
    st.write("No iteration data yet. Start the agent to see debug output.")

# --- Shutdown button ---
st.divider()
if st.button("⏹️ Shutdown App", type="secondary"):
    logger = logging.getLogger("player")
    logger.info("User requested app shutdown")
    st.warning("Shutting down...")
    import time
    import keyboard
    import psutil
    time.sleep(0.5)
    logging.shutdown()
    keyboard.press_and_release("ctrl+w")
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()
