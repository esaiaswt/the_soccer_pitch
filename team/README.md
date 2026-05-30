# Multi-Agent Team

A coordinated multi-agent soccer team that orchestrates a **Coach agent** and four **Player sub-agents** (Goalkeeper, Defender, Midfielder, Striker) to play as a team on the Pitch server. The Coach observes the full game state, detects patterns, and issues natural-language tactical instructions. Each Player runs an independent Look-Think-Act loop on its own thread, making autonomous movement decisions with coach guidance as advisory context.

## Video Demo

https://github.com/user-attachments/assets/01984543-63ee-4e3c-873d-3bec643513d0



## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    team/ Application                      │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ State Poller │───▶│ SharedState  │◀── all agents read │
│  └──────────────┘    └──────────────┘                   │
│         │                                                │
│  ┌──────────────┐    ┌──────────────────┐               │
│  │ Coach Agent  │───▶│ InstructionStore │◀── players read│
│  └──────────────┘    └──────────────────┘               │
│                                                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────┐ │
│  │ Goalkeeper │ │  Defender  │ │ Midfielder │ │Striker│ │
│  └────────────┘ └────────────┘ └────────────┘ └──────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Streamlit Team Dashboard                │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   GET /api/state                POST /api/action
   (Pitch Server)                (Pitch Server)
```

- **State Poller** — Single thread polling `GET /api/state`, shares snapshot with all agents
- **Coach Agent** — Stronger LLM (70B-class) analyzing game state and issuing tactical instructions
- **Player Agents** — Lighter LLM (8B-class) per player, each on its own thread with a 1.5s decision cycle
- **Streamlit Dashboard** — Team control, live debug panels, per-player overrides

## Quick Start

### 1. Prerequisites

- Python 3.11+
- The Pitch server running (see root README)
- An NVIDIA NIM API key ([get one here](https://build.nvidia.com/))

### 2. Install

```bash
cd team
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `team/.env` and set your `NVIDIA_API_KEY`. All other parameters have sensible defaults.

### 4. Run

```bash
python main.py
```

This launches the Streamlit Team Dashboard. Select a team color (Red or Blue), then click **Start** to begin playing.

Alternatively, run the dashboard directly:

```bash
streamlit run app.py
```

## Configuration

All parameters are loaded from `team/.env`:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `NVIDIA_API_KEY` | *(required)* | — | Your NVIDIA NIM API key |
| `PITCH_HOST` | `localhost` | — | Pitch server hostname |
| `PITCH_PORT` | `8000` | — | Pitch server port |
| `COACH_MODEL` | `meta/llama-3.3-70b-instruct` | — | LLM model for the Coach |
| `PLAYER_MODEL` | `meta/llama-3.1-8b-instruct` | — | LLM model for Players |
| `COACHING_FREQUENCY` | `7` | 2–30s | How often the Coach issues instructions |
| `POLL_INTERVAL` | `1` | 0.1–10s | How often game state is polled |
| `STREAMLIT_PORT` | *(auto)* | 1024–65535 | Dashboard port (auto-scans 8501–8510 if empty) |
| `TEAM_COLOR` | `Red` | Red/Blue | Default team (can override in dashboard) |
| `COACH_MEMORY_SIZE` | `50` | — | Max game state snapshots in Coach memory |

## How It Works

### Coach Agent

The Coach runs on a configurable cadence (default 7s):
1. Reads the latest game state from SharedState
2. Adds it to a rolling memory buffer for pattern detection
3. Invokes the Coach LLM with game state + recent history
4. Parses the response into per-player tactical instructions
5. Stores instructions in the InstructionStore for players to read

### Player Agents

Each Player runs a continuous Look-Think-Act loop (1.5s cycle):
1. **Look** — Read the shared game state
2. **Think** — Invoke the Player LLM with game state + Coach instruction (if fresh)
3. **Act** — Parse the LLM response into movement (dx, dy, kick) and POST to the Pitch server

### Resilience

- **LLM timeout (10s)** → Player submits a Brake Action (dx=0, dy=0, kick=false)
- **LLM error** → Brake Action, log error, continue next cycle
- **Coach failure** → Players continue autonomously without coach context
- **Stale instructions** (>3× coaching frequency old) → Excluded from player context
- **State Poller error** → Preserves last good snapshot, retries next interval

## Multi-Instance Support

Run two teams simultaneously for AI-vs-AI matches:

```bash
# Terminal 1 — Red team
TEAM_COLOR=Red python main.py

# Terminal 2 — Blue team
TEAM_COLOR=Blue STREAMLIT_PORT=8502 python main.py
```

Each instance has fully isolated state — no shared mutable data between them. Logs go to separate files (`team_red.log`, `team_blue.log`).

## Dashboard Features

- **Team selection** — Choose Red or Blue before starting
- **Start/Stop** — Launch or halt all agents with one click
- **Coach override** — Inject a team-level tactical instruction (max 500 chars)
- **Player overrides** — Per-player behavior injection (max 500 chars each)
- **Debug panels** — Live view of each player's state, action, and coach instruction
- **Coach history** — 10 most recent observations and issued instructions

## Running Tests

```bash
cd team
python -m pytest tests/ -v
```

The test suite includes:
- Unit tests for each module
- Property-based tests (Hypothesis) validating 12 correctness properties
- Integration tests for end-to-end data flow and multi-instance isolation

No API key or running Pitch server is needed to run tests.

## Project Structure

```
team/
├── __init__.py           # Package marker
├── main.py               # CLI entry point
├── app.py                # Streamlit dashboard
├── config.py             # Configuration loading and validation
├── orchestrator.py       # Thread lifecycle management
├── state_poller.py       # Polls Pitch server for game state
├── coach_agent.py        # Coach memory + LLM instruction generation
├── player_agent.py       # Player Look-Think-Act loop
├── shared_state.py       # Thread-safe game state container
├── instruction_store.py  # Thread-safe Coach-to-Player instructions
├── debug_store.py        # Debug data for dashboard
├── logging_config.py     # Structured logging setup
├── port_scanner.py       # Streamlit port auto-assignment
├── requirements.txt      # Python dependencies
├── .env.example          # Configuration template
└── tests/                # Test suite
    ├── test_properties.py    # Property-based tests (Hypothesis)
    ├── test_integration.py   # End-to-end integration tests
    ├── test_config.py
    ├── test_shared_state.py
    ├── test_instruction_store.py
    ├── test_debug_store.py
    ├── test_logging_config.py
    ├── test_state_poller.py
    ├── test_coach_memory.py
    ├── test_coach_agent.py
    ├── test_player_agent.py
    ├── test_orchestrator.py
    ├── test_port_scanner.py
    └── test_main.py
```
