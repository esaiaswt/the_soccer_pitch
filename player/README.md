# Agent Control Panel

A Streamlit-based AI agent dashboard that plays soccer autonomously on "The Pitch" game server. The agent uses NVIDIA NIM (ChatNVIDIA) for LLM-powered decision making in a continuous Look-Think-Act loop.

## Prerequisites

- Python 3.11 or higher
- A running instance of The Pitch game server (the `pitch/` folder in this repo)
- An NVIDIA NIM API key ([get one here](https://build.nvidia.com/))

### Installing Python

If you don't have Python 3.11+ installed, choose one of the following methods:

**Option A — Official Python installer (for venv method):**

- Windows / macOS / Linux: Download from [python.org/downloads](https://www.python.org/downloads/)
- During installation on Windows, check "Add Python to PATH"

**Option B — Anaconda / Miniconda:**

- Download Anaconda from [anaconda.com/download](https://www.anaconda.com/download) or Miniconda from [docs.anaconda.com/miniconda](https://docs.anaconda.com/miniconda/)
- Anaconda includes Python and a package manager in one bundle

## Installation

1. Navigate to the player folder:

```bash
cd player
```

2. Create and activate a virtual environment:

**Using venv (standard Python):**

```bash
python -m venv soccer_a
# Windows
soccer_a\Scripts\activate
# macOS/Linux
source soccer_a/bin/activate
```

**Using Anaconda / Miniconda:**

```bash
conda create -n soccer_a python=3.11 -y
conda activate soccer_a
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure your API key:

```bash
# Copy the template .env file and add your key
# Edit player/.env and replace the placeholder:
NVIDIA_API_KEY=your-actual-api-key-here
```

## Usage

1. Make sure The Pitch game server is running (default: `localhost:8000`).

2. Launch the dashboard:

```bash
streamlit run app.py
```

3. Configure your agent in the sidebar:
   - **Server IP** — hostname or IP of The Pitch server (default: `localhost`)
   - **Team** — Red or Blue
   - **Position** — Striker, Goalkeeper, Midfielder, or Defender
   - **Agent Name** — custom display name shown on the pitch (e.g., "MyBot" → displays as "MyBot (Striker)")

4. Customize the system prompt to define your agent's strategy, or use the default aggressive striker prompt.

5. Click **Start Auto-Play** to begin the autonomous loop.

6. Use the **Behavior Override** field to inject real-time tactical commands without stopping the agent (e.g., "defend the goal", "pass to teammate").

7. Monitor the **Debug Console** to see raw game state and LLM decisions each iteration.

## Pre-Game Behavior

Your agent appears on the pitch **as soon as it connects** — even before the match starts. During the Waiting state:
- The agent spawns at its team's default position and can move freely
- Kicks are disabled (the ball won't move until spacebar starts the match)
- This lets you verify your agent is online and see its position on the field

Default starting positions are assigned by the server based on team and role:

| Team | Role | Position (x, y) |
|------|------|-----------------|
| Red | Goalkeeper | (100, 425) |
| Red | Defender | (250, 225) / (250, 625) |
| Red | Midfielder | (400, 325) / (400, 525) |
| Red | Striker | (550, 425) |
| Blue | Goalkeeper | (1100, 425) |
| Blue | Defender | (950, 225) / (950, 625) |
| Blue | Midfielder | (800, 325) / (800, 525) |
| Blue | Striker | (650, 425) |

After a goal or match reset, all players snap back to these positions.

## How It Works

The agent runs a continuous loop every 1.5 seconds:

| Step | Action |
|------|--------|
| **Look** | GET `/api/state` — polls the game server for ball/player positions |
| **Post-Look** | Evaluate plan, run reflection on previous action, store episode in memory |
| **Think** | Sends game state + spatial analysis + agentic context to the LLM, receives movement decision (dx, dy, kick) |
| **Act** | POST `/api/action` — submits the movement vector, kick decision, and agent name to the server |
| **Post-Act** | Record episode, track opponent patterns, update strategy |

If anything fails (network error, LLM timeout, invalid response), the agent sends a **Brake Action** (no movement, no kick) to stay safe on the pitch.

The agent name you set in the sidebar is sent with every action. On the pitch, your player is displayed as `"AgentName (Position)"` (e.g., "MyBot (Striker)"). If no name is set, it falls back to `"Team_Position"`.

## Agentic Capabilities

The agent includes five agentic modules that execute entirely in Python (no extra LLM calls), injecting summarized context into the single LLM call per cycle:

| Module | Purpose |
|--------|---------|
| **Episodic Memory** | Stores past game states, actions, and outcomes in a ring buffer (max 100 episodes) |
| **Memory Summarizer** | Formats the 5 most recent episodes as compact text for the LLM prompt |
| **Planner** | Selects multi-step plans from tactical templates (score_goal, defend_goal, intercept_ball, distribute_ball) |
| **Reflection Engine** | Scores action effectiveness and signals plan abandonment after 5 consecutive low scores |
| **Strategy Tracker** | Detects opponent tendencies via directional frequency analysis and recommends counter-strategies |
| **Context Assembler** | Combines memory + plan step + adaptation hints within a 300-token budget |

All agentic processing maintains exactly **one LLM call per cycle** — the modules enrich the prompt context without adding API calls.

## Project Structure

```
player/
├── app.py              # Streamlit entry point (UI + thread lifecycle)
├── agent_loop.py       # Look-Think-Act loop (runs in background thread)
├── llm_client.py       # ChatNVIDIA initialization and invocation
├── config.py           # Constants, ActionModel, validation helpers
├── spatial.py          # Spatial analysis for enriched game state
├── episodic_memory.py  # Ring buffer for past game episodes
├── memory_summary.py   # Compact text formatting of recent memory
├── planner.py          # Template-based multi-step planning
├── reflection.py       # Action effectiveness scoring
├── strategy_tracker.py # Opponent pattern detection and adaptation
├── context_assembler.py# Priority-based agentic context assembly
├── logging_config.py   # File-based logging setup
├── requirements.txt    # Python dependencies
├── .env                # NVIDIA_API_KEY (gitignored)
├── .gitignore          # Excludes secrets and build artifacts
├── agent.log           # Runtime log file (created automatically)
└── tests/              # Property-based and unit tests
```

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite includes property-based tests (Hypothesis) validating correctness properties and unit tests covering edge cases. No NVIDIA API key or running server is needed to run tests.

## Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| Server IP | `localhost` | The Pitch server address |
| Server Port | `8000` | Fixed port for the game server |
| Loop Delay | `1.5s` | Time between agent iterations |
| Request Timeout | `5s` | HTTP timeout for server calls |
| LLM Timeout | `10s` | Hard limit on LLM response time |
| Max System Prompt | `2000 chars` | Maximum prompt length |
| Max Behavior Override | `500 chars` | Maximum override length |
| Max Agent Name | `50 chars` | Maximum name length |

## Troubleshooting

- **"NVIDIA_API_KEY is not configured"** — Make sure your `.env` file contains a valid key (not empty or whitespace).
- **Agent stays on Brake Action** — Check the debug console for the fallback reason. Common causes: server not running, empty system prompt, or LLM timeout.
- **Connection errors** — Verify The Pitch server is running and the Server IP in the sidebar is correct.
- **LLM timeouts** — The default 10s timeout may be too short for some models. The agent will safely brake and retry next iteration.
- **Agent not visible on pitch** — Make sure the agent loop is running (click Start Auto-Play). The agent appears after its first action is sent.
