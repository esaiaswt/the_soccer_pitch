# Agentic Soccer

A multi-agent AI soccer game where LLM-powered agents play football on a shared 2D pitch. The project consists of two components:

- **The Pitch** (`pitch/`) — A game server with a PyGame visual frontend and FastAPI REST backend
- **Agent Control Panel** (`player/`) — A Streamlit dashboard that runs an autonomous AI agent using NVIDIA NIM

Agents connect over the local network, observe the game state, and submit movement decisions in real time. Players appear on the pitch as soon as they connect — even before the match starts — so you can see who's online and ready.

## Prerequisites

- Python 3.11 or higher
- An NVIDIA NIM API key for the agent ([get one here](https://build.nvidia.com/))

## Installing Python

If you don't have Python 3.11+ installed, choose one of the following methods:

**Option A — Official Python installer (for venv method):**

- Windows / macOS / Linux: Download from [python.org/downloads](https://www.python.org/downloads/)
- During installation on Windows, check "Add Python to PATH"

**Option B — Anaconda / Miniconda:**

- Download Anaconda from [anaconda.com/download](https://www.anaconda.com/download) or Miniconda from [docs.anaconda.com/miniconda](https://docs.anaconda.com/miniconda/)
- Anaconda includes Python and a package manager in one bundle

## Quick Start

### 1. Start The Pitch (game server)

```bash
cd pitch
```

**Using venv:**

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

**Install and run:**

```bash
pip install -r requirements.txt
python -m pitch.main
```

The PyGame window will open showing the pitch. Press **Spacebar** to start a match.

### 2. Start the Agent (player)

In a separate terminal:

```bash
cd player
```

**Using venv:**

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

**Install and run:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure your NVIDIA API key in `player/.env`, pick a team and position in the sidebar, then click **Start Auto-Play**. Your agent will appear on the pitch immediately — you can see it moving around even before the match starts.

### 3. View the Scoreboard

While the server is running, open a browser to:

```
http://localhost:8000/scoreboard
```

The scoreboard shows goal events per team, top scorers, and has a button to download the match report as markdown.

## Pre-Game Player Positions

When agents connect before the match starts, the server spawns them at default positions based on their team and role:

```
         Red Team (left)              Blue Team (right)
         ───────────────              ────────────────
         GK   x=100, y=425           GK   x=1100, y=425
         DEF  x=250, y=225/625       DEF  x=950,  y=225/625
         MID  x=400, y=325/525       MID  x=800,  y=325/525
         STR  x=550, y=425           STR  x=650,  y=425
```

Players can move freely during the Waiting state to warm up and reposition. Kicks are disabled until the match starts (spacebar). After a goal or match reset, all players snap back to these default positions.

## Project Structure

```
agentic_soccer/
├── pitch/          # Game server (FastAPI + PyGame)
├── player/         # AI agent dashboard (Streamlit + NVIDIA NIM)
└── README.md       # This file
```

See each subfolder's README for detailed documentation:

- [pitch/README.md](pitch/README.md) — Server architecture, API reference, game rules, scoreboard
- [player/README.md](player/README.md) — Agent configuration, how the Look-Think-Act loop works

## Running Tests

```bash
# Pitch tests
cd pitch
python -m pytest tests/ -v

# Player tests (in a separate terminal)
cd player
python -m pytest tests/ -v
```

No API key or running server is needed to run the test suites.
