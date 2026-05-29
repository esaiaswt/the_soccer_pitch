# The Pitch

A LAN-based Agentic Football game server combining a FastAPI REST backend with a PyGame 2D visual frontend. AI agents connect over the local network to control players on a top-down football pitch rendered in real time.

## Features

- **REST API** for AI agents to query game state and submit player actions
- **PyGame renderer** displaying a 1200×800 top-down football pitch at 30+ FPS
- **Deterministic physics** running at 60 ticks/second with friction, boundary reflection, and velocity capping
- **Match lifecycle** controlled via spacebar (Waiting → Playing → Waiting)
- **Goal detection** with audio feedback, score tracking, and automatic position resets
- **Thread-safe** shared state supporting multiple concurrent agents
- **LAN discovery** — detects and displays the server's local IP for easy agent connection

## Architecture

```
┌─────────────────────────────────────────────────┐
│              The Pitch Process                   │
│                                                 │
│  Main Thread     API Thread      Physics Thread │
│  ┌──────────┐   ┌───────────┐   ┌───────────┐ │
│  │ PyGame   │   │ Uvicorn   │   │ Physics   │ │
│  │ Renderer │   │ FastAPI   │   │ Engine    │ │
│  │ 30 FPS   │   │ :8000     │   │ 60 Hz     │ │
│  └────┬─────┘   └─────┬─────┘   └─────┬─────┘ │
│       │               │               │        │
│       └───────────────┼───────────────┘        │
│                       │                         │
│              ┌────────┴────────┐                │
│              │   Game State    │                │
│              │   + Lock        │                │
│              └─────────────────┘                │
└─────────────────────────────────────────────────┘
         ▲               ▲               ▲
         │               │               │
    Agent 1 (HTTP)  Agent 2 (HTTP)  Agent N (HTTP)
```

## Installation

### Prerequisites

- Python 3.11 or higher
- A display (for the PyGame window)

### Setup

```bash
cd pitch

# Create and activate a virtual environment
python -m venv soccer_a
# Windows
soccer_a\Scripts\activate
# macOS/Linux
source soccer_a/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Server

From the project root:

```bash
cd pitch
python -m pitch.main
```

Or from the parent directory:

```bash
python -m pitch.main
```

The server will:
1. Detect your local IP address and display it on screen
2. Start the REST API on `0.0.0.0:8000`
3. Open the PyGame window showing the pitch

### Controlling the Match

- **Spacebar** — Start the match (transitions from Waiting to Playing)
- **Close window** — Shut down the server

Matches last 90 seconds. After the timer expires, the match returns to Waiting and you can press Spacebar to start a new one. Scores are preserved between matches.

### Configuration

Edit `pitch/.env` to change network settings:

```env
HOST=0.0.0.0
PORT=8000
```

## Connecting Agents

When the server starts, the PyGame window displays the server's LAN IP address in the top-right corner of the HUD (e.g., `IP: 192.168.1.50`). Agents on the same network use this IP and port 8000 to connect:

```
http://<displayed-ip>:8000/api/state
http://<displayed-ip>:8000/api/action
```

No registration or handshake is needed. An agent "joins" simply by sending its first `POST /api/action` — the server automatically spawns the player at the team's default position. Multiple agents can control different players on the same team.

## API Reference

### GET /api/state

Returns the current game state.

**Response (200):**
```json
{
  "match_state": "Waiting",
  "time_left": 90.0,
  "score": {"Red": 0, "Blue": 0},
  "ball": {"x": 600.0, "y": 400.0},
  "players": {
    "Red_Striker": {"x": 550.0, "y": 400.0},
    "Blue_Goalkeeper": {"x": 1100.0, "y": 400.0}
  }
}
```

### POST /api/action

Submit a player movement and/or kick action.

**Request body:**
```json
{
  "team": "Red",
  "position": "Striker",
  "vector": {"dx": 0.5, "dy": -0.3},
  "kick": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `team` | string | `"Red"` or `"Blue"` |
| `position` | string | Player position name (e.g., `"Striker"`, `"Goalkeeper"`) |
| `vector` | object | Movement direction. `dx` and `dy` are clamped to [-1, 1] and multiplied by 20 px |
| `kick` | boolean | Attempt to kick the ball (only works within 30px of the ball) |

**Response (200):**
```json
{"status": "ok", "player": "Red_Striker"}
```

**Error responses:**

| Status | Condition |
|--------|-----------|
| 400 | Invalid team (not "Red" or "Blue") |
| 403 | Match has not started (Waiting state) |
| 503 | Server temporarily unable to process (lock timeout) |

### Agent Loop Example

```python
import httpx
import time

SERVER = "http://192.168.1.50:8000"

while True:
    # Read state
    state = httpx.get(f"{SERVER}/api/state").json()
    
    if state["match_state"] != "Playing":
        time.sleep(0.5)
        continue
    
    # Decide action based on state
    ball = state["ball"]
    # ... your AI logic here ...
    
    # Submit action
    httpx.post(f"{SERVER}/api/action", json={
        "team": "Red",
        "position": "Striker",
        "vector": {"dx": 0.8, "dy": -0.2},
        "kick": False,
    })
    
    time.sleep(0.1)  # ~10 actions per second
```

## Game Rules

- **Pitch**: 1200×800 pixel grid
- **Teams**: Red (left half) and Blue (right half)
- **Match duration**: 90 seconds
- **Ball physics**: Friction (0.97/tick), max speed 40 px/tick, bounces off boundaries
- **Kick**: Must be within 30px of the ball. Applies a 20 px/tick impulse toward the ball
- **Goals**: Left zone (x 0–30, y 300–500) scores for Blue. Right zone (x 1170–1200, y 300–500) scores for Red
- **After goal**: 2-second pause, then ball and players reset to starting positions
- **Player naming**: `{Team}_{Position}` (e.g., `Red_Striker`)
- **New players**: Automatically spawned at default team positions on first action

## Running Tests

```bash
cd pitch
python -m pytest tests/ -v
```

The test suite includes 99 tests: unit tests, property-based tests (Hypothesis), and integration tests.

## Project Structure

```
pitch/
├── __init__.py
├── main.py              # Entry point, thread orchestration
├── config.py            # Configuration dataclass
├── state.py             # Game state models and StateManager
├── physics.py           # Physics engine (60Hz)
├── api.py               # FastAPI REST endpoints
├── renderer.py          # PyGame renderer (30 FPS)
├── audio.py             # Goal sound playback
├── logging_config.py    # Logging setup
├── .env                 # Network configuration
├── requirements.txt     # Python dependencies
└── tests/
    ├── test_state_unit.py
    ├── test_state_properties.py
    ├── test_physics_unit.py
    ├── test_physics_properties.py
    ├── test_api_unit.py
    ├── test_api_properties.py
    ├── test_audio_unit.py
    ├── test_renderer_unit.py
    ├── test_logging_properties.py
    ├── test_main_unit.py
    └── test_integration.py
```

## Logging

All events are logged to `pitch/pitch.log` in append mode with ISO 8601 timestamps:

```
2024-01-15T10:30:00.123456 INFO Server starting - Local_IP=192.168.1.50, host=0.0.0.0, port=8000
2024-01-15T10:30:05.456789 INFO State transition: Waiting -> Playing (trigger=spacebar)
2024-01-15T10:31:12.789012 INFO Goal scored by Blue - Score: Red=0, Blue=1
```
