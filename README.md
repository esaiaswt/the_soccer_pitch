# Agentic Soccer

An intranet-first take on the [AWS Agentic Football Cup](https://aws.amazon.com/startups/events/agentic-football-cup-singapore-build-ai-agents-that-play-football). Instead of deploying AI agents to the cloud, this project lets you build and run them locally on your LAN — designed to be developed entirely with [AWS Kiro](https://kiro.dev).

## Inspiration

The AWS Agentic Football Cup challenges teams to build AI agents that play football autonomously. This project adapts that concept for local development: a self-hosted game server renders the pitch on a projector screen while AI agents connect over the intranet via a simple REST API. No cloud infrastructure required — just a machine on your network running the server and agents that can make HTTP calls.

## How It Works

```
┌──────────────────────┐         ┌──────────────────────┐
│   The Pitch Server   │  HTTP   │     AI Agent(s)      │
│                      │◄────────│                      │
│  • PyGame display    │         │  • Poll GET /state   │
│  • FastAPI on :8000  │────────►│  • POST /action      │
│  • Physics @ 60Hz    │         │  • Any language/LLM  │
└──────────────────────┘         └──────────────────────┘
        LAN (e.g., 192.168.x.x)
```

1. **Start the pitch server** on a machine with a display (projector, monitor, etc.)
2. **Read the IP** shown on the PyGame HUD
3. **Point your agents** at `http://<ip>:8000` and start playing

Agents can be written in any language. They poll the game state, decide on actions, and submit movement/kick commands over HTTP. Kiro can generate these agents for you using its spec-driven workflow.

## Project Structure

```
agentic_soccer/
├── README.md              ← You are here
├── .gitignore
├── .kiro/                 # Kiro spec documents (requirements, design, tasks)
│   └── specs/the-pitch/
└── pitch/                 # The game server
    ├── README.md          # Detailed server docs, API reference, game rules
    ├── main.py            # Entry point
    ├── api.py             # REST endpoints
    ├── state.py           # Game state models
    ├── physics.py         # Physics engine
    ├── renderer.py        # PyGame frontend
    ├── audio.py           # Goal sounds
    ├── config.py          # Configuration
    ├── requirements.txt   # Python dependencies
    └── tests/             # 99 tests (unit + property-based + integration)
```

## Quick Start

### Prerequisites

- Python 3.11+
- A display for the PyGame window

### Install and Run

```bash
# Clone the repo
git clone https://github.com/esaiaswt/the_soccer_pitch.git
cd the_soccer_pitch

# Set up the virtual environment
cd pitch
python -m venv soccer_a

# Activate (Windows)
soccer_a\Scripts\activate
# Activate (macOS/Linux)
# source soccer_a/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m pitch.main
```

The PyGame window opens showing the pitch. The server's LAN IP is displayed in the top-right corner.

### Start a Match

Press **Spacebar** to begin. The 90-second countdown starts and agents can submit actions.

### Connect an Agent

Agents join by sending HTTP requests to the displayed IP:

```python
import httpx, time

SERVER = "http://192.168.1.50:8000"  # Use the IP shown on screen

while True:
    state = httpx.get(f"{SERVER}/api/state").json()

    if state["match_state"] != "Playing":
        time.sleep(0.5)
        continue

    # Your AI logic here
    ball = state["ball"]

    httpx.post(f"{SERVER}/api/action", json={
        "team": "Red",
        "position": "Striker",
        "vector": {"dx": 0.8, "dy": -0.2},
        "kick": False,
    })

    time.sleep(0.1)
```

No registration needed — the server spawns a player automatically on the first action.

## API at a Glance

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Current game state (ball, players, score, timer) |
| `/api/action` | POST | Submit movement and/or kick for a player |

See [`pitch/README.md`](pitch/README.md) for the full API reference, game rules, and architecture details.

## Running Tests

```bash
cd pitch
python -m pytest tests/ -v
```

99 tests covering state management, physics, API endpoints, rendering, logging, and full integration scenarios.

## Built With

- **[AWS Kiro](https://kiro.dev)** — Spec-driven development (requirements → design → tasks → implementation)
- **[FastAPI](https://fastapi.tiangolo.com/)** — REST API framework
- **[PyGame](https://www.pygame.org/)** — 2D rendering
- **[Hypothesis](https://hypothesis.readthedocs.io/)** — Property-based testing
- **Python 3.11+**

## License

This project is for educational and experimental purposes.
