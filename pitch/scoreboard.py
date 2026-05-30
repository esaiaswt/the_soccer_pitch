"""Scoreboard module for The Pitch.

Provides a web-based scoreboard page at /scoreboard showing goal events
per team, top scorers, and a download button for markdown export.
"""

from collections import Counter

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from pitch.state import StateManager

router = APIRouter()

# Module-level state manager reference, set by main.py
state_manager: StateManager = None  # type: ignore


def _get_scoreboard_data() -> dict:
    """Read goal log from state and build scoreboard data.

    If the current match has no goals (e.g., just reset), shows the
    previous match data so it can still be downloaded.

    Returns a dict with red_goals, blue_goals lists and top_scorers per team.
    """
    if not state_manager.acquire(timeout=2.0):
        return {"red_goals": [], "blue_goals": [], "red_top": [], "blue_top": [], "score": {"Red": 0, "Blue": 0}, "is_previous": False}

    try:
        state = state_manager.state
        score = dict(state.score)
        goal_log = state.goal_log
        is_previous = False

        # If current match has no goals, show previous match data
        if not goal_log and state_manager.previous_match:
            score = state_manager.previous_match["score"]
            goal_log = state_manager.previous_match["goal_log"]
            is_previous = True

        red_goals = []
        blue_goals = []

        for event in goal_log:
            entry = {
                "time": f"{event.time:.1f}s",
                "scorer": event.scorer,
            }
            if event.team == "Red":
                red_goals.append(entry)
            else:
                blue_goals.append(entry)

        # Compute top scorers per team
        red_scorers = Counter(e.scorer for e in goal_log if e.team == "Red")
        blue_scorers = Counter(e.scorer for e in goal_log if e.team == "Blue")

        red_top = [{"name": name, "goals": count} for name, count in red_scorers.most_common()]
        blue_top = [{"name": name, "goals": count} for name, count in blue_scorers.most_common()]

        return {
            "red_goals": red_goals,
            "blue_goals": blue_goals,
            "red_top": red_top,
            "blue_top": blue_top,
            "score": score,
            "is_previous": is_previous,
        }
    finally:
        state_manager.release()


@router.get("/api/scoreboard")
async def get_scoreboard_json() -> JSONResponse:
    """Return scoreboard data as JSON."""
    data = _get_scoreboard_data()
    return JSONResponse(status_code=200, content=data)


@router.get("/api/scoreboard/download")
async def download_scoreboard_md() -> PlainTextResponse:
    """Return scoreboard as downloadable markdown."""
    data = _get_scoreboard_data()
    md = _build_markdown(data)
    return PlainTextResponse(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=scoreboard.md"},
    )


@router.get("/scoreboard")
async def scoreboard_page() -> HTMLResponse:
    """Serve the scoreboard HTML page."""
    return HTMLResponse(content=SCOREBOARD_HTML)


def _build_markdown(data: dict) -> str:
    """Build a markdown string from scoreboard data."""
    lines = []
    lines.append("# Match Scoreboard")
    lines.append("")
    lines.append(f"**Red {data['score'].get('Red', 0)} - {data['score'].get('Blue', 0)} Blue**")
    lines.append("")

    # Red team goals
    lines.append("## Red Team Goals")
    lines.append("")
    if data["red_goals"]:
        lines.append("| Goal Time | Agent Name |")
        lines.append("|-----------|------------|")
        for g in data["red_goals"]:
            lines.append(f"| {g['time']} | {g['scorer']} |")
    else:
        lines.append("_No goals scored._")
    lines.append("")

    # Blue team goals
    lines.append("## Blue Team Goals")
    lines.append("")
    if data["blue_goals"]:
        lines.append("| Goal Time | Agent Name |")
        lines.append("|-----------|------------|")
        for g in data["blue_goals"]:
            lines.append(f"| {g['time']} | {g['scorer']} |")
    else:
        lines.append("_No goals scored._")
    lines.append("")

    # Top scorers
    lines.append("## Top Scorers")
    lines.append("")
    lines.append("### Red Team")
    lines.append("")
    if data["red_top"]:
        lines.append("| Agent Name | Goals |")
        lines.append("|------------|-------|")
        for s in data["red_top"]:
            lines.append(f"| {s['name']} | {s['goals']} |")
    else:
        lines.append("_No scorers._")
    lines.append("")

    lines.append("### Blue Team")
    lines.append("")
    if data["blue_top"]:
        lines.append("| Agent Name | Goals |")
        lines.append("|------------|-------|")
        for s in data["blue_top"]:
            lines.append(f"| {s['name']} | {s['goals']} |")
    else:
        lines.append("_No scorers._")
    lines.append("")

    return "\n".join(lines)


SCOREBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Match Scoreboard - The Pitch</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2em;
            color: #fff;
        }
        .score-header {
            text-align: center;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding: 10px;
            background: #16213e;
            border-radius: 8px;
        }
        .score-header .red { color: #ff6b6b; }
        .score-header .blue { color: #4dabf7; }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            max-width: 1000px;
            margin: 0 auto 20px;
        }
        .team-panel {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
        }
        .team-panel.red { border-top: 4px solid #ff6b6b; }
        .team-panel.blue { border-top: 4px solid #4dabf7; }
        .team-panel h2 {
            margin-bottom: 12px;
            font-size: 1.2em;
        }
        .team-panel.red h2 { color: #ff6b6b; }
        .team-panel.blue h2 { color: #4dabf7; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 16px;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #2a2a4a;
        }
        th {
            background: #0f3460;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td { font-size: 0.95em; }
        .top-scorers {
            margin-top: 16px;
            padding-top: 12px;
            border-top: 2px solid #2a2a4a;
        }
        .top-scorers h3 {
            font-size: 0.95em;
            margin-bottom: 8px;
            color: #aaa;
        }
        .empty { color: #666; font-style: italic; padding: 8px 0; }
        .download-bar {
            text-align: center;
            margin-top: 20px;
        }
        .download-btn {
            background: #0f3460;
            color: #fff;
            border: 1px solid #4dabf7;
            padding: 10px 24px;
            border-radius: 6px;
            font-size: 1em;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
        .download-btn:hover { background: #1a4a7a; }
        .auto-refresh { text-align: center; color: #666; font-size: 0.8em; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>&#9917; Match Scoreboard</h1>
    <div class="score-header" id="score-header">
        <span class="red">Red 0</span> &mdash; <span class="blue">0 Blue</span>
    </div>

    <div class="container">
        <div class="team-panel red">
            <h2>Red Team</h2>
            <table>
                <thead><tr><th>Goal Time</th><th>Agent Name</th></tr></thead>
                <tbody id="red-goals"><tr><td colspan="2" class="empty">No goals yet</td></tr></tbody>
            </table>
            <div class="top-scorers">
                <h3>Top Scorers</h3>
                <div id="red-top"><span class="empty">—</span></div>
            </div>
        </div>
        <div class="team-panel blue">
            <h2>Blue Team</h2>
            <table>
                <thead><tr><th>Goal Time</th><th>Agent Name</th></tr></thead>
                <tbody id="blue-goals"><tr><td colspan="2" class="empty">No goals yet</td></tr></tbody>
            </table>
            <div class="top-scorers">
                <h3>Top Scorers</h3>
                <div id="blue-top"><span class="empty">—</span></div>
            </div>
        </div>
    </div>

    <div class="download-bar">
        <a class="download-btn" href="/api/scoreboard/download">&#11015; Download as Markdown</a>
    </div>
    <div class="auto-refresh">Auto-refreshes every 2 seconds</div>

    <script>
        async function refresh() {
            try {
                const res = await fetch('/api/scoreboard');
                const data = await res.json();
                updateUI(data);
            } catch (e) { /* ignore fetch errors */ }
        }

        function updateUI(data) {
            // Score header
            const sh = document.getElementById('score-header');
            const label = data.is_previous ? ' <span style="font-size:0.7em;color:#888">(Previous Match)</span>' : '';
            sh.innerHTML = `<span class="red">Red ${data.score.Red}</span> &mdash; <span class="blue">${data.score.Blue} Blue</span>${label}`;

            // Red goals table
            const rg = document.getElementById('red-goals');
            if (data.red_goals.length === 0) {
                rg.innerHTML = '<tr><td colspan="2" class="empty">No goals yet</td></tr>';
            } else {
                rg.innerHTML = data.red_goals.map(g => `<tr><td>${g.time}</td><td>${g.scorer}</td></tr>`).join('');
            }

            // Blue goals table
            const bg = document.getElementById('blue-goals');
            if (data.blue_goals.length === 0) {
                bg.innerHTML = '<tr><td colspan="2" class="empty">No goals yet</td></tr>';
            } else {
                bg.innerHTML = data.blue_goals.map(g => `<tr><td>${g.time}</td><td>${g.scorer}</td></tr>`).join('');
            }

            // Red top scorers
            const rt = document.getElementById('red-top');
            if (data.red_top.length === 0) {
                rt.innerHTML = '<span class="empty">&mdash;</span>';
            } else {
                rt.innerHTML = data.red_top.map(s => `<div>${s.name}: ${s.goals} goal${s.goals > 1 ? 's' : ''}</div>`).join('');
            }

            // Blue top scorers
            const bt = document.getElementById('blue-top');
            if (data.blue_top.length === 0) {
                bt.innerHTML = '<span class="empty">&mdash;</span>';
            } else {
                bt.innerHTML = data.blue_top.map(s => `<div>${s.name}: ${s.goals} goal${s.goals > 1 ? 's' : ''}</div>`).join('');
            }
        }

        // Initial load + auto-refresh every 2s
        refresh();
        setInterval(refresh, 2000);
    </script>
</body>
</html>
"""
