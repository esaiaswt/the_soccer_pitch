"""CLI entry point for the multi-agent team application.

Loads configuration from team/.env, validates all parameters, sets up logging,
finds an available port for the Streamlit dashboard, and launches the dashboard
as a subprocess.

Usage:
    python team/main.py
    python -m team.main

Requirements: 9.1, 10.1, 10.2, 10.3
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `team` package is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from team.config import load_config
from team.logging_config import setup_logging
from team.port_scanner import PortUnavailableError, find_available_port


def main() -> None:
    """Entry point: load config, find port, launch Streamlit dashboard."""
    # 1. Load and validate config from team/.env (Req 9.1)
    config = load_config()

    # 2. Set up logging for this team instance
    setup_logging(config.team_color)

    # 3. Find an available port for the Streamlit dashboard
    try:
        port = find_available_port(config.streamlit_port)
    except PortUnavailableError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Launch Streamlit dashboard
    app_path = str(Path(__file__).parent / "app.py")
    url = f"http://localhost:{port}"
    print(f"Starting Team Dashboard at {url}")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_path,
                f"--server.port={port}",
                "--server.headless=true",
            ],
            check=False,
        )
    except KeyboardInterrupt:
        print("\nShutting down Team Dashboard...")
        sys.exit(0)


if __name__ == "__main__":
    main()
