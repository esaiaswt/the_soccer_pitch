"""Main entry point for The Pitch.

Orchestrates all application threads:
- Main thread: PyGame renderer (required by OS windowing systems)
- Daemon thread: Uvicorn serving the FastAPI REST API on 0.0.0.0:8000
- Daemon thread: Physics engine running at 60Hz

Handles startup (IP detection, .env loading, component initialization),
graceful shutdown (PyGame quit → daemon threads terminate with process),
and port-in-use errors.
"""

import logging
import os
import socket
import sys
import threading

import pygame
import uvicorn
from dotenv import load_dotenv

from pitch import api
from pitch.audio import AudioManager
from pitch.config import Config
from pitch.logging_config import log_startup, setup_logging
from pitch.physics import PhysicsEngine
from pitch.renderer import Renderer
from pitch.state import StateManager

logger = logging.getLogger("pitch")


def detect_local_ip() -> str:
    """Detect the machine's LAN IP address.

    Creates a UDP socket and connects to an external address (8.8.8.8:80)
    without actually sending data. The socket's own address reveals the
    local IP used for LAN routing.

    Returns:
        The detected LAN IP address, or "127.0.0.1" on failure.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip
    except Exception:
        logger.warning("Could not detect local IP, falling back to 127.0.0.1")
        return "127.0.0.1"


def main() -> None:
    """Application entry point.

    Initializes all components and starts threads:
    1. Load .env configuration
    2. Set up logging
    3. Detect local IP
    4. Initialize StateManager
    5. Wire StateManager into the API module
    6. Initialize AudioManager (with pygame.mixer)
    7. Initialize PhysicsEngine with goal callback
    8. Start Uvicorn in a daemon thread
    9. Start PhysicsEngine in a daemon thread
    10. Run Renderer on the main thread (blocks until quit)
    11. Clean up on exit
    """
    # Load .env from the pitch directory
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

    # Initialize logging
    setup_logging()

    # Load configuration
    config = Config()

    # Detect local IP
    local_ip = detect_local_ip()

    # Log startup info
    log_startup(local_ip, config.HOST, config.PORT)
    print(f"The Pitch - Local IP: {local_ip}")
    print(f"Server binding to {config.HOST}:{config.PORT}")

    # Initialize StateManager
    state_manager = StateManager()

    # Wire state_manager into the API module
    api.state_manager = state_manager

    # Initialize pygame.mixer for audio
    try:
        pygame.mixer.init()
    except Exception as e:
        logger.warning("Failed to initialize pygame.mixer: %s. Audio disabled.", e)

    # Initialize AudioManager
    sound_path = os.path.join(os.path.dirname(__file__), "goal.wav")
    audio_manager = AudioManager(sound_path=sound_path)

    # Initialize PhysicsEngine with goal callback wired to audio
    physics_engine = PhysicsEngine(
        state_manager=state_manager,
        on_goal=audio_manager.play_goal_sound,
    )

    # Start Uvicorn in a daemon thread
    def start_uvicorn() -> None:
        try:
            uvicorn.run(
                api.app,
                host=config.HOST,
                port=config.PORT,
                log_level="info",
            )
        except OSError as e:
            if "address already in use" in str(e).lower() or "10048" in str(e):
                logger.error(
                    "Port %d is already in use. Cannot start server.", config.PORT
                )
                print(
                    f"ERROR: Port {config.PORT} is already in use. "
                    "Please close the other application or change the PORT in .env."
                )
                os._exit(1)
            else:
                logger.error("Failed to start Uvicorn: %s", e)
                print(f"ERROR: Failed to start server: {e}")
                os._exit(1)

    uvicorn_thread = threading.Thread(target=start_uvicorn, daemon=True)
    uvicorn_thread.start()

    # Start PhysicsEngine in a daemon thread
    physics_thread = threading.Thread(target=physics_engine.run, daemon=True)
    physics_thread.start()

    # Run Renderer on the main thread (blocks until PyGame quit)
    renderer = Renderer(state_manager=state_manager, local_ip=local_ip)
    try:
        renderer.run()
    except Exception as e:
        logger.error("Renderer crashed: %s", e)
    finally:
        # Stop physics engine gracefully
        physics_engine.stop()
        logger.info("The Pitch shutting down.")
        print("The Pitch shutting down. Goodbye!")


if __name__ == "__main__":
    main()
