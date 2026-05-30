"""Pytest configuration for player tests.

Adds the player/ directory to sys.path so that tests can import modules
using either 'from player.config import ...' or 'from config import ...'.
"""

import sys
from pathlib import Path

# Add the player/ directory to sys.path so direct imports work
_player_dir = str(Path(__file__).resolve().parent.parent)
if _player_dir not in sys.path:
    sys.path.insert(0, _player_dir)
