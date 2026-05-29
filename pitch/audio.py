"""Audio module for The Pitch.

Provides the AudioManager class that handles loading and playing
goal sound effects via pygame.mixer.
"""

import logging
import os

import pygame.mixer

logger = logging.getLogger(__name__)


class AudioManager:
    """Manages audio playback for game events.

    Loads a goal sound file and plays it when a goal is scored.
    Handles missing files and playback failures gracefully by
    logging warnings and continuing without sound.
    """

    def __init__(self, sound_path: str = "goal.wav") -> None:
        """Initialize the AudioManager.

        Args:
            sound_path: Path to the goal sound WAV file.
                        Defaults to "goal.wav".
        """
        self._sound_path = sound_path
        self._sound: pygame.mixer.Sound | None = None
        self._load_sound()

    def _load_sound(self) -> None:
        """Attempt to load the sound file.

        Logs a warning and continues if the file is missing or
        cannot be loaded.
        """
        if not os.path.isfile(self._sound_path):
            logger.warning(
                "Audio file not found: %s. Goal sounds will be disabled.",
                self._sound_path,
            )
            return

        try:
            self._sound = pygame.mixer.Sound(self._sound_path)
        except Exception as e:
            logger.warning(
                "Failed to load audio file '%s': %s. Goal sounds will be disabled.",
                self._sound_path,
                e,
            )
            self._sound = None

    def play_goal_sound(self) -> None:
        """Play the goal sound effect.

        If the sound was not loaded (missing file or load failure),
        this method does nothing. If playback fails at runtime,
        logs a warning and continues.
        """
        if self._sound is None:
            logger.warning("No goal sound loaded, skipping playback.")
            return

        try:
            self._sound.play()
        except Exception as e:
            logger.warning("Failed to play goal sound: %s", e)
