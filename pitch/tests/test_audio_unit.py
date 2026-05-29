"""Unit tests for the AudioManager class."""

import logging
from unittest.mock import MagicMock, patch

import pytest


class TestAudioManager:
    """Tests for AudioManager initialization and playback."""

    @patch("pitch.audio.pygame.mixer.Sound")
    @patch("pitch.audio.os.path.isfile", return_value=True)
    def test_load_sound_success(self, mock_isfile, mock_sound_cls):
        """AudioManager loads sound when file exists."""
        from pitch.audio import AudioManager

        manager = AudioManager("goal.wav")
        mock_sound_cls.assert_called_once_with("goal.wav")
        assert manager._sound is not None

    @patch("pitch.audio.os.path.isfile", return_value=False)
    def test_load_sound_missing_file(self, mock_isfile, caplog):
        """AudioManager logs warning when file is missing."""
        from pitch.audio import AudioManager

        with caplog.at_level(logging.WARNING):
            manager = AudioManager("missing.wav")

        assert manager._sound is None
        assert "Audio file not found" in caplog.text

    @patch("pitch.audio.pygame.mixer.Sound", side_effect=Exception("load error"))
    @patch("pitch.audio.os.path.isfile", return_value=True)
    def test_load_sound_failure(self, mock_isfile, mock_sound_cls, caplog):
        """AudioManager logs warning when sound loading fails."""
        from pitch.audio import AudioManager

        with caplog.at_level(logging.WARNING):
            manager = AudioManager("bad.wav")

        assert manager._sound is None
        assert "Failed to load audio file" in caplog.text

    @patch("pitch.audio.pygame.mixer.Sound")
    @patch("pitch.audio.os.path.isfile", return_value=True)
    def test_play_goal_sound_success(self, mock_isfile, mock_sound_cls):
        """play_goal_sound() calls play on the loaded sound."""
        from pitch.audio import AudioManager

        mock_sound_instance = MagicMock()
        mock_sound_cls.return_value = mock_sound_instance

        manager = AudioManager("goal.wav")
        manager.play_goal_sound()

        mock_sound_instance.play.assert_called_once()

    @patch("pitch.audio.os.path.isfile", return_value=False)
    def test_play_goal_sound_no_sound_loaded(self, mock_isfile, caplog):
        """play_goal_sound() logs warning when no sound is loaded."""
        from pitch.audio import AudioManager

        with caplog.at_level(logging.WARNING):
            manager = AudioManager("missing.wav")
            manager.play_goal_sound()

        assert "No goal sound loaded" in caplog.text

    @patch("pitch.audio.pygame.mixer.Sound")
    @patch("pitch.audio.os.path.isfile", return_value=True)
    def test_play_goal_sound_playback_failure(self, mock_isfile, mock_sound_cls, caplog):
        """play_goal_sound() logs warning on playback failure."""
        from pitch.audio import AudioManager

        mock_sound_instance = MagicMock()
        mock_sound_instance.play.side_effect = Exception("playback error")
        mock_sound_cls.return_value = mock_sound_instance

        with caplog.at_level(logging.WARNING):
            manager = AudioManager("goal.wav")
            manager.play_goal_sound()

        assert "Failed to play goal sound" in caplog.text

    @patch("pitch.audio.pygame.mixer.Sound")
    @patch("pitch.audio.os.path.isfile", return_value=True)
    def test_custom_sound_path(self, mock_isfile, mock_sound_cls):
        """AudioManager accepts custom sound path."""
        from pitch.audio import AudioManager

        manager = AudioManager("/custom/path/cheer.wav")
        mock_sound_cls.assert_called_once_with("/custom/path/cheer.wav")
