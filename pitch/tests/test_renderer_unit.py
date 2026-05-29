"""Unit tests for the Renderer event handling.

Tests handle_events() behavior including QUIT events, spacebar transitions,
and state management during event processing. Uses mocked PyGame components
for headless testing.
"""

from unittest.mock import MagicMock, patch

import pygame
import pytest

from pitch.state import StateManager
from pitch.renderer import Renderer


@pytest.fixture
def state_manager():
    """Create a real StateManager for testing event handling."""
    return StateManager()


@pytest.fixture
def renderer(state_manager):
    """Create a Renderer instance with mocked pygame event system."""
    r = Renderer(state_manager=state_manager, local_ip="192.168.1.100")
    return r


class TestHandleEventsQuit:
    """Tests for QUIT event handling."""

    @patch("pygame.event.get")
    def test_quit_event_returns_false(self, mock_event_get, renderer):
        """handle_events() returns False when a QUIT event is in the queue."""
        quit_event = MagicMock()
        quit_event.type = pygame.QUIT

        mock_event_get.return_value = [quit_event]

        result = renderer.handle_events()

        assert result is False

    @patch("pygame.event.get")
    def test_no_events_returns_true(self, mock_event_get, renderer):
        """handle_events() returns True when no events are in the queue."""
        mock_event_get.return_value = []

        result = renderer.handle_events()

        assert result is True

    @patch("pygame.event.get")
    def test_non_quit_events_returns_true(self, mock_event_get, renderer):
        """handle_events() returns True when only non-QUIT events are present."""
        other_event = MagicMock()
        other_event.type = 999  # Some other event type

        mock_event_get.return_value = [other_event]

        result = renderer.handle_events()

        assert result is True


class TestHandleEventsSpacebar:
    """Tests for spacebar key event handling."""

    @patch("pygame.event.get")
    def test_spacebar_waiting_transitions_to_playing(self, mock_event_get, renderer, state_manager):
        """Pressing SPACE when match_state is WAITING transitions to PLAYING."""
        from pitch.state import MatchState

        # Ensure state is WAITING
        assert state_manager.state.match_state == MatchState.WAITING

        space_event = MagicMock()
        space_event.type = pygame.KEYDOWN
        space_event.key = pygame.K_SPACE

        mock_event_get.return_value = [space_event]

        result = renderer.handle_events()

        assert result is True
        assert state_manager.state.match_state == MatchState.PLAYING

    @patch("pygame.event.get")
    def test_spacebar_playing_does_nothing(self, mock_event_get, renderer, state_manager):
        """Pressing SPACE when match_state is PLAYING does not change state."""
        from pitch.state import MatchState

        # Set state to PLAYING
        state_manager.state.match_state = MatchState.PLAYING

        space_event = MagicMock()
        space_event.type = pygame.KEYDOWN
        space_event.key = pygame.K_SPACE

        mock_event_get.return_value = [space_event]

        result = renderer.handle_events()

        assert result is True
        assert state_manager.state.match_state == MatchState.PLAYING

    @patch("pygame.event.get")
    def test_spacebar_sets_time_left_to_90(self, mock_event_get, renderer, state_manager):
        """Pressing SPACE when transitioning to PLAYING sets time_left to 90.0."""
        from pitch.state import MatchState

        # Set time_left to something other than 90 to verify it gets reset
        state_manager.state.time_left = 45.0
        state_manager.state.match_state = MatchState.WAITING

        space_event = MagicMock()
        space_event.type = pygame.KEYDOWN
        space_event.key = pygame.K_SPACE

        mock_event_get.return_value = [space_event]

        renderer.handle_events()

        assert state_manager.state.match_state == MatchState.PLAYING
        assert state_manager.state.time_left == 90.0
