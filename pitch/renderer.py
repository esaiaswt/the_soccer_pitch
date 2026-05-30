"""PyGame renderer module for The Pitch.

Renders a top-down 2D view of the football pitch at 30+ FPS,
including players, ball, field lines, goal zones, and HUD.
Handles keyboard events (spacebar to start, quit to exit).
"""

import time

import pygame

from pitch.config import Config
from pitch.state import MatchState, StateManager

_config = Config()

# Display constants
SCREEN_WIDTH: int = 1200
SCREEN_HEIGHT: int = 800
FPS: int = 30
FONT_SIZE_HUD: int = 24

# Colors
COLOR_GREEN = (34, 139, 34)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (220, 50, 50)
COLOR_BLUE = (50, 100, 220)
COLOR_GOAL_LEFT = (200, 200, 50)
COLOR_GOAL_RIGHT = (200, 200, 50)
COLOR_HUD_BG = (30, 30, 30)
COLOR_BALL_FILL = (255, 255, 255)
COLOR_BALL_OUTLINE = (0, 0, 0)

# Pitch layout
GOAL_ZONE_LEFT = pygame.Rect(0, 300, 30, 200)
GOAL_ZONE_RIGHT = pygame.Rect(1170, 300, 30, 200)
CENTER_CIRCLE_RADIUS = 80
PENALTY_AREA_WIDTH = 120
PENALTY_AREA_HEIGHT = 300
HUD_HEIGHT = 50


class Renderer:
    """PyGame renderer for the football pitch.

    Reads game state via the StateManager lock and renders
    the pitch, players, ball, and HUD at 30+ FPS.
    """

    def __init__(self, state_manager: StateManager, local_ip: str) -> None:
        """Initialize the renderer.

        Args:
            state_manager: Thread-safe state manager for reading game state.
            local_ip: The detected local IP address to display in the HUD.
        """
        self._state_manager = state_manager
        self._local_ip = local_ip
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._ip_refresh_interval: float = 5.0  # seconds between IP re-checks
        self._last_ip_check: float = 0.0
        self._detect_ip_func = None  # set externally for IP refresh

    def run(self) -> None:
        """Main render loop (runs on main thread).

        Initializes PyGame display, creates clock, and enters the
        main event/render loop at 30+ FPS. Exits when handle_events
        returns False (quit event).
        """
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("The Pitch - Agentic Football")
        self._clock = pygame.time.Clock()

        try:
            self._font = pygame.font.SysFont("Arial", FONT_SIZE_HUD)
        except Exception:
            self._font = pygame.font.Font(None, FONT_SIZE_HUD)

        running = True
        while running:
            running = self.handle_events()
            self._maybe_refresh_ip()
            self.render_frame()
            pygame.display.flip()
            self._clock.tick(FPS)

        pygame.quit()

    def _maybe_refresh_ip(self) -> None:
        """Re-detect the local IP every few seconds to handle network changes."""
        if self._detect_ip_func is None:
            return
        now = time.time()
        if now - self._last_ip_check >= self._ip_refresh_interval:
            self._last_ip_check = now
            new_ip = self._detect_ip_func()
            if new_ip != self._local_ip:
                self._local_ip = new_ip

    def render_frame(self) -> None:
        """Render a complete frame: pitch, players, ball, and HUD.

        Acquires the state lock to read a consistent snapshot,
        then draws all elements.
        """
        if not self._state_manager.acquire():
            return

        try:
            state = self._state_manager.state
            match_state = state.match_state
            time_left = state.time_left
            score = dict(state.score)
            ball_x = state.ball.x
            ball_y = state.ball.y
            players = {
                name: (p.team, p.x, p.y, p.name)
                for name, p in state.players.items()
            }
        finally:
            self._state_manager.release()

        # Draw layers
        self.render_pitch()
        self.render_players(players)
        self.render_ball(ball_x, ball_y)
        self.render_hud(score, time_left, match_state)

    def render_pitch(self) -> None:
        """Draw the pitch background, field lines, and goal zones."""
        self._screen.fill(COLOR_GREEN)

        # Center line
        pygame.draw.line(
            self._screen, COLOR_WHITE,
            (SCREEN_WIDTH // 2, HUD_HEIGHT),
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT),
            2,
        )

        # Center circle
        center_y = HUD_HEIGHT + (SCREEN_HEIGHT - HUD_HEIGHT) // 2
        pygame.draw.circle(
            self._screen, COLOR_WHITE,
            (SCREEN_WIDTH // 2, center_y),
            CENTER_CIRCLE_RADIUS,
            2,
        )

        # Center dot
        pygame.draw.circle(
            self._screen, COLOR_WHITE,
            (SCREEN_WIDTH // 2, center_y),
            5,
        )

        # Left penalty area
        penalty_left = pygame.Rect(
            0,
            center_y - PENALTY_AREA_HEIGHT // 2,
            PENALTY_AREA_WIDTH,
            PENALTY_AREA_HEIGHT,
        )
        pygame.draw.rect(self._screen, COLOR_WHITE, penalty_left, 2)

        # Right penalty area
        penalty_right = pygame.Rect(
            SCREEN_WIDTH - PENALTY_AREA_WIDTH,
            center_y - PENALTY_AREA_HEIGHT // 2,
            PENALTY_AREA_WIDTH,
            PENALTY_AREA_HEIGHT,
        )
        pygame.draw.rect(self._screen, COLOR_WHITE, penalty_right, 2)

        # Goal zones
        pygame.draw.rect(self._screen, COLOR_GOAL_LEFT, GOAL_ZONE_LEFT)
        pygame.draw.rect(self._screen, COLOR_GOAL_RIGHT, GOAL_ZONE_RIGHT)

        # Outer boundary
        boundary = pygame.Rect(0, HUD_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT - HUD_HEIGHT)
        pygame.draw.rect(self._screen, COLOR_WHITE, boundary, 3)

    def render_players(self, players: dict) -> None:
        """Draw player circles with team colors and name labels.

        Args:
            players: Dict mapping name to (team, x, y, display_name) tuples.
        """
        if not self._font:
            return

        for name, (team, x, y, display_name) in players.items():
            color = COLOR_RED if team == "Red" else COLOR_BLUE
            pos = (int(x), int(y))

            # Player circle
            pygame.draw.circle(self._screen, color, pos, 12)
            pygame.draw.circle(self._screen, COLOR_BLACK, pos, 12, 2)

            # Name label below player
            label = self._font.render(display_name, True, COLOR_WHITE)
            label_rect = label.get_rect(midtop=(pos[0], pos[1] + 14))
            self._screen.blit(label, label_rect)

    def render_ball(self, ball_x: float, ball_y: float) -> None:
        """Draw the ball as a white circle with black outline.

        Args:
            ball_x: Ball x position.
            ball_y: Ball y position.
        """
        pos = (int(ball_x), int(ball_y))
        pygame.draw.circle(self._screen, COLOR_BALL_FILL, pos, 8)
        pygame.draw.circle(self._screen, COLOR_BALL_OUTLINE, pos, 8, 2)

    def render_hud(self, score: dict, time_left: float, match_state: MatchState) -> None:
        """Draw the HUD header with score, time, match state, and IP.

        Args:
            score: Dict with "Red" and "Blue" integer scores.
            time_left: Remaining match time in seconds.
            match_state: Current match state enum value.
        """
        if not self._font:
            return

        # HUD background
        hud_rect = pygame.Rect(0, 0, SCREEN_WIDTH, HUD_HEIGHT)
        pygame.draw.rect(self._screen, COLOR_HUD_BG, hud_rect)

        # Score text (left side)
        score_text = f"Red {score.get('Red', 0)} - {score.get('Blue', 0)} Blue"
        score_surface = self._font.render(score_text, True, COLOR_WHITE)
        self._screen.blit(score_surface, (20, 12))

        # Time and state (center)
        time_text = f"{match_state.value} | Time: {time_left:.1f}s"
        time_surface = self._font.render(time_text, True, COLOR_WHITE)
        time_rect = time_surface.get_rect(center=(SCREEN_WIDTH // 2, HUD_HEIGHT // 2))
        self._screen.blit(time_surface, time_rect)

        # Local IP (right side)
        ip_text = f"IP: {self._local_ip}"
        ip_surface = self._font.render(ip_text, True, COLOR_WHITE)
        ip_rect = ip_surface.get_rect(midright=(SCREEN_WIDTH - 20, HUD_HEIGHT // 2))
        self._screen.blit(ip_surface, ip_rect)

    def handle_events(self) -> bool:
        """Process PyGame events.

        Handles QUIT event (returns False to stop the loop) and
        KEYDOWN SPACE (transitions Waiting→Playing).

        Returns:
            True to continue running, False to exit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if self._state_manager.acquire():
                    try:
                        state = self._state_manager.state
                        if state.match_state == MatchState.WAITING:
                            state.match_state = MatchState.PLAYING
                            state.time_left = 90.0
                    finally:
                        self._state_manager.release()

        return True
