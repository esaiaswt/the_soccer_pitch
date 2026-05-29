# Implementation Plan: The Pitch

## Overview

Implement "The Pitch" as a monolithic Python application combining a FastAPI REST backend and PyGame 2D frontend. The implementation follows an incremental approach: project scaffolding → core state management → physics engine → API endpoints → renderer → audio → integration wiring. Each step builds on the previous, ensuring no orphaned code.

## Tasks

- [x] 1. Set up project structure, configuration, and dependencies
  - [x] 1.1 Create project directory structure and configuration files
    - Create `pitch/` folder with subdirectories for source and tests
    - Create `pitch/.env` with HOST=0.0.0.0, PORT=8000 defaults
    - Create `pitch/.gitignore` including `.env`, `__pycache__/`, `soccer_a/`, `*.log`
    - Create `pitch/requirements.txt` with pinned versions: fastapi>=0.100, uvicorn>=0.23, pygame>=2.5, python-dotenv>=1.0, hypothesis>=6.82, pytest>=7.4, pytest-asyncio>=0.21, httpx>=0.24
    - Create `pitch/config.py` with the `Config` dataclass containing all constants (HOST, PORT, PITCH_WIDTH, PITCH_HEIGHT, MAX_SPEED, POSSESSION_RANGE, MATCH_DURATION, PHYSICS_TICK_RATE, RENDER_FPS, FRICTION, MAX_BALL_SPEED, KICK_IMPULSE, GOAL_PAUSE, LOCK_TIMEOUT)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6_

- [x] 2. Implement game state module
  - [x] 2.1 Create state data models and StateManager
    - Create `pitch/state.py`
    - Implement `MatchState` enum with WAITING and PLAYING values
    - Implement `Ball` dataclass with x, y, vx, vy fields (defaults: 600.0, 400.0, 0.0, 0.0)
    - Implement `Player` dataclass with name, team, x, y fields
    - Implement `GameState` dataclass with match_state, time_left, score, ball, players, goal_scored_flag
    - Implement `StateManager` class with threading.Lock, acquire/release with 5s timeout, read_snapshot(), apply_action(), reset_after_goal(), reset_match()
    - Default starting positions: Red team x=100–550, Blue team x=650–1100
    - Player name convention: `{Team}_{Position}`
    - _Requirements: 5.1, 5.2, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 8.2, 8.3, 8.4_

  - [x] 2.2 Write property test for score preservation across match reset
    - **Property 9: Score preservation across match reset**
    - **Validates: Requirements 5.6**
    - Use Hypothesis to generate arbitrary score states and verify scores remain unchanged after Playing→Waiting transition

  - [x] 2.3 Write property test for post-goal reset invariant
    - **Property 11: Post-goal reset invariant**
    - **Validates: Requirements 9.5, 9.6**
    - Use Hypothesis to generate arbitrary player configurations and verify ball resets to (600, 400) with zero velocity and players return to defaults after goal

  - [x] 2.4 Write property test for movement vector calculation
    - **Property 5: Movement vector calculation**
    - **Validates: Requirements 8.2, 8.3**
    - Use Hypothesis to generate arbitrary float dx/dy values and verify applied movement equals (clamp(dx,-1,1)*20, clamp(dy,-1,1)*20)

- [x] 3. Implement physics engine
  - [x] 3.1 Create physics engine with ball mechanics
    - Create `pitch/physics.py`
    - Implement `PhysicsEngine` class with reference to StateManager
    - Implement `run()` method with 60Hz fixed-timestep loop using time.sleep
    - Implement `tick()` method that acquires lock, applies friction, updates position, handles boundaries, caps velocity, checks goals, decrements timer
    - Implement `apply_friction()`: multiply vx, vy by FRICTION (0.97)
    - Implement `update_ball_position()`: add vx to x, vy to y
    - Implement `handle_boundary_collision()`: clamp position to [0,1200]×[0,800], negate velocity component on boundary hit
    - Implement `cap_velocity()`: if magnitude > 40, scale down to 40
    - Implement `check_goal()`: detect ball in goal zones (left: x 0–30, y 300–500 → Blue scores; right: x 1170–1200, y 300–500 → Red scores), use goal_scored_flag to prevent double-scoring
    - Implement `decrement_timer()`: subtract dt, clamp to 0.0, trigger match end on zero
    - Implement goal pause: 2-second freeze after goal before resuming
    - Handle NaN/Inf ball positions by resetting to center
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.4, 9.1, 9.2, 9.3, 9.5, 9.6, 9.7_

  - [x] 3.2 Write property test for friction convergence
    - **Property 1: Friction convergence**
    - **Validates: Requirements 4.2**
    - Use Hypothesis to generate arbitrary velocities within [-40, 40] and verify speed reduces below 0.1 within bounded ticks

  - [x] 3.3 Write property test for ball boundary invariant
    - **Property 2: Ball boundary invariant**
    - **Validates: Requirements 4.3, 4.4**
    - Use Hypothesis to generate ball positions and velocities, verify position stays within [0,1200]×[0,800] after tick and velocity reflects at boundaries

  - [x] 3.4 Write property test for ball velocity cap
    - **Property 3: Ball velocity cap**
    - **Validates: Requirements 4.7**
    - Use Hypothesis to generate large velocity values and verify magnitude never exceeds 40 after any operation

  - [x] 3.5 Write property test for kick distance threshold
    - **Property 4: Kick distance threshold**
    - **Validates: Requirements 4.5, 4.6**
    - Use Hypothesis to generate player/ball positions around the 30px threshold and verify kick applies impulse only when distance < 30

  - [x] 3.6 Write property test for timer decrement
    - **Property 8: Timer decrement**
    - **Validates: Requirements 5.4**
    - Use Hypothesis to generate time_left values (0–90) and dt values, verify time_left decreases by exactly dt and never goes negative

  - [x] 3.7 Write property test for goal detection and scoring
    - **Property 10: Goal detection and scoring**
    - **Validates: Requirements 9.2, 9.3**
    - Use Hypothesis to generate ball positions near/in goal zones and verify score increments exactly once per entry

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement FastAPI application
  - [x] 5.1 Create API endpoints with validation
    - Create `pitch/api.py`
    - Implement FastAPI app instance
    - Implement `ActionRequest` Pydantic model with team, position, vector (dx, dy), kick fields
    - Implement `GET /api/state` endpoint: acquire lock with timeout, return JSON snapshot with match_state, time_left, score, ball, players; return 503 on lock timeout; return 500 on unhandled exception
    - Implement `POST /api/action` endpoint: validate team is "Red" or "Blue" (400 on invalid), reject if Waiting (403), acquire lock, clamp dx/dy to [-1,1], multiply by MAX_SPEED (20), apply movement, handle kick logic (check possession range < 30px), spawn player if not exists, return 200 on success; return 503 on lock timeout
    - _Requirements: 2.1, 2.5, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x] 5.2 Write property test for actions rejected in Waiting state
    - **Property 6: Actions rejected in Waiting state**
    - **Validates: Requirements 5.2, 8.5**
    - Use Hypothesis to generate arbitrary valid action payloads and verify 403 response when match_state is Waiting

  - [x] 5.3 Write property test for invalid team rejection
    - **Property 7: Invalid team rejection**
    - **Validates: Requirements 8.6**
    - Use Hypothesis to generate arbitrary strings (not "Red" or "Blue") and verify 400 response

  - [x] 5.4 Write property test for API response schema completeness
    - **Property 12: API response schema completeness**
    - **Validates: Requirements 7.2**
    - Use Hypothesis to generate arbitrary GameState instances and verify serialized JSON contains all required fields with correct types

- [x] 6. Implement PyGame renderer
  - [x] 6.1 Create renderer with pitch, players, ball, and HUD
    - Create `pitch/renderer.py`
    - Implement `Renderer` class with StateManager reference and local_ip string
    - Implement `run()` method: initialize PyGame display (1200x800), set caption, create clock, enter main loop
    - Implement `render_frame()`: clear screen, draw pitch, players, ball, HUD
    - Implement `render_pitch()`: green background, white field lines, goal zone rectangles
    - Implement `render_players()`: draw circles with team colors (Red/Blue), player name labels
    - Implement `render_ball()`: draw white/black circle at ball position
    - Implement `render_hud()`: display score, time_left, match_state, and Local_IP in top header area (font ≥ 24px)
    - Implement `handle_events()`: process QUIT event (return False to stop), process KEYDOWN SPACE (transition Waiting→Playing), acquire lock for state mutations
    - Maintain 30+ FPS via pygame.time.Clock
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 5.3, 2.4_

  - [x] 6.2 Write unit tests for renderer event handling
    - Test spacebar triggers Waiting→Playing transition
    - Test quit event returns False
    - Mock PyGame display for headless testing
    - _Requirements: 5.3, 3.1_

- [x] 7. Implement audio module
  - [x] 7.1 Create audio manager for goal sounds
    - Create `pitch/audio.py`
    - Implement `AudioManager` class with sound_path parameter (default "goal.wav")
    - Implement `play_goal_sound()`: load and play WAV file via pygame.mixer
    - Handle missing audio file gracefully (log warning, continue without sound)
    - Handle playback failure gracefully (log warning, continue)
    - _Requirements: 9.4_

- [x] 8. Implement logging configuration
  - [x] 8.1 Create logging setup with file handler
    - Create logging configuration in `pitch/main.py` (or a dedicated `pitch/logging_config.py`)
    - Configure file handler writing to `pitch/pitch.log` in append mode
    - Set format: `{ISO8601_TIMESTAMP} {LEVEL} {message}`
    - Set level to INFO
    - Log server startup info (Local_IP, host, port)
    - Log player spawns, goals, state transitions, API requests
    - Log errors with full traceback at ERROR level
    - _Requirements: 1.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

  - [x] 8.2 Write property test for log entry format
    - **Property 13: Log entry format**
    - **Validates: Requirements 10.1**
    - Use Hypothesis to generate arbitrary log messages and verify output matches `{ISO8601_TIMESTAMP} {LEVEL} {message}` pattern

- [x] 9. Implement application entry point and wire components together
  - [x] 9.1 Create main entry point with thread orchestration
    - Create `pitch/main.py` (or extend if logging config is already there)
    - Implement `detect_local_ip()`: use socket to detect LAN IP, fallback to 127.0.0.1
    - Load `.env` configuration via python-dotenv
    - Initialize StateManager
    - Initialize AudioManager
    - Initialize PhysicsEngine with StateManager reference
    - Start Uvicorn in a daemon thread serving the FastAPI app on 0.0.0.0:8000
    - Start PhysicsEngine in a daemon thread
    - Initialize and run Renderer on main thread (PyGame requires main thread)
    - Wire goal detection in physics to trigger AudioManager.play_goal_sound()
    - Handle graceful shutdown: PyGame quit → daemon threads terminate with process
    - Handle port-in-use error with clear error message
    - _Requirements: 2.2, 2.3, 2.6, 2.7, 3.1, 5.3_

  - [x] 9.2 Write unit tests for IP detection and startup
    - Test detect_local_ip() with mocked socket (success returns LAN IP)
    - Test detect_local_ip() with mocked socket failure (returns 127.0.0.1)
    - Test port-in-use error handling
    - _Requirements: 2.3, 2.6, 2.7_

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Integration testing
  - [x] 11.1 Write integration tests for full API lifecycle
    - Test GET /api/state returns correct JSON schema in Waiting state
    - Test POST /api/action returns 403 in Waiting state
    - Test full match lifecycle: start → action → goal → reset → timer expiry
    - Test concurrent action submissions (verify no lost writes)
    - Test lock timeout behavior (503 response)
    - Use httpx AsyncClient with FastAPI TestClient
    - _Requirements: 6.2, 7.1, 7.2, 7.3, 7.4, 8.5, 8.7_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation language is Python 3.11+ as specified in the design
- PyGame must run on the main thread; FastAPI and physics run as daemon threads
- All state access is protected by a single threading.Lock with 5-second timeout

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "6.1", "7.1", "8.1"] },
    { "id": 5, "tasks": ["6.2", "8.2", "9.1"] },
    { "id": 6, "tasks": ["9.2", "11.1"] }
  ]
}
```
