# Requirements Document

## Introduction

"The Pitch" is the central game server for a local LAN-based Agentic Football game. It provides a FastAPI REST backend that AI agents (clients) connect to over the local network, combined with a PyGame-based 2D visual frontend rendered locally on a projector screen. The server manages match state, physics simulation, player actions, scoring, and real-time rendering of a top-down football pitch on a 1200x800 grid.

## Glossary

- **The_Pitch**: The central game server application combining a FastAPI REST backend and a PyGame visual frontend.
- **Game_State**: The global dictionary holding all match data including time remaining, scores, ball position, and player positions.
- **Match_State**: An enumeration representing the current phase of a match, either `Waiting` or `Playing`.
- **Agent**: An external AI client that connects to The_Pitch over the LAN via REST API calls.
- **Player**: A named entity within the Game_State representing a team member (e.g., `Red_Striker`, `Blue_Goalkeeper`).
- **Ball**: The single game ball entity with position and velocity properties on the 2D grid.
- **Pitch_Grid**: The 1200x800 pixel 2D coordinate system used by both the physics engine and PyGame renderer.
- **Goal_Zone**: Defined rectangular areas at the far left and far right edges of the Pitch_Grid where scoring occurs.
- **Possession_Range**: A distance threshold of 30 pixels between a Player and the Ball, within which a kick action is valid.
- **MAX_SPEED**: A constant (20 pixels) representing the maximum movement applied per API call.
- **Friction**: A continuous deceleration factor applied to the Ball each physics tick to simulate natural slowdown.
- **State_Lock**: A Python `asyncio.Lock()` or `threading.Lock()` used to prevent race conditions when multiple clients update the Game_State concurrently.
- **PyGame_Thread**: A separate thread running the PyGame rendering loop alongside the FastAPI server.
- **Local_IP**: The machine's primary local network IP address (e.g., 192.168.x.x) detected programmatically.

## Requirements

### Requirement 1: Project Structure and Environment Setup

**User Story:** As a developer, I want the project to be organized in a dedicated `pitch` folder with a Python virtual environment and proper configuration, so that the application is self-contained and reproducible.

#### Acceptance Criteria

1. THE The_Pitch SHALL be built entirely within a dedicated folder named `pitch`, with all application source code, configuration files, and resources residing within this folder.
2. THE The_Pitch SHALL use a Python virtual environment named `soccer_a` located within the `pitch` folder.
3. THE The_Pitch SHALL store all local network configurations (including host addresses, port numbers, and connection credentials) in a `.env` file within the `pitch` folder.
4. THE The_Pitch SHALL include `.env` in the `.gitignore` file during initial project setup, even before the `.env` file is created, to prevent secrets from being committed.
5. THE The_Pitch SHALL maintain a log file named `pitch.log` within the `pitch` folder, recording timestamped entries at INFO level or above for application startup, shutdown, errors, and key operational events.
6. THE The_Pitch SHALL include a `requirements.txt` file within the `pitch` folder listing all Python package dependencies with pinned versions required to reproduce the environment.

### Requirement 2: FastAPI REST Server Binding and LAN Discovery

**User Story:** As a developer, I want the server to bind to all network interfaces and detect the local IP address, so that AI agents on the LAN can discover and connect to the server.

#### Acceptance Criteria

1. THE The_Pitch SHALL serve standard HTTP REST endpoints using FastAPI.
2. THE The_Pitch SHALL bind to host `0.0.0.0` on port `8000` to allow LAN traffic from any connected device.
3. WHEN the server starts, THE The_Pitch SHALL use Python's `socket` library to programmatically detect the machine's local IP address by selecting the IP address of the interface that routes to external networks.
4. THE The_Pitch SHALL display the detected Local_IP on the PyGame UI in a font size no smaller than 24px, positioned within the top 10% of the screen height.
5. THE The_Pitch SHALL NOT use WebSockets for any communication.
6. IF the local IP address cannot be detected, THEN THE The_Pitch SHALL display `127.0.0.1` as the fallback address on the PyGame UI.
7. IF port `8000` is already in use or any other startup failure occurs (such as network interface problems or FastAPI initialization errors), THEN THE The_Pitch SHALL terminate and display an error message indicating the failure reason. IF termination fails, THE The_Pitch SHALL attempt to bind to an alternative port or retry before giving up.

### Requirement 3: PyGame Visual Frontend in Separate Thread

**User Story:** As a spectator viewing the projector, I want to see a smooth 2D top-down view of the football pitch, so that I can follow the match in real time.

#### Acceptance Criteria

1. THE The_Pitch SHALL run PyGame in a separate thread (PyGame_Thread) alongside the FastAPI server.
2. THE The_Pitch SHALL render hardware-accelerated 2D graphics using PyGame on the local display.
3. THE The_Pitch SHALL render a top-down view of the Pitch_Grid at a minimum of 30 frames per second, updating entity positions based on the Game_State.
4. THE The_Pitch SHALL visually distinguish Players by team using different colors for each team, and SHALL render the Ball as a distinct entity distinguishable from Players.
5. THE The_Pitch SHALL display the remaining match time, the current score, and the host Local_IP in a fixed header area at the top of the PyGame screen, rendered at a font size no smaller than 20 pixels.
6. WHILE the PyGame_Thread is reading the Game_State for rendering, THE The_Pitch SHALL acquire the State_Lock to prevent reading inconsistent data.

### Requirement 4: Coordinate System and Physics Engine

**User Story:** As a developer, I want a strict 1200x800 2D coordinate system with realistic ball physics, so that the game simulation is consistent and visually coherent.

#### Acceptance Criteria

1. THE The_Pitch SHALL use a strict 1200 x 800 2D grid (Pitch_Grid) for all physics calculations and PyGame rendering.
2. THE The_Pitch SHALL run the physics simulation at a fixed rate of 60 ticks per second and apply Friction each tick by multiplying the Ball's velocity by a factor between 0.90 and 0.99, so that the Ball decelerates to below 0.1 pixels per tick within a bounded number of ticks.
3. THE The_Pitch SHALL treat the edges of the Pitch_Grid as solid boundaries and clamp the Ball position to remain within coordinates (0, 0) to (1200, 800).
4. WHEN the Ball reaches a boundary of the Pitch_Grid, THE The_Pitch SHALL reflect the Ball's velocity component perpendicular to that boundary (multiply by -1) so that the Ball bounces off the edge and remains within bounds.
5. WHEN a Player action includes a kick and the distance between the Player and the Ball is less than 30 pixels (Possession_Range), THE The_Pitch SHALL apply a velocity impulse of a fixed magnitude between 15 and 30 pixels per tick to the Ball in the direction the Player is facing.
6. WHEN a Player action includes a kick and the distance between the Player and the Ball is 30 pixels or greater (inclusive of exactly 30 pixels), THE The_Pitch SHALL ignore the kick action.
7. THE The_Pitch SHALL cap the Ball's velocity at a maximum of 40 pixels per tick to prevent the Ball from tunneling through boundaries or other entities.

### Requirement 5: Match Lifecycle and State Management

**User Story:** As a game operator, I want to control when the match starts using the spacebar and have a 90-second timer, so that I can coordinate the game flow for spectators and agents.

#### Acceptance Criteria

1. THE The_Pitch SHALL implement a Match_State with two values: `Waiting` and `Playing`, initialized to `Waiting` when the server starts.
2. WHILE the Match_State is `Waiting`, THE The_Pitch SHALL prevent player movement and ball physics from being applied, keeping all entity positions and velocities unchanged.
3. WHILE the Match_State is `Waiting`, WHEN the PyGame UI captures a SPACEBAR keypress, THE The_Pitch SHALL immediately transition the Match_State to `Playing`, start the 90-second game timer counting down from 90.0 to 0.0, and enable player movement and ball physics, with no additional conditions or cooldown required between matches.
4. WHILE the Match_State is `Playing`, THE The_Pitch SHALL decrement the game timer each physics tick, ignore SPACEBAR keypresses, and allow physics ticks to trigger state transitions (such as timer expiry).
5. WHEN the game timer reaches zero, THE The_Pitch SHALL transition the Match_State back to `Waiting`, stop all player movement, set the Ball velocity to zero, reset the Ball position to center (600, 400), and reset all Players to their default starting coordinates.
6. WHEN the Match_State transitions from `Playing` to `Waiting` due to timer expiry, THE The_Pitch SHALL preserve the current score, allowing the operator to start a new match by pressing SPACEBAR again.

### Requirement 6: Concurrency and Race Condition Prevention

**User Story:** As a developer, I want thread-safe access to the game state, so that simultaneous API calls from multiple agents do not corrupt the shared state.

#### Acceptance Criteria

1. WHEN a POST /api/action request is received, THE The_Pitch SHALL acquire the State_Lock before reading or modifying the Game_State dictionary and SHALL NOT proceed with the update until the lock is acquired.
2. WHEN multiple clients submit POST /api/action requests simultaneously, THE The_Pitch SHALL serialize all Game_State mutations such that the final Game_State reflects each update applied exactly once in acquisition order with no lost writes.
3. WHEN a state update completes or raises an error during processing, THE The_Pitch SHALL release the State_Lock within the same code path, ensuring the lock is never held after the request handler returns.
4. WHEN a GET /api/state request is received, THE The_Pitch SHALL acquire the State_Lock before reading the Game_State to ensure the response contains a consistent snapshot rather than a partially updated state.
5. IF the State_Lock cannot be acquired within 5 seconds, THEN THE The_Pitch SHALL abandon the pending operation and return an error response indicating the server is temporarily unable to process the request.

### Requirement 7: GET /api/state Endpoint

**User Story:** As an AI agent, I want to retrieve the current game state via a GET request, so that I can make informed decisions about my next action.

#### Acceptance Criteria

1. WHEN a GET request is received at `/api/state`, THE The_Pitch SHALL return an HTTP 200 response with a JSON body containing the current Game_State within 200 milliseconds.
2. THE The_Pitch SHALL include the following fields in the GET /api/state response: `match_state` (string, either "Waiting" or "Playing"), `time_left` (float, range 0.0 to 90.0), `score` (object with `Red` and `Blue` integer values, each 0 or greater), `ball` (object with `x` float in range 0.0 to 1200.0 and `y` float in range 0.0 to 800.0), and `players` (object mapping player names to position objects with `x` float in range 0.0 to 1200.0 and `y` float in range 0.0 to 800.0).
3. IF the Match_State is `Waiting`, THEN THE The_Pitch SHALL return an HTTP 200 response with default values in the GET /api/state response: `time_left` of 90.0, `score` of `{"Red": 0, "Blue": 0}`, `ball` at `{"x": 600.0, "y": 400.0}`, and an empty `players` object if no players have joined. THE The_Pitch SHALL return HTTP 200 for all successful /api/state responses regardless of match state.
4. IF an unexpected error occurs while processing a GET /api/state request, THEN THE The_Pitch SHALL return an HTTP 500 response with a JSON body containing an error message indicating the failure reason.

### Requirement 8: POST /api/action Endpoint

**User Story:** As an AI agent, I want to submit movement and kick actions via a POST request, so that I can control my player on the pitch.

#### Acceptance Criteria

1. WHEN a POST request is received at `/api/action`, THE The_Pitch SHALL accept a JSON payload with fields: `team` (string, one of "Red" or "Blue"), `position` (string), `vector` (object with `dx` and `dy` floats), and `kick` (boolean).
2. WHEN the incoming `dx` or `dy` values exceed the range of -1.0 to 1.0, THE The_Pitch SHALL clamp the values to the valid range before applying movement.
3. THE The_Pitch SHALL multiply the clamped vector by MAX_SPEED (20 pixels) to calculate the actual movement applied to the Player per API call, and apply it as a velocity modifier to that Player's position.
4. WHEN a POST /api/action request references a Player that does not exist in the Game_State, THE The_Pitch SHALL spawn that Player at default starting coordinates for the specified team (Red team on the left half, Blue team on the right half of the Pitch_Grid).
5. WHEN the Match_State is `Waiting`, THE The_Pitch SHALL reject all actions (both movement and kick) and return an HTTP 403 response with a JSON body indicating the match has not started.
6. WHEN a POST /api/action request contains an invalid `team` value (not "Red" or "Blue"), THE The_Pitch SHALL return an HTTP 400 response with a JSON body indicating the invalid team.
7. WHEN a valid POST /api/action request is processed successfully, THE The_Pitch SHALL return an HTTP 200 response with a JSON body confirming the action was applied.

### Requirement 9: Goal Scoring and Reset

**User Story:** As a spectator, I want to see goals registered with audio feedback and automatic position resets, so that the match flow is clear and exciting.

#### Acceptance Criteria

1. THE The_Pitch SHALL define Goal_Zones as rectangular areas: the left Goal_Zone spanning x coordinates 0 to 30 and y coordinates 300 to 500, and the right Goal_Zone spanning x coordinates 1170 to 1200 and y coordinates 300 to 500.
2. WHEN the Ball's center position enters the left Goal_Zone, THE The_Pitch SHALL increment the Blue team's score by 1. THE The_Pitch SHALL only increment the score once per goal zone entry, requiring the Ball to exit the Goal_Zone and re-enter before another goal can be scored.
3. WHEN the Ball's center position enters the right Goal_Zone, THE The_Pitch SHALL increment the Red team's score by 1. THE The_Pitch SHALL only increment the score once per goal zone entry, requiring the Ball to exit the Goal_Zone and re-enter before another goal can be scored.
4. WHEN a goal is scored, THE The_Pitch SHALL play an audio file (WAV or MP3 format) to signal the goal event, and SHALL continue normal operation if the audio file is missing or playback fails.
5. WHEN a goal is scored, THE The_Pitch SHALL reset the Ball to the center of the Pitch_Grid (600, 400) with velocity set to zero.
6. WHEN a goal is scored, THE The_Pitch SHALL reset all Players to their default starting coordinates (Red team on the left half, Blue team on the right half).
7. WHEN a goal is scored, THE The_Pitch SHALL pause physics and player movement for 2 seconds before resuming play to allow spectators to register the goal event.

### Requirement 10: Logging and Debugging

**User Story:** As a developer, I want comprehensive logging throughout the application, so that I can debug issues and monitor server behavior during matches.

#### Acceptance Criteria

1. THE The_Pitch SHALL write all log entries to a dedicated log file named `pitch.log` within the `pitch` folder, with each entry prefixed by an ISO 8601 timestamp and a log level (INFO, WARNING, or ERROR).
2. WHEN the server starts, THE The_Pitch SHALL log at INFO level the detected Local_IP, the bound host, and the bound port.
3. WHEN a Player is spawned for the first time, THE The_Pitch SHALL log at INFO level the player name, team, and starting coordinates.
4. WHEN a goal is scored, THE The_Pitch SHALL log at INFO level the scoring team, the new score, and the ISO 8601 timestamp of the goal event.
5. WHEN the Match_State transitions, THE The_Pitch SHALL log at INFO level the previous state, the new state, and the trigger event that caused the transition (spacebar press or timer expiry).
6. IF an unhandled exception occurs during processing of a GET or POST API request, THEN THE The_Pitch SHALL log at ERROR level the exception type, message, and full traceback. THE The_Pitch SHALL continue operating for recoverable exceptions, but SHALL be allowed to terminate for severe exceptions that compromise system integrity, after logging the error.
7. WHEN a GET or POST API request is processed successfully, THE The_Pitch SHALL log at INFO level the request method, endpoint path, and response status code.
8. THE The_Pitch SHALL append to the existing log file on each server start rather than overwriting previous entries.
