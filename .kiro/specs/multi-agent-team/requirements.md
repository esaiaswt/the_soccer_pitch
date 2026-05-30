# Requirements Document

## Introduction

A multi-agent soccer team application (`team/`) that orchestrates a Coach agent and four Player sub-agents (Goalkeeper, Defender, Midfielder, Striker) to play as a coordinated team on the existing Pitch server. The Coach observes the full game state, detects patterns, and issues natural-language tactical instructions to each player. Each player runs an independent Look-Think-Act loop on its own thread, receiving coach guidance as advisory context while making autonomous movement decisions. A Streamlit dashboard provides team-level control, live debugging, and per-player overrides. The system operates entirely through the existing Pitch REST API without modifying the `pitch/` server code.

## Glossary

- **Coach_Agent**: An off-field LLM-powered agent that observes game state, maintains memory of past states, detects patterns, and issues tactical instructions to Player_Agents
- **Player_Agent**: An on-field LLM-powered sub-agent that runs an independent Look-Think-Act loop, receives advisory instructions from the Coach_Agent, and submits movement/kick actions to the Pitch server
- **State_Poller**: A dedicated thread that periodically polls `GET /api/state` from the Pitch server and shares the resulting snapshot with all agents
- **Coach_Instruction**: A free-text natural-language tactical message sent from the Coach_Agent to a specific Player_Agent
- **Look_Think_Act_Loop**: A continuous cycle where a Player_Agent reads game state, invokes its LLM for a decision, and posts an action to the Pitch server
- **Brake_Action**: A safe fallback action (dx=0, dy=0, kick=false) used when an agent cannot obtain a valid decision
- **Team_Dashboard**: A Streamlit web application providing team configuration, start/stop controls, per-player overrides, and live debug information
- **Coach_Memory**: A rolling buffer of past game state snapshots maintained by the Coach_Agent for pattern detection
- **Coaching_Frequency**: The configurable interval (in seconds) at which the Coach_Agent issues new instructions to Player_Agents
- **Pitch_Server**: The existing FastAPI game server exposing `GET /api/state` and `POST /api/action` endpoints
- **Team_Instance**: A single running instance of the team application controlling one team (Red or Blue)

## Requirements

### Requirement 1: Shared State Polling

**User Story:** As a team operator, I want a single thread to poll game state and share it with all agents, so that API calls to the Pitch server are minimized.

#### Acceptance Criteria

1. WHEN the Team_Instance starts, THE State_Poller SHALL begin polling `GET /api/state` from the Pitch_Server at a configurable interval that defaults to 1.5 seconds
2. WHEN the State_Poller receives a successful response, THE State_Poller SHALL make the game state snapshot available to the Coach_Agent and all four Player_Agents before the next polling cycle begins
3. IF the State_Poller receives an HTTP error or connection timeout (5-second limit) from the Pitch_Server, THEN THE State_Poller SHALL log the error including the HTTP status code or exception type and retry on the next polling interval without crashing, preserving the most recent successful snapshot for agents to read
4. WHILE the Team_Instance is running, THE State_Poller SHALL be the only component that calls `GET /api/state` on the Pitch_Server
5. WHEN the Team_Instance is stopped, THE State_Poller SHALL cease polling within one polling interval and release its thread

### Requirement 2: Coach Agent Observation and Memory

**User Story:** As a team operator, I want the Coach to observe the full game state and remember past states, so that it can detect opponent patterns and make informed tactical decisions.

#### Acceptance Criteria

1. WHEN a new game state snapshot is available from the State_Poller, THE Coach_Agent SHALL read and store the ball position, all player positions, score, time remaining, match state, and the timestamp at which the snapshot was received
2. THE Coach_Agent SHALL maintain a Coach_Memory buffer of past game state snapshots in chronological order up to a configurable maximum size that defaults to 50 snapshots
3. WHEN adding a new snapshot would cause the Coach_Memory buffer to exceed its maximum size, THE Coach_Agent SHALL discard the oldest snapshot before adding the new one
4. WHEN analyzing game state, THE Coach_Agent SHALL have access to the full Coach_Memory buffer to detect patterns across multiple snapshots
5. IF a game state snapshot from the State_Poller is missing required fields (ball position, player positions, score, time remaining, or match state), THEN THE Coach_Agent SHALL discard that snapshot without adding it to Coach_Memory and log a warning

### Requirement 3: Coach Agent Tactical Instructions

**User Story:** As a team operator, I want the Coach to issue personalized tactical instructions to each player at a regular cadence, so that the team plays with coordinated strategy.

#### Acceptance Criteria

1. WHILE the Team_Instance is running, THE Coach_Agent SHALL issue a Coach_Instruction to each of the four Player_Agents once per Coaching_Frequency interval
2. THE Coach_Agent SHALL use the Coach LLM model (configured separately from the Player LLM model) via ChatNVIDIA to generate Coach_Instructions
3. WHEN generating a Coach_Instruction, THE Coach_Agent SHALL produce free-text natural-language guidance that references the target player's role name (Goalkeeper, Defender, Midfielder, or Striker) and relates to that role's tactical responsibilities
4. THE Coach_Instruction SHOULD target 500 characters or fewer in length, but the system SHALL accept and deliver longer instructions when the LLM generates them
5. THE Coaching_Frequency SHALL default to 5 seconds and be adjustable by the user via configuration within the range of 2 to 30 seconds
6. IF the Coach_Agent LLM invocation fails or times out when generating a Coach_Instruction, THEN THE Coach_Agent SHALL log the error, skip that coaching cycle, and retry on the next Coaching_Frequency interval without crashing. The Coach_Agent MAY also skip coaching cycles for other reasons at its discretion.

### Requirement 4: Player Agent Look-Think-Act Loop

**User Story:** As a team operator, I want each player to run its own decision loop on a separate thread, so that players act independently and in parallel.

#### Acceptance Criteria

1. WHEN the Team_Instance starts, THE Team_Instance SHALL launch four Player_Agent threads, one for each position (Goalkeeper, Defender, Midfielder, Striker)
2. WHILE a Player_Agent thread is running, THE Player_Agent SHALL execute a continuous Look_Think_Act_Loop that reads the shared game state from the State_Poller, invokes its LLM, posts an action to the Pitch_Server via `POST /api/action`, and then waits 1.5 seconds before starting the next iteration. IF reading the shared game state fails, THE Player_Agent SHALL still proceed with LLM invocation and action posting using the most recently available snapshot
3. THE Player_Agent SHALL use a lighter LLM model (8B-parameter class or smaller) via ChatNVIDIA for its decision-making invocations, distinct from the stronger model used by the Coach_Agent
4. WHEN a Player_Agent invokes its LLM, THE Player_Agent SHALL include the latest Coach_Instruction as advisory context alongside the game state
5. IF no Coach_Instruction has been received yet or the Coach_Agent has stopped producing instructions, THEN THE Player_Agent SHALL invoke its LLM using only the game state without coach context and continue operating autonomously
6. THE Player_Agent SHALL make the final movement and kick decision based solely on its own LLM output, regardless of the Coach_Instruction content

### Requirement 5: Player Agent Resilience

**User Story:** As a team operator, I want players to keep operating even when the coach or LLM fails, so that the team never stops playing mid-match.

#### Acceptance Criteria

1. IF the Coach_Agent thread crashes or the most recent Coach_Instruction timestamp is older than 3 consecutive Coaching_Frequency intervals, THEN THE Player_Agent SHALL continue operating using only the shared game state without including any Coach_Instruction in its LLM context
2. IF a Player_Agent LLM invocation does not return within 10 seconds, THEN THE Player_Agent SHALL submit a Brake_Action (dx=0, dy=0, kick=false) and continue to the next loop iteration. The Player_Agent SHALL NOT submit a Brake_Action during normal LLM processing that completes within the timeout.
3. IF a Player_Agent LLM invocation returns an error, THEN THE Player_Agent SHALL submit a Brake_Action (dx=0, dy=0, kick=false), log the error with structured context including agent identity and error type, and continue to the next loop iteration
4. IF the State_Poller fails to update the shared game state for more than 2 consecutive polling intervals, THEN THE Player_Agent SHALL use the most recently available snapshot for its decision and continue its Look_Think_Act_Loop without interruption
5. WHILE a Player_Agent is operating without Coach_Instructions due to coach failure, THE Player_Agent SHALL resume incorporating Coach_Instructions once a new Coach_Instruction with a current timestamp is received

### Requirement 6: Streamlit Team Dashboard

**User Story:** As a team operator, I want a web dashboard to configure, control, and monitor the entire team from one interface.

#### Acceptance Criteria

1. WHEN the Team_Dashboard launches, THE Team_Dashboard SHALL present a team selection control allowing the user to choose Red or Blue
2. THE Team_Dashboard SHALL provide a single button to start all agents (Coach_Agent and four Player_Agents) and a single button to stop all agents, where pressing stop SHALL signal all agent threads to terminate and wait up to 30 seconds for confirmation before reporting them as stopped
3. IF the user presses the start button without having selected a team, THEN THE Team_Dashboard SHALL display an error message indicating that team selection is required and SHALL NOT start any agents
4. THE Team_Dashboard SHALL provide a per-player free-text input field (maximum 500 characters) for injecting behavior overrides into each Player_Agent
5. THE Team_Dashboard SHALL provide a team-level free-text input field (maximum 500 characters) for injecting a tactical override into the Coach_Agent
6. THE Team_Dashboard SHALL display a debug panel for each Player_Agent showing that player's latest game state, most recent action, and current Coach_Instruction, updated on each Streamlit rerun cycle via a thread-safe shared data structure
7. THE Team_Dashboard SHALL display a Coach_Memory and history view showing the Coach_Agent's 10 most recent observations and issued instructions

### Requirement 7: Observability and Logging

**User Story:** As a developer, I want structured logs of all agent communication and performance metrics, so that I can debug and tune the system.

#### Acceptance Criteria

1. THE Team_Instance SHALL log every Coach_Instruction sent from the Coach_Agent to a Player_Agent with a structured format including an ISO 8601 timestamp, target player identity, and instruction content
2. WHEN an LLM invocation returns token usage metadata, THE Team_Instance SHALL log prompt token count, completion token count, and total token count for that invocation along with the agent identity
3. IF an LLM invocation response does not include token usage metadata, THEN THE Team_Instance SHALL always log a warning indicating token data was unavailable for that invocation along with the agent identity, regardless of the reason for the missing metadata
4. THE Team_Instance SHALL log decision latency in milliseconds (time from LLM invocation start to response received) for each Coach_Agent and Player_Agent invocation
5. IF any agent encounters an error, THEN THE Team_Instance SHALL log the error with structured context including agent identity, error type, the current match state, and the action the agent was attempting
6. THE Team_Instance SHALL write all logs to a dedicated log file per Team_Instance in append mode using ISO 8601 timestamps with microsecond precision, matching the format used by the Pitch_Server
7. THE Team_Instance SHALL use thread-safe logging so that concurrent writes from the State_Poller, Coach_Agent, and Player_Agent threads do not interleave or corrupt log entries

### Requirement 8: Multi-Instance Support

**User Story:** As a developer, I want to run two team instances simultaneously on the same machine, so that I can stage full AI-vs-AI matches.

#### Acceptance Criteria

1. THE Team_Instance SHALL support running two instances concurrently on the same machine without shared mutable state, where each instance controls a different team (Red or Blue) and both connect to the same Pitch_Server
2. WHEN a Team_Instance starts its Team_Dashboard and no port is configured via environment variable, THE Team_Instance SHALL scan ports starting from 8501 up to 8510 and bind to the first port not already in use
3. WHERE the user configures a specific port via the STREAMLIT_PORT environment variable, THE Team_Instance SHALL use that port for the Team_Dashboard
4. IF the configured or auto-assigned port is already in use, THEN THE Team_Instance SHALL display an error message indicating the port conflict and refuse to start the Team_Dashboard
5. THE Team_Instance SHALL write logs to a team-specific log file named by team color (e.g., `team_red.log`, `team_blue.log`) so that concurrent instances do not interleave log output

### Requirement 9: Configuration and Environment

**User Story:** As a team operator, I want all tunable parameters in a configuration file, so that I can adjust behavior without modifying code.

#### Acceptance Criteria

1. WHEN the Team_Instance starts, THE Team_Instance SHALL load configuration from a `.env` file in the `team/` directory before initializing any agents or the State_Poller
2. THE Team_Instance SHALL support configuration of the following parameters: Pitch_Server address (host and port), NVIDIA API key, Coach LLM model name, Player LLM model name, Coaching_Frequency (in seconds, valid range 2 to 30), State_Poller interval (in seconds, valid range 0.1 to 10), and Streamlit port (valid range 1024 to 65535). IF any parameter value falls outside its valid range, THEN THE Team_Instance SHALL reject the value, display an error message indicating the invalid parameter and its valid range, and exit without starting any agents.
3. IF the NVIDIA API key is missing or empty in the configuration, THEN THE Team_Instance SHALL display an error message indicating the missing key and exit without starting any agents
4. WHERE the Pitch_Server address is not specified, THE Team_Instance SHALL default to `localhost:8000`
5. WHERE the Coaching_Frequency is not specified, THE Team_Instance SHALL default to 7 seconds
6. WHERE the State_Poller interval is not specified, THE Team_Instance SHALL default to 1 second
7. WHERE the Streamlit port is not specified, THE Team_Instance SHALL auto-assign an available port. IF a configured port is unavailable, THE Team_Instance SHALL fall back to auto-assigning an available port using the same scanning mechanism (ports 8501 to 8510).

### Requirement 10: Isolation from Existing Components

**User Story:** As a developer, I want the team app to be fully independent from the existing player app, so that both can coexist without conflicts.

#### Acceptance Criteria

1. THE Team_Instance SHALL place all source code, tests, configuration files, and virtual environment within the `team/` directory at the project root
2. THE Team_Instance SHALL interact with the Pitch_Server exclusively through the existing `GET /api/state` and `POST /api/action` REST endpoints
3. THE Team_Instance SHALL not contain any import statements referencing modules from the `pitch/` or `player/` directories, and SHALL not add those directories to its Python path
4. THE Team_Instance SHALL maintain its own `requirements.txt` located at `team/requirements.txt`, listing pinned or bounded dependency versions independent of those in `pitch/requirements.txt` or `player/requirements.txt`
5. THE Team_Instance SHALL not define a top-level Python package named `pitch` or `player` within the `team/` directory
