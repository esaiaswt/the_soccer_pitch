# Requirements Document

## Introduction

The Agent Control Panel is a client-side Streamlit dashboard that enables participants in a local Agentic Football LAN tournament to control an AI-powered player agent. The dashboard communicates with an existing game server ("The Pitch") via REST API, using an LLM (NVIDIA NIM via ChatNVIDIA) to make real-time gameplay decisions. The agent follows a continuous Look-Think-Act loop: polling game state, reasoning about movement via structured LLM output, and posting actions back to the server.

## Glossary

- **Control_Panel**: The Streamlit-based client dashboard application that participants use to configure and run their AI agent
- **Pitch_Server**: The existing FastAPI game server ("The Pitch") running at a configurable IP address on port 8000, exposing REST endpoints for game state and player actions
- **Agent_Loop**: The continuous while-loop cycle (Look → Think → Act → Rate Limit) that drives autonomous gameplay
- **ChatNVIDIA**: The LangChain NVIDIA AI Endpoints integration class used to invoke LLM inference with structured output
- **Action_Model**: The Pydantic BaseModel defining the structured output schema for LLM responses (dx, dy, kick)
- **Brake_Action**: The default safe action (dx=0.0, dy=0.0, kick=False) sent when the LLM fails to produce a valid response
- **Possession_Range**: The 30-pixel radius within which a player can kick the ball
- **Pitch_Dimensions**: The 1200x800 pixel playing field dimensions
- **System_Prompt**: The configurable text prompt that instructs the LLM on gameplay strategy
- **Behavior_Override**: A real-time text injection that modifies agent behavior during live gameplay

## Requirements

### Requirement 1: Project Structure and Environment Setup

**User Story:** As a tournament participant, I want a self-contained project folder with a Python virtual environment, so that I can run the agent without dependency conflicts.

#### Acceptance Criteria

1. THE Control_Panel SHALL be built within a dedicated `player` folder at the workspace root
2. WHEN the project is initialized, THE Control_Panel SHALL create a Python virtual environment named `soccer_a` within the `player` folder using Python 3.10 or higher
3. THE Control_Panel SHALL store the `NVIDIA_API_KEY` in a `.env` file within the `player` folder using the format `NVIDIA_API_KEY=<value>`
4. THE Control_Panel SHALL include `.env` in the `.gitignore` file within the `player` folder to prevent secret exposure
5. IF the `NVIDIA_API_KEY` environment variable is missing or empty at runtime, THEN THE Control_Panel SHALL exit with a non-zero exit code and an error message indicating that the API key is not configured

### Requirement 2: Server Connection Configuration

**User Story:** As a tournament participant, I want to configure the server connection details and my player identity, so that my agent connects to the correct game and plays on the right team.

#### Acceptance Criteria

1. THE Control_Panel SHALL provide a text input field for the Pitch_Server IP address with a default value of "localhost"
2. THE Control_Panel SHALL provide a dropdown selector for team selection with options "Red" and "Blue"
3. THE Control_Panel SHALL provide a dropdown selector for position selection with the options: "Striker", "Goalkeeper", "Midfielder", "Defender"
4. THE Control_Panel SHALL provide a text input field for the agent name with a maximum length of 50 characters
5. WHEN the agent constructs API URLs, THE Control_Panel SHALL use the format `http://{SERVER_IP}:8000/api/state` and `http://{SERVER_IP}:8000/api/action`

### Requirement 3: Game State Retrieval (Look Step)

**User Story:** As a tournament participant, I want my agent to continuously poll the game server for the current state, so that the LLM can make informed decisions based on live data.

#### Acceptance Criteria

1. WHEN the Agent_Loop executes the Look step, THE Control_Panel SHALL send a GET request to `http://{SERVER_IP}:8000/api/state` with a timeout of 5 seconds
2. WHEN the GET request returns an HTTP 200 response, THE Control_Panel SHALL parse the JSON response containing match_state, time_left, score, ball position (x, y), and player positions (x, y per player)
3. WHEN the GET request returns an HTTP 200 response, THE Control_Panel SHALL proceed to the Think step with the retrieved game state and SHALL NOT use the Brake_Action
4. IF the GET request returns a non-200 HTTP status code, times out, or raises a connection error, THEN THE Control_Panel SHALL continue the loop with the Brake_Action regardless of whether error logging succeeds

### Requirement 4: LLM Decision Making (Think Step)

**User Story:** As a tournament participant, I want the agent to use an LLM with structured output to decide movement and kick actions, so that gameplay decisions are intelligent and schema-compliant.

#### Acceptance Criteria

1. THE Action_Model SHALL define three fields: `dx` (float, constrained to -1.0 through 1.0), `dy` (float, constrained to -1.0 through 1.0), and `kick` (boolean)
2. THE Control_Panel SHALL initialize ChatNVIDIA with a configurable model identifier (default: `meta/llama3-8b-instruct`)
3. THE Control_Panel SHALL bind the Action_Model to ChatNVIDIA using the `.with_structured_output()` method
4. WHEN the Think step executes, THE Control_Panel SHALL pass the System_Prompt as the system message and the current game state JSON as the user message to the structured LLM
5. IF the LLM invocation raises an exception or does not return a response within 10 seconds during the Think step, THEN THE Control_Panel SHALL complete the Think step using the Brake_Action as the fallback decision
6. WHEN a Behavior_Override text input contains a non-empty string, THE Control_Panel SHALL append the override text after the game state JSON in the user message sent to the LLM
7. THE Control_Panel SHALL limit the Behavior_Override text input to a maximum of 500 characters

### Requirement 5: Action Submission (Act Step)

**User Story:** As a tournament participant, I want the agent to submit validated actions to the game server, so that my player moves and kicks on the pitch.

#### Acceptance Criteria

1. WHEN the Agent_Loop executes the Act step, THE Control_Panel SHALL send a POST request to `http://{SERVER_IP}:8000/api/action` with a JSON body containing `team` (string), `position` (string), `vector` (object with `dx` and `dy` float fields), and `kick` (boolean) fields
2. THE Control_Panel SHALL use the dx and dy values from the Action_Model response as the `dx` and `dy` fields within the `vector` object
3. IF the POST request fails due to a network error or returns a non-2xx HTTP status code, THEN THE Control_Panel SHALL log the error details to the debug console and continue to the next Agent_Loop iteration
4. THE Control_Panel SHALL enforce a timeout of 5 seconds on the POST request to prevent the Agent_Loop from blocking indefinitely

### Requirement 6: Rate Limiting

**User Story:** As a tournament participant, I want the agent loop to enforce a delay between iterations, so that the server is not overwhelmed with requests.

#### Acceptance Criteria

1. WHEN the Agent_Loop completes one iteration (Look, Think, Act), THE Control_Panel SHALL wait 1.5 seconds before starting the next iteration regardless of whether the iteration succeeded or used the Brake_Action due to errors
2. WHILE the match_state is not "Playing", THE Control_Panel SHALL continue to enforce the 1.5-second delay between Agent_Loop iterations
3. IF the Agent_Loop toggle is deactivated during the 1.5-second wait, THEN THE Control_Panel SHALL complete the wait period before stopping the loop

### Requirement 7: Error Handling and Safety

**User Story:** As a tournament participant, I want the agent to handle LLM failures gracefully, so that my player does not crash or send invalid data to the server.

#### Acceptance Criteria

1. THE Control_Panel SHALL catch all exceptions raised during the structured LLM invocation without terminating the Agent_Loop
2. IF the LLM invocation raises an exception, THEN THE Control_Panel SHALL log the exception type and error message to the Streamlit debug console and use the Brake_Action (dx=0.0, dy=0.0, kick=False) as the action for that iteration
3. IF the LLM returns a response that fails Pydantic validation against the Action_Model, THEN THE Control_Panel SHALL log the validation error to the Streamlit debug console and use the Brake_Action as the action for that iteration
4. IF the LLM invocation does not return a response within 10 seconds, THEN THE Control_Panel SHALL abort the invocation, log a timeout warning to the Streamlit debug console, and use the Brake_Action as the action for that iteration
5. IF the LLM returns a None or empty response that is not a valid Action_Model instance, THEN THE Control_Panel SHALL treat it as a validation failure and use the Brake_Action as the action for that iteration

### Requirement 8: Agent Loop Control

**User Story:** As a tournament participant, I want a toggle switch to start and stop the autonomous agent loop, so that I have control over when the agent plays.

#### Acceptance Criteria

1. THE Control_Panel SHALL provide a "Start Auto-Play" toggle switch in the Streamlit UI with an initial state of off (deactivated) on application load
2. WHEN the toggle is activated, THE Control_Panel SHALL begin executing the Agent_Loop continuously, following the rate limit defined in Requirement 6
3. WHEN the toggle is deactivated, THE Control_Panel SHALL stop the Agent_Loop after the current iteration completes, within a maximum of 30 seconds from deactivation
4. WHILE the Agent_Loop is running, THE Control_Panel SHALL display a visible status indicator showing that the agent is actively playing
5. IF the toggle is activated and the server IP address or team selection is not configured, THEN THE Control_Panel SHALL not start the Agent_Loop and SHALL display an error message indicating the missing configuration

### Requirement 9: Prompt Engineering Panel

**User Story:** As a tournament participant, I want to customize the system prompt sent to the LLM, so that I can experiment with different strategies during the tournament.

#### Acceptance Criteria

1. THE Control_Panel SHALL provide a text area with a minimum of 6 visible rows and a maximum input length of 2000 characters for editing the System_Prompt
2. THE Control_Panel SHALL pre-fill the System_Prompt text area with: "You are an aggressive striker on a 1200x800 pitch. Calculate the shortest vector to the ball (normalized between -1.0 and 1.0). If you are within 30 pixels of the ball, set 'kick' to True to shoot towards the goal."
3. THE Control_Panel SHALL provide a separate single-line text input labeled "Current Behavior Override" with a maximum input length of 500 characters for injecting tactical commands during live gameplay
4. WHEN the System_Prompt text area is modified while the Agent_Loop is running, THE Control_Panel SHALL use the updated System_Prompt starting from the next Agent_Loop iteration
5. IF the System_Prompt text area is empty when the Think step executes, THEN THE Control_Panel SHALL use the Brake_Action for that iteration and display a warning indicating that a System_Prompt is required
6. WHEN the Behavior_Override text input is cleared during live gameplay, THE Control_Panel SHALL send only the System_Prompt without any override text on the next Agent_Loop iteration

### Requirement 10: Debug Console

**User Story:** As a tournament participant, I want to see raw game state and LLM reasoning in real-time, so that I can debug and understand my agent's behavior.

#### Acceptance Criteria

1. THE Control_Panel SHALL display the raw incoming JSON game state in a debug console area showing only the most recent iteration's data
2. THE Control_Panel SHALL display the LLM reasoning output (the structured Action_Model response) in the debug console area showing only the most recent iteration's data
3. WHEN the Agent_Loop completes an iteration, THE Control_Panel SHALL update the debug console with the latest game state and LLM response
4. IF the Brake_Action is used for an iteration, THEN THE Control_Panel SHALL display the Brake_Action values in the debug console along with the reason for fallback (e.g., "LLM timeout", "validation error", "connection error")

### Requirement 11: File-Based Logging

**User Story:** As a tournament participant, I want a persistent log file for debugging and troubleshooting, so that I can review agent behavior after gameplay.

#### Acceptance Criteria

1. THE Control_Panel SHALL create a log file named `agent.log` within the `player` folder using Python's standard `logging` module with append mode
2. THE Control_Panel SHALL log each game state retrieval (success or failure), LLM invocation (input summary and output), action submission (payload and response status), and error occurrence (exception type and message) to the log file
3. THE Control_Panel SHALL include ISO 8601 timestamps (YYYY-MM-DDTHH:MM:SS) in all log entries along with the log level (DEBUG, INFO, WARNING, ERROR)

### Requirement 12: Dependency Management

**User Story:** As a tournament participant, I want clearly defined dependencies, so that I can set up the project quickly on any machine.

#### Acceptance Criteria

1. THE Control_Panel SHALL use Streamlit as the frontend framework
2. THE Control_Panel SHALL use the Python `requests` library for HTTP communication with the Pitch_Server
3. THE Control_Panel SHALL use `langchain-nvidia-ai-endpoints` (specifically the ChatNVIDIA class) for LLM inference
4. THE Control_Panel SHALL use Pydantic (version 2.x) for defining the structured output schema
5. THE Control_Panel SHALL use `python-dotenv` for loading environment variables from the `.env` file
6. THE Control_Panel SHALL NOT use LangGraph, multi-agent orchestration libraries, or any library whose primary purpose is agent workflow routing
7. THE Control_Panel SHALL include a `requirements.txt` file in the `player` folder listing all direct dependencies with minimum version constraints (using `>=` notation)
8. THE Control_Panel SHALL require Python 3.10 or higher as specified in the project documentation
