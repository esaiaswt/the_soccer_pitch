# Implementation Plan: Agent Control Panel

## Overview

Build a Streamlit-based AI agent control panel in a `player/` folder that communicates with the existing Pitch game server via REST API. The agent uses NVIDIA NIM (ChatNVIDIA) for LLM-based decision making in a continuous Look-Think-Act loop. Implementation proceeds from project scaffolding, through core models and utilities, to the agent loop, LLM integration, and finally the Streamlit UI wiring.

## Tasks

- [x] 1. Set up project structure and configuration
  - [x] 1.1 Create player/ folder with project scaffolding
    - Create `player/` directory at workspace root
    - Create `player/.gitignore` with entries for `.env`, `__pycache__/`, `*.pyc`, `venv/`, `soccer_a/`, `agent.log`
    - Create `player/.env` with placeholder `NVIDIA_API_KEY=<your-key-here>`
    - Create `player/requirements.txt` listing: `streamlit>=1.28`, `requests>=2.31`, `langchain-nvidia-ai-endpoints>=0.1`, `pydantic>=2.0`, `python-dotenv>=1.0`, `hypothesis>=6.82`, `pytest>=7.4`, `pytest-mock>=3.11`, `requests-mock>=1.11`
    - Create `player/tests/__init__.py` (empty)
    - _Requirements: 1.1, 1.3, 1.4, 12.1-12.8_

  - [x] 1.2 Implement config.py with constants and ActionModel
    - Create `player/config.py` with all constants: `DEFAULT_SERVER_IP`, `SERVER_PORT`, `REQUEST_TIMEOUT`, `LLM_TIMEOUT`, `LOOP_DELAY`, `MAX_AGENT_NAME_LENGTH`, `MAX_SYSTEM_PROMPT_LENGTH`, `MAX_BEHAVIOR_OVERRIDE_LENGTH`, `DEFAULT_SYSTEM_PROMPT`, `TEAMS`, `POSITIONS`
    - Define `ActionModel` Pydantic BaseModel with `dx` (float, ge=-1.0, le=1.0), `dy` (float, ge=-1.0, le=1.0), `kick` (bool)
    - Define `BRAKE_ACTION = ActionModel(dx=0.0, dy=0.0, kick=False)`
    - Add API key validation function that rejects empty, None, or whitespace-only values
    - Add URL construction helper function: `build_url(server_ip: str, endpoint: str) -> str`
    - _Requirements: 1.5, 2.5, 4.1_

  - [x] 1.3 Write property tests for config module (Properties 1, 2, 3)
    - **Property 1: API key validation rejects all empty-like values**
    - **Validates: Requirements 1.5**
    - **Property 2: URL construction produces correct format**
    - **Validates: Requirements 2.5**
    - **Property 3: ActionModel accepts valid ranges and rejects invalid ranges**
    - **Validates: Requirements 4.1**
    - Create `player/tests/test_config_properties.py` and `player/tests/test_url_properties.py`

  - [x] 1.4 Implement logging_config.py
    - Create `player/logging_config.py` with `setup_logging()` function
    - Configure file-based logging to `agent.log` with append mode
    - Use ISO 8601 timestamp format (`YYYY-MM-DDTHH:MM:SS`)
    - Log format: `{ISO_8601_TIMESTAMP} | {LEVEL} | {message}`
    - _Requirements: 11.1, 11.2, 11.3_

  - [x] 1.5 Write property tests for logging (Property 11)
    - **Property 11: Log entries contain ISO 8601 timestamps and required event data**
    - **Validates: Requirements 11.2, 11.3**
    - Create `player/tests/test_logging_properties.py`

- [x] 2. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement LLM client module
  - [x] 3.1 Implement llm_client.py with ChatNVIDIA integration
    - Create `player/llm_client.py`
    - Implement `create_llm_client(model: str = "meta/llama3-8b-instruct") -> StructuredLLM` that initializes ChatNVIDIA and binds ActionModel via `.with_structured_output()`
    - Implement `invoke_llm(client, system_prompt, game_state_json, behavior_override="", timeout=10.0) -> ActionModel` that assembles messages and invokes the LLM
    - Implement message assembly: system message = system_prompt; user message = game_state_json + (newline + behavior_override if non-empty)
    - Handle timeout via threading or signal mechanism (10s hard limit)
    - _Requirements: 4.2, 4.3, 4.4, 4.6_

  - [x] 3.2 Write property tests for message assembly (Property 6)
    - **Property 6: LLM message assembly with optional behavior override**
    - **Validates: Requirements 4.4, 4.6**
    - Create `player/tests/test_message_properties.py`

  - [x] 3.3 Write unit tests for LLM client setup
    - Test that `create_llm_client` initializes with correct model parameter
    - Test that structured output binding is applied
    - Create `player/tests/test_llm_client_unit.py`
    - _Requirements: 4.2, 4.3_

- [x] 4. Implement agent loop module
  - [x] 4.1 Implement agent_loop.py with Look-Think-Act cycle
    - Create `player/agent_loop.py`
    - Define `IterationResult` dataclass with fields: `game_state`, `action`, `fallback_reason`, `error_details`, `timestamp`
    - Implement `AgentLoop` class with constructor accepting: `server_ip`, `team`, `position`, `llm_client`, `get_system_prompt` callable, `get_behavior_override` callable, `on_iteration` callback, `stop_event` (threading.Event)
    - Implement `run()` method with the main loop: look → think → act → sleep 1.5s, stopping on `stop_event`
    - **Look step**: GET `/api/state` with 5s timeout; on any failure return `IterationResult` with `BRAKE_ACTION` and fallback reason
    - **Think step**: invoke LLM with system prompt + game state + optional override; on any exception or empty/None response, return `BRAKE_ACTION`; if system prompt is empty/whitespace, return `BRAKE_ACTION` with "system prompt required" reason
    - **Act step**: POST `/api/action` with JSON payload `{team, position, vector: {dx, dy}, kick}`; log errors but continue loop
    - Validate configuration (server_ip non-empty, team selected) before starting loop; raise error if missing
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 7.4, 7.5, 8.2, 8.3, 8.5, 9.4, 9.5_

  - [x] 4.2 Write property tests for agent loop (Properties 4, 5, 7, 8, 9, 10)
    - **Property 4: Game state parsing extracts all required fields**
    - **Validates: Requirements 3.2**
    - **Property 5: Look step errors always produce Brake_Action**
    - **Validates: Requirements 3.4**
    - **Property 7: LLM invocation failure always produces Brake_Action**
    - **Validates: Requirements 4.5, 7.1, 7.2, 7.3, 7.5**
    - **Property 8: Action payload construction preserves ActionModel values**
    - **Validates: Requirements 5.1, 5.2**
    - **Property 9: Missing configuration prevents loop start**
    - **Validates: Requirements 8.5**
    - **Property 10: Empty system prompt produces Brake_Action**
    - **Validates: Requirements 9.5**
    - Create `player/tests/test_agent_loop_properties.py`, `player/tests/test_parsing_properties.py`, `player/tests/test_payload_properties.py`

  - [x] 4.3 Write unit tests for agent loop control
    - Test thread start/stop lifecycle via stop_event
    - Test rate limiting (1.5s delay between iterations)
    - Test that on_iteration callback is invoked each cycle
    - Create `player/tests/test_agent_loop_unit.py`
    - _Requirements: 6.1, 6.2, 6.3, 8.2, 8.3_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Streamlit dashboard
  - [x] 6.1 Implement app.py with UI layout and session state
    - Create `player/app.py` as the Streamlit entry point
    - Load environment variables via `python-dotenv`; validate `NVIDIA_API_KEY` at startup (exit with error if missing/empty)
    - Initialize session state keys: `agent_thread`, `stop_event`, `latest_iteration`, `is_running`
    - **Sidebar**: Server IP text input (default "localhost"), Team dropdown ("Red"/"Blue"), Position dropdown (Striker/Goalkeeper/Midfielder/Defender), Agent Name text input (max 50 chars)
    - **Main area top**: Start/Stop Auto-Play toggle button, status indicator (running/stopped)
    - **Main area middle**: System Prompt text area (min 6 rows, max 2000 chars, pre-filled with DEFAULT_SYSTEM_PROMPT), Behavior Override single-line input (max 500 chars)
    - **Main area bottom**: Debug console showing latest game state JSON and LLM response/action; display fallback reason when Brake_Action is used
    - _Requirements: 1.5, 2.1, 2.2, 2.3, 2.4, 8.1, 8.4, 9.1, 9.2, 9.3, 9.6, 10.1, 10.2, 10.3, 10.4_

  - [x] 6.2 Wire agent loop thread lifecycle to UI toggle
    - On toggle activation: validate server_ip and team are configured (show error if not); create `threading.Event` for stop signal; instantiate `AgentLoop` with current config; start background thread running `agent_loop.run()`; set `is_running = True`
    - On toggle deactivation: set stop_event; wait for thread to complete (max 30s); set `is_running = False`
    - Pass `get_system_prompt` and `get_behavior_override` callables that read current widget values so changes take effect on next iteration
    - Pass `on_iteration` callback that updates `st.session_state.latest_iteration` for debug console display
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 9.4_

  - [x] 6.3 Write unit tests for app UI behavior
    - Test initial widget defaults (toggle off, default IP, default prompt)
    - Test that missing config blocks loop start
    - Test debug console updates on iteration callback
    - Create `player/tests/test_app_unit.py`
    - _Requirements: 2.1, 8.1, 8.5, 10.3_

- [x] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The project uses Python with Hypothesis for property-based testing and pytest as the test runner
- All code lives in the `player/` folder, independent of the existing `pitch/` game server

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4"] },
    { "id": 2, "tasks": ["1.3", "1.5", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "4.1"] },
    { "id": 4, "tasks": ["4.2", "4.3"] },
    { "id": 5, "tasks": ["6.1"] },
    { "id": 6, "tasks": ["6.2"] },
    { "id": 7, "tasks": ["6.3"] }
  ]
}
```
