# Implementation Plan: Multi-Agent Team

## Overview

Implement a multi-agent soccer team application in `team/` that orchestrates a Coach agent and four Player sub-agents (Goalkeeper, Defender, Midfielder, Striker) to play coordinated soccer on the existing Pitch server. The implementation follows a bottom-up approach: configuration and shared data structures first, then individual agent threads, then orchestration, and finally the Streamlit dashboard.

## Tasks

- [x] 1. Set up project structure and configuration
  - [x] 1.1 Create `team/` directory structure with `__init__.py` files, `requirements.txt`, `.env.example`, and `team/tests/` directory
    - Create `team/`, `team/tests/`, `team/tests/__init__.py`
    - Create `team/requirements.txt` with pinned dependencies: langchain-nvidia-ai-endpoints, python-dotenv, requests, streamlit, hypothesis, pytest, pytest-mock, requests-mock
    - Create `team/.env.example` with all configurable parameters documented
    - _Requirements: 10.1, 10.4_

  - [x] 1.2 Implement `team/config.py` with `TeamConfig` dataclass and `.env` loading
    - Define frozen `TeamConfig` dataclass with all fields: pitch_host, pitch_port, nvidia_api_key, coach_model, player_model, coaching_frequency, poll_interval, streamlit_port, team_color, coach_memory_size
    - Implement `load_config()` function that reads from `team/.env` using python-dotenv
    - Validate ranges: coaching_frequency (2-30s), poll_interval (0.1-10s), streamlit_port (1024-65535)
    - Exit with error message if nvidia_api_key is missing/empty or any parameter is out of range
    - Apply defaults: pitch_host="localhost", pitch_port=8000, coaching_frequency=7, poll_interval=1, coach_memory_size=50
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 1.3 Write property test for configuration validation
    - **Property 10: Configuration parameter validation**
    - **Validates: Requirements 9.2, 3.5**

- [x] 2. Implement shared data containers
  - [x] 2.1 Implement `team/shared_state.py` with thread-safe `SharedState` class
    - Use `threading.Lock` for atomic get/set of the game state snapshot
    - Implement `get_snapshot()`, `set_snapshot(snapshot)`, `get_last_update_time()`
    - Store timestamp of last successful update
    - _Requirements: 1.2, 1.3_

  - [x] 2.2 Write property test for state snapshot propagation
    - **Property 1: State snapshot propagation**
    - **Validates: Requirements 1.2**

  - [x] 2.3 Implement `team/instruction_store.py` with `CoachInstruction` dataclass and `InstructionStore` class
    - Define `CoachInstruction` dataclass with content (str), timestamp (float), target_position (str)
    - Implement thread-safe `InstructionStore` with `set_instruction()`, `get_instruction()`, `get_all_instructions()`
    - Use `threading.Lock` for concurrent access safety
    - _Requirements: 3.4, 4.4_

  - [x] 2.4 Write property test for instruction delivery integrity
    - **Property 5: Instruction delivery integrity**
    - **Validates: Requirements 3.4**

  - [x] 2.5 Implement `team/debug_store.py` with `PlayerDebugInfo` dataclass and `DebugStore` class
    - Define `PlayerDebugInfo` dataclass with latest_state, latest_action, latest_instruction, last_update
    - Implement thread-safe `DebugStore` with `update_player()`, `get_player()`, `update_coach()`, `get_coach()`
    - _Requirements: 6.6, 6.7_

- [x] 3. Implement logging infrastructure
  - [x] 3.1 Implement `team/logging_config.py` with structured, thread-safe logging
    - Configure logging to write to team-specific log file (e.g., `team_red.log`, `team_blue.log`)
    - Use ISO 8601 timestamps with microsecond precision
    - Format: `{timestamp} | {level} | {agent_identity} | {message} | {structured_context}`
    - Ensure thread-safe logging with `logging.handlers` or lock-based approach
    - Implement helper functions for logging coach instructions, token usage, decision latency, and errors
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 8.5_

  - [x] 3.2 Write property test for structured log completeness
    - **Property 12: Structured log completeness**
    - **Validates: Requirements 7.1, 7.2, 7.5**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement State Poller
  - [x] 5.1 Implement `team/state_poller.py` with `StatePoller` class
    - Create thread target `run()` method that polls `GET /api/state` at configurable interval
    - Update `SharedState` on successful response
    - Handle HTTP errors and connection timeouts (5s limit) by logging and preserving last good snapshot
    - Respect `stop_event` for clean shutdown within one polling interval
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 5.2 Write property test for State Poller error resilience
    - **Property 2: State Poller error resilience with snapshot preservation**
    - **Validates: Requirements 1.3**

- [x] 6. Implement Coach Agent
  - [x] 6.1 Implement `team/coach_agent.py` with `CoachMemory` class
    - Implement rolling buffer with configurable max size (default 50)
    - Maintain chronological insertion order
    - Discard oldest snapshot when buffer is full
    - Validate snapshots have required fields before adding (ball, players, score, time_left, match_state)
    - Provide `get_history()` and `get_recent(n)` methods
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 6.2 Write property tests for Coach Memory
    - **Property 3: Coach Memory buffer invariants**
    - **Property 4: Invalid snapshot rejection**
    - **Validates: Requirements 2.2, 2.3, 2.5**

  - [x] 6.3 Implement `CoachAgent` class with LLM-powered tactical instruction generation
    - Create thread target `run()` method that loops at Coaching_Frequency interval
    - Read SharedState and CoachMemory for context
    - Invoke Coach LLM (ChatNVIDIA with configured coach_model) to generate per-player instructions
    - Store instructions in InstructionStore with current timestamp
    - Handle LLM failures gracefully: log error, skip cycle, continue
    - Log each instruction with structured format (timestamp, target player, content)
    - Log token usage and decision latency
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 7.1, 7.2, 7.3, 7.4_

  - [x] 6.4 Write property test for Coach error resilience
    - **Property 6: Coach error resilience**
    - **Validates: Requirements 3.6**

- [x] 7. Implement Player Agent
  - [x] 7.1 Implement `team/player_agent.py` with `PlayerAgent` class
    - Create thread target `run()` method with Look-Think-Act loop (1.5s cycle)
    - Read SharedState for current game state
    - Read InstructionStore for latest Coach Instruction (include if not stale)
    - Detect stale instructions: exclude if timestamp > 3 × coaching_frequency old
    - Invoke Player LLM (ChatNVIDIA with configured player_model) with 10s timeout
    - Parse LLM response into action (dx, dy, kick) and POST to Pitch server
    - Submit Brake_Action on LLM timeout or error
    - Update DebugStore with latest state, action, and instruction
    - Log token usage, decision latency, and errors with structured context
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 5.5, 7.2, 7.3, 7.4, 7.5_

  - [x] 7.2 Write property test for Coach instruction staleness detection
    - **Property 8: Coach instruction staleness detection**
    - **Validates: Requirements 5.1**

  - [x] 7.3 Write property test for Player Brake Action on LLM failure
    - **Property 9: Player Brake Action on LLM failure**
    - **Validates: Requirements 5.2, 5.3**

  - [x] 7.4 Write property test for Coach instruction inclusion in player context
    - **Property 7: Coach instruction inclusion in player context**
    - **Validates: Requirements 4.4**

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Team Orchestrator
  - [x] 9.1 Implement `team/orchestrator.py` with `TeamOrchestrator` class
    - Implement `start()`: create and launch State Poller thread, Coach Agent thread, and four Player Agent threads
    - Implement `stop(timeout=30.0)`: signal all threads via stop_event, join with timeout, report status
    - Implement `is_running()` to check thread health
    - Ensure no shared mutable state between instances for multi-instance support
    - _Requirements: 4.1, 6.2, 8.1_

  - [x] 9.2 Implement port auto-assignment logic in orchestrator or dashboard launcher
    - Scan ports 8501-8510 for first available port when no port configured
    - Use configured port from STREAMLIT_PORT env var if set
    - Display error and refuse to start if configured/auto-assigned port is in use
    - _Requirements: 8.2, 8.3, 8.4, 9.7_

  - [x] 9.3 Write property test for port auto-assignment
    - **Property 11: Port auto-assignment**
    - **Validates: Requirements 8.2**

- [x] 10. Implement Streamlit Dashboard
  - [x] 10.1 Implement `team/app.py` with Streamlit-based Team Dashboard
    - Team selection control (Red/Blue) with validation (error if not selected before start)
    - Start/Stop buttons that invoke TeamOrchestrator.start()/stop()
    - Per-player free-text override input fields (max 500 chars)
    - Team-level tactical override input field for Coach (max 500 chars)
    - Debug panel per Player showing latest state, action, and coach instruction from DebugStore
    - Coach memory/history view showing 10 most recent observations and instructions
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 11. Integration wiring and final validation
  - [x] 11.1 Create `team/main.py` entry point that loads config, creates orchestrator, and launches Streamlit
    - Load config from `team/.env`
    - Validate all parameters before starting
    - Wire all components together (SharedState, InstructionStore, DebugStore, Orchestrator)
    - Launch Streamlit app with correct port
    - _Requirements: 9.1, 10.1, 10.2, 10.3_

  - [x] 11.2 Write integration tests for end-to-end data flow
    - Test State Poller → SharedState → Player Agent flow with mock Pitch server
    - Test Coach Agent → InstructionStore → Player Agent instruction delivery
    - Test multi-thread concurrent access to SharedState and InstructionStore
    - Test two-instance isolation (no shared mutable state)
    - _Requirements: 1.2, 3.1, 4.4, 8.1_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation uses Python with LangChain (langchain-nvidia-ai-endpoints), Streamlit, and Hypothesis
- All code resides in `team/` directory, fully isolated from `pitch/` and `player/`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1", "2.3", "2.5", "3.1"] },
    { "id": 3, "tasks": ["2.2", "2.4", "3.2", "5.1"] },
    { "id": 4, "tasks": ["5.2", "6.1"] },
    { "id": 5, "tasks": ["6.2", "6.3"] },
    { "id": 6, "tasks": ["6.4", "7.1"] },
    { "id": 7, "tasks": ["7.2", "7.3", "7.4", "9.1"] },
    { "id": 8, "tasks": ["9.2"] },
    { "id": 9, "tasks": ["9.3", "10.1"] },
    { "id": 10, "tasks": ["11.1"] },
    { "id": 11, "tasks": ["11.2"] }
  ]
}
```
