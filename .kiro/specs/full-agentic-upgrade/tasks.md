# Implementation Plan: Full Agentic Upgrade

## Overview

This plan implements five agentic capabilities (episodic memory, multi-step planning, self-reflection, learning/adaptation, and inter-player communication) across both `player/` and `team/` applications. Each module is implemented independently in both apps per the application independence requirement. Integration wires the modules into the existing Look-Think-Act cycle, maintaining exactly one LLM call per cycle.

## Tasks

- [x] 1. Implement Episodic Memory module
  - [x] 1.1 Create `player/episodic_memory.py` with Episode dataclass and EpisodicMemory class
    - Define `Episode` dataclass with fields: cycle, game_state, action, next_state_delta, effectiveness
    - Implement `EpisodicMemory` using `collections.deque(maxlen=max_capacity)` for O(1) append with automatic eviction
    - Implement `add()`, `get_all()` (chronological order), `get_recent(n)`, and `__len__()` methods
    - Default max_capacity = 100
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6_

  - [x] 1.2 Create `team/episodic_memory.py` with identical Episode dataclass and EpisodicMemory class
    - Duplicate the implementation from player/ with no cross-package imports
    - Same interface: Episode dataclass, EpisodicMemory class with add/get_all/get_recent/__len__
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 10.2, 10.3_

  - [x] 1.3 Write property tests for EpisodicMemory in `player/tests/test_episodic_memory_properties.py`
    - **Property 1: Episode storage round-trip with chronological ordering**
    - **Property 2: Capacity invariant with oldest-first eviction**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**

  - [x] 1.4 Write property tests for EpisodicMemory in `team/tests/test_episodic_memory_properties.py`
    - **Property 1: Episode storage round-trip with chronological ordering**
    - **Property 2: Capacity invariant with oldest-first eviction**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**

- [x] 2. Implement Memory Summarizer module
  - [x] 2.1 Create `player/memory_summary.py` with `summarize_memory()` function
    - Format each episode as a single line: "Cycle {n}: {action_verb} → {outcome_class}"
    - Outcome classification: positive (effectiveness >= 0.6), neutral (0.3–0.6), negative (< 0.3)
    - Limit to at most 5 most recent episodes
    - Enforce 500 character max, truncating older episodes first while preserving the most recent
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 2.2 Create `team/memory_summary.py` with identical `summarize_memory()` function
    - Duplicate implementation from player/ with no cross-package imports
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 10.2, 10.3_

  - [x] 2.3 Write property tests for memory summarizer in `player/tests/test_memory_summary_properties.py`
    - **Property 3: Memory summary episode count limit**
    - **Property 4: Memory summary line format**
    - **Property 5: Memory summary truncation preserves most recent**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

  - [x] 2.4 Write property tests for memory summarizer in `team/tests/test_memory_summary_properties.py`
    - **Property 3: Memory summary episode count limit**
    - **Property 4: Memory summary line format**
    - **Property 5: Memory summary truncation preserves most recent**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Planner module
  - [x] 4.1 Create `player/planner.py` with SubGoal, Plan, PlanTemplate dataclasses and Planner class
    - Define `SubGoal` with description and target_condition callable
    - Define `Plan` with name, sub_goals (max 5), current_index, completed flag
    - Define `PlanTemplate` with name, trigger_condition, priority dict, sub_goals
    - Implement `Planner` with evaluate(), advance(), should_abandon() methods
    - Template library: score_goal, defend_goal, intercept_ball, distribute_ball
    - Template selection by highest priority for agent's position when multiple match
    - Plan replacement when higher-priority template matches
    - All logic in Python, no LLM calls
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 4.2 Create `team/planner.py` with identical Planner implementation
    - Duplicate implementation from player/ with no cross-package imports
    - _Requirements: 3.1–3.8, 4.1–4.6, 10.2, 10.3_

  - [x] 4.3 Write property tests for Planner in `player/tests/test_planner_properties.py`
    - **Property 6: Plan state machine advancement**
    - **Property 7: Plan abandonment on unachievable goal**
    - **Property 8: Plan replacement with higher-priority template**
    - **Property 9: Plan sub-goal count limit**
    - **Property 11: Template selection by highest priority**
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 4.1, 4.2, 4.3, 4.5**

  - [x] 4.4 Write property tests for Planner in `team/tests/test_planner_properties.py`
    - **Property 6: Plan state machine advancement**
    - **Property 7: Plan abandonment on unachievable goal**
    - **Property 8: Plan replacement with higher-priority template**
    - **Property 9: Plan sub-goal count limit**
    - **Property 11: Template selection by highest priority**
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 4.1, 4.2, 4.3, 4.5**

- [x] 5. Implement Reflection Engine module
  - [x] 5.1 Create `player/reflection.py` with ReflectionResult dataclass and ReflectionEngine class
    - Define `ReflectionResult` with effectiveness_score (0.0–1.0) and should_abandon_plan flag
    - Implement scoring formula: Δ ball_distance (weight 0.4) + Δ goal_distance (weight 0.4) + possession_change (weight 0.2)
    - Clamp raw score to [0.0, 1.0] before threshold checks
    - Track recent scores; signal abandonment when last 5 consecutive scores all < 0.3
    - Handle edge cases: missing previous state (skip, return None), division by zero (neutral 0.5)
    - All logic in Python, no LLM calls
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 5.2 Create `team/reflection.py` with identical ReflectionEngine implementation
    - Duplicate implementation from player/ with no cross-package imports
    - _Requirements: 5.1–5.7, 10.2, 10.3_

  - [x] 5.3 Write property tests for ReflectionEngine in `player/tests/test_reflection_properties.py`
    - **Property 12: Effectiveness score range invariant**
    - **Property 13: Effectiveness score stored in episode**
    - **Property 14: Abandonment signal on consecutive low scores**
    - **Validates: Requirements 5.3, 5.4, 5.5, 5.7**

  - [x] 5.4 Write property tests for ReflectionEngine in `team/tests/test_reflection_properties.py`
    - **Property 12: Effectiveness score range invariant**
    - **Property 13: Effectiveness score stored in episode**
    - **Property 14: Abandonment signal on consecutive low scores**
    - **Validates: Requirements 5.3, 5.4, 5.5, 5.7**

- [x] 6. Implement Strategy Tracker module
  - [x] 6.1 Create `player/strategy_tracker.py` with PatternEntry, AdaptationRecord dataclasses and StrategyTracker class
    - Define `PatternEntry` with opponent_positions, ball_position, effectiveness
    - Define `AdaptationRecord` with observed_pattern, counter_strategy, confidence
    - Implement `StrategyTracker` with record(), analyze(), get_active_adaptations(max_count=2), reset_for_new_match()
    - Analysis: compute directional frequency distributions; generate AdaptationRecord when direction bucket > 70% confidence
    - reset_for_new_match() retains AdaptationRecords, clears raw pattern entries
    - All logic in Python, no LLM calls
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 6.2 Create `team/strategy_tracker.py` with identical StrategyTracker implementation
    - Duplicate implementation from player/ with no cross-package imports
    - _Requirements: 6.1–6.7, 10.2, 10.3_

  - [x] 6.3 Write property tests for StrategyTracker in `player/tests/test_strategy_tracker_properties.py`
    - **Property 15: Strategy tracker pattern recording**
    - **Property 16: Directional analysis produces adaptation above confidence threshold**
    - **Property 17: Active adaptations count limit**
    - **Property 18: Match reset preserves adaptations but clears raw entries**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6**

  - [x] 6.4 Write property tests for StrategyTracker in `team/tests/test_strategy_tracker_properties.py`
    - **Property 15: Strategy tracker pattern recording**
    - **Property 16: Directional analysis produces adaptation above confidence threshold**
    - **Property 17: Active adaptations count limit**
    - **Property 18: Match reset preserves adaptations but clears raw entries**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Context Assembler module
  - [x] 8.1 Create `player/context_assembler.py` with `assemble_agentic_context()` function
    - Accept memory_summary, plan_step, adaptation_hints parameters
    - Priority-based truncation order (lowest first): memory summary → adaptation hints → plan step
    - Token estimation: ~4 characters per token, max 300 tokens
    - Return empty string when all components are empty
    - _Requirements: 9.3, 9.4_

  - [x] 8.2 Create `team/context_assembler.py` with `assemble_agentic_context()` function including signals parameter
    - Same as player/ version but with additional `signals` parameter
    - Priority order (highest first): plan step → signals → adaptation hints → memory summary
    - _Requirements: 9.3, 9.4_

  - [x] 8.3 Write property tests for Context Assembler in `player/tests/test_context_assembler_properties.py`
    - **Property 10: Plan context inclusion iff plan is active**
    - **Property 24: Context priority-based truncation**
    - **Validates: Requirements 3.9, 3.10, 9.3, 9.4**

  - [x] 8.4 Write property tests for Context Assembler in `team/tests/test_context_assembler_properties.py`
    - **Property 10: Plan context inclusion iff plan is active**
    - **Property 24: Context priority-based truncation**
    - **Validates: Requirements 3.9, 3.10, 9.3, 9.4**

- [x] 9. Implement Signal Bus module (team/ only)
  - [x] 9.1 Create `team/signal_bus.py` with Signal dataclass and SignalBus class
    - Define `Signal` with sender_position, signal_type, payload (max 50 chars), timestamp
    - Implement `SignalBus` with publish(), read_all(exclude_position), clear() methods
    - Thread safety: `threading.Semaphore(4)` for read and write concurrency limits, `threading.Lock` for internal dict
    - Retain only most recent signal per sender position
    - Reject signals with payload > 50 characters
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6, 7.7_

  - [x] 9.2 Write property tests for SignalBus in `team/tests/test_signal_bus_properties.py`
    - **Property 19: Signal Bus publish/read/replace invariant**
    - **Property 20: Dead ball clears all signals**
    - **Validates: Requirements 7.1, 7.2, 7.3, 8.5**

- [x] 10. Implement Signal Generator module (team/ only)
  - [x] 10.1 Create `team/signal_generator.py` with SignalGenerator class
    - Implement `generate()` method accepting plan, game_state, team, position
    - Rules: requesting_pass when sub-goal benefits from teammate awareness and not dead ball
    - ready_to_pass when in kick range (ball_distance <= 30) and teammate making a run
    - supporting when nearest to ball carrier, with current zone in payload
    - Return None during dead ball situations
    - All logic in Python, no LLM calls
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 10.2 Write property tests for SignalGenerator in `team/tests/test_signal_generator_properties.py`
    - **Property 21: Signal generation from awareness-benefiting sub-goals**
    - **Property 22: Ready-to-pass signal generation**
    - **Property 23: Supporting signal generation**
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Integrate agentic modules into player/ agent loop
  - [x] 12.1 Modify `player/agent_loop.py` to integrate episodic memory, planner, reflection, strategy tracker, and context assembler
    - Add instance attributes for EpisodicMemory, Planner, ReflectionEngine, StrategyTracker, and cycle counter
    - After `_look()`: evaluate active plan via Planner, run ReflectionEngine on previous action, store episode in memory
    - Before `_think()`: assemble agentic context (memory summary + plan step + adaptation hints) via context_assembler
    - Append agentic context to `enriched_state` after spatial summary (Req 2.5)
    - After `_act()`: record new episode, record pattern entry in StrategyTracker
    - Include current sub-goal description in prompt when plan is active (Req 3.9), omit when no plan (Req 3.10)
    - Maintain exactly one LLM call per cycle
    - _Requirements: 2.5, 3.2, 3.6, 3.9, 3.10, 5.1, 5.5, 9.1, 9.2, 10.1_

  - [x] 12.2 Write integration tests in `player/tests/test_agentic_integration.py`
    - Test single LLM call per cycle with mocked LLM client
    - Test no LLM calls from agentic modules
    - Test agentic context appended after spatial analysis
    - _Requirements: 9.1, 9.2_

- [x] 13. Integrate agentic modules into team/ player agent
  - [x] 13.1 Modify `team/player_agent.py` to integrate all agentic modules including Signal Bus
    - Add instance attributes for EpisodicMemory, Planner, ReflectionEngine, StrategyTracker, and cycle counter
    - Accept SignalBus instance in constructor (shared across all PlayerAgents)
    - After look: evaluate plan, run reflection, store episode
    - Before think: read signals from SignalBus, assemble agentic context (memory + plan + adaptations + signals)
    - Append agentic context in `_build_messages()` after spatial analysis block
    - After act: record episode, record pattern in StrategyTracker, generate and publish signal via SignalGenerator
    - Clear SignalBus on dead ball detection
    - Maintain exactly one LLM call per cycle
    - _Requirements: 2.5, 3.2, 3.6, 3.9, 3.10, 5.1, 5.5, 7.2, 7.4, 8.1, 9.1, 9.2, 10.2_

  - [x] 13.2 Modify `team/orchestrator.py` to instantiate shared SignalBus and pass to PlayerAgents
    - Create single SignalBus instance in orchestrator
    - Pass SignalBus reference to each PlayerAgent constructor
    - _Requirements: 7.6_

- [x] 14. Integrate adaptation data into Coach Agent (team/ only)
  - [x] 14.1 Modify `team/coach_agent.py` to aggregate player adaptation data into coaching prompt
    - Accept references to player StrategyTrackers in constructor
    - In `_build_prompt()`: collect active AdaptationRecords from all players (at most 1 sentence per player)
    - Detect shared opponent tendencies across multiple players and issue coordinated instructions
    - Limit additional adaptation context to 200 tokens
    - Do not directly modify player memories, plans, or trackers; communicate via InstructionStore only
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [x] 14.2 Write property tests for Coach integration in `team/tests/test_coach_integration_properties.py`
    - **Property 25: Coach adaptation summary within token limit**
    - **Property 26: Coach coordinated instructions on shared tendency**
    - **Validates: Requirements 11.1, 11.2, 11.4**

  - [x] 14.3 Write integration tests in `team/tests/test_agentic_integration.py`
    - Test single LLM call per player cycle with mocked LLM client
    - Test no LLM calls from agentic modules
    - Test Signal Bus thread safety with concurrent readers/writers
    - Test application independence (no cross-package imports via static analysis)
    - _Requirements: 7.5, 9.1, 9.2, 10.1, 10.2_

- [x] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All modules are implemented in Python using dataclasses consistent with existing codebase patterns
- The `player/` and `team/` implementations are fully independent with no shared imports (Req 10)
- The design mandates duplicated implementations rather than a shared library

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "1.4", "2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "2.4", "4.1", "4.2", "5.1", "5.2", "6.1", "6.2"] },
    { "id": 3, "tasks": ["4.3", "4.4", "5.3", "5.4", "6.3", "6.4", "8.1", "8.2", "9.1"] },
    { "id": 4, "tasks": ["8.3", "8.4", "9.2", "10.1"] },
    { "id": 5, "tasks": ["10.2", "12.1"] },
    { "id": 6, "tasks": ["12.2", "13.1", "13.2"] },
    { "id": 7, "tasks": ["14.1"] },
    { "id": 8, "tasks": ["14.2", "14.3"] }
  ]
}
```
