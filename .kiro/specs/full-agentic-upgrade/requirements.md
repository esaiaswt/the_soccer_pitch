# Requirements Document

## Introduction

This feature upgrades both the `player/` (single-agent) and `team/` (multi-agent) soccer applications to be "fully agentic" by adding episodic memory, multi-step planning, self-reflection, learning/adaptation, and inter-player communication. All new capabilities are implemented primarily in Python code to avoid increasing LLM API call rates, with the LLM receiving only summarized context to keep token counts manageable.

## Glossary

- **Episodic_Memory**: A per-player Python data structure that stores past game states, actions taken, and outcomes as timestamped episodes, with configurable maximum capacity and automatic eviction of oldest entries.
- **Plan**: A multi-step sequence of sub-goals decomposed from a high-level objective (e.g., "score a goal" → position behind ball → approach ball → kick), maintained in Python and executed across multiple Look-Think-Act cycles.
- **Reflection_Engine**: A Python module that evaluates the effectiveness of the most recent action by comparing expected outcomes against actual game state changes, producing an effectiveness score without additional LLM calls.
- **Strategy_Tracker**: A Python module that records which tactical patterns succeed or fail during a match and produces adaptation recommendations based on observed opponent behavior.
- **Signal_Bus**: A thread-safe communication channel in the `team/` application that allows Player agents to broadcast short intention messages to teammates without requiring LLM calls.
- **Memory_Summary**: A compact text representation of relevant episodic memory entries, formatted for inclusion in the LLM prompt without exceeding token budgets.
- **Player_Agent**: An LLM-powered agent running a Look-Think-Act loop that controls a single soccer player position.
- **Coach_Agent**: An LLM-powered agent in the `team/` application that generates tactical instructions for all Player agents.
- **Pitch_Server**: The REST API server providing game state (GET /api/state) and accepting actions (POST /api/action).
- **Look-Think-Act_Cycle**: The repeating loop where an agent reads game state, invokes the LLM for a decision, and posts an action.
- **Effectiveness_Score**: A numeric value between 0.0 and 1.0 computed by the Reflection_Engine indicating how well the last action achieved its intended outcome.
- **Adaptation_Record**: A structured entry in the Strategy_Tracker containing an observed opponent pattern, the recommended counter-strategy, and a confidence score.

## Requirements

### Requirement 1: Episodic Memory Storage

**User Story:** As a developer, I want each Player_Agent to maintain an episodic memory of past game states, actions, and outcomes, so that the agent can make decisions informed by recent history.

#### Acceptance Criteria

1. THE Episodic_Memory SHALL store each episode as a record containing the game state snapshot, the action taken, and the resulting next-state delta.
2. WHEN a new episode is added and the Episodic_Memory has reached its configured maximum capacity, THE Episodic_Memory SHALL evict the oldest episode to make room.
3. THE Episodic_Memory SHALL support a configurable maximum capacity with a default of 100 episodes.
4. WHEN the Episodic_Memory is queried, THE Episodic_Memory SHALL return episodes in chronological order from oldest to newest.
5. THE Episodic_Memory SHALL operate independently in `player/` and `team/` applications without shared state between the two codebases.
6. THE Episodic_Memory SHALL add a new episode within O(1) amortized time complexity to avoid impacting the Look-Think-Act_Cycle timing.

### Requirement 2: Memory Summarization for LLM Context

**User Story:** As a developer, I want episodic memory to be summarized into a compact text format before being sent to the LLM, so that token counts remain manageable and the LLM receives relevant context.

#### Acceptance Criteria

1. WHEN the LLM prompt is constructed, THE Memory_Summary SHALL contain at most 5 of the most relevant recent episodes.
2. THE Memory_Summary SHALL format each episode as a single line containing the cycle number, action taken, and outcome classification (positive, neutral, or negative).
3. THE Memory_Summary SHALL not exceed 500 characters in total length.
4. IF the Memory_Summary would exceed 500 characters, THEN THE Memory_Summary SHALL truncate older episodes first while preserving the most recent episode.
5. THE Memory_Summary SHALL be appended to the existing spatial analysis section in the LLM prompt without replacing existing context.

### Requirement 3: Multi-Step Planning

**User Story:** As a developer, I want each Player_Agent to decompose high-level goals into multi-step plans and execute them across multiple cycles, so that agents exhibit purposeful multi-turn behavior.

#### Acceptance Criteria

1. THE Plan SHALL consist of an ordered sequence of sub-goals, where each sub-goal has a target condition that can be evaluated against the current game state.
2. WHEN a Player_Agent begins a new Look-Think-Act_Cycle and has an active Plan, THE Player_Agent SHALL evaluate whether the current sub-goal's target condition is satisfied.
3. WHEN the current sub-goal's target condition is satisfied, THE Plan SHALL advance to the next sub-goal in the sequence.
4. WHEN the Player_Agent reaches the final sub-goal position and its target condition is satisfied, THE Plan SHALL be marked as completed and cleared from the Player_Agent.
5. IF the game state changes such that the Plan's high-level goal is no longer achievable (e.g., opponent gains possession), THEN THE Player_Agent SHALL abandon the current Plan.
6. THE Plan evaluation and advancement logic SHALL execute in Python without making additional LLM API calls during active plan execution; LLM calls SHALL be permitted during cleanup of inactive plans.
7. WHEN an agent already has an active Plan and game conditions change to match a different template with higher priority, THE Player_Agent SHALL replace the active Plan with the new matching template immediately.
8. THE Plan SHALL contain at most 5 sub-goals to keep execution tractable within the cycle timing constraints.
9. WHEN the LLM is invoked and a Plan is active, THE Player_Agent SHALL include the current active sub-goal description in the prompt context so the LLM can make decisions aligned with the plan.
10. WHEN the LLM is invoked and no Plan is active, THE Player_Agent SHALL skip including sub-goal context in the prompt entirely.

### Requirement 4: Plan Generation

**User Story:** As a developer, I want plans to be generated from predefined tactical templates based on game state conditions, so that planning does not require extra LLM calls.

#### Acceptance Criteria

1. THE Player_Agent SHALL select a Plan from a library of predefined plan templates based on the current game state conditions (ball possession, field position, score differential).
2. WHEN the Player_Agent has no active Plan and the game state matches a template's trigger condition, THE Player_Agent SHALL instantiate a new Plan from that template.
3. WHEN the Player_Agent already has an active Plan and the game state changes to match a different template, THE Player_Agent SHALL replace the active Plan with the new matching template immediately.
3. THE plan template library SHALL include at minimum templates for: score a goal, defend own goal, intercept ball, and distribute ball (goalkeeper).
4. WHEN multiple plan templates match the current game state, THE Player_Agent SHALL select the template with the highest priority for the agent's position.
5. THE plan template selection logic SHALL execute in Python without making LLM API calls.

### Requirement 5: Self-Reflection After Actions

**User Story:** As a developer, I want each Player_Agent to evaluate the effectiveness of its last action after each cycle, so that the agent can detect when its strategy is failing.

#### Acceptance Criteria

1. WHEN a Look-Think-Act_Cycle completes and the next game state is available, THE Reflection_Engine SHALL compute an Effectiveness_Score by comparing the expected outcome of the action against the actual state change.
2. THE Reflection_Engine SHALL compute the Effectiveness_Score using measurable criteria: change in ball distance, change in goal distance, and whether possession changed.
3. THE Effectiveness_Score SHALL be a value between 0.0 (action had no positive effect or was counterproductive) and 1.0 (action fully achieved its intended outcome).
4. WHEN the Reflection_Engine computes a raw score outside the 0.0–1.0 range, THE Reflection_Engine SHALL clamp the value to the nearest boundary (0.0 or 1.0) before applying abandonment threshold checks.
5. THE Reflection_Engine SHALL store the Effectiveness_Score in the corresponding Episodic_Memory episode.
6. THE Reflection_Engine SHALL execute entirely in Python using arithmetic comparisons on game state values without making LLM API calls.
7. IF the Effectiveness_Score for the last 5 consecutive actions is below 0.3, THEN THE Reflection_Engine SHALL signal the Player_Agent to abandon its current Plan.

### Requirement 6: Learning and Adaptation Within a Match

**User Story:** As a developer, I want each Player_Agent to track which strategies work against the current opponent and adapt its behavior accordingly, so that agents improve during a match.

#### Acceptance Criteria

1. THE Strategy_Tracker SHALL record each action's context (opponent positions, ball position) and outcome (Effectiveness_Score) as a pattern entry.
2. WHEN the Strategy_Tracker has accumulated at least 10 pattern entries, THE Strategy_Tracker SHALL analyze opponent tendencies by computing directional frequency distributions of opponent movements.
3. WHEN an opponent tendency is detected with confidence above 0.7, THE Strategy_Tracker SHALL produce an Adaptation_Record containing the observed pattern and recommended counter-strategy.
4. THE Strategy_Tracker SHALL include at most 2 active Adaptation_Records in the LLM prompt context to avoid token bloat.
5. THE Strategy_Tracker SHALL execute pattern analysis in Python using statistical aggregation without making LLM API calls.
6. WHEN the match resets (new kickoff detected) and the Strategy_Tracker has accumulated Adaptation_Records, THE Strategy_Tracker SHALL retain those Adaptation_Records but reset the raw pattern entries.
7. WHEN the match resets and no Adaptation_Records have been created, THE Strategy_Tracker SHALL NOT consider the retention requirement satisfied; the requirement applies only when actual Adaptation_Records exist to retain.

### Requirement 7: Inter-Player Communication via Signal Bus (team/ only)

**User Story:** As a developer, I want Player agents in the `team/` application to broadcast short intention signals to teammates, so that players can coordinate without relying solely on Coach instructions.

#### Acceptance Criteria

1. THE Signal_Bus SHALL allow any Player_Agent to publish a signal containing the sender position, a signal type (e.g., "requesting_pass", "making_run", "covering_zone"), and a brief payload of at most 50 characters.
2. WHEN a Player_Agent publishes a signal, THE Signal_Bus SHALL make the signal available to all other Player agents within the same Look-Think-Act_Cycle.
3. THE Signal_Bus SHALL retain only the most recent signal from each player position, automatically replacing older signals.
4. WHEN a Player_Agent reads signals from the Signal_Bus, THE Player_Agent SHALL include relevant teammate signals in its LLM prompt context.
5. THE Signal_Bus SHALL be thread-safe, supporting concurrent reads from 4 Player agents and concurrent writes from 4 Player agents; WHEN more than 4 concurrent readers or 4 concurrent writers attempt access simultaneously, THE Signal_Bus SHALL block excess agents until concurrency drops below the limit.
6. THE Signal_Bus SHALL operate exclusively within the `team/` application and SHALL have no dependency on or effect on the `player/` application.
7. THE Signal_Bus SHALL not require additional LLM API calls for signal generation; signals SHALL be generated by Python logic based on the current Plan and game state.

### Requirement 8: Signal Generation Logic (team/ only)

**User Story:** As a developer, I want signals to be generated automatically from the player's current plan and game state, so that communication happens without extra LLM overhead.

#### Acceptance Criteria

1. WHEN a Player_Agent has an active Plan with a sub-goal that benefits from teammate awareness (e.g., "receive pass") and the game state is not a dead ball situation, THE Player_Agent SHALL publish a corresponding signal to the Signal_Bus.
2. WHEN a Player_Agent is in kick range and has a teammate making a run, THE Player_Agent SHALL publish a "ready_to_pass" signal.
3. WHEN a Player_Agent detects it is the nearest teammate to the ball carrier, THE Player_Agent SHALL publish a "supporting" signal with its current zone.
4. THE signal generation logic SHALL execute in Python based on spatial analysis data without making LLM API calls.
5. WHEN the game state indicates a dead ball (kickoff, goal scored), THE Signal_Bus SHALL clear all active signals.

### Requirement 9: Rate Limit Compliance

**User Story:** As a developer, I want all new agentic capabilities to operate without increasing the LLM API call rate, so that the system remains within NVIDIA NIM free tier rate limits.

#### Acceptance Criteria

1. THE Episodic_Memory, Plan evaluation, Reflection_Engine, Strategy_Tracker, and Signal_Bus SHALL execute entirely in Python without making LLM API calls.
2. THE Player_Agent SHALL continue to make exactly one LLM API call per Look-Think-Act_Cycle, with the new agentic context (memory summary, current plan step, adaptation hints, teammate signals) included in the existing prompt; no additional calls SHALL be made for error recovery if the primary call fails.
3. THE total additional token count from agentic context (Memory_Summary + plan step + adaptation hints + signals) SHALL not exceed 300 tokens per LLM invocation.
4. IF the combined agentic context exceeds 300 tokens, THEN THE Player_Agent SHALL prioritize context in this order: current plan step, teammate signals, adaptation hints, memory summary — truncating from the lowest priority first.

### Requirement 10: Application Independence

**User Story:** As a developer, I want changes to `player/` and `team/` to be independent of each other, so that each application can be developed, tested, and deployed separately.

#### Acceptance Criteria

1. THE `player/` application SHALL contain its own implementations of Episodic_Memory, Plan, Reflection_Engine, and Strategy_Tracker with no imports from the `team/` package.
2. THE `team/` application SHALL contain its own implementations of Episodic_Memory, Plan, Reflection_Engine, Strategy_Tracker, and Signal_Bus with no imports from the `player/` package.
3. WHEN a shared algorithm is needed by both applications, THE implementation SHALL be duplicated in each application's package rather than extracted to a shared library.
4. WHEN an algorithm could potentially benefit both applications but is not currently used by either, THE implementation SHALL be proactively duplicated into both application packages.
4. THE `team/` application SHALL maintain the existing Coach_Agent → Player_Agent hierarchy where the Coach provides strategic guidance and Players make tactical decisions.

### Requirement 11: Coach Integration with Agentic Capabilities (team/ only)

**User Story:** As a developer, I want the Coach_Agent to incorporate aggregated player memory and adaptation data into its tactical analysis, so that coaching decisions reflect what players have learned during the match.

#### Acceptance Criteria

1. WHEN the Coach_Agent builds its LLM prompt, THE Coach_Agent SHALL include a summary of active Adaptation_Records from all Player agents (at most 1 sentence per player).
2. WHEN the Coach_Agent detects that multiple players report the same opponent tendency, THE Coach_Agent SHALL issue coordinated tactical instructions that address the tendency.
3. THE Coach_Agent SHALL not directly modify Player episodic memories, plans, or strategy trackers; communication SHALL remain through the existing InstructionStore.
4. THE additional context added to the Coach_Agent prompt from player adaptation data SHALL not exceed 200 tokens.
