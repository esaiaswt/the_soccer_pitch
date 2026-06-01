"""Reflection Engine module for the team/ application.

Evaluates the effectiveness of the most recent action by comparing expected
outcomes against actual game state changes. Produces an effectiveness score
without additional LLM calls.

Scoring formula (weighted combination):
- Δ ball_distance (did we get closer to ball?): weight 0.4
- Δ goal_distance (did ball get closer to opponent goal?): weight 0.4
- possession_change (did we gain/keep possession?): weight 0.2

This is an independent implementation with no imports from the player/ package.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class ReflectionResult:
    """Result of a reflection evaluation.

    Attributes:
        effectiveness_score: Score between 0.0 and 1.0 (clamped).
        should_abandon_plan: True if last 5 scores are all below 0.3.
    """

    effectiveness_score: float  # 0.0 to 1.0, clamped
    should_abandon_plan: bool  # True if last 5 scores all < 0.3


class ReflectionEngine:
    """Evaluates action effectiveness using game state comparisons.

    Tracks recent effectiveness scores and signals plan abandonment
    when consecutive scores fall below the abandonment threshold.
    All logic executes in Python without LLM API calls.
    """

    def __init__(
        self,
        abandonment_window: int = 5,
        abandonment_threshold: float = 0.3,
    ) -> None:
        """Initialize the reflection engine.

        Args:
            abandonment_window: Number of recent scores to check for abandonment.
                Defaults to 5.
            abandonment_threshold: Score threshold below which an action is
                considered ineffective. Defaults to 0.3.
        """
        self._abandonment_window = abandonment_window
        self._abandonment_threshold = abandonment_threshold
        self._recent_scores: deque[float] = deque(maxlen=abandonment_window)

    def evaluate(
        self,
        action: dict,
        expected_outcome: dict,
        actual_state: dict,
        previous_state: dict | None,
    ) -> ReflectionResult | None:
        """Evaluate the effectiveness of the last action.

        Compares the expected outcome against the actual state change to
        compute an effectiveness score.

        Args:
            action: The action that was taken (e.g., {"dx": float, "dy": float, "kick": bool}).
            expected_outcome: What the action was expected to achieve.
            actual_state: The game state after the action was executed.
            previous_state: The game state before the action was executed.
                If None (first cycle), reflection is skipped.

        Returns:
            A ReflectionResult with the effectiveness score and abandonment signal,
            or None if previous_state is not available (first cycle).
        """
        if previous_state is None:
            return None

        score = self._compute_score(actual_state, previous_state)
        self._recent_scores.append(score)
        should_abandon = self._check_abandonment()

        return ReflectionResult(
            effectiveness_score=score,
            should_abandon_plan=should_abandon,
        )

    def get_recent_scores(self) -> list[float]:
        """Return the list of recent effectiveness scores.

        Returns:
            List of recent scores, oldest first.
        """
        return list(self._recent_scores)

    def _compute_score(self, actual_state: dict, previous_state: dict) -> float:
        """Compute the effectiveness score from state comparison.

        Scoring formula (weighted combination):
        - Δ ball_distance: weight 0.4
        - Δ goal_distance: weight 0.4
        - possession_change: weight 0.2

        Args:
            actual_state: The game state after the action.
            previous_state: The game state before the action.

        Returns:
            Clamped effectiveness score between 0.0 and 1.0.
        """
        ball_score = self._compute_ball_distance_score(actual_state, previous_state)
        goal_score = self._compute_goal_distance_score(actual_state, previous_state)
        possession_score = self._compute_possession_score(actual_state, previous_state)

        raw_score = (
            0.4 * ball_score
            + 0.4 * goal_score
            + 0.2 * possession_score
        )

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, raw_score))

    def _compute_ball_distance_score(
        self, actual_state: dict, previous_state: dict
    ) -> float:
        """Compute the ball distance component score.

        A score of 1.0 means we got significantly closer to the ball.
        A score of 0.5 means no change (neutral).
        A score of 0.0 means we got significantly farther from the ball.

        If previous distance is zero, returns 0.5 (neutral) to avoid division by zero.
        """
        prev_distance = previous_state.get("ball_distance", 0.0)
        curr_distance = actual_state.get("ball_distance", 0.0)

        if prev_distance == 0.0:
            # Division by zero edge case: treat as neutral
            return 0.5

        # Positive delta means we got closer (good)
        delta = prev_distance - curr_distance
        # Normalize by previous distance to get a relative improvement
        normalized = delta / prev_distance

        # Map from [-1, 1] range to [0, 1] range
        return 0.5 + (normalized * 0.5)

    def _compute_goal_distance_score(
        self, actual_state: dict, previous_state: dict
    ) -> float:
        """Compute the goal distance component score.

        A score of 1.0 means the ball got significantly closer to the opponent goal.
        A score of 0.5 means no change (neutral).
        A score of 0.0 means the ball got farther from the opponent goal.

        If previous distance is zero, returns 0.5 (neutral) to avoid division by zero.
        """
        prev_distance = previous_state.get("goal_distance", 0.0)
        curr_distance = actual_state.get("goal_distance", 0.0)

        if prev_distance == 0.0:
            # Division by zero edge case: treat as neutral
            return 0.5

        # Positive delta means ball got closer to goal (good)
        delta = prev_distance - curr_distance
        # Normalize by previous distance
        normalized = delta / prev_distance

        # Map from [-1, 1] range to [0, 1] range
        return 0.5 + (normalized * 0.5)

    def _compute_possession_score(
        self, actual_state: dict, previous_state: dict
    ) -> float:
        """Compute the possession change component score.

        Returns:
            1.0 if we gained or kept possession.
            0.0 if we lost possession.
            0.5 if possession state is unchanged and we don't have it.
        """
        had_possession = previous_state.get("has_possession", False)
        have_possession = actual_state.get("has_possession", False)

        if have_possession:
            # We have possession now (gained or kept) — good
            return 1.0
        elif had_possession and not have_possession:
            # We lost possession — bad
            return 0.0
        else:
            # We didn't have it before and still don't — neutral
            return 0.5

    def _check_abandonment(self) -> bool:
        """Check if the plan should be abandoned.

        Returns True if the last `abandonment_window` consecutive scores
        are all below the abandonment threshold.

        Returns:
            True if plan should be abandoned, False otherwise.
        """
        if len(self._recent_scores) < self._abandonment_window:
            return False

        return all(
            score < self._abandonment_threshold
            for score in self._recent_scores
        )
