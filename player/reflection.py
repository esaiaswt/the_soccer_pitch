"""Reflection engine for evaluating action effectiveness.

Computes an effectiveness score by comparing expected outcomes against actual
game state changes. Uses a weighted scoring formula based on ball distance
change, goal distance change, and possession change. All logic is pure Python
with no LLM calls.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass


@dataclass
class ReflectionResult:
    """Result of a reflection evaluation.

    Attributes:
        effectiveness_score: A value in [0.0, 1.0] indicating how effective
            the last action was. Higher is better.
        should_abandon_plan: True if the last 5 consecutive scores are all
            below the abandonment threshold (0.3).
    """

    effectiveness_score: float  # 0.0 to 1.0, clamped
    should_abandon_plan: bool  # True if last 5 scores all < 0.3


class ReflectionEngine:
    """Evaluates action effectiveness using game state comparisons.

    Scoring formula (weighted combination):
    - Δ ball_distance (did we get closer to ball?): weight 0.4
    - Δ goal_distance (did ball get closer to opponent goal?): weight 0.4
    - possession_change (did we gain/keep possession?): weight 0.2

    The raw score is clamped to [0.0, 1.0] before any threshold checks.
    Signals plan abandonment when the last N consecutive scores are all
    below the abandonment threshold.
    """

    def __init__(
        self,
        abandonment_window: int = 5,
        abandonment_threshold: float = 0.3,
    ) -> None:
        self._abandonment_window = abandonment_window
        self._abandonment_threshold = abandonment_threshold
        self._recent_scores: collections.deque[float] = collections.deque(
            maxlen=abandonment_window
        )

    def evaluate(
        self,
        action: dict,
        expected_outcome: dict,
        actual_state: dict,
        previous_state: dict | None,
    ) -> ReflectionResult | None:
        """Evaluate the effectiveness of an action.

        Args:
            action: The action that was taken (e.g., {"dx": 1.0, "dy": 0.0, "kick": False}).
            expected_outcome: The expected outcome (not used in scoring formula
                but kept for interface consistency).
            actual_state: The game state after the action was taken.
            previous_state: The game state before the action. If None (first
                cycle), reflection is skipped and None is returned.

        Returns:
            A ReflectionResult with the clamped effectiveness score and
            abandonment signal, or None if previous_state is missing.
        """
        if previous_state is None:
            return None

        ball_distance_score = self._compute_ball_distance_score(
            previous_state, actual_state
        )
        goal_distance_score = self._compute_goal_distance_score(
            previous_state, actual_state
        )
        possession_score = self._compute_possession_score(
            previous_state, actual_state
        )

        raw_score = (
            0.4 * ball_distance_score
            + 0.4 * goal_distance_score
            + 0.2 * possession_score
        )

        # Clamp to [0.0, 1.0]
        clamped_score = max(0.0, min(1.0, raw_score))

        self._recent_scores.append(clamped_score)

        should_abandon = self._check_abandonment()

        return ReflectionResult(
            effectiveness_score=clamped_score,
            should_abandon_plan=should_abandon,
        )

    def get_recent_scores(self) -> list[float]:
        """Return the list of recent effectiveness scores."""
        return list(self._recent_scores)

    def _check_abandonment(self) -> bool:
        """Check if the plan should be abandoned.

        Returns True only when the last `abandonment_window` consecutive
        scores are all below the abandonment threshold.
        """
        if len(self._recent_scores) < self._abandonment_window:
            return False
        return all(
            score < self._abandonment_threshold
            for score in self._recent_scores
        )

    def _compute_ball_distance_score(
        self, previous_state: dict, actual_state: dict
    ) -> float:
        """Compute score component for ball distance change.

        A decrease in ball distance (getting closer) yields a higher score.
        If previous distance is zero, returns neutral 0.5.
        """
        prev_dist = previous_state.get("ball_distance", 0.0)
        curr_dist = actual_state.get("ball_distance", 0.0)

        if prev_dist == 0.0:
            return 0.5

        # Positive delta means we got closer (improvement)
        delta = prev_dist - curr_dist
        # Normalize: full improvement (delta == prev_dist) -> 1.0
        # No change (delta == 0) -> 0.5
        # Got further away (delta == -prev_dist) -> 0.0
        score = 0.5 + (delta / (2.0 * prev_dist))
        return score

    def _compute_goal_distance_score(
        self, previous_state: dict, actual_state: dict
    ) -> float:
        """Compute score component for goal distance change.

        A decrease in goal distance (ball getting closer to opponent goal)
        yields a higher score. If previous distance is zero, returns neutral 0.5.
        """
        prev_dist = previous_state.get("goal_distance", 0.0)
        curr_dist = actual_state.get("goal_distance", 0.0)

        if prev_dist == 0.0:
            return 0.5

        # Positive delta means ball got closer to goal (improvement)
        delta = prev_dist - curr_dist
        # Normalize similarly to ball distance
        score = 0.5 + (delta / (2.0 * prev_dist))
        return score

    def _compute_possession_score(
        self, previous_state: dict, actual_state: dict
    ) -> float:
        """Compute score component for possession change.

        Returns:
            1.0 if we gained possession (didn't have it, now we do)
            0.75 if we kept possession
            0.25 if we never had possession (no change, still don't have it)
            0.0 if we lost possession
        """
        prev_possession = previous_state.get("has_possession", False)
        curr_possession = actual_state.get("has_possession", False)

        if curr_possession and not prev_possession:
            return 1.0  # Gained possession
        elif curr_possession and prev_possession:
            return 0.75  # Kept possession
        elif not curr_possession and not prev_possession:
            return 0.25  # Never had it
        else:
            return 0.0  # Lost possession
