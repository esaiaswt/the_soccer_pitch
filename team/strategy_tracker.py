"""Strategy Tracker module for the team/ application.

Records tactical patterns (opponent positions, ball position, effectiveness)
and analyzes opponent tendencies using directional frequency distributions.
When a directional bucket exceeds 70% confidence, generates an AdaptationRecord
with a recommended counter-strategy.

This is an independent implementation with no imports from the player/ package.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PatternEntry:
    """A single pattern entry recording opponent context and outcome."""

    opponent_positions: list[dict]  # [{x, y}, ...]
    ball_position: dict  # {x, y}
    effectiveness: float


@dataclass
class AdaptationRecord:
    """A detected opponent tendency with a recommended counter-strategy."""

    observed_pattern: str  # e.g., "opponent_favors_left_flank"
    counter_strategy: str  # e.g., "shift_defensive_coverage_left"
    confidence: float  # 0.0 to 1.0


# Mapping from directional bucket to pattern description and counter-strategy
_DIRECTION_PATTERNS: dict[str, tuple[str, str]] = {
    "left": ("opponent_favors_left_flank", "shift_defensive_coverage_left"),
    "right": ("opponent_favors_right_flank", "shift_defensive_coverage_right"),
    "center": ("opponent_favors_center", "compact_defensive_center"),
    "forward": ("opponent_pushes_forward", "drop_defensive_line_deeper"),
    "back": ("opponent_sits_back", "push_higher_press"),
}


class StrategyTracker:
    """Tracks tactical patterns and produces adaptation recommendations.

    Records action contexts and outcomes, then analyzes opponent tendencies
    by computing directional frequency distributions. When a direction bucket
    exceeds 70% of entries, generates an AdaptationRecord.
    """

    def __init__(self, min_entries_for_analysis: int = 10) -> None:
        """Initialize the strategy tracker.

        Args:
            min_entries_for_analysis: Minimum pattern entries required before
                analysis can produce results. Defaults to 10.
        """
        self._min_entries_for_analysis = min_entries_for_analysis
        self._entries: list[PatternEntry] = []
        self._adaptations: list[AdaptationRecord] = []

    def record(self, entry: PatternEntry) -> None:
        """Record a pattern entry for later analysis.

        Args:
            entry: The pattern entry containing opponent positions,
                ball position, and effectiveness score.
        """
        self._entries.append(entry)

    def analyze(self) -> list[AdaptationRecord]:
        """Analyze recorded patterns and generate adaptation records.

        Computes directional frequency distributions of opponent positions
        relative to the ball. When any directional bucket exceeds 70% of
        entries, generates an AdaptationRecord for that tendency.

        Returns:
            List of newly generated AdaptationRecords. Returns empty list
            if fewer than min_entries_for_analysis entries have been recorded.
        """
        if len(self._entries) < self._min_entries_for_analysis:
            return []

        # Compute directional buckets
        buckets: dict[str, int] = {
            "left": 0,
            "right": 0,
            "center": 0,
            "forward": 0,
            "back": 0,
        }

        total_entries = len(self._entries)

        for entry in self._entries:
            direction = self._classify_direction(
                entry.opponent_positions, entry.ball_position
            )
            buckets[direction] += 1

        # Check for dominant directions exceeding 70% threshold
        new_adaptations: list[AdaptationRecord] = []
        confidence_threshold = 0.7

        for direction, count in buckets.items():
            confidence = count / total_entries
            if confidence > confidence_threshold:
                pattern_name, counter = _DIRECTION_PATTERNS[direction]
                # Avoid duplicating existing adaptations
                if not any(
                    a.observed_pattern == pattern_name for a in self._adaptations
                ):
                    record = AdaptationRecord(
                        observed_pattern=pattern_name,
                        counter_strategy=counter,
                        confidence=confidence,
                    )
                    new_adaptations.append(record)
                    self._adaptations.append(record)

        return new_adaptations

    def get_active_adaptations(self, max_count: int = 2) -> list[AdaptationRecord]:
        """Return the most confident active adaptation records.

        Args:
            max_count: Maximum number of records to return. Defaults to 2.

        Returns:
            List of at most max_count AdaptationRecords, sorted by
            confidence (highest first).
        """
        sorted_adaptations = sorted(
            self._adaptations, key=lambda a: a.confidence, reverse=True
        )
        return sorted_adaptations[:max_count]

    def reset_for_new_match(self) -> None:
        """Reset for a new match, retaining adaptations but clearing raw entries.

        Clears all raw pattern entries while preserving any AdaptationRecords
        that have been generated from previous analysis.
        """
        self._entries = []

    def _classify_direction(
        self, opponent_positions: list[dict], ball_position: dict
    ) -> str:
        """Classify the average opponent position relative to ball into a directional bucket.

        Computes the average position of all opponents relative to the ball,
        then classifies into one of: left, right, center, forward, back.

        Args:
            opponent_positions: List of opponent position dicts with 'x' and 'y' keys.
            ball_position: Ball position dict with 'x' and 'y' keys.

        Returns:
            One of "left", "right", "center", "forward", "back".
        """
        if not opponent_positions:
            return "center"

        ball_x = ball_position.get("x", 0.0)
        ball_y = ball_position.get("y", 0.0)

        # Compute average relative position of opponents to ball
        total_dx = 0.0
        total_dy = 0.0
        count = len(opponent_positions)

        for pos in opponent_positions:
            total_dx += pos.get("x", 0.0) - ball_x
            total_dy += pos.get("y", 0.0) - ball_y

        avg_dx = total_dx / count
        avg_dy = total_dy / count

        # Classify based on which component is dominant
        # Use absolute values to determine primary axis
        abs_dx = abs(avg_dx)
        abs_dy = abs(avg_dy)

        if abs_dx < 5.0 and abs_dy < 5.0:
            # Near the ball - classify as center
            return "center"

        if abs_dx >= abs_dy:
            # Horizontal movement is dominant
            return "left" if avg_dx < 0 else "right"
        else:
            # Vertical movement is dominant
            # Positive y = forward (toward opponent goal), negative y = back
            return "forward" if avg_dy > 0 else "back"
