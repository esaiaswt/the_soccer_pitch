"""Strategy tracker module for learning and adaptation within a match.

Records tactical patterns (opponent positions, ball position, effectiveness)
and analyzes opponent tendencies by computing directional frequency distributions.
When a directional bucket exceeds 70% confidence, generates an AdaptationRecord
with a recommended counter-strategy. All logic executes in pure Python without
LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatternEntry:
    """A single recorded pattern of opponent behavior and its effectiveness."""

    opponent_positions: list[dict]  # [{"x": float, "y": float}, ...]
    ball_position: dict  # {"x": float, "y": float}
    effectiveness: float


@dataclass
class AdaptationRecord:
    """A detected opponent tendency with a recommended counter-strategy."""

    observed_pattern: str  # e.g., "opponent_favors_left_flank"
    counter_strategy: str  # e.g., "shift_defensive_coverage_left"
    confidence: float  # 0.0 to 1.0


# Mapping from directional bucket to pattern description and counter-strategy.
_DIRECTION_MAP: dict[str, tuple[str, str]] = {
    "left": ("opponent_favors_left_flank", "shift_defensive_coverage_left"),
    "right": ("opponent_favors_right_flank", "shift_defensive_coverage_right"),
    "center": ("opponent_favors_center", "compact_defensive_center"),
    "forward": ("opponent_pushes_forward", "drop_defensive_line_deeper"),
    "back": ("opponent_sits_back", "push_higher_press"),
}


class StrategyTracker:
    """Tracks opponent patterns and produces adaptation recommendations.

    Records each action's context and outcome, then analyzes directional
    frequency distributions of opponent movements relative to the ball.
    When a direction bucket exceeds 70% of entries, an AdaptationRecord
    is generated.
    """

    def __init__(self, min_entries_for_analysis: int = 10) -> None:
        self._min_entries = min_entries_for_analysis
        self._entries: list[PatternEntry] = []
        self._adaptations: list[AdaptationRecord] = []

    def record(self, entry: PatternEntry) -> None:
        """Record a new pattern entry."""
        self._entries.append(entry)

    def analyze(self) -> list[AdaptationRecord]:
        """Analyze recorded patterns and generate AdaptationRecords.

        Computes directional frequency distributions of opponent positions
        relative to the ball. When any direction bucket exceeds 70% of
        entries, generates an AdaptationRecord for that tendency.

        Returns the list of newly generated AdaptationRecords (also stored
        internally as active adaptations).
        """
        if len(self._entries) < self._min_entries:
            return []

        # Classify each entry into a directional bucket
        bucket_counts: dict[str, int] = {
            "left": 0,
            "right": 0,
            "center": 0,
            "forward": 0,
            "back": 0,
        }
        total_entries = len(self._entries)

        for entry in self._entries:
            bucket = self._classify_direction(entry)
            bucket_counts[bucket] += 1

        # Generate AdaptationRecords for buckets exceeding 70% threshold
        new_adaptations: list[AdaptationRecord] = []
        for direction, count in bucket_counts.items():
            confidence = count / total_entries
            if confidence > 0.7:
                pattern_name, counter = _DIRECTION_MAP[direction]
                record = AdaptationRecord(
                    observed_pattern=pattern_name,
                    counter_strategy=counter,
                    confidence=confidence,
                )
                new_adaptations.append(record)

        # Update stored adaptations with new findings
        if new_adaptations:
            self._adaptations = new_adaptations

        return new_adaptations

    def get_active_adaptations(self, max_count: int = 2) -> list[AdaptationRecord]:
        """Return at most max_count active AdaptationRecords.

        Returns the highest-confidence adaptations, limited to max_count.
        """
        sorted_adaptations = sorted(
            self._adaptations, key=lambda a: a.confidence, reverse=True
        )
        return sorted_adaptations[:max_count]

    def reset_for_new_match(self) -> None:
        """Reset for a new match: retain AdaptationRecords, clear raw entries."""
        self._entries = []

    def _classify_direction(self, entry: PatternEntry) -> str:
        """Classify the average opponent position relative to ball into a bucket.

        Computes the average position of all opponents, then determines the
        direction relative to the ball position. Uses x-axis for left/right
        and y-axis for forward/back, with a center threshold.
        """
        if not entry.opponent_positions:
            return "center"

        avg_x = sum(p.get("x", 0.0) for p in entry.opponent_positions) / len(
            entry.opponent_positions
        )
        avg_y = sum(p.get("y", 0.0) for p in entry.opponent_positions) / len(
            entry.opponent_positions
        )

        ball_x = entry.ball_position.get("x", 0.0)
        ball_y = entry.ball_position.get("y", 0.0)

        dx = avg_x - ball_x
        dy = avg_y - ball_y

        # Determine primary direction based on larger displacement
        if abs(dx) > abs(dy):
            # Horizontal displacement dominates
            if dx < -10:
                return "left"
            elif dx > 10:
                return "right"
            else:
                return "center"
        else:
            # Vertical displacement dominates
            if dy > 10:
                return "forward"
            elif dy < -10:
                return "back"
            else:
                return "center"
