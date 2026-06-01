"""Memory summarization module for compact LLM context.

Formats recent episodic memory entries into a compact text representation
suitable for inclusion in the LLM prompt without exceeding token budgets.
Each episode is rendered as a single line with cycle number, action verb,
and outcome classification.
"""

from __future__ import annotations

from episodic_memory import EpisodicMemory, Episode


def _classify_outcome(effectiveness: float | None) -> str:
    """Classify an effectiveness score into positive, neutral, or negative.

    - positive: effectiveness >= 0.6
    - neutral: 0.3 <= effectiveness < 0.6, or effectiveness is None
    - negative: effectiveness < 0.3
    """
    if effectiveness is None:
        return "neutral"
    if effectiveness >= 0.6:
        return "positive"
    if effectiveness < 0.3:
        return "negative"
    return "neutral"


def _derive_action_verb(action: dict) -> str:
    """Derive a human-readable action verb from an action dict.

    - "kicked" if kick is True
    - "moved" if dx or dy is non-zero
    - "waited" otherwise
    """
    if action.get("kick", False):
        return "kicked"
    dx = action.get("dx", 0)
    dy = action.get("dy", 0)
    if dx != 0 or dy != 0:
        return "moved"
    return "waited"


def _format_episode(episode: Episode) -> str:
    """Format a single episode as a summary line."""
    action_verb = _derive_action_verb(episode.action)
    outcome_class = _classify_outcome(episode.effectiveness)
    return f"Cycle {episode.cycle}: {action_verb} \u2192 {outcome_class}"


def summarize_memory(
    memory: EpisodicMemory, max_episodes: int = 5, max_chars: int = 500
) -> str:
    """Format recent episodes as compact text for LLM context.

    Returns a string with at most max_episodes entries, each on one line:
    "Cycle {n}: {action_verb} → {outcome_class}"

    Truncates older episodes first if exceeding max_chars.
    """
    episodes = memory.get_recent(max_episodes)
    if not episodes:
        return ""

    # Format all episode lines
    lines = [_format_episode(ep) for ep in episodes]

    # Build result, truncating older episodes first to stay within max_chars
    # Start with all lines and remove from the front (oldest) until within limit
    while lines and len("\n".join(lines)) > max_chars:
        if len(lines) == 1:
            # Preserve the most recent episode even if it alone exceeds max_chars
            break
        lines.pop(0)

    return "\n".join(lines)
