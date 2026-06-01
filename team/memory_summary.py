"""Memory Summarizer module for the team/ application.

Formats recent episodic memory entries as compact text for inclusion
in the LLM prompt context without exceeding token budgets.

This is an independent implementation with no imports from the player/ package.
"""

from __future__ import annotations

from team.episodic_memory import EpisodicMemory


def _action_verb(action: dict) -> str:
    """Derive a human-readable action verb from an action dict.

    Args:
        action: Action dictionary with keys like dx, dy, kick.

    Returns:
        A short verb describing the action taken.
    """
    if action.get("kick"):
        return "kicked"
    dx = action.get("dx", 0)
    dy = action.get("dy", 0)
    if dx != 0 or dy != 0:
        return "moved"
    return "waited"


def _outcome_class(effectiveness: float | None) -> str:
    """Classify effectiveness score into an outcome category.

    Args:
        effectiveness: Score between 0.0 and 1.0, or None.

    Returns:
        One of "positive", "neutral", or "negative".
    """
    if effectiveness is None:
        return "neutral"
    if effectiveness >= 0.6:
        return "positive"
    if effectiveness < 0.3:
        return "negative"
    return "neutral"


def summarize_memory(
    memory: EpisodicMemory, max_episodes: int = 5, max_chars: int = 500
) -> str:
    """Format recent episodes as compact text for LLM context.

    Returns a string with at most max_episodes entries, each on one line:
    "Cycle {n}: {action_verb} → {outcome_class}"

    Truncates older episodes first if exceeding max_chars.

    Args:
        memory: The episodic memory to summarize.
        max_episodes: Maximum number of episodes to include. Defaults to 5.
        max_chars: Maximum total character length. Defaults to 500.

    Returns:
        A compact multi-line string summarizing recent memory.
    """
    episodes = memory.get_recent(max_episodes)
    if not episodes:
        return ""

    # Format each episode as a single line
    lines: list[str] = []
    for ep in episodes:
        verb = _action_verb(ep.action)
        outcome = _outcome_class(ep.effectiveness)
        lines.append(f"Cycle {ep.cycle}: {verb} \u2192 {outcome}")

    # Join all lines and check character limit
    result = "\n".join(lines)
    if len(result) <= max_chars:
        return result

    # Truncate older episodes first while preserving the most recent
    while len(lines) > 1:
        lines.pop(0)
        result = "\n".join(lines)
        if len(result) <= max_chars:
            return result

    # If even a single line exceeds max_chars, truncate it
    return lines[0][:max_chars]
