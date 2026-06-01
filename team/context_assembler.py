"""Context Assembler module for the team/ application.

Assembles agentic context (memory summary, plan step, adaptation hints,
and teammate signals) with priority-based truncation to stay within
token budgets for the LLM prompt.

This is an independent implementation with no imports from the player/ package.
"""

from __future__ import annotations


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text using ~4 characters per token.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    return len(text) // 4


def _truncate_to_budget(text: str, max_chars: int) -> str:
    """Truncate text to fit within a character budget.

    Args:
        text: The text to truncate.
        max_chars: Maximum allowed characters.

    Returns:
        Truncated text, cut at the last space before the limit if possible.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to cut at a word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space]
    return truncated


def assemble_agentic_context(
    memory_summary: str,
    plan_step: str | None,
    adaptation_hints: list[str],
    signals: list[str] | None = None,
    max_tokens: int = 300,
) -> str:
    """Assemble agentic context with priority-based truncation.

    Priority order (highest first):
    1. Current plan step
    2. Teammate signals (team/ only)
    3. Adaptation hints
    4. Memory summary

    Truncates from lowest priority first if exceeding max_tokens.
    Token estimation: ~4 characters per token.

    Args:
        memory_summary: Compact text summary of recent episodic memory.
        plan_step: Current active sub-goal description, or None if no plan.
        adaptation_hints: List of adaptation strategy hints.
        signals: List of teammate signal strings (team/ only).
        max_tokens: Maximum token budget. Defaults to 300.

    Returns:
        Assembled context string, or empty string if all components are empty.
    """
    max_chars = max_tokens * 4

    # Build sections in priority order (highest first)
    sections: list[str] = []

    # Priority 1: Plan step (highest)
    plan_section = ""
    if plan_step:
        plan_section = f"[Plan] {plan_step}"

    # Priority 2: Signals
    signals_section = ""
    if signals:
        signals_section = "[Signals] " + "; ".join(signals)

    # Priority 3: Adaptation hints
    adaptation_section = ""
    if adaptation_hints:
        adaptation_section = "[Adapt] " + "; ".join(adaptation_hints)

    # Priority 4: Memory summary (lowest)
    memory_section = ""
    if memory_summary:
        memory_section = f"[Memory] {memory_summary}"

    # Collect all non-empty sections in priority order (highest first)
    priority_sections = []
    if plan_section:
        priority_sections.append(plan_section)
    if signals_section:
        priority_sections.append(signals_section)
    if adaptation_section:
        priority_sections.append(adaptation_section)
    if memory_section:
        priority_sections.append(memory_section)

    if not priority_sections:
        return ""

    # Check if everything fits
    separator = "\n"
    result = separator.join(priority_sections)
    if len(result) <= max_chars:
        return result

    # Truncate from lowest priority first
    # Work backwards through priority_sections (lowest priority = last)
    while len(priority_sections) > 1:
        # Try truncating the lowest priority section
        current_result = separator.join(priority_sections)
        if len(current_result) <= max_chars:
            return current_result

        # Calculate budget available for the lowest priority section
        higher_priority = separator.join(priority_sections[:-1])
        remaining_chars = max_chars - len(higher_priority) - len(separator)

        if remaining_chars <= 0:
            # No room for lowest priority section at all, remove it
            priority_sections.pop()
        else:
            # Try to fit a truncated version
            truncated = _truncate_to_budget(
                priority_sections[-1], remaining_chars
            )
            if truncated:
                priority_sections[-1] = truncated
                result = separator.join(priority_sections)
                if len(result) <= max_chars:
                    return result
            # Still too long or empty after truncation, remove it
            priority_sections.pop()

    # Only highest priority section remains, truncate if needed
    return _truncate_to_budget(priority_sections[0], max_chars)
