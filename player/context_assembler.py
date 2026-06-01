"""Context assembler module for player agentic context.

Assembles agentic context (memory summary, plan step, adaptation hints)
into a single string for inclusion in the LLM prompt. Applies priority-based
truncation to stay within the token budget.

Priority order (highest first):
1. Current plan step
2. Adaptation hints
3. Memory summary

Truncates from lowest priority first if exceeding max_tokens.
Token estimation: ~4 characters per token.
"""

from __future__ import annotations


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length (~4 chars per token)."""
    return len(text) // 4


def assemble_agentic_context(
    memory_summary: str,
    plan_step: str | None,
    adaptation_hints: list[str],
    max_tokens: int = 300,
) -> str:
    """Assemble agentic context with priority-based truncation.

    Priority order (highest first):
    1. Current plan step
    2. Adaptation hints
    3. Memory summary

    Truncates from lowest priority first if exceeding max_tokens.
    Token estimation: ~4 characters per token.

    Returns empty string when all components are empty.
    """
    # Build sections in priority order (highest first)
    sections: list[str] = []

    # Priority 1 (highest): plan step
    plan_section = ""
    if plan_step:
        plan_section = f"[Plan] {plan_step}"

    # Priority 2: adaptation hints
    hints_section = ""
    if adaptation_hints:
        formatted_hints = "; ".join(adaptation_hints)
        hints_section = f"[Adapt] {formatted_hints}"

    # Priority 3 (lowest): memory summary
    memory_section = ""
    if memory_summary:
        memory_section = f"[Memory] {memory_summary}"

    # Collect all non-empty sections in priority order (lowest first for truncation)
    # We'll build from lowest priority and truncate from the front
    prioritized = []  # (priority, section) - lower index = lower priority
    if memory_section:
        prioritized.append(memory_section)
    if hints_section:
        prioritized.append(hints_section)
    if plan_section:
        prioritized.append(plan_section)

    if not prioritized:
        return ""

    # Check if total fits within budget
    max_chars = max_tokens * 4

    # Try with all sections joined
    result = "\n".join(prioritized)
    if _estimate_tokens(result) <= max_tokens:
        return result

    # Truncate from lowest priority first
    # prioritized[0] is lowest priority (memory), then hints, then plan
    while prioritized and _estimate_tokens("\n".join(prioritized)) > max_tokens:
        if len(prioritized) == 1:
            # Only highest priority item left - truncate it to fit
            remaining_chars = max_tokens * 4
            prioritized[0] = prioritized[0][:remaining_chars]
            break
        # Remove lowest priority item
        prioritized.pop(0)

    return "\n".join(prioritized)
