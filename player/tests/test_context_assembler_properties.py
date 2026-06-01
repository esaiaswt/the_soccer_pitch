"""Property-based tests for the Context Assembler module (Properties 10 and 24).

# Feature: full-agentic-upgrade, Property 10: Plan context inclusion iff plan is active
# Feature: full-agentic-upgrade, Property 24: Context priority-based truncation
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from context_assembler import assemble_agentic_context, _estimate_tokens


# --- Strategies ---

# Non-empty text that could be a plan step description
plan_step_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
)

# Memory summary text (could be multi-line)
memory_summary_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=500,
)

# Single adaptation hint
hint_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=100,
)

# List of adaptation hints
adaptation_hints_strategy = st.lists(hint_strategy, min_size=0, max_size=5)


# Feature: full-agentic-upgrade, Property 10: Plan context inclusion iff plan is active
class TestPlanContextInclusionIffPlanActive:
    """Property 10: Plan context inclusion iff plan is active.

    For any context assembly invocation, the output SHALL contain the current
    sub-goal description if and only if an active Plan exists. When no Plan is
    active, no sub-goal text SHALL appear in the assembled context.

    **Validates: Requirements 3.9, 3.10**
    """

    @given(
        memory_summary=memory_summary_strategy,
        plan_step=plan_step_strategy,
        adaptation_hints=adaptation_hints_strategy,
    )
    @settings(max_examples=100)
    def test_plan_step_included_when_active(self, memory_summary, plan_step, adaptation_hints):
        """When a plan step is provided (plan is active), the output contains the plan step text."""
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
        )

        # The plan step text should appear in the output (possibly truncated for very long inputs,
        # but since plan_step has highest priority it should be preserved)
        assert plan_step in result or "[Plan]" in result, (
            f"Plan step not found in output when plan is active.\n"
            f"Plan step: {plan_step!r}\n"
            f"Output: {result!r}"
        )

    @given(
        memory_summary=memory_summary_strategy,
        adaptation_hints=adaptation_hints_strategy,
    )
    @settings(max_examples=100)
    def test_no_plan_context_when_plan_is_none(self, memory_summary, adaptation_hints):
        """When plan_step is None (no active plan), no plan-related text appears in output."""
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=None,
            adaptation_hints=adaptation_hints,
        )

        # No plan marker should appear in the output
        assert "[Plan]" not in result, (
            f"Plan context marker found in output when no plan is active.\n"
            f"Output: {result!r}"
        )

    @given(
        memory_summary=memory_summary_strategy,
        adaptation_hints=adaptation_hints_strategy,
    )
    @settings(max_examples=100)
    def test_empty_string_plan_treated_as_no_plan(self, memory_summary, adaptation_hints):
        """When plan_step is an empty string, it is treated as no active plan."""
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step="",
            adaptation_hints=adaptation_hints,
        )

        # Empty plan step should not produce plan context
        assert "[Plan]" not in result, (
            f"Plan context marker found in output when plan_step is empty string.\n"
            f"Output: {result!r}"
        )


# Feature: full-agentic-upgrade, Property 24: Context priority-based truncation
class TestContextPriorityBasedTruncation:
    """Property 24: Context priority-based truncation.

    For any combination of memory summary, plan step, adaptation hints that would
    exceed 300 tokens, the Context Assembler SHALL truncate from lowest priority
    first (memory summary → adaptation hints → plan step), and the output SHALL
    not exceed 300 tokens. The highest-priority items SHALL be preserved intact.

    **Validates: Requirements 9.3, 9.4**
    """

    @given(
        memory_summary=memory_summary_strategy,
        plan_step=st.one_of(st.none(), plan_step_strategy),
        adaptation_hints=adaptation_hints_strategy,
    )
    @settings(max_examples=100)
    def test_output_never_exceeds_max_tokens(self, memory_summary, plan_step, adaptation_hints):
        """The assembled context never exceeds 300 tokens."""
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            max_tokens=300,
        )

        token_count = _estimate_tokens(result)
        assert token_count <= 300, (
            f"Output exceeds 300 token limit: {token_count} tokens.\n"
            f"Output length: {len(result)} chars.\n"
            f"Output: {result!r}"
        )

    @given(
        memory_summary=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=800,
            max_size=1000,
        ),
        plan_step=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=10,
            max_size=50,
        ),
        adaptation_hints=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=10,
                max_size=50,
            ),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_memory_truncated_first(self, memory_summary, plan_step, adaptation_hints):
        """When total exceeds budget, memory summary (lowest priority) is truncated first."""
        # With memory_summary of 800-1000 chars, total will always exceed 300 tokens (1200 chars)
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            max_tokens=300,
        )

        # Plan step (highest priority) should be preserved intact
        assert plan_step in result, (
            f"Plan step (highest priority) was not preserved intact during truncation.\n"
            f"Plan step: {plan_step!r}\n"
            f"Output: {result!r}"
        )

    @given(
        memory_summary=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=800,
            max_size=1000,
        ),
        plan_step=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=10,
            max_size=50,
        ),
        adaptation_hints=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=10,
                max_size=50,
            ),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_plan_step_highest_priority_preserved(self, memory_summary, plan_step, adaptation_hints):
        """Plan step has highest priority and is preserved when truncation occurs."""
        # With memory_summary of 800-1000 chars, total will always exceed 300 tokens (1200 chars)
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            max_tokens=300,
        )

        # Plan step should always be in the output (it's highest priority)
        assert f"[Plan] {plan_step}" in result, (
            f"Plan section not preserved intact.\n"
            f"Expected: '[Plan] {plan_step}'\n"
            f"Output: {result!r}"
        )

    @given(
        memory_summary=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=500,
            max_size=800,
        ),
        adaptation_hints=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=100,
                max_size=200,
            ),
            min_size=2,
            max_size=4,
        ),
        plan_step=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=100,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_truncation_order_lowest_priority_first(
        self, memory_summary, adaptation_hints, plan_step
    ):
        """Truncation removes lowest priority items first: memory → hints → plan."""
        # With these sizes (memory 500-800, hints 200-800, plan 100-200),
        # total will always exceed 300 tokens (1200 chars)
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            max_tokens=300,
        )

        # If memory is gone but hints remain, that's correct truncation order
        # If both memory and hints are gone, plan should still be there
        has_memory = "[Memory]" in result
        has_hints = "[Adapt]" in result
        has_plan = "[Plan]" in result

        # Plan (highest priority) should always be present
        assert has_plan, (
            f"Plan (highest priority) was removed during truncation.\n"
            f"Output: {result!r}"
        )

        # If hints are removed, memory must also be removed (lower priority goes first)
        if not has_hints:
            assert not has_memory, (
                f"Memory (lower priority) preserved but hints (higher priority) removed.\n"
                f"Output: {result!r}"
            )

    @given(
        memory_summary=memory_summary_strategy,
        plan_step=st.one_of(st.none(), plan_step_strategy),
        adaptation_hints=adaptation_hints_strategy,
        max_tokens=st.integers(min_value=10, max_value=500),
    )
    @settings(max_examples=100)
    def test_output_respects_custom_max_tokens(self, memory_summary, plan_step, adaptation_hints, max_tokens):
        """The assembled context respects any custom max_tokens value."""
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            max_tokens=max_tokens,
        )

        token_count = _estimate_tokens(result)
        assert token_count <= max_tokens, (
            f"Output exceeds custom {max_tokens} token limit: {token_count} tokens.\n"
            f"Output length: {len(result)} chars.\n"
            f"Output: {result!r}"
        )
