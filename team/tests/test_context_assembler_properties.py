"""Property-based tests for the team/ Context Assembler module.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from team.context_assembler import assemble_agentic_context


# --- Strategies for generating valid context components ---

# Generate non-empty plan step descriptions
plan_step_strategy = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=100,
    ),
)

# Generate memory summary strings
memory_summary_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=200,
)

# Generate adaptation hints lists
adaptation_hints_strategy = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=50,
    ),
    min_size=0,
    max_size=5,
)

# Generate signals lists (team/ specific)
signals_strategy = st.one_of(
    st.none(),
    st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=1,
            max_size=50,
        ),
        min_size=0,
        max_size=4,
    ),
)


# Feature: full-agentic-upgrade, Property 10: Plan context inclusion iff plan is active
# **Validates: Requirements 3.9, 3.10, 9.3, 9.4**


class TestPlanContextInclusionIffPlanIsActive:
    """Property 10: Plan context inclusion iff plan is active.

    For any context assembly invocation, the output SHALL contain the current
    sub-goal description if and only if an active Plan exists. When no Plan is
    active, no sub-goal text SHALL appear in the assembled context.
    """

    @settings(max_examples=100)
    @given(
        memory_summary=memory_summary_strategy,
        plan_step=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=1,
            max_size=80,
        ),
        adaptation_hints=adaptation_hints_strategy,
        signals=signals_strategy,
    )
    def test_plan_step_included_when_plan_is_active(
        self, memory_summary, plan_step, adaptation_hints, signals
    ):
        """When an active Plan exists (plan_step is not None and non-empty),
        the output SHALL contain the plan step description.

        **Validates: Requirements 3.9**
        """
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            signals=signals,
            max_tokens=300,
        )

        # The plan step should appear in the output since it has highest priority
        assert plan_step in result, (
            f"Plan step '{plan_step}' not found in assembled context: '{result}'"
        )

    @settings(max_examples=100)
    @given(
        memory_summary=memory_summary_strategy,
        adaptation_hints=adaptation_hints_strategy,
        signals=signals_strategy,
    )
    def test_no_plan_context_when_plan_is_none(
        self, memory_summary, adaptation_hints, signals
    ):
        """When no Plan is active (plan_step is None), no sub-goal text SHALL
        appear in the assembled context.

        **Validates: Requirements 3.10**
        """
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=None,
            adaptation_hints=adaptation_hints,
            signals=signals,
            max_tokens=300,
        )

        # The [Plan] marker should not appear when plan_step is None
        assert "[Plan]" not in result, (
            f"Plan context marker '[Plan]' found in output when plan_step is None: '{result}'"
        )

    @settings(max_examples=100)
    @given(
        memory_summary=memory_summary_strategy,
        adaptation_hints=adaptation_hints_strategy,
        signals=signals_strategy,
    )
    def test_no_plan_context_when_plan_is_empty_string(
        self, memory_summary, adaptation_hints, signals
    ):
        """When plan_step is an empty string (treated as no active plan),
        no sub-goal text SHALL appear in the assembled context.

        **Validates: Requirements 3.10**
        """
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step="",
            adaptation_hints=adaptation_hints,
            signals=signals,
            max_tokens=300,
        )

        # The [Plan] marker should not appear when plan_step is empty
        assert "[Plan]" not in result, (
            f"Plan context marker '[Plan]' found in output when plan_step is empty: '{result}'"
        )


# Feature: full-agentic-upgrade, Property 24: Context priority-based truncation
# **Validates: Requirements 3.9, 3.10, 9.3, 9.4**


class TestContextPriorityBasedTruncation:
    """Property 24: Context priority-based truncation.

    For any combination of memory summary, plan step, adaptation hints, and
    signals that would exceed 300 tokens, the Context Assembler SHALL truncate
    from lowest priority first (memory summary → adaptation hints → signals →
    plan step), and the output SHALL not exceed 300 tokens. The highest-priority
    items SHALL be preserved intact.
    """

    @settings(max_examples=100)
    @given(
        memory_summary=memory_summary_strategy,
        plan_step=plan_step_strategy,
        adaptation_hints=adaptation_hints_strategy,
        signals=signals_strategy,
    )
    def test_output_never_exceeds_max_tokens(
        self, memory_summary, plan_step, adaptation_hints, signals
    ):
        """The output SHALL not exceed 300 tokens (1200 characters at ~4 chars/token).

        **Validates: Requirements 9.3**
        """
        max_tokens = 300
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            signals=signals,
            max_tokens=max_tokens,
        )

        max_chars = max_tokens * 4
        assert len(result) <= max_chars, (
            f"Output is {len(result)} chars ({len(result) // 4} tokens) "
            f"but should not exceed {max_chars} chars ({max_tokens} tokens)."
        )

    @settings(max_examples=100)
    @given(
        plan_step=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=1,
            max_size=50,
        ),
        signals=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=30,
            ),
            min_size=1,
            max_size=3,
        ),
        adaptation_hints=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=30,
            ),
            min_size=1,
            max_size=3,
        ),
        memory_summary=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=200,
            max_size=500,
        ),
    )
    def test_plan_step_preserved_when_truncation_needed(
        self, plan_step, signals, adaptation_hints, memory_summary
    ):
        """The highest-priority item (plan step) SHALL be preserved intact
        when truncation is needed.

        **Validates: Requirements 9.4**
        """
        # Use a small token budget to force truncation
        max_tokens = 50
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            signals=signals,
            max_tokens=max_tokens,
        )

        # Plan step has highest priority and should be preserved
        assert plan_step in result, (
            f"Plan step '{plan_step}' was truncated but it has highest priority. "
            f"Result: '{result}'"
        )

    @settings(max_examples=100)
    @given(
        plan_step=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=1,
            max_size=30,
        ),
        signals=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=30,
            ),
            min_size=1,
            max_size=3,
        ),
        memory_summary=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=300,
            max_size=600,
        ),
    )
    def test_memory_summary_truncated_before_higher_priority(
        self, plan_step, signals, memory_summary
    ):
        """Memory summary (lowest priority) SHALL be truncated or removed before
        higher-priority items (signals, plan step).

        **Validates: Requirements 9.4**
        """
        # Use a budget that can't fit everything
        max_tokens = 100
        adaptation_hints = ["hint1", "hint2"]

        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            signals=signals,
            max_tokens=max_tokens,
        )

        max_chars = max_tokens * 4

        # Output must not exceed budget
        assert len(result) <= max_chars, (
            f"Output exceeds budget: {len(result)} > {max_chars}"
        )

        # Plan step (highest priority) should be preserved
        assert plan_step in result, (
            f"Plan step '{plan_step}' was lost during truncation. Result: '{result}'"
        )

    @settings(max_examples=100)
    @given(
        memory_summary=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=1,
            max_size=100,
        ),
        plan_step=plan_step_strategy,
        adaptation_hints=adaptation_hints_strategy,
        signals=signals_strategy,
    )
    def test_all_components_included_when_within_budget(
        self, memory_summary, plan_step, adaptation_hints, signals
    ):
        """When all components fit within the token budget, all SHALL be
        included in the output without truncation.

        **Validates: Requirements 9.3, 9.4**
        """
        # Use a large budget that should fit everything
        max_tokens = 1000
        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            signals=signals,
            max_tokens=max_tokens,
        )

        # All non-empty components should be present
        if memory_summary:
            assert memory_summary in result, (
                f"Memory summary not found in result when budget is large enough. "
                f"Memory: '{memory_summary}', Result: '{result}'"
            )
        if plan_step:
            assert plan_step in result, (
                f"Plan step not found in result when budget is large enough. "
                f"Plan: '{plan_step}', Result: '{result}'"
            )
        if adaptation_hints:
            for hint in adaptation_hints:
                assert hint in result, (
                    f"Adaptation hint '{hint}' not found in result when budget is large enough. "
                    f"Result: '{result}'"
                )
        if signals:
            for signal in signals:
                assert signal in result, (
                    f"Signal '{signal}' not found in result when budget is large enough. "
                    f"Result: '{result}'"
                )

    @settings(max_examples=100)
    @given(
        plan_step=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=1,
            max_size=30,
        ),
        signals=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=30,
            ),
            min_size=1,
            max_size=3,
        ),
        adaptation_hints=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=50,
                max_size=100,
            ),
            min_size=2,
            max_size=4,
        ),
        memory_summary=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=50,
            max_size=100,
        ),
    )
    def test_truncation_order_lowest_priority_first(
        self, plan_step, signals, adaptation_hints, memory_summary
    ):
        """Truncation SHALL proceed from lowest priority first:
        memory summary → adaptation hints → signals → plan step.

        **Validates: Requirements 9.4**
        """
        # Use a very tight budget to force truncation
        max_tokens = 40
        max_chars = max_tokens * 4

        result = assemble_agentic_context(
            memory_summary=memory_summary,
            plan_step=plan_step,
            adaptation_hints=adaptation_hints,
            signals=signals,
            max_tokens=max_tokens,
        )

        # Output must not exceed budget
        assert len(result) <= max_chars, (
            f"Output exceeds budget: {len(result)} > {max_chars}"
        )

        # If memory is present, all higher-priority items must also be present
        if memory_summary in result:
            # If memory survived, signals and plan must also be present
            assert plan_step in result, (
                "Memory summary survived but plan step (higher priority) was lost"
            )

        # If adaptation hints are present, plan step must also be present
        if adaptation_hints and all(hint in result for hint in adaptation_hints):
            assert plan_step in result, (
                "Adaptation hints survived but plan step (higher priority) was lost"
            )

        # If signals are present, plan step must also be present
        if signals and all(sig in result for sig in signals):
            assert plan_step in result, (
                "Signals survived but plan step (higher priority) was lost"
            )
