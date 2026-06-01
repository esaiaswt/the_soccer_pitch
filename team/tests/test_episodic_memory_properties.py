"""Property-based tests for the team/ EpisodicMemory module.

Uses Hypothesis to verify correctness properties from the design document.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from team.episodic_memory import Episode, EpisodicMemory


# --- Strategies for generating valid episodes ---

# JSON-compatible primitive values for game state and action dicts
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(max_size=20),
)

json_values = st.recursive(
    json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=3),
    ),
    max_leaves=10,
)

game_state_strategy = st.dictionaries(
    st.text(min_size=1, max_size=10),
    json_values,
    min_size=1,
    max_size=5,
)

action_strategy = st.fixed_dictionaries({
    "dx": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "dy": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "kick": st.booleans(),
})

next_state_delta_strategy = st.fixed_dictionaries({
    "ball_dx": st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    "ball_dy": st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    "possession_changed": st.booleans(),
})

effectiveness_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


def episode_strategy(cycle: int | None = None):
    """Generate a valid Episode with an optional fixed cycle number."""
    cycle_st = st.just(cycle) if cycle is not None else st.integers(min_value=0, max_value=100_000)
    return st.builds(
        Episode,
        cycle=cycle_st,
        game_state=game_state_strategy,
        action=action_strategy,
        next_state_delta=next_state_delta_strategy,
        effectiveness=effectiveness_strategy,
    )


# Strategy for a list of episodes with unique, sorted cycle numbers (chronological)
def episodes_list_strategy(min_size: int = 1, max_size: int = 200):
    """Generate a list of episodes with strictly increasing cycle numbers."""
    return st.lists(
        st.integers(min_value=0, max_value=100_000),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    ).flatmap(
        lambda cycles: st.tuples(
            *[episode_strategy(c) for c in sorted(cycles)]
        ).map(list)
    )


# Feature: full-agentic-upgrade, Property 1: Episode storage round-trip with chronological ordering
# **Validates: Requirements 1.1, 1.4**


class TestEpisodeStorageRoundTrip:
    """Property 1: Episode storage round-trip with chronological ordering.

    For any sequence of valid episodes added to an EpisodicMemory, retrieving
    all episodes SHALL return them in chronological order (by cycle number) with
    all fields (game_state, action, next_state_delta) intact and equal to the
    original values.
    """

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=50))
    def test_episodes_returned_in_chronological_order(self, episodes):
        """Episodes retrieved via get_all() SHALL be in chronological order by cycle number.

        **Validates: Requirements 1.1, 1.4**
        """
        memory = EpisodicMemory(max_capacity=200)

        for ep in episodes:
            memory.add(ep)

        retrieved = memory.get_all()

        # Verify chronological ordering
        cycles = [ep.cycle for ep in retrieved]
        assert cycles == sorted(cycles), (
            f"Episodes not in chronological order: {cycles}"
        )

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=50))
    def test_episode_fields_intact_after_storage(self, episodes):
        """All episode fields SHALL be intact and equal to original values after retrieval.

        **Validates: Requirements 1.1, 1.4**
        """
        memory = EpisodicMemory(max_capacity=200)

        for ep in episodes:
            memory.add(ep)

        retrieved = memory.get_all()

        assert len(retrieved) == len(episodes)

        for original, stored in zip(episodes, retrieved):
            assert stored.cycle == original.cycle
            assert stored.game_state == original.game_state
            assert stored.action == original.action
            assert stored.next_state_delta == original.next_state_delta
            assert stored.effectiveness == original.effectiveness

    @settings(max_examples=100)
    @given(episodes=episodes_list_strategy(min_size=1, max_size=50))
    def test_get_all_count_matches_added(self, episodes):
        """The number of episodes returned by get_all() SHALL equal the number added
        (when within capacity).

        **Validates: Requirements 1.1**
        """
        memory = EpisodicMemory(max_capacity=200)

        for ep in episodes:
            memory.add(ep)

        retrieved = memory.get_all()
        assert len(retrieved) == len(episodes)
        assert len(memory) == len(episodes)


# Feature: full-agentic-upgrade, Property 2: Capacity invariant with oldest-first eviction
# **Validates: Requirements 1.2, 1.3**


class TestCapacityInvariantWithEviction:
    """Property 2: Capacity invariant with oldest-first eviction.

    For any configured maximum capacity N and any sequence of episodes added to
    an EpisodicMemory, the memory size SHALL never exceed N, and when at capacity,
    the evicted episode SHALL always be the one with the lowest cycle number.
    """

    @settings(max_examples=100)
    @given(
        capacity=st.integers(min_value=1, max_value=50),
        episodes=episodes_list_strategy(min_size=1, max_size=100),
    )
    def test_memory_size_never_exceeds_capacity(self, capacity, episodes):
        """Memory size SHALL never exceed the configured maximum capacity N.

        **Validates: Requirements 1.2, 1.3**
        """
        memory = EpisodicMemory(max_capacity=capacity)

        for ep in episodes:
            memory.add(ep)
            assert len(memory) <= capacity, (
                f"Memory size {len(memory)} exceeds capacity {capacity}"
            )

    @settings(max_examples=100)
    @given(
        capacity=st.integers(min_value=1, max_value=20),
        episodes=episodes_list_strategy(min_size=2, max_size=60),
    )
    def test_oldest_episode_evicted_when_at_capacity(self, episodes, capacity):
        """When at capacity, the evicted episode SHALL always be the one with the
        lowest cycle number.

        **Validates: Requirements 1.2, 1.3**
        """
        memory = EpisodicMemory(max_capacity=capacity)

        for ep in episodes:
            memory.add(ep)

        retrieved = memory.get_all()

        # After adding all episodes, the memory should contain the most recent ones
        expected_count = min(len(episodes), capacity)
        assert len(retrieved) == expected_count

        # The retained episodes should be the last `capacity` episodes added
        # (i.e., the ones with the highest cycle numbers since they were added in order)
        expected_episodes = episodes[-capacity:]
        for stored, expected in zip(retrieved, expected_episodes):
            assert stored.cycle == expected.cycle, (
                f"Expected cycle {expected.cycle} but got {stored.cycle}. "
                f"Oldest episodes should have been evicted first."
            )

    @settings(max_examples=100)
    @given(
        capacity=st.integers(min_value=1, max_value=20),
        episodes=episodes_list_strategy(min_size=1, max_size=60),
    )
    def test_retained_episodes_are_most_recent(self, episodes, capacity):
        """After eviction, the retained episodes SHALL be the most recent ones
        (highest cycle numbers).

        **Validates: Requirements 1.2, 1.3**
        """
        memory = EpisodicMemory(max_capacity=capacity)

        for ep in episodes:
            memory.add(ep)

        retrieved = memory.get_all()

        if len(episodes) > capacity:
            # All retained episodes should have cycle numbers >= the evicted ones
            retained_cycles = {ep.cycle for ep in retrieved}
            evicted_episodes = episodes[:-capacity]
            for evicted_ep in evicted_episodes:
                assert evicted_ep.cycle not in retained_cycles, (
                    f"Evicted episode with cycle {evicted_ep.cycle} still in memory"
                )
                # Every retained cycle should be greater than every evicted cycle
                for retained_cycle in retained_cycles:
                    assert retained_cycle > evicted_ep.cycle, (
                        f"Retained cycle {retained_cycle} is not greater than "
                        f"evicted cycle {evicted_ep.cycle}"
                    )
