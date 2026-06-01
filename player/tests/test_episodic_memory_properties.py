"""Property-based tests for the EpisodicMemory module (Properties 1 and 2).

# Feature: full-agentic-upgrade, Property 1: Episode storage round-trip with chronological ordering
# Feature: full-agentic-upgrade, Property 2: Capacity invariant with oldest-first eviction
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from episodic_memory import Episode, EpisodicMemory


# --- Strategies ---

game_state_strategy = st.fixed_dictionaries(
    {
        "ball_x": st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
        "ball_y": st.floats(min_value=-350, max_value=350, allow_nan=False, allow_infinity=False),
        "player_x": st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
        "player_y": st.floats(min_value=-350, max_value=350, allow_nan=False, allow_infinity=False),
    }
)

action_strategy = st.fixed_dictionaries(
    {
        "dx": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "dy": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "kick": st.booleans(),
    }
)

next_state_delta_strategy = st.fixed_dictionaries(
    {
        "ball_dx": st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        "ball_dy": st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        "possession_changed": st.booleans(),
    }
)

effectiveness_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


def episode_strategy(cycle: int | None = None):
    """Generate a valid Episode with an optional fixed cycle number."""
    cycle_st = st.just(cycle) if cycle is not None else st.integers(min_value=0, max_value=10000)
    return st.builds(
        Episode,
        cycle=cycle_st,
        game_state=game_state_strategy,
        action=action_strategy,
        next_state_delta=next_state_delta_strategy,
        effectiveness=effectiveness_strategy,
    )


episodes_with_unique_cycles = st.lists(
    st.integers(min_value=0, max_value=100000), min_size=1, max_size=50, unique=True
).flatmap(
    lambda cycles: st.tuples(
        st.just(sorted(cycles)),
        st.tuples(*[episode_strategy(c) for c in sorted(cycles)]),
    )
)


# Feature: full-agentic-upgrade, Property 1: Episode storage round-trip with chronological ordering
class TestEpisodeStorageRoundTripChronologicalOrdering:
    """Property 1: Episode storage round-trip with chronological ordering.

    For any sequence of valid episodes added to an EpisodicMemory, retrieving
    all episodes SHALL return them in chronological order (by cycle number)
    with all fields (game_state, action, next_state_delta) intact and equal
    to the original values.

    **Validates: Requirements 1.1, 1.4**
    """

    @given(data=episodes_with_unique_cycles)
    @settings(max_examples=100)
    def test_episodes_returned_in_chronological_order(self, data):
        """Episodes are returned in chronological order by cycle number."""
        cycles, episodes = data
        memory = EpisodicMemory(max_capacity=100)

        for episode in episodes:
            memory.add(episode)

        retrieved = memory.get_all()

        # Verify chronological ordering
        retrieved_cycles = [ep.cycle for ep in retrieved]
        assert retrieved_cycles == sorted(retrieved_cycles), (
            f"Episodes not in chronological order: {retrieved_cycles}"
        )

    @given(data=episodes_with_unique_cycles)
    @settings(max_examples=100)
    def test_episode_fields_intact_after_storage(self, data):
        """All episode fields are preserved after storage and retrieval."""
        cycles, episodes = data
        memory = EpisodicMemory(max_capacity=100)

        for episode in episodes:
            memory.add(episode)

        retrieved = memory.get_all()

        # Build a lookup by cycle for comparison
        original_by_cycle = {ep.cycle: ep for ep in episodes}

        for ep in retrieved:
            original = original_by_cycle[ep.cycle]
            assert ep.game_state == original.game_state, (
                f"game_state mismatch at cycle {ep.cycle}"
            )
            assert ep.action == original.action, (
                f"action mismatch at cycle {ep.cycle}"
            )
            assert ep.next_state_delta == original.next_state_delta, (
                f"next_state_delta mismatch at cycle {ep.cycle}"
            )
            assert ep.effectiveness == original.effectiveness, (
                f"effectiveness mismatch at cycle {ep.cycle}"
            )

    @given(
        episodes=st.lists(episode_strategy(), min_size=1, max_size=30),
    )
    @settings(max_examples=100)
    def test_episode_count_matches_additions_within_capacity(self, episodes):
        """Number of stored episodes equals number added when within capacity."""
        memory = EpisodicMemory(max_capacity=100)

        for episode in episodes:
            memory.add(episode)

        assert len(memory) == len(episodes)
        assert len(memory.get_all()) == len(episodes)


# Feature: full-agentic-upgrade, Property 2: Capacity invariant with oldest-first eviction
class TestCapacityInvariantOldestFirstEviction:
    """Property 2: Capacity invariant with oldest-first eviction.

    For any configured maximum capacity N and any sequence of episodes added
    to an EpisodicMemory, the memory size SHALL never exceed N, and when at
    capacity, the evicted episode SHALL always be the one with the lowest
    cycle number.

    **Validates: Requirements 1.2, 1.3**
    """

    @given(
        max_capacity=st.integers(min_value=1, max_value=20),
        num_episodes=st.integers(min_value=1, max_value=50),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_memory_size_never_exceeds_capacity(self, max_capacity, num_episodes, data):
        """Memory size never exceeds the configured maximum capacity."""
        memory = EpisodicMemory(max_capacity=max_capacity)

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)
            assert len(memory) <= max_capacity, (
                f"Memory size {len(memory)} exceeds capacity {max_capacity}"
            )

    @given(
        max_capacity=st.integers(min_value=1, max_value=10),
        num_episodes=st.integers(min_value=1, max_value=40),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_oldest_episode_evicted_when_at_capacity(self, max_capacity, num_episodes, data):
        """When at capacity, the evicted episode is always the one with the lowest cycle number."""
        memory = EpisodicMemory(max_capacity=max_capacity)
        all_added = []

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)
            all_added.append(episode)

        # The memory should contain the most recent max_capacity episodes
        expected_episodes = all_added[-max_capacity:]
        retrieved = memory.get_all()

        assert len(retrieved) == len(expected_episodes)

        # Verify the retained episodes are the most recent ones (highest cycles)
        for retrieved_ep, expected_ep in zip(retrieved, expected_episodes):
            assert retrieved_ep.cycle == expected_ep.cycle, (
                f"Expected cycle {expected_ep.cycle}, got {retrieved_ep.cycle}. "
                f"Oldest episodes should be evicted first."
            )

    @given(
        max_capacity=st.integers(min_value=1, max_value=15),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_eviction_preserves_chronological_order(self, max_capacity, data):
        """After eviction, remaining episodes are still in chronological order."""
        memory = EpisodicMemory(max_capacity=max_capacity)
        num_episodes = data.draw(st.integers(min_value=max_capacity + 1, max_value=max_capacity * 3))

        for i in range(num_episodes):
            episode = data.draw(episode_strategy(cycle=i))
            memory.add(episode)

        retrieved = memory.get_all()
        retrieved_cycles = [ep.cycle for ep in retrieved]

        assert retrieved_cycles == sorted(retrieved_cycles), (
            f"Episodes not in chronological order after eviction: {retrieved_cycles}"
        )
        assert len(retrieved) == max_capacity

