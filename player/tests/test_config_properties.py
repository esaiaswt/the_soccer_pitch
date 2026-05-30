"""Property-based tests for the config module (Properties 1 and 3)."""

from hypothesis import given, settings
import hypothesis.strategies as st
from pydantic import ValidationError

from config import validate_api_key, ActionModel


# Feature: agent-control-panel, Property 1: API key validation rejects all empty-like values
class TestApiKeyValidationRejectsEmptyLikeValues:
    """Property 1: API key validation rejects all empty-like values.

    For any string that is empty, None, or composed entirely of whitespace
    characters, the API key validation function SHALL reject it.

    **Validates: Requirements 1.5**
    """

    @given(
        whitespace=st.text(
            alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"]),
            min_size=0,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_whitespace_only_strings_are_rejected(self, whitespace: str):
        """Any string composed entirely of whitespace characters is rejected."""
        assert validate_api_key(whitespace) is False

    def test_none_is_rejected(self):
        """None is rejected by the API key validation."""
        assert validate_api_key(None) is False

    def test_empty_string_is_rejected(self):
        """Empty string is rejected by the API key validation."""
        assert validate_api_key("") is False

    @given(
        key=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != "")
    )
    @settings(max_examples=100)
    def test_non_empty_non_whitespace_strings_are_accepted(self, key: str):
        """Any string with at least one non-whitespace character is accepted."""
        assert validate_api_key(key) is True


# Feature: agent-control-panel, Property 3: ActionModel accepts valid ranges and rejects invalid ranges
class TestActionModelValidRanges:
    """Property 3: ActionModel accepts valid ranges and rejects invalid ranges.

    For any float value for dx and dy, the ActionModel SHALL accept values
    within [-1.0, 1.0] inclusive, and SHALL reject values outside that range.

    **Validates: Requirements 4.1**
    """

    @given(
        dx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        kick=st.booleans(),
    )
    @settings(max_examples=100)
    def test_valid_range_values_are_accepted(self, dx: float, dy: float, kick: bool):
        """Values within [-1.0, 1.0] for dx and dy are accepted."""
        model = ActionModel(dx=dx, dy=dy, kick=kick)
        assert model.dx == dx
        assert model.dy == dy
        assert model.kick == kick

    @given(
        dx=st.floats(min_value=1.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        kick=st.booleans(),
    )
    @settings(max_examples=100)
    def test_dx_above_range_is_rejected(self, dx: float, dy: float, kick: bool):
        """dx values above 1.0 are rejected with ValidationError."""
        try:
            ActionModel(dx=dx, dy=dy, kick=kick)
            assert False, f"Expected ValidationError for dx={dx}"
        except ValidationError:
            pass

    @given(
        dx=st.floats(min_value=-1000.0, max_value=-1.01, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        kick=st.booleans(),
    )
    @settings(max_examples=100)
    def test_dx_below_range_is_rejected(self, dx: float, dy: float, kick: bool):
        """dx values below -1.0 are rejected with ValidationError."""
        try:
            ActionModel(dx=dx, dy=dy, kick=kick)
            assert False, f"Expected ValidationError for dx={dx}"
        except ValidationError:
            pass

    @given(
        dx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=1.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        kick=st.booleans(),
    )
    @settings(max_examples=100)
    def test_dy_above_range_is_rejected(self, dx: float, dy: float, kick: bool):
        """dy values above 1.0 are rejected with ValidationError."""
        try:
            ActionModel(dx=dx, dy=dy, kick=kick)
            assert False, f"Expected ValidationError for dy={dy}"
        except ValidationError:
            pass

    @given(
        dx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=-1000.0, max_value=-1.01, allow_nan=False, allow_infinity=False),
        kick=st.booleans(),
    )
    @settings(max_examples=100)
    def test_dy_below_range_is_rejected(self, dx: float, dy: float, kick: bool):
        """dy values below -1.0 are rejected with ValidationError."""
        try:
            ActionModel(dx=dx, dy=dy, kick=kick)
            assert False, f"Expected ValidationError for dy={dy}"
        except ValidationError:
            pass
