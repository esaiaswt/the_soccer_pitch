"""Property-based tests for URL construction (Property 2)."""

from hypothesis import given, settings
import hypothesis.strategies as st

from config import build_url


# Strategies for generating valid server IP/hostname strings
_ip_octets = st.integers(min_value=0, max_value=255)
_ipv4_addresses = st.tuples(_ip_octets, _ip_octets, _ip_octets, _ip_octets).map(
    lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}"
)

_hostname_labels = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-"),
    min_size=1,
    max_size=20,
).filter(lambda s: not s.startswith("-") and not s.endswith("-"))

_hostnames = st.lists(_hostname_labels, min_size=1, max_size=3).map(
    lambda labels: ".".join(labels)
)

_server_ips = st.one_of(_ipv4_addresses, _hostnames, st.just("localhost"))


# Feature: agent-control-panel, Property 2: URL construction produces correct format
class TestUrlConstructionFormat:
    """Property 2: URL construction produces correct format.

    For any valid server IP string, the URL construction function SHALL produce
    URLs matching `http://{SERVER_IP}:8000/api/state` and
    `http://{SERVER_IP}:8000/api/action` exactly.

    **Validates: Requirements 2.5**
    """

    @given(server_ip=_server_ips)
    @settings(max_examples=100)
    def test_state_url_format(self, server_ip: str):
        """URL for 'state' endpoint matches http://{server_ip}:8000/api/state."""
        url = build_url(server_ip, "state")
        expected = f"http://{server_ip}:8000/api/state"
        assert url == expected, f"Expected '{expected}', got '{url}'"

    @given(server_ip=_server_ips)
    @settings(max_examples=100)
    def test_action_url_format(self, server_ip: str):
        """URL for 'action' endpoint matches http://{server_ip}:8000/api/action."""
        url = build_url(server_ip, "action")
        expected = f"http://{server_ip}:8000/api/action"
        assert url == expected, f"Expected '{expected}', got '{url}'"

    @given(server_ip=_server_ips)
    @settings(max_examples=100)
    def test_url_has_no_extra_path_segments_or_query_params(self, server_ip: str):
        """Constructed URLs have no extra path segments or query parameters."""
        for endpoint in ["state", "action"]:
            url = build_url(server_ip, endpoint)
            # Should not contain ? or # (no query params or fragments)
            assert "?" not in url, f"URL contains query params: {url}"
            assert "#" not in url, f"URL contains fragment: {url}"
            # Should have exactly the expected path segments
            path_part = url.split(f"{server_ip}:8000")[1]
            assert path_part == f"/api/{endpoint}", (
                f"Unexpected path: {path_part}"
            )
