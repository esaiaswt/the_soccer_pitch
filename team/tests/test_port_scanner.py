"""Unit tests for the port scanner module.

Tests cover:
- is_port_available() correctly detects available and occupied ports
- find_available_port() with no configured port scans 8501-8510
- find_available_port() with configured port uses it if available
- find_available_port() falls back to scanning when configured port is occupied
- find_available_port() raises PortUnavailableError when all ports occupied
- Error message content for different failure scenarios

Requirements: 8.2, 8.3, 8.4, 9.7
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from team.port_scanner import (
    PORT_RANGE_END,
    PORT_RANGE_START,
    PortUnavailableError,
    find_available_port,
    is_port_available,
)


class TestIsPortAvailable:
    """Tests for is_port_available()."""

    def test_available_port_returns_true(self):
        """An unoccupied port should return True."""
        # Use a high port that's very unlikely to be in use
        with patch("team.port_scanner.socket.socket") as mock_socket_cls:
            mock_sock = mock_socket_cls.return_value.__enter__.return_value
            mock_sock.bind.return_value = None  # bind succeeds
            assert is_port_available(9999) is True

    def test_occupied_port_returns_false(self):
        """An occupied port should return False."""
        with patch("team.port_scanner.socket.socket") as mock_socket_cls:
            mock_sock = mock_socket_cls.return_value.__enter__.return_value
            mock_sock.bind.side_effect = OSError("Address already in use")
            assert is_port_available(8501) is False


class TestFindAvailablePortNoConfig:
    """Tests for find_available_port() with no configured port (Req 8.2)."""

    def test_returns_first_available_port_in_range(self):
        """Should return the first available port starting from 8501."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            mock_check.return_value = True
            result = find_available_port(configured_port=None)
            assert result == PORT_RANGE_START
            mock_check.assert_called_with(PORT_RANGE_START)

    def test_skips_occupied_ports(self):
        """Should skip occupied ports and return the first available one."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            # 8501 and 8502 occupied, 8503 available
            def side_effect(port):
                return port >= 8503

            mock_check.side_effect = side_effect
            result = find_available_port(configured_port=None)
            assert result == 8503

    def test_returns_last_port_in_range_if_others_occupied(self):
        """Should return 8510 if all others in range are occupied."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            def side_effect(port):
                return port == PORT_RANGE_END

            mock_check.side_effect = side_effect
            result = find_available_port(configured_port=None)
            assert result == PORT_RANGE_END

    def test_raises_error_when_all_ports_occupied(self):
        """Should raise PortUnavailableError when all ports 8501-8510 are occupied."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            mock_check.return_value = False
            with pytest.raises(PortUnavailableError) as exc_info:
                find_available_port(configured_port=None)
            assert str(PORT_RANGE_START) in str(exc_info.value)
            assert str(PORT_RANGE_END) in str(exc_info.value)


class TestFindAvailablePortWithConfig:
    """Tests for find_available_port() with configured port (Req 8.3, 9.7)."""

    def test_uses_configured_port_when_available(self):
        """Should use the configured port if it's available (Req 8.3)."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            mock_check.return_value = True
            result = find_available_port(configured_port=8888)
            assert result == 8888
            mock_check.assert_called_once_with(8888)

    def test_falls_back_to_scan_when_configured_port_occupied(self):
        """Should fall back to scanning 8501-8510 when configured port is occupied (Req 9.7)."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            def side_effect(port):
                if port == 8888:
                    return False  # configured port occupied
                return port == 8501  # first scan port available

            mock_check.side_effect = side_effect
            result = find_available_port(configured_port=8888)
            assert result == 8501

    def test_raises_error_when_configured_and_all_scan_ports_occupied(self):
        """Should raise PortUnavailableError when configured port and all scan ports are occupied (Req 8.4)."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            mock_check.return_value = False
            with pytest.raises(PortUnavailableError) as exc_info:
                find_available_port(configured_port=8888)
            assert "8888" in str(exc_info.value)
            assert str(PORT_RANGE_START) in str(exc_info.value)

    def test_configured_port_not_in_scan_range(self):
        """Configured port outside 8501-8510 should still be tried first."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            mock_check.return_value = True
            result = find_available_port(configured_port=3000)
            assert result == 3000


class TestFindAvailablePortScanOrder:
    """Tests verifying the scan order is lowest-to-highest (Req 8.2)."""

    def test_scans_in_ascending_order(self):
        """Should scan ports in ascending order from 8501 to 8510."""
        with patch("team.port_scanner.is_port_available") as mock_check:
            # Make all ports unavailable to capture all calls
            mock_check.return_value = False
            with pytest.raises(PortUnavailableError):
                find_available_port(configured_port=None)

            # Verify all ports in range were checked in order
            called_ports = [call.args[0] for call in mock_check.call_args_list]
            expected_ports = list(range(PORT_RANGE_START, PORT_RANGE_END + 1))
            assert called_ports == expected_ports
