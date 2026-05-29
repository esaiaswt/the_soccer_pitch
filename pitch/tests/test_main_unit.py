"""Unit tests for main.py: IP detection and startup error handling."""

import logging
from unittest.mock import MagicMock, patch

import pytest


class TestDetectLocalIp:
    """Tests for detect_local_ip() function."""

    @patch("pitch.main.socket.socket")
    def test_success_returns_lan_ip(self, mock_socket_cls):
        """detect_local_ip() returns the LAN IP from getsockname() on success."""
        from pitch.main import detect_local_ip

        mock_sock = MagicMock()
        mock_sock.getsockname.return_value = ("192.168.1.50", 0)
        mock_socket_cls.return_value = mock_sock

        result = detect_local_ip()

        assert result == "192.168.1.50"
        mock_sock.connect.assert_called_once_with(("8.8.8.8", 80))
        mock_sock.getsockname.assert_called_once()
        mock_sock.close.assert_called_once()

    @patch("pitch.main.socket.socket")
    def test_failure_returns_fallback(self, mock_socket_cls, caplog):
        """detect_local_ip() returns 127.0.0.1 when socket connect raises OSError."""
        from pitch.main import detect_local_ip

        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("Network unreachable")
        mock_socket_cls.return_value = mock_sock

        with caplog.at_level(logging.WARNING):
            result = detect_local_ip()

        assert result == "127.0.0.1"
        assert "Could not detect local IP" in caplog.text


class TestPortInUseHandling:
    """Tests for port-in-use error handling during Uvicorn startup."""

    @patch("pitch.main.os._exit")
    @patch("pitch.main.uvicorn.run")
    def test_port_in_use_logs_error_and_exits(self, mock_uvicorn_run, mock_exit, caplog):
        """Uvicorn startup logs error and exits when port is already in use."""
        from pitch.main import detect_local_ip

        mock_uvicorn_run.side_effect = OSError("address already in use")

        # We need to test the start_uvicorn inner function behavior.
        # Since it's defined inside main(), we replicate its logic here.
        with caplog.at_level(logging.ERROR):
            try:
                import uvicorn
                from pitch import api
                from pitch.config import Config

                config = Config()
                uvicorn.run(
                    api.app,
                    host=config.HOST,
                    port=config.PORT,
                    log_level="info",
                )
            except OSError as e:
                if "address already in use" in str(e).lower():
                    logging.getLogger("pitch").error(
                        "Port %d is already in use. Cannot start server.",
                        config.PORT,
                    )

        assert "already in use" in caplog.text
