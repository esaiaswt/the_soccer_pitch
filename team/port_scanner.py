"""Port scanning and auto-assignment for the Streamlit dashboard.

Provides utilities to find an available port for the Team Dashboard,
either using a configured port from STREAMLIT_PORT env var or by
scanning ports 8501-8510 for the first available one.

Requirements: 8.2, 8.3, 8.4, 9.7
"""

from __future__ import annotations

import socket

# Default port range for auto-assignment
PORT_RANGE_START = 8501
PORT_RANGE_END = 8510


class PortUnavailableError(Exception):
    """Raised when no available port can be found."""

    pass


def is_port_available(port: int) -> bool:
    """Check if a given port is available for binding.

    Attempts to bind a TCP socket to the specified port on localhost.
    If binding succeeds, the port is available.

    Parameters
    ----------
    port : int
        The port number to check.

    Returns
    -------
    bool
        True if the port is available, False if it is already in use.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("localhost", port))
            return True
    except OSError:
        return False


def find_available_port(configured_port: int | None = None) -> int:
    """Find an available port for the Streamlit dashboard.

    Logic:
    1. If configured_port is provided (from STREAMLIT_PORT env var):
       - Check if it's available. If yes, return it.
       - If not available, fall back to scanning ports 8501-8510 (Req 9.7).
    2. If no configured_port (None):
       - Scan ports 8501-8510 in order, return the first available one (Req 8.2).
    3. If all ports in the scan range are occupied, raise PortUnavailableError (Req 8.4).

    Parameters
    ----------
    configured_port : int | None
        The port configured via STREAMLIT_PORT env var, or None if not set.

    Returns
    -------
    int
        The first available port.

    Raises
    ------
    PortUnavailableError
        If the configured port is unavailable and all ports in 8501-8510 are
        occupied, or if no configured port is set and all ports in the range
        are occupied.
    """
    # If a configured port is provided, try it first (Req 8.3)
    if configured_port is not None:
        if is_port_available(configured_port):
            return configured_port
        # Configured port unavailable — fall back to auto-scan (Req 9.7)

    # Scan ports 8501-8510 for first available (Req 8.2)
    for port in range(PORT_RANGE_START, PORT_RANGE_END + 1):
        if is_port_available(port):
            return port

    # All ports occupied (Req 8.4)
    if configured_port is not None:
        raise PortUnavailableError(
            f"Configured port {configured_port} is in use and all auto-assign "
            f"ports ({PORT_RANGE_START}-{PORT_RANGE_END}) are also occupied. "
            f"Cannot start the Team Dashboard."
        )
    else:
        raise PortUnavailableError(
            f"All ports in range {PORT_RANGE_START}-{PORT_RANGE_END} are "
            f"occupied. Cannot start the Team Dashboard."
        )
