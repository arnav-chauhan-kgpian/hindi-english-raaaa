"""Offline guard for the official scored run. After model warmup the harness calls
block_network(); any outbound socket to a non-loopback address then raises. Loopback
stays open so a *local* ASR server (127.0.0.1) is allowed — that's still local-only.
"""
from __future__ import annotations
import socket

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _is_local(addr) -> bool:
    try:
        host = addr[0]
    except (TypeError, IndexError):
        return False
    return host in _LOOPBACK or str(host).startswith("127.")


class NetworkBlocked(RuntimeError):
    pass


def block_network():
    def guarded_connect(self, addr):
        if not _is_local(addr):
            raise NetworkBlocked(f"outbound network blocked during scoring: {addr}")
        return _real_connect(self, addr)

    def guarded_connect_ex(self, addr):
        if not _is_local(addr):
            raise NetworkBlocked(f"outbound network blocked during scoring: {addr}")
        return _real_connect_ex(self, addr)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


def restore_network():
    socket.socket.connect = _real_connect
    socket.socket.connect_ex = _real_connect_ex
