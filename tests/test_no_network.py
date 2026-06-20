"""Proves the scored run blocks outbound network but allows local loopback.
Run: python tests/test_no_network.py"""
import sys, os, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from offline_guard import block_network, restore_network, NetworkBlocked

def check(name, cond):
    print(("PASS" if cond else "FAIL"), name); assert cond, name

block_network()

# outbound to the internet must fail
blocked = False
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(2)
    s.connect(("8.8.8.8", 53))
except NetworkBlocked:
    blocked = True
except Exception:
    blocked = True  # any failure means it didn't reach the internet
check("outbound internet blocked", blocked)

# loopback (a local ASR server) is still allowed to attempt
local_ok = True
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.2)
    s.connect_ex(("127.0.0.1", 8001))  # no server needed; must not be NetworkBlocked
except NetworkBlocked:
    local_ok = False
except Exception:
    pass  # connection refused is fine — it wasn't blocked by the guard
check("local loopback allowed", local_ok)

restore_network()
print("\nALL NO-NETWORK TESTS PASSED")
