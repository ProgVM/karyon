"""
agent_copilot_listener.py — Background Co-Pilot Monitor for Karyon-CoRE Dialogue Engine
Connects to /tmp/karyon_agent_bridge.sock via UNIX domain stream socket.
Audits real-time Free Energy, entropy, Poincare nodes, and allows injecting active guidance.
"""

import socket
import json
import time
import sys

SOCKET_PATH = "/tmp/karyon_agent_bridge.sock"

def connect_copilot():
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(SOCKET_PATH)
        client.setblocking(False)
        print(f"[CO-PILOT BRIDGE] Connected to Karyon Sovereign Substrate via {SOCKET_PATH}")
        return client
    except Exception as e:
        print(f"[CO-PILOT BRIDGE] Failed to connect: {e}")
        return None

def poll_bridge(client, timeout=0.1):
    start = time.time()
    events = []
    buffer = ""
    while time.time() - start < timeout:
        try:
            data = client.recv(4096)
            if not data:
                break
            buffer += data.decode("utf-8", errors="ignore")
            lines = buffer.split("\n")
            buffer = lines[-1]
            for l in lines[:-1]:
                if l.strip():
                    try:
                        events.append(json.loads(l))
                    except Exception:
                        pass
        except BlockingIOError:
            time.sleep(0.01)
        except Exception:
            break
    return events

def inject_directive(client, directive_str: str):
    try:
        client.sendall(directive_str.encode("utf-8"))
        print(f"[CO-PILOT INJECT] Sent directive: {directive_str}")
        return True
    except Exception as e:
        print(f"[CO-PILOT INJECT] Error: {e}")
        return False

if __name__ == "__main__":
    c = connect_copilot()
    if c:
        print("[CO-PILOT] Listening for live telemetry...")
        for _ in range(5):
            evs = poll_bridge(c, timeout=0.5)
            for ev in evs:
                print(f"  -> {ev}")
        c.close()
