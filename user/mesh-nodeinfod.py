import socket, subprocess, sys, os
SCRIPT = os.path.expanduser("~/.local/mesh/bin/mesh-nodeinfo.sh")
s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try: s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
except Exception: pass
s.bind(("::", 8100)); s.listen(32)
while True:
    try:
        c, _ = s.accept()
    except Exception:
        continue
    try:
        c.sendall(subprocess.run(["/bin/bash", SCRIPT], capture_output=True, timeout=8).stdout)
    except Exception:
        pass
    finally:
        try: c.close()
        except Exception: pass
