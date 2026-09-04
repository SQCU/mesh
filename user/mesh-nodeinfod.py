import socket, subprocess, sys, os, struct
SCRIPT = os.environ.get("MESH_NODEINFO", os.path.expanduser("~/.local/mesh/bin/mesh-nodeinfo.sh"))
PORT = int(os.environ.get("MESH_NODEINFO_PORT", "8100"))
s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try: s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
except Exception as error: print(f"nodeinfo dual-stack: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
s.bind(("::", PORT)); s.listen(64)
while True:
    try: c, _ = s.accept()
    except Exception as error:
        print(f"nodeinfo accept: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        continue
    try:
        body = subprocess.run(["/bin/bash", SCRIPT], capture_output=True, timeout=8).stdout
        c.sendall(b"mesh1 %d\n" % len(body) + body)
        c.shutdown(socket.SHUT_WR)
        c.settimeout(5)
        try:
            while c.recv(4096):
                continue
        except Exception as error:
            print(f"nodeinfo receive: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    except Exception as error:
        print(f"nodeinfo response: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        try: c.close()
        except Exception as error: print(f"nodeinfo close: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
