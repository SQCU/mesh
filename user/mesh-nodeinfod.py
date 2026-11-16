import socket, subprocess, sys, os, threading, time
SCRIPT = os.environ.get("MESH_NODEINFO", os.path.expanduser("~/.local/mesh/bin/mesh-nodeinfo.sh"))
PORT = int(os.environ.get("MESH_NODEINFO_PORT", "8100"))
LATEST = (b"", "starting", 0.0)

def sample():
    global LATEST
    while True:
        lines, status = [], "failed"
        try:
            with subprocess.Popen(["/bin/bash", SCRIPT], stdout=subprocess.PIPE) as process:
                for line in process.stdout:
                    if line.strip() != b"end": lines.append(line)
                status = "complete" if process.wait() == 0 else "failed"
        except Exception as error:
            print(f"nodeinfo sample: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        LATEST = (b"".join(lines), status, time.time()) if lines else (LATEST[0], status, LATEST[2])
        time.sleep(5)

s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try: s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
except Exception as error: print(f"nodeinfo dual-stack: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
s.bind(("::", PORT)); s.listen(64)
threading.Thread(target=sample, daemon=True).start()
while True:
    try: c, _ = s.accept()
    except Exception as error:
        print(f"nodeinfo accept: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        continue
    try:
        sample_body, status, updated = LATEST
        body = sample_body + f"sample_status={status} sample_updated={updated:.3f} sample_age={max(0, time.time()-updated):.3f}\nend\n".encode()
        c.settimeout(5)
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
