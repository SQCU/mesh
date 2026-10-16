from __future__ import annotations

import errno
import ipaddress
import os
import selectors
import socket
import subprocess
import time
from dataclasses import dataclass, field

BROWSE_SERVICE = "_meshnode._tcp"
NODEINFO_PORTS = (8100, 8099)

KIND_RANK = {"fabric-v4ll": 0, "fabric-routed": 1, "fabric-adjacent": 2,
             "lan": 3, "lan-v6": 4}

def classify(addr: str) -> str | None:
    scope = None
    if "%" in addr:
        addr, scope = addr.split("%", 1)

        if scope == "<0>":
            scope = None
        elif scope == "lo0":
            return None
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return None
    if ip.is_loopback or ip.is_unspecified:
        return None
    if ip.version == 4:
        if ip.is_link_local:
            return "fabric-v4ll"
        if ip.is_private:
            return "lan"
        return None
    if ip.is_link_local:
        return "fabric-adjacent" if scope else None
    if addr.lower().startswith("fd6d:6573:68:"):
        return "fabric-routed"
    if ip.is_global:
        return "lan-v6"
    return None

@dataclass
class Node:
    name: str
    is_self: bool = False
    paths: list[tuple[str, str]] = field(default_factory=list)

    def ranked(self) -> list[tuple[str, str]]:
        return sorted(self.paths, key=lambda p: KIND_RANK.get(p[0], 99))

def _dns_sd(args: list[str], timeout: float) -> list[str]:
    try:
        p = subprocess.run(["dns-sd", *args], capture_output=True, text=True,
                           timeout=timeout)
        return p.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        output = getattr(e, "stdout", None) or ""
        return (output.decode(errors="replace") if isinstance(output, bytes) else output).splitlines()

def browse(timeout: float = 3.0) -> list[str]:
    names = []
    for line in _dns_sd(["-t", str(int(timeout)), "-B", BROWSE_SERVICE, "local"], timeout + 1):
        f = line.split()
        if len(f) >= 7 and f[1] == "Add":
            names.append(f[-1])
    return sorted(set(names))

def local_name() -> str:
    try:
        return subprocess.run(["scutil", "--get", "LocalHostName"],
                              capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return socket.gethostname().split(".")[0]

def addresses(name: str, timeout: float = 3.0) -> list[tuple[str, str]]:
    out = []
    for line in _dns_sd(["-t", str(int(timeout)), "-G", "v4v6", f"{name}.local"], timeout + 1):
        f = line.split()
        if len(f) >= 6 and f[1] == "Add":
            kind = classify(f[5])
            if kind:
                out.append((kind, f[5]))
    return sorted(set(out), key=lambda p: KIND_RANK.get(p[0], 99))

def discover(timeout: float = 3.0) -> list[Node]:
    me = local_name()
    return [Node(n, n == me, addresses(n, timeout)) for n in browse(timeout)]

def sample(data: bytes, limit: int) -> str:
    if len(data) > limit:
        raise ValueError("nodeinfo exceeds local byte limit")
    header, separator, body = data.partition(b"\n")
    if header.startswith(b"mesh1 "):
        if not separator or len(body) != int(header.split()[1]):
            raise ValueError("incomplete nodeinfo frame")
    else:
        body = data
    if not body.rstrip().endswith(b"end"):
        raise ValueError("incomplete nodeinfo sample")
    return body.decode()

def where(target: tuple) -> str:
    """A socket address as address%interface:port (the interface where it has a scope)."""
    scope = target[3] if len(target) > 3 else 0
    return f"{target[0]}%{socket.if_indextoname(scope)}:{target[1]}" if scope else f"{target[0]}:{target[1]}"

def nodeinfo(address: str, timeout: float = 2.0, limit: int = 1 << 20,
             stagger: float = 0.25) -> tuple[str, str, list[str]]:
    """A node's node-info sample from `address` (a name or an address): every port of NODEINFO_PORTS on an address at
    once (ports are services on one path), the name's addresses in resolver order `stagger` seconds apart, or at once
    when no attempt is pending; the first whole sample answers, and the reading ends `timeout` seconds past the last
    address's turn, so every address has at least `timeout`.  (the sample, the address:port that gave it, why each
    attempt that ended before it gave none); OSError, naming every attempt, where none gives one."""
    addresses = [[(family, kind, proto, (sockaddr[0], port, *sockaddr[2:])) for port in NODEINFO_PORTS]
                 for family, kind, proto, _, sockaddr in socket.getaddrinfo(address, None, type=socket.SOCK_STREAM)]
    deadline = time.monotonic() + timeout + stagger * max(0, len(addresses) - 1)
    chooser, pending, errors, started = selectors.DefaultSelector(), {}, [], 0.0

    def fail(client, why):
        chooser.unregister(client)
        target = pending.pop(client)[0]
        errors.append(f"{where(target)}: {why}")
        client.close()

    try:
        while time.monotonic() < deadline and (addresses or pending):
            if addresses and (not pending or time.monotonic() >= started + stagger):
                started = time.monotonic()
                for family, kind, proto, target in addresses.pop(0):
                    client = socket.socket(family, kind, proto)
                    client.setblocking(False)
                    code = client.connect_ex(target)
                    if code in (0, errno.EINPROGRESS):
                        pending[client] = [target, None]
                        chooser.register(client, selectors.EVENT_WRITE)
                    else:
                        errors.append(f"{where(target)}: {os.strerror(code)}")
                        client.close()
                continue
            wait = deadline - time.monotonic()
            for key, _ in chooser.select(max(0.0, min(wait, started + stagger - time.monotonic()) if addresses else wait)):
                client, (target, data) = key.fileobj, pending[key.fileobj]
                if data is None:
                    code = client.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                    if code:
                        fail(client, os.strerror(code))
                        continue
                    try:
                        client.shutdown(socket.SHUT_WR)
                    except OSError as error:
                        fail(client, error)
                        continue
                    pending[client][1] = bytearray()
                    chooser.modify(client, selectors.EVENT_READ)
                    continue
                try:
                    chunk = client.recv(65536)
                except BlockingIOError:
                    continue
                except OSError as error:
                    fail(client, error)
                    continue
                data.extend(chunk)
                if chunk and len(data) <= limit:
                    continue
                try:
                    return sample(bytes(data), limit), where(target), errors
                except ValueError as error:
                    fail(client, error)
        errors += [f"{where(target)}: no sample in {timeout} s" for target, _ in pending.values()]
        raise OSError("; ".join(errors) or f"{address}: no address")
    finally:
        for client in pending:
            client.close()
        chooser.close()

def observation(body: str) -> dict:
    facts = {}
    for line in body.splitlines():
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        if not fields.keys() & {"port", "route"}:
            facts.update(fields)
    return facts

def reachable(addr: str, port: int = 22, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((addr, port), timeout=timeout):
            return True
    except OSError:
        return False

def resolve(pattern: str, port: int = 22, timeout: float = 3.0) -> str | None:
    pat = pattern.lower().replace("mesh-", "")
    for node in discover(timeout):
        if node.is_self or pat not in node.name.lower().replace("-", ""):
            continue
        for _kind, addr in node.ranked():
            if reachable(addr, port, timeout=2.0):
                return addr
    return None

def ssh_argv(pattern: str, *command: str, identity: str | None = None,
             timeout: float = 3.0) -> list[str] | None:
    node = None
    pat = pattern.lower().replace("mesh-", "")
    for n in discover(timeout):
        if not n.is_self and pat in n.name.lower().replace("-", ""):
            node = n
            break
    if node is None:
        return None
    for _kind, addr in node.ranked():
        if reachable(addr, 22, timeout=2.0):

            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                    "-o", f"HostKeyAlias={node.name}",
                    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
            if identity:
                argv += ["-i", identity, "-o", "IdentitiesOnly=yes"]
            return argv + [addr, *command]
    return None

if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["nodeinfo"] and len(sys.argv) == 3:
        try:
            print(nodeinfo(sys.argv[2])[0], end="")
        except (OSError, ValueError) as error:
            sys.exit(f"nodeinfo {sys.argv[2]}: {error}")
        sys.exit(0)
    if len(sys.argv) > 1:
        a = resolve(sys.argv[1])
        print(a if a else "", end="\n" if a else "")
        sys.exit(0 if a else 1)
    for n in discover():
        print(f"node {n.name} {'self' if n.is_self else 'peer'}")
        for kind, addr in n.ranked():
            print(f"  {kind:16} {addr}")
