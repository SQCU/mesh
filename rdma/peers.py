from __future__ import annotations

import ipaddress
import socket
import subprocess
from dataclasses import dataclass, field

BROWSE_SERVICE = "_meshnode._tcp"
NODEINFO_PORTS = (8099, 8100)

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
        return getattr(e, "stdout", None) and e.stdout.splitlines() or []

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
    if len(sys.argv) > 1:
        a = resolve(sys.argv[1])
        print(a if a else "", end="\n" if a else "")
        sys.exit(0 if a else 1)
    for n in discover():
        print(f"node {n.name} {'self' if n.is_self else 'peer'}")
        for kind, addr in n.ranked():
            print(f"  {kind:16} {addr}")
