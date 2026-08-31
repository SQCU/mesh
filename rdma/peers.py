"""peers — where a mesh node IS, right now. The one way to find a peer.

An address is a property of an EDGE, not of a node. A mesh is defined by nodes
that partition and reconnect on different edges, so binding a name to an address
is guaranteed to break; the only question is when. This module binds names to
IDENTITIES and resolves identities to edges at the moment of use.

It exists because the demo spent a day reporting "mesh-mini unreachable" while
the node was up. `~/.ssh/config` said HostName 192.168.1.183 -- a subnet this
machine has not been on for some time -- and every consumer named the peer by
that string. The node was answering on 10.0.0.165 and on the Thunderbolt fabric
at 169.254.225.22 the entire time, and the fabric's own discovery knew both. One
stale string took out the policy responder, the telemetry viewer, and the
conclusion that any of it was working.

Replaces the discovery half of bin/mesh-peers.sh, which did address
classification with shell globs (`172.1[6-9].*`), framing with `nc` and
tempfiles, and concurrency with background jobs. Classification here is
`ipaddress`, which is exact; probing is `socket`, which reports its own errors.
`dns-sd` is still invoked because it IS the OS discovery primitive, but nothing
is parsed by awk and no logic lives in a subshell.
"""
from __future__ import annotations

import ipaddress
import socket
import subprocess
from dataclasses import dataclass, field

BROWSE_SERVICE = "_meshnode._tcp"
NODEINFO_PORTS = (8099, 8100)

# Preference order. Fabric first: a direct Thunderbolt link has no router in it
# and survives LAN re-addressing, which is the failure this module exists for.
KIND_RANK = {"fabric-v4ll": 0, "fabric-routed": 1, "fabric-adjacent": 2,
             "lan": 3, "lan-v6": 4}


def classify(addr: str) -> str | None:
    """Which kind of edge an address represents, or None if unusable."""
    scope = None
    if "%" in addr:
        addr, scope = addr.split("%", 1)
        # dns-sd writes "%<0>" to mean "no scope"; it is not an interface and an
        # address carrying it is not connectable.
        if scope == "<0>":
            scope = None
        elif scope == "lo0":
            return None          # loopback is not an edge to anywhere
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
    paths: list[tuple[str, str]] = field(default_factory=list)   # (kind, addr)

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
    """Node names advertising the mesh service."""
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
    """Every usable (kind, addr) currently published for one node."""
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
    """A TCP connect, which answers the question actually being asked --
    'can I talk to it' -- rather than ICMP, which a host may drop while serving."""
    try:
        with socket.create_connection((addr, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve(pattern: str, port: int = 22, timeout: float = 3.0) -> str | None:
    """The ONE address to use for a node, verified reachable. None if it has no
    live edge -- which is a real answer, distinct from a stale string."""
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
    """argv for ssh'ing to a node, addressed by identity. None if no live edge.

    The host key is pinned to the NODE NAME via HostKeyAlias, not to the address.
    A fabric edge is link-local (169.254/16) and therefore ephemeral, so writing
    it into known_hosts memoizes something that is not a fact about the node --
    which this repo had done five times before anyone noticed, and which I did
    once more today with StrictHostKeyChecking=accept-new. Pinning the alias
    means the trust survives re-addressing and the file never accumulates dead
    addresses.

    There is deliberately no ~/.ssh/config Host alias in this path. That file is
    a memoized resolution table; its entry for this node pointed at a subnet the
    machine had left, and every consumer inherited the error.
    """
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
            # accept-new applies to the ALIAS, so known_hosts accumulates node
            # NAMES -- stable, one entry per node for its lifetime -- instead of
            # a new line every time a link-local edge is renumbered.
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                    "-o", f"HostKeyAlias={node.name}",
                    "-o", "StrictHostKeyChecking=accept-new"]
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
