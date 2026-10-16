from __future__ import annotations

import concurrent.futures
import errno
import functools
import ipaddress
import os
import re
import selectors
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

BROWSE_SERVICE = "_meshnode._tcp"
NODEINFO_PORTS = (8100, 8099)
HOST_LABEL = re.compile(r"[A-Za-z0-9-]+")
CONTROL = Path.home() / ".local/share/mesh/ssh"

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

def host_of(instance: str, timeout: float = 1.0) -> str | None:
    """The host label an instance of the beacon's service can be reached at: the instance's own name where it is a host
    label, else its SRV target (dns-sd -L; mDNS renames an instance whose name another holds 'name (2)', and its host
    'name-2'); None where no answer comes within `timeout`."""
    if HOST_LABEL.fullmatch(instance):
        return instance
    for line in _dns_sd(["-t", str(max(1, int(timeout))), "-L", instance, BROWSE_SERVICE, "local"], timeout + 1):
        if " can be reached at " in line:
            return line.split(" can be reached at ", 1)[1].split()[0].rsplit(":", 1)[0].removesuffix(".").removesuffix(".local")
    return None

def browse(timeout: float = 3.0) -> list[str]:
    """The host label of every instance of the beacon's service that announces itself within `timeout` seconds: the
    instance name is the whole of dns-sd's last column, spaces included."""
    instances = set()
    for line in _dns_sd(["-t", str(int(timeout)), "-B", BROWSE_SERVICE, "local"], timeout + 1):
        f = line.split(None, 6)
        if len(f) == 7 and f[1] == "Add":
            instances.add(f[6].strip())
    with concurrent.futures.ThreadPoolExecutor(max(1, len(instances))) as pool:
        return sorted({h for h in pool.map(host_of, sorted(instances)) if h})

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
    """Every node the beacon's service announces within `timeout` seconds (browse), each with its addresses, every
    node's looked up at once."""
    me, names = local_name(), browse(timeout)
    with concurrent.futures.ThreadPoolExecutor(max(1, len(names))) as pool:
        found = list(pool.map(lambda n: addresses(n, timeout), names))
    return [Node(n, n == me, paths) for n, paths in zip(names, found)]

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

def node_number(node_id: str) -> int:
    """The bridge node number of the host whose IOPlatformUUID is `node_id` (node-info's node_id): mesh-flow.c main's
    FNV-1 of the host UUID's 16 bytes (gethostuuid), its low 31 bits, 1 for 0."""
    number = 0
    for byte in uuid.UUID(node_id).bytes:
        number = (number * 0x01000193 & 0xffffffff) ^ byte
    return number & 0x7fffffff or 1

def identity(body: str) -> int | None:
    """The bridge node number a node-info sample's node_id names, None where it names none."""
    try:
        return node_number(observation(body)["node_id"])
    except (KeyError, ValueError):
        return None

def routes(body: str) -> list[str]:
    """The fabric addresses a node-info sample's node routes to (its `route=` lines: babeld's routes to other nodes'
    ULAs), in its order."""
    return [fields["route"] for line in body.splitlines()
            if "route" in (fields := dict(word.split("=", 1) for word in line.split() if "=" in word))]

def resolved(address: str, timeout: float) -> list:
    """socket.getaddrinfo of `address` within `timeout` seconds (an mDNS name's lookup is not otherwise bounded);
    OSError naming it where none comes."""
    done, out = threading.Event(), {}

    def look():
        try:
            out["found"] = socket.getaddrinfo(address, None, type=socket.SOCK_STREAM)
        except OSError as error:
            out["error"] = error
        done.set()
    threading.Thread(target=look, daemon=True).start()
    if not done.wait(timeout):
        raise OSError(f"{address}: no address in {timeout} s")
    if "error" in out:
        raise OSError(f"{address}: {out['error']}")
    return out["found"]

def where(target: tuple) -> str:
    """A socket address as address%interface:port (the interface where it has a scope)."""
    scope = target[3] if len(target) > 3 else 0
    return f"{target[0]}%{socket.if_indextoname(scope)}:{target[1]}" if scope else f"{target[0]}:{target[1]}"

def nodeinfo(address: str | list[str], timeout: float = 2.0, limit: int = 1 << 20, stagger: float = 0.25,
             accept=None) -> tuple[str, str, list[str]]:
    """A node's node-info sample from `address` (a name or an address, or a list of them in the order they are
    tried): every port of NODEINFO_PORTS on an address at once (ports are services on one path), the addresses in
    order (each name's in resolver order, the names resolved at once) `stagger` seconds apart, or at once when no
    attempt is pending; the first whole sample that `accept` takes (accept(sample): None, else why not, as another
    node's) answers, and the reading ends `timeout` seconds past the last address's turn, so every address has at
    least `timeout`; the names' resolution is inside that bound.  (the sample, the address:port that gave it, why each
    attempt that ended before it gave none); OSError, naming every attempt, where none gives one."""
    began, asked, errors = time.monotonic(), [address] if isinstance(address, str) else list(address), []

    def look(name):
        try:
            return resolved(name, timeout)
        except OSError as error:
            errors.append(str(error))
            return []
    with concurrent.futures.ThreadPoolExecutor(max(1, len(asked))) as pool:
        found = list(pool.map(look, asked))
    addresses = [[(family, kind, proto, (sockaddr[0], port, *sockaddr[2:])) for port in NODEINFO_PORTS]
                 for each in found for family, kind, proto, _, sockaddr in each]
    deadline = began + timeout + stagger * max(0, len(addresses) - 1)
    chooser, pending, started = selectors.DefaultSelector(), {}, 0.0

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
                    text = sample(bytes(data), limit)
                except ValueError as error:
                    fail(client, error)
                    continue
                refused = accept(text) if accept else None
                if refused:
                    fail(client, refused)
                    continue
                return text, where(target), errors
        errors += [f"{where(target)}: no sample in {timeout} s" for target, _ in pending.values()]
        raise OSError("; ".join(errors) or f"{', '.join(asked)}: no address")
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

def resolve(name: str, port: int = 22, timeout: float = 3.0) -> str | None:
    """The first address answering on `port` of the node announced as `name` (its host label, matched whole, ignoring
    case); None where no address answers; ValueError naming them where several nodes carry it."""
    found = [n for n in discover(timeout) if not n.is_self and n.name.lower() == name.lower()]
    if len(found) > 1:
        raise ValueError(f"{name} names {len(found)} nodes: their addresses {[n.ranked() for n in found]}")
    for _kind, addr in (found[0].ranked() if found else []):
        if reachable(addr, port, timeout=2.0):
            return addr
    return None

@functools.lru_cache(maxsize=1)
def roster() -> tuple[str, ...]:
    """The private keys of ~/.ssh whose public halves the mesh's roster authorizes (this checkout's keys/authorized_keys,
    else etc/authorized_keys; bin/mesh-run.sh passes the same)."""
    root = Path(__file__).resolve().parent.parent
    listed = next((p for p in (root / "keys/authorized_keys", root / "etc/authorized_keys") if p.is_file()), None)
    blobs = {line.split()[1] for line in (listed.read_text().splitlines() if listed else [])
             if len(line.split()) >= 2 and not line.startswith("#")}
    return tuple(str(public.with_suffix("")) for public in sorted((Path.home() / ".ssh").glob("*.pub"))
                 if public.with_suffix("").is_file() and len(public.read_text().split()) >= 2
                 and public.read_text().split()[1] in blobs)

def ssh_argv(name: str, address: str, node: int, account: str | None = None) -> list[str]:
    """ssh's argv, up to its command, for the node whose bridge is node `node`: the destination its name, so the
    driver's ssh configuration's Host entry for that name applies (its User and IdentityFile); HostName `address`,
    where its node-info answered with that node's identity (identity), so a node with no shared LAN is reached along
    the fabric's routes; its host key kept under mesh-<node>, the first accepted unasked; the account its node-info
    names (`account`: the node's mesh account) where it publishes one; every key of the driver's the roster authorizes;
    batch mode, a 5 s connect, and a multiplexed connection of the mesh's own (CONTROL) whose keepalive ends it within
    about 3 s of the node falling silent."""
    CONTROL.mkdir(mode=0o700, parents=True, exist_ok=True)
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=1",
            "-o", "ServerAliveCountMax=3", "-o", "ControlMaster=auto", "-o", f"ControlPath={CONTROL}/%C",
            "-o", "ControlPersist=60", "-o", f"HostName={address}", "-o", f"HostKeyAlias=mesh-{node}",
            "-o", "StrictHostKeyChecking=accept-new"]
    for key in roster():
        argv += ["-i", key]
    return argv + (["-l", account] if account else []) + [name]

if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["nodeinfo"] and len(sys.argv) == 3:
        try:
            print(nodeinfo(sys.argv[2])[0], end="")
        except (OSError, ValueError) as error:
            sys.exit(f"nodeinfo {sys.argv[2]}: {error}")
        sys.exit(0)
    if len(sys.argv) > 1:
        try:
            a = resolve(sys.argv[1])
        except ValueError as error:
            sys.exit(str(error))
        print(a if a else "", end="\n" if a else "")
        sys.exit(0 if a else 1)
    for n in discover():
        print(f"node {n.name} {'self' if n.is_self else 'peer'}")
        for kind, addr in n.ranked():
            print(f"  {kind:16} {addr}")
