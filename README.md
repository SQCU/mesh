# mesh

Provisioning for a fabric of Apple Silicon Macs wired together with Thunderbolt and
talking RDMA. The invariant: **a node may never become unreachable, and may never
decide on its own to stop executing.**

Nodes are expected to be unplugged, carried to another room, and replugged into a
different position in the mesh. Nothing here encodes cable position.

**The threat is the false negative** — a node that should be working and reachable
and silently is not, for any reason, however locally sensible. Stock macOS ships in
exactly that vulnerable mode: it sleeps, it locks, it waits for consent, and it
reboots itself when it judges that wise. Correcting that default is the only work
that justifies a human being physically present. See [THREAT-MODEL.md](THREAT-MODEL.md)
before changing settings that look wrong.

## The one step that cannot be automated

`rdma_ctl enable` only runs from Recovery OS:

```
$ sudo rdma_ctl enable
rdma_ctl: This tool needs to be executed from Recovery OS.    exit=77
```

Entering 1TR requires physically holding the power button, which is a deliberate
Apple anti-tampering property — no MDM, script, or remote console can do it.
**Budget exactly one physical visit per machine, forever.** Everything else here
exists to make that visit the only one.

## Bringing up a new node

Hands-on, once, ~5 minutes:

1. **Setup Assistant** — create the admin account. Leave FileVault **off**: it would
   demand a keyboard at every boot, which breaks the invariant. Encrypt payloads at
   the application layer instead. Confirm Activation Lock is Disabled.
2. **Recovery** — hold power → Terminal → `rdma_ctl enable` → reboot. Writes
   `rdma-enable=1` to NVRAM, which then persists.
3. **Bootstrap** — open Terminal and run:

   ```
   curl -fsSL https://raw.githubusercontent.com/SQCU/mesh/main/bootstrap.sh | sudo bash
   ```

   This opens sshd *first*, then fetches and runs `install.sh`. Confirm port 22
   answers before walking away.

Everything after that is remote: `ssh node.local` and re-run `sudo install.sh`.

> The failure to guard against here is not "the wrong code ran" — that is answered by
> replacing the machine. It is a bootstrap that half-completes and leaves the node
> unreachable. That is why `bootstrap.sh` opens sshd *before* it fetches anything:
> if every later step fails, the machine is still remotely recoverable.
> Set `MESH_BRANCH` to fetch a branch other than `main`. Never index by commit —
> see **Never index version control by commit** in [AGENTS.md](AGENTS.md).

## What provisioning does that looks wrong

Repo reasoning, not operator policy — see [THREAT-MODEL.md](THREAT-MODEL.md) for what
was actually asked for. Each of these is open to revisiting:

- **FileVault stays off.** It demands a keyboard at every boot, so a node under it
  cannot reboot unattended. Encrypt payloads at the application layer instead,
  whenever that is wanted — nothing here argues data at rest is unimportant.
- **The application firewall is off**, and the keeper re-asserts that. It can only
  subtract reachability. This is the weakest-supported item here and the first to
  revisit if a node ever lives somewhere with less physical control.
- **Passwordless sudo** via `/etc/sudoers.d/mesh`, `visudo -c` validated before it is
  written. A password prompt on a keyboard-less node blocks both recovery and any
  future agent that provisions a machine and then has to administer it.
- **The provisioning source is a public repo**, so a node reaches a consistent
  definition with no credentials.

> `nc -w` does not bound the connect on macOS. Probing an unreachable node with
> `nc -w 2` blocks for **75 seconds**; `nc -G 2 -w 2` returns immediately. Anything
> that enumerates the fabric must use `-G`, or one dead node stalls the whole report
> and the cost grows with the fleet.

## Traversal is a stream, and nodes patch themselves

`mesh-peers` emits facts as it learns them and never waits to prove a negative:

```
node <addr> via=<iface>     emitted from the kernel routing table, at t=0
info <addr> name=... ...    emitted when that node answers, in arrival order
```

The `node` lines are the traversal result and they are already complete — babeld did
that work, and reading the table costs microseconds. The `info` lines are best-effort
enrichment that streams in unordered. A node that never answers simply has no `info`
line; its absence is the signal, and observing it costs nothing.

`MESH_DEADLINE` (default 2s) is a horizon, not a per-node timeout. It does not grow
with the fleet: a hundred nodes still finish in one deadline, and one dead node slows
nothing. The consumer reads a stream and deals with out-of-order arrival. There is no
mode that buffers into an aligned table, because aligning columns means waiting for
the slowest row, which means one unreachable node delays the report on every
reachable one.

`mesh-run` follows the same shape: every node runs in parallel and each output line is
prefixed with its node, streaming as it arrives.

**`io.mesh.update` re-converges from the branch every 15 minutes**, as root. Before
this existed, every fix required a human at a keyboard typing `sudo` on each node —
which does not scale and is the same shape of problem as a node that cannot be
reached. Now a push propagates to the fleet on its own. The node follows the branch
recorded in `/usr/local/mesh/revision`, always at that branch's newest commit.

## Joining without root

A machine can become a fully identified, relaying fabric partner with **no `sudo` at
all**, provided the privileged parts are already in place (bridge down, ports up and
addressed, `ip6.forwarding=1`, babeld running). Those are one-time and survive; the
parts that change often do not need root:

```
./install-user.sh
```

It installs a node-identity responder as a **user LaunchAgent** on port 8100 and puts
`mesh-peers` / `mesh-run` in `~/.local/mesh/bin`. Nothing touches `/usr` or `/Library`.

Port 8100 rather than 8099 because a stale system `io.mesh.nodeinfo` may already hold
8099 in the root domain. `mesh-peers` probes 8099 first and falls back to 8100, so a
node whose system responder is broken or outdated is still fully identified by its
userspace shim.

`inetdCompatibility` does **not** work in the GUI/user domain — launchd accepts the
connection, closes it, and never runs the program (`runs = 0`), while the socket looks
perfectly healthy. Hence a plain listener under `KeepAlive` instead.

### Discovery does not require routing

`mesh-peers` also reads `ndp` for each fabric port and emits its neighbours:

```
neigh fe80::34b5:96ff:feb2:55c8%en2 on=en2
info  fe80::34b5:96ff:feb2:55c8%en2 name=Ms-Mac-mini model=Mac16,11 ...
```

Adjacency is knowable from the kernel with no protocol involved, so a node is findable
over a cable even if it never advertises a routable address. Relaying is independent of
this too: babeld re-advertises *learned* routes by default — the `redistribute` filter
only governs a node's own local routes — so a node forwards for its neighbours whether
or not it announces itself.

## RDMA: measured, and the constraint that bites first

Verified between `Mac16,11` and `Mac17,6` over one Thunderbolt 5 cable with
`ibv_uc_pingpong`, GID index 0 (the EUI-64 link-local this repo assigns):

| message | rx-depth | frames | throughput | latency |
|---|---|---|---|---|
| 4 KB | 100 | 100 | 6.4 Gbit/s | 10.2 us |
| 64 KB | 60 | 960 | 47.1 Gbit/s | 22.3 us |
| 256 KB | 15 | 960 | 64.5 Gbit/s | 65.0 us |
| 1 MB | 3 | 768 | 70.9 Gbit/s | 237 us |

**Queues are counted in 4 KB frames, not in messages.** `size/4096 * rx_depth` must
stay under 4095. The first attempt here asked for 64 KB messages at the default
rx-depth of 500 — 8000 frames — and failed as `Failed to modify QP to RTR` on one side
and `Operation not permitted` on the other, which reads like a permissions or fabric
problem and is neither.

Other limits worth knowing before writing against this, all from TN3205: send/receive
only (`IBV_WR_SEND`, `IBV_ACCESS_LOCAL_WRITE`), 10 UC queue pairs per device, ~16 MB
maximum message, sender and receiver must post the **same frame count**, no hardware
ACK so integrity and retransmit are the application's job, and a per-controller IOMMU
so a buffer is registered separately per device.

RDMA never routes. It is point-to-point per cable, and the out-of-band GID/QPN
exchange only needs a socket to the directly connected peer — which is why none of
this depends on the routing layer.

> [RDMA-FIRST.md](RDMA-FIRST.md) is the design economics: what is cheap, what is
> expensive, and why the usual backpressure and framing machinery is the wrong shape
> here. Read it before designing; read RDMA-RULES.md before running.

> Before writing or running anything that opens a verbs device, read
> [RDMA-RULES.md](RDMA-RULES.md). Killing a stuck verbs process can wedge a node
> badly enough to need a physical power cycle.

## Links, and knowing when you only have one

A node reaches the world over Wi-Fi/LAN and over the Thunderbolt fabric. It is
tempting to call the first a control plane and the second a data plane. That is
backwards, and the label hides a real dependency:

- **Wi-Fi is opportunistic.** Whether a network exists is a property of the room, not
  of the node. A MacBook cannot host one for the minis, so a venue with no Wi-Fi means
  no Wi-Fi, and nothing on the node can change that.
- **The fabric is the dependable one.** Copper, point to point, between two machines
  we own. It needs no infrastructure that anyone else controls.

So the fabric can be a node's *only* link, and control has to ride it when that is the
case — `ssh`, `mesh-peers` and `mesh-run` all work over `fabric-adjacent` link-local
addresses for exactly this reason. Breaking the fabric is a reachability failure, not
an inconvenience, which is why [RDMA-RULES.md](RDMA-RULES.md) is a reachability
document.

The invariant is not "every link comes back" — that cannot be promised for a network
that may not exist. It is: **a node comes back with every link the room actually
offers, and reports all of them**, so a drop to one is visible rather than discovered
later:

```
LINKS
  wifi/lan:  up via en1
  fabric:    up
  redundant: 2 of 2
```

`mesh-nodeinfo` carries `planes=N lan=<iface> fabric=<0|1>`, and the keeper logs
`DEGRADED` every pass while only one remains. Do not run anything that can disrupt the
remaining link while degraded.

### The list has to have more than one entry

macOS auto-joins any known network in range, so "find an available network" needs no
code. It needs a list, and a list of one is a venue change away from needing a
keyboard. `networks.conf` (gitignored — it holds live credentials) seeds every node,
`install.sh` copies it to `/usr/local/mesh/networks.conf`, and the keeper re-asserts
it. Entries are only added, never removed. Format is in `networks.conf.example`.

## Workload design studies

`design/` holds design studies for workloads that run on the fabric. They are
proposals with their reasoning and their adversarial critique attached, not settled
fact, and each states plainly which of its numbers are measured and which are
estimates.

- [`design/mesh-coprocessor-demo.md`](design/mesh-coprocessor-demo.md) — **current.**
  The mini as a tile-streaming matrix coprocessor, demonstrated through a multi-team
  Xonotic payload match. Measured hardware, three superlinear levers, the ABI, and the
  attacks each part of the workload survives.
- [`design/mesh-coprocessor-demo-plain.md`](design/mesh-coprocessor-demo-plain.md) — the
  same design at reading grade 3.7: what the linear algebra computes, and what the game
  must show so the solver is visibly present and visibly necessary.
- [`design/xonotic-bot-compute.md`](design/xonotic-bot-compute.md) — **superseded.** The
  earlier mean-field study. Kept for its discard table and its adversarial critiques; its
  throughput figures were estimates and its crossover analysis is void.

## Adding a machine to the fabric

Steps 1-3 above, then cable it to any free Thunderbolt port on any existing node —
position does not matter and nothing records it. Then verify, from either end:

```
mesh-status                     # rdma_enX PORT_ACTIVE, io.mesh.* all loaded
mesh-peers                      # the new node appears, with a name and a VIA
mesh-run all 'sysctl -n hw.model'
```

What each step proves, in order, so a failure tells you where to look:

| check | passes when | fails if |
|---|---|---|
| `rdma_ctl status` = enabled | the Recovery visit happened | needs a physical 1TR visit; nothing remote fixes it |
| some `rdma_enX` is `PORT_ACTIVE` | a cable is in, both ends RDMA-enabled | cable, or the far end never had `rdma_ctl enable` |
| `ifconfig <port>` has an `fe80::` address | `mesh-fabric-init` ran | check `/usr/local/mesh/log/fabric.log` |
| the peer appears in `mesh-peers` | babeld exchanged routes | check `/usr/local/mesh/log/router.log`; confirm both nodes advertise |
| `mesh-run <name> true` succeeds | the roster reached the node | re-run `install.sh`; keys merge, they never truncate |

A node that answers `mesh-peers` and `mesh-run` is administrable and workloadable.
Nothing else is required of it.

## Is anything missing?

```
mesh-peers        # every node announcing on the fabric right now
mesh-status       # this node's power, lifelines, RDMA, daemons, and peers
```

## Why sshd is enabled via launchctl

The documented path is gated behind Full Disk Access, which can only be granted
through the GUI or an MDM profile — impossible on a headless virgin machine:

```
$ sudo systemsetup -f -setremotelogin on
setremotelogin: Turning Remote Login on or off requires Full Disk Access privileges.
```

`launchctl bootstrap` needs only root and no TCC grant, so that is what `install.sh`
and `bootstrap.sh` use. This is the hinge the whole remote-provisioning story turns on.

## The Thunderbolt bridge is torn down, not configured

macOS bridges every Thunderbolt port into `bridge0` by default. `mesh-fabric-init`
destroys it on every node, every boot, and every keeper pass. Two independent
reasons, both measured rather than assumed:

**It has no loop protection.** `ifconfig bridge0` advertises `proto stp`, but the
bridge id is all zeros, no root is elected, and members carry only
`<LEARNING,DISCOVER>` with no port state. STP is named, not running. Cable any ring
and frames circulate forever. [TN3205][tn] says the same and tells you to disable
the bridge before connecting Macs in loops. Rings are the point of this fabric: with
three ports per machine, anything past a chain has a cycle in it.

**It strips the RDMA GID table.** A bridged member interface holds no addresses of
its own — they live on `bridge0` — and an RDMA device derives its GIDs from its
paired IP interface. Bridged, every `rdma_enX` reports `GID[0]` alone. TN3205's own
sample code queries GID index 1, which does not exist in that state.

Apple's documented fix is not sufficient. Measured on `Mac16,11`: after
`networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off`, `bridge0` was
still `UP`, `RUNNING`, `status: active`, with all three members attached and a live
address-cache entry. Making the service inactive drops the IPv4 address and nothing
else. The bridge interface itself has to be destroyed.

[tn]: https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt

## Addressing

No static IPs. Identity travels with the machine:

- Bonjour `<LocalHostName>.local`
- IPv6 link-local on each fabric port, EUI-64 from that port's own MAC

That link-local is *exactly* the value the paired RDMA device reports as `GID[0]`,
so a port's IP address and its RDMA identity are the same number. Nothing depends on
a GID index, and the address encodes node identity rather than cable position — it
survives being unplugged and replugged into a different port on a different machine.

macOS does not autoconfigure link-local on an interface with no network service, so
`mesh-fabric-init` assigns it. It deliberately does **not** create SystemConfiguration
services for these ports: those writes need Full Disk Access and fail headlessly with
`Unable to access the System Configuration database`, and configd has no business
renumbering an RDMA fabric.

`net.inet6.ip6.forwarding` is on, because in a ring a node must transit for its
neighbours. RDMA itself never routes — TN3205 is explicit that the application
forwards — so this reaches nodes, it does not carry fabric payload.

Discover neighbours on a given port:

```
ping6 -c3 -I en4 ff02::1
```

None of this state persists on its own: configd rebuilds `bridge0` on Thunderbolt
hotplug and `ifconfig` addresses are runtime-only. So it runs at boot as
`io.mesh.fabric` and the keeper re-asserts it every pass. A silently re-bridged node
is one that will storm the moment the fabric grows a ring.

The Thunderbolt bridge service is kept **last** in network service order for as long
as it exists, so a peer can never advertise itself as the default route and blackhole
the node's uplink.

## Layout

| path | what |
|---|---|
| `install.sh` | provision/converge a node. Idempotent — re-running is the drift fix. |
| `bootstrap.sh` | one-shot entry for the physical visit. Opens sshd, then runs `install.sh`. |
| `enable-autologin.sh` | boot into a GUI session so session-bound workloads survive a reboot. |
| `AGENTS.md` | **the contract.** Accessibility is required; rejection filtering is banned. Read before changing anything. |
| `THREAT-MODEL.md` | why several settings here look wrong under a normal hardening model. |
| `bin/mesh-status.sh` | one-screen health report. Installed as `mesh-status`. |
| `bin/mesh-peers.sh` | enumerate live nodes on the fabric. Installed as `mesh-peers`. Absence is the alarm. |
| `bin/mesh-beacon.sh` | continuous `_meshnode._tcp` announcement. Must never exit voluntarily. |
| `bin/mesh-keeper.sh` | 60s watchdog: re-asserts power policy, re-bootstraps sshd/screen sharing/beacon. |
| `hmi-epilogue.sh` | post-provision operator override for power + firewall on a human-used machine. Never affects reachability. |
| `bin/mesh-devtools-init.sh` | headless Command Line Tools install, so every node can build RDMA code. Idempotent. |
| `bin/mesh-router-init.sh` | identity `/128` on `lo0`, babeld across every fabric port. Resident; keeper restarts it. |
| `vendor/babeld-arm64` | prebuilt Babel daemon, so a node needs no toolchain. Provenance in `vendor/PROVENANCE.md`. |
| `bin/mesh-nodeinfo.sh` | socket-activated node identity + topology responder on 8099. |
| `bin/mesh-run.sh` | run a command on one node, `all`, or `others`. Installed as `mesh-run`. |
| `bin/mesh-fabric-init.sh` | tear down the Thunderbolt bridge, address each fabric port, enable IPv6 forwarding. Idempotent; run at boot and every keeper pass. |
| `bin/mesh-rdma-init.sh` | boot-time fabric verification. Verify-only by necessity. |
| `keys/authorized_keys` | the pubkey roster every node trusts. Public keys only. |
| `templates/` | LaunchDaemon template for workloads. Boot-time, KeepAlive, no login needed. |
| `docs/` | the bring-up writeup. `./serve.sh` to read it locally. |

## No GUI, including for the toolchain

Nodes need Apple's SDK: `infiniband/verbs.h` and `librdma.tbd` ship only in the
Command Line Tools, and TN3205 requires both to build anything against RDMA. So
every node needs CLT, not just ones that compile the router.

CLT installs headlessly. The trap is only the *implicit* path — invoking bare `cc`
with no toolchain opens the GUI installer. The explicit path never does:

```
sudo touch /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
softwareupdate -l          # now lists "Command Line Tools for Xcode 26.6-26.6"
sudo softwareupdate -i "Command Line Tools for Xcode 26.6-26.6" --verbose
sudo rm /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
```

The marker file is load-bearing, not folklore: without it `softwareupdate -l` lists
**zero** Command Line Tools labels; with it, two. `mesh-devtools-init` does this and
is idempotent — it exits immediately when `verbs.h` is already resolvable.

Nothing else in the stack needs a toolchain either. `uv` is a prebuilt binary linking
only system frameworks, and `uv python list` offers prebuilt CPython, so Python is
available on a virgin node with no compiler at all.

## Routing, and why the node list comes from the routing table

With the bridge gone, each Thunderbolt port is its own point-to-point link. Bonjour
and link-local reach exactly one cable hop, so multicast discovery stops describing
the fabric the moment a third node exists. Anything built on `ff02::1` silently
degrades from "the mesh" to "my neighbours" — a report that still prints a table and
no longer means what it says.

So every node takes a routable identity address and the fabric is routed:

- **Identity**: a `/128` under `fd6d:6573:68::/48` on `lo0`, derived from the node's
  `IOPlatformUUID`. On `lo0` because with no bridge there is no single segment for it
  to live on, and it must not change when a cable moves to another port.
- **Routing**: `babeld` on every fabric port, using the per-port link-locals as
  next-hops. It is designed for topologies that churn, reconverges on replug with no
  per-node configuration, and uses every link rather than blocking one the way a
  spanning tree would — which is the whole reason to cable a ring.
- **Enumeration**: `mesh-peers` reads the routing table. Every node babeld knows about
  is a node, at any hop count and any fleet size, with no central registry and no
  multicast.

babeld is vendored as a prebuilt arm64 binary — see [vendor/PROVENANCE.md](vendor/PROVENANCE.md)
for the source, hashes, and the one-line build fix. A node must never need a
toolchain: invoking `cc` on a virgin Mac opens the Command Line Tools GUI installer.

RDMA itself is untouched by any of this and still never routes. TN3205 is explicit
that the application forwards across a topology. This layer exists so that nodes can
*find and reach* each other at unlimited fleet size; the RDMA data plane rides the
cables directly, point to point.

## Names, traversal, and running things

`mesh-peers` shows what the fabric is, not just which addresses exist:

```
  NODE               MODEL        CORES MEM    RDMA      ADDRESS                                VIA
  Ms-Mac-mini        Mac16,11     12    24G    enabled   fd6d:6573:68:3af8:1a3c:9700:3034:715d  self
```

That comes from `io.mesh.nodeinfo`, a launchd socket-activated service on port 8099.
There is no resident daemon and no language runtime: launchd accepts the connection
and hands the socket to a shell script as stdin/stdout. Each node reports its name,
hardware, macOS, RDMA and SDK state, every fabric port with its state, GID and the
peer seen on it, and every route it knows.

That last part is what makes the topology discoverable. Ask each node what it can see
and you can reconstruct the whole graph — no central registry, no multicast, no
hop limit.

Running work anywhere:

```
mesh-run all      'sysctl -n hw.model'
mesh-run others   'uptime'
mesh-run Ms-Mac-mini 'ls /usr/local/mesh'
```

Job submission needs no new machinery. Every node already runs sshd and trusts the
operator roster; the only missing piece was knowing which nodes exist, which the
routing table answers.

> If you write a plist to `/Library/LaunchDaemons` with `inetdCompatibility`, do not
> also set `StandardOutPath`. It overrides the socket launchd dups onto stdout, and
> the service answers every connection with silence while looking perfectly healthy.

## Which definition a node is running

Fetching is by branch only — `main`, or whatever `MESH_BRANCH` names. Never by commit;
the reasoning is in [AGENTS.md](AGENTS.md), and it is not negotiable because a pinned
node cannot receive the change that would unpin it.

Branch-addressed content on `raw.githubusercontent.com` carries `cache-control:
max-age=300`, so a node can briefly fetch a copy up to five minutes old. That is a
wait, not a fault: it clears itself, and a node that needs the newest definition
sooner can simply be reached and re-run. Do not reach for a commit hash to dodge it —
that trades a transient for a permanent.

What matters is that convergence is visible rather than silent. Each node records the
branch it converged from and when:

```
mesh-status         branch main converged 2026-08-28T00:56:27Z
mesh-peers          BRANCH / CONVERGED columns across the whole fabric
```

A node that has not converged recently is the thing worth noticing, and now you can
see it from any machine.

## HMI epilogue

Provisioning gives every node the same policy: never sleep, firewall off, no screen
lock. That is correct for a machine in a rack and wrong for one someone is typing on.

`hmi-epilogue.sh` is an operator action run *after* provisioning, on a machine a human
uses:

```
sudo ./hmi-epilogue.sh
MESH_HMI_SLEEP=30 MESH_HMI_DISPLAYSLEEP=10 sudo ./hmi-epilogue.sh
```

It writes `/usr/local/mesh/policy`. The keeper does not consult a flag and decide
whether to act — it always asserts whatever that file says. `install.sh` writes the
fleet default; the epilogue overwrites it; the keeper converges on it either way. So
there is no branch anywhere that can skip a capability, only one policy the keeper
always enforces. Return to fleet policy by re-running `sudo ./install.sh`.

Everything governing *reachability* is outside the policy file entirely and is
asserted unconditionally every pass: sshd, screen sharing, the beacon, the Thunderbolt
fabric, routing, network time, and the RDMA alarm.

## Reading the docs

```
./serve.sh          # http://localhost:8080
```

No venv, no dependencies — `uv run --no-project python -m http.server`.

## Adding a workload

Copy `templates/io.mesh.job-EXAMPLE.plist` to
`/Library/LaunchDaemons/io.mesh.job-<name>.plist`, point it at your script, then:

```
sudo launchctl bootstrap system /Library/LaunchDaemons/io.mesh.job-<name>.plist
```

Runs as root at boot with no login session. `KeepAlive` restarts it forever.

## Using the mesh

An application sees three functions, in `rdma/mesh.h`. Everything below a page — queue pairs,
memory regions, lkeys, wire headers, residency, retransmission — belongs to the bridge and
does not appear.

```c
void  *mesh_open(size_t bytes, size_t *stride, size_t *usable);
size_t mesh_write(const void *p, size_t nbytes, int node);
size_t mesh_read(void **p, int *from);
```

`mesh_open` maps a share of this node and returns memory. `mesh_write` hands a range to
another node and returns how much it took, so a caller loops on the remainder; there is no
size at which it refuses. `mesh_read` returns one arrived slot and recycles the previous one,
which is why there is no release call.

Memory comes in slots rather than one flat span because the device reports `max_sge: 1` and
has no immediate data, so every page carries its own header contiguously. The gap is 16 bytes
in 4096.

**One thing an application must know.** `mesh_write` takes the bytes, but nothing tells you
when a page is free again. Rotate through your arena rather than reusing the same slots; the
head of the arena may still be in flight. Reusing it corrupted 34 of 18041 rows before this
was understood.

From Python, `rdma/mesh.py` binds the same three functions and returns numpy views of mesh
memory, so MLX computes on pages the NIC wrote without a copy.

### Running it

```
bin/mesh-bridge.sh start | stop | restart | status
```

It is a launchd service, like `io.mesh.beacon` and `io.mesh.router`. Settings come from
`etc/bridge.conf`: how much of the node the mesh holds, and how much your own programs expect
to use. The wire limit is raised to match, and startup is refused above 90% of RAM, because
wiring past that can leave a node that cannot boot without someone physically present.
Changing the file and restarting is the supported way to change how much of a node the mesh
holds; nothing is tuned while it runs.

Never reach for `pkill`. A process holding a verbs device does not die from SIGKILL — it sits
in uninterruptible kernel sleep still holding the device, every other process on the machine
loses RDMA, the ports drop with the cable attached, `shutdown` hangs, and a human has to walk
to the machine and pull power. `stop` sends SIGTERM, waits for the teardown that releases the
device, and reports rather than escalating. The plist sets `ExitTimeOut` to 0, which in launchd
means infinity, so launchd will not escalate either. `bin/mesh-kill-guard.sh` enforces this;
`AGENTS.md` describes what it covers and what it cannot.

On a headless node there is no GUI launchd domain, so the script re-execs under `sudo` when it
can. Running as root there means the region is created world-readable on purpose: a region
only root can open is a mesh no application can attach to.

The bridge itself takes only what it cannot infer: `-I` node index, `-M` share of the machine,
`-s` region, `-T` duration, and a peer. There is no switch for the page size, the registration
chunk, the send window or the device — each of those can select a run that looks healthy and
moves corrupt data. Unknown options are refused, and the link is found by looking for the port
that is up.

### Building, and keeping nodes identical

```
make -C rdma all
```

One command, every binary. Converge a node with git — `git fetch && git reset --hard
origin/main && make -C rdma all` — never by copying a hand-picked list of files. Mismatched
binaries between two nodes are indistinguishable from a transport regression when the only
thing you look at is throughput, and that mistake accounted for every wild number measured
during this bridge's development.

### Measuring

There is one instrument. The bridge publishes a census of the page table and its variance to
the region once a second; `rdma/mesh-stat` reads it; `viz/serve.py` polls it and serves the
viewer. Rates come from differencing `sent` and `recvd` against `uptime_ms`.

Applications do not measure. `rdma/mesh-app` keeps no clock and computes no throughput; it
reports only what it alone knows, which is whether bytes came back wrong. A second measurement
path only disagrees with the first, and reading a rate off an application whose peer had
silently failed to start is how several wrong numbers were reported here.

### What runs today

`rdma/mesh_coproc.py` holds a weight matrix resident on one node and applies it to rows
streamed from the other. 54337 rows verified exactly, `wrong=0`. That path runs at
0.12 Gbit/s, bounded by a Python loop touching one slot at a time; the C client carries
59 Gbit/s over the same link with `corrupt=0`. Trust the correctness number, not the rate.
