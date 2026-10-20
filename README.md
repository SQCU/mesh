# mesh

The [collective goal](GOAL.md): the lowest attainable overhead collective operations for
arbitrary programs on any number of Macs in any cabling ([rdma/NCCL.md](rdma/NCCL.md)). No ordering or
routing is committed to ahead of the hardware.

Provisioning for a fabric of Apple Silicon Macs wired together with Thunderbolt and
talking RDMA. The invariant: **a node may never become unreachable, and may never
decide on its own to stop executing.**

Nodes are expected to be unplugged, carried to another room, and replugged into a
different position in the mesh. Nothing here encodes cable position.

**The threat is the false negative** — a node that should be working and reachable
and silently is not, for any reason, however locally sensible. Stock macOS ships in
exactly that vulnerable mode: it sleeps, it locks, it waits for consent, and it
reboots itself on an internal policy. Correcting that default is the only work
that justifies a human being physically present. See [THREAT-MODEL.md](THREAT-MODEL.md)
before changing settings that look wrong.

A node stays awake by its power policy (`sleep 0`, `standby 0`, `autopoweroff 0`, `autorestart 1`, re-asserted by
the keeper), never by a process holding assertions. Until 2026-10-13 `io.mesh.caffeinate` ran `caffeinate -dimsu`,
which kept the display on and declared the user active forever, so `dasd` never found the node idle and never ran
the system's own idle maintenance: the Neural Engine daemon's compiled-model cache, which SIP lets only
`aned`'s garbage collection delete (`com.apple.aned.danglingModelsGC`, `com.apple.aned.modelCacheGC`, refused 7,750
times in two weeks on "Device Activity Policy"), grew to about 390 GB of a 460 GB disk. A full disk is the false
negative this fleet exists to prevent; the display sleeps after 10 minutes so the node can be idle by macOS's
definition while it keeps computing.

Each keeper pass runs `bin/mesh-health.sh`, which writes `/var/run/mesh-health`: disk free, used and floor, and every
process holding a user-activity or display assertion now, with how long it has held it (what keeps `dasd` from finding
the node idle). Below twice the floor those holders become a warning (an attended machine may keep its display on for
good, and starved housekeeping only matters once space runs short), and the report also measures what `du` can read
against what is used (the rest is space only the OS accounts for) and the largest directories under /Users, once each
time the disk enters that state. Each condition worth attention is a `warn=` line; the keeper writes the warnings
whenever they change to `keeper.log` and to the unified log (`log show --last 1d --predicate 'process == "logger"'`,
which is written even when the disk is under the floor and the keeper's own log is not). `mesh-status` prints the
report, and node-info carries its numbers and `health_warnings`.

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

**Nodes of your own.** The roster every node trusts is `keys/authorized_keys` (public keys, each labeled
`owner:machine`). Fork this repository, append your public key to it, and bootstrap from the fork:

```
curl -fsSL https://raw.githubusercontent.com/<you>/mesh/main/bootstrap.sh | sudo MESH_REPO=<you>/mesh bash
```

A node remembers the repository it was bootstrapped from (`/usr/local/mesh/repo`) and converges from it
(`mesh-update`), the bridge and the library with it (`rdma/NCCL.md` "Build and run").

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

## Traversal is a stream, and nodes patch themselves

`mesh-peers` emits facts as they arrive and never waits to prove a negative; it runs until it is stopped:

```
node <name> self|peer               each node this node's bridge holds a record of (its map), at once; then each node
                                    that announces _meshnode._tcp (Bonjour) on the shared LAN, as it announces
info <name> via=bridge node=... model=... memory=... window=...   its bridge's record of the node
info <name> via=mdns model=... rdma=...                           the node's announcement, as it is resolved
path <name> kind=<kind> addr=<a>    each address the node advertises: lan, lan-v6, fabric-adjacent (a link-local
                                    on one of this node's RDMA ports), fabric-routed (its fd6d:6573:68:: identity),
                                    fabric-v4ll
gone <name>                         an announcement withdrawn
```

The fabric's nodes are the bridge's map, so a node cabled to this one is known along its cable and never asked over
Wi-Fi (the operator, 2026-10-16); its facts are node-info's, which its resident rank answers (metal-microbench
docs/elastic.md#call-delivery). Discovery on the shared LAN is its mDNS (git 1587656: control
traffic needs no fabric), so it is not limited to one cable hop; a node on no shared LAN is found only over its own
cables. Every line arrives when its event does: a slow node delays only its own lines, and a node that is not there
is simply absent. The consumer reads a stream and deals with out-of-order arrival.

**`io.mesh.update` re-converges from the branch every 15 minutes**, as root. Before
this existed, every fix required a human at a keyboard typing `sudo` on each node —
which does not scale and is the same shape of problem as a node that cannot be
reached. Now a push propagates to the fleet on its own. The node follows the branch
recorded in `/usr/local/mesh/revision`, always at that branch's newest commit. Each pass
installs the files that changed and loads each daemon that is not loaded; a loaded daemon is left running, and picks up
what changed by restarting itself (launchd, below). The
admin user is the one `/usr/local/mesh/admin-user` records (`MESH_USER` changes it).

## Joining without root

A machine can become a fully identified, relaying fabric partner with **no `sudo` at
all**, provided the privileged parts are already in place (bridge down, ports up and
addressed, `ip6.forwarding=1`, babeld running). Those are one-time and survive; the
parts that change often do not need root:

```
./install-user.sh
```

It puts `mesh-peers` in `~/.local/mesh/bin`. Nothing touches `/usr` or `/Library`.

### Reaching a node does not require routing

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
ACK, and a per-controller IOMMU so a buffer is registered separately per device.

RDMA never routes. It is point-to-point per cable, and the out-of-band GID/QPN
exchange only needs a socket to the directly connected peer — which is why none of
this depends on the routing layer.

> [design/transport.md](design/transport.md) is the transport: who owns which pages and addresses, why a
> send or a receive does not fail, and which kernel calls can wedge a node (killing a verbs process can need a
> physical power cycle). Read it before writing or running anything that opens a verbs device.

## Links, and knowing when you only have one

A node reaches the world over Wi-Fi/LAN and over the Thunderbolt fabric. It is
tempting to call the first a control plane and the second a data plane. That is
backwards, and the label hides a real dependency:

- **Wi-Fi is opportunistic.** Whether a network exists is a property of the room, not
  of the node. A node cannot host one for the others, so a venue with no Wi-Fi means
  no Wi-Fi, and nothing on the node can change that.
- **The fabric is the dependable one.** Copper, point to point, between two machines
  we own. It needs no infrastructure that anyone else controls.

So the fabric can be a node's *only* link, and control has to ride it when that is the
case — `ssh` and `mesh-peers` both work over `fabric-adjacent` link-local
addresses for exactly this reason. Breaking the fabric is a reachability failure, not
an inconvenience, which is why the transport's stop (design/transport.md) is a reachability
matter.

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

The plane count is the keeper's: the LAN, and a fabric port whose cable link is up with an RDMA peer
(`/var/run/mesh-fabric`, the fabric init's). The keeper logs `DEGRADED` every pass while only one remains, and
`mesh-top` shows that line. Do not run anything that can disrupt the remaining link while degraded.

### The list has to have more than one entry

macOS auto-joins any known network in range, so "find an available network" needs no
code. It needs a list, and a list of one is a venue change away from needing a
keyboard. `networks.conf` (gitignored — it holds live credentials) seeds every node,
`install.sh` copies it to `/usr/local/mesh/networks.conf`, and the keeper re-asserts
it. Entries are only added, never removed. Format is in `networks.conf.example`.

## Adding a machine to the fabric

Steps 1-3 above, then cable it to any free Thunderbolt port on any existing node —
position does not matter and nothing records it. Then verify, from either end:

```
mesh-status                     # rdma_enX PORT_ACTIVE, io.mesh.* all loaded
mesh-peers                      # the new node appears: in its bridge's map, and its announcement
```

What each step proves, in order, so a failure tells you where to look:

| check | passes when | fails if |
|---|---|---|
| `rdma_ctl status` = enabled | the Recovery visit happened | needs a physical 1TR visit; nothing remote fixes it |
| some `rdma_enX` is `PORT_ACTIVE` | a cable is in, both ends RDMA-enabled | cable, or the far end never had `rdma_ctl enable` |
| `ifconfig <port>` has an `fe80::` address | `mesh-fabric-init` ran | check `/usr/local/mesh/log/fabric.log` |
| the peer appears in `mesh-peers` | its bridge's record reached this one, or its beacon announced it | check `~/Library/Logs/io.mesh.bridge.log`, `/usr/local/mesh/log/router.log`; confirm both nodes advertise |

A node that answers `mesh-peers` is administrable and workloadable.
Nothing else is required of it.

## Is anything missing?

```
mesh-peers        # every node in the bridge's map, then every node announcing, as it announces
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

A node that forwards is a router, so it keeps the weak end-system model (RFC 1122 §3.3.4.2):
`net.inet6.ip6.check_interface` and `net.inet6.ip6.select_src_strong_end` are 0. Under macOS's strong
model the ULA identity carried nothing: a ULA destination took a cable's link-local address as its
source. macOS also scopes a datagram's route to the interface holding its source address, so the identity
lives on each fabric port, not `lo0`, where a datagram bound to it found no route along a cable.

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
| `bin/mesh-peers.sh` | the nodes this node's bridge holds a record of, then the nodes announcing on the LAN, as a stream. Installed as `mesh-peers`. |
| `bin/mesh-beacon.sh` | continuous `_meshnode._tcp` announcement. Must never exit voluntarily. |
| `bin/mesh-keeper.sh` | a pass every 60 s (drift has no event to wait on): re-asserts power policy, re-bootstraps sshd/screen sharing/beacon, runs the fabric init (port recovery), writes health warnings when they change, loads each unloaded `io.mesh.*` system daemon (one whose userspace provider runs excepted) and background agent; loads the admin user's own LaunchAgents that declare the Background session type into its background domain (user/<uid>) on every node, so the user's services (the bridge, a rank) run from boot with nobody at the node and never in a console session, where Local Network privacy would hold them. |
| `bin/mesh-health.sh` | the node's health report (`/var/run/mesh-health`): disk against the floor, idle blockers, ports without their RDMA peer. |
| `bin/mesh-devtools-init.sh` | headless Command Line Tools install, so every node can build RDMA code. Idempotent. |
| `bin/mesh-router-init.sh` | identity `/128` on each fabric port, babeld across every fabric port. Resident; launchd restarts it. |
| `vendor/babeld-arm64` | prebuilt Babel daemon, so a node needs no toolchain. Provenance in `vendor/PROVENANCE.md`. |
| `bin/mesh-nodeinfo.sh` | the node's identity, facts and topology, run by the node's resident rank for a `{"facts"}` request (metal-microbench docs/elastic.md#call-delivery). |
| `bin/mesh-agent.sh` | `mesh-agent.sh NEW [DOMAIN [PLIST]]` puts a launchd job's definition NEW in place as PLIST (default NEW without `.new`) and loads the job in DOMAIN (default the user's background domain user/<uid>, which Local Network privacy does not hold) where it is not loaded (`launchctl print`, else `launchctl list` where print refuses a user domain); a definition that does not parse is refused. A loaded job is left running: launchd removes a booted-out service only some time after its process exits, with no event, so replacing a loaded definition races by construction, and nothing does it. A definition never changes with what its job runs (no digest, nothing derived from binaries or configuration), so a changed one is the repo's own edit, in place for the job's next load (login or boot), and said so. **launchd.** Every resident job is KeepAlive with `ThrottleInterval` 1 (launchd starts it again at once after an exit, at most once a second; its default is 10) and picks up a new program or configuration by exiting: the bridge on its binary (design/transport.md); telemetry and the observer on a kqueue watch of their own files and the documents they read at their start (`bin/mesh_restart.py`); the beacon and the router, scripts that exec `dns-sd` and `babeld`, by `install.sh`'s `launchctl kickstart -k` (launchd's own restart of an unchanged definition) where their programs changed; periodic jobs (the keeper, the updates) and the convergence's passes read their programs at each run. The bridge's `ExitTimeOut` is 60, the longest launchd gives a stop at logout or shutdown before its SIGKILL. Nothing boots a job out but the operator's `mesh-converge.sh off`, which removes definitions and lets each loop end. |
| `bin/mesh-converge.sh` | the node's convergence (Convergence, below): with no argument the mesh checkout's pass (`io.mesh.converge`); `held` and `stamp` are the node's convergence for any consumer's layer, `off` takes the node out. |
| `bin/mesh-stat` | the bridge as its own build reads it: runs the `mesh-stat` built beside the admin user's running bridge (`/usr/local/mesh/admin-user`), a file that user owns, so a reader never meets a region of another MESH_VERSION. It runs the reader as the admin user, root's call through `sudo -n -u` that user, never as root; any other caller is refused (`"errno":1`), as is a node with no bridge of that user's (`"errno":3`) or a reader that user does not own (`"errno":2`). |
| `bin/mesh-fabric-init.sh` | tear down the Thunderbolt bridge, address each fabric port, enable IPv6 forwarding; write each port's link and RDMA peer to `/var/run/mesh-fabric`; a port whose cable link is up with no RDMA peer is logged and held, never brought down (design/rubrics.md M3). Idempotent; run at boot and every keeper pass. |
| `bin/mesh-rdma-init.sh` | boot-time fabric record: RDMA enablement and each device's link, as boot found them. Verify-only by necessity. |
| `keys/authorized_keys` | the pubkey roster every node trusts. Public keys only. |
| `templates/` | LaunchDaemon template for workloads. Boot-time, KeepAlive, no login needed. |
| `viz/` | the whole-fabric observer (`io.mesh.observer`, :8787) and its page: the nodes in its bridge's records, each sampled every `period_s` of its configuration document (`etc/observer.json`, installed as the node's where it has none; no default in code). |
| `docs/` | the bring-up writeup. `./serve.sh` to read it locally. |
| `rdma/` substrate | the node's bridge and what it publishes: `mesh.h`, `mesh-verbs.h`, `mesh-link.c`, `mesh-program.c` and `mesh-bridge.h` (the bridge), `mesh-call.c`/`.h`, `mesh-dataflow.c`/`.h`, `mesh-metal.m`/`.h`, `mesh.metal` (the GPU wait, payload read and publication), `mesh-disk.h`, `mesh_disk.py`, `mesh-stat.c`, `mesh.py`, `peers.py`, `store.py`, `mesh_c_build.py`, `mesh-coreml.m`, and torch-mesh's `_stream.mm` (torch's MPS stream as the library's stream) |
| `rdma/` collectives | NCCL's API over the substrate: `nccl.h`, `nccl-mesh.c` (its sessions), `nccl-mesh-metal.m`/`.h`, `mesh-collective.c`/`.h`, `mesh_mpi.py`, `allocate.py`, and torch-mesh's `backend.py`, `partition.py`, `evidence.py`, `engines.py` and its kernels (`streamed.metal`, `attention.metal`, `argmax.metal`, `speculate.metal`, `drafter.metal`) |
| `rdma/mesh-plan.h` | the one header both publish: the substrate's read side (`mesh_observe`, `mesh_map`, `mesh_map_wait`) and the collectives' planning |
| consumers | everything that calls them: metal-microbench, `xonotic/`, `rdma/workload.py` and `xonwire.def` |

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

## Routing

With the bridge gone, each Thunderbolt port is its own point-to-point link. Bonjour
and link-local reach exactly one cable hop, so multicast discovery stops describing
the fabric the moment a third node exists. Anything built on `ff02::1` silently
degrades from "the mesh" to "my neighbours" — a report that still prints a table and
no longer means what it says.

So every node takes a routable identity address and the fabric is routed:

- **Identity**: a `/128` under `fd6d:6573:68::/48` on each fabric port, derived from the node's
  `IOPlatformUUID`, the same on every port, so it does not change when a cable moves to another port (Addressing,
  above, says why not `lo0`).
- **Routing**: `babeld` on every fabric port, each declared a wired link (`type wired`: a cable's fixed cost, no
  wireless loss estimate), using the per-port link-locals as next-hops. It is designed for topologies that churn, reconverges on replug with no
  per-node configuration, and uses every link rather than blocking one the way a
  spanning tree would — which is the whole reason to cable a ring.
- **Enumeration**: the bridges' map names every node on the fabric, `mesh-peers` adds the shared LAN's (above), and
  each node's record lists its routes.

babeld is vendored as a prebuilt arm64 binary — see [vendor/PROVENANCE.md](vendor/PROVENANCE.md)
for the source, hashes, and the one-line build fix. A node must never need a
toolchain: invoking `cc` on a virgin Mac opens the Command Line Tools GUI installer.

RDMA itself is untouched by any of this and still never routes. TN3205 is explicit
that the application forwards across a topology. This layer exists so that nodes can
*find and reach* each other at unlimited fleet size; the RDMA data plane rides the
cables directly, point to point.

## Names, traversal, and running things

`mesh-peers` shows what the fabric is, not just which addresses exist (abridged):

```
node <self> self
node <peer> peer
info <peer> via=bridge node=<n> model=Mac16,11 memory=... window=...
path <peer> kind=lan addr=<a>
path <peer> kind=fabric-adjacent addr=fe80::<b>%<port>
path <self> kind=fabric-routed addr=fd6d:6573:68:<c>
```

A node's own facts (`name=<peer> node_id=... boot_id=... ula=fd6d:6573:68:<d>` and the rest below) come from
`mesh-nodeinfo.sh`, which the node's resident rank runs for a `{"facts"}` request (metal-microbench
docs/elastic.md#call-delivery), as the rank's user, so no root process runs the user's checkouts or bridge reader. Every value is one token (a space in it is written `_`: `chip=Apple_M4_Pro`), so a reader
that splits a line on whitespace reads it whole; `rdma/peers.py` `fields` reads a value to the next `key=` of its line
and takes either form.
Each node reports its name, hardware (its chip and GPU core count among it), macOS, RDMA enablement and SDK state, its
bridge's window (`window`, mesh-stat's own top-level figure: since mesh-stat prints every node's record, the last
`"window"` of its line was another node's, and from 67d50f2 each of the pair published the other's), its admin account,
the maker of its instance evidence (`instances`, `<chip>/<cores>`, written `incomplete:<chip>/<cores>` where the file
lacks its fit (coefficients and bandwidth), its served and attention readings (a fit before torch-mesh 771ae16) or an
attention line of either plan: torch-mesh refuses the first two and cannot price a layer's plan without the third), what each of its
layers installed as that layer's install last recorded it (`revisions` and `images`, `<label>:<revision>` and
`<label>:<digest>`, from `~/.cache/mesh/facts/<label>`, whose other lines it publishes as they are), each layer's last
convergence outcome (`converge`, `<label>:held|staging|failed|converging|converged|off`), the model files it holds
(`models`, `<dir>/<file>:<bytes>` under `~/models`, links followed), its health report's numbers, each fabric
port's line as the fabric init last wrote it (`port=<port> link=<status> rdma_peer=present|absent`), and its routes.
`rdma_peer` is whether the IORegistry holds an `AppleThunderboltRDMAPeerInterface` for the port's RDMA interface (its
`RID`); it is not a verbs port measurement, though on every port read so far it was present exactly when the verbs
port was `PORT_ACTIVE`. A port's address and neighbours are telemetry's (`io.mesh.telemetry`, its fabric section).

The RDMA graph is the bridges' map: each bridge floods its ports' record to every other, so every node holds the
whole graph (`mesh-plan.h` `mesh_map`) — no central registry, no multicast, no hop limit.

Job submission needs no new machinery. Every node already runs sshd and trusts the
operator roster; the only missing piece was knowing which nodes exist, which the bridge's map and `mesh-peers` (the
LAN) answer.

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
node-info           branch=main converged=2026-08-28T00:56:27Z, every node's through its bridge
```

A node that has not converged since a push is the thing worth noticing, and you can see it from any machine.

## Convergence

The root layer above is `io.mesh.update`'s. The user layer (the admin user's mesh checkout, its bridge and
library, and any consumer's layer on top, such as metal-microbench's resident rank) follows its upstreams by itself on
a node given a convergence document (`bin/mesh-user.sh CONFIGURATION CHECKOUT CONVERGENCE`, etc/converge/README.md):
the user's agent `io.mesh.converge` runs `bin/mesh-converge.sh follow`, a pass every `period_s`, the upstream's own copy of the script continuing the loop by `exec` wherever it differs from the running one, and a consumer's own agent does the
same for its layer with that script's subcommands. A layer changes only to images built and checked beforehand:

1. **Held.** A checkout with local work (changes to tracked files, commits its upstream lacks, no upstream) is left as
   it is, nothing fetched.
2. **Fetched**, and the upstream's own copy of the script run, so a broken script is mended by the next push. A layer
   whose upstream and documents are what its install last recorded (`~/.cache/mesh/converge/<label>`) is done.
3. **Staged.** The upstream revision is built in a worktree of its own (`~/.cache/mesh/stage/<repository>`) at
   background priority: for the mesh, `make -C rdma all install`, the bindings and torch-mesh's extension. A failed
   build changes nothing on the node (`failed`, with its log beside the state), and the next pass builds again. A staged
   bridge signed ad hoc where the running one carries the signing identity is not installed, since the Local Network
   grant keyed on that identity would not carry over to it.
4. **Installed.** The checkout is fast-forwarded, each staged image whose bytes differ is installed by rename (the
   libraries first, the bridge's own binary last, and the binary anew wherever libmesh changed, since the bridge
   watches its binary alone), `bin/mesh-user.sh` runs on them (`MESH_USER_BUILT`: the bindings, torch-mesh, the
   configuration and the bridge's agent), and what it installed is recorded.

Nothing stops a bridge but the bridge. Its agent's definition names its program and its configuration's path alone, so
it changes only where the checkout moves; what the bridge runs changes by rename: a new binary or libmesh (the
installation above, or `make all install` by hand, which installs the binary anew whenever libmesh changed) or a new
configuration (`mesh-user.sh` renames it into place and then gives the binary a new file), and `mesh-user.sh` stops
the bridge by SIGTERM, its programs failing as values, so launchd starts the new one (design/transport.md section 9); a
process already running keeps the files it mapped. No loaded definition is replaced (`bin/mesh-agent.sh`).

Each pass's outcome, where it changed, is a line of `~/Library/Logs/io.mesh.converge.log` and node-info's `converge`; what
each layer installed is node-info's `revisions` and `images`. A MESH_VERSION bump parts the map into bridges of the two
versions until every node has converged, and a consumer's process holding the older library fails its calls as values
until its own layer converges, a pass later. Whether a node converges is the operator's choice, one
command each way: `bin/mesh-user.sh CONFIGURATION CHECKOUT CONVERGENCE` (then each consumer's own setup, which writes
its agent where the document is recorded) puts it in; `bin/mesh-converge.sh off` takes it out, removing the recorded
document and every layer's agent definition (`<label>.converge`; each layer's follow ends at its next pass) and
marking each layer's outcome `off`, so neither a later `mesh-user.sh` without CONVERGENCE nor a consumer's setup writes
them again. A `launchctl disable` alone is undone by the next `mesh-user.sh` or setup, whose agent install enables the
label.

## Reading the docs

```
./serve.sh          # http://localhost:8080
```

The same locked mesh runtime serves these docs through `mesh-python`.

## Adding a workload

Copy `templates/io.mesh.job-EXAMPLE.plist` to
`/Library/LaunchDaemons/io.mesh.job-<name>.plist`, point it at your script, then:

```
sudo launchctl bootstrap system /Library/LaunchDaemons/io.mesh.job-<name>.plist
```

Runs as root at boot with no login session. `KeepAlive` restarts it forever.

## Collectives

The mesh's collectives are NCCL's API over the bridges: [rdma/NCCL.md](rdma/NCCL.md) (libnccl-mesh, the C
library; `rdma/mesh_mpi.py` for NumPy), [rdma/torch-mesh/README.md](rdma/torch-mesh/README.md)
(torch.distributed's `mesh` backend) and [design/heterogeneity.md](design/heterogeneity.md) (nodes of
different speeds, links of different costs). A node's bridge (`rdma/mesh-flow`) is its resident service, the admin user's agent
`io.mesh.bridge` run from the user's mesh checkout with the node's bridge configuration, a document its consumer writes and passes
(`bin/mesh-user.sh CONFIGURATION [CHECKOUT [CONVERGENCE]]`; etc/bridge/README.md, with the two didactic examples; no default; design/transport.md; Convergence above). What it owns and does:
design/transport.md. It forwards nothing, runs no command and carries no request.

The transport, every page and word with its writer: [design/transport.md](design/transport.md).
