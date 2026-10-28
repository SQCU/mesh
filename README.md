# mesh

The [collective goal](GOAL.md): the lowest attainable overhead collective operations for
arbitrary programs on any number of Macs in any cabling ([rdma/NCCL.md](rdma/NCCL.md)). The operator, 2026-10-03:
"i cannot and will not precommit to a specific ordering or routing!"

Provisioning for a fabric of Apple Silicon Macs wired together with Thunderbolt and
talking RDMA. The operator, 2026-08-27 ([THREAT-MODEL.md](THREAT-MODEL.md)): "the *absolute* requirement for this
project is that machines may not become unreachable and may never 'choose' to turn off or stop executing rdma chained
intermediates or results for larper reasons".

Nothing here encodes cable position. The operator, 2026-09-25: "absolutely no part of the project's scope or
requirements ever gave off the impression that mesh units are permanent, stationary, stay online forever, or that
anyone is allowed to stop working, whether services or agents working on services, if a meshed compute network is
unplugged, turned off, turned on, retopoed, etc."

The operator, 2026-08-27 (THREAT-MODEL.md): "the most important class of compromise is the false negative: the
computer should be on and receiving work and doing work, but isn't, for any reason, no matter how sensible the
'reason' might sound", and "correcting this default is the first, urgent, and only human-intervention-justifying
priority". Stock macOS sleeps, locks, waits for consent, and reboots itself on an internal policy.

A node stays awake by its power policy (`sleep 0`, `standby 0`, `autopoweroff 0`, `autorestart 1`, set by each run
of `install.sh`), not by a process holding assertions. Until 2026-10-13 `io.mesh.caffeinate` ran `caffeinate -dimsu`,
which kept the display on and declared the user active forever, so `dasd` never found the node idle and never ran
the system's own idle maintenance: the Neural Engine daemon's compiled-model cache, which SIP lets only
`aned`'s garbage collection delete (`com.apple.aned.danglingModelsGC`, `com.apple.aned.modelCacheGC`, refused 7,750
times in two weeks on "Device Activity Policy"), grew to about 390 GB of a 460 GB disk. A full disk is the false
negative this fleet exists to prevent; the display sleeps after 10 minutes so the node can be idle by macOS's
definition while it keeps computing.

`bin/mesh-health.sh`, run by `install.sh` (or by hand, as root), writes `/var/run/mesh-health`: disk free, used and floor, and every
process holding a user-activity or display assertion now, with how long it has held it (what keeps `dasd` from finding
the node idle). Below twice the floor those holders become a warning (an attended machine may keep its display on for
good, and starved housekeeping only matters once space runs short), and the report also measures what `du` can read
against what is used (the rest is space only the OS accounts for) and the largest directories under /Users, once each
time the disk enters that state. Each condition worth attention is a `warn=` line. `mesh-status` prints the report, and
node-info carries its numbers and `health_warnings`; the report is as of its `health_updated` line, since nothing
writes it on a schedule.

## The one step that cannot be automated

`rdma_ctl enable` only runs from Recovery OS:

```
$ sudo rdma_ctl enable
rdma_ctl: This tool needs to be executed from Recovery OS.    exit=77
```

Entering 1TR requires physically holding the power button, which is a deliberate
Apple anti-tampering property — no MDM, script, or remote console can do it. It is
the one step of a node's provisioning that takes a physical visit; every other step
runs remotely.

## Bringing up a new node

Hands-on, once, ~5 minutes:

1. **Setup Assistant** creates the admin account, with FileVault off (with it on,
   every boot waits at a keyboard) and Activation Lock disabled.
2. **Recovery**: power held → Terminal → `rdma_ctl enable` → reboot. This writes
   `rdma-enable=1` to NVRAM, which then persists.
3. **Bootstrap**, in Terminal:

   ```
   curl -fsSL https://raw.githubusercontent.com/SQCU/mesh/main/bootstrap.sh | sudo bash
   ```

   This opens sshd *first*, then fetches and runs `install.sh`; port 22 answers
   from then on.

Everything after that is remote: `ssh node.local` and a rerun of `sudo install.sh`.

The two gitignored credential files, `networks.conf` (every known Wi-Fi network) and `tailscale.key` (a reusable,
pre-approved Tailscale auth key, not ephemeral), are not in the fetched tarball. They reach a node from a machine that
holds them, once:

```
for f in networks.conf tailscale.key; do ssh node.local "sudo sh -c 'umask 077; cat > /usr/local/mesh/$f'" < $f; done
```

Then `ssh node.local 'sudo /usr/local/mesh/bin/mesh-networks.sh; sudo /usr/local/mesh/bin/mesh-tailscale-init.sh'` (or a
rerun of `install.sh`) adds the networks and joins the tailnet labeled with the node's hardware serial number, not a
host name (the operator, 2026-10-21: "we never rely on something like a localhostname as a name or identifier for
something which we are programming"). A node's key expires unless its expiry is disabled in the Tailscale admin console
or it joined with a tagged key, whose nodes do not expire; a node whose key expired logs in again.

**Gitignored documents.** Git carries everything a node converges to except two kinds of document, which reach a node
over ssh from a peer that holds them, when the node joins the mesh and to every node again when one changes (a network
added, a key replaced, a standup recorded):
- credentials (`networks.conf`, `tailscale.key`), because the repository is public;
- `NODE-STANDUPS.md` at the checkout's root, the record of which macOS and toolchains each node had and what went
  wrong at its standup (`ssh <node> 'cat > mesh/NODE-STANDUPS.md' < NODE-STANDUPS.md`). The operator, 2026-10-22: it
  is gitignored for "preventing agents from treating the happenstance of os versions as permanent or part of the
  repo's versioning".

Build products are gitignored too; each node builds its own.

**Nodes of your own.** The roster every node trusts is `keys/authorized_keys` (public keys, each labeled
`owner:machine`). A fork of this repository with its owner's public key appended bootstraps its owner's nodes:

```
curl -fsSL https://raw.githubusercontent.com/<you>/mesh/main/bootstrap.sh | sudo MESH_REPO=<you>/mesh bash
```

A node remembers the repository and branch it was bootstrapped from (`/usr/local/mesh/repo`, `/usr/local/mesh/branch`)
and converges from them when someone reruns its bootstrap (below); the bridge and the library are built by hand
(`rdma/NCCL.md` "Build and run").

`bootstrap.sh` opens sshd *before* it fetches anything, so a node whose later steps
fail is still reachable. `MESH_BRANCH` selects a branch other than `main` (Which
definition a node is running, below).

## What provisioning does that looks wrong

The repository's reasons, with no operator quote behind them ([THREAT-MODEL.md](THREAT-MODEL.md)
has the operator's words):

- **FileVault stays off.** It demands a keyboard at every boot, so a node under it
  cannot reboot unattended.
- **The application firewall is off**, set by each run of `install.sh`. It can only
  subtract reachability.
- **Passwordless sudo** via `/etc/sudoers.d/mesh`, `visudo -c` validated before it is
  written. A password prompt on a keyboard-less node blocks both recovery and any
  future agent that provisions a machine and then has to administer it.
- **The provisioning source is a public repo**, so a node reaches a consistent
  definition with no credentials.

## Traversal is a stream, and a node converges when its installer runs

`mesh-peers` emits facts as they arrive and does not wait to prove a negative; it runs until it is stopped:

```
node <n> self|peer [name=<name>]    each node this node's bridge holds a record of (its map), at once; then each node
                                    that announces _meshnode._tcp (Bonjour) on the shared LAN, as it announces
info <n> via=bridge model=... memory=... region=...                its bridge's record of the node
info <n> via=mdns node=<n> node_id=<uuid> serial=... model=... rdma=...   the node's announcement, as it is resolved
path <n> kind=<kind> addr=<a>       each address the node advertises: lan, lan-v6, fabric-adjacent (a link-local
                                    on one of this node's RDMA ports), fabric-routed (its fd6d:6573:68:: identity),
                                    fabric-v4ll
gone <n>                            an announcement withdrawn
```

`<n>` is the node's identity, its bridge node number (FNV-1 of its host UUID, `rdma/peers.py` `node_number`): a
bridge record and an announcement are the same node when their numbers are, and `self` is this machine's own number.
`name=` is the name table's (`etc/mesh-nodes.json`, entries keyed by host UUID, `rdma/peers.py` `table`), absent for a
node it does not list; a name is a label for people, which nothing matches to find a node (AGENTS.md at 513638a4, "Node
identity and names"). The beacon announces the host UUID as its instance name and its number, UUID and serial in its TXT
record; a node still running an older beacon (one that announced its LocalHostName) shows as
`unidentified-<key>` until `install.sh` next runs there. The SRV target `dns-sd -L` resolves is used only as
mDNS's pointer to the node's address records.

The fabric's nodes are the bridge's map, so a node cabled to this one is known along its cable (the operator,
2026-10-16: "you should never be using networking over wifi etc when it is physically and literally tethered / cabled
to a host machine"); its facts are node-info's, which its resident rank answers (metal-microbench
docs/elastic.md#call-delivery). Discovery on the shared LAN is its mDNS (git 1587656: control
traffic needs no fabric), so it is not limited to one cable hop; a node on no shared LAN is found only over its own
cables. Every line arrives when its event does: a slow node delays only its own lines, and a node that is not there
is simply absent.

**A node converges when someone runs its installer, and at no other time.** Nothing fetches, installs, builds or
restarts on a schedule: the operator, 2026-10-21, "this is a genre of code where automatically updating and building
code can bootloop a system indefinitely". `io.mesh.update` (a fetch and rerun of `install.sh` every 15 minutes) and
`io.mesh.keeper` (a pass every 60 s) are gone, and `install.sh` deletes their definitions where they remain. A push
reaches a node when someone reaches the node and reruns its bootstrap, which takes the recorded branch's newest commit:

```
R=$(cat /usr/local/mesh/repo) B=$(cat /usr/local/mesh/branch); curl -fsSL https://raw.githubusercontent.com/$R/$B/bootstrap.sh | sudo MESH_REPO=$R MESH_BRANCH=$B bash
```

Each run installs the files that changed and loads each daemon that is not loaded; a loaded daemon is left running,
and the resident ones take a changed program by `install.sh`'s restart of them (launchd, below). Drift (power policy,
the firewall, sshd and screen sharing, the fabric's ports, the known networks, the tailnet, an unloaded job) is
corrected by the next run. The admin user is the one `/usr/local/mesh/admin-user` records (`MESH_USER` changes it).

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

The following is an earlier single-cable measurement, not a description of the current cluster topology or its
aggregate bandwidth. Communicators discover the physical multigraph and retain parallel cables
([rdma/NCCL.md](rdma/NCCL.md#communicators)).

Verified between `Mac16,11` and `Mac17,6` over one Thunderbolt 5 cable with
`ibv_uc_pingpong`, GID index 0 (the EUI-64 link-local this repo assigns):

| message | rx-depth | frames | throughput | latency |
|---|---|---|---|---|
| 4 KB | 100 | 100 | 6.4 Gbit/s | 10.2 us |
| 64 KB | 60 | 960 | 47.1 Gbit/s | 22.3 us |
| 256 KB | 15 | 960 | 64.5 Gbit/s | 65.0 us |
| 1 MB | 3 | 768 | 70.9 Gbit/s | 237 us |

**Queues are counted in 4 KB frames, not in messages.** A queue holds under 4095
frames (`size/4096 * rx_depth`). The first attempt here asked for 64 KB messages at the default
rx-depth of 500 — 8000 frames — and failed as `Failed to modify QP to RTR` on one side
and `Operation not permitted` on the other, which reads like a permissions or fabric
problem and is neither.

Other limits, all from TN3205: send/receive only (`IBV_WR_SEND`,
`IBV_ACCESS_LOCAL_WRITE`), 10 UC queue pairs per device, ~16 MB maximum message, the
**same frame count** posted by sender and receiver, no hardware ACK, and a
per-controller IOMMU so a buffer is registered separately per device.

The queue pairs are not the same everywhere: the M5 Ultra on macOS 27 reports `max_qp` 3
on every device, not 11, and its driver keeps a queue pair of each device for itself. A
link's two bridges state what their devices hold and make the lanes both can, ten where
both hold ten and one stream's two where either holds two
([design/HETEROGENEOUS-LINKS.md](design/HETEROGENEOUS-LINKS.md)).

The verbs transfer is point-to-point per cable, and its out-of-band GID/QPN exchange needs a socket to the directly
connected peer. Multihop data movement is compiled by the collective layer into these physical transfers; relays
forward their landed pages. This data route is separate from the IP routing layer.

[design/transport.md](design/transport.md) is the transport: who owns which pages and addresses, why a
send or a receive does not fail, and which kernel calls can wedge a node (killing a verbs process can need a
physical power cycle).

## Links, and knowing when you only have one

A node reaches the world over Wi-Fi/LAN and over the Thunderbolt fabric. It is
tempting to call the first a control plane and the second a data plane. That is
backwards, and the label hides a real dependency:

- **Wi-Fi is opportunistic.** Whether a network exists is a property of the room, not
  of the node. A node cannot host one for the others, so a venue with no Wi-Fi means
  no Wi-Fi, and nothing on the node can change that.
- **The fabric is the dependable one.** Copper, point to point, between two machines
  we own. It needs no infrastructure that anyone else controls.

So the fabric can be a node's *only* link, and control rides it when that is the
case — `ssh` and `mesh-peers` both work over `fabric-adjacent` link-local
addresses for exactly this reason. Breaking the fabric is a reachability failure, not
an inconvenience, which is why the transport's stop (design/transport.md) is a reachability
matter.

A node comes back with every link the room offers, and `mesh-status` reports all of
them, so a drop to one is visible rather than discovered later:

```
LINKS
  wifi/lan:  up via en1
  fabric:    up
  redundant: 2 of 2
```

The plane count is `mesh-status`'s: the LAN, and a fabric port whose cable link is up (each port's RDMA peer is in
its RDMA section, from `/var/run/mesh-fabric`, the fabric init's). It prints `DEGRADED` while only one remains.

### The list of known networks

macOS auto-joins any known network in range, so "find an available network" needs no
code. It needs a list, and a list of one is a venue change away from needing a
keyboard. `networks.conf` (gitignored — it holds live credentials) seeds every node,
`install.sh` copies it to `/usr/local/mesh/networks.conf` and seeds it
(`mesh-networks.sh`) at each run. `mesh-networks.sh` adds entries and removes none. Format is in `networks.conf.example`.

## Adding a machine to the fabric

Steps 1-3 above, and a cable to any free Thunderbolt port on any existing node;
position does not matter and nothing records it. From either end:

```
mesh-status                     # rdma_enX PORT_ACTIVE, io.mesh.* all loaded
mesh-peers                      # the new node appears: in its bridge's map, and its announcement
```

What each reading shows, in order:

| reading | shows | otherwise |
|---|---|---|
| `rdma_ctl status` = enabled | the Recovery visit happened | a physical 1TR visit; nothing remote enables it |
| some `rdma_enX` is `PORT_ACTIVE` | a cable is in, both ends RDMA-enabled | the cable, or the far end never had `rdma_ctl enable` |
| `ifconfig <port>` has an `fe80::` address | `mesh-fabric-init` ran | its log is `/usr/local/mesh/log/fabric.log` |
| the peer appears in `mesh-peers` | its bridge's record reached this one, or its beacon announced it | the logs are `~/Library/Logs/io.mesh.bridge.log` and `/usr/local/mesh/log/router.log` |

A node that answers `mesh-peers` is administrable and workloadable.

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
destroys it on every node, every boot, and every run of `install.sh`. Two independent
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
else. `mesh-fabric-init` destroys the bridge interface itself.

[tn]: https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt

## Addressing

No static IPs. Identity travels with the machine:

- the host UUID (`IOPlatformUUID`), from which the bridge's node id and the fabric identity address below are drawn
- IPv6 link-local on each fabric port, EUI-64 from that port's own MAC

A Bonjour `<LocalHostName>.local` name is not identity: macOS renames a running machine when another claims its name
(AGENTS.md at 513638a4, "Node identity and names").

That link-local is *exactly* the value the paired RDMA device reports as one of its GIDs (`GID[0]` on macOS 26;
on macOS 27 the M5 Ultra's devices list another address first, `23ff:…`, then the link-local at `GID[1]`), so a port's
IP address and its RDMA identity are the same number. The bridge chooses the GID that is the link-local
(`rdma/mesh-link.c` `device_gid`), not an index, and the address encodes node identity rather than cable position — it
survives being unplugged and replugged into a different port on a different machine.

macOS does not autoconfigure link-local on an interface with no network service, so
`mesh-fabric-init` assigns it. It deliberately does **not** create SystemConfiguration
services for these ports: those writes need Full Disk Access and fail headlessly with
`Unable to access the System Configuration database`.

`net.inet6.ip6.forwarding` is on, because in a ring a node transits for its
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
`io.mesh.fabric` and at each run of `install.sh`; configd's rebuild of `bridge0` after a
hotplug stands until one of them runs again. A silently re-bridged node is one that
will storm the moment the fabric grows a ring.

The Thunderbolt bridge service is kept **last** in network service order for as long
as it exists, so a peer can never advertise itself as the default route and blackhole
the node's uplink.

## Layout

| path | what |
|---|---|
| `install.sh` | provision/converge a node, run by hand. Idempotent — re-running is the drift fix, and nothing reruns it. Each run sets the power policy, the firewall, sshd, screen sharing and NTP, runs the networks, fabric and tailnet inits and the health report, loads every unloaded `io.mesh.*` system daemon (one whose userspace provider runs excepted) and the admin user's LaunchAgents that declare the Background session type (into user/<uid>), and starts the beacon, telemetry and the observer where they are not running. It deletes `io.mesh.update`, `io.mesh.keeper`, `io.mesh.cartlane.learner` and `io.mesh.cartlane.viewer` where they remain, the update's bootout last since that job may be the run's own parent. It runs with root's home: macOS `sudo` keeps the caller's `HOME`, so the first hand-run install on the M5 Ultra Studio (2026-10-21) left uv's cache, Pythons and receipt root-owned in the admin's home, and the admin's own build (`mesh-user.sh`) could not open them. |
| `bootstrap.sh` | one-shot entry for the physical visit. Opens sshd, then runs `install.sh`. |
| `enable-autologin.sh` | boot into a GUI session so session-bound workloads survive a reboot. |
| `CLAUDE.md` | where to start: `design/rubric-actionable.md`, then `THREAT-MODEL.md` and `INDEX.md`. |
| `INDEX.md` | the documents that describe the current system, each with a short note. |
| `THREAT-MODEL.md` | the operator's words on access, verbatim (2026-08-27). |
| `bin/mesh-status.sh` | one-screen health report. Installed as `mesh-status`. |
| `bin/mesh-peers.sh` | the nodes this node's bridge holds a record of, then the nodes announcing on the LAN, as a stream. Installed as `mesh-peers`. |
| `bin/mesh-tailscale-init.sh` | the node's own tailnet edge (design/ACCESS-TOPOLOGY-SPEC.md: a stationary node is not carried by a laptop's tailnet). Builds the open-source `tailscale` and `tailscaled` at a pinned version with a pinned Go toolchain (fetched, used and deleted with its caches; Tailscale ships no CLI-only macOS binary, and the App Store and standalone apps need a GUI approval), runs `tailscaled` as `io.mesh.tailscale` (state in `/usr/local/mesh/tailscale`), and brings it up from `/usr/local/mesh/tailscale.key` where it is not running; with no key it logs the login URL to `tailscale.log`. Run by `install.sh`; it builds only when the pinned version changes and only with 4 GiB free above the disk floor. A Mac with `/Applications/Tailscale.app` already has its edge from the app, and a second client would contend for the tailnet's routes and DNS, so there it does nothing and `mesh-status` names the app. |
| `bin/mesh-rdma.sh` | the node's RDMA devices, one line each: `<port> device=<rdma_port> rid=<RID> rdma_peer=present\|absent`. Every script that names fabric ports reads them here. It reads the IORegistry's archive form (`ioreg -a`, a property list, through `plutil`) and keys each device by its registry entry ID, because a tree walk lists an object once for each path to it: on macOS 27 (the M5 Ultra Studio, 2026-10-21) each `AppleThunderboltRDMAInterface` is listed twice, so the earlier text walk named every port twice and its RID lookup matched no `AppleThunderboltRDMAPeerInterface`, reporting `rdma_peer=absent` on a cabled port. |
| `bin/mesh-beacon.sh` | continuous `_meshnode._tcp` announcement, its instance name the host UUID and its TXT record the node's number, UUID, serial, model and RDMA state. It execs `dns-sd -R`, which runs until it is stopped. |
| `etc/mesh-nodes.json` | the name table and the administrator's one configuration (`rdma/mesh_nodes.py` `load`, a `Cluster`): how to reach a node, not who is a member (the live fabric is): each node's name for people, keyed by its host UUID (`rdma/peers.py` `table` and `label`; `node <number>` for a node it does not list), its bridge node number, administration and fabric addresses, `roles`, `memory_gib` and the rank guard's floor `guard_headroom_gib` (memorystatus's numerator: free, speculative and pageable memory, less the compressor's on a `workstation`); the administrator's ssh `access` (`key`, `user`; absent, the environment's), the hourly's `run_seconds`, the nodes' `python`, their per-commit `checkouts` under home and how many are kept (`checkouts_kept`), the first `rank_port` and how long a bridge swap waits for attached programs to detach (`detach_seconds`). A workload's `place` names a role (default `compute`) and least values; its ranks are the `compute` nodes that meet it, the `root` first, the rest by bridge node number. Names are read by `mesh-peers`, node-info, `mesh-top`, the observer and telemetry; by nothing that finds a node. |
| `bin/mesh-health.sh` | the node's health report (`/var/run/mesh-health`): disk against the floor, idle blockers, ports without their RDMA peer. |
| `bin/mesh-devtools-init.sh` | headless Command Line Tools install, so every node can build RDMA code, and every `/Applications/Xcode*.app` readied for the library's kernels (license, first launch, Metal toolchain). Idempotent. |
| `bin/mesh-router-init.sh` | identity `/128` on each fabric port, babeld across every fabric port. Resident; launchd restarts it. |
| `vendor/babeld-arm64` | prebuilt Babel daemon, so a node needs no toolchain. Provenance in `vendor/PROVENANCE.md`. |
| `bin/mesh-nodeinfo.sh` | the node's identity, facts and topology, run by the node's resident rank for a `{"facts"}` request (metal-microbench docs/elastic.md#call-delivery). |
| `bin/mesh-launchd-agent.sh` | `mesh-launchd-agent.sh NEW [DOMAIN [PLIST]]` puts a launchd job's definition NEW in place as PLIST (default NEW without `.new`) and loads the job in DOMAIN (default the user's background domain user/<uid>, which Local Network privacy does not hold) where it is not loaded (`launchctl print`, else `launchctl list` where print refuses a user domain); a definition that does not parse is refused. A loaded job is left running: launchd removes a booted-out service only some time after its process exits, with no event, so replacing a loaded definition races by construction, and nothing does it. A definition never changes with what its job runs (no digest, nothing derived from binaries or configuration), so a changed one is the repo's own edit, in place for the job's next load (login or boot), and said so. **launchd.** No job runs on a schedule. The router and the tailnet's `tailscaled` are KeepAlive with `ThrottleInterval` 1 (launchd starts one again at once after an exit, at most once a second; its default is 10); the beacon, telemetry and the observer run from their load (`RunAtLoad`), once a boot, and launchd does not start one again after an exit (`install.sh` starts one that is not running); the bridge's is its own: its job runs its heartbeat (`ThrottleInterval` 60,
KeepAlive {SuccessfulExit: false}), which restarts the bridge, and itself, on a deploy's SIGHUP (design/transport.md
section 2). The beacon, the router, telemetry and the observer take a new program by `install.sh`'s `launchctl kickstart -k` (launchd's own restart of an unchanged definition) where their programs, or the documents telemetry and the observer read at their start, changed (`install-user.sh`'s for a userspace provider); an edit to a node's `observer.json` takes effect at the observer's next start (`sudo launchctl kickstart -k system/io.mesh.observer`). The bridge's `ExitTimeOut` is 60, the longest launchd gives a stop at logout or shutdown before its SIGKILL; the heartbeat
ends when its bridge has ended after its SIGTERM, and expires the region (ASKED_END): the bridge's stop has no wait, and
a bridge whose loop has stopped turning is ended by the heartbeat's watch (design/transport.md section 16); a bridge the
watch cannot end (a thread inside a kernel call) leaves the heartbeat to launchd's SIGKILL at the `ExitTimeOut`. |
| `bin/mesh-stat` | the bridge as its own build reads it: runs the `mesh-stat` built beside the admin user's running bridge (`/usr/local/mesh/admin-user`), a file that user owns, so a reader never meets a region of another layout (MESH_LAYOUT). It runs the reader as the admin user, root's call through `sudo -n -u` that user, never as root; any other caller is refused (`"errno":1`), as is a node with no bridge of that user's (`"errno":3`) or a reader that user does not own (`"errno":2`). |
| `bin/mesh-fabric-init.sh` | tear down the Thunderbolt bridge, address each fabric port, enable IPv6 forwarding; write each port's link and RDMA peer to `/var/run/mesh-fabric`; a port whose cable link is up with no RDMA peer is logged and held, never brought down (design/rubrics.md M3). Idempotent; run at boot (`io.mesh.fabric`) and by each run of `install.sh`. |
| `bin/mesh-rdma-init.sh` | boot-time fabric record: RDMA enablement and each device's link, as boot found them. Verify-only by necessity. |
| `keys/authorized_keys` | the pubkey roster every node trusts. Public keys only. |
| `viz/` | the whole-fabric observer (`io.mesh.observer`, :8787) and its page: the nodes in its bridge's records, each sampled every `period_s` of its configuration document (`etc/observer.json`, installed as the node's where it has none; no default in code). |
| `docs/` | the bring-up writeup. `./serve.sh` to read it locally. |
| `rdma/` substrate | the node's bridge and what it publishes: `mesh.h`, `mesh-link.c`, `mesh-observe.h` and `mesh-bridge.h` (the bridge), `mesh-heartbeat.c` (its heartbeat), `mesh-dataflow.c` and `mesh-call.h`, `mesh.metal` (the GPU wait, payload read and publication), `mesh-disk.h`, `mesh_disk.py`, `mesh-stat.c`, `mesh.py`, `peers.py`, `store.py`, `mesh_c_build.py`, `mesh-coreml.m`, and torch-mesh's `_stream.mm` (torch's MPS stream as the library's stream) |
| `rdma/` collectives | NCCL's API over the substrate: `nccl.h`, `nccl-mesh.c` (its communicators and groups), `nccl-mesh-metal.m`/`.h`, `nccl-mesh-kernels.h`/`.m` (its kernels), `mesh-collective.c`, `mesh_mpi.py`, `allocate.py`, and torch-mesh's `backend.py`, `partition.py`, `evidence.py`, `engines.py`, `attention.py` and its kernels (`streamed.metal`, `attention.metal`, `argmax.metal`, `speculate.metal`, `drafter.metal`) |
| `rdma/mesh-plan.h` | the one header both publish: the substrate's read side (`mesh_reached`, `mesh_region`, `mesh_map`, `mesh_nodes_health`, defined in `mesh-dataflow.c`) and the collectives' planning |
| consumers | everything that calls them: metal-microbench, `xonotic/`, `rdma/workload.py` and `xonwire.def` |

## The toolchain, installed headlessly

`infiniband/verbs.h` and `librdma.tbd`, which TN3205's RDMA programs build against,
ship only in the Command Line Tools. So `mesh-devtools-init` installs CLT on every
node, not just ones that compile the router.

CLT installs headlessly. The trap is only the *implicit* path — invoking bare `cc`
with no toolchain opens the GUI installer. The explicit path never does:

```
sudo touch /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
softwareupdate -l          # now lists "Command Line Tools for Xcode <version>"
sudo softwareupdate -i "Command Line Tools for Xcode <version>" --verbose
sudo rm /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
```

The marker file is load-bearing, not folklore: without it `softwareupdate -l` lists
**zero** Command Line Tools labels; with it, two. `mesh-devtools-init` does this and
is idempotent — it exits immediately when `verbs.h` is already resolvable.

The Metal compiler is not in the Command Line Tools: it is an Xcode component, and
`softwareupdate` offers no Xcode. Every node has one, so that any node builds the
library's kernels itself (`rdma/Makefile`, `kernels`), copied once per node from a
machine that has one, over the tailnet:

```
ditto -c --keepParent /Applications/Xcode-<version>.app - | ssh <node> 'sudo ditto -x - /Applications'
```

`mesh-devtools-init` then readies every `/Applications/Xcode*.app` headlessly
(`xcodebuild -license accept`, `-runFirstLaunch`, `-downloadComponent MetalToolchain`,
`xcrun --kill-cache` for root and the admin user), without changing the active
developer directory, which stays the Command Line Tools. Which Xcode, Metal toolchain
and macOS each node has, and what broke while setting them up, is in `NODE-STANDUPS.md`
(gitignored; "Gitignored documents go to peers directly", above).

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

babeld is vendored as a prebuilt arm64 binary ([vendor/PROVENANCE.md](vendor/PROVENANCE.md):
the source, hashes, and the one-line build fix), so the router runs on a node that has
no toolchain yet: invoking `cc` on a virgin Mac opens the Command Line Tools GUI installer.

RDMA itself is untouched by any of this and still never routes. TN3205 is explicit
that the application forwards across a topology. This layer exists so that nodes can
*find and reach* each other at unlimited fleet size; the RDMA data plane rides the
cables directly, point to point.

## Names, traversal, and running things

`mesh-peers` shows what the fabric is, not just which addresses exist (abridged):

```
node 1980260065 self name=Ms-MacBook-Pro
node 1974797816 peer name=Ms-Mac-mini
info 1974797816 via=bridge model=Mac16,11 memory=... region=...
path 1974797816 kind=lan addr=<a>
path 1974797816 kind=fabric-adjacent addr=fe80::<b>%<port>
path 1980260065 kind=fabric-routed addr=fd6d:6573:68:<c>
```

A node's own facts (`node_id=<host UUID> serial=... name=<the name table's> boot_id=... ula=fd6d:6573:68:<d>` and the rest below) come from
`mesh-nodeinfo.sh`, which the node's resident rank runs for a `{"facts"}` request (metal-microbench
docs/elastic.md#call-delivery), as the rank's user, so no root process runs the user's checkouts or bridge reader. Every value is one token (a space in it is written `_`: `chip=Apple_M4_Pro`), so a reader
that splits a line on whitespace reads it whole; `rdma/peers.py` `fields` reads a value to the next `key=` of its line
and takes either form.
Each node reports its name, hardware (its chip and GPU core count among it), macOS, RDMA enablement and SDK state, its
bridge's region (`region`, mesh-stat's own top-level figure: since mesh-stat prints every node's record, the last
`"window"` (now `"region"`) of its line was another node's, and from 67d50f2 each of the pair published the other's), its admin account,
the maker of its instance evidence (`instances`, `<chip>/<cores>`, written `incomplete:<chip>/<cores>` where the file
lacks its fit (coefficients and bandwidth), its served and attention readings (a fit before torch-mesh 771ae16) or an
attention line of either plan: torch-mesh refuses the first two and cannot price a layer's plan without the third), what each of its
layers installed as that layer's install last recorded it (`revisions` and `images`, `<label>:<revision>` and
`<label>:<digest>`, from `~/.cache/mesh/facts/<label>`, whose other lines it publishes as they are), the model files it holds
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

In a `/Library/LaunchDaemons` plist with `inetdCompatibility`, `StandardOutPath`
overrides the socket launchd dups onto stdout, and the service answers every
connection with silence while looking perfectly healthy.

## Which definition a node is running

Fetching is by branch: `main`, or the branch `MESH_BRANCH` names. The operator,
2026-08-27 ([THREAT-MODEL.md](THREAT-MODEL.md)): "you can *never* index by commit *ever*
unless you register that commit as a branch". A node pinned at a commit would fetch its
bootstrap at that commit too, so no rerun would bring it the change that unpins it.

Branch-addressed content on `raw.githubusercontent.com` carries `cache-control:
max-age=300`, so a node can briefly fetch a copy up to five minutes old. That is a
wait, not a fault: it clears itself, and a node that needs the newest definition
sooner can simply be reached and re-run.

Each node records the branch its last `install.sh` run installed and when that run ended, in
`/usr/local/mesh/revision` (install.sh:218), and prints both under the name `converged`:

```
mesh-status         branch main converged 2026-08-28T00:56:27Z
node-info           branch=main converged=2026-08-28T00:56:27Z
```

`mesh-status` reads the node's own file (bin/mesh-status.sh:4). node-info's lines are `bin/mesh-nodeinfo.sh`'s
(lines 28-29), which the node's resident rank runs for a `{"facts"}` request (metal-microbench
docs/elastic.md#call-delivery); the bridge carries no request. Whether a node was installed since a push reads from
its own `mesh-status` or from its rank's node-info.

## A new bridge on a node

Nothing stops a bridge but its heartbeat, on a signal from outside (design/transport.md sections 2 and 9). Its launchd agent's definition names its program and its configuration's path alone, so
it changes only where the checkout moves; what the bridge runs changes by rename: a new binary or libmesh (`make all
install` by hand, which installs the binary anew whenever libmesh changed) or a new configuration (`mesh-user.sh`
renames it into place and then gives the binary a new file), and `mesh-user.sh` sends the bridge's heartbeat SIGHUP,
which restarts the bridge keeping its region and runs its own new image (design/transport.md section 2); a process
already running keeps the files it mapped. No loaded definition is replaced (`bin/mesh-launchd-agent.sh`). What each layer
installed is node-info's `revisions` and `images`. A MESH_VERSION bump parts nothing: no version enters the wire's
magic and no reader tests a message's `version`, so bridges of any version pair (design/transport.md section 14). A
MESH_LAYOUT change expires the old region, and a consumer's process holding the older library fails its calls as values
until it is started again on the new one.

## Reading the docs

```
./serve.sh          # http://localhost:8080
```

The same locked mesh runtime serves these docs through `mesh-python`.

## Collectives

A program that uses the mesh, step by step, with what the bridge, the collective client and your own service each do:
[docs/writing-a-service.md](docs/writing-a-service.md) (its program: `examples/chain-service.py`).

The mesh's collectives are NCCL's API over the bridges: [rdma/NCCL.md](rdma/NCCL.md) (libnccl-mesh, the C
library; `rdma/mesh_mpi.py` for NumPy), [rdma/torch-mesh/README.md](rdma/torch-mesh/README.md)
(torch.distributed's `mesh` backend) and [design/heterogeneity.md](design/heterogeneity.md) (nodes of
different speeds, links of different costs). A node's bridge (`rdma/mesh-flow`) is its resident service, the admin user's launchd agent
`io.mesh.bridge` run from the user's mesh checkout with the node's bridge configuration, a document its consumer writes and passes
(`bin/mesh-user.sh CONFIGURATION [CHECKOUT]`; etc/bridge/README.md, with the two didactic examples; no default; design/transport.md; A new bridge on a node, above). What it owns and does:
design/transport.md. It runs no command and carries no request.

The transport, every page and word with its writer: [design/transport.md](design/transport.md).
