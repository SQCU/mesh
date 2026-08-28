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
