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
> Pin `MESH_REF` to a tag or SHA when you want a reproducible definition.

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
| `bin/mesh-fabric-init.sh` | tear down the Thunderbolt bridge, address each fabric port, enable IPv6 forwarding. Idempotent; run at boot and every keeper pass. |
| `bin/mesh-rdma-init.sh` | boot-time fabric verification. Verify-only by necessity. |
| `keys/authorized_keys` | the pubkey roster every node trusts. Public keys only. |
| `templates/` | LaunchDaemon template for workloads. Boot-time, KeepAlive, no login needed. |
| `docs/` | the bring-up writeup. `./serve.sh` to read it locally. |

## HMI epilogue

Provisioning gives every node the same policy: never sleep, firewall off, no screen
lock. That is correct for a machine in a rack and wrong for one someone is typing on.

`hmi-epilogue.sh` is an operator action run *after* provisioning, on a machine a human
uses:

```
sudo ./hmi-epilogue.sh
MESH_HMI_SLEEP=30 MESH_HMI_DISPLAYSLEEP=10 sudo ./hmi-epilogue.sh
```

It writes `/usr/local/mesh/hmi`, which is the only thing the keeper consults before
re-asserting power policy and firewall state. Everything that governs *reachability*
is untouched and still re-asserted every 60 seconds: sshd, screen sharing, the beacon,
the Thunderbolt fabric, network time, and the RDMA alarm. Undo with
`sudo rm /usr/local/mesh/hmi`.

The marker is deliberately narrow. It cannot make a node unreachable — it can only
let a laptop close its lid and keep its firewall, which is what its operator asked
for. A machine under it still announces itself, still routes, still answers.

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
