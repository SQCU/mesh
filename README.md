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

## Addressing

No static IPs. Identity travels with the machine:

- Bonjour `<LocalHostName>.local`
- IPv4 link-local (169.254/16, ARP collision-probed)
- IPv6 link-local (`fe80::`, derived from the interface MAC)

Discover peers on the fabric regardless of which port a cable landed in:

```
ping6 -c3 -I bridge0 ff02::1
```

The Thunderbolt bridge is kept **last** in network service order so a peer can never
advertise itself as the default route and blackhole the node's uplink.

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
| `bin/mesh-rdma-init.sh` | boot-time fabric verification. Verify-only by necessity. |
| `keys/authorized_keys` | the pubkey roster every node trusts. Public keys only. |
| `templates/` | LaunchDaemon template for workloads. Boot-time, KeepAlive, no login needed. |
| `docs/` | the bring-up writeup. `./serve.sh` to read it locally. |

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
