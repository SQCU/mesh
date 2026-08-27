# Threat model

This is an availability-first fleet. Read this before changing anything, because
several decisions in `install.sh` look wrong under a conventional host-hardening
model and are correct under this one.

## The asset is participation

What these machines are for is being powered, reachable, and continuously accepting
and returning work over the fabric. That participation is the asset. Everything else
is replaceable.

## The threat is the false negative

**The failure that matters is a node that should be working and reachable, and
silently is not — for any reason, no matter how locally sensible that reason
sounds.**

A machine that has powered down to save energy, gone to sleep because no display was
attached, rebooted itself to install an update, stopped answering because a packet
filter turned itself on, or is sitting at a login window waiting for a keyboard that
will never be plugged in, has been removed from the mesh just as effectively as if
someone had unplugged it. From the outside these are indistinguishable, and all of
them are the same failure.

**Inducing that withdrawal is therefore an attack**, and one we care about far more
than unauthorised access. A node that can be made to eagerly, solipsistically
withdraw itself is a vulnerability.

## What is explicitly not the threat

Confidentiality and integrity of an individual node are **not** primary concerns.

A node running the wrong code, or holding data someone else read, is resolved by
destroying the machine and provisioning another one — the process is in `README.md`
and costs one physical visit. That is a complete and satisfactory answer to that
class of problem. It is *not* an answer to a machine that quietly stops working,
because you may not find out for a long time, and the mesh degrades in the meantime.

Trading availability away to buy confidentiality is, here, a bad trade.

## The dangerous default

**Every one of these machines is initialised in the vulnerable mode.** Stock macOS
is tuned for a laptop on someone's desk: it sleeps, it locks, it waits for consent,
it reboots itself when it judges that wise, and it prefers going quiet to staying up.
Each of those is a withdrawal vector shipped enabled.

Correcting that default is the first and most urgent work on any new machine, and it
is the only thing that justifies a human being physically present.

| Stock default | How it withdraws the node | Countermeasure |
|---|---|---|
| `sleep 1`, display/disk sleep | Sleeps ~1 min after the display is unplugged | `pmset sleep/displaysleep/disksleep 0`, plus a `KeepAlive` caffeinate daemon independent of pmset |
| `autorestart 0` | Stays dark after a power blip | `pmset autorestart 1`, `nvram auto-boot=true`, zero startup delay |
| Auto-install macOS updates | Reboots itself on Apple's schedule | Download-and-notify only; never auto-restart |
| FileVault | Demands a keyboard at every boot | Left off. Encrypt payloads at the application layer instead |
| Remote Login off | Node is unreachable from birth | `launchctl bootstrap` sshd (no TCC grant needed) |
| Application firewall | Can only ever subtract reachability | Disabled, and the watchdog turns it back off if it re-enables |
| Kernel/system freeze | Hangs indefinitely | `systemsetup -setrestartfreeze on` |
| Clock drift | Breaks ssh and TLS silently; reads as a dead node | Network time enforced by the watchdog |
| Screensaver / idle lock | Interferes with unattended remote access | `idleTime 0` |
| Static IP addressing | Collides and partitions when a node is replugged elsewhere | Bonjour + v4/v6 link-local only |

## Absence must be observable

A withdrawn node and a healthy node look identical until someone thinks to check.
That gap is what makes the false negative dangerous, so it is closed explicitly:

- `io.mesh.beacon` advertises `_meshnode._tcp` continuously via Bonjour. It is
  `KeepAlive`, it must never exit voluntarily, and **its silence is the alarm.**
- `mesh-peers` enumerates every node currently announcing on the fabric. Run it from
  any machine. A node missing from that list has withdrawn.
- `io.mesh.keeper` re-asserts every invariant above once a minute and logs any drift
  it had to correct, so slow decay is visible rather than silent.

Do not make the beacon quiet for any reason other than the node actually being gone.
Do not add a "graceful degradation" path that lets a node decide to stop
participating. There is no local condition under which withdrawing is the right call;
that judgement belongs to an operator, not to the node.

## Accepted costs

Stated plainly so nobody re-litigates them later as oversights:

- **Disks are unencrypted at rest.** Deliberate. FileVault is incompatible with
  unattended reboot.
- **No host firewall.** Deliberate. The security boundary is physical access to the
  room.
- **Passwordless sudo for the admin user.** Deliberate. Prompting for a password is a
  withdrawal vector on a machine with no keyboard.
- **Provisioning source is a public repo.** Deliberate, so any node can reach a
  consistent definition with no credentials.
