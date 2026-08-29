# Working on RDMA machines

Written after wedging a node badly enough that it needed a physical power cycle.
None of this is theoretical; every line below was observed on `Ms-Mac-mini`.

## The failure

Five `mesh-allreduce` processes were `SIGKILL`ed during a size sweep. They did not
die. `ps` showed them in state `U` — uninterruptible kernel sleep, blocked inside a
verbs call:

```
PID   STAT COMMAND
55027 U    /tmp/mesh-allreduce -d rdma_en4 ...
55143 U    ibv_devinfo -d rdma_en4
```

`SIGKILL` cannot remove a process in that state. Each one held its protection
domain, queue pairs and device context forever. Then, in order:

1. `ibv_alloc_pd` began failing for *every* process, including Apple's own
   `ibv_uc_pingpong`.
2. All three Thunderbolt ports went to `PORT_DOWN` / "No device connected", with
   the cable still physically attached.
3. `ibv_devinfo` itself hung.
4. `shutdown -r now` never completed, because shutdown waits on those processes.
5. Recovery required someone physically power-cycling the machine.

The device has `max_qp: 11` and `max_mr: 100`. It does not take many leaks.

## Rules

- **Never `SIGKILL` a process that has opened a verbs device.** It may be
  unkillable, and the leak is permanent until the machine is power-cycled.
- **Every verbs program installs a signal handler** that runs `ibv_destroy_qp`,
  `ibv_dereg_mr`, `ibv_dealloc_pd`, `ibv_close_device`, then exits.
- **Never poll a completion queue unbounded.** `while(ibv_poll_cq(...)<1);` with a
  peer that never sends is how a process gets stuck where signals cannot reach it.
  Always bound the wait and exit through the cleanup path.
- **Check the port before allocating anything.** `ibv_query_port` and refuse on
  anything but `IBV_PORT_ACTIVE`. A clear "the port is down" beats failing later at
  `alloc_pd` and blaming your own code, which is what happened here.
- **Bound every blocking call, not just the one you remembered.** The first hardened
  version of `mesh-hop` had a deadline on `ibv_poll_cq` and still hung forever, because
  `accept()`, `connect()` and the out-of-band `read()` in front of it had none. A
  program with one unbounded syscall is an unbounded program.
- **One experiment at a time.** A sweep that launches a process per size, killing
  the previous one, is a leak amplifier.

## Habits that are safe elsewhere and are not safe here

These are normal, good practice in most environments. They are load-bearing
mistakes on this hardware:

| habit | why it is normally fine | why it is not fine here |
|---|---|---|
| kill and retry the dev loop | the OS reclaims everything on exit | the OS cannot reclaim a process wedged in the driver |
| unbounded blocking wait | a supervisor or Ctrl-C can always kill you | the kill does not arrive; the process is uninterruptible |
| rely on process exit for cleanup | kernel frees fds and memory | the device context is not freed if the process never exits |
| run a parameter sweep in a loop | each run is independent | each leak accumulates against `max_qp: 11` |
| assume a failure is your bug | usually it is | check the substrate first; Apple's own tools failing means it is not you |

Containers, GC'd runtimes and timeout-wrapped test harnesses normally conceal all
of the above. They do not help here — the leaked state is in the kernel driver, on
the other side of every abstraction you have.

## Diagnosing

```
ps -o pid,stat,command -p <pid>     # STAT U   = wedged, unkillable
ibv_devinfo -d <dev>                # hanging  = itself a symptom
ibv_uc_pingpong -d <dev>            # if Apple's tool also fails, it is the device
mesh-status                         # PORT_ACTIVE per device, check this first
```

## Recovery

There is no software recovery. A clean reboot may hang on the wedged processes. The
ladder is: stop launching new verbs processes, then power-cycle the machine.

Which is precisely the physical visit this repo exists to avoid, so treat these
rules as protecting the node's reachability, not as style.

## Warm the neighbour cache before RTR

`ibv_modify_qp(..., IBV_QPS_RTR)` fails on both peers when the IPv6 neighbour cache has no
entry for the destination GID, even though both ports report `PORT_ACTIVE`, the cable is
attached, and `system_profiler SPThunderboltDataType` shows the peer device. Apple's stack
resolves the destination GID to a link address through the neighbour cache; an empty cache is
a failed transition, not a delayed one.

This happens after any replug, because the RDMA device names follow the interface and the new
port has never done discovery. Symptoms are symmetric — both sides die at RTR — which reads
like a bad out-of-band exchange and is not.

```
ping6 -c 2 'ff02::1%<iface>'    # populates the cache, needs no root
ndp -an | grep <iface>          # confirm the peer appears
```

Do this before blaming the QP setup. Verify the pairing is real rather than assumed: the two
`GID[0]` values from `ibv_devinfo -v` must be the two link-local addresses `ndp` shows on that
interface.

## A false-loss storm can panic the kernel

On 2026-08-28 a MacBook Pro kernel panicked inside `com.apple.driver.AppleThunderboltRDMA`
with an MTE tag check fault, which is the kernel catching a corrupted or freed pointer. It
was not caused by killing anything. The link was carrying a normal bidirectional workload.

What the surviving peer's log showed, in order:

| symptom | value |
|---|---|
| sequence gaps recorded | 7,321,292 in five seconds, against 32,821 pages received |
| double frees caught by the guard | 55,335 |
| conservation invariant `nfree+posted+nready+sending` | fell from 125,829 to 81,691 |
| local protection errors | 1, on a page in state POSTED |

The order matters. The gaps came first, the double frees followed, the accounting drained,
and only then did a receive buffer complete with `IBV_WC_LOC_PROT_ERR` — the hardware being
handed a buffer whose registration or state no longer made sense. The panic followed about
seventy seconds later.

**The amplifier was an unbounded loop in the data plane**, which this document already
forbids in another form:

```c
for(uint32_t m = expected; m != h->seq; m++) { ... }   /* runs h->seq - expected times */
```

One arriving page drove that loop as many times as the sequence number claimed had been
missed. A gap wider than the repair bitmap cannot be repaired from anyway, so the loop was
doing unbounded work to record something unusable, while generating retransmit requests that
produced more traffic and more apparent loss. It is now bounded by the width of the miss
window, and anything wider is counted as `farseq` rather than walked.

Rules this adds:

- **A single received page must never drive work proportional to a value it carries.** A
  length, a count or a sequence delta on the wire is an input, and the peer that sent it may
  be wrong. Bound every loop by a local constant, not by a remote number.
- **A conservation invariant that drifts is an emergency, not telemetry.** `sum` fell by a
  third of the pool in three seconds and the program kept running. It should have stopped.
- **A guard firing 55,335 times is a guard that was ignored.** The double-free check printed
  one line per event and the flood was mistaken for noise. Count them, print the first few,
  and treat a nonzero count as a failure.

## Pairing leaks a per-boot driver resource, and RTR's EFAULT is the fuel gauge

`ibv_modify_qp` to RTR can fail with EFAULT while every argument is valid, the port is
active, and the vendor's own pingpong succeeds beside it. The discriminator, found by role
swaps and a region-size bisect: the fault tracks the amount of memory the process has
registered. Early in a boot a 4 GB registration pairs instantly; after a few hundred
queue-pair setup/teardown cycles on the same boot the same registration fails forever while a
0.14 GB one pairs in seconds, and a machine that has cycled less still affords more. The
driver appears to install per-region DMA state at pairing time from a finite per-boot pool
that teardown does not fully return.

Consequences:
- A healing loop is itself the leak amplifier. Converge in as few pairing attempts as
  possible; never spin QP bring-up at high frequency.
- If RTR starts returning EFAULT with valid arguments, do not debug the arguments. Shrink
  the region to confirm the diagnosis.
- Measured: a cable replug re-enumerates the controller and the mesh heals across it, onto a
  different physical port even, but the pool does not come back — it is host-side kext state,
  scoped to the boot.
- The pool is spent at pairing time, so the bridge rendezvouses over TCP before touching
  verbs at all: an attempt with no peer present costs nothing, and waiting happens at
  exponential backoff, which is free. Only a completed rendezvous spends registration and a
  queue pair.
- When exhaustion is real — repeated RTR faults at the configured size after completed
  rendezvous — the node recovers itself: it dumps state, and where the bridge runs as root it
  schedules its own reboot with a thirty-minute cooldown. Everything returns by supervision:
  the bridge at boot, pairing by the heal loop, clients by the generation remap. A reboot is
  therefore just another physical event the mesh heals through, self-scheduled, never a
  human's errand. On a node whose bridge is not root, the exhaustion is logged loudly with
  the recommendation instead.
- Size regions with headroom for the boot's remaining budget, not the machine's RAM.
- Failed pairing attempts consume the pool too: a size that paired an hour ago can be
  unaffordable after a retry storm at that size. Cap retries low and shrink before retrying.
