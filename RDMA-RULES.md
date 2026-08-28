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
