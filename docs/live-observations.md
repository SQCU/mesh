# Inspect a running application

The committed hourly driver serves `GET /STATS` at the address in
[`etc/statistics.json`](../etc/statistics.json). Its default address is
`http://127.0.0.1:18619/STATS`. The driver prints that address when it starts.

```
curl --silent http://127.0.0.1:18619/STATS
```

The reply includes the selected commits, current stage, discovered nodes,
installation status and workloads. Each node's `commands` holds its
administrative command states. A running workload's `ranks` holds the process
states by rank. Each process exposes:

- Its PID, start time, status, exit code and finish time when known.
- `output`: stdout and stderr chunks, available as the pipe delivers them.
- `updated_unix`: when output most recently arrived.
- `records`: complete JSON records decoded from that output, including paired
  whole-call latency and useful throughput.
- `diagnostics`: non-record output, also used for failure reporting without
  printing all numerical records again.

Output does not wait for a newline, process exit or another rank. Exit status
does not wait for the other ranks. A quiet process is reported as running;
silence alone is not reported as numerical progress or failure. The PID on a
remote node's entry is the local administration process, which names that node
in its command.

[`mesh_nodes.sh`](../rdma/mesh_nodes.py) supplies this stream to every shared
node command. [`mesh_statistics.serving`](../rdma/mesh_statistics.py) exposes the
owner's state directly. It has no numerical executor, synchronization command,
disk cache or write endpoint. Other administrative callers can serve their own
state with the same function and an explicit listen address.

Discovery reads every configured node's live map (its mesh-stat, over its
administration path in `etc/mesh-nodes.json`) and takes the union; the launcher
need not be a fabric member. The fabric is the union's largest linked part, and it
alone is the membership: a member with no configured entry is reached at its
fabric-routed address through a configured member (bin/mesh-peers.sh), or recorded
unreachable. A
configured node that is unreachable, whose bridge is not up, or that is outside the
fabric is recorded under `absent` with the reason, and the run goes on: every
reachable node is installed, each workload is placed over the live, installed
members, and a workload that none meets, or that fails, is recorded failed while
the others still run, each gated by its own pairing check. Pairing
is checked against the union's links. Each command is visible in `discovery` as it
runs. It does not browse DNS, probe unrelated Tailnet hosts, or fetch source to
discover a machine. The parallel installation commands verify each endpoint's
actual bridge identity.

Each workload's ranks are the nodes its `place` selects in `etc/mesh-nodes.json`
(`Cluster.place`: `compute` nodes with a role and least values, the `root` first).
Each rank runs under `mesh_nodes.py node guard` in its own session with the run's
token (`MESH_RUN`), its operands the first line of a stdin its caller holds open.
The guard ends it at its deadline, when the node's headroom falls below its
`guard_headroom_gib`, when that stdin closes or on SIGTERM or SIGHUP. The headroom
is memorystatus's numerator A, free + speculative + pageable internal and external
pages, read every 0.1 s by `sysctlbyname` (`host_statistics64` is rate-limited);
on a `workstation` it is A less `vm.compressor_bytes_used`. A failed read ends the
rank too. It then SIGKILLs the rank and each descendant it
tracked by parent pid, each pid checked against its start time, until none remains,
and then every process of the user still carrying the token (`KERN_PROCARGS2`); it
exits 125 naming any survivor. The rank's own alarm ends it should the guard die.
A descendant still escapes when it leaves its parent between two samples and runs a
platform binary (whose environment macOS hides) or with `MESH_RUN` removed.
A run ends by `run_seconds`: every command of its driver is bounded a minute
earlier, so the driver returns its record. A stop (the bound, SIGTERM, SIGHUP,
SIGINT) cancels the bound, has the driver close its ranks' stdin and return its
record, kills its session past a minute, and ignores later stops while the record
is written. Every hourly appends one line to log.jsonl, one that cannot start
included, naming the commits its ranks loaded and the runs whose ranks shared its
nodes. There is no node lock: runs of different commits share a bridge as clients.

Each run uses its own commit's mesh checkout on every node
(`<checkouts>/mesh/<sha>`, the `checkouts_kept` most recently used kept). node prepare
runs make there (which rebuilds only what is stale), _mesh_c and torch_mesh's
extension in place, and its ranks load them by path (PYTHONPATH and the libraries'
rpath), never from the shared venv; each reports the commit it loaded. So programs
of different commits attach to the same resident bridge as ordinary clients. The
bridge runs from its own checkout beside them (`current`), which node prepare moves
to the run's commit and installs with bin/mesh-user.sh as it is: that restarts the
bridge (SIGHUP, its region kept) only when its images changed. A node's preparations
take turns.

The application observer emits each completed whole-call record through
`mesh_statistics.emit`. Late peer observations still enter the existing
feedback collector. An unfinished later call does not hide an earlier completed
call's record. At driver exit, the existing compact hourly record stores the
aggregate measurements and failures; the live endpoint ends with its owner.

A Metal command's completion alone does not establish that its collective
finished. Preparation attaches the function's communicators to its shared
schedule. Both native callbacks and the Torch observer read the same command
and collective outcome. This reads existing error records; it adds no
acknowledgment, GPU command or wait to the numerical function.
