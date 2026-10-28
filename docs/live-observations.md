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

Discovery reads the local bridge's existing cluster map and binds the administration
and fabric addresses in `etc/mesh-nodes.json`. Each command is visible in
`discovery` as it runs. It does not browse DNS, probe unrelated Tailnet hosts, or
fetch source to discover a machine. A missing configured node fails preparation.
The parallel installation commands verify each endpoint's actual bridge identity.

Each workload's ranks are the nodes its `place` selects in `etc/mesh-nodes.json`
(`Cluster.place`: `compute` nodes with a role and least values, the `root` first). Each rank runs under
`mesh_nodes.py node guard` on its node, in its own process group, which the guard
kills at the rank's deadline or below the node's `guard_free_pages`, then confirms
empty. A run ends by `run_seconds`; its workloads stop a minute earlier so the
driver returns its record. The cluster lease is a `flock`ed file naming its owner
and pid; it ends with its holder.

Each node keeps one `current` Git build checkout per repository. It fetches a
selected commit from origin when needed and retains incremental build products.
One administration command prepares both repositories and the installation.
An installation receipt names the commit, OS build, SDK build and Python ABI;
unchanged inputs reuse the installation after live bridge and package checks.
Changed inputs enter the ordinary build dependencies. The Metal binary archive
also depends on the OS/SDK record, rather than rebuilding on every invocation.
Source revisions replace installer content hashes and temporary source copies.

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
