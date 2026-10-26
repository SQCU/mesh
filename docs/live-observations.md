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

The application observer emits each completed whole-call record through
`mesh_statistics.emit`. Late peer observations still enter the existing
feedback collector. An unfinished later call does not hide an earlier completed
call's record. At driver exit, the existing compact hourly record stores the
aggregate measurements and failures; the live endpoint ends with its owner.
