# Inspect a running application

A mesh program reports as it runs: each completed whole call's record (latency, useful bytes, its report) goes to its
stdout through `mesh_statistics.emit`, one JSON line, as the call completes, whether or not the other ranks have
finished. Output that is not a record is the program's own diagnostics. A late peer's observation still enters the
feedback collector, and an unfinished later call does not hide an earlier completed call's record.

The hourly (`rdma/mesh_hourly.py`, every node's `io.mesh.hourly` job) runs each program of metal-microbench's
`tools/mesh/hourly/workloads.json` as anyone runs it. Its record is the program's own output, in the job's log on each
node, `/usr/local/mesh/log/user/io.mesh.hourly.log` (newsyslog rotates it past 4 MB, two kept): a line naming the
program and the nodes before each run, the program's records and diagnostics, and a line with its exit status after.

A program keeps its own observations, which its placement reads before it runs and its close appends after
(`rdma/rollout.py`: a path as named, or a bare name's `~/.cache/mesh/observations/<name>.jsonl`). Live connectivity
and the bandwidth landed through each port are `mesh-top`'s, which starts its own telemetry sampler for as long as it
runs.

A preparation that installs new bridge images restarts the bridge (SIGHUP, its region kept), and that fails the ranks
of any program attached at that moment: each link is lost and paired again, and each communicator must be made again
(design/transport.md section 14). On 2026-10-28 one run's preparation restarted the three headless bridges 3 s into
another run's chain-n3-long (20261028T221222Z-G), whose three ranks failed so.

A Metal command's completion alone does not establish that its collective finished. Preparation attaches the
function's communicators to its shared schedule. Both native callbacks and the Torch observer read the same command
and collective outcome. This reads existing error records; it adds no acknowledgment, GPU command or wait to the
numerical function.
