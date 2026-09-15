# Counterexample: explicit remote-fill synchronization

This is deliberately the path **not used by default**. The operator requested it
so that synchronization remains visible and reviewable rather than being mistaken
for a missing backend feature. `send`, `reduce_scatter`, `all_gather`, and
`all_reduce` never call `sync_on_remote_fill`.

## Explicit API

```python
from mesh.collective import sync_on_remote_fill

remote = program.export(remote_ref)  # setup, before realize
program.realize()
# Publish all inputs needed to produce remote_ref before blocking this thread.
sync_on_remote_fill(remote)
use(remote.array)
remote.consume()
```

The arguments are previously exported `Result`s. The call blocks its calling thread
until all supplied regions are present. It does not initiate communication, publish
inputs, consume results, insert a graph edge, or wait on unrelated regions. It does
not discover whether a Result came from a remote producer; the caller names the
remote results. Exported reader ownership retains those values until explicit
`consume()`. An already consumed result cannot satisfy a wait for that generation.
The empty conjunction returns immediately. A reported native error raises `OSError`.
There is no timeout, clock condition, implicit cancellation, or deadlock detector.
Python interruption remains possible; it is not successful completion.

MPI's separation of nonblocking initiation and explicit completion is the prior art
([MPI-4.1, communication completion](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node74.htm)).
This is a polling completion wait over mesh Results, not an MPI implementation or a
CUDA warpgroup synchronization primitive.

## Same numerical graph, different caller order

[The executable example](../examples/sync-on-remote-fill.py) allocates independent
registered source, receive, and reply blocks for every trial before execution.
Each block follows exactly this path:

```
root write[i] -> SEND -> peer received[i] -> add(1) -> SEND -> root reply[i]
```

The peer's addition uses the existing expression kernel. Each block is independent;
there is no mathematical edge between blocks. Mesh's existing numerical and transport
threads perform all progress. The peer host's `signal.pause()` only keeps the process
alive; it owns no publication needed by those threads. Terminating that client with
SIGTERM after a run closes its Program normally. Do not signal the bridge.

`--mode stream` publishes all blocks, then explicitly waits for the final result set
at the host observation boundary. `--mode sync` explicitly waits for each reply before
publishing the next input. Both perform the same writes, additions and transfers.
The latter introduces the artificial edge `reply[i] -> write[i+1]`, making the caller
pay a complete round trip between publications. For independent rounds of latency L,
the serialized span contains N*L; the streaming graph permits overlap. Measured wall
times also include submission costs, polling and contention, so a slowdown is a
measurement to report, not a universal timing assertion.

One separately allocated warm-up round trip precedes timing in both finite modes.
Timing uses only the root's clock, outside tensor functions. Every trial uses fresh
configured blocks. The output reports each elapsed time and online count, mean and
sample variance; numerical values are checked outside the timed interval. No sleeps,
artificial delays, mocked transport, or clock rendezvous are used.

## Independent progress with an absent input and retained results

`--mode independent` realizes the same graph, leaves `source[0]` unpublished,
and publishes every other source. The root observes available replies without
calling `sync_on_remote_fill`. It retains every returned value through the end of
the example: no `consume()` releases an earlier result to permit a later result.
Only the root's observation loop checks readiness; numerical and RDMA progress
remain on their own threads. There is no sleep, deadline, forced timing order,
or peer rendezvous in the numerical flow.

The source therefore makes two unwanted dependencies reviewable:
`source[0] -> reply[i]` for `i > 0`, and `consume(reply[i]) -> reply[j]` for
distinct configured values. Neither edge exists in the addition/transfer algebra.
The missing reply remains exported and is reported at the end, rather than being
removed from the graph. A newly introduced whole-input barrier prevents the
independent outputs from appearing; a result-retirement barrier prevents them
from coexisting. No timeout converts those failures into apparent completion.

This mode does not prove receive preposting, overlapping reuse of the same Ref,
or absence of every possible guard. The current announcement and output-claim
branches still fail G4. Those defects require source changes; this demonstration
is not a substitute for them. The mode has been added in source, not executed.

## The permanent wait

`--mode deadlock` moves the explicit wait for `reply[0]` before the only write that
could produce `source[0]`. All storage and functions are already configured.

```
wait returns -> root write[0] -> peer received[0] -> peer add[0]
     ^                                               |
     +---------------- root reply[0] <---------------+
```

Audit of every possible progress source:

- Only the root host publishes `source[0]`, after the wait returns.
- Only that transfer can publish the peer's `received[0]`.
- The peer addition needs `received[0]`; its other operand, constant one, is ready.
- Only the addition's output can supply the reply transfer.
- Only receipt of that reply can satisfy the root wait.
- The warm-up and all trial inputs remain unpublished. No other thread, transfer,
  timer or fallback publishes them.

Thus no event in the cycle can be first. Faster transport, more polling threads and
more buffer capacity cannot break it. The wait has no timeout and remains forever
unless something outside this dataflow interrupts or changes the program. Observing
it for a finite duration cannot prove forever; the closed cycle proves nontermination.

This audit is specific to the complete example. It is not a general proof that an
arbitrary placement of a completion wait is safe. Never place such a wait in a
numerical callback or before a publication needed to satisfy that wait.

## Running on two configured participants

Build the package on both nodes with
`uv run --no-project --with setuptools --with numpy python setup.py build_py`.
Set `PYTHONPATH` to the generated `build/lib.*` directory and `DYLD_LIBRARY_PATH`
to its `mesh/lib` directory. On each node, run the same command, using the actual
configured node IDs:

```sh
python examples/sync-on-remote-fill.py --root 0 --peer 1 --mode stream
python examples/sync-on-remote-fill.py --root 0 --peer 1 --mode sync
```

Run each pair separately. The root prints `DONE`; send SIGTERM to the peer **client
PID printed by the example** before starting the next pair. The deliberately
nonterminating mode is opt-in:

```sh
python examples/sync-on-remote-fill.py --root 0 --peer 1 --mode deadlock
```

The root prints its wait cycle and never prints a trial or `DONE`. Interrupt both
example clients with SIGTERM or Ctrl-C after inspection. This is an executable
counterexample, not a watchdog, acceptance harness, default library path, or reason
to introduce timeouts into tensor dataflow.

## Recorded two-node demonstration

Source `eabd9c5`, local node 0 (M5 Max) and node 1 (M4 Pro), existing Thunderbolt
RDMA bridges, one configured data queue pair per participant, CPU addition, FP32,
32 independent blocks of 16,384 elements per trial, five trials, and 42,270,720
planned arena bytes per participant. Package builds passed on both nodes. Both
finite modes completed and checked every output against the expected addition.

| Caller order | n | Mean per 32-block batch (ms) | Sample variance (ms²) |
|---|---:|---:|---:|
| Streaming submissions, final observation wait | 5 | 0.640367 | 0.0198608914341 |
| Explicit wait after each submission | 5 | 4.128258 | 0.0126009682811 |

Individual batch times in milliseconds:

- Stream: 0.886084, 0.627625, 0.541042, 0.580875, 0.566208.
- Sync: 4.308250, 4.060833, 4.154917, 4.019542, 4.097750.

The ratio of measured means is approximately 6.45. Runs were sequential, stream
then sync, without randomized ordering; this demonstrates this workload and setup,
not a universal slowdown factor. The arithmetic, payload sizes and graph were the
same. Only caller ordering changed.

The deadlock mode reached the printed cycle on the root. Both bridges were paired
with code zero and both client processes remained live; no trial or `DONE` appeared.
After inspecting that state, both clients received SIGTERM and exited normally.
Both bridges remained running and returned to idle, client zero, code zero. This
finite observation is consistent with the source proof above; it is not used as a
substitute for the proof.
