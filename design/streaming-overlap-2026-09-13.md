# Measured producer/link/consumer overlap

Source revision `fb7828b`, September 13, 2026. The public API demonstration is
[`examples/streaming-overlap.py`](../examples/streaming-overlap.py). The
[raw samples and summaries](../measurements/streaming-overlap-2026-09-13.json)
include every measured latency and all four paired runs, including the slower
fourth pair.

## Operation and comparison

The operation is `Y = (X @ W) @ V`. The M5 Max produces `X @ W`, sends those
output sections over canonical Thunderbolt RDMA, and the M4 Pro consumes them
in the second contraction. Its output sections return over the second queue.

`X` is FP32 `(4096,256)`; both weights are `(256,256)`. Each numerical call
multiplies a `(64,256)` tile by a `(256,256)` weight matrix. There are 64 tiles
per contraction and 128 numerical calls per distributed result. NumPy uses
Accelerate on both participants (2.4.6 on the M5, 2.0.2 on the M4). Neither mode
changes the math library, arithmetic calls, tile sizes, layouts, or participants.

Both modes keep two separately configured inputs in flight. The host refills
available input slots and scans all configured functions; it does not block one
input's pipeline on another input's return. Ten completions warm up each run,
then 3,000 completions are measured. Run order alternates whole/streamed and
streamed/whole across four pairs. Input values change between invocations.

In whole mode, one grid function executes the tiled contraction calls and
publishes the whole output allocation on return. The consumer therefore requires
the whole received operand before invoking its same numerical tiles. In streamed
mode, `BlockSpec.region_map` selects one 64 KiB section for each grid function.
Each numerical tile completes independently through the native publication path.
`call_native` prepares all array views and submissions during setup.

This comparison measures the gain from finer publication and consumption at the
same input concurrency. It does not compare a pipelined run against a one-input
request/response baseline. It also charges the streamed implementation for its
additional submission/completion overhead.

## Measurements

| Quantity | Whole operand | Streamed sections | Count per mode |
|---|---:|---:|---:|
| Mean throughput, results/s | 962.9283 | 1429.2196 | 4 runs |
| Throughput sample variance, (results/s)² | 3096.0368 | 3160.7359 | 4 runs |
| Mean latency, ms | 2.006869 | 1.326312 | 12,000 invocations |
| Latency sample variance, ms² | 0.980220 | 0.827785 | 12,000 invocations |
| Consumer launches before full operand reception | 0 | 758,887 | includes warmups |
| Total consumer launches | 12,040 | 770,560 | includes warmups |

Paired throughput gain: **48.5513%**. Its online statistics are count **4**, mean
**48.5513%**, sample variance **8.7497 squared percentage points**. Each sample
is `100 * (streamed_results_per_second / whole_results_per_second - 1)` for
one paired run; it is not the ratio of the two aggregate means.

Paired mean-latency reduction: **33.8516%**. Its online statistics are count
**4**, mean **33.8516%**, sample variance **1.6517 squared percentage points**.
Every retained sample contributes to the Welford summaries; no outlier was removed.

Numerical checking occurs after steady-state timing, on the terminal result in
each input slot, against the reference with that invocation's input scale.
Maximum absolute error was **5.739744e-6** in every run. This verifies the checked
terminal results, not a claim that every intermediate was separately evaluated.

## What establishes overlap

Before the M4 executes each consumer tile, the measurement reads physical
PRESENT bits for the full received operand. These are independent of an observer's
READ bits. The count therefore establishes that computation starts while other
parts of that operand remain unreceived. It adds no numerical dependency or
release condition. The earlier pilot used unreadness instead of presence and is
excluded from these results.

Producer publication is also regional: each submitted tile writes its declared
section, invokes completion, and `mesh_complete` publishes that section and marks
its configured send eligible. Later producer tiles are separate submissions.
The bridge independently posts registered source pages. The consumer repeats
that publication path for the return link. This source trace plus early-consumer
observations and the paired end-to-end gain establish the demonstrated overlap.

In Amdahl terms, the same numerical work loses a serial communication/computation
dependency and completes faster. These measurements establish that advantage for
this workload and window. They do not measure hardware utilization, prove that
all stalls disappear, or establish a universal speedup. A smaller pilot with
three inputs in flight had essentially equal throughput; more invocation-level
concurrency can already hide the dependency that region streaming removes.

The M4 NumPy process emitted matmul floating-point warnings during setup/warmup
execution in both modes; their text is retained in the raw record. Terminal
numerical checks passed. The report does not infer a cause from those warnings.

## Reproduction

On each participant, install the current main checkout with the local Python:

```sh
python -m pip install . --no-deps --target .build/overlap-package --upgrade
```

Use the Python environment with NumPy/Accelerate on that participant. With the
canonical bridge running, start the consumer on the M4:

```sh
PYTHONPATH=.build/overlap-package python -u examples/streaming-overlap.py 1 streamed --rows 4096 --depth 2
```

Run the producer on the M5:

```sh
PYTHONPATH=.build/overlap-package python -u examples/streaming-overlap.py 0 streamed --rows 4096 --depth 2
```

The consumer prints its PID and stays available while the producer runs. Once
the producer exits successfully, send SIGTERM to that consumer PID; it prints
its observation counts and detaches normally. Repeat with `whole` on both
participants, alternating order for paired repetitions. Process startup is not
a numerical rendezvous, and no synchronized wall-clock launch is used.
