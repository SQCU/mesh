# Mesh telemetry implementation claims

Controlling specifications: [`../mesh-coprocessor-demo.md`](../mesh-coprocessor-demo.md),
[`../rl-training-spec.md`](../rl-training-spec.md), and
[`../ALGORITHM-CONTRACTS.md`](../ALGORITHM-CONTRACTS.md).

Implementation surfaces:

- `user/mesh-telemetry.py` owns each node's in-memory leased telemetry stream.
- `viz/serve.py` owns discovery, membership, live whole-fabric aggregation, and the
  page-table visualization feed.
- `rdma/workload.py` owns workload-independent operation envelopes and measures.
- `xonotic/solver/strat/roofline.py` owns measured operating-point search.
- `xonotic/solver/strat/expert_worker.py` and `strat_responder.py` own distributed-scale
  placement and counterfactual measures.

The claims are substantiated only when active aggregates have the same support as the
reachable lease, stale capacity is reported only as inventory, missing objective
coordinates retain their own support measures, and both forward and optimization work
are observed on distinct mesh hosts.

Page-table capacity is the configured region disintegrated into receive ownership and
application arena ownership. Neither component is capped by a historical workload count,
and failed pairing attempts do not change either coordinate.

Arbitrary-extent stream closure is driven by byte offsets and transport completions. The
sender emits its first FIN after the complete data extent and retries it only after the
prior FIN page is reclaimed; an application spin count is not a response window. The
queued-copy store grows from submitted row mass with checked address-space arithmetic.

Machine-activity FLOP bounds are workload-independent. Characterized CPU capacity times
the maximum live CPU-cluster residency and characterized GPU capacity times GPU residency
form concurrent upper-bound components; their sum brackets a CPU-only, GPU-only, or mixed
workload without naming it. A rootless telemetry owner invokes only the root sampler through
noninteractive sudo and keeps the same in-memory ring, port, and sequence protocol.
Mach CPU ticks provide the unprivileged whole-CPU residency coordinate; privileged sampling
adds power, frequency, and thermal coordinates but is not the source of CPU availability.
An instrumented interval with unspecified arithmetic publishes a zero lower bound and a
missing upper bound. It never publishes zero as an invented upper bound; component support
and missing capacity remain explicit coordinates.

CPU FP32 characterization doubles a square SGEMM realization from matrix order one until
one measured multiplication occupies the requested sampling duration or allocation
capacity is exhausted. The characterized rate is the maximum observed rate along that
hardware-driven trajectory; neither a matrix order nor a workload row count is a release
ceiling.

The M4 Pro trajectory reached its requested interval at square order 8192 and measured a
3,488.27664 GFLOP/s maximum at the 8192 by 8192 by 1024 coordinate. The complete
characterization stream remains the measurement artifact rather than a preset workload
size.

Ports 8788 and 8787 should each have one answering provider. Installation retains a
loaded system job or an answering userspace provider rather than forcing an ownership
transfer. User installation measures the system job's socket ownership and response
before relying on it. It can bootstrap the same userspace interface when the system
provider is absent. Definition refresh and any pre-existing supervisor overlap require
a continuity-preserving handoff; routine installation no longer resolves them by stopping
the working provider. Retained jobs are not reported as newly deployed generations.

The Xonotic operating profile is indexed by the literal player, team, and cart counts
reported by the workload. Runtime bot changes trace a player-count slice at fixed team
and cart coordinates. Match realization moves between those slices. Aggregation never
merges observations whose three coordinates differ.
