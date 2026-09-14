# Sparse routing source and operational evidence

The existing streaming-algebra program at `4beaeb1` accepts
`--scatter-destinations` independently of update count and tile size. Additional
destinations receive ordinary early updates where available; empty destinations
still return their base. Destination 2 retains the withheld update/factor tile.
The second occurrence changes routes, values and factors, including an invalid
wide integer destination. Every returned destination is numerically checked.
This extends the existing numerical workflow, not a separate evaluator.

## Before shared ownership

The installed library is `a6305fb`. Local CPU and Metal runs use float32,
67 updates, 33-row input tiles, 17 destinations and three measured gold
invocations after warmup. Both complete the gold, independent scatter consumers,
changing routing occurrences and 65-way source reuse. A CPU run with four
destinations provides the smaller routing comparison. Raw observations and traces
are in `measurements/lowering-2026-09-13/sparse-before-*.json.gz`.

| Destinations | Backend | Configured functions | Ordinary input maps | Indexed candidate entries |
| --- | --- | ---: | ---: | ---: |
| 4 | CPU | 1672 | 3141 | 404 |
| 17 | CPU | 1711 | 3206 | 1275 |
| 17 | Metal | 1711 | 3206 | 1275 |

These are whole-workflow trace counts, including unchanged gold and side cases.
The extra 13 destinations retain 871 additional candidate entries: exactly
13 times 67. Source `_lower_indexed_add` constructs every candidate for every
destination, independently confirming the multiplication. Counts describe actual
retained bindings, not allocated bytes or physical occupancy. Local timing data
is retained but these runs do not establish a speedup or distributed throughput.

The replacement must expose its shared candidate and consumer identities in
diagnostics; omitting candidates from a trace does not prove eliminated storage.
The [shared routing design](shared-routing-ownership.md) separately identifies
domain ownership, active numerical work and physical allocation granularity.

## Initial shared-domain implementation

At `232e3b4`, local CPU and Metal complete the same wide scatter case and the
optional actual Xonotic gather/concatenate graph. The Xonotic observations include
canonical row identities for the source, indices, gathered intermediate, tail and
outputs. Both changing occurrences return exact values while the withheld source
and index blocks remain writable and the corresponding output remains unavailable.

A paired CPU float16 run also completes the gold, wide scatter, Xonotic case and
65 remote fanout branches, including second source reuse. Gold maximum absolute
error is 0.001953125 under the existing tolerance. The peer exits normally by
SIGTERM. Scatter domains and the optional Xonotic side graph execute locally on
rank zero in this workload; this is not evidence for remote candidate storage in
a shared route domain. The gold and fanout exercise the actual RDMA link.

The matched local CPU workflow without the optional Xonotic case changes as
follows, retaining 67 updates and 17 destinations:

| Quantity | Before | Initial shared domain |
| --- | ---: | ---: |
| Configured functions | 1711 | 1696 |
| Ordinary input maps | 3206 | 3225 |
| Indexed candidate entries | 1275 | 136 |
| Shared route candidate entries | 0 | 67 |
| Completed numerical submissions | 1989 | 1959 |

The removed 1139 indexed candidate entries equal 17 times 67. Their replacement
has one 67-candidate table and 17 explicit consumers. The 15 fewer configured
functions eliminate per-destination boundary lookup in favor of shared directory
production; two scatter occurrences account for 30 fewer numerical submissions.
These counts are source/trace evidence, not a throughput percentage. Gold timing
variation remains unresolved and no no-regression or speedup claim follows.

`sparse-initial-*.json.gz` records installed source `232e3b4`, raw observations and
traces. Further review found redundant ordinary directory readers in each
consumer, scaling with directory page count. Removing those readers and repeated
full-directory presence scans is a subsequent native correction; this initial
version is not the final sparse metadata bound. Potential segment allocations and
empty numerical launches remain separate work in both versions.

## Shared directory ownership and larger execution

`465ecfc`, `d864402` and `5843a8b` complete explicit sorted-key ownership and
subtract domain-held directory and immutable table rows from each consumer's
ordinary reader maps. Numerical argument pointers remain unchanged. Prepared
means the domain holds its metadata; it clears before the first metadata reader
is released. Source arrival and consumer readiness no longer scan the complete
directory's presence rows. Resolved trace flag 8 preserves the distinction from
an active prepared domain after every candidate and consumer has finished.

Installed `5843a8b` passes local CPU and Metal with 1025 updates, 513-row input
tiles and 17 destinations, including both changing scatter occurrences, delayed
factors, Xonotic indexed composition and reused fanout. Both traces contain one
1025-candidate domain, 17 consumer identities and a two-page immutable address
table. Each final consumer retains one ordinary reader map for its base; shared
metadata and selected candidates are represented by the domain. Each whole
workflow completes 3915 numerical submissions. These are actual trace identities
and counts, not inferred physical overlap.

For that shape, the previous final-scatter descriptors alone would allocate
17 times (2 times 1025 plus 2), or 34884 logical result rows. The configured
substrate has 32768 rows, so that representation cannot fit even before its
operand and other lifetime rows are included. The shared domain uses 1025 plus
17 plus 1, or 1043 result rows for the corresponding ownership facts. This
comparison follows directly from allocation source; no attempted oversized run
was required. It does not include the still-separate partial and input-reader
storage costs or prove arbitrary-size execution.

Paired CPU and Metal at `5843a8b` also pass the 67-update float16 workflow on the
actual RDMA substrate. Gold maximum errors are respectively 0.001953125 and
0.00390625 under unchanged tolerances. Both return all 65 remote fanout results
over two source occurrences. Both peers terminate normally. The same locality
limit applies: scatter domains and Xonotic side computations are on rank zero;
remote domain candidates and derivatives still need applicable workload evidence.

`sparse-final-*.json.gz` retains these observations, both peer traces where
applicable, and exact installed revisions. This completes the shared sparse
candidate/directory ownership increment, not the nine-step lowering plan.
Remaining work includes active segment grids, physical partial storage, general
expression composition/rank coverage, caller derivatives, collective placement,
and matched performance acceptance. No universal no-overhead claim is made.
