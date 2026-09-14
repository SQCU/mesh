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
