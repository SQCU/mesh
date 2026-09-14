# Explicit transfer descriptors and the composed gold run

Source measured: `602f3cd`, on Apple M5 Max and M4 Pro over Thunderbolt RDMA.
Raw observations: [indexed-gold-2026-09-13.json](../measurements/indexed-gold-2026-09-13.json).

The historical CPU harness `examples/streaming-algebra.py` composed
FFN → RMSNorm → summed learned embeddings → FFN → RMSNorm.
The first FFN's hidden sections move from node zero to node one; the second
FFN's hidden sections move back. Both directions use data queue zero. One
additional RDMA queue carries explicit index tuples. No operand staging buffer
is used: each payload receive names the retained peer pages directly.

All three invocations withheld row section zero until section one completed
the entire chain. Every invocation succeeded. The peer also observed FFN1,
RMSNorm1, and the embedding sum for section one while section zero remained
absent. These observations exercise the old absent-first-send failure on a
single data queue, rather than avoiding it with separate queues per section.

The maximum absolute numerical error against the float64 host reference was
2.7124965829017356e-6. Node zero reported 228 submitted and 228 completed CPU
functions, with error code zero. Both bridges reported code zero after the run.

| Observation | Count | Mean (ms) | Sample variance (ms²) |
| --- | ---: | ---: | ---: |
| First completed final section, all invocations | 3 | 21.4757223333 | 1255.7835123558 |
| All final sections, all invocations | 3 | 22.3458056667 | 1254.2653624668 |
| First completed final section, subsequent invocations | 2 | 1.016146 | 0.001442274632 |
| All final sections, subsequent invocations | 2 | 1.8986045 | 0.002511632813 |

The first invocation's 62.394875 ms first-section latency includes cold
startup/pairing. It is retained in the artifact and separated explicitly above.
These are host-observed latencies, including observation-loop granularity.
Peer monotonic timestamps have their own origin and must not be subtracted
from local timestamps. This is evidence of independent section progress and
numerical correctness, not a throughput speedup claim or a per-kernel overlap
measurement. Two subsequent invocations are a small sample.

Each peer registered 137,560,064 bytes: 8,192 arena pages of 16 KiB, four pages
per payload block, plus the canonical metadata and preallocated index buffers.
All graph instances and numerical storage were realized before input publication.
The peer process was terminated gracefully after observation; both bridges remain
running with no attached clients.

The source review and parallel audit additionally retained occurrence indices,
per-transfer fanout reader ownership, announced payload continuation after a
failed post, Metal argument-buffer indices, CoreML specialization identities,
canonical backing addresses/layouts, execution provenance, compiler operand IDs,
physical strides, graph ownership, frame refs, tensor dtypes, declared output
partitions, and supplied deployment identities. Details are in the algorithm
sources and the associated audit documents. Ordinary contraction indexing and
transport storage-ring counters remain; they do not substitute for destinations.
