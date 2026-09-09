# Stream dispatch index

`mesh_turn_window` builds a lookup table over the current stream pointer array,
then uses it for incoming frames. The table belongs to the client context and
retains its largest allocation until detach. It is rebuilt on every progress
turn, so binding, retirement and fairness rotation require no invalidation
protocol. A failed growth allocation uses the previous linear lookup.

The hash key is `(channel, receive direction, epoch)`. Matching still compares
all existing fields, including the source node. Node is deliberately outside
the hash: legacy wildcard receivers can acquire their node identity during the
same progress turn. The hashed fields are immutable while that stream is live.
Different source nodes with the same key remain separate candidates.

Insertion follows stream-array order. Linear probing visits matching candidates
in that order; the last match preserves the prior loop's selection. An empty
bucket proves that no later insertion for the same key is beyond that bucket.
The table has at least twice as many buckets as streams. Hash collisions never
suppress candidate matching. The fallback traverses the original pointer array.

For S streams and F frames, the expected work per turn is O(S + F), replacing
O(S*F). Collision-heavy keys can still require O(S*F); this is not a worst-case
constant-time claim. Neither probe count nor allocation extent depends on a
frame's advertised payload length or sequence gap.

This is dispatch indexing, not payload validation or error correction. Stream
agreement, numerical readiness, lease ownership and wire behavior are unchanged.
The bridge and shared-memory header do not change. Rebuild static clients and
consumers together after changing the public client-context declaration; the
microbench Makefile includes the transitive Metal-header dependencies.

Operational evidence comes from the existing Gemma full-model evaluator in
`metal-microbench/docs/model_tp_performance.md`. The indexed-only exploratory
cohort did not establish an end-to-end speedup. Larger live-stream workloads and
paired measurements remain necessary to quantify the indexing benefit.
