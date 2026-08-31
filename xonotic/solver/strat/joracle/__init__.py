"""j-oracle: a read-only side channel onto the live strategy responder.

Nothing in this package writes to the game, the responder, the mesh, or any
checkpoint.  It tails the responder's own telemetry JSONL and republishes it,
plus rolling linear probes on the live IR, over a local HTTP port.

Design rule carried from design/jspace-probe.md: a probe score without its
controls is meaningless, so every probe here is reported alongside a random
projection of the same raw inputs and a shuffled-label run.  Fields that are
absent from the stream are reported as absent; nothing is defaulted to zero and
then rendered as if it were measured.
"""
