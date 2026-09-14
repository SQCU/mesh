# Explicit metadata audit: Python and Xonotic

The frame adapter now retains and traverses its actual output references and
input results. Reservation selects writable references, and reception delivers
any ready reference. There is no sent/received modulo counter from which a
buffer identity is reconstructed, and an absent earlier buffer does not hide
a later ready buffer. Fixed backing-buffer ownership remains in canonical mesh.

The FFN composition preserves input dtype in its output specifications. This
avoids converting an F16 up-projection into an F32 activation accidentally and
then attempting a mixed-type down contraction.

The generated Metal compiler retains operand IDs at source construction, binds
only the actual operand/output view list, and records actual physical strides.
Logical indexing and physical layout are distinct fields. Registered-buffer
addresses are direct; the old pointer-table page division is gone. Per-dispatch
argument-buffer selection is explicit, including restoration after output clear.

Retained examples and the planner no longer invoke the deleted host numerical
scan. Peer examples sleep for process shutdown signals while native execution
runs independently; terminal-result observation remains a host concern.

The obsolete tensor runtime's persistent policy, remote graph protocol, worker,
strategy execution, learner, and responder entry points were deleted. No
replacement runtime or compatibility shim was introduced. Their numerical
model/compiler source and the canonical planner/Frames callers remain. Their
obsolete curriculum, demo, distributed evaluation and responder-liveness launch
flows were also removed. The live J-oracle server and game engine remain.

The compiler preserves graph owner-to-peer mappings. Distributed realization
passes the same `root_peer` on each participant; root owner zero maps to that
peer, and named graph regions retain their declared peers. Cross-owner values
have explicit destination tensors and `Program.copy` edges. Replica reuse is
keyed by value ID and destination peer. Each declared output is allocated by
`kernel_call` on every participant, while `peer` selects its numerical executor
at setup. There is no remote graph scheduler.

The canonical kernel call retains the declared block partition. Application
compiler expansion and a separate gold-chain qualification are not collective
requirements; the [asynchronous contract](async-collectives.md) defines the scope.
