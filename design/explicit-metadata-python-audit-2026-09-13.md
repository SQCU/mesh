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

The obsolete tensor runtime's persistent policy, remote graph protocol, and
worker entry point were deleted. No replacement runtime or compatibility shim
was introduced. The higher-level strategy execution module still names the
deleted policy adapter and requires a real composition migration before use.

Remaining compiler limitations are explicit: kernel_calls still binds one
whole region per operand and lowers every graph node locally even if its graph
owner names another peer. General BlockSpec region lowering and preservation
of that ownership belong to the retained compiler migration, not a second
remote graph scheduler. The gold distributed chain remains outstanding.
