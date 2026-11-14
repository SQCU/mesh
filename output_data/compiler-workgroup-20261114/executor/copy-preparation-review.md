# Typed copies and preparation

The native consumer now composes the actual typed prefix landing, retained body append, and suffix staging/output
copies before construction. It consumes `CopyRecord`, original input slots, and `InputPart.landing`; it does not
derive partitions from read footprints. One actual wire root owns its typed views. The append's value map and
resulting storage-root map stay separate and reach `Prepared.remap` and the final row recipe.

New caller/result roots are external operands before construction. Wire roots remain planned storage. Local
`Arena` preparation initializes each zeroed function/root copy using the solved first/offset views and retained
byte extent. Source and archive-loaded local owners take this same preparation path.

Review corrected a per-view backing split, a stale tuple unpack of a scalar landing root, and the omitted local
state-copy initialization. SAME047 implemented the connected source; supervisor AST and diff checks passed.
InputPart's strided layout comes directly from its typed producer. No new numerical or network execution is
claimed here.

The next existing cost client reached real local planning but stopped before invocation: command0 binding45
references value68 without placement. Its concrete Value/Storage/operand correspondence is under review. This
is an open consumer failure, not a measured cost result. Prior CPU/archive observations remain separately bounded
in current-owner-review.md.
