# Shared Xonotic gather and concatenation

September 14, 2026. This increment migrates gather and concatenate with one- or
two-dimensional sources/indices and at most two-dimensional results to the
existing expression compiler. It adds no new compiler primitive or execution
interface. Higher-rank operations and shape reinterpretations that cannot be
represented by the existing direct views retain their current numerical source
until their shared lowering is implemented.

## Retained numerical maps

`tensor.py` supplies gather's existing mapping tuple, advanced-index shape and
adjacency flag. `tensor_metal.gather_expression` translates that same mapping
into expressions rather than generated Metal statements. Output coordinates are
`program_id * block_extent + indices`, so the coordinates remain global even when
an independently published output block has a ragged tail.

Slices preserve their start and step, including negative steps. Fixed indices
remain fixed. Advanced index operands use the existing broadcast-axis placement;
singleton dimensions use coordinate zero. Negative dynamic indices are normalized
against the corresponding logical source axis before the source load. Inserted
axes consume output dimensions but not source dimensions. Integer indices remain
integer expressions, and source loads preserve the declared value dtype.

Concatenation retains the configured axis and cumulative source lengths. Each
output coordinate selects the corresponding source and subtracts its source's
origin on that axis. Nested selects contain ordinary indexed loads; their branch
predicates are retained by the existing expression dependency analysis. A tile
spanning a concatenation boundary legitimately reads both source regions. A tile
inside one source does not require another source's missing values.

The mechanism follows the [Pallas reference and index-map model](https://docs.jax.dev/en/latest/pallas/design/design.html)
and uses mesh's already implemented conditional indexed-load and selected-reader
lowering. The caller does not construct selection buffers, dynamic reader state,
backend source, or a second numerical scheduler.

## Storage, publication and derivatives

Logical vectors use the existing 1 by N representation. Matrix views preserve
canonical backing pointers and actual strides. No gathered input staging tensor
is created; the gather output itself is the intended numerical result, stored in
its independently publishable output blocks. Concatenation likewise writes its
actual result without first assembling a whole input.

`Program.kernel_call` realizes these block outputs and bindings during setup.
The shared `.at` lowering supplies selected-page dependencies and source holds.
Changing caller syntax has not changed that runtime protocol. The default output
grid uses the existing tile_rows/tile_columns choices; it does not impose input
block alignment because indexed loads retain their source block maps.

The graph's derivative rules are unchanged. Concatenation's derivative slices
its cotangent, so covered slices now use this shared gather lowering. Gather's
reverse operation remains the existing gather_vjp implementation with explicit
block-address metadata; duplicate gradient indices still follow that operation's
existing collision semantics. This increment does not claim to have migrated
that reverse scatter, take_along_axis, or higher-rank derivatives.

## Evidence and remaining work

Python source compilation passes. Source review compares gather's translated
mapping directly with the previous gather_address routine and traces source and
output indices through the existing expression binder. No numerical workload was
run for this increment. The existing Xonotic workflow must validate actual emitted
kernels, numerical output and derivatives; syntax compilation does not establish
those results or a performance improvement.

Remaining work includes higher-rank view lowering, gather VJP scatter composition,
take_along_axis and its VJP, and compiled cumulative-sum/search operations.
Selected-load metadata and kernel launch costs remain measurable overhead; the
migration establishes region dependency structure without claiming zero cost.
The old custom implementations remain only for still-unmigrated cases, not as a
second completed public implementation style.
