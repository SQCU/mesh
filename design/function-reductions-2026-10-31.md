# Supplied reduction functions

`Reduction` captures a two-input combine and an identity with an explicit scalar
type and independent-value shape. The callable is source. A prepared `Function`
runs the resulting page commands; it never calls the Python reducer.

```python
from torch_mesh.reduction import Reduction

combine = Reduction(torch.add, 0, dtype=values.dtype,
                    element_shape=tuple(values.shape[1:]),
                    reassociate=True, topology=chosen_basis)

def source(values):
    return combine(values, group)
```

`element_shape` describes one whole value, such as all fields of a softmax
summary. Cuts between values use that explicit schema. `root` selects a rooted
result; nonrecipients have a zero-width result. `counts` selects reduce-scatter
counts in independent values. With neither, every rank receives the result.
The example in [examples/function_reduction.py](../examples/function_reduction.py)
uses the ordinary `Function`, group declaration, and preparation interfaces.

`active` is an optional ordinary page operand with one Boolean or 0/1 value per
independent contribution. Source lowering selects the local value or the
declared neutral identity before expanding the chosen fold. Identity evaluation
uses a typed zero prototype, so it does not read an absent value. Capture and
autograd see that same selection. A changing mask does not select another
executor. This source input establishes numerical semantics; automatic projection
of it into every frontend's transport coverage remains integration work.

## The chosen fold is a source operand

The default preserves rank order and a left fold. `reassociate=True` permits a
new parenthesization; `reorder=True` separately permits a new leaf order. These
are algebraic permissions from the caller, including the permitted change in
floating-point rounding.

`topology` supplies the existing tree basis as ordinary configuration:

- `contributors` is the native rank bitset, stored in uint64 words.
- `segments` names independent value domains. For reduce-scatter, `counts`
  supplies those domains.
- `parents` contains one rank-parent array per tree; `UINT32_MAX` excludes a
  rank. `segment` assigns each tree to its domain, and `share` supplies its
  nonnegative allocation within that domain.
- `identity_owner` assigns each empty-contribution domain to a rank. Its default
  follows the rooted/scattered result owner, or rank zero for an all-reduce.

An omitted tree list seeds the declared rank order. `ncclMeshFoldPrepare` retains
those columns and their shares, adds permitted neighboring associations, and
uses the same page-range apportionment and binary-tree decoder as native
reduction placement. A zero-share neighbor remains a solver operand; it emits
no source work until assigned a nonempty domain. Reordering can consume explicitly
supplied reordered trees. The association-only neighborhood preserves rank order.

The result contains exact binary nodes with left input, right input, owner, and
round, plus each tree's value range. Source lowering expands this chosen DAG
before autograd and functionalization. A numerical node consumes the actual two
values named by that node. Selected page exchanges carry each value to its
owner. One contributor aliases its pages. No contributors evaluate the supplied
identity at the declared identity owner.

`Reduction.fold_choices` and the captured graph's `mesh_fold_choices` retain the
expanded basis and selected ranges. `Function.fold_choices` exposes the same
source choices. A subsequent proposal supplies revised `topology` operands and
recaptures the source. The later group routing solve routes the already-declared
page deliveries; it does not choose a second, different fold behind autograd's
back. There is no claim that local rotations or a continuous share optimizer
find a global optimum in a finite fixed number of steps.

## Differentiation follows the actual association

There is no opaque `torch_mesh::reduce_function` marker. Both forward-only and
joint forward/backward capture see the chosen source graph. Ordinary autograd
therefore differentiates each supplied combine at its actual left and right
operands, in the chosen association. The selected exchange's transpose adds the
cotangents of its recipients and returns them to the original source pages.

This does not assume that an associative forward function has a gradient which
can be reduced with the same function. Derivatives belong to the same captured
`Function`, parameter-gradient outputs, ownership pass, and page compiler. There
is no separate backward executor or manual distributed derivative call.

The source can use ordinary `torch.cat` to describe assembled results. After
joint capture, the common algebra rewrite turns concatenation into the existing
page-assembly equalities. Producers and consumers share those pages. A consumer's
real requirement for contiguous storage remains an operand of placement.

## Lowering precedes recording

The frontend retains numerical and transfer declarations after ownership and
page placement. `ncclMeshCommLower` then fixes each declaration-owned occurrence:
its issue pages, cuts, source-readiness words, and typed arrival/publication
relations. Each prepared occurrence owns fresh source-readiness words, including
occurrences which reuse the same payload bindings. Frames within that occurrence
use increasing thresholds. No runtime reset or allocation supplies this state.

The page-effect compiler projects those relations onto each numerical command's
actual reads and final writes. It solves all channels first, then emits the
command's source once. Independent completed-prefix domains share one metadata
page operand and the command's required observation/publication fences. CPU,
ANE, generated Metal, explicit native kernels, and layout copies retain their
source and access relations until this step finishes.

An input with no numerical producer is an explicit source input to its existing
transfer descriptor. Earlier remote arrivals become that descriptor's actual
prerequisites. Its existing presentation publishes the initial source prefix;
later numerical publishers retain the exact input-page dependencies when their
prefix can pass those initial pages. A returned remote value with no numerical
consumer contributes a terminal word to the recorder's existing nonblocking
completion observation. Numerical completion already covers consumed arrival
words. Neither case adds a numerical wait or publication dispatch.

Only then does `Program.prepare` bind numerical leaves, and
`ncclMeshCommEncode(comm, occurrence, commands, count, stream)` record the exact occurrence which
was lowered. The runtime consumes that common recorded graph. It does not recover
page relationships by decoding backend addresses or repeat the source solve.

The former `ncclMeshReductionFunction` callback and `_stream.Reduction` registration
are removed. They previously started another numerical compiler during native
recording. Summary merges now enter through the same `Reduction` source before
capture and differentiation. Native numerical providers contribute their ordinary
source declarations and access relations to that same lowering sequence.

## Activity in native fold leaves

Native fold source retains each incoming transport identity and its numerical
value cells. Coverage binds the actual active page to those cells before the
shared source solve. The same fold equation skips an inactive input's payload
observation and combine. A preceding COPY may also be inactive; its declared
producer relation then selects the reduction identity as the initial accumulator.
The mask does not become an arithmetic weight, and `expected` remains completion
metadata. Static folds specialize the predicate to true without an activity load.

Incoming availability, output overwrites of outstanding SEND pages, and result
publication now enter the shared page-effect solve. An arrival constrains reads;
a prior reader's completion constrains writes. These use the exact same byte
relations as numerical operands. A masked observation retains its active page
and cut, including the dependencies needed to read that mask. The compiler puts
those dependencies before the predicate; it does not join differently masked
inputs into one unconditional threshold.

The Metal and CPU fold bodies contain the combine equation only. They no longer
walk ready-word arrays, check overwrite counters, maintain completion state, or
publish their own prefixes. Their runtime cell tables retain element ranges and
numerical addresses. The shared emitter supplies the required page effects.

Common numerical cells now end at packet boundaries as well as work-width
boundaries. Each cell publishes its actual completed contribution prefix; a
regular multi-page address run no longer joins those arrivals solely because it
is contiguous. Physical fragments of one independent numerical value carry the
same activity. A byte mask which disagrees within that value does not define a
numerical contributor.

Native local input activity must be supplied by its numerical source equation;
masking an arbitrary outgoing transport does not declare that rank absent from
the fold. Coverage now enters `ncclMeshCall` before packetization, in the byte
coordinates of an original logical SEND. Its transaction identity survives
every routed hop and physical refinement. Active/count bindings use the same
occurrence slots as other page operands. There is no late occurrence-coverage
API which can silently leave a semantic cut inside an indivisible DMA span.

Python WRITE projection applies the same retained byte-order relation to payload
and coverage. A one-source/one-recipient WRITE keeps transaction zero; splitting
calls also transduces their source-call boundaries. A general collective rewrite
can change its logical SEND graph, contributor stages, and transaction ordinals.
It still needs a retained original-transaction-to-generated-transaction relation
before automatic coverage can traverse that rewrite. Reusing an old integer ID
or choosing a channel cannot supply that relation. Direct final-call declarations
and projected WRITEs are connected; generalized collective coverage synthesis
remains unfinished.

Source checks cover static/dynamic Metal fold bodies and generated CPU equations;
no payload run or performance improvement is claimed.
