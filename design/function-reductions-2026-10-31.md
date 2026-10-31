# Supplied reduction functions

A reduction is a function supplied to the compiler. `Reduction` captures a
two-input combine and an identity with an explicit scalar type and value shape.
The callable runs while its source is captured. Execution uses ordinary commands
over the pages chosen for the collective.

`element_shape` names one complete value. A value can contain several fields, as
in a stable softmax summary. The transfer compiler cuts only between values. It
passes each selected cell's ordered source pages and output pages to the same
numerical compiler used by `Function`.

The default keeps rank order and a left fold. `reassociate=True` permits a new
parenthesization; `reorder=True` permits a new leaf order. These are declarations
about the supplied function. Floating-point callers choose whether they permit
the resulting change in rounding. Empty contribution sets use the identity.
One contribution uses the same page placement equations as any other identity
function.

```python
from torch_mesh.reduction import Reduction

combine = Reduction(torch.add, 0, dtype=values.dtype,
                    element_shape=tuple(values.shape[1:]),
                    reassociate=True, reorder=True)

def source(values):
    return combine(values, group)
```

This source belongs to an ordinary `Function`. `root` chooses a rooted reduction;
`counts` supplies reduce-scatter counts in independent values. With neither,
the result is an all-reduce. The complete preparation is in
[examples/function_reduction.py](../examples/function_reduction.py).

`reduction.sum`, `reduction.maximum`, and `reduction.partials` supply ordinary
functions through this interface. The Python attention-summary declaration now
uses `partials`; it has no private numerical collective implementation.

The C preparation contract is `ncclMeshReductionFunction`. Its callback receives
resolved page cells and the enclosing recording stream. The callback contributes
ordinary CPU or Metal commands to that same recorder. Source page versions,
including reads of a producer declared later, go through its normal dependency
solver. Required pages are retained by that function. No reducer callback,
operator lookup, shape decision, or source capture remains in the payload run.

The compiler retains a numerical template for each value shape, contributor
count, and source/destination storage-alias relation. Binding it to another
occurrence changes its page operands. Compiled index
readers and normal scalar fusion apply to the supplied combine just as they do
to the rest of the function.

This contract exposes the forward combine. A source that differentiates a custom
distributed reduction also needs the derivative's distributed page relations;
the declaration does not infer them from an associative forward claim.
