# Supplied functions through reduce-scatter

The caller is [`examples/coreml-chain.swift`](../examples/coreml-chain.swift).
The importable interface is `Mesh` and `TensorFunction`; Mesh has no model-stage
API. The stage list below belongs to this caller and is realized before invocation.

For each stage, input i is owned by its configured participant. That participant
applies each supplied function Fij to Xi, producing contribution Cij. The caller
passes the contributions to `reduceScatter`, whose supplied Accelerate function
adds them at their output owners. Finishing calls select named input partials
from that result and from earlier values, then publish any number of outputs.

`Cij = Fij(Xi)`
`Yj = sum_i Cij`
`Xnext = G(selected earlier values, selected Y)` or `Y` when no finishing calls are declared.

For a linear tensor operator, Fij is its block from input section i to output
section j. Thus Yj is the corresponding section of T(X). No inverse, replicated
full input, or gathered intermediate is needed. G may implement a supplied
nonlinear consumer, residual update, or function returning several sections.
The caller is responsible for a mathematically valid
partition and useful placement; a transport fragment never changes that partition.

The source iterates the configured input and output lists independently; neither
list nor world size has a two-participant restriction. Every remote contribution
uses its two declared endpoints and configured peer channel. With two sections
owned by ranks 0 and 1, one instance of this general relation is:

| Rank | Local numerical work | Outgoing contribution | Reduction consumed locally |
|---|---|---|---|
| 0 | F00(X0), F01(X0) | C01 to rank 1 | Y0 = C00 + C10 |
| 1 | F10(X1), F11(X1) | C10 to rank 0 | Y1 = C01 + C11 |

F00 and F01 use separate configured numerical workers, as do F10 and F11.
Each native completion publishes its own contribution. A Y0 completion can
issue a G0 reading Y0 and an earlier X0, and the next uses of Xnext,0, while Y1 is unfinished. No node waits for
a global stage number to advance. The stage loop constructs this graph at setup;
it is absent from numerical execution. With D configured stages, this is the same
relation repeated D times, including the requested 8–100-stage compositions.

An FFN residual block can use two such stages. The first computes the distributed
up projection and a supplied activation; the second computes the down projection
and a supplied residual consumer:

`Uj = phi_j(sum_i Aij(Xi))`

`Xnext,j = H_j(sum_i Bij(Ui), Xj)`.

For gated FFNs, the first stage has separate reduced gate and up sections, and
the supplied phi reads both. No activation, contraction or residual arithmetic
is implemented by Mesh. Xj is another declared input of H_j: the same registered
partial feeds its up-projection users and this later use. Mesh retains it until
those users complete, without a caller stamp or release. For block b represented
by stages 2b and 2b+1, the original X is group 4b, the activated U is group 4b+2,
the down result is group 4b+3, and Xnext is group 4b+4. Repeating this relation
does not introduce a block-wide completion event.

The input producers call `vDSP_vramp` into Mesh operands. The F and G calls use
supplied compiled Core ML models. The reduction calls `vDSP_vadd` directly on
resolved operand pointers. Those functions come from existing Apple libraries;
this caller and Mesh contain no replacement numerical kernel. The example uses
float32 model inputs and outputs and supports arbitrary contiguous tensor shapes.
Models are loaded only for work owned by this participant. Repeated paths reuse
the loaded model, while operand-specific views are prepared independently.
Finishing inputs go through an explicit `gather` to the supplied call owner;
outputs go through an explicit `scatter` to their declared owners. Those calls
retain their meanings for any input count, output count and world size.
When several calls request the same earlier partial at the same destination and
queue, setup shares the received value. Each reader retains its own input use;
the payload is sent once rather than once per reader. The construction history
and delivery map are discarded before numerical work begins.

Run the built caller as:

```
rdma/coreml-chain RANK WORLD_SIZE REGION CONFIGURATION.json
```

The configuration has `count` invocation indices, `workers` numerical workers,
model feature names `inputName` and `outputName`, initial `inputs`, and `stages`.
Each operand describes its `owner` and `shape`. For every stage, `functions[i][j]`
is the path to the supplied compiled Fij model and `outputs[j]` is its reduced
section's shape and owner. `finish` is a list of supplied calls. Each call names
its `model`, `owner`, `worker`, named `inputs`, and named `outputs` with operand
shapes and destination owners. An empty list uses the reduced sections directly.
The previous stage's outputs determine the next stage's input layouts. Put each desired stage
in the list; Mesh does not infer a depth, architecture, kernel, or placement.

An input names a `group` and `part` in the caller's construction history. Group 0
is the initial input list. For zero-based stage d, group 2d+1 contains its reduced
sections and group 2d+2 contains its finishing-call outputs, in declaration order.
A finishing call can read an earlier call's outputs in that group, as well as any
earlier group. These indices resolve before `start()` into ordinary `TensorPart`
arguments. The history is discarded before invocation; neither group numbers nor
the stage/call configuration are interpreted by the numerical workers.

This is one stage entry illustrating the configuration fields:

```json
{
  "count": 8,
  "workers": 2,
  "inputName": "x",
  "outputName": "y",
  "inputs": [
    {"owner": 0, "shape": [32, 5324]},
    {"owner": 1, "shape": [32, 5324]}
  ],
  "stages": [{
    "outputs": [
      {"owner": 0, "shape": [32, 5324]},
      {"owner": 1, "shape": [32, 5324]}
    ],
    "functions": [["F00.mlmodelc", "F01.mlmodelc"],
                  ["F10.mlmodelc", "F11.mlmodelc"]],
    "finish": [
      {
        "model": "H0.mlmodelc", "owner": 0, "worker": 0,
        "inputs": [
          {"name": "value", "group": 1, "part": 0},
          {"name": "skip", "group": 0, "part": 0}
        ],
        "outputs": [{"name": "y", "operand": {"owner": 0, "shape": [32, 5324]}}]
      },
      {
        "model": "H1.mlmodelc", "owner": 1, "worker": 0,
        "inputs": [
          {"name": "value", "group": 1, "part": 1},
          {"name": "skip", "group": 0, "part": 1}
        ],
        "outputs": [{"name": "y", "operand": {"owner": 1, "shape": [32, 5324]}}]
      }
    ]
  }]
}
```

Those paths are caller inputs, not model artifacts supplied by this repository.
The illustration is not an executed eight-stage experiment: `count` describes
independent invocation indices, while `stages.count` describes depth. The source
accepts the full configured stage sequence without rebuilding functions between
indices or restricting tensor size to a transport request. The concrete shapes
shown partition a [32, 10648] tensor into two 681,472-byte operands; transport
chunk count remains internal.

A world-size-one placement assigns all operand owners to rank 0 and uses the
same functions and arithmetic graph. The distributed placement above divides
that work and communicates only the off-diagonal contributions. This describes
actual source dependencies and work ownership. No executed model artifacts,
throughput measurement, speedup claim, or complete reusable-stream lifecycle is
established by this change. The finite index extent still cannot be rearmed.

Native operand preparation belongs to `TensorFunction`, so that same
function value is accepted by `call`, `map`, `reduce`, `reduceScatter` and
`allReduce`. The old special native `Mesh.map` overload is gone. Factories,
Core ML feature providers and output backings are prepared before execution;
the runtime calls the resolved CPU/Metal/prediction submission directly.
`TensorFunction.prediction(model, inputs:…, outputs:…)` takes named view factories
in operand order. Its per-index feature provider reads the realized operand array
and selects each input's prepared feature value independently. Output options use
the local result bindings prepared for that index. No Cartesian product of input
addresses, feature-value creation or output-dictionary construction is needed
at invocation. The previous raw prediction-provider callback is removed.
