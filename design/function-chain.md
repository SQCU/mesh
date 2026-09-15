# Supplied functions through reduce-scatter

The caller is [`examples/coreml-chain.swift`](../examples/coreml-chain.swift).
The importable interface is `Mesh` and `TensorFunction`; Mesh has no model-stage
API. The stage list below belongs to this caller and is realized before invocation.

For each stage, input i is owned by its configured participant. That participant
applies each supplied function Fij to Xi, producing contribution Cij. The caller
passes the contributions to `reduceScatter`, whose supplied Accelerate function
adds them at their output owners. Each output can then invoke its supplied G.

`Cij = Fij(Xi)`
`Yj = sum_i Cij`
`Xnext,j = Gj(Yj)` or `Yj` when G is absent.

For a linear tensor operator, Fij is its block from input section i to output
section j. Thus Yj is the corresponding section of T(X). No inverse, replicated
full input, or gathered intermediate is needed. G may implement a supplied
nonlinear local consumer. The caller is responsible for a mathematically valid
partition and useful placement; a transport fragment never changes that partition.

With two sections owned by ranks 0 and 1:

| Rank | Local numerical work | Outgoing contribution | Reduction consumed locally |
|---|---|---|---|
| 0 | F00(X0), F01(X0) | C01 to rank 1 | Y0 = C00 + C10 |
| 1 | F10(X1), F11(X1) | C10 to rank 0 | Y1 = C01 + C11 |

F00 and F01 use separate configured numerical workers, as do F10 and F11.
Each native completion publishes its own contribution. A Y0 completion can
issue G0 and the next uses of Xnext,0 while Y1 is unfinished. No node waits for
a global stage number to advance. The stage loop constructs this graph at setup;
it is absent from numerical execution. With D configured stages, this is the same
relation repeated D times, including the requested 8–100-stage compositions.

The input producers call `vDSP_vramp` into Mesh operands. The F and G calls use
supplied compiled Core ML models. The reduction calls `vDSP_vadd` directly on
resolved operand pointers. Those functions come from existing Apple libraries;
this caller and Mesh contain no replacement numerical kernel. The example uses
float32 model inputs and outputs and supports arbitrary contiguous tensor shapes.
Models are loaded only for work owned by this participant. Repeated paths reuse
the loaded model, while operand-specific views are prepared independently.

Run the built caller as:

```
rdma/coreml-chain RANK WORLD_SIZE REGION CONFIGURATION.json
```

The configuration has `count` invocation indices, `workers` numerical workers,
model feature names `inputName` and `outputName`, initial `inputs`, and `stages`.
Each operand describes its `owner` and `shape`. For every stage, `functions[i][j]`
is the path to the supplied compiled Fij model, `outputs[j]` is its output
section's shape and owner, and `finish[j]` is the path to Gj or null. The previous
stage's outputs determine the next stage's input layouts. Put each desired stage
in the list; Mesh does not infer a depth, architecture, kernel, or placement.

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
    "finish": ["G0.mlmodelc", "G1.mlmodelc"]
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

Native operand preparation now belongs to `TensorFunction`, so that same
function value is accepted by `call`, `map`, `reduce`, `reduceScatter` and
`allReduce`. The old special native `Mesh.map` overload is gone. Factories,
Core ML feature providers and output backings are prepared before execution;
the runtime calls the resolved CPU/Metal/prediction submission directly.
