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
  "inFlight": 2,
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

[`examples/coreml-models.py`](../examples/coreml-models.py) prepares a concrete
instance using Apple's Core ML Tools builder and compiler. The caller supplies
owners, per-section input and hidden widths, row count, block count and invocation
count. Each block has distinct dense weight blocks drawn from the specified NumPy
seed. The exported functions are matmul, GELU after the up-projection reduction,
and addition of the down-projection result to the earlier input. This is the
FFN residual algebra above, with no arithmetic or model interpretation added to
Mesh. The [README commands](../README.md#collective-implementation) instantiate
four blocks, eight projection stages and four independent invocation indices.
Its [128, 512] and [128, 1024] sections span four and eight 64-KiB transport chunks.

The chain binds one terminal consumer per final section through the existing
`TensorFunction.cpu` interface. That consumer reads two scalar samples and
submits their report to the process's main queue. Formatting and output occur
there, outside numerical execution. These terminal reports are not dependencies
of any projection, reduction or subsequent stage; there is no stage barrier or
result polling. They expose actual final outputs, not timing or a speedup claim.

A world-size-one placement assigns all operand owners to rank 0 and uses the
same functions and arithmetic graph. The distributed placement above divides
that work and communicates only the off-diagonal contributions. The paired run
below executes this graph. It does not establish throughput superiority or a
complete reusable-stream lifecycle. The finite index extent still cannot be rearmed.

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

## P1 paired Core ML run

On September 15, 2026, source `f253955` built and ran with the README commands.
The Mini used a fresh clone at `/tmp/mesh-import-p1`; its separately linked caller
was `/tmp/mesh-coreml-chain`, with the module's directory encoded as an rpath.
No loader environment variables were supplied. The other caller linked the same
module sources from the canonical laptop checkout. Both bridges used ABI 43,
65,536 pages of 16 KiB, four pages per transport chunk and one configured link:
M5 Max `rdma_en6` to M4 Pro `rdma_en3`. The deployed link configuration was
`p1/bridge-links` at `4aa0823`.

Core ML Tools 9.0 compiled 32 distinct projection models plus the activation and
residual models. The generator arguments were exactly the README example:
owners `[0, 1]`, input widths `[512, 512]`, hidden widths `[1024, 1024]`, 128 rows,
four FFN residual blocks, four invocation indices, two numerical workers and
seed 73. Both participants applied their supplied projections and consumed
reduce-scatter results at every stage. The native completion path reached all
eight final consumers. Their returned samples, in observation order, were:

```text
rank=0 part=0 index=0 first=-920.8167 last=-166650.17
rank=0 part=0 index=1 first=-921.58374 last=-166652.75
rank=0 part=0 index=2 first=-922.3454 last=-166655.39
rank=0 part=0 index=3 first=-923.0958 last=-166658.1
rank=1 part=1 index=1 first=878.302 last=32572.629
rank=1 part=1 index=3 first=883.70654 last=32573.672
rank=1 part=1 index=2 first=880.97437 last=32573.219
rank=1 part=1 index=0 first=875.62726 last=32572.254
```

After both complete result sets arrived, the demonstration clients received
SIGTERM. Ordinary bridge restarts completed their verbs-owner cleanup and returned
both nodes to ABI 43 with `client:0`. This is P1's operational evidence, not an
E2/E3 public-path performance measurement; output order does not prove speedup.

## G1 Gram and projection chain

[`examples/gram-chain.swift`](../examples/gram-chain.swift) imports Mesh and
supplies Accelerate functions. Its one `contract` composition takes execution
owners, output owners and byte counts, workers, a combine function, and a setup
factory for each product's function and operands. It declares two-operand calls
and passes their contributions to `reduceScatter`. It contains no matrix shape,
transpose convention, tensor arithmetic, or distinction between a Gram product
and a projection. The [canonical citation](algorithm-sources.md#programkernel_call)
identifies the panel decomposition being used.

The configuration supplies `owners`, `rows`, `latent`, `hidden`, `width`, `blocks`,
`count` and `workers`. Each dimensions list has the same length as `owners`, but
its entries need not match. Multiple partitions may share an owner; world size
does not determine an operation or partition count. The included configuration
uses eight blocks, four independent instances and unequal rectangular tiles.
Changing all owners to zero gives the same algebra and functions on one rank.

For block b, let `Z[i,l]` have shape `rows[i] × width[l]`, and let the registered
constant tiles be `A[l,h]`, `R[i,j]`, and `W[h,l]`, with the corresponding width,
hidden and latent dimensions. The source declares:

\[
\begin{aligned}
H_{ih} &= \sum_l Z_{il} A_{lh},\\
U_{jh} &= \sum_i R_{ij}^{\mathsf T} H_{ih},\\
Y_{ih} &= \sum_j R_{ij} U_{jh},\\
D_{il} &= \sum_h Y_{ih} W_{hl},\\
Z^{\mathrm{next}}_{il} &= Z_{il} + D_{il}.
\end{aligned}
\]

Thus each block is `Znext = Z + R(Rᵀ(ZA))W`. The loop constructs the entire
configured block sequence before `start()`. It is not a host loop that launches
one block and waits for its outputs before declaring the next.

| Value | Owner | Direct consumer |
|---|---|---|
| `Z[i,l]` | `owners[l]` | up projection and residual addition |
| `H[i,h]` | `owners[h]` | two-input `RᵀH` product |
| `U[j,h]` | `owners[j]` | two-input `RU` product |
| `Y[i,h]` | `owners[h]` | two-input down projection with `W[h,l]` |
| `D[i,l]` | `owners[l]` | residual addition with the original `Z[i,l]` |
| `Znext[i,l]` | `owners[l]` | next block's up projection |

For each product, `contract` gathers exactly its declared operands to that term's
execution owner, calls the supplied function, and declares a reduce-scatter to
the output owners. The gather is a list of sends of those operands; there is no
gather of the complete matrix. Every reduction result is a `TensorPart` passed
directly to the next product. In particular, `y[...]` is the down projection's
first operand without caller-side flattening, staging or tensor reconstruction.
Mesh's ordinary remote edges handle placements that differ; ABI 51 materializes
fragmented multi-chunk inputs in canonical storage for contiguous BLAS operands,
on the numerical worker. That copy is accounted for as layout work.

The bound numerical closures contain a BLAS call, a vDSP addition, or a vDSP
input ramp. They contain no allocation, presence read, completion query, wait,
or backend selection. Transposition and leading dimensions are fixed when
`product` is constructed. Weight initialization, shape/index expressions,
function factories and graph construction all execute before `start()`.
The driver submits the finite configured instances and retains the final parts;
it does not install a reporting callback or poll their contents.

Build with `make -C rdma gram-chain`. With a bridge matching the built module,
the caller commands for the included placement are:

```sh
rdma/gram-chain 0 2 /mesh0 examples/gram-chain.json
rdma/gram-chain 1 2 /mesh0 examples/gram-chain.json
```

The same program accepts explicit routes. `examples/gram-chain-star.json` places
the numerical work on ranks 0 and 2 and routes both directions through rank 1.
Configure physical links `0 ↔ 1 ↔ 2`; each rank reads the same placement:

```sh
rdma/gram-chain 0 3 /mesh0 examples/gram-chain-star.json
rdma/gram-chain 1 3 /mesh0 examples/gram-chain-star.json
rdma/gram-chain 2 3 /mesh0 examples/gram-chain-star.json
```

The JSON routes include both endpoints; the caller converts each path into a
`Placement.Edge` key and subsequent-rank list. `0 → 2` and `2 → 0` are separately
declared. Rank 1 has transfer bindings but no tensor functions, and starts no
numerical worker. Only the configured numerical owners submit local roots; the
relay never calls `submit`. Its received section is bound directly as the onward SEND
source. This configuration is source usage of T2, not a three-node run or a
performance measurement.

`examples/gram-chain-ring.json` uses four numerical owners with explicit directed
routes over `0 ↔ 1 ↔ 2 ↔ 3 ↔ 0`. Each node has two peers; opposite-rank traffic
forwards through a configured neighbor. It retains eight blocks and uses two
instances, four workers and unequal rectangular tiles with 512–896 rows. Several
inputs span multiple transport chunks at the current 64 KiB payload geometry.
The ordinary `contract` calls exercise received constants, contractions,
reduce-scatter, residual reuse and continued consumption across blocks. No
ring-specific numerical function, transport call, or result-handling path is added:

```sh
rdma/gram-chain 0 4 /mesh0 examples/gram-chain-ring.json
rdma/gram-chain 1 4 /mesh0 examples/gram-chain-ring.json
rdma/gram-chain 2 4 /mesh0 examples/gram-chain-ring.json
rdma/gram-chain 3 4 /mesh0 examples/gram-chain-ring.json
```

These commands describe source integration with matching bridges and configured
links; the four-node execution has not been run or measured.

The current configurations separate total `count` from resident `inFlight`.
The ring requests eight invocations over two resident slots; the other Gram
configurations request four over two. The Core ML configuration above does the
same eight-over-two submission. Their ordinary main driver retries immediate
busy admission and advances the label only on success. This is explicit caller
admission policy, outside tensor functions and dedicated transport threads; it
does not wait for a particular tensor or introduce a batch barrier.

The callers build and satisfy G1's source composition check. They have not been
run or measured. W6's native invocation path has a source ownership derivation;
the remaining receive/admission and result/recovery requirements leave G1 and N1
partial. These declarations do not establish repeated-execution correctness,
overlap, or E3 performance superiority.
