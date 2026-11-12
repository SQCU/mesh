# The compiler

## The turn

The operator, 2026-11-10:

> are we done doing cognitive restructuring wrt the conflict between 'working examples of outcomes' (laudatory, unportable, unextensible) and 'specifications' (transcendent, greenfielded if necessary, ever-more-extensible)? i still see a very basic need for the very basic and very literal compiler-which-does-the-meshproject's-equivalent-to-the-compact-counterfactual-lowerer to be described as a litearl and immediate task which can be worked on immediately without qualification and complex subtle nuance..

## The task

Write mesh's compiler. It is the program quoted in [data-flow.md](data-flow.md), written in mesh at `compiler/`: the same
modules, the same functions with the same arguments and results, and the same two types, `Exported` and `Scheduled`.

```
compile_module(module, args, processors, placement, cost) -> Scheduled
    export -> place -> on_devices -> schedule -> argument_buffers
Program(scheduled, functions)      runs it
lower(scheduled) -> functions      one function per processor
save(directory, scheduled), load(directory) -> Scheduled
```

Three things are mesh's and are added to it:

1. **A processor is a device of a machine.** `processors` is `{name: (machine, torch device)}`.
   - M. Abadi et al., TensorFlow, arXiv:1603.04467, section 3, "Devices": "Device names are composed of pieces that
     identify the device's type, the device's index within the worker," and "an identification of the job and task of
     the worker"; "Example device names are "/job:localhost/device:cpu:0" or "/job:worker/task:17/device:gpu:3"."
   - A device may name an engine that is not a torch device: `(machine, 'ane')` is the machine's Neural Engine, reached
     through Core ML; "the device's type" is read as covering it. The values of such a processor that `export`,
     `on_devices` and `schedule` handle are host tensors: `Scheduled.devices` holds the host for it and
     `Scheduled.engines` is `{processor: engine}`, which the archive saves and loads. The engine is where the
     processor's function runs.
2. **A channel between two machines is carried by the bridge.** Where a `send` and its `recv` are on processors of
   different machines, `send` writes the tensor's pages to the bridge and `recv` returns a tensor over the pages
   received.
   - Abadi et al., section 3.2.2: "Any cross-device edge from x to y is removed and replaced by an edge from x to a new
     Send node in x's subgraph"; "At runtime, the implementations of the Send and Receive nodes coordinate to transfer
     data across devices." Section 3.3: "Distributed execution of a graph is very similar to multi-device execution.";
     "Send/Receive node pairs that communicate across worker processes".
   - That the transfer is of pages, written by one side, is the operator's (session 15c469fa, 2026-10-28): "the data
     model is so important that no previous code must run or needs to be run until the one-sided write
     unified-use-of-pages approach has been imposed upon the source."
3. **A placement also shards.** The placement operand gives a tensor's sharding over machines. A collective that the
   sharding requires is `send` and `recv` nodes that `place` adds.
   - The sharding and the collectives it requires are those of Xu et al., GSPMD, 2021, as
     [unequal-shards-change-request.md](unequal-shards-change-request.md) quotes them, for example section 4.5: "It uses
     AllGather to replicate data along sharded dimensions".
   - [aot-compilation-change-request.md](aot-compilation-change-request.md), A4: "The compilation must lower every
     collective A2 inserts to sends and receives of pages."

## The work, in order

1. `compiler/` holds `ir`, `export`, `placement`, `devices`, `schedule`, `driver`, `runtime`, `archive` and the check.
   The check is `~/schedule-emit` `samples.py` at f41c64a, brought in as `compiler/samples.py`. With processors on one
   machine's CPU and GPU, every case of the check passes. (Done: mesh ce2e8ac9; 22 cases match eager execution.)
   - The graph functions `schedule.py` uses (the rank and the list scheduling of `heft_placed`, the slot search of
     `earliest_slot`, the cycle test of `argument_buffers`) are functions of `rdma/mesh_graph.py`, given the caller's
     operands, and `schedule.py` calls them. The operator: "the literal graph algorithms header which was introduced
     ages ago in the repository, and is the only sensible place to put graph algorithms".
2. Processors on several machines: items 1 and 2 above. The check's cases pass with their processors on two machines.
   (Written: mesh 6d4f007b, 1a3cfe60. At dcc51b67 the 22 cases that have two or more processors pass with two processes whose
   entries the check carries between two regions of the bridge's layout; the 33 is the count on one machine. The run under two bridges is
   `MESH_GROUP=<run> python samples.py --machines <A> <B>` on each machine, and has not been made.)
3. Sharding: item 3 above. `examples/tp.py`, `cp.py`, `ep.py` and `pl.py` are compiled by `compile_module` and run by
   `Program`. (Done: mesh 297c3cda, 8e78b145, 16681b92, 3d51955d, 0d70ec49. The four examples match eager execution on
   two GPU processors, worst relative error 2.3e-06; 11 sharded cases of the check pass on every path.)
4. `lower` for Metal: a function of `Scheduled` that returns one function per processor.
   - Core ML: `lower_coreml.lower` returns a function for each processor whose engine is `ane`. The processor's graph,
     in scheduled order, is cut at each `send`, `recv` and copy to another device, which the function calls as torch
     does; after `lower_metal.lower` no copy to the Metal device is left in it. Each run of nodes between two cuts is
     one ML Program, written by the module as the protobuf of coremltools 9.0 `mlmodel/format/MIL.proto` and
     `Model.proto` (coremltools is not imported), compiled by `MLModel compileModelAtURL:` when `lower` runs and loaded
     with `MLComputeUnitsCPUAndNeuralEngine`.
     - State that no output mutates is a constant of the program, and a node whose operands are all constants is
       evaluated when `lower` runs. A floating value is float16 inside the program; a float32 tensor that enters or
       leaves it is converted by a `cast` operation of the program.
     - An argument is an `MLMultiArray` over the caller's pages (MLMultiArray.h: "Creates the object with existing data
       without copy."). A result is a tensor of its node's compiled shape and strides, and its pages are proposed as
       its output backing; MLPredictionOptions.h: "The framework may not use the specified backing object and instead
       allocates one by itself if the outputBacking dictionary doesn't contain the entry for the feature name, the
       model doesn't support the user allocated buffers, or in the batch prediction mode." Where it does not, the
       result is transferred into those pages (`transferToMultiArray:`) and the function's `transferred` names the
       model.
     - Where an operation runs is Core ML's choice. The function's `plan()` is MLComputePlan's preferred and supported
       devices per operation (MLComputePlan.h: "The anticipated compute devices that would be used for executing an ML
       Program operation").
     - The check's path `coreml` runs every case that has an `ane` processor and passes at a relative error of 2^-6
       against eager float32 where its other paths pass at 1e-4; the case of float16 values, `examples/engines.py`'s
       model, passes at 2^-6 on every path. Float16 has an 11-bit significand, a unit roundoff
       of 2^-11. A contraction of n terms accumulates at most n roundoffs relative to the sum of its terms' magnitudes
       (N. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., section 3.1), and the square root of n
       where the roundoffs' signs are independent; the widest contraction of the cases has 1024 terms, and 32 * 2^-11
       is 2^-6. The check prints, per case and `ane` processor, the plan's device for each kind of operation.
   - `driver.lowered(scheduled, lowerings)` applies the caller's lowerings in order, `(lower_metal.lower,
     lower_coreml.lower)`. Each lowers its own processors and returns the others' GraphModules, which the next one is
     given: `lower_metal.lower` returns them without their copies to the Metal device, whose function takes host pages,
     so it is first (design/torch-source-lowering-change-request.md, item 1: "no torch Metal kernel is ever submitted or
     recorded again."). A processor no lowering takes keeps the last GraphModule returned for it.
5. `rdma/mesh_compile.py` and metal-microbench's `metal_recording`, `metal_lower`, `metal_ir` and `metal_archive` are
   deleted when no example calls them.
   - `examples/engines.py` is compiled by `compile_module` and run by `Program` (mesh 5098de5f), its functions those of
     `lowered(scheduled, (lower_metal.lower, lower_coreml.lower))`: `lower_metal.lower` lowers a float16 product, its
     cooperative tensor of the product's type, and `aten.gelu` with `approximate='tanh'` (torch 2.15,
     `torch/nn/modules/activation.py`, GELU: "When the approximate argument is 'tanh', Gelu is estimated with:" 0.5 *
     x * (1 + Tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))). Its `cpu` processor runs its GraphModule.
