# The compiler

## The turn

The operator, 2026-11-10:

> are we done doing cognitive restructuring wrt the conflict between 'working examples of outcomes' (laudatory, unportable, unextensible) and 'specifications' (transcendent, greenfielded if necessary, ever-more-extensible)? i still see a very basic need for the very basic and very literal compiler-which-does-the-meshproject's-equivalent-to-the-compact-counterfactual-lowerer to be described as a litearl and immediate task which can be worked on immediately without qualification and complex subtle nuance..

## The task

Write mesh's compiler at `compiler/` with the same data flow and assignment of stage responsibilities as the
reference program quoted in [data-flow.md](data-flow.md). That document quotes the operator's assignment:

> remmber that our goal here is to articulate what the data flow will be for the mesh modules that have the same role in their own data flow as this counterfactual example code.

The reference's interfaces are:

```
compile_module(module, args, processors, placement, cost) -> Scheduled
    export -> place -> on_devices -> schedule -> argument_buffers
Program(scheduled, functions)      runs it
lower(scheduled) -> functions      one function per processor
save(directory, scheduled), load(directory) -> Scheduled
```

Three things are mesh's and are added to it:

1. **A processor is a device of a machine.** `processors` is `{name: (machine, device)}`, the device a torch device or
   an engine named below.
   - M. Abadi et al., TensorFlow, arXiv:1603.04467, section 3, "Devices": "Device names are composed of pieces that
     identify the device's type, the device's index within the worker," and "an identification of the job and task of
     the worker"; "Example device names are "/job:localhost/device:cpu:0" or "/job:worker/task:17/device:gpu:3"."
   - A device may name an engine that is not a torch device: `(machine, 'ane')` is the machine's Neural Engine, reached
     through Core ML; "the device's type" is read as covering it. That reading is Claude's, made so that `examples/engines.py`, whose
     engines are `mps,cpu,ane`, is compiled by this compiler; it is not Abadi et al.'s statement and not an operator
     turn.
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

## The work

Claude implements this compilation flow in `compiler/mesh_compiler/`.

The steps, their order and what each one replaces are the eleven items of
[implementation-order-2026-11-09.md](implementation-order-2026-11-09.md).
