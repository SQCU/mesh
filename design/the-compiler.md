# The compiler

## The turn

The operator, 2026-11-10:

> are we done doing cognitive restructuring wrt the conflict between 'working examples of outcomes' (laudatory, unportable, unextensible) and 'specifications' (transcendent, greenfielded if necessary, ever-more-extensible)? i still see a very basic need for the very basic and very literal compiler-which-does-the-meshproject's-equivalent-to-the-compact-counterfactual-lowerer to be described as a litearl and immediate task which can be worked on immediately without qualification and complex subtle nuance..

The operator, 2026-11-13, on the scope of the integration work:

> alright. lets do a more holistic review. are there any serious documentary or practical gaps in the repository scopes which would stop us even from a moment from claudeminding a bunch of subagents to do narrowly focused work on the already-designed already-cited compiler integration work (which means retopologying a weird and strange data flow (which means not reusing code or idiom from any of the related existing code!!)) very clearly circumscribed by the chagne request documentation and related memoranda?

The operator, continuing the same review:

> "- One placement edit remains uncommitted. Its owner must review and account for it before overlapping edits; other scopes can proceed." does this edit come from a failcommit which was operationally ignoring or supervening the literal work assignment? you can revert and throw out failcommits from claudes which were lying to each other about what they were assigned or 'allowed' to do. the literal work assignments are *really simple* and do not imply labrythine structures of hundreds of thousands of lines of code. if you look at the 'short list' of features the user has been 'asking' for over the last month, you will note that they imply a small library for the ahead of time scheduling of memory (so that a 'bridge app' can jointly optimize register and buffer use by multiple programs with aot-knowable lifecycles and 'bursty' memory use. so that functions can be typed and prologue-fused or epilogue-fused with each other or with collective communications operations *without respect for* their internal numerics in black-box or anonymous form. so that functions which take tensor-typed inputs and outputs can be 'load balanced' s.t. very little aot compilation work is invalidated or needs to be recompiled to iteratively optimize transport routing or tensor parallelism placement if either of those properties are parametrically optimized between collective-function-calls.). this repo is unusually *clean* in what it asks for; there are almost no np hard problems! there are almost no problems which require just in time compilation. there are almost no problems whihc require exotic type systems or new approaches not in literature. therea re almost no problems which require defensive programming, guards, or asserts. there are almost no problesm which require ingenuity, craft, or iteratively testing code to figure out whether it'll work or how to write it. it's almost like someone completely designed an approach and fed it to the subagents one chunk at a time with total omniscience over how the program must be eventually implemented to reach some specific performance targets?

## The task

Write mesh's compiler at `compiler/` with the same data flow and assignment of stage responsibilities as the
reference program quoted in [data-flow.md](data-flow.md). That document quotes the operator's assignment:

> remmber that our goal here is to articulate what the data flow will be for the mesh modules that have the same role in their own data flow as this counterfactual example code.

Implement the requests' cited algorithms and data structures in that flow. The existing mesh and metal-microbench
implementation bodies and their idiom are the code to replace. The [change-request standard](change-requests.md)
states the source vocabulary and how to revise naming defects while continuing the assigned work.

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
