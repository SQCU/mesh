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
2. **A channel between two machines is carried by the bridge.** Where a `send` and its `recv` are on processors of
   different machines, `send` writes the tensor's pages to the bridge and `recv` returns a tensor over the pages
   received.
3. **A placement also shards.** The placement operand gives a tensor's sharding over machines. A collective that the
   sharding requires is `send` and `recv` nodes that `place` adds.

## The work, in order

1. `compiler/` holds `ir`, `export`, `placement`, `devices`, `schedule`, `driver`, `runtime`, `archive` and the check.
   With processors on one machine's CPU and GPU, every case of the check passes.
2. Processors on several machines: items 1 and 2 above. The check's cases pass with their processors on two machines.
3. Sharding: item 3 above. `examples/tp.py`, `cp.py`, `ep.py` and `pl.py` are compiled by `compile_module` and run by
   `Program`.
4. `lower` for Metal: a function of `Scheduled` that returns one function per processor.
5. `rdma/mesh_compile.py` and metal-microbench's `metal_recording`, `metal_lower`, `metal_ir` and `metal_archive` are
   deleted when no example calls them.

Order 1 starts now.
