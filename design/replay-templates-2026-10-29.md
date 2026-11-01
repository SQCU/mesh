# Reusing captured source and native records

Updated 2026-10-31.

A reusable function retains its source graph, operand descriptions and configuration.
`Function.capture` keeps that graph. `Function.repeated` composes declared occurrences
without recapturing the Python function. `Function.declare_graph` uses the same
ownership, placement, command, page-effect and backend lowering as a first declaration.
Each realization binds the current pages and transport occurrence before execution.

Native recorder and replay objects still hold the commands for one prepared function.
Arithmetic libraries, pipeline declarations and pipeline caches remain reusable. Those
objects do not authorize replaying an earlier occurrence's transport counters.

The command-snapshot persistence interface was removed on 2026-10-31:

- `_stream.save` and `_stream.load`, and the unused `DNN.bound` and `DNN.load` methods;
- `MetalScheduleTemplate`, `MetalScheduleInstantiate`, and their file read, write and
  archive helpers;
- snapshot-only script/plan tracking, host/crossing serialization declarations and
  CPU/ANE reconstruction from serialized addresses.

The removed loader bound numerical commands from a snapshot, then lowered a fresh
transport occurrence. Saved effect tables still named the former occurrence's words,
thresholds and cuts. Changing buffer addresses could not fix those relations. A fresh
route can also change the number of cuts and contributors, so replacing counters alone
would leave the numerical schedule wrong.

The compiler now keeps source and page relations through effect solving, then binds
native commands. CPU workspace, ANE arguments and exact access metadata retain their
ordinary prepared lifetimes; they are not serialized to make another executor.

Persistent source-template save/load is not supplied by this change. A future format
must retain the source declarations and explicit configuration needed by the same
lowering sequence. It must not restore transport state from a completed function.
