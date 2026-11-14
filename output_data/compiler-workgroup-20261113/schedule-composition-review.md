# Same-target schedule composition

Source checkpoint: `compiler/mesh_compiler/schedule.py`. This is the producer half of compilation-composition
requirements 2, 3, 6 and 7, paired with the driver's explicit target grouping and the whole-body target consumer.
This note records source review and focused source-stage checks, not a machine-checked semantic proof or a
numerical/performance result.

## Local contract

Input: the device-adjusted placed graph, its retained transfer endpoints, the existing global task/dependence/order
result, and an explicit original-processor-to-group operand. The driver groups by machine, selected engine/target
and full device; CPU storage alone does not identify an ANE target as CPU.

`schedule` keeps the original task IDs, resource lanes, cost operands and Plan. After its existing order is solved,
`assemble` copies a group's actual global nodes once. An internal SEND/RECV pair becomes its copied sender value;
genuine device copies and cross-group effects remain. Canonical `mesh_graph.contract` projects dependencies through
only the erased endpoints. It is not called when no endpoints are erased. There is no added serial-chain relation.

The result has group modules plus the original Plan. Module metadata contains original `members`, copied-node
`tasks` identities, copied-node `dependences`, and sender physical `received_strides` for remaining receives.
Placeholders and the original output/mutation tuple use the same copy correspondence. `argument_buffers` consumes
the group projection using its existing SCC owner. Driver maps result/device/engine keys to actual returned modules.

After substitution, the copied consumers of erased receives are passed to canonical `ir.views`; its existing
alias/destination rules update the changed use-def relation. This corrected a concrete loss of CAT destinations in
the linear example. No destination policy or view-name rewriting was added here. The source establishes equal
FakeTensor geometry: `placement.insert_send_recv` assigns the sender value to the receive, and `devices.on_devices`
updates it to the real inserted-copy value where needed. Substitution therefore does not introduce a new
non-viewable reshape. The checked examples inserted no new nodes during this refresh. A future transformation that
changes geometry and causes `ir.evaluated` to insert nodes must also expose those nodes' task correspondence; that
different transformation is not implemented or claimed here.

The target consumer retains member/source provenance in saved Targets. Current `archive.load` restores those
packages, with no FX graph/module reconstruction. A single compiled numerical entry can discharge its internal
task relation; surviving native effects still need their actual dependence/geometry operands and remain executor
work. This checkpoint does not certify those effects, bridge execution or mixed-target completion.

## Review and evidence

Same Claude Opus 5.5 session `dc4b988b-a100-4e17-8c80-ff3f90ac97e3`, explicitly selected and verified as
`claude-opus-5-5`. Work orders/logs: `compiler/schedule-compose.md`, `compiler/schedule-compose.inbox.md`,
`compiler/schedule.resume2.stream.jsonl`, `compiler/schedule.resume3.stream.jsonl`. Codex reviewed source/consumer
correspondence and directed three corrections: reuse unchanged edges, construct each group membership tuple once,
and refresh affected views through their canonical owner. Worker is held after the correction.

Source-stage runs use the actual Engines FFN/linear and TP models through export, placement, devices, schedule and
argument buffers. They do not run an alternate numerical evaluator:

- Engines, `ENGINES=cpu,cpu ROWS=2 K=32 N=48 LAYERS=1`: all four internal channels disappear in one group; exact
  inputs/outputs, task/lane provenance and sender-value substitution remain. The two expert branches gain no
  dependence path. Mixed grouping retains the two external channels and their physical strides.
- Linear: `mm_1` and `mm_2` acquire the two original CAT destinations through `ir.views`; mixed grouping keeps the
  external receive's destination. Other view/destination records remain unchanged.
- TP, `LAYERS=1 SEQ=16 DIM=64 HEADS=4 FFN=128 BATCH=1`: all 23 same-target channels disappear; retained task and
  dependency endpoints belong to the composed module, without stale view roots. No view refresh creates nodes.
- Worker `py_compile`/`pyflakes` and supervisor `py_compile`/named `git diff --check` pass. Scratch observations live
  under `/tmp/schedule-compose-check`; no permanent test harness was added.

These checks support this local representation/consumer contract. Native numerical execution, save/load execution,
transport behavior and throughput require their respective evidence; none is inferred from this checkpoint.
