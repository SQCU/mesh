# Source output realization

SAME974 sessions75–77, terminal/HOLD. Only `placement.py` changed. Supervisor
AST and `git diff --check` passed; source SHA256
`18ec833bf191184d545c5ab272c42ee80f4e647643468ea437f83e8c548c602e`.

The existing `crossing-output-1.json` records a logical 16-byte broadcast
(5,8)/(0,1) paired with an 80-byte contiguous CAT by a scalar delta. The prior
684c7b42 shortcut removed a redundant CAT only. The current change retains the
original alias parents/operands before placement, consumes canonical
`mesh_graph.postorder` and `forward`, and realizes shared source bases once from
the existing placed result and whole-value cache. Original input aliases keep
that input. Mutation outputs keep their original destination path. Other outputs
retain the original output transfer.

Required fresh-base geometry follows Torch2.14
`_inductor.ir.ExternKernel.require_strides`: compact zero-stride dimensions,
copy into the original geometry, expand. Its storage is allocated at the source's
full extent, with the source offset/strides applied by `aten.as_strided`. Existing
CLONE.out/typed copy owns numerical work. Canonical FX DCE removes unused paths
before transfers while retaining source effects and input-part endpoints.
No new native interface, archive record, or runtime decision was added.

Root reviewed the source approach. The actual native pairing remains to be
observed using the approved existing Crossing column-shard caller after the
matching v39/source release. No native result is claimed here. Command/order and
worker records are in the metal workgroup `frontends/torch/materialized-output-*`
and `session-75.jsonl` through `session-77.jsonl`.
