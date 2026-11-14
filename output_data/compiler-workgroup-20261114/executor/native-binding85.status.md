# Indirect argument operand handoff (native-binding85): SOURCE-READY, HOLD

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5.
- Scope: capture arguments, Storages use, loaded Contents. Source only.
- No build, run, SSH, stage, commit or binary replacement. `metal_recording.m` is unchanged.
- This status replaces native-binding84 §1 and its "Concrete observation".

## Live snapshot (args67, from native84's `distributed/dispatch3-snapshot.stdout`)

- Row 0/call 0/function 0/instance 0, dispatch 3: kind `METAL_IR_INDIRECT`, kernel 3, `arguments` 67, 22 bindings,
  24 effects, numeric kernel 0, members 0..20, later 1.
- Value 67 is a TEMPORARY U32 root of 12 bytes, with no data, buffer or transfer. Its `word` is `METAL_IR_DATA` (the
  `Value` default, metal_ir.h:639).
- The shader bindings omit 67. Effect: a 12-byte read of 67.
- Producer: `Transport` (metal_transport.mm:338–374) sets `later.arguments = arguments` and
  `added {arguments, 0, 12, read}`, and Artifact branch C turns that into the view. `compaction.repeat = k.repeat` and
  `later = k`, so the compaction and the later pass share one innermost region.

## 1. Capture: the operand in the lookup producer (`metal_capture.m` `-prepare:`)

- The native84 exception after a failed lookup is gone. The effect loop is byte-identical to HEAD: lookup, then raise
  on `k == slots`, then `keyed.key = slotOf[k]`.
- After the binding loop, in the existing `[made->bindings + 1]` arrays, an INDIRECT dispatch with
  `made->arguments < V` registers one more operand: `slotOf = command.slots`, `accessOf = 1`,
  `storageOf = info[made->arguments].storage`.
- Dispatch 3's read of root 67 now resolves through the ordinary lookup.
- Unchanged: `command.slots`, `command.bindings`, `read`/`write`, `capture->buffers`, and `MetalLaterAdd`'s
  `record->arguments` (metal_capture.m:141–143).

**Domains.**
- **Logical command operand key** (`slotOf`, `keyed.key`): shader slots `given->slot` in `[0, command.slots)`, then
  `command.slots` for the arguments.
- **Physical replay index** (metal_capture.m:1019–1037): per repetition, `[pass->slot[0..pass->slots)…, arguments,
  step]`, stride `pass->slots + 2`. `pass->slots` counts only the active (access ≠ 0) shader slots, so the arguments
  sit at physical index `pass->slots`. `MetalNativeIndirect` reads `bound[e·stride + slots]` (metal_program.m:225).
- `command.slots` is a logical key, not that physical index. The two agree only when every shader slot is active.
- No live reader of the keyed effects exists (native84). The key is consistent with the slot domain, not read.

## 2. Storages: the same operand through the existing usage (`metal_ir.mm` `Storages`)

- The DATA/STEP body of the binding loop is now the lambda `use(value, access)`, without any change in what it does.
  It runs per binding with `given.access`, and once per INDIRECT dispatch as `use(made.arguments, 1)`, right after the
  existing alignment, at that dispatch's own `orderEarliest`/`orderLatest`.
- Through `use`, the read reaches the canonical root `values[arguments].storage`: `used` (`lastUse[root]`, `points`)
  and the `held.begin`/`held.end` and `stored`/`loaded` updates.
- **979 repeat projection.** `domain` now takes a value. Its per-binding body is the lambda `touch(value, access)`,
  also called as `touch(made.arguments, 1)` for an in-span INDIRECT dispatch.
- The region's touched-domain projection consumes this actual read just as it consumes shader reads; the source
  producer keeps compaction and later in the same innermost region. No new region-boundary scenario is assumed.
- Nothing is added beyond that: no allocator, Function or Local change, no new solve, no shader argument, and no
  blanket lifetime.

**Correction of native84.** native84 said the root's `end` "necessarily stays at the compaction". That was wrong.
- Before this unit, the root's only binding was the compaction's write. So `loaded` stayed false, and the post-loop
  set `storage.end = last`.
- `lastUse[67]` was still only the compaction's `orderLatest`. That is wrong for its consumers below.
- Now `end` and `lastUse` include the later pass's `orderLatest`. When that is the root's latest point, `loaded` is
  true and `end` is that position, not `last`. The actual positions of dispatch 3 and its compaction were not read.

**Consumers reviewed (unchanged):**
- **capture** (metal_capture.m:568–584): `used(1 + v, last[v])`, the End record, at the later pass's `orderLatest`.
  It feeds `handover` (≈1081).
- **native.py `storages`**: storage `begin`/`end` becomes the existing `mesh.Value` allocation interval.
- **`metal_ir_place` access pairs** (metal_ir.mm ≈4625): these already `Touch` the placed arguments of each later
  INDIRECT dispatch.
- **`p.lastLater`**: unchanged.
- **archive** (`x.Vec(p.storages); x.Vec(p.lastUse)`): unchanged.

## 3. Loaded CONSTANT contents (preserved)

`metal_ir_loaded` keeps the accepted line: `!Contents(*p, k).data && !p->values[v.storage].buffer`.

Evidence (`frontends/torch/observe/static-constant-construction-1.json`):
- typed root 4 is CONSTANT, data 2.0, 4 bytes;
- its logical scalar view 5 is CONSTANT, storage 4, data NULL.

`Contents` returns the view's data or else its canonical root's.

## Source standard

Each added comment is only a citation: `metal_ir.h metal_ir_dispatch \`arguments\`` and this file. The existing binding
comment stays above its loop.

## Checks

Strict `-fsyntax-only -Wall -Wextra -Werror`, using the Makefile's actual flags, rc=0 on:
- `metal_capture.m` and `metal_program.m`;
- all 8 `METAL_IR_SRCS` (`TINT_CFLAGS`);
- `tools/native_bench.m` and `tools/e2b_replay.m` (`PROVIDER_CFLAGS`).

`metal_recording.m` is unchanged and was not compiled.

## Not established

1. **Replay timing.** Does replay run a later pass within its `[orderEarliest, orderLatest]`? The dispatch's own
   shader bindings already rely on that window, so the arguments use it the same way. This was not verified here.
2. **Native later.** The native later pass (`METAL_IR_TASK`, the `mesh_arguments` view of the native entry) is a
   different dispatch kind. Both guards are `METAL_IR_INDIRECT`, and this path was not inspected.

**Changed (against HEAD):** metal-microbench `metal_capture.m` (+6) and `metal_ir.mm` (`Storages` refactor plus
operand, and the `Contents` line), 40 insertions and 25 deletions. HOLD for review, checkpoint and a matched build.
