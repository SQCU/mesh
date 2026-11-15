# Native131 → 132: prepared-unit handoff (`b3273f38` lifecycle), exact state

Claude Opus 5.5, 2026-11-15. metal-microbench working tree on `51988113`. From now on 131 does not edit these hunks.

## Committed and mine (compile-unit scheduling)

These rode in `51988113`: 135's commit took them from the shared index; the content is exactly what I staged and
checked.
- `Emit`: each unit's first kernel is its one compiler (`compiledBy`, `unitFirst`).
- `UpstreamTask`: a unit's members are taken across passes.
- `Configure`: no compile for unit members.

Already in `c384f1df`: `metal_ir_upstream.unit`, `Upstream.unit`, the append renumbering, `UpstreamRunner`, and one
`TaskLibrary` per unit.

## In the working tree, untouched, handed to 132 as is (uncommitted)

- `metal_ir.h`: `int metal_ir_upstream_unit(struct metal_ir_program *, uint32_t unit, const char *prepare,
  const char *release);` (after `struct metal_ir_upstream_parameter`), and `Program::Upstream`
  `std::string prepare, release;` (after `unit`).
- `metal_ir.mm`: the `metal_ir_upstream_unit` definition, which sets `prepare`/`release` on every upstream of that
  unit and returns ENOENT where none has it.
- `metal_lower.mm` `UpstreamTask` (~1716): `if (!u.prepare.empty()) { exported.push_back(u.prepare);
  exported.push_back(u.release); }`, the unit library's export of both entries. This is the prepared-export boundary
  inside my `NativeBuild` call: keep it in your commit, or say so and I will commit it after your fields land.
- `metal_archive.mm` `UpstreamModule` (~421): `x.Pod(u.unit); x.Str(u.prepare); x.Str(u.release);`. `u.unit` is
  mine; it survives Records only once this line is committed.

## Removed by 131 before the stop arrived (in answer to the reassignment), never committed

Exact text, for 132 to re-apply as it chooses:
- `metal_ir.h`, after `struct metal_ir_upstream_parameter`:
  `#define METAL_IR_UPSTREAM_PREPARED 0xfffffffdu` (= `LITERT_MESH_UNIT_PREPARED`).
- `Program::Upstream`: `uint32_t prepared = UINT32_MAX;` (the unit's prepared operand's value, shared by its
  upstreams).
- `NativeBinary`: `std::string prepare, release; uint32_t value = UINT32_MAX; void *object = nullptr;
  void (*destroy)(void *) = nullptr;`. **`object` clashes with `NativeBinary::object` (the bytes); rename it, for
  example `preparedObject`.**
- `int Prepare(Program &p, NativeBinary &library, char *error, size_t room)` (`metal_lower.mm`, before `TaskLibrary`):
  - return 0 where `library.prepare` is empty or the object is already made;
  - otherwise `dlsym` the prepare and release entries ("its unit's prepare or release export is missing" if either is
    absent);
  - set the object to `prepare()`;
  - copy the 8-byte pointer into `p.values[library.value].data` where that is held.
- `Close`: `if (object && destroy) destroy(object);` before `dlclose(handle)`.
- `UpstreamTask`, after `Present`: `compiled.prepare = u.prepare; compiled.release = u.release;
  compiled.value = u.prepared;` then `Prepare(p, compiled, …)`, calling `Close` on failure.
- `TargetBind`: `Prepare(p, *k.library, …)` after `k.library->handle = handle`.
- `metal_ir_upstream_emitted`: a parameter whose slot is `METAL_IR_UPSTREAM_PREPARED` (first pass of a unit; anything
  else is an explicit "unfinished" EINVAL) gets the unit's one value:
  - a CONSTANT `METAL_IR_U64` `{1}` value, `owned`, 8 held zero bytes, made at the unit's first projection and shared
    through `prepared`;
  - bound at `first.bindings.size()` with access 1, 8 bytes, alignment 8, plus `first.added` {value, 0, 8};
  - the parameter's slot rewritten to that binding's slot.
- Archive, task library: `o.Str(library.prepare); o.Str(library.release); o.Pod(library.value);` after its images,
  and the matching `in.Str`/`in.Pod`. In `Load`, after the values loop, each library with an object copies it into its
  value's held bytes (`Prepare` ran at its `TargetBind`).
- `UpstreamModule`: `x.Pod(u.prepared)` after `x.Str(u.release)`.

Design as drafted: one prepared object per unit, made by the library that owns the unit code and released before that
library closes. It reaches every projection, in both the task library's host steps and the whole program's linked
object, as the value its CONSTANT word binding points at. The runner passes `*pointers[slot]` (`pointer` 0).
