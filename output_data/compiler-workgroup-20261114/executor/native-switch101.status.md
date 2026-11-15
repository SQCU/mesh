# Scalar switch encoder — source held

Source checkpoint: `a00eb204`. SAME047 session
`047e12be-cd27-4637-a3bb-7d46f6e8216b`, explicit `claude-opus-5-5`.
No native import, build or invocation has validated this source yet.

- `lower_metal.instructed` consumes actual canonical branch GraphModules and
  explicit operands, recursively using the existing `_instruction` for leaves.
  The original source task retains every emitted leaf and branch result write.
- `Builder.switch` emits existing `_copy` writes inside each arm, before its end
  ordinal. Original yield/common ValueIDs use `_stored`; C registration moves no
  data and retains that provenance.
- One construction stack covers Repeat and Switch. Each constructor receives
  only its actual direct children as `{kind, region, arm}`, in source postorder.
  Equal/empty spans and mixed nesting need no interval hierarchy inference.
- Published C kinds are Repeat=1 and Switch=2. `RegionChild` is 12 bytes,
  `RepeatRegion` 44, `SwitchRegion` 48. Constructor argument order matches
  `metal_ir.h`; both append child count and child-array pointer.
- Repeat and switch APIs use `_compiler`'s canonical binding table. The displaced
  lazy repeat cache/helper is removed. No stale-library compatibility path.
- `native.py`, Bound, and the frontend output/destination normalization are
  unchanged by this unit.

Supervisor AST, pyflakes and diff checks passed on both edited Python files.
Native selector/range, result/completion and all-engine consumers remain the
paired C unit's obligation. Fresh imports stay held until its matching build.
