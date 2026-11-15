# Numerical replacement working checkpoint

This is an ordinary source checkpoint of the ongoing whole Scheduled → upstream compiler → native/archive replacement. It does not establish import, runtime, target, precision or complete numerical coverage. The upstream composition patch is retained source, not installed or executed. Current active same-session workers continue from these edits.

Producer65f:93c2b497-5654-46eb-8756-73bc2268151a, exec59562; consumer66f:476e57ac-cf94-49e8-801a-a3ded988dd77, exec78017; methods67b:211a403e-ed56-4d48-a343-ddc67d416cc4, exec52090. Base before checkpoint: `c599a8e51b9d45d45804b13dffd967c321c5b749`. Native1113ac4d/v42 build and isolated ABI passed independently; full Python imports remain held.

The source now retains processor Scheduled graphs, complete selected-target maps, original GraphLowering/layout/control owners and per-call launch results. Generic `_instruction`/SOURCES dispatch was removed, and custom algorithm tensor bodies are replacing its old record compiler. General Buffer-role reboxing/copied `_metal` emitter was removed. These changes remain connected work in progress.

At this snapshot the copied compile_fx pass sequence/`_ordered`, single-tag rewrite provenance, remaining method record/meta-tensor bridge, final CPU composition/ABI, precision and ANE target connections still require correction. Existing native LAPACK workspace/implementation ownership must be consumed without a duplicate query or integer-ABI mismatch. Canonical error workspace is native-owned STATE with atomic append effects and completion reset; current Python mirrors/old references are being displaced. Actual lowered graph/source correspondence and final native saved operands are the archive boundary; load must not rerun upstream compilation.

Only the named files below are staged. Hashes identify the index snapshot even if a live worker edits the worktree afterward. Static parse is source structure evidence only.

| File | Lines | SHA-256 | Check |
|---|---:|---|---|
| compiler/mesh_compiler/driver.py | 130 | dce59a52216d62ed9ec06e5f1b273c5c93797431f5c19a2db16d6f2abb36bf22 | AST pass |
| compiler/mesh_compiler/ir.py | 350 | 1367dea46d8b56963f9ce36af342d6e006502735f7b90cb4b0e5ee4848f4ead8 | AST pass |
| compiler/mesh_compiler/lower_inductor.py | 711 | 5391299b4ae694e2f627d2e813a2e95f3e95f6738197c7d0241bebb99b5169f7 | AST pass |
| compiler/mesh_compiler/lower_metal.py | 711 | 00147d14016d91c4c92d32f8f53e1dffe88d059520627b224867b892e2efce1e | AST pass |
| compiler/mesh_compiler/metal.py | 1140 | 149db27d33a930e216f11453483201f24b177825d2b2ceca12a1ececeacc2ee3 | AST pass |
| compiler/mesh_compiler/operations.py | 4775 | 83843c028c18afded5c5df05b77866e1c24e36e33e63c7b924b694e347511569 | AST pass |
| compiler/mesh_compiler/decompose.py | 568 | 2af20f0fe921fe4dd72e25cffe1906770b1e375bafea4d66581e0cfd5ae22eec | AST pass |
| compiler/patches/torch-inductor-upstream-composition.patch | 190 | 15ba8679a9fc00bae38a1ad0b35dc21f1d9ddc9a470688ea3ec6145efba2f102 | document/patch |
| compiler/patches/README.md | 432 | 7e3c19a5631680f3f16ca43dcdf9452eba5b8903da1828888fd8d177ef995348 | document/patch |

## Stopped65f source boundary

65f PID18391/exec59562 interrupted and collected exit0. Its CPU original-emitter composition hook, final per-pass ABI and canonical native error-workspace displacement are preserved. The new `_inductor` remains rejected: it wraps AOT export/compile_fx_forward but manually reproduces both outer and inner compiler pass order in nested `compiled`. A fresh Opus context takes the original compile_fx/AOT result hook; individual upstream function calls are not ownership of the original pipeline. No source import or patch application.

- `compiler/mesh_compiler/lower_inductor.py`: `524d20839f2c52d431fdfa916b3347450d4484419587b671c20cf0281388f39a`.
- `compiler/patches/torch-inductor-upstream-composition.patch`: `a08d2ded96ccb7910e7ef24959cc3eade47255ab6b705c9442075091abd87b07`.
