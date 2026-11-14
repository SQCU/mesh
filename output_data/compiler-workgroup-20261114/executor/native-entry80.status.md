# Multi-function CPU entry linkage (native-entry80, root correction) — SOURCE-READY, HOLD

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5, on top of metal 9fa86439.

**Withdrawn.** The earlier uncommitted 19-line Emit hash/rename hunk in `metal_lower.mm` is removed by contextual
edit. The tree is clean again against 9fa86439.

## Exact producer branch (published before the edit)

**The owner.** `provider/dawn.patch`, `src/tint/lang/msl/writer/printer/printer.cc`, the Printer's compute-entry
emission (`case PipelineStage::kCompute`). For `options_.host_cpp` the patch emits `extern "C" ` before the entry.

**Why it collides.**
- The same patch already opens an unnamed namespace for host_cpp output (`if (options_.host_cpp) Line() << "namespace
  {";`, closed at the end of `msl`).
- So every helper and the `<entry>_host` trampoline has internal linkage: neither is in `nm` of `body0.o` or
  `body1.o`.
- Only the `extern "C"` entry has C language linkage, which gives it an unmangled external symbol: the observed
  `T _mesh_cpu_0` in both objects.

**Nothing needs it external.**
- CompileTask exports only its own `extern "C"` `mesh_native_*` wrappers, distinct per compiled output.
- It calls `<entry>_host` from its runner in the same translation unit.
- No caller dlsyms the Tint entry: the only `dlsym`s are of `nativeSymbol`.

**The correction.** In that branch, host_cpp emits no linkage specifier. The entry is then a module-local function in
the existing unnamed namespace. The Metal branch (`[[max_total_threads_per_threadgroup]] kernel`) is unchanged.

**Unchanged:**
- CompileTask wrappers and `mesh_native_*` exports;
- the shared NativeBinary and its one object addition;
- names, archive and the metal-microbench sources.

## Implemented (`provider/dawn.patch` only)

**The hunk.** printer.cc's hunk `@@ -366,9 +387,10 @@`, formerly `+387,12`, changes from
```
+                    if (options_.host_cpp) {
+                        out << "extern \"C\" ";
+                    } else {
+                        out << "[[max_total_threads_per_threadgroup(" << total_threads << ")]]\n";
+                        out << "kernel ";
+                    }
```
to
```
+                    if (!options_.host_cpp) {
+                        out << "[[max_total_threads_per_threadgroup(" << total_threads << ")]]\n";
+                        out << "kernel ";
+                    }
```
- The later printer.cc hunk headers' new-side starts shift by −2. No other hunk or file changes.
- **Result:** a host_cpp compute entry `<entry>` now has no linkage specifier inside the patch's existing unnamed
  namespace, so it has internal linkage, like its `<entry>_host` trampoline. Each task object's only external symbol is
  CompileTask's `extern "C"` `mesh_native_*` wrapper. The two `mesh_cpu_0` definitions are module-local and no longer
  collide at LowerProgram's link.

**Local dependency, by the Makefile's own recipe.** `make .build/provider/dawn-source` reversed `dawn.applied`, applied
the canonical `provider/dawn.patch` cleanly, copied it to `dawn.applied` (`cmp` equal), and touched the stamp. Nothing
was built. The stamp means the next canonical build rebuilds `$(DAWN_LIB)`, and from it `libmetal_ir`.

## Checks

- **printer.cc:** a private `-fsyntax-only` of the patched file with its exact Dawn compile command (`ninja -t
  commands`, nothing built): rc=0.
  - Its two `-Wunsafe-buffer-usage` warnings are at printer.cc:1094/1104, in the pre-existing patched
    `EmitNativeLapack`, not this branch.
  - The Dawn command has no `-Werror`.
- **metal-microbench:** no source change. The earlier Emit hunk is removed, and `git status` shows only
  `provider/dawn.patch`.
- `metal_recording.m` and the poisoned paths are untouched.

## Remaining gate

- **Checkpoint first.** Fleet remote main is 9fa86439 with provider dependencies not built, so this patch must be
  checkpointed before any shared local or remote canonical build.
- **Then build and retry.** The canonical build (Dawn rebuild, then `libmetal_ir`), then the same frontend Chained A/B
  input-gradient client. Not run here.
- **Tolerated warnings.** An entry or `_host` trampoline left unused in its own task TU would warn
  (`-Wunused-function`). NativeBuild compiles tasks without `-Wall`/`-Werror`, so that is not an error.

**Changed:** `provider/dawn.patch` only. HOLD.
