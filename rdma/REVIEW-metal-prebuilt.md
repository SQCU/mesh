# Review: libnccl-mesh's kernels compiled at build time (branch metal-prebuilt, 2026-10-21)

The regression: libnccl-mesh compiled its own Metal kernels in every process, at its first collective
(`newLibraryWithSource` on about 30 KB of text, then one pipeline per kind at first use). The operator, 10-21: "then
this is a regression and it is certainly the case that this does not need to be dynamically built and linked in this
situation." The measurements behind it are in metal-microbench `output_data/build-time-20261021/review.md`
("Metal").

Order of work. The source was written and these pairs were drafted first. An independent reviewer then ranked the
diff against design/rubrics.md before anything was compiled. Its defects were fixed (listed below), and only then was
anything built, on the M5, in this worktree. `A ≺ B` means A ranks below B. Line numbers are this branch's.

## Kinds

- **C (collective code):** libnccl-mesh (`rdma/nccl-mesh-metal.m`); its kernels (`rdma/nccl-mesh-kernels.h`, moved
  verbatim from `nccl-mesh-metal.m` except a header comment and two comment lines); their build tool
  (`rdma/nccl-mesh-kernels.m`); the AIR and its record (`vendor/libnccl-mesh.metallib`, `vendor/libnccl-mesh.sha256`);
  comment citations in `nccl.h`, `nccl-mesh.c`, `nccl-mesh-metal.h` and `nccl-mesh-host.c`; and the documents
  (`NCCL.md`, `vendor/PROVENANCE.md`).
- **`rdma/Makefile`:** builds both M and C images. The changed rules are C's. The `install` split touches how M's
  images install: the same loop and the same order, now ahead of libnccl-mesh.
- **No M source changed:** `mesh-flow`, `libmesh.dylib` and `mesh-heartbeat` have the same sources.
- **No D code changed.**
- **The ABI is unchanged:** no declaration in `nccl.h` changed, and `ncclMeshMetalSteps()` returns the same text.

## What the change does

1. **Build time.**
   - `.build/nccl-mesh-kernels` writes the kernels' text: `mesh.metal`'s primitives, then `STEPS`, `KERNELS`, `TRACKS`
     and `TRACK_KERNEL`. That is the text `kernels_of` compiled before.
   - The kernels' SHA-256 is the hash of the compile line (`METAL`) and the text (Makefile:116).
   - `make -C rdma kernels` compares `vendor/libnccl-mesh.sha256` with two values: that SHA-256, and the committed AIR's
     own SHA-256 (Makefile:130-142). Where both match, the committed AIR is used.
   - Where either differs, the kernels are compiled into `vendor/`. A machine without the toolchain fails there, naming
     the build to run.
2. **Install.**
   - `install-images` installs every image except libnccl-mesh, as before (Makefile:33).
   - `install-nccl` comes next (Makefile:43). It makes this node's binary archive of every kind's pipeline (`NAME`: 71
     kinds and `tracks`) with the build tool. It installs `libnccl-mesh.<SHA-256>.metallib` and `.binarchive`, then
     `libnccl-mesh.dylib`.
   - Earlier kernels' files stay. `clean-kernels` removes them, run by hand.
   - Nothing runs after install, and nothing starts by itself.
3. **Run time.**
   - `kernels_of` (nccl-mesh-metal.m:89) finds its own image with `dladdr`. It loads the two files named by
     `KERNELS_SHA256`, which is compiled into the dylib.
   - It makes every kind's pipeline from the archive with `MTLPipelineOptionFailOnBinaryArchiveMiss`.
   - A missing file or a miss fails the collective as a value. The message starts with `make -C rdma install` and
     names the file.
   - The per-kind `pipeline()` is gone. A dispatch reads `k->pipelines[kind]` (nccl-mesh-metal.m:303, 541).

## The independent review's defects, and what was done

1. **The committed AIR was trusted on the text's hash alone, so a merge that took one file of two would install AIR of
   other kernels.** Fixed. The record is now two lines, the kernels' SHA-256 and the AIR's own, and both are checked
   (Makefile:131-132). Checked by hand: a record whose second line was wrong failed `make` on a machine without the
   toolchain (`output_data/metal-prebuilt-20261021/no-toolchain.log`).
2. **Install deleted other kernels' files.** A process that mapped the earlier dylib but had not yet loaded its
   kernels would then fail. Fixed. Install deletes nothing, and `clean-kernels` is a separate target, run by hand
   (Makefile:53).
3. **A stale AIR stopped the bridge's images from installing.** Fixed in the Makefile. The AIR is no longer in `all`.
   `install` installs the other images first, then libnccl-mesh with its kernels. Checked by hand: with a bad record
   and no toolchain, `make install` installed `mesh-stat` and then failed at the kernels.
   - Still open: `bin/mesh-user.sh` runs `make all install || exit 1`, so its later steps (torch-mesh, the bridge's
     configuration) still stop there. That file belongs to the agents now editing provisioning and was not touched.
4. **The vendor prerequisites had no rule.** Fixed with `$(wildcard ...)`. A missing file falls into the compile
   branch.
5. **The toolchain probe (`xcrun -f metal`) was weak.** Removed. The compile is tried, and the named-build message is
   printed when it fails.
6. **The hash covered the text but not the flags.** Fixed. The compile line is hashed with the text. The language
   version is now stated (`-std=metal4.0`); it gave the same bytes as the default.
7. **`-Werror` had no deprecation exception.** Added `-Wno-error=deprecated-declarations`. The first build also
   showed two unused functions from `mesh.metal` (`mesh_remote_wait`, `mesh_remote_wait_bounded`), which other
   consumers call. So `-Wno-unused-function` was added; every other warning is an error.
8. **The M5's "commit these" notice went to stdout**, which `mesh-user.sh` discards. Moved to stderr.
9. **Messages.** A nil `error` is guarded. The instruction comes first, so the 512-byte buffer never cuts it off. A
   missing metallib now reads "installs", not "builds".
10. **All 72 pipelines are made at the first collective.** Measured below: 1.6-1.8 ms for 72, against 0.55-0.63 ms for
    the five a two-rank float32 all-reduce uses. That is about 1.1 ms more, once a process. Kept (see the C4 and
    Quantities pairs).

## Measured on the M5

All runs are by hand, in new processes, on this M5 (Apple M5 Max, macOS 26.3.2). The logs are in metal-microbench
`output_data/metal-prebuilt-20261021/` (`check.log`, `kernels-check.m`).

| | library | pipelines | total |
|---|---|---|---|
| before: run-time compile, text seen before | 10.8-14.8 ms | 86-93 ms for the five | about 100 ms |
| before: run-time compile, new text | 151-177 ms | 86-123 ms for the five | about 250-300 ms |
| after: `kernels_of` as built | 0.2 ms | archive load 0.9-1.1 ms, then 72 lookups 1.6-1.8 ms | 2.9-3.5 ms |

- Every lookup hit: 0 misses of 72, three times.
- `make install` makes the archive (1.48 MB) in 0.8-0.9 s. Its bytes differ from run to run, so install replaces it
  every time.
- The AIR is 418,375 bytes. A second compile of the same text and flags gave the same bytes.
- This is one device. The pair's first-run and steady step times are the main session's to read (rubrics,
  Quantities).

## Ranked pairs

**Quantities: overhead.** A process's Metal start of about 100 ms (250-300 ms after a change of the text), paid
inside its first collective ≺ about 3 ms. The operator's "1% overhead is not acceptable" (10-20) is the scale. The
earlier first chain run took 94 µs a step against 48.4 µs on its rerun (build-time review).

**C4, asynchrony.** A host compile between a group's steps ≺ no compile on any call's path.
- Before: the library on the first call, and each kind's pipeline at its first dispatch (origin/main `pipeline()`,
  reached from `dispatch_as`).
- Now: one load at the first collective, and a dispatch reads an array (nccl-mesh-metal.m:303).
- Eager or lazy lookups: making the five used kinds lazily would save about 1.1 ms once a process. But it would put a
  lookup (about 0.03 ms) between a later call's steps whenever a new kind first runs, and a miss on a rare kind would
  show up mid-run. The two differ by under 3% of any run, so they rank equal on time, and C4 and C6 decide: lazy ≺
  eager.

**C6, failure's reach, on an archive miss.** Two forms:
- compile once more (the build-time review's "real code should not fail on a miss, only fall back", text an agent
  wrote without a citation);
- fail the collective as a value, naming `make -C rdma install` (nccl-mesh-metal.m:113-126).

Ranked: the first ≺ the second.
- The operator: "code which can't perform well can fail instead of being run". A miss means the system's GPU compiler
  changed (an OS update). The fallback would pay the old 100 ms in every process, silently.
- Failure across interfaces 2: a call made to succeed on a broken substrate ≺ a call that returns its error. Failure
  across interfaces 5: machinery for a rare event ≺ none. Quantities: a regression patched ≺ a regression unwritten.
- Uncited agent text ≺ the operator's text (rubrics, "How to read them").
- `kernels_of` keeps nothing when it fails. So the same process's next call reads the files again and succeeds once the
  install has run. No restart is needed, and nothing is left for the next call to repair. Checked by hand: the three
  failure paths gave their messages (`failure-paths.log`).

**C6, files outside the process.** Install removing other kernels' files ≺ keeping them until `clean-kernels`. The
draft removed them; the review found it, and it was fixed.

**Failure across interfaces 4, a halt.** A prebuilt metallib under a fixed name, loaded whatever it was built from ≺
files named by the kernels' SHA-256 compiled into the dylib, with a record that also pins the AIR's bytes. Kernels of
other text could read `struct mesh_args` in another layout. A wait reading the wrong word spins to its bound or past
it, and on a node that is a watchdog reset. With the named files, the call simply fails.

**Failure across interfaces 3, no co-dying parts (M4, M6 for the install).** A stale AIR stopping the bridge's images
(the draft: the AIR first in `all`) ≺ the images installed first, and libnccl-mesh failing alone with the node's
earlier libnccl-mesh left in place (Makefile:31-51). The caller's `make all install || exit 1` in `bin/mesh-user.sh`
still ends that script there. Open, for that file's owners.

**Strings are messages.** A recorded hash read from a file and compared in the library ≺ the hash in the files' names,
which the file system resolves; the library compares no text. Kernel names are still looked up by `newFunctionWithName`
(an interface the library does not implement): once a process for every kind, and once an install. No dispatch looks a
name up.

**C9, one primitive offered as a header.** The kernels' text inside `nccl-mesh-metal.m`, which a tool could reach only
by including a `.m` file ≺ the same single copy in `nccl-mesh-kernels.h`, included by the library and by its build
tool. The dylib no longer holds the 30 KB of text or `mesh.metal`'s bytes; it keeps `STEPS` for
`ncclMeshMetalSteps()`. The committed AIR is a derived copy, held in step by its two-line record.

**C1, provenance.** NCCL builds its device code with the library (cubins for each listed GPU, PTX for the rest), not at
the first collective. The new form has that shape: AIR built once with the library, in PTX's place, and a per-node
binary archive, in the cubin's place. It deviates in one way: where NCCL's driver would compile PTX for an unlisted
GPU, this library fails (C6 above).

**C2, invariance.** Equal. The AIR runs on every Apple GPU. Each node makes its own archive the same way: the M5 Max,
the M4 Pro and the Studios alike, with no table of GPUs and no branch for one. Committing an archive per GPU kind (a
table by chip, D6's lower pole) was not done.

**C8, numerical candor.** Equal in what is declared, and not bit-identical. `MTLMathModeSafe` becomes
`-fmetal-math-mode=safe`, and Metal 4.0 is what the run-time compile took by default (0x40000 here). But Xcode 26.4.1's
toolchain does not produce the run-time compiler's AIR: an archive made from the run-time compile of the same text
served none of the 72 offline functions, under every flag set tried (`check.log`). The kernels are the same text in
the same math mode, compiled by a different compiler build. By C8, last-bit differences are accepted. Not confirmed:
the pair's outputs under the new AIR.

**Strings are messages 5, declared, not inferred.** The AIR's target and language taken from the toolchain's defaults
(the building machine's OS) ≺ declared on the compile line: `-std=metal4.0 -mmacosx-version-min=26.0`. These gave the
same bytes as the defaults on this M5, and they are now hashed with the text.

**Inversions, "lives longer".** Nothing resident was added: no daemon, no launchd job, no watcher. The files are build
products keyed by content. A person or agent makes them by building and installing. D8 ranks caches keyed by content
high.

## Effect on other callers

- `ncclMeshMetalSteps()` returns the same `STEPS` text (nccl-mesh-metal.m:613). A caller that compiles its own kernels
  after the steps (metal-microbench, torch-mesh's `_stream.pipeline`) still compiles them in its own process. That
  compile is the caller's, and this change does not touch it.
- Each of them loads `~/mesh/rdma/libnccl-mesh.dylib` by rpath, so `kernels_of` finds the files `install` put beside
  it. Nothing copies the dylib elsewhere.
- Each node must run `make -C rdma install` once with this change before its first Metal collective. Until then, the
  collective fails as a value, naming that install.
- The host build (`make host`, `nccl-mesh-host.c`) has no Metal and is unchanged.
- The branch fused-stream edits `STEPS` in `nccl-mesh-metal.m`. Those lines now live in `nccl-mesh-kernels.h`, so a
  merge of that branch must carry the edit there, and must commit a new AIR built on the M5.
