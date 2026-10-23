# Shared kernel interface: callback removal

Source and installed packages on both participants: `a4110d4`, September 14,
2026. Native algebra and Python binding were rebuilt from that commit. The
example remains the `66c6673` implementation. Existing ABI-25 bridges and their
1 GiB arenas were retained. No Xonotic work, arena resize or tile sweep occurred.

The change removes the arbitrary Python numerical callback path, ctypes callback
types and public native callback registration. Compiled CPU and Metal kernels
retain one private native binder. Tracked-source review found no external uses
of the removed registration. Python compilation, native Objective-C syntax
checking, package builds on both nodes and `git diff --check` succeeded.

The existing streaming-algebra program ran with these common CPU arguments:

```
--backend cpu --dtype float32 --runs 1 --samples 20 --depth 40
--tile-rows 64 --tile-k 128 --tile-columns 128
```

Local command: `PYTHONPATH=.build/overlap-package .build/xonotic-env/bin/python
examples/streaming-algebra.py 0 --local`, followed by those arguments and
`--trace .build/callback-removal-cpu-40-trace.json`.

Paired commands use rank 1 on the peer with
`PYTHONPATH=.build/gold-package`, then rank 0 locally, both without `--local`.
Their trace names are `callback-removal-paired-cpu-40-peer-trace.json` and
`callback-removal-paired-cpu-40-trace.json`. All logs and full traces are retained
as gzip files alongside this record.

| Run | Samples | Batch mean (ms) | Sample variance (ms²) | Maximum absolute error | Root completed functions |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU local, depth 40 | 20 | 27.7902229 | 5.7354092888 | 4.0206757203e-5 | 75148 |
| CPU paired, depth 40 | 20 | 27.82642095 | 0.3651594056 | 4.0206757203e-5 | 31338 |
| Metal local, depth 1 | 2 | 5.9910835 | 0.1384700575 | 1.4976386491e-6 | 1596 |

The Metal command uses `--local --backend metal --dtype float32 --runs 1
--samples 2 --depth 1`, default geometry, and
`--trace .build/callback-removal-metal-trace.json`. It exercises the retained
compiled Metal binder; two samples do not establish a performance conclusion.

All three root commands exited 0 with equal submitted/completed counts and zero
runtime error. Warmup withheld one input region until another traversed the
chain; timed samples were fully supplied. The paired peer was terminated with
SIGTERM after root completion, and its original SSH command exited 0 before
artifact retrieval. Both bridges subsequently reported ready, client 0,
unpaired, with the same PIDs (33497 local, 9229 peer); retained status codes
32/domain 1 and 36/domain 1 were not reset or interpreted as numerical errors.

This is evidence for removal of the callback interface while retaining ordinary
numerical execution. It does not establish a distributed speedup or fix the
compulsory reduction-to-epilogue dispatch boundary. Raw compiled backend code
also remains capable of expressing control flow; removing Python callbacks is
not proof that every remaining backend body satisfies streaming semantics.

The next native publication change must separate making a region PRESENT from
ending its producer's ownership. Current `mesh_publish` also clears PRODUCING.
Source review shows readers and send posting can use PRESENT while PRODUCING
retains storage ownership; final completion must still retire the invocation's
input readers. This identifies reusable canonical state, not a newly implemented
partial-publication ABI. Generated arithmetic still needs to emit publication
for exact configured regions while continuing computation, and shared lowering
still needs to preserve independently runnable consumer regions.
