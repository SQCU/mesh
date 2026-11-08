# The asked and built torch lowering: the mechanisms invoked without their literature

The operator's question of 2026-11-07 19:34 (−08:00), Claude's answer of 2026-11-07 19:48, and the operator's reply of
2026-11-07 20:02, each copied by script from the session transcript
(.claude-personal/projects/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550.jsonl). The answer is
verbatim, without its opening emoticon; its times are UTC, and its code is cited at mesh 97a6bb8c and metal-microbench
bed56a95.

## The question

> wrt this specific asked/built gap, which repo types, algorithms, and data structures were implicitly invoked by the user or assisatnt turns proximate to that work and not spoken explicitly & w/ citations to the explicit external literature? i am only interested in the *citational* failures inherent to this history here as they are the most strongly predictive of what goals were recognized and misapplied or misgeneralized in the repo so far (so we can, as usual, construct change request documents of the simple and citationally-closed form...)

## The answer

Eight mechanisms carry the gap. Each was invoked by idiom or by an API's name instead of by its literature. For each one below: where it was invoked, the repo code, the literature that names it, and the goal it shows was misread. Times are UTC.

- **"In FL"** means already quoted in design/final-lowering-sources-2026-11-06.md. "§4.x" alone means a section of the PyTorch 2 paper (Ansel et al., ASPLOS 2024) as FL quotes it.
- **"Crossref-checked"** means I confirmed the bibliographic data, but nothing from the work is quoted in any record yet.

**1. Capture used in place of compilation**
- **Invoked:**
  - User 22:03: "a cudagraphs equivalent parser *in the library* which lets us aot replace a torchpython script with an IR".
  - Assistant 22:45: "the CUDA-graph-trees pattern"; assistant 00:49: "CUDA Graphs' stream capture done on Metal".
  - User 02:08: it "names no data structure, no algorithm, no authors".
- **Repo:** `_captured` (mesh_compile.py:1775), `MetalRecorderCapture`, `struct MetalReplay`.
- **Literature, in FL:** §4.7, "TorchInductor uses CUDA Graphs [20] to completely eliminate the overhead from wrapper code."
  - In that pipeline, record/replay comes after lowering (§4.3), scheduling (§4.4) and codegen (§4.5), and replays the generated kernels.
  - Your order was the same: "use, compile, and replay" (09-12); "after tracing, compilation, lowering, etc" (10-28).
- **Goal misread:** no host overhead at execution time. Capture achieved it for eager kernels by skipping the three stages before replay.
  - In the specialization record's terms (Definition 4.1), the recording is a residual program of torch's eager interpreter. So its operations are that interpreter's kernels.

**2. Every node is a fallback**
- **Invoked:** user 22:41, "use torch generally *and also* be fast"; implementer prompt 22:48, "the engine's kernels run wherever they compute an ATen operator".
- **Repo:** `_steps` (:1697) → `_called` (:1540) → `_apply` (:1521), which runs each node's operator eagerly on MPS.
- **Literature, in FL:** §4.3, "Lowering is done by symbolically interpreting the FX graph and applying lowering functions which do the conversion for a single operator… If an unknown operator is encountered, it is automatically converted into a fallback kernel node which runs the original PyTorch code."
  - `_apply` interprets the graph concretely, so every node is that fallback. No lowering function exists.
- **Goal misread:** generality, obtained by falling back everywhere.

**3. Kernels registered per ATen operator**
- **Invoked:** assistant 22:45, "The fast kernels plug into torch without new code", citing torch/library.py's docstring; user 22:47, "have a subagent integrate this".
- **Repo:** `_kernels` (:1394) and `torch.library.impl('aten::mm', 'mps')` (:1400), with the half / unit-stride side condition and a fallback.
- **Literature:**
  - Choosing target operations to cover IR nodes is instruction selection by tree or DAG covering: Aho, Ganapathi & Tjiang, TOPLAS 11(4), 1989; Ertl, POPL 1999; Koes & Goldstein, CGO 2008 (Crossref-checked).
  - In FL, a library kernel is §4.4's `ExternKernelSchedulerNode`, and matmul is §4.5's template.
- **The fused kernels.** By their names, several engine kernels compute subgraphs, not single operators: `rms_norm_add`, `moe_gelu_mul_fused`, `dense_gemv_q8_0_v7_rmsnorm_qkv`, `vision_ffn_gate_up_gelu_fp32_mma`.
  - A per-operator registration only matches one-node patterns, so it can never select them. Selecting them is the DAG case.
- **Goal misgeneralized:** "plug in the fast kernels" was satisfied by mm, the one kernel whose pattern is a single node.

**4. Dividing a recorded dispatch among partitions** (the stop in items 3 and 9)
- **Invoked:**
  - Assistant 22:03: "Torch's MPS operators take one contiguous strided buffer per tensor, run as whole synchronous operations, and keep no partial results in registers". That describes the synchronous, complete, contiguous consumer.
  - Assistant 00:00: "the recorder rewrites each reader's kernel source…, and torch's kernels ship compiled, with no source".
  - User 00:12, on "predicated off and run later"; user 01:21, on "MSL's barrier rule requires it".
- **Repo:** `MetalMaskedPass` (metal_recording.m:1447), `MetalCompactionSource` (:1596), `MetalMaskReaders` (:1618).
- **Literature:** Halide (PLDI 2013) §4.2, "Bounds Inference". FL retrieved the PDF; I verified this passage in it, but FL doesn't quote it: "it symbolically evaluates the bounds of each dimension based on the bounds required of its caller and the symbolic indices at which the caller invokes it".
  - The general forms are array regions (Triolet, Irigoin & Feautrier, SIGPLAN CC 1986) and array dataflow analysis (Feautrier, IJPP 20(1), 1991), both Crossref-checked.
- **Why it stopped:** these methods need index expressions, and a precompiled recorded dispatch has none. So the stop comes from capturing instead of lowering.
  - Every cited consumer (MPI_PARRIVED; FLUX, TileLink and Triton-distributed in FL) is code that knows which partition each of its parts reads.
- "MSL's barrier rule" is barrier divergence: Aiken & Gay, "Barrier inference", POPL 1998; Betts et al., GPUVerify, OOPSLA 2012 (Crossref-checked).
- **Goal misapplied:** U41/U71's consumption as data lands, retrofitted onto opaque kernels.

**5. The replay container taken as the target**
- **Invoked:**
  - Assistant 00:00: "Metal's indirect command buffers hold no copies".
  - Assistant, quoted by you at 01:21: "an indirect command buffer holds only draws and dispatches".
  - 03297e52's fill stop.
- **Repo:** the indirect command buffer's `commandTypes` (metal_recording.m:3020), and `MetalReplayEncode` (:3383), which cuts the replay into runs.
- **Citational failure:**
  - Item 1's sources (apitrace, GFXReconstruct, RenderDoc's "every API call is serialised out in order") replay calls in order. None says to store them as a command list, or where to cut one.
  - The replay-grounding record's row 1 names the lineage (channel program, display list, bundle, CUDA graph, indirect command buffer), but gives no rule for choosing among them.
- **In a lowering**, a fill or copy is a graph node. It is either fused away (TVM §3, verified in FL's PDF: "without saving the intermediate results in memory") or emitted as a dispatch.
- **Goal misgeneralized:** "Metal 4 only" was read as "one Metal 4 container".

**6. Memory planned over torch's eager allocations** (items 6 and 7)
- **Invoked:** 03297e52, "every buffer torch made during the capture is a value of the capture, given to mesh_plan"; assistant 02:12, "One contiguous offset per value"; your 02:31 turn on bin packing contiguous and noncontiguous pages.
- **Literature, in FL:** §4.4 plans memory "for buffer removal and/or reuse" after deciding "which operators get fused". The capture plans a page for every intermediate, including those a fusion would remove.
- **Item 6's stop** ("where that memory block lies among the pages") was answered on 11-04 and never written: one run of frames by the buddy system (Knowlton, CACM 1965). packing-and-coflows-2026-11-06.md records it as "never taken up".
  - The mixed problem has unsplittable contiguous items and splittable any-page items. That is bin packing with item fragmentation (Mandal, Chakrabarti & Ghose 1998; Menakerman & Rom, WADS 2001; Crossref-checked). No record cites it.
- **Goal misapplied:** the AOT compiled memory plan, run over the wrong set of values.

**7. Item 8's stop: "which IR carries one on Metal"**
- **Invoked:** the stop itself, against your 02:31 instruction to "cite other dsl/hardware pairs to analogize".
- **Literature, in FL:** the IR is §4.3's "define-by-run IR", or TVM's and Halide's split of compute from schedule (TVM §4.1).
  - The target enters only at codegen: "Triton codegen is responsible for mapping TorchInductor's IR to output Triton kernels" (§4.5).
  - FL §14 already quotes the Metal side: `matmul2d`, `cooperative_tensor`, execution scopes, `MTLTensor`.
- **Goal misgeneralized:** "cite everything" became "a citation must name the target".

**8. "Lower through LiteRT"** (rejected at 02:31)
- One front end's representation was chosen for another front end's programs.
- **Literature:** several front ends sharing one middle representation (Lattner & Adve, LLVM, CGO 2004; from memory, not retrieved).
- compiler-and-runtime.md §2 now lists three front ends, but cites no source for that split.

**What this predicts.** Each goal was recognized but attached to the wrong stage of a pipeline FL already quotes:

```
asked:  ExportedProgram → lowering functions + fallback (§4.3) → loop-level IR
        → fusion, order, memory planning (§4.4) → Metal codegen (§4.5; FL §14)
        → kernels → record/replay of their launches (§4.7)
built:  ExportedProgram → fallback at every node, plus mm under the dispatcher
        → torch's eager kernels → record/replay
        + memory planned over eager allocations
        + partition reads retrofitted onto recorded dispatches
```

The asked stages give five change requests, each closed by the sources above:
1. lowering functions with fallback;
2. engine kernels selected by DAG covering;
3. fusion with memory planning, with buddy runs for contiguous operands;
4. partition reads by bounds inference;
5. record/replay of the generated launches.

Want me to write them? I'd first retrieve and quote the Crossref-checked works.

## The operator's reply

> document this specific finding independently of any change request i guess, verbatim. moving on, 
> "
> What this predicts. Each goal was recognized but attached to the wrong stage of a pipeline FL already quotes:
> "
> this is an *annoying* but *easy* problem. no huge research or testing-heavy changes are required! the scope of impact of these 'change requests' is remarkably small and simple! this is the best case we could ever find from code review. 
> "Want me to write them? "
> sure
