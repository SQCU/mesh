Sources for design/torch-capture-change-request.md item 3 and the completion times mesh_plan publishes, retrieved 2026-11-08 and recorded here by script: each quotation is the source's text as the script copied it, with the corrections listed under it, each read from the rendered page. The script and the downloads were in the session's scratchpad directory, lit-hall/, and are not kept here.

# List scheduling in the order of a linear program's completion times: sources and verbatim passages

## 1. L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein: Schedule-by-C̄j

**Bibliographic.** L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, "Scheduling to Minimize Average Completion Time: Off-line and On-line Approximation Algorithms", Mathematics of Operations Research 22(3), 1997.

**Read.** The authors' preprint: Technische Universität Berlin, Fachbereich 3 Mathematik, Preprint No. 516/1996, 35 pages, from https://depositonce.tu-berlin.de/items/4cb97e87-6de9-440d-b21e-3fe9313bc406 (bitstream be0af06f-f5fa-4f3c-9760-82e2b4004000, a scan with an OCR text layer by ocrmypdf 6.1.2 and Tesseract 4.1.1). The script copied the repository's extracted text of it (bitstream 33d1ea06-c740-4bd8-b594-b394e8f344b1), from the line "2.1 Single-machine scheduling" on. Page numbers are the printed ones.

> We begin by presenting a 2-approximation algorithm for 1|prec|ΣwjCj based on the linear programming formulation that minimizes Σ_{j=1}^n wjCj subject to constraints (3) and (5).

— §2.1, p. 5.

> first, we obtain an optimal solution to the linear program, C̄1, ..., C̄n; and then we schedule the jobs in order of non-decreasing C̄j, where ties are broken by choosing an order that is consistent with the precedence relation. We refer to this algorithm as Schedule-by-C̄j, since the jobs are ordered according to their completion times in the linear programming solution.

— §2.1, p. 5.

> Observe that constraints (3) ensure that the resulting schedule will be consistent with the precedence constraints.

— §2.1, p. 5.

Corrections, each read from the rendered page (the PDF's page 7, printed page 5): the text layer's line breaks inside words joined, its displaced words "the linear" put back before "program,", and its symbols as the page prints them:
- `the linear ⏎the following heuristic for producing a schedule: first, we obtain an optimal solution to ⏎program,` read as `the following heuristic for producing a schedule: first, we obtain an optimal solution to the linear ⏎program,`;
- `linear pro- ⏎gramming` read as `linear programming`;
- `Conside⏎r` read as `Consider`;
- `ti⏎es` read as `ties`;
- `re⏎fer` read as `refer`;
- `ti⏎mes` read as `times`;
- `sch⏎edule` read as `schedule`;
- `©1,...,CGn;` read as `C̄1, ..., C̄n;`;
- `non-decreasing C;,` read as `non-decreasing C̄j,`;
- `Schedule-by-C;,` read as `Schedule-by-C̄j,`;
- `1|prec| > wj;C;` read as `1|prec|ΣwjCj`;
- `)7;_, wjCj` read as `Σ_{j=1}^n wjCj`;

## 2. R. L. Graham: the task list

**Bibliographic.** R. L. Graham, "Bounds for Certain Multiprocessing Anomalies", The Bell System Technical Journal 45(9), November 1966, pp. 1563–1581.

**Read.** The scan at https://archive.org/download/bstj45-9-1563/bstj45-9-1563.pdf (20 pages, no text layer). The script copied Tesseract's OCR of its page 2 (printed p. 1564), rendered at 300 dpi.

> We are given a linear ordering L: (T_k1, ···, T_km) of T called a task list (or priority list). In general, at any time t a P_i completes a task, it immediately (and instantaneously) scans the list L (starting from the beginning) until it comes to the first task T_j which has not yet begun to be executed. If all the predecessors of T_j (i.e., those T_i ≺ T_j) have been completed by time t then P_i begins working on T_j. Otherwise P_i proceeds to the next task T_j′, in L which has not yet begun to be executed, etc.

— §1.1, p. 1564.

> If P_i proceeds through the entire list L without finding a task to execute then P_i becomes idle (we shall also say that P_i is working on an empty task). P_i remains idle until some other P_j completes a task at which time P_i (and of course P_j) immediately scans the list L as before for possible tasks to execute.

— §1.1, p. 1564.

Corrections, each read from the rendered page (subscripts written with an underscore):
- `(7;,, --- , 7's,,)` read as `(T_k1, ···, T_km)`;
- `time ¢ a P;` read as `time t a P_i`;
- `list Z` read as `list L`;
- `first task 7’; which` read as `first task T_j which`;
- `predecessors of 7’; (i.e., those 7’; < 7';)` read as `predecessors of T_j (i.e., those T_i ≺ T_j)`;
- `by time ¢ then P; begins working on 7’; .` read as `by time t then P_i begins working on T_j.`;
- `P; proceeds to the next task 7, in L` read as `P_i proceeds to the next task T_j′, in L`;
- `If P; proceeds through` read as `If P_i proceeds through`;
- `then P; becomes idle` read as `then P_i becomes idle`;
- `say that P; is` read as `say that P_i is`;
- `P; remains idle until some other P; com-⏎pletes` read as `P_i remains idle until some other P_j completes`;
- `time P; (and of course P;)` read as `time P_i (and of course P_j)`;

## 3. Apple, MTL4ComputeCommandEncoder.h: a dispatch with an indirect buffer

**Read.** /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers/MTL4ComputeCommandEncoder.h, lines 90 to 112, as the script copied them (macOS SDK 26.4.1).

```
/// Encodes a compute dispatch command with a grid that aligns to threadgroup boundaries, using an indirect buffer
/// for arguments.
///
/// This method allows you to supply the threadgroups-per-grid counts indirectly via an ``MTLBuffer`` index. This
/// enables you to calculate this value in the GPU timeline from a shader function, enabling GPU-driven workflows.
///
/// Metal assumes that the buffer contents correspond to the layout of struct ``MTLDispatchThreadgroupsIndirectArguments``.
/// You are responsible for ensuring this address aligns to 4-bytes.
///
/// Use an instance of ``MTLResidencySet`` to mark residency of the indirect buffer that the `indirectBuffer`
/// parameter references.
///
/// - Parameters:
///   - indirectBuffer:        GPUAddress of a ``MTLBuffer`` instance providing compute parameters.
///                            Lay out the data in this buffer as described in the
///                            ``MTLDispatchThreadgroupsIndirectArguments`` structure. This address
///                            requires 4-byte alignment.
///   - threadsPerThreadgroup: A ``MTLSize`` instance that represents the number of threads in one
///                            threadgroup, in each dimension.
- (void)dispatchThreadgroupsWithIndirectBuffer:(MTLGPUAddress)indirectBuffer
                         threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;

/// Encodes a compute dispatch command with an arbitrarily sized grid, using an indirect buffer for arguments.
```
