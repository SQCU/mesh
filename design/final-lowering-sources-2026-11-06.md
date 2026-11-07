# Final lowering: sources' own statements

Retrieved 2026-11-06. This file quotes the sources and adds no argument or recommendation.

## Retrieval record and conventions

- Downloads were in new directories under
  `lit2/`, a session's temporary folder that is not kept in this repository
  (`fl00_*` holds metadata; `fl01_*` to `fl14_*` hold the sources). Local paths are given in full below.
- Text was extracted from PDFs with `pdftotext` (poppler). Each quotation was string-matched against that
  extraction (whitespace normalised, ligatures expanded) and its page found from the match.
- **Pages.** TVM's proceedings PDF prints proceedings page numbers, and those are cited. The Triton, Halide
  and CoCoNet PDFs print no page numbers, so they are cited by PDF page; their proceedings page ranges
  appear in the bibliography. All other sources are cited by PDF page.
- Names typeset in small capitals in a PDF (CuTe in Cecka; FlashAttention in the NeurIPS PDF; Flux) are
  written in ordinary case. Hyphens at line ends are kept where the layout-mode extraction shows them and
  the compound appears hyphenated elsewhere in the same source. Bracketed numbers are the sources' own
  reference markers, and `[...]` marks an omission.
- Header quotations are reproduced byte-for-byte in code blocks, with their line numbers. Header typos
  are left as they are.
- Nothing is quoted from a source that was not read. Items not read are listed in the last section.

---

## A. Tile-level kernels mapped to the hardware's matrix units and memory hierarchy

### 1. Triton (Tillet, Kung, Cox — MAPL 2019)

**Bibliography.** Philippe Tillet, H. T. Kung, David Cox. "Triton: An Intermediate Language and Compiler
for Tiled Neural Network Computations." In *Proceedings of the 3rd ACM SIGPLAN International Workshop on
Machine Learning and Programming Languages (MAPL '19)*, June 22, 2019, Phoenix, AZ, USA. ACM, pp. 10–19.
DOI 10.1145/3315508.3329973. Pages and DOI are confirmed by Crossref.

**Read.** I read the author-hosted PDF, which carries the ACM rights block and DOI:
https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf →
`lit2/fl01_triton_mapl19/2019-mapl-tillet-kung-cox.pdf`
(10 pages). I read the abstract and §§1–3 and §5.

**Property it names.** It names "tile-level operations and optimizations", with tiles decomposed "to fit
a machine’s compute capabilities and memory hierarchy as tightly as possible".

> We present Triton, a language and compiler centered around the concept of tile, i.e., statically shaped
> multi-dimensional sub-arrays. Our approach revolves around (1) a C-based language and an LLVM-based
> intermediate representation (IR) for expressing tensor programs in terms of operations on parametric tile
> variables and (2) a set of novel tile-level optimization passes for compiling these programs into
> efficient GPU code.

— Abstract, PDF p. 1.

> By contrast, Triton relies on the addition of tile-level operations and optimizations into traditional
> compilation pipelines.

— §2 Related Work, PDF p. 3.

> First, it simplifies the structure of tensor programs by hiding important performance details pertaining
> to intra-tile memory coalescing [12], cache management [32] and specialized hardware utilization [27].
> Second, it opens the door for compilers to perform these optimizations automatically, as discussed in
> Section 5.

— §3.2.1 Tile semantics, PDF p. 3.

> The Triton programming model is similar, but each kernel is single-threaded – though automatically
> parallelized – and associated with a set of global ranges that varies from instance to instance (see
> Figure 3).

— §3.3 Programming Model, PDF p. 4.

> Tile-level memory operation inside loops can be problematic, as they may induce severe latency that cannot
> be hidden in the absence of enough independent instructions. It is however possible to mitigate this
> problem in Triton-IR directly by detecting loops and adding adequate prefetching code where necessary (See
> Listing 7).

— §5.1.1 Pre-Fetching, PDF p. 5.

> Specifically, the optimizations performed by Triton-JIT consist of (1) hierarchical tiling, (2) memory
> coalescing, (3) shared memory allocation and (4) shared memory synchronization.

— §5.2 Machine-Dependent Passes, PDF p. 6.

> Nested tiling strategies (see Figure 5) aim at decomposing tiles into micro-tiles and eventually
> nano-tiles in order to fit a machine’s compute capabilities and memory hierarchy as tightly as possible.

— §5.2.1 Hierarchical Tiling, PDF p. 6.

> Tile-level operations that have high arithmetic intensity (e.g., dot) can benefit from temporarily storing
> their operands in fast shared memory. The purpose of the Shared Memory Allocation pass is to determine when
> and where a tile should be stashed to this space.

— §5.2.3 Shared Memory Allocation, PDF p. 6.

### 2. ThunderKittens (Spector et al.)

**Bibliography.** Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, Christopher Ré.
"ThunderKittens: Simple, Fast, and Adorable AI Kernels." arXiv:2410.20399v1 [cs.LG], 27 Oct 2024. The arXiv
license is CC0 1.0. OpenReview lists the work as "ThunderKittens: Simple, Fast, and *Adorable* Kernels",
ICLR 2025 Spotlight (forum `0fJfVOSUra`). I did not read the ICLR version.

**Read.** https://arxiv.org/pdf/2410.20399 (v1) →
`lit2/fl02_thunderkittens/2410.20399.pdf`.
I read the abstract, §1, §2, §3.1, parts of §3.2–3.3, and §5.

**Property it names.** It names three things: "Tile data structures with managed layouts", "Program
template for asynchronous work" and "Grid scheduling for pipelining thread-blocks". It also states the
aim of "keeping tensor cores at high utilization".

> Our abstractions map to the three levels of the GPU hierarchy: (1) at the warp-level, we provide 16x16
> matrix tiles as basic data structures and PyTorch-like parallel compute operations over tiles, (2) at the
> thread-block level, we provide a template for overlapping asynchronous operations across parallel warps,
> and (3) at the grid-level, we provide support to help hide the block launch and tear-down, and memory
> costs.

— Abstract, p. 1.

> The main vector of growth for accelerated compute is in specialized matrix multiply units. On the NVIDIA
> A100 and NVIDIA H100 GPUs, BF16 tensor cores represent 16× the FLOPs available relative to general-purpose
> BF16 / FP32 compute. Consequently, any high performance framework must prioritize keeping tensor cores at
> high utilization whenever possible. However, all kernels have other operations too (like memory loads or
> the softmax in attention), and it is crucial to minimize the overhead of non-tensor core operations.

— §1, p. 1.

> We discuss NVIDIA, but the parallelism types hold across architectures, including AMD and Apple GPUs.

— §1, footnote 1, p. 1.

> At the warp level, we use a 16 × 16 matrix tile as our basic data structure, maximizing compatibility with
> and encouraging the use of tensor cores. TK automatically picks the optimal memory layouts for the tiles to
> minimize bank conflicts while remaining compatible with specialized hardware instructions, avoiding user
> effort.

— §1, principle 1 ("Tile data structures with managed layouts"), p. 2.

> The developer’s effort reduces to populating a few boilerplate functions within this model, using our
> PyTorch-like operands, and the template internally hides latencies through memory pipelines and
> synchronization primitives (Figure 1).

— §1, principle 2 ("Program template for asynchronous work"), p. 2.

> Our template supports a persistent grid, where we overlap memory loads across thread block boundaries.

— §1, principle 3 ("Grid scheduling for pipelining thread-blocks"), p. 2.

> This model represents the ideal case of perfect overlapping between memory, compute, and tensor core costs.

— §2.2 Cost model, p. 4.

> The operations are executed by a “worker” abstraction, or a warp or warpgroup (4 warps) of threads that
> collaboratively own and operate on a piece of data.

— §3.1, p. 5.

> Layouts specify how logical data elements are mapped to physical thread ownership. Different execution
> units, tile sizes and types, and hardware-accelerated instructions benefit from different layouts.

— §3.1 "Choosing a memory layout", p. 5.

> We want our register tiles (the fastest GPU memory) to by-default keep memory in the layouts required by
> tensor core units (the fastest GPU compute units).

— §3.1, p. 5.

> We first define the template, which has four steps – load-compute-store-finish (LCSF for short) – and
> builds on the classical producer-consumer paradigm [7, 16].

— §3.2, p. 6.

> As discussed in Section 2, the general pattern of an AI kernel is to load tiles of large tensors from HBM to
> SRAM, perform computation in fast memory, store the result for the tile back to HBM, and repeat this for the
> next tiles.

— §3.2 "Programming abstractions", p. 7.

### 3. NVIDIA CUTLASS / CuTe layouts

**Repository.** NVIDIA/cutlass, branch `main`, commit `b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd`
(committer date 2026-11-06T09:43:14Z).

**License.** `LICENSE.txt` at that commit reads "Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved." and "SPDX-License-Identifier: BSD-3-Clause". It states one exception: "The files
located in the `python/CuTeDSL` directory are licensed under the NVIDIA End User License Agreement (EULA)."
GitHub's license API reports `NOASSERTION` for the repository.

**Read.** Raw files at that commit →
`lit2/fl03_cutlass_docs/`:
`media/docs/cpp/cute/00_quickstart.md`, `01_layout.md`, `02_layout_algebra.md`, `0t_mma_atom.md` (opening
sections), and the "CuTe" section of `README.md`. These are the documentation files the request names; they
are NVIDIA/cutlass Markdown, not this project's. I also downloaded `03_tensor.md` and
`0x_gemm_tutorial.md` but quote nothing from them.

**Property it names.** It names "hierarchically multidimensional layouts of threads and data" and an
"algebra of `Layout`s", on which "tiling and partitioning" are built. MMA "Atom"s expose architecture-specific
MMA instructions through layouts.

> CuTe is a collection of C++ CUDA template abstractions for defining and operating on hierarchically
> multidimensional layouts of threads and data. CuTe provides `Layout` and `Tensor` objects that compactly
> packages the type, shape, memory space, and layout of data, while performing the complicated indexing for
> the user.

— `media/docs/cpp/cute/00_quickstart.md`, first paragraph.

> Layouts can also be combined and manipulated via functional composition, on which we build a large set of
> common operations such as tiling and partitioning.

— `00_quickstart.md`, second paragraph.

> Fundamentally, a `Layout` maps from coordinate space(s) to an index space.

— `01_layout.md`, opening.

> CuTe also provides an "algebra of `Layout`s." `Layout`s can be combined and manipulated to construct more
> complicated layouts and to tile layouts across other layouts. This can help users do things like partition
> layouts of data over layouts of threads.

— `01_layout.md`, opening.

> Common utilities for building complicated layouts from simpler ones depend on the `Layout` product. Common
> utilities for partitioning layouts (of data, for example) across other layouts (of threads, for example)
> depend on the `Layout` divide. All of these utilities rely on the functional composition of `Layout`s.

— `02_layout_algebra.md`, opening.

> MMAs are architecture-specific. Different generations of GPU architectures introduce different sets of MMA
> instructions. However, CuTe features such as `Layout` makes it possible to expose MMAs for use in generic
> CUDA C++ code.

— `0t_mma_atom.md`, opening.

> Combining potentially multiple Atoms, a "TiledMMA" provides utilities for building more complex
> partitioning patterns by creating layouts and interleavings of Atoms.

— `0t_mma_atom.md`, opening list, item 4.

> Together, these two types comprise an "Atom" that decouples the complexity of thread and data layouts from
> the call site of the PTX instruction.

— `0t_mma_atom.md`, § "CuTe MMA Atoms".

**3a. Paper on the layout algebra.** Cris Cecka (NVIDIA Research). "CuTe Layout Representation and
Algebra." arXiv:2603.02298v2 [cs.MS], 29 Jul 2026 (v1 2 Mar 2026). The arXiv license is CC BY 4.0.
Read: https://arxiv.org/pdf/2603.02298 (v2) →
`lit2/fl03_cute_paper/2603.02298.pdf`.
I read the abstract and §1.

**Property it names.** Specialized tensor instructions "prescribe fixed, often complex data layouts that
must be correctly propagated through the entire execution pipeline".

> Modern architectures for high-performance computing and deep learning increasingly incorporate specialized
> tensor instructions, including tensor cores for matrix multiplication and hardware-optimized copy
> operations for multi-dimensional data. These instructions prescribe fixed, often complex data layouts that
> must be correctly propagated through the entire execution pipeline to ensure both correctness and optimal
> performance.

— Abstract, p. 1.

> CuTe layouts provide a framework for managing both data layouts and thread arrangements in GPU kernels,
> while the layout algebra enables powerful compile-time reasoning about layout properties and the expression
> of generic tensor transformations.

— Abstract, p. 1.

> Data storage layouts have always affected performance by dictating how and when memory accesses occur, but
> as hardware instructions become larger and prescribe fixed layouts for inputs and outputs the effect on
> correctness also becomes vital. These layouts must be propagated through the entire execution pipeline to
> ensure correct invocation of hardware instructions and ensure optimized memory access patterns.

— §1, p. 3.

> Separation of concerns: Data layouts are declared independently of algorithmic logic, promoting clarity
> and modularity.

— §1, p. 3.

> Static analysis and optimization: Sophisticated algebraic techniques empower the inspection, reordering,
> and partitioning of tensor arguments according to architectural constraints.

— §1, p. 3.

**3b. Second paper (abstract only).** Jack Carlisle, Jay Shah, Reuben Stern, Paul VanKoughnett (Colfax
Research). "Categorical Foundations for CuTe Layouts." arXiv:2601.05972v1 [cs.PL], 9 Jan 2026, 174 pp.
The arXiv license is non-exclusive distribution 1.0. File:
`lit2/fl03_cute_paper/2601.05972.pdf`.

> These methods are conceptually grounded in the abstract notion of a CuTe layout and a rich algebra of such
> layouts, including operations such as composition, logical product, and logical division.

— Abstract, PDF p. 2.

### 4. FlashAttention (Dao et al. — NeurIPS 2022)

**Bibliography.** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. "FlashAttention: Fast
and Memory-Efficient Exact Attention with IO-Awareness." In *Advances in Neural Information Processing
Systems 35 (NeurIPS 2022)*, Main Conference Track, eds. S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave,
K. Cho, A. Oh. Curran Associates, pp. 16344–16359. DOI 10.52202/068431-1189, as given in the proceedings
BibTeX. Also published as arXiv:2205.14135v2 (23 Jun 2022).

**Read.** NeurIPS proceedings PDF
https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf
→ `lit2/fl00_neurips/fa_neurips.pdf`.
The quotes and PDF pages below come from this file. I also read arXiv v2 →
`lit2/fl04_flashattention/2205.14135.pdf`.
It has the same passages, except that it writes "Q, K, V".

**Property it names.** It names being "IO-aware—accounting for reads and writes between levels of GPU
memory", and "uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory
(HBM) and GPU on-chip SRAM".

> We argue that a missing principle is making attention algorithms IO-aware—accounting for reads and writes
> between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses
> tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip
> SRAM.

— Abstract, PDF p. 1.

> However, common Python interfaces to deep learning such as PyTorch and Tensorflow do not allow fine-grained
> control of memory access.

— §1, PDF p. 2.

> Kernel fusion. The most common approach to accelerate memory-bound operations is kernel fusion: if there are
> multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple
> times for each operation.

— §2.1 Hardware Performance, PDF p. 3.

> The main idea is that we split the inputs Q,K,V into blocks, load them from slow HBM to fast SRAM, then
> compute the attention output with respect to those blocks. By scaling the output of each block by the
> right normalization factor before adding them up, we get the correct result at the end.

— §3.1 An Efficient Attention Algorithm With Tiling and Recomputation, PDF p. 4.

> Implementation details: Kernel fusion. Tiling enables us to implement our algorithm in one CUDA kernel,
> loading input from HBM, performing all the computation steps (matrix multiply, softmax, optionally masking
> and dropout, matrix multiply), then write the result back to HBM (masking and dropout in Appendix B). This
> avoids repeatedly reading and writing of inputs and outputs from and to HBM.

— §3.1, PDF p. 5.

> Using multiple GPUs adds an additional layer to IO analysis—accounting for data transfer between GPUs.

— §5 Limitations and Future Directions ("Multi-GPU IO-Aware Methods"), PDF p. 10.

---

## B. Whole-graph lowering into fused kernels

### 5. PyTorch 2 (Ansel et al. — ASPLOS 2024)

**Bibliography.** Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky,
Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban
Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock
Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai
Lu, CK Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi,
Helen Suk, Michael Suo, Phil Tillet, Eikan Wang, Xiaodong Wang, William Wen, Shunting Zhang, Xu Zhao, Keren
Zhou, Richard Zou, Ajit Mathews, Gregory Chanan, Peng Wu, Soumith Chintala. "PyTorch 2: Faster Machine
Learning Through Dynamic Python Bytecode Transformation and Graph Compilation." In *29th ACM International
Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24)*,
April 27–May 1, 2024, La Jolla, CA, USA. ACM, pp. 929–947. DOI 10.1145/3620665.3640366 (Crossref).

**Read.** https://pytorch.org/assets/pytorch2-2.pdf, which redirects to
https://docs.pytorch.org/assets/pytorch2-2.pdf. It is an 18-page copy carrying the ACM rights block, DOI and
ACM Reference Format →
`lit2/fl05_pytorch2/pytorch2-2.pdf`.
Pages below are this PDF's own; I did not map them to 929–947. I read the abstract, §1 and §4.

**Property it names.** It names "graph-level optimizations" that "cross operator boundaries", reached by
"lowering from an FX graph of PyTorch operations into TorchInductor’s define-by-run IR". A scheduler then
"determines which operators get fused", and "Triton codegen" follows.

> TorchInductor is the default compiler backend for TorchDynamo, which translates PyTorch programs into
> OpenAI’s Triton for GPUs and C++ for CPUs.

— Abstract, PDF p. 2.

> The downside of eager mode frameworks is that they make it harder to apply graph-level optimizations through
> compilers. The framework only has visibility of a single operator at a time, and thus cannot automatically
> perform optimizations, like fusion or scheduling, that cross operator boundaries.

— §1, PDF p. 2.

> Tensors have exposed strides that can be manipulated by users, aliasing views are commonplace, and both data
> and metadata can be mutated in-place.

— §4.1 ("PyTorch Native"), PDF p. 8.

> The next phase of compilation is lowering from an FX graph of PyTorch operations into TorchInductor’s
> define-by-run IR.

— §4.3 Lowerings and Define-By-Run Loop-Level IR, PDF p. 8.

> Lowering is done by symbolically interpreting the FX graph and applying lowering functions which do the
> conversion for a single operator. At the time of writing, TorchInductor has lowerings for 433 PyTorch
> operators (1605 including overloads). If an unknown operator is encountered, it is automatically converted
> into a fallback kernel node which runs the original PyTorch code.

— §4.3, PDF p. 9.

> TensorBox and StorageBox are abstractions that match PyTorch torch.Tensor and torch.Storage objects and
> allow the handling of views, aliasing, and mutation during the lowering process.

— §4.3, PDF p. 9.

> The scheduling phase of TorchInductor determines which operators get fused, what order kernels run in, and
> does memory planning for buffer removal and/or reuse.

— §4.4 Scheduling, PDF p. 9.

> ExternKernelSchedulerNode represents calls to library code or user-defined kernels. [...] Finally, a
> FusedSchedulerNode represents a set of two or more SchedulerNodes fused into a single kernel.

— §4.4, PDF p. 9.

> The fusion score orders fusions by: 1) the category of the fusion (e.g. pointwise/reduction/template); 2)
> estimated bytes of memory traffic saved by the fusion; and 3) shorter distance between nodes in the
> original graph

— §4.4, PDF p. 10.

> Triton codegen is responsible for mapping TorchInductor’s IR to output Triton [46] kernels.

— §4.5 Triton Code Generation, PDF p. 10.

> For more complex operations (matmuls and convolutions), TorchInductor has its own template system for
> generating Triton code that mixes handwritten Triton with generated Triton.

— §4.5, PDF p. 10.

> When enabled with mode="reduce-overhead", TorchInductor uses CUDA Graphs [20] to completely eliminate the
> overhead from wrapper code. CUDA Graphs records and replays kernel launches at the CUDA driver level and is
> lower overhead than even the C++ wrapper code.

— §4.7 Wrapper Codegen, PDF p. 10.

### 6. TVM (Chen et al. — OSDI 2018)

**Bibliography.** Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen
Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy. "TVM: An Automated End-to-End
Optimizing Compiler for Deep Learning." In *Proceedings of the 13th USENIX Symposium on Operating Systems
Design and Implementation (OSDI '18)*, October 8–10, 2018, Carlsbad, CA, USA. USENIX Association, ISBN
978-1-939133-08-3. The PDF's printed pages run 579–594. Also published as arXiv:1802.04799v3 (5 Oct 2018).

**Read.** https://www.usenix.org/system/files/osdi18-chen.pdf →
`lit2/fl06_tvm/osdi18-chen.pdf`.
The quotes and printed page numbers below come from this file. I also read arXiv v3 →
`lit2/fl06_tvm/1802.04799.pdf`,
which has the same text.

**Property it names.** It names "high-level operator fusion, mapping to arbitrary hardware primitives, and
memory latency hiding", and calls the mapping onto tensor instructions "tensorization".

> We propose TVM, a compiler that exposes graph-level and operator-level optimizations to provide
> performance portability to deep learning workloads across diverse hardware back-ends. TVM solves
> optimization challenges specific to deep learning, such as high-level operator fusion, mapping to arbitrary
> hardware primitives, and memory latency hiding.

— Abstract, p. 579.

> Computational graphs provide a global view of operators, but they avoid specifying how each operator must
> be implemented.

— §3 Optimizing Computational Graphs, p. 581.

> Operator fusion combines multiple operators into a single kernel without saving the intermediate results in
> memory. This optimization can greatly reduce execution time, particularly in GPUs and specialized
> accelerators.

— §3 "Operator Fusion", p. 581.

> This approach is no longer sustainable when targeting an increasing number of hardware back-ends since the
> required number of fused pattern implementations grows combinatorially with the number of data layouts,
> data types, and accelerator intrinsics that must be supported.

— §3, p. 582.

> This process builds on Halide’s idea of decoupling descriptions from computation rules (or schedule
> optimizations) [32] and extends it to support new optimizations (nested parallelism, tensorization, and
> latency hiding) and a wide array of hardware back-ends.

— §4 Generating Tensor Operations, p. 582.

> Adopting the decoupled compute/schedule principle from Halide [32], we use a schedule to denote a specific
> mapping from a tensor expression to low-level code.

— §4.1 Tensor Expression and Schedule Space, p. 583.

> We introduce the concept of memory scopes to the schedule space so that a compute stage (AS and BS in the
> code) can be marked as shared.

— §4.2 Nested Parallelism with Cooperation, p. 584.

> These new primitives create both opportunities and challenges for schedule-based compilation; while using
> them can improve performance, the compilation framework must seamlessly integrate them. We dub this
> tensorization: it is analogous to vectorization for SIMD architectures but has significant differences.
> Instruction inputs are multidimensional, with fixed or variable lengths, and each has different data
> layouts.

— §4.3 Tensorization, p. 584.

> We make tensorization extensible by separating the target hardware intrinsic from the schedule with a
> mechanism for tensor-intrinsic declaration. We use the same tensor expression language to declare both the
> behavior of each new hardware intrinsic and the lowering rule associated with it.

— §4.3, p. 584.

> Additionally, we introduce a tensorize schedule primitive to replace a unit of computation with the
> corresponding intrinsics. The compiler matches the computation pattern with a hardware declaration and
> lowers it to the corresponding hardware intrinsic. Tensorization decouples the schedule from specific
> hardware primitives, making it easy to extend TVM to support new hardware architectures.

— §4.3, p. 584.

> Latency hiding refers to the process of overlapping memory operations with computation to maximize
> utilization of memory and compute resources.

— §4.4 Explicit Memory Latency Hiding, p. 584.

### 7. Halide (Ragan-Kelley et al. — PLDI 2013)

**Bibliography.** Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, Saman
Amarasinghe. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in
Image Processing Pipelines." In *Proceedings of the 34th ACM SIGPLAN Conference on Programming Language
Design and Implementation (PLDI '13)*, June 16–21, 2013, Seattle. ACM, pp. 519–530. DOI
10.1145/2491956.2462176 (Crossref).

**Read.** I read the author-hosted PDF, which carries the block "PLDI '13 June 16-21, Seattle. Copyright c
2013 ACM 978-1-4503-2014-6/13/06": https://people.csail.mit.edu/jrk/halide-pldi13.pdf →
`lit2/fl07_halide/halide-pldi13.pdf`
(12 pages). I read the abstract, §1, the opening of §2, §3.2, §4.6 and part of §7.

**Property it names.** It names "decoupling the algorithm definition from its execution strategy" by "a
schedule representation", together with "fusion for producer-consumer locality across stages".

> We present a systematic model of the tradeoff space fundamental to stencil pipelines, a schedule
> representation which describes concrete points in this space for each stage in an image processing
> pipeline, and an optimizing compiler for the Halide image processing language that synthesizes high
> performance implementations from a Halide algorithm and a schedule.

— Abstract, PDF p. 1.

> Libraries of optimized subroutines do not solve the problem, either, since many critical optimizations
> involve fusion for producer-consumer locality across stages.

— §1 Introduction, PDF p. 1.

> We address this challenge by raising the level of abstraction, and decoupling the algorithm definition from
> its execution strategy, to improve portability and composability, while automating the search for
> optimized mappings of the resulting pipelines to parallel machines and complex memory hierarchies.

— §1, PDF p. 1.

> a DSL compiler based on this representation that combines Halide programs and schedule descriptions to
> synthesize points anywhere in this space, using a design where the choices for how to execute a program are
> separated not just from the definition of what to compute, but are pulled all the way outside the black box
> of the compiler;

— §1.2 Contributions, PDF p. 2.

> Our split representation, which separates schedules from the underlying algorithm, combined with the
> inside-out design of our compiler, allows our compiler to automatically search for the best schedule.

— §1.2, PDF p. 2.

> We introduce a model for the space of important choices in scheduling stencil pipelines, based on each stage
> choosing at what granularity to compute each of its inputs, at what granularity to store each for reuse,
> and within those grains, in what order its domain should be traversed (Fig. 4).

— §3.2 A Model for the Scheduling Choice Space, PDF p. 5.

> GPU kernel launches are modeled as dimensions (loops) scheduled to be parallel and annotated with the GPU
> block and thread dimensions to which they correspond.

— §4.6 Back-end Code Generation ("GPU Code Generation"), PDF p. 8.

> The end result is not individual GPU kernels, but large graphs of hybrid CPU/GPU execution, described by
> the same scheduling model which drives the CPU backends.

— §4.6, PDF p. 9.

---

## C. Sharding and collectives in the compiled graph

### 8. GSPMD (Xu et al.)

**Bibliography.** Yuanzhong Xu, HyoukJoong Lee, Dehao Chen, Blake Hechtman, Yanping Huang, Rahul Joshi,
Maxim Krikun, Dmitry Lepikhin, Andy Ly, Marcello Maggioni, Ruoming Pang, Noam Shazeer, Shibo Wang, Tao Wang,
Yonghui Wu, Zhifeng Chen (Google). "GSPMD: General and Scalable Parallelization for ML Computation Graphs."
arXiv:2105.04663v2 [cs.DC], 23 Dec 2021 (v1 10 May 2021). The arXiv license is non-exclusive distribution
1.0. I did not establish any other venue (see the last section).

**Read.** https://arxiv.org/pdf/2105.04663 (v2) →
`lit2/fl08_gspmd/2105.04663.pdf`.
I read the abstract, §1, §2.2, §3.1, §3.5 and parts of §4.

**Property it names.** It "separates the concerns of machine learning model programming and parallelism".
The user gives annotations, and "compiler passes [...] complete the sharding specification on the entire
computation graph". The output is "a single program for all partitions".

> We present GSPMD, an automatic, compiler-based parallelization system for common machine learning
> computations. It allows users to write programs in the same way as for a single device, then give hints
> through a few annotations on how to distribute tensors, based on which GSPMD will parallelize the
> computation.

— Abstract, p. 1.

> GSPMD separates the concerns of machine learning model programming and parallelism. It allows users to
> write programs with giant tensors as if there were a single giant device. Then the user can insert
> annotations in a few places that specify how tensors are distributed across devices; GSPMD will run
> compiler passes that complete the sharding specification on the entire computation graph, and transform it
> into a mathematically equivalent, parallelized computation to run on each device.

— §1 Introduction, p. 1.

> Generating one program for each partition would increase compilation time significantly, so GSPMD instead
> produces a single program for all partitions. This property is called Single Program Multiple Data (SPMD),
> and is crucial for scaling to thousands of partitions.

— §1, p. 1.

> The automatic partitioner in GSPMD is implemented as transformation passes in the XLA compiler [3].

— §2.2 XLA and TensorFlow, p. 2.

> This representation is simple yet general, because it treats all tensor dimensions in the same way and does
> not specialize for batch dimensions (data parallelism [22]), weight dimensions (model-parallelism or
> weight-update sharding [40]), or image dimensions (spatial partitioning [11]).

— §3.1 Representation of tensor sharding, p. 3.

> Devices can be organized in a logical multi-dimensional tensor called a device mesh, which can be used with
> an intuitive API.

— §3.1, p. 3.

> When generating collective communication operators, GSPMD preserves the order of devices in device_mesh.
> Therefore, device_mesh can be configured by the user in order to optimize communication based on the
> topology of the device network.

— §3.1, p. 3.

> Iterative, priority-based sharding propagation. To complete the sharding assignment on all tensors, GSPMD
> runs the propagation pass on the entire graph in multiple iterations, and alternates between forward
> propagation (input to output) and backward propagation (output to input).

— §3.5 Intuitive sharding completion, p. 5.

> Since the partitioner forces all the devices to run the same program, the communication patterns are
> regular. We use XLA’s operators that provide MPI-style collective communications [26].

— §4.2 Communication primitives, p. 7.

### 9. Mesh-TensorFlow (Shazeer et al. — NeurIPS 2018)

**Bibliography.** Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish Vaswani, Penporn Koanantakool,
Peter Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff Young, Ryan Sepassi, Blake Hechtman (Google Brain).
"Mesh-TensorFlow: Deep Learning for Supercomputers." In *Advances in Neural Information Processing Systems
31 (NeurIPS 2018)*, eds. S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, R. Garnett.
Curran Associates. The proceedings BibTeX gives no page range. Also published as arXiv:1811.02084v1
(5 Nov 2018).

**Read.** NeurIPS proceedings PDF
https://proceedings.neurips.cc/paper_files/paper/2018/file/3a37abdeefe1dab1b30f7c5c7e581b93-Paper.pdf →
`lit2/fl00_neurips/mtf_neurips.pdf`.
The quotes and pages below come from this file. I also read arXiv v1 →
`lit2/fl09_meshtf/1811.02084.pdf`.
I read §§1–8.

**Property it names.** "the user can specify any tensor-dimensions to be split across any dimensions of a
multi-dimensional mesh of processors". The graph "compiles into a SPMD program consisting of parallel
operations coupled with collective communication primitives".

> We introduce Mesh-TensorFlow, a language for specifying a general class of distributed tensor computations.
> Where data-parallelism can be viewed as splitting tensors and operations along the "batch" dimension, in
> Mesh-TensorFlow, the user can specify any tensor-dimensions to be split across any dimensions of a
> multi-dimensional mesh of processors. A Mesh-TensorFlow graph compiles into a SPMD program consisting of
> parallel operations coupled with collective communication primitives such as Allreduce.

— Abstract, PDF p. 1.

> We define a mesh as an n-dimensional array of such processors. The mesh is only a naming abstraction and
> does not imply a physical network topology.

— §2 Hardware Assumptions, PDF p. 2.

> A global "computation layout" is a partial map from tensor-dimension to mesh-dimension specifying which
> tensor-dimensions are split across which dimensions of the processor-mesh.

— §4 Mesh-TensorFlow: Beyond Batch Splitting, PDF p. 3.

> The layout of a tensor is an injective partial map from the tensor’s dimensions to dimensions of the mesh,
> and is computed as the restriction of the global computation layout to that tensor’s dimensions.

— §5 Tensor Representations, PDF p. 3.

> Each operation is implemented by parallel computation on every processor, and sometimes collective
> communication.

— §6 Operation Implementation, PDF p. 3.

> Einsum is implemented by parallel einsum operations on each processor of that processor’s input slices,
> followed by MPI-allreduce across any mesh dimensions corresponding to reduced-out Tensor dimensions.

— §6 ("Einstein Summation"), PDF p. 3.

> The user builds a Mesh-TensorFlow graph in python, which the library "lowers" to generate part of a
> TensorFlow graph.

— §7 Mesh-TensorFlow syntax, PDF p. 4.

> The code above defines only the mathematical model. We now discuss several different computation layouts.
> Each will produce identical results, but will have different performance characteristics.

— §8 Example: Two Fully-Connected Layers, PDF p. 4.

> When computing y, the split hidden dimension is reduced out. Consequently, the results of that computation
> get allreduced across all processors.

— §8.2 Model-Parallel Layout, PDF p. 5.

---

## D. Communication fused into compute tiles

### 10. FLUX (Chang et al.)

**Bibliography.** Li-Wen Chang, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Yinmin Zhong, Xuanrun
Zhang, Zuquan Song, Ziheng Jiang, Chengji Yao, Haibin Lin, Xin Jin, Xin Liu (ByteDance Ltd; Peking
University). "FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion."
arXiv:2406.06858v5 [cs.LG], 23 Oct 2024 (v1 11 Jun 2024). The arXiv license is non-exclusive distribution
1.0. I did not establish any other venue.

**Read.** https://arxiv.org/pdf/2406.06858 (v5) →
`lit2/fl10_flux/2406.06858.pdf`.
I read the abstract and §§1–3.

**Property it names.** It "overdecomposes communication and computation operations into much finer-grained
operations and further fuses them into a larger kernel". ReduceScatter becomes "epilogue fusion" and
AllGather "prologue fusion".

> This paper proposes a novel method, Flux, to significantly hide communication latencies with dependent
> computations for GPUs. Flux overdecomposes communication and computation operations into much finer-grained
> operations and further fuses them into a larger kernel to effectively hide communication without
> compromising kernel efficiency.

— Abstract, p. 1.

> These methods might have several limitations on GPUs, such as no precise control of execution timing on
> GPUs when using streams, and poor GPU utilization for executing multiple smaller kernel instances.

— §1, p. 2.

> [...] Fine-grained Communication Overlapping (Flux), that decomposes the original communication and
> computation into much finer-grained tiles than the existing methods, and then fuses tiled computation and
> communication into a single larger kernel. In the fused kernel, each dependent computation and
> communication tile is mapped into each thread block

— §1, p. 2. The sentence ends with footnote marker 1, whose text reads: "Depending on GEMM implementations, a
tile can also be mapped into a warp or a thread block cluster as well."

> Flux optimizes communication together with computation, including kernel fusion, tile coordinate
> swizzling, GPU instruction selection, communication order selection, etc.

— §1, p. 2.

> The common partitioning strategy we discuss in the paper is an extended Megatron-LM [24] with sharded
> activation [13, 25].

— §2.1 Common Partitioning and Communication Patterns, p. 3.

> The first GEMM shards the weight (W1) along the the row direction, and AllGathers the sharded input
> activations along the column direction before GEMM, while the second GEMM shards the weight (W2) along the
> column direction, and ReduceScatters the output activation along the column direction.

— §2.1, p. 3.

> While the execution order and concurrent execution among GPU kernels can be achieved through streams and
> events<sup>3</sup>, however, the execution timing is not trivially controlled by most GPU programming
> models.

— §2.2 Conventional Communication Overlapping Strategies, p. 4.

> Here, since the computation operation is GEMM, and most high-performance GEMM kernels on GPUs are written
> with tiling, such as thread block tiling or warp tiling, our decomposition can naturally map into existing
> tiling in the kernels. Flux fuses dependent communication and/or wait logic into a GEMM kernel, and
> launches only one fused kernel, compared to the prior methods launching multiple split GEMM kernels.

— §2.3 Overview of Fused GEMM with Communication, p. 6.

> In Flux, ReduceScatter is implemented as epilogue fusion into a GEMM kernel. More specifically,
> ReduceScatter communication is fused into the epilogue of the GEMM kernel.

— §3.1 ReduceScatter Overlapping, p. 6.

> Unlike a standard GEMM kernel having only one single output pointer, the number of output pointers (Cs) in
> the fused GEMM kernel is increased to the number of devices in tensor parallelism [...]

— §3.1, p. 6.

> This algorithm requires GPUs with peer-to-peer (P2P) supports, which modern NVIDIA GPUs within a node
> already have, regardless with NVLink or PCIe interconnects.

— §3.1, p. 6.

> Different from ReduceScatter, AllGather is implemented as prologue fusion into a GEMM kernel. More
> specifically, AllGather signal checking is fused into the prologue of the GEMM kernel.

— §3.2 AllGather Overlapping, p. 7.

> On the kernel side, GEMM tile computation is blocked by the function WaitSignal until the value contained
> in the signal is set to true.

— §3.2, p. 7.

> Meanwhile, in AllGather, the tiling strategy of communication (tiles<sub>comm</sub>) is decoupled from the
> tiling strategy of GEMM computation.

— §3.2, p. 8.

### 11. TileLink (Zheng et al. — MLSys 2025)

**Bibliography.** Size Zheng, Jin Fang, Xuegui Zheng, Qi Hou, Wenlei Bao, Ningxin Zheng, Ziheng Jiang,
Dongyang Wang, Jianxi Ye, Haibin Lin, Li-Wen Chang, Xin Liu (ByteDance Seed). "TileLink: Generating
Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives." In *Proceedings of
Machine Learning and Systems 7 (MLSys 2025)*, eds. M. Zaharia, G. Joshi, Y. Lin. This is taken from the MLSys
proceedings page and BibTeX, which give no page range. Also published as arXiv:2503.20313v3 [cs.DC],
3 Apr 2025, with arXiv license CC BY-NC-ND 4.0.

**Read.** I read arXiv v3, https://arxiv.org/pdf/2503.20313 →
`lit2/fl11_tilelink/2503.20313.pdf`
(abstract, §1, §2, §3.2, §4.1–4.3). I did not read the MLSys PDF.

**Property it names.** It "decouples the design space of communication and computation, linking these two
parts via tile-centric primitives". These are signal primitives with release/acquire semantics and data
primitives, compiled by "tile-centric mapping".

> In the frontend, TileLink decouples the design space of communication and computation, linking these two
> parts via tile-centric primitives. In the backend, TileLink translates these primitives into low-level
> communication instructions, integrating the communication and computation components to achieve
> overlapped execution.

— Abstract, p. 1.

> The core idea is to map communication and computation to distinct hardware units, allowing them to operate
> concurrently. To handle data dependency between communication and computation operators, synchronization
> or barriers are inserted into the loop of data transfer and computation.

— §1, p. 2.

> However, synchronization between these decomposed kernels necessitates host intervention, introducing
> non-negligible overhead at runtime.

— §1, p. 2.

> On the other hand, the kernel fusion method [5, 15] combines communication and computation kernels into
> one fused kernel to overcome the above disadvantages. Within fused kernels, communication is mapped to
> either DMA (Direct Memory Access) engines or processing cores (e.g., streaming multiprocessors on a GPU),
> while computation is executed simultaneously on other processing cores. Data dependencies are managed using
> on-device barriers, and processing cores responsible for data transfer communicate with computation cores
> through atomic or communication instructions.

— §1, p. 2.

> These primitives need to perform system synchronization before and after data transfer to follow the SPMD
> (Single Program, Multiple Data) programming model and integrate seamlessly with other operators. However,
> coarse-grained synchronization can cause computational units to be idle during communication, thus
> reducing computational efficiency. We call these communication primitives operator-centric primitives.

— §2 ("Operator-Centric Communication Primitives"), p. 3.

> These primitives introduce memory consistency semantics and adhere to the tile-level abstraction utilized
> in the compiler, distinguishing them from the operator-centric primitives provided by previous frameworks
> [1, 2] and libraries [26].

— §3.2 Tile-Centric Primitives, p. 6.

> Signal primitives are designed to manage barriers between communication and computation.

— §3.2.1 Signal Primitives, p. 7.

> The notify primitives carry release semantics, ensuring that any memory access occurring before
> producer(peer)_tile_notify and rank_notify cannot be executed after these notify primitives. Conversely,
> the wait primitives carry acquire semantics, ensuring that any memory access following
> consumer(peer)_tile_wait and rank_wait cannot be executed before these wait primitives.

— §3.2.1 ("Memory Consistency"), p. 7.

> The device-side tile_push(pull)_data primitive maps communication to processing cores, while the host-side
> rank_copy_data primitive maps communication to the copy engine.

— §3.2.2 Data Primitives, p. 7.

> Tile-centric mapping consists of three components: shape mapping, rank mapping, and channel mapping. Shape
> mapping associates each tile_id with a specific tensor shape slice. Rank mapping links each tile_id to a
> device rank. Channel mapping assigns each tile_id to a communication barrier.

— §4.1 Tile-Centric Mapping, p. 7.

> Static mapping refers to mappings that can be decided at compile time. Static mapping is commonly used when
> data sharding strategy is fixed such as tensor-parallel MLP and sequence-parallel self-attention.

— §4.1, p. 8.

> To address this issue, TileLink enforces strict data dependencies between its primitives and their
> following load/store operations so that its primitives can be correctly reordered and unrolled by
> pipelining passes.

— §4.2 Compilation for Memory Consistency, p. 8.

### 12. CoCoNet (Jangda et al. — ASPLOS 2022)

**Bibliography.** Abhinav Jangda, Jun Huang, Guodong Liu, Amir Hossein Nodehi Sabet, Saeed Maleki, Youshan
Miao, Madanlal Musuvathi, Todd Mytkowicz, Olli Saarikivi. "Breaking the Computation and Communication
Abstraction Barrier in Distributed Machine Learning Workloads." In *Proceedings of the 27th ACM International
Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '22)*, February
28 – March 4, 2022, Lausanne, Switzerland. ACM, pp. 402–416. DOI 10.1145/3503222.3507778 (Crossref).

**Read.** I read arXiv:2105.05720v5 (26 Mar 2022), which carries the ASPLOS '22 rights block, DOI and ACM
Reference Format →
`lit2/fl12_coconet/2105.05720.pdf`
(15 pages). I read the abstract, §1, §2, the opening of §3, §3.1, §3.4, §5.2 and §5.3.

**Property it names.** It names breaking "the current logical separation between computation and
communication kernels", with "computation and communication as first class constructs". The optimizations
are "fusion or overlapping of communication and computation" and reorder.

> However, the current logical separation between computation and communication kernels in machine learning
> frameworks misses optimization opportunities across this barrier.

— Abstract, PDF p. 1.

> Therefore, we present CoCoNet, which contains (i) a domain specific language to express a distributed
> machine learning program in the form of computation and communication operations, (ii) a set of semantics
> preserving transformations to optimize the program, and (iii) a compiler to generate jointly optimized
> communication and computation GPU kernels.

— Abstract, PDF p. 1.

> In machine learning systems today, computation and communication are treated as independent abstractions
> implemented in different libraries.

— §1 Introduction, PDF p. 1.

> Fusion optimization decreases memory bandwidth usage by generating a single kernel to perform multiple
> communication and computation operations. Reorder optimization moves the computation before or after the
> communication, thereby either distributing the computation or enabling new fusion possibilities. Finally,
> overlapping optimization orchestrates multiple computation and communication operations in a fine-grained
> manner to fully utilize both network and compute resources.

— §1, PDF p. 2.

> The idea is to slice the output into smaller chunks and start the AllReduce communication on a chunk as
> soon as the MatMul kernel has computed it. To ensure minimum wait time for the AllReduce kernel, we need to
> schedule the MatMul kernel to compute chunks in the order the AllReduce kernel communicates them.

— §1, PDF p. 2.

> Furthermore, we need to invoke only one MatMul kernel and AllReduce kernel to avoid the overhead of
> launching multiple kernels.

— §1, PDF p. 2.

> Inspired by Halide [42], CoCoNet includes a scheduling language to specify an execution schedule of the
> program using a set of transformations.

— §1, PDF p. 2.

> Besides item datatype, like FP32 and FP16, and shape, a CoCoNet tensor also includes a layout that
> describes the distributed allocation of tensor’s data across a set of ranks. There are three layouts for a
> tensor: sliced, replicated, and local.

— §2.1 Tensor Layout, PDF p. 2.

> FusedAllReduce avoids such stores and loads by directly passing the output of communication to following
> computations through registers.

— §2.3 Fused Collective Communication Operations, PDF p. 3.

> CoCoNet provides four semantics preserving transformations to optimize a program written in the DSL.

— §3 CoCoNet Transformations, PDF p. 4.

> We call an order of transformations a schedule.

— §3, PDF p. 4.

> CoCoNet provides the overlap transformation to overlap a series of producer-consumer operations to utilize
> multiple resources of hardware simultaneously.

— §3.4 Overlapping Operations, PDF p. 5.

> A similar approach for overlapping of computation and communication operations for a GPU workload would
> involve dividing all operations into sub-operations and ensuring dependency between sub-operations using
> CUDA streams. However, this approach would provide sub-optimal performance because each sub-operation is
> performed on only a part of data, which leads to in-efficient computation and under-utilization of
> communication bandwidth.

— §5.3 Overlapping of Communication and Computation, PDF p. 7.

> First, it schedules the MatMul kernel (based on CUTLASS [4]) to produce chunks in the same order as the
> AllReduce consumes them.

— §5.3, PDF p. 7. The PDF's running header falls between "(based on" and "CUTLASS" at a column break.

> Second, CoCoNet invokes both kernels only once on different streams and synchronizes the AllReduce with
> the MatMul using an efficient fine-grained spin-lock on a memory buffer to ensure that the AllReduce wakes
> up as soon as the MatMul produces a chunk.

— §5.3, PDF p. 7.

### 13. Triton-distributed (Zheng et al.)

**Bibliography.** Size Zheng, Wenlei Bao, Qi Hou, Xuegui Zheng, Jin Fang, Chenhui Huang, Tianqi Li, Haojie
Duanmu, Renze Chen, Ruifan Xu, Yifan Guo, Ningxin Zheng, Ziheng Jiang, Xinyi Di, Dongyang Wang, Jianxi Ye,
Haibin Lin, Li-Wen Chang, Liqiang Lu, Yun Liang, Jidong Zhai, Xin Liu (ByteDance Seed; Tsinghua; Peking;
Shanghai Jiao Tong; Zhejiang). "Triton-distributed: Programming Overlapping Kernels on Distributed AI
Systems with the Triton Compiler." arXiv:2504.19442v3 [cs.DC], 5 Jun 2025 (v1 28 Apr 2025). The arXiv
license is non-exclusive distribution 1.0. I did not establish any other venue.

**Read.** https://arxiv.org/pdf/2504.19442 (v3) →
`lit2/fl13_triton_distributed/2504.19442.pdf`.
I read the abstract, §1, §2 and §3.1.

**Property it names.** It names "native overlapping optimizations for distributed AI workloads", with
communication primitives "compliant with the OpenSHMEM standard" integrated "into the compiler". The model
rests on "symmetric memory, signal exchange, and async-task".

> In a distributed system, there are three fundamental activities occur concurrently: computation, memory
> access, and communication. In existing training/inference frameworks, these aspects are often optimized
> independently at different programming levels. As a result, it is difficult for these activities to
> coordinate with each other and unleash the full performance potential of the cluster.

— Abstract, p. 1.

> First, we integrate communication primitives compliant with the OpenSHMEM standard into the compiler. This
> enables programmers to utilize these primitives with a higher-level Python programming model.

— Abstract, p. 1.

> Triton-distributed supports native fine-grained computation-communication overlapping using
> compiler-assisted primitives.

— §1, p. 4.

> Our compiler translates these primitives into NVSHMEM/ROCSHMEM implementations during compilation,
> enabling communication across both single-node and multi-node GPU clusters. Notably, implementing these
> workloads requires minimal modifications to Triton’s original compute code: developers only focus on adding
> communication logic.

— §1, p. 4.

> The communication part is implemented with OpenSHMEM primitives, which are further lowered to LLVM IR along
> with a bitcode library. The computation part goes through the original Triton compilation stack to LLVM IR.

— §1, p. 4.

> Our programming model follows the MPMD (multiple programs multiple data) model. The MPMD model allows
> communication and computation tasks to run in parallel and cooperate with each other to complete a global
> task. The core of our programming model includes three concepts: symmetric memory, signal exchange, and
> async-task.

— §2.1 MPMD Programming: Symmetric Memory, Signal Exchange, and Async-Task, p. 4.

> Symmetric Memory: Each rank allocates a memory buffer in the global scope with the same size.

— §2.1, p. 4.

> Signal Exchange: Operations on each rank use signals to communicate with each other in a consistent
> manner. A signal is a data object that resides in symmetric memory.

— §2.1, p. 5.

> Async-Task: Operations such as data transfer and computation are treated as asynchronous tasks that run in
> parallel. Async-tasks can be synchronized through signals.

— §2.1, p. 5.

Table 1 ("Our Communication Primitives"), p. 5, gives these non-OpenSHMEM entries:

- `wait`: "Locally wait a signal until it equals to some given value"
- `ld_acquire`: "Load with acquire semantic"
- `red_release`: "Reduction add with release semantic"

> For a distributed system, the essence of designing primitives is to model communication patterns
> effectively. The design of communication primitives depends on the system’s architecture, such as the
> interconnection topology and bandwidth.

— §2.2 Communication Primitives, p. 6.

> For example, wait is used with consume_token to construct data dependency between signal operations, and
> the following MMA operations for better compiler-based pipelining.

— §2.2, p. 6.

> Intra-Node Swizzle: Swizzle is to change the order of communication operations and computation operations
> so that they can better overlap with other.

— §3.1 Optimization Approaches for Overlapping, p. 8.

> Copy Engine: GPUs and NPUs employ dedicated memory copy units to perform communication. Using these units
> for communication is also important for overlapping within single node.

— §3.1, p. 8.

---

## E. The Metal hardware's own primitives (local SDK)

**Source.** Apple Inc., macOS 26.4 SDK. `xcrun --sdk macosx --show-sdk-path` gives
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`, a symlink to `MacOSX26.4.sdk` (`--show-sdk-version`
26.4.1, build 25E251; SDKSettings `CanonicalName` "macosx26.4"). The MSL standard-library headers are not in
the SDK. They are in the Metal toolchain that `xcrun -f metal` resolves under `DEVELOPER_DIR=/Applications/Xcode-26.4.1.app/Contents/Developer`:
`/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.5.188.0.R8x4KP/Metal.xctoolchain`
(`metal --version`: "Apple metal version 32023.883 (metalfe-32023.883)"; toolchain identifier
`com.apple.dt.toolchain.Metal.32023.883`). The headers carry "Copyright (c) 2025 Apple Inc. All rights
reserved" (MPP and MSL headers) and "Copyright © 2024 Apple, Inc. All rights reserved." (`MTLTensor.h`). No
SDK license-agreement file was read. These files were read in place and not copied.

**Files read.**

- `/Library/Developer/CommandLineTools/SDKs/MacOSX26.4.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h` (642 lines, read in full)
- `.../MetalPerformancePrimitives.framework/Headers/MetalPerformancePrimitives.h`, `__impl/MPPTensorOpsBase.h`, `__impl/MPPTensorOpsTypes.h` (read in full)
- `/Library/Developer/CommandLineTools/SDKs/MacOSX26.4.sdk/System/Library/Frameworks/Metal.framework/Headers/MTLTensor.h` (216 lines, read in full); `MTLBuffer.h` and `MTLDevice.h` (the `newTensorWithDescriptor` declarations)
- `/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.5.188.0.R8x4KP/Metal.xctoolchain/usr/metal/32023/lib/clang/32023.883/include/metal/metal_cooperative_tensor` (all of its comments) and `.../include/metal/metal_tensor` (all of its comments, plus the declarations quoted)

**Property it names.** `matmul2d` "performs generalized matrix multiplication operation", executed by an
"execution scope". A `cooperative_tensor` "owns thread private data and divides the data for entire tensor
among threads". The tensor types are `MTLTensor`, "A resource representing a multi-dimensional array", and
the MSL `tensor` in "Host-bound" (`tensor_handle`) and "Shader-Allocated" (`tensor_inline`) forms.

#### 14a. `matmul2d` (MPPTensorOpsMatMul2d.h)

Lines 7–10:

```
// This API performs generalized matrix multiplication operation
//             C = A*B + C;
// A and B can be tensor_handle, tensor_offset, and tensor_inline.
// C can be tensor_handle, tensor_offset, tensor_inline or cooperative_tensor.
```

Lines 63–68:

```
// Basic usage is in the following example which takes M x K matrix A of type
// half, K x N matrix B of type half, both in device memory and produces M x N
// matrix C of type float in device memory. It tiles this matrix multiplication
// in thread groups, where each thread group computes a 64 x 32 tile of output
// but multiplying 64 x K tile of A with K x 32 tile of B. This compute kernel
// will be launched with dispatch grid of
```

Lines 101–104:

```
//    // create matmul op from above descriptor with 4 SIMD-Groups. All 4
//    // SIMD-Groups in this threadgroup will execute this
//    // matmul cooperatively. More on this scope below.
//    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
```

Lines 159–161:

```
// User can also take ownership of looping over reduction or k-dimension by
// choosing appropriate chunk size in k (called k-tile or tilek) For following
// example, we choose 16.
```

Lines 328, 349–360 and 391–395 (declarations):

```
#if defined(__METAL_VERSION__) && defined(__HAVE_TENSOR__)
```

```
struct matmul2d_descriptor
{
  enum class mode
  {
    multiply,
    multiply_accumulate,
  };

  int m, n, k;
  bool transpose_left, transpose_right;
  bool relaxed_precision;
  mode matmul_mode;
```

```
template <matmul2d_descriptor Descriptor, typename Scope, class... Args>
class matmul2d : __tensor_ops_detail::op
{
  static_assert(__tensor_ops_detail::__is_tensorops_execution_scope_v<Scope>,
                "Scope template argument should be of op_scope type");
```

#### 14b. `cooperative_tensor` (MPPTensorOpsMatMul2d.h and MSL `metal_cooperative_tensor`)

MPPTensorOpsMatMul2d.h lines 207–225:

```
// Often times, we need to do some post processing on computed results before
// storing to device or threadgroup memory. For example, in machine learning we
// need to apply activation function on compute value. One can do GEMM as above
// which writes the result to device memory, read the value back, call post
// processing function and write again. This results in wasted bandwidth,
// performance and power. User can apply post processing in-register where GEMM
// output is computed using cooperative_tensor. Unlike tensor_handle,
// tensor_offset and tensor_inline which are non-owning meaning these are
// wrappers around resource in device, threadgroup or thread address space,
// cooperative_tensor owns thread private data and divides the data for entire
// tensor among threads (participating in the scope of operation) in implementation
// defined manner. This thread private memory is allocated at construction of
// cooperative_tensor and deallocated when this cooperative_tensor goes out of
// scope. The layout of cooperative_tensor depends on operation, data type,
// number of threads in opscope with which op was created. Note that
// cooperative_tensor created from an op is only valid for threads that are part
// of execution scope on which op was created. Though the layout of cooperative_tensor
// is implementation defined, we provide accessor functions as shown in the
// example below
```

MPPTensorOpsMatMul2d.h lines 269–275:

```
//    // create cooperative bias tensor with same layout as destination
//    // cooperative_tensor of matmul
//    auto biasT = matmulOp.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
//
//    // load data from bias tensor_handle into biasT cooperative_tensor using
//    // layout and distribution of element among threads of scope on which matmul was created.
//    biasT.load(bias);
```

MPPTensorOpsMatMul2d.h lines 342–347 and 587–591 (reductions over cooperative tensors):

```
enum class reduction_operation
{
  sum,
  max,
  min,
};
```

```
template <class ElementType, class SrcExtents, class DstExtents, class SrcLayout, class DstLayout>
inline void reduce_rows(
    thread metal::cooperative_tensor<ElementType, SrcExtents, SrcLayout> &source,
    thread metal::cooperative_tensor<ElementType, DstExtents, DstLayout> &destination,
    reduction_operation op = reduction_operation::sum,
```

MPPTensorOpsMatMul2d.h lines 611 and 622–625:

```
// Returns whether the iterators are compatible between a source and destination cooperative tensor.
```

```
//     else {
//          // Fall back to storing sourceCT to threadgroup memory and access via
//          // destCT's multidimensional indices
//     }
```

`metal_cooperative_tensor` lines 239–249, 279–296 and 304–307:

```
// struct Layout {
//   // Returns the amount of storage each threads needs to allocate
//   // for this cooperative tensor.
//   static size_t thread_storage_size();
//
//   // Returns the alignment of the storage allocated in each thread
//   // for this cooperative_tensor.
//   static constexpr size_t thread_storage_align();
//
//   // Constructs the per-thread object.
//   static void construct(thread void *);
```

```
//   // Given a 1-D index, returns whether the corresponding element
//   // of the per-thread data is masked out.
//   static bool is_valid_element(const thread void *, uint16);
//
//   // Given a 1-D index, return the set of multi-dimension index.
//   template <class OtherIndexType, size_t Rank>
//   static array<OtherIndexType, Rank>
//   get_multidimensional_index(const thread void *, uint16_t);
//
//   // Load elements belonging to this thread into per-thread storage from
//   // the source tensor.
//   template <class T, class E, class D, class... Tags>
//   static void load(thread void *storage, const thread tensor<T, E, D, Tags...> &);
//
//   // Store elements belonging to this thread from per-thread storage into
//   // the destination tensor.
//   template <class T, class E, class D, class... Tags>
//   static void store(const thread void *storage, const thread tensor<T, E, D, Tags...> &);
```

```
template <class ElementType,
          class Extents,
          class Layout>
class [[sizeas(Layout::thread_storage_size())]] alignas(Layout::thread_storage_align()) cooperative_tensor
```

#### 14c. Execution scopes (MPPTensorOpsMatMul2d.h lines 296–321)

```
// Note on scope of operation
// ==========================
// A tensor operation may be executed on a single thread entirely or
// cooperatively among a set of SIMD groups. The set of threads is called the
// "execution scope" of the tensor operation. A tensor op must be created with
// an execution scope provided as template argument. All the threads in this
// execution scope must enter the run method i.e. call to run methods must be
// "execution scope" uniform. Use the following types to configure the execution
// modes of each operation:
//     metal::execution_thread: The operation will be run on a single thread.
//                              Fragment shaders only support this execution scope.
//     metal::execution_simdgroup: The operation will be run cooperatively by all
//                                 threads in the SIMD group. May be used for finer
//                                 control over tiling by slicing tensors with SIMD IDs.
//     metal::execution_simdgroups<N>: The operation will be executed cooperatively by N
//                                     SIMD groups. Must be used when all threads in a
//                                     threadgroup are cooperatively performing the operation.
//
// It is undefined behavior if the number of SIMD groups dispatched does not
// match the number of SIMD groups that the operation was configured with.
//
// Even though each thread in the execution scope can potentially independently
// enter and exit run method, the threads in the execution scope are working
// do not necessarily operate completely independently. For example, the tensor
// operation may need for (for correctness or performance) to synchronize
// among the threads in execution scope it was created with.
```

#### 14d. Metal 4 tensor types

`MTLTensor.h` lines 18, 35–36 and 154–156:

```
/// The possible data types for the elements of a tensor.
```

```
/// The largest rank a tensor can have.
#define MTL_TENSOR_MAX_RANK 16
```

```
/// A resource representing a multi-dimensional array that you can use with machine learning workloads.
MTL_EXTERN API_AVAILABLE(macos(26.0), ios(26.0))
@protocol MTLTensor <MTLResource>
```

`MTLTensor.h` lines 84–99:

```
/// The type that represents the different contexts for a tensor.
typedef NS_OPTIONS(NSUInteger, MTLTensorUsage)
{
    /// A tensor context that applies to compute encoders.
    ///
    /// You can use tensors with this context in ``MTL4ComputeCommandEncoder`` or ``MTLComputeCommandEncoder`` instances.
    MTLTensorUsageCompute = 1 << 0,
    /// A tensor context that applies to render encoders.
    ///
    /// You can use tensors with this context in ``MTL4RenderCommandEncoder`` or ``MTLRenderCommandEncoder`` instances.
    MTLTensorUsageRender  = 1 << 1,
    /// A tensor context that applies to machine learning encoders.
    ///
    /// You can use tensors with this context in ``MTL4MachineLearningCommandEncoder`` instances.
    MTLTensorUsageMachineLearning      = 1 << 2,
} API_AVAILABLE(macos(26.0), ios(26.0));
```

`MTLTensor.h` lines 111–119:

```
/// An array of strides, in elements, one for each dimension in the tensors you create with this descriptor, if applicable.
///
/// You are responsible for ensuring `strides` meets the following requirements:
/// - The first element of `strides` is one.
/// - If ``usage`` contains ``MTLTensorUsage/MTLTensorUsageMachineLearning``, the second element of `strides` is aligned to 64 bytes, and for any `i` larger than one, `strides[i]` is equal to `strides[i-1] * dimensions[i-1]`.
/// - If ``dataType`` is a sub-byte ``MTLTensorDataType``, for any `i` greater than or equal to 1, `strides[i]` is aligned to 128 bytes. This is not a requirement for non-sub-byte data types, but following this convention improves performance.
///
/// Only set this property when creating tensors from a buffer.
@property (readwrite, nonatomic, copy, nullable) MTLTensorExtents *strides;
```

`MTLTensor.h` lines 161–165:

```
/// A buffer instance this tensor shares its storage with or nil if this tensor does not wrap an underlying buffer.
@property (nullable, readonly) id<MTLBuffer> buffer;

/// An offset, in bytes, into the buffer instance this tensor shares its storage with, or zero if this tensor does not wrap an underlying buffer.
@property (readonly) NSUInteger bufferOffset;
```

`MTLBuffer.h` lines 68–70 and `MTLDevice.h` line 1338:

```
/// Creates a tensor that shares storage with this buffer.
///
/// `offset` must be 0 when ``MTLTensorDescriptor/usage`` contains ``MTLTensorUsage/MTLTensorUsageMachineLearning``.
```

```
/// Creates a tensor by allocating new memory.
```

MSL `metal_tensor` lines 221, 224–232, 322–330, 1392–1397, 1779 and 1803–1806:

```
constexpr bool __is_tensor_rank(size_t Rank) { return Rank <= 16; }
```

```
struct tensor_inline
{
};
struct tensor_handle
{
};
struct tensor_offset
{
};
```

```
template <class ElementType,
          class Extents,
          class Descriptor = tensor_handle,
          class... Tags>
struct tensor
{
  static_assert(__tensor_detail::__is_tensor_descriptor_v<Descriptor>,
                "tensor: Descriptor template parameter must be one of tensor_handle or tensor_inline");
};
```

```
// Host-bound tensors.

template <class ElementType,
          class Extents,
          class... Tags>
struct tensor<ElementType, Extents, tensor_handle, Tags...>
```

```
// Shader-Allocated Tensors
```

```
template <class ElementType,
          class Extents,
          class... Tags>
struct __TENSOR_INLINE_ATTRS(Extents) tensor<ElementType, Extents, tensor_inline, Tags...>
```

---

## Not read, or read only in part

- **ThunderKittens**: I read the arXiv v1 PDF, not the ICLR 2025 version. **TileLink**: I read arXiv v3, not the
  MLSys 2025 PDF. **Triton, Halide, PyTorch 2**: I read author-hosted copies that carry the ACM rights blocks,
  not the ACM Digital Library files. **CoCoNet**: I read arXiv v5, which carries the ASPLOS '22 block.
- **"Categorical Foundations for CuTe Layouts"** (arXiv:2601.05972): abstract only.
- **CuTe `03_tensor.md` and `0x_gemm_tutorial.md`**: downloaded, not quoted.
- **Venues of GSPMD, FLUX and Triton-distributed**: not established. DBLP's API returned a bot check and HTTP
  429, and Semantic Scholar's API returned HTTP 429. These three are cited as arXiv preprints.

---

## Table: properties of a final lowering, and the sources that state them

The first column quotes the phrase from the first source named. The second column gives verbatim
fragments from the passages above.

| Property (a source's phrase) | Sources that state it (verbatim fragment — locator) |
|---|---|
| "tile-level operations and optimizations" (Triton §2) | Triton: "the addition of tile-level operations and optimizations into traditional compilation pipelines" — §2<br>ThunderKittens: "we use a 16 × 16 matrix tile as our basic data structure" — §1<br>CuTe docs: "a large set of common operations such as tiling and partitioning" — 00_quickstart.md<br>MPP: "It tiles this matrix multiplication in thread groups" — MPPTensorOpsMatMul2d.h l. 65–66 |
| "fit a machine’s compute capabilities and memory hierarchy as tightly as possible" (Triton §5.2.1) | Triton: "decomposing tiles into micro-tiles and eventually nano-tiles" — §5.2.1<br>ThunderKittens: "Our abstractions map to the three levels of the GPU hierarchy" — Abstract<br>CuTe docs: "hierarchically multidimensional layouts of threads and data" — 00_quickstart.md<br>Halide: "optimized mappings of the resulting pipelines to parallel machines and complex memory hierarchies" — §1 |
| instructions "prescribe fixed, often complex data layouts" (Cecka, Abstract) | Cecka: "must be correctly propagated through the entire execution pipeline" — Abstract; "ensure correct invocation of hardware instructions" — §1<br>ThunderKittens: "keep memory in the layouts required by tensor core units" — §3.1<br>CuTe docs: "decouples the complexity of thread and data layouts from the call site of the PTX instruction" — 0t_mma_atom.md<br>TVM: "Instruction inputs are multidimensional, with fixed or variable lengths, and each has different data layouts." — §4.3<br>MPP: "The layout of cooperative_tensor depends on operation, data type, number of threads in opscope with which op was created." — l. 220–221 |
| "mapping to arbitrary hardware primitives" (TVM, Abstract) | TVM: "a tensorize schedule primitive to replace a unit of computation with the corresponding intrinsics" — §4.3<br>CuTe docs: "makes it possible to expose MMAs for use in generic CUDA C++ code" — 0t_mma_atom.md<br>ThunderKittens: "keeping tensor cores at high utilization whenever possible" — §1<br>MPP: "This API performs generalized matrix multiplication operation" — l. 7 |
| "IO-aware" (FlashAttention, Abstract) | FlashAttention: "uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM" — Abstract<br>Triton: "temporarily storing their operands in fast shared memory" — §5.2.3<br>ThunderKittens: "load tiles of large tensors from HBM to SRAM, perform computation in fast memory" — §3.2 |
| "Operator fusion" (TVM §3) | TVM: "combines multiple operators into a single kernel without saving the intermediate results in memory" — §3<br>FlashAttention: "Tiling enables us to implement our algorithm in one CUDA kernel" — §3.1<br>PyTorch 2: "a FusedSchedulerNode represents a set of two or more SchedulerNodes fused into a single kernel" — §4.4<br>Halide: "many critical optimizations involve fusion for producer-consumer locality across stages" — §1<br>MPP: "User can apply post processing in-register where GEMM output is computed using cooperative_tensor." — l. 212–213 |
| optimizations "that cross operator boundaries" (PyTorch 2 §1) | PyTorch 2: "The framework only has visibility of a single operator at a time" — §1<br>TVM: "Computational graphs provide a global view of operators" — §3; "the required number of fused pattern implementations grows combinatorially" — §3<br>Halide: "Libraries of optimized subroutines do not solve the problem, either" — §1<br>CoCoNet: "misses optimization opportunities across this barrier" — Abstract |
| "decoupling the algorithm definition from its execution strategy" (Halide §1) | Halide: "Our split representation, which separates schedules from the underlying algorithm" — §1.2<br>TVM: "Adopting the decoupled compute/schedule principle from Halide" — §4.1<br>CoCoNet: "Inspired by Halide [42], CoCoNet includes a scheduling language" — §1<br>Mesh-TensorFlow: "The code above defines only the mathematical model." — §8<br>GSPMD: "GSPMD separates the concerns of machine learning model programming and parallelism." — §1<br>Cecka: "Data layouts are declared independently of algorithmic logic" — §1 |
| "memory latency hiding" (TVM, Abstract) | TVM: "overlapping memory operations with computation to maximize utilization of memory and compute resources" — §4.4<br>Triton: "adding adequate prefetching code where necessary" — §5.1.1<br>ThunderKittens: "the template internally hides latencies through memory pipelines and synchronization primitives" — §1; "perfect overlapping between memory, compute, and tensor core costs" — §2.2 |
| "complete the sharding specification on the entire computation graph" (GSPMD §1) | GSPMD: "runs the propagation pass on the entire graph in multiple iterations" — §3.5<br>Mesh-TensorFlow: "A global "computation layout" is a partial map from tensor-dimension to mesh-dimension" — §4<br>CoCoNet: "There are three layouts for a tensor: sliced, replicated, and local." — §2.1 |
| "device mesh" (GSPMD §3.1) | Mesh-TensorFlow: "The mesh is only a naming abstraction and does not imply a physical network topology." — §2<br>GSPMD: "device_mesh can be configured by the user in order to optimize communication based on the topology of the device network" — §3.1 |
| "parallel operations coupled with collective communication primitives" (Mesh-TensorFlow, Abstract) | Mesh-TensorFlow: "followed by MPI-allreduce across any mesh dimensions corresponding to reduced-out Tensor dimensions" — §6<br>GSPMD: "We use XLA’s operators that provide MPI-style collective communications [26]." — §4.2<br>FLUX: "AllGathers the sharded input activations along the column direction before GEMM" ... "ReduceScatters the output activation along the column direction" — §2.1 |
| "a single program for all partitions" (GSPMD §1) | Mesh-TensorFlow: "compiles into a SPMD program" — Abstract<br>TileLink: "to follow the SPMD (Single Program, Multiple Data) programming model" — §2<br>Triton-distributed: "Our programming model follows the MPMD (multiple programs multiple data) model." — §2.1 |
| "fuses tiled computation and communication into a single larger kernel" (FLUX §1) | FLUX: "ReduceScatter communication is fused into the epilogue of the GEMM kernel" — §3.1; "AllGather signal checking is fused into the prologue of the GEMM kernel" — §3.2<br>TileLink: "combines communication and computation kernels into one fused kernel" — §1<br>CoCoNet: "generating a single kernel to perform multiple communication and computation operations" — §1; "directly passing the output of communication to following computations through registers" — §2.3<br>Triton-distributed: "developers only focus on adding communication logic" — §1 |
| "Signal primitives are designed to manage barriers between communication and computation." (TileLink §3.2.1) | TileLink: "The notify primitives carry release semantics" ... "the wait primitives carry acquire semantics" — §3.2.1<br>Triton-distributed: "Load with acquire semantic"; "Reduction add with release semantic"; "Locally wait a signal until it equals to some given value" — Table 1<br>FLUX: "GEMM tile computation is blocked by the function WaitSignal until the value contained in the signal is set to true" — §3.2<br>CoCoNet: "an efficient fine-grained spin-lock on a memory buffer" — §5.3 |
| "compute chunks in the order the AllReduce kernel communicates them" (CoCoNet §1) | CoCoNet: "to produce chunks in the same order as the AllReduce consumes them" — §5.3<br>FLUX: "tile coordinate swizzling" ... "communication order selection" — §1<br>Triton-distributed: "Swizzle is to change the order of communication operations and computation operations so that they can better overlap with other." — §3.1 |
| "launches only one fused kernel" (FLUX §2.3) | CoCoNet: "we need to invoke only one MatMul kernel and AllReduce kernel to avoid the overhead of launching multiple kernels" — §1<br>TileLink: "synchronization between these decomposed kernels necessitates host intervention, introducing non-negligible overhead at runtime" — §1<br>FLUX: "no precise control of execution timing on GPUs when using streams" — §1 |
| "map communication and computation to distinct hardware units" (TileLink §1) | TileLink: "communication is mapped to either DMA (Direct Memory Access) engines or processing cores" — §1; "the host-side rank_copy_data primitive maps communication to the copy engine" — §3.2.2<br>Triton-distributed: "GPUs and NPUs employ dedicated memory copy units to perform communication." — §3.1 |
| "Symmetric Memory" (Triton-distributed §2.1) | Triton-distributed: "Each rank allocates a memory buffer in the global scope with the same size." — §2.1<br>FLUX: "This algorithm requires GPUs with peer-to-peer (P2P) supports" — §3.1; "the number of output pointers (Cs) in the fused GEMM kernel is increased to the number of devices in tensor parallelism" — §3.1 |
| "execution scope" (MPP, l. 299–300) | MPP: "The set of threads is called the "execution scope" of the tensor operation." — l. 299–300<br>ThunderKittens: "a warp or warpgroup (4 warps) of threads that collaboratively own and operate on a piece of data" — §3.1<br>Triton: "each kernel is single-threaded – though automatically parallelized" — §3.3 |
| "records and replays kernel launches" (PyTorch 2 §4.7) | PyTorch 2: "TorchInductor uses CUDA Graphs [20] to completely eliminate the overhead from wrapper code" — §4.7 |
| "A resource representing a multi-dimensional array" (MTLTensor.h l. 154) | MTLBuffer.h: "Creates a tensor that shares storage with this buffer." — l. 68<br>metal_tensor: "// Host-bound tensors." (`tensor_handle`) — l. 1392; "// Shader-Allocated Tensors" (`tensor_inline`) — l. 1779<br>MPP: "non-owning meaning these are wrappers around resource in device, threadgroup or thread address space" — l. 214–215<br>CuTe docs: "`Layout` and `Tensor` objects that compactly packages the type, shape, memory space, and layout of data" — 00_quickstart.md<br>PyTorch 2: "Tensors have exposed strides that can be manipulated by users, aliasing views are commonplace" — §4.1 |
