# Joint, nested and sequential optimization of compiled schedules, collectives and shardings: sources, quoted verbatim

Retrieved 2026-11-07. Each PDF was fetched with `curl` into `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit10/` and its text taken by `pdftotext` in reading order (`<name>.raw.txt`). Each block below was copied by script from the cited lines; nothing in a block is retyped. pdftotext joins hyphenated line breaks in places ("interoperator" for "inter-operator").

## 1. Unity (OSDI 2022): joint optimization of algebraic transformations and parallelization

- **Source:** C. Unger, Z. Jia, W. Wu, S. Lin, M. Baines, C. E. Quintero Narvaez, V. Ramakrishnaiah, N. Prajapati, P. McCormick, J. Mohd-Yusof, X. Luo, D. Mudigere, J. Park, M. Smelyanskiy and A. Aiken, "Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization", 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), 2022. https://www.usenix.org/system/files/osdi22-unger.pdf (USENIX open access).

`unity.raw.txt` L31–40, abstract:

```text
This paper presents Unity, the first system that jointly optimizes algebraic transformations and parallelization in distributed DNN training. Unity represents both parallelization
and algebraic transformations as substitutions on a unified
parallel computation graph (PCG), which simultaneously expresses the computation, parallelization, and communication
of a distributed DNN training procedure.
Optimizations, in the form of graph substitutions, are automatically generated given a list of operator specifications,
and are formally verified correct using an automated theorem
prover. Unity then uses a novel hierarchical search algorithm
to jointly optimize algebraic transformations and parallelization while maintaining scalability. The combination of these
techniques provides a generic and extensible approach to optimizing distributed DNN training, capable of integrating new
DNN operators, parallelization strategies, and model architectures with minimal manual effort.
```

`unity.raw.txt` L200–212, L262–264, §2, sequential and joint optimization:

```text
replaced). Workarounds can be used to fix invalid solutions by
using default parallelization strategies or copying the strategies of nearby operators, but it is easy to find cases in which
these workarounds lead to suboptimal solutions. As such,
applying algebraic optimization before parallelization is the
only option, but as we see in the next example, it can miss
significant optimization opportunities.
Consider the computation graph shown in Figure 1, which
represents a 2-layer multilayer perceptron (MLP). If we are
optimizing independently (also referred to as “sequentially”),
we start by applying algebraic transformations without considering parallelism. A typical algebraic optimizer will fuse
the MatMul and ReLU operators to remove redundant memory
loads and stores. The model is then parallelized (we consider only 2 GPUs for simplicity) resulting in Figure 2a: data
parallelism is used for both operators and thus the weight
the cost of not fusing the first ReLU.
As this example shows, joint optimization is necessary to
maximize performance. However, it also poses significant
```

## 2. Alpa (OSDI 2022): nested optimization of inter- and intra-operator parallelism

- **Source:** L. Zheng, Z. Li, H. Zhang, Y. Zhuang, Z. Chen, Y. Huang, Y. Wang, Y. Xu, D. Zhuo, E. P. Xing, J. E. Gonzalez and I. Stoica, "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning", OSDI 22, 2022. arXiv 2201.12023v3.

`alpa.raw.txt` L96–99, §1:

```text
lower bandwidth in between. Second, this hierarchical design
allows us to solve each level near-optimally as an individual
tractable sub-problem. While the joint execution plan is not
guaranteed globally optimal, they demonstrate strong performance empirically for training various large models.
```

`alpa.raw.txt` L558–572, §3, the passes:

```text
passes as Fig. 3 shows. Given a model description, in the
form of a Jax [9] intermediate representation (IR), and a cluster configuration, the inter-op compilation pass slices the IR
into a number of stages, and slices the device cluster into a
number of device meshes. The inter-op pass uses a Dynamic
Programming (DP) algorithm to assign stages to meshes and
invokes the intra-op compilation pass on each stage-mesh pair,
to query the execution cost of this assignment. Once invoked,
the intra-op pass optimizes the intra-op parallel execution
plan of the stage running on its assigned mesh, by minimizing
its execution cost using an Integer Linear Programming (ILP)
formulation, and reports the cost back to the inter-op pass. By
repeatedly querying the intra-op pass for each allocation of
a stage-mesh pair, the inter-op pass uses the DP to minimize
the inter-op parallel execution latency and obtains the best
slicing scheme of stages and meshes.
```

## 3. CoCoNet (ASPLOS 2022): computation and communication optimized jointly

- **Source:** A. Jangda, J. Huang, G. Liu, A. H. Nodehi Sabet, S. Maleki, Y. Miao, M. Musuvathi, T. Mytkowicz and O. Saarikivi, "Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads", ASPLOS 22, 2022. DOI 10.1145/3503222.3507778; arXiv 2105.05720v5.

`coconet.raw.txt` L61–70, abstract:

```text
Therefore, we present CoCoNet, which contains (i) a domain
specific language to express a distributed machine learning program
in the form of computation and communication operations, (ii) a set
of semantics preserving transformations to optimize the program,
and (iii) a compiler to generate jointly optimized communication
and computation GPU kernels. Providing both computation and
communication as first class constructs allows users to work on
a high-level abstraction and apply powerful optimizations, such
as fusion or overlapping of communication and computation. CoCoNet enabled us to optimize data-, model- and pipeline-parallel
workloads in large language models with only a few lines of code.
```

## 4. TACCL (NSDI 2023): collective algorithm synthesis, routing then scheduling

- **Source:** A. Shah, M. Cowan, J. Nelson, V. Chidambaram, S. Maleki and O. Saarikivi (the arXiv v4 header), "TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches", NSDI 23, 2023. arXiv 2111.04867v4.

`taccl.raw.txt` L162–168, §1:

```text
used by modern machine learning workloads. We present a
novel mixed integer linear programming (MILP) encoding
of the collective algorithm synthesis problem that improves
scalability by first solving a bandwidth-relaxed version of the
problem to decide on routing, followed by ordering heuristics and a second bandwidth-constrained problem to find a
valid scheduling (§5). In addition to improving scalability,
TACCL’s MILP formulation allows modeling of heterogeneous links with different per-message overhead characteristics. This overcomes the limitation in SCCL [9] that prevents
```

## 5. GSPMD (2021): sharding from annotations

- **Source:** Y. Xu, H. Lee, D. Chen, B. Hechtman, Y. Huang, R. Joshi, M. Krikun, D. Lepikhin, A. Ly, M. Maggioni, R. Pang, N. Shazeer, S. Wang, T. Wang, Y. Wu and Z. Chen, "GSPMD: General and Scalable Parallelization for ML Computation Graphs", arXiv 2105.04663v2, 2021.

`gspmd.raw.txt` L46–49, abstract:

```text
We present GSPMD, an automatic, compiler-based parallelization system for common machine learning computations. It allows users to write programs in the same way as
for a single device, then give hints through a few annotations
on how to distribute tensors, based on which GSPMD will
parallelize the computation. Its representation of partitioning is simple yet general, allowing it to express different or
```

## 6. Halide (PLDI 2013): the algorithm decoupled from its schedule

- **Source:** J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand and S. Amarasinghe, "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines", PLDI 2013. Authors' copy https://people.csail.mit.edu/jrk/halide-pldi13.pdf.

`halide.raw.txt` L102–106, §1:

```text
for producer-consumer locality across stages.
We address this challenge by raising the level of abstraction, and
decoupling the algorithm definition from its execution strategy, to
improve portability and composability, while automating the search
for optimized mappings of the resulting pipelines to parallel machines and complex memory hierarchies. Effective abstraction and
```

## 7. Bilevel optimization: one problem nested in another

- **Source:** A. Sinha, P. Malo and K. Deb, "A Review on Bilevel Optimization: From Classical to Evolutionary Approaches and Applications", IEEE Transactions on Evolutionary Computation 22(2), 2018. arXiv 1705.06270.

`bilevel.raw.txt` L3–4, abstract:

```text
Abstract—Bilevel optimization is defined as a mathematical
program, where an optimization problem contains another optimization problem as a constraint. These problems have received significant attention from the mathematical programming
```

## 8. Coordinate descent: blocks of variables optimized in turn

- **Source:** S. J. Wright, "Coordinate Descent Algorithms", Mathematical Programming 151, 2015. arXiv 1502.04759v1.

`bcd.raw.txt` L12–13, abstract:

```text
Abstract Coordinate descent algorithms solve optimization problems by successively performing approximate minimization along coordinate directions or
coordinate hyperplanes. They have been used in applications for many years,
```
