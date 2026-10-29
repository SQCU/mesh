# Lowering closure, 2026-10-29

The question: does "general" or "arbitrary" lowering of allocation and indexing reach a finite, decidable end? Or does
each closed case keep revealing new ones? It is asked for a fixed class of programs.

## The class

The class is tensor-to-tensor programs, ahead-of-time compiled, run across any number of CPU, GPU and ANE units on a
collective, whose memory is region pages rather than pointers and offsets. The minimum frontends, on the operator's
statement of 2026-10-29:

- typed MPS programs written for this repo;
- every PyTorch program that `torch.export` / FX captures for a collective of MPS machines with ANE backends.
  Exported graphs contain only Core ATen operators and custom operators, and are fully functional (no in-place or
  `_out` variants) ([PyTorch IRs](https://docs.pytorch.org/docs/2.3/torch.compiler_ir.html));
- the whole LiteRT engine on Metal, which is a fixed set of builtin operators;
- at least one non-transformer DNN written into the demos.
  - examples/lstm.py exists on the torch_mesh path.
  - examples/fabric-dnn.py is an FFN stack, the transformer's own block.
  - Neither goes through the LiteRT or native frontends.

Every frontend therefore reduces to a finite operator inventory. The question becomes whether one finite algebra
lowers each operator's indexing, and how they compose.

## The answer: three layers, each finite or decidable

### 1. Index arithmetic: Presburger relations

Nearly every operator's indexing is quasi-affine: integer linear terms with floor division and modulus by constants.
Slice, stride, permute, broadcast, pad, concat, split, tile, block and block-cyclic distribution are all quasi-affine.
So is reshape, which is mixed-radix division and modulus.

- **The swizzles are included.** Bit i of x is ⌊x/2ⁱ⌋ mod 2, and XOR of bits is their sum mod 2. Triton's F₂ linear
  layouts ([Zhou et al., ASPLOS '26](https://arxiv.org/abs/2505.23819)) are therefore quasi-affine once expanded.
- **The operations are closed and decidable.** Relations over these terms are Presburger sets. Presburger arithmetic is
  decidable, and its relations are closed under composition, inverse, image, preimage, intersection, union and
  projection. These are implemented in isl ([Verdoolaege](https://src.rcs.uwaterloo.ca/xref/netbsd-src/external/mit/isl/dist/doc/reading.tex)),
  Omega ([calculator](https://www.cs.umd.edu/projects/omega/calculator_doc/node1.html)) and
  [MLIR's Presburger library](https://mlir.llvm.org/doxygen/classmlir_1_1presburger_1_1PresburgerRelation.html).
- **The general inverse is lexmin.** The closed form that code needs (the first preimage, as a piecewise quasi-affine
  function of the parameters) comes from parametric integer programming. That algorithm always terminates
  ([Feautrier 1988](https://www.rairo-ro.org/articles/ro/pdf/1988/03/ro1988220302431.pdf)).
- **Code generation is solved.** Index source is generated from piecewise quasi-affine expressions
  ([Grosser, Verdoolaege, Cohen, TOPLAS 2015](https://unpaywall.org/10.1145%2F2743016)).
- **Both production layout systems fit inside it.** CuTe and Triton linear layouts both embed in isl relations, with
  composition, inversion and complement done by isl
  ([Bhaskaracharya et al. 2025](https://arxiv.org/abs/2511.10374)). Their one warning applies here: CuTe's implicit
  promotion yields mappings beyond relational composition, and code built on it needs bounds checks. CuTe's own
  algebra has a categorical account ([Carlisle et al. 2026](https://arxiv.org/pdf/2601.05972)).
- **Symbolic sizes are parameters.** `torch.export`'s backed dynamic dimensions (SymInts with ranges) are exactly the
  parameters of parametric Presburger sets.

### 2. Pages: a finite table, or a function of a prefix

A page table maps a page number to a bridge page. At preparation the table is known. A relation composed with it is
piecewise quasi-affine, with one piece per run of contiguous pages, so it is finite.

As a symbol, the table is a function applied to the address's outermost coordinate (page, then offset). Presburger
arithmetic with free function symbols is undecidable in general (Downey 1972). Omega stays decidable by applying
function symbols only to a prefix of a tuple
([Omega docs](https://www.cs.umd.edu/projects/omega/calculator_doc/node9.html)), and a page coordinate placed
outermost is such a prefix. PagedAttention's block table is this layer in production
([Kwon et al., SOSP 2023](https://arxiv.org/pdf/2309.06180)).
[vAttention](https://arxiv.org/html/2405.04437v3) puts the same table into the virtual memory system instead.

### 3. Data-dependent indices: operands, not arithmetic

Gather, scatter, index_select, embedding, nonzero, top-k, MoE routing and sparse coordinates are not Presburger. The
standard treatment keeps one mechanism for all of them:

- **Index arrays as functions.** An index array is an uninterpreted function at compile time and an operand at run
  time, as in the Sparse Polyhedral Framework's inspector/executor
  ([Venkat, Hall, Strout, PLDI '15](https://mstrout.cs.arizona.edu/Papers/Papers15/PLDI15-Venkat.pdf)).
- **Sparse formats.** A sparse format is a composition of a small fixed set of per-dimension level formats
  ([Chou, Kjolstad, Amarasinghe, OOPSLA '18](https://arxiv.com/abs/1804.10112)).
- **Sizes.** On the size side, `torch.export`'s unbacked SymInts (from nonzero and item) are the counterpart
  ([PyTorch docs](https://docs.pytorch.org/docs/main/compile/dynamic_shapes_backed_unbacked.html)).

## Allocation and communication within the same algebra

- **Allocation sizes and reuse.** A buffer's size is the cardinality of an image set. Storage reuse is a modular
  folding of that set. Lattice-based allocation gives the smallest size and dimension within the class of modular
  mappings ([Darte, Schreiber, Villard, IEEE TC 2005](https://perso.ens-lyon.fr/gilles.villard/BIBLIOGRAPHIE/PDF/DarteSchreiberVillard.pdf)).
- **Communication sets.** What a peer must send is a flow-out set: the image of the consumer's accesses over the
  producer's pages, computed as a Presburger set, with packing, unpacking and partners derived from it
  ([Bondhugula, SC'13](https://www.csa.iisc.ac.in/~uday/publications/uday-sc13-slides.pdf)). In this repo that set is
  the input to a WRITE plan, and it stops at nccl.h's CommPlan.
- **Lower bounds.** For linear subscripts, the Hölder-Brascamp-Lieb linear program bounds any loop nest's
  communication ([Christ et al. 2013](https://arxiv.org/pdf/1308.0068)). It is the yardstick for C1's "fewer
  transactions" and for D10.

## Where it does not reach an end

- **Polynomial subscripts.** Index expressions such as packed-triangular i(i+1)/2 are outside Presburger. Deciding
  integer solutions of polynomial equations is undecidable (Matiyasevich 1970); that this reaches dependence analysis
  is inference, not a cited result. The remedy is layer 3: such a storage order is an index table, carried as an
  operand.
- **Placement.** Choosing the best assignment across units is a finite search with no closed form. It is
  optimization, not a gap in lowering: D6's estimator between calls, not the compiler.
- **Cost.** Presburger decision is at least doubly exponential in the worst case. Tensor relations have few
  dimensions, and the work runs at preparation only.

## The source against the algebra

MetalTensorLayout (rdma/mesh-tensor-layout.m, 1552 lines at 3171f4b4) is a hand-written subset of layers 1 and 3:

- `TensorSum`, `TensorScale` and `TensorQuotient` are quasi-affine expression trees;
- `maps` and the `gather` / `select:indices:` forms are layer-3 tables;
- MetalTensorPages `bind:` is layer 2;
- slice, select, permute, reshape, gather, components, shift and rebase are the constructors.

The cases operand-contract.md lists as remaining are each one composition, inverse or image in layer 1: reversed and
repeated selections, selections across resized source domains, shape-changing collective outputs, and external and
signature selections. A general inverse would cover them all; three partial inversions do not
(`TensorSelectedRebase`, `TensorCoordinateRebase`, `TensorFactorRebase`). The general one is lexmin over the
relation. With a general inverse, the remaining-case list stops growing. Until then, each case is a new branch.

D4 ranks foreign code called above foreign code rewritten. isl and MLIR's Presburger library implement layer 1,
lexmin and the generation of piecewise quasi-affine code. Calling either in lowering ranks above extending the subset.
Lowering is ahead of time, so a solver's heap never reaches a run. Only its output does: emitted index source,
constants and page relations, all serialized. Where foreign code does run in a mesh process, mesh already moves its
memory into the region: torch's CPU and MPS allocators are region leases (e91d8091, e961c8d4).

## The work, as tickets

Each ticket is source-scoped, and none is transport. T7 ends at CommPlan's operands.

| Ticket | Scope |
|---|---|
| T1 | The relation type: piecewise quasi-affine, logical to storage, with symbolic sizes as parameters |
| T2 | The closed operations: compose, inverse by lexmin, image, preimage, intersect, union, project, equality |
| T3 | Emission: relation to kernel index source and compact constants (`declaration:`, `constants:`) |
| T4 | Sizing and reuse: cardinality and modular folding for region-page allocations |
| T5 | Pages: the page table as a prefix function composed into T1 |
| T6 | Data-dependent indices as operands; sparse levels from a fixed level-format set |
| T7 | Communication sets: each peer's flow-out image over owner pages, as WRITE plan inputs |
| T8 | Operator tables, one ticket per family, for each of the three frontends (Core ATen, LiteRT builtins, MPS) |
| T9 | A non-transformer DNN demo through the LiteRT and native frontends, not torch_mesh alone |

T8 is the bulk, and it parallelizes. The families are elementwise and broadcast, view (reshape, permute, slice, pad,
concat, split), reductions, contractions (matmul, conv, einsum), indexing (gather, scatter, embedding), normalization,
attention, pooling, and quantized or compressed operands. Each maps an operator to T1 relations and calls T2–T6; none
adds an operation to them.
