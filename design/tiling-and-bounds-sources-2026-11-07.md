# Tiling an expression DAG, and bounds inference: sources, quoted verbatim

Retrieved 2026-11-07. Each excerpt below was copied by script from the cited lines of the text file named; nothing in a
block is retyped. The text files are `pdftotext` (pdftotext version 26.09.0) output of the PDFs named. In Listing 1
the typeset small capitals come out spaced, as `B OTTOM U P DP` for BottomUpDP.

## 1. Koes and Goldstein, CGO 2008: optimal instruction tiling, and NOLTIS

- **Source:** D. R. Koes, S. C. Goldstein, "Near-optimal instruction selection on DAGs", Proceedings of the 6th annual
  IEEE/ACM international symposium on Code generation and optimization (CGO '08), pp. 45–54, 2008, DOI
  10.1145/1356058.1356065 (Crossref). The ACM copy answered 403 to a script; the authors' copy was read:
  https://www.cs.cmu.edu/~dkoes/research/dkoes_cgo08.pdf, 10 pages, PDF created 2008-01-17, SHA-256
  b711c8ed662d22175971bd3036c7a573325cf92a510dd1e4934c7f0f736fd86c →
  `scratchpad/lit-lg/koes2008/dkoes_cgo08.pdf`, text `dkoes_cgo08.txt`.

§3, `dkoes_cgo08.txt` L156–177:

```text
PROBLEM DESCRIPTION

Given an expression DAG which represents the computation of
a basic block and a set of architecture specific instruction tiles, we
wish to find an optimal tiling of the DAG which corresponds to the
minimum cost instruction sequence. The expression DAG consists
of nodes representing operations (such as add or load) and operands
(such as a constant or memory location). We refer to a node with
multiple parents as a shared node. The set of tiles consists of a
collection of expression trees each with an assigned cost. If a leaf
of an expression tree is not an operand, it is assumed that the inputs
for that leaf node will be available from a register1 . Similarly, the
output of the tree is assumed to be written to a register. A tile
matches a node in the DAG if the root of the tile is the same kind
of node as the DAG node and the subtrees of the tile recursively
match the children of the DAG node. In order for a tiling to be
legal and complete, the inputs of each tile must be available as the
outputs of other tiles in the tiling, and all the root nodes of the
DAG (those nodes with zero in-degree) must be matched to tiles.
The optimal tiling is the legal and complete tiling where the sum
of the costs of the tiles is minimized. More formally, we define an
optimal instruction tiling as follows:
```

Figure 1's caption, L253–255:

```text
Figure 1: An example of instruction selection on a DAG. (a) The tile set used (commutative tiles are omitted). (b) Two possible tilings.
In a simple cost model where every tile has a unit cost the top tiling would be optimal, but with the cost model shown the lower tiling
is optimal.
```

§4, L369–410:

```text
4.

NOLTIS

The NP-complete nature of the optimal instruction tiling problem necessitates the use of heuristics when performing instruction
selection. A common approach is to first decompose the DAG into
a forest of trees and then use an optimal tree tiling algorithm to tile
each tree. Every common subexpression in the DAG is therefore
at the root of a tree in the forest. However, as we will show in
Section 7, this approach is not as successful as algorithms which
work directly on the DAG. For example, if all the tiles in Figure 1
were assigned a unit cost, the tree decomposition solution would be
suboptimal.
In this section we present NOLTIS, a linear-time algorithm which
obtains near-optimal tilings of expression DAGs. The algorithm
applies tree tiling directly to the DAG without first performing tree
decomposition, uses this tiling to decide which parts of the DAG
can be productively decomposed into trees, and then retiles the partially decomposed DAG.
First we apply dynamic programming on the DAG ignoring the
presence of shared nodes using the procedure B OTTOM U P DP from
Listing 1. Conceptually, we are tiling the tree that would be formed
if every shared node (and its descendants) was duplicated to convert
the DAG into a potentially exponentially larger tree. However, the
algorithm remains linear since each node is visited only once. Once
dynamic programming has labeled each node with the best tile for
that node, a top down pass, T OP D OWN S ELECT, creates a tiling

of the DAG. The existence of shared nodes may result in a tiling
where nodes are covered by multiple tiles (i.e., the tiles overlap).
However, since no node will be at the root of two tiles (this would
imply that the exact same value would be computed twice), the
number of tiles in a tiling is proportional to the number of nodes.
Consequently, the top down pass, which traverses tiles, has linear
time complexity.
The tiling found by the first tiling pass ignores the impact of
shared nodes in the DAG and therefore may have excessive amounts
of overlap. In the next step of the algorithm, we identify shared
nodes where removing overlap locally improves the overall tiling.
These nodes are added to the fixedNodes set. We then perform
another tiling pass. In this pass, tiles are prohibited from spanning
nodes in the fixedNodes set; these nodes must be matched to the
root of a tile.
```

Listing 1, L570–611:

```text
Listing 1 Dynamic programming instruction selection with modifications for near-optimal DAG selection
1: DAG : expression DAG
2: bestChoiceForNode : Node → (Tile × int)
3: fixedNodes : set of Node
4: matchedTiles : set of Tile
5: coveringTiles : Node → set of Tile
6: procedure S ELECT
7: f ixedN odes ← {}
8:
B OTTOM U P DP()
. initializes bestChoiceForNode
9:
T OP D OWN S ELECT()
. initializes coveringTiles
10:
I MPROVE CSED ECISIONS()
. initializes fixedNodes
11:
B OTTOM U P DP()
. uses fixedNodes
12:
T OP D OWN S ELECT() . puts final tiling in matchedTiles
13: procedure B OTTOM U P DP
14:
for n ∈ reverseTopologicalSort(DAG) do
15:
bestChoiceForNode[n].cost ← ∞
16:
for tn ∈ matchingTiles(n) do
17:
if ¬hasInteriorFixedNode(tn , fixedNodes) then
18:
valP
← cost(t)+
0
n 0 ∈edgeNodes(tn ) bestChoiceForNode[n ].cost
19:
if val < bestChoiceForNode[n].cost then
20:
bestChoiceForNode[n].cost ← val
21:
bestChoiceForNode[n].tile ← tn
```

§7, L650–654:

```text
cse-all The expression DAG is completely decomposed into trees
and dynamic programming is performed on each tree. That
is, every shared node is fixed. This is the conventional method
for applying tree tiling to a DAG [5].

```

## 2. Halide, PLDI 2013: bounds inference

- **Source:** J. Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and
  Recomputation in Image Processing Pipelines", PLDI '13, pp. 519–530, DOI 10.1145/2491956.2462176, read as
  design/final-lowering-sources-2026-11-06.md §7 records it (https://people.csail.mit.edu/jrk/halide-pldi13.pdf →
  `scratchpad/lit2/fl07_halide/halide-pldi13.pdf`), here its text `halide-pldi13.txt`.

§4.2, `halide-pldi13.txt` L1048–1070:

```text
4.2

Bounds Inference

At this stage, for allocation sizes and loop bounds the pipeline relies
on symbolic bounds variables for each dimension of each function.
The next stage of lowering generates and injects appropriate definitions for these variables. Like function lowering, bounds inference
proceeds recursively back from the output. For each function, it
symbolically evaluates the bounds of each dimension based on the
bounds required of its caller and the symbolic indices at which the
caller invokes it. At each step, the required bounds of each dimension are computed by interval analysis of the expressions in the
caller which index that dimension, given the previously computed
bounds of all downstream functions.
After bounds inference has recursed to the top of the pipeline, it
walks back down to the output, injecting definitions for the bounds
variables used as stand-ins during lowering. They are defined by
expressions which compute concrete bounds as a preamble at each
loop level (e.g., in Fig. 5, right, the minimum bound of blurx.y is
computed from interval analysis of the index expressions at which it
is accessed combined with the bounds of the calling function, out).
In practice, hoisting dynamic bounds evaluation expressions to the
outermost loop level possible makes the runtime overhead of more
complex bounds expressions negligible.
```
