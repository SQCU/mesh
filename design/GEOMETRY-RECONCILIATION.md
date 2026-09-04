# Geometry reconciliation numerics

## Scope

Geometry construction reconciles proposed embeddings with a continuous 3D domain. It
does not discard identities whose first coordinates intersect a boundary. The input is
a reference embedding, its incidence relation, physical hull dimensions, and the
compiled geometric domain. The output is a source-to-realized coordinate relation,
the transduced incidence relation, and measures of every displacement and residual.

The reusable curve kernel is declared by
`xonotic/payload/tools/curve_reconcile.h` and implemented by
`xonotic/payload/tools/curve_reconcile.c`. It has no BSP, game, waypoint, team, or
transport semantics. The Python binding in `curve_reconcile.py` passes contiguous
double-precision coordinate rows through that ABI.

## Discrete energy coordinates

For current vertices (p_i\in\mathbb R^3), reference vertices (q_i), edge vectors
(e_i=p_{i+1}-p_i), and explicit physical scales (L) and (R), the kernel accumulates
six independently weighted coordinates.

The anchor energy

\[
E_a=\sum_i\lVert p_i-q_i\rVert^2/L^2
\]

measures displacement from the supplied construction. The strain energy

\[
E_s=\sum_i(\lVert e_i\rVert-\lVert q_{i+1}-q_i\rVert)^2/L^2
\]

retains the reference sampling density without fixing positions. The bending energy

\[
E_b=\sum_i\lVert p_{i+1}-2p_i+p_{i-1}\rVert^2/L^2
\]

smooths tangent variation.

For adjacent unit tangents (u_i) and (v_i), the cusp coordinate is

\[
E_c=\sum_i\left((1+u_i\cdot v_i+\epsilon_c)^{-1}
-(2+\epsilon_c)^{-1}\right).
\]

It approaches its barrier as a curve reverses direction and is zero for a straight
continuation. The kernel also reports the minimum turn cosine and the number of turn
atoms.

For a curve point (p_i), its unit tangent (t_i), and a non-neighbor point (p_j),
let (d=p_j-p_i). The discrete inverse squared tangent-point radius is

\[
\rho_{ij}=4(\lVert d\rVert^2-(d\cdot t_i)^2)
/(\lVert d\rVert^2+\epsilon_x^2)^2.
\]

The tangent-point coordinate accumulates ((L^2\rho_{ij})^{q/2}) over the directed
non-neighbor relation. This distinguishes global strand proximity from local bending.
Its power is an interface value, not an implementation constant.

For every pair of nonincident segments, the thickness coordinate accumulates

\[
E_t=\sum_{a,b}(R^2/(d(a,b)^2+\epsilon_x^2))^{s/2},
\]

where (d(a,b)) is the exact closest distance between the two finite segments. The
kernel reports the minimum such distance and the segment-pair mass. The analytic
gradient differentiates through closest-point barycentric coordinates using the
envelope property.

The ABI returns each energy separately, their sum, their analytic gradient, and the
geometric measures. Weights select an optimization objective without erasing the
underlying coordinates.

## Clipping and obstacle reconciliation

A collision domain is represented independently from curve energy. For a convex solid
with half-spaces (n_k\cdot x\le d_k), sweeping a hull with extrema (m,M) changes
each plane distance by its support value. Compiled planes are retained as independent
contiguous normal-x, normal-y, normal-z, and distance streams with a solid-to-plane offset
relation. Solid bounds occupy a
Morton-ordered linear bounding hierarchy. Compacted owner-node rows traverse its levels
together, and repeated owner indices make plane evaluation and segmented maximum
reduction SIMD operations rather than per-point dictionary and face walks.

For every intersecting point-solid pair, the local operation selects the face with
maximum signed coordinate (s_{ij}) and its unit normal (n_{ij}). With clearance
(\tau), one point's direction is the minimizer of the damped least-squares coordinate

\[
\frac12\lVert\delta_i\rVert^2+
\frac12\sum_j\left(n_{ij}^{\mathsf T}\delta_i-(\tau-s_{ij})\right)^2.
\]

Its normal equations are the three-dimensional proximal system

\[
\left(I+\sum_j n_{ij}n_{ij}^{\mathsf T}\right)\delta_i
=\sum_j(\tau-s_{ij})n_{ij}.
\]

The matrices are symmetric positive definite by construction. Their batched Cholesky
factorization operates directly on six independent structure-of-arrays coefficients
and three right-hand-side streams. Coefficient assembly reuses one contact-row work
stream instead of retaining nine products. It neither materializes an `N x 3 x 3` tensor nor
dispatches thousands of tiny opaque linear-algebra calls. Solid centroids are
quantized into 63-bit Morton coordinates and their bounds are reduced into a linear
bounding-volume hierarchy. Active point hulls, swept hulls, and rays descend that same
hierarchy as level-synchronous owner-node relations. Each level evaluates every active
box overlap or ray-box slab in contiguous arrays and compacts the relation before
descending. A swept hull is represented by its center ray against solid bounds widened
by the exact physical support, so neither domain length nor the ray bounding box creates
grid-cell rows. Work follows intersected spatial bounds rather than the Cartesian product
of queries and solids. For every remaining convex solid, affine half-space coordinates
provide its exact infinite-ray interval. Owner-sorted interval rows, a segmented prefix
maximum, and a segmented reduction extend the boundary of the union component containing
the input coordinate. Each point retains that scalar boundary rather than retaining and
re-sorting previously reached interval rows. The first horizon is the exact exit from the solids containing the
input. Each subsequent packet traverses only the previously unexamined ray shell out to
twice the proven connected distance. Newly reached solid identities contribute their
full-line interval once. A gap below the examined horizon proves completion; otherwise the
horizon doubles. Thus a connected chain spanning distance (D) from an initial exit (d)
needs at most logarithmically many packet fronts, while an isolated contact never
materializes the rest of the ray. The point advances directly past the completed component.
Projection has no configured radius or iteration limit, no grid-cell materialization, and
no repeated plane evaluation for an already reached solid. A second direction field toward
the domain center is evaluated only for compacted identities whose proximal component
reaches the domain boundary.

A local measurement over 8,192 simultaneous proximal systems and 32,768 incident
constraints takes 0.776 milliseconds and returns unit directions with maximum norm error
of (3.34\times10^{-16}). For 256 points embedded in 10,000 spatially separated solids,
projection takes 1.64 milliseconds, evaluates 512 point-solid candidates and 3,072 plane
coordinates in one ray front, and leaves zero residual penetration. An adversarial chain of
4,096 face-connected solids takes 4.94 milliseconds: 13 doubled fronts, 4,097 candidate
identities, 24,582 plane coordinates, and zero residual penetration. The chain length
therefore changes the front count logarithmically while the arithmetic work follows the
reached relation linearly. The complete machine-readable observation is
[`../measurements/projection-release-closure.json`](../measurements/projection-release-closure.json).

The hierarchy construction follows the Morton-order linear-BVH formulation in
[Karras](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees).
The packet traversal uses the same structure-of-arrays ray-box organization as wide
SIMD BVH traversal; the binary topology keeps every refinement step a mathematical
bisection rather than a machine-specific lane-count constant.

This is the static convex specialization of local/global constraint projection used by
[Projective Dynamics](https://www.projectivedynamics.org/projectivedynamics.pdf).
The explicit positional constraint and multiplier interpretation follows
[XPBD](https://mmacklin.com/xpbd.pdf). For deforming strands, swept feasibility and
barrier contact follow the stronger continuous treatment described by
[Incremental Potential Contact](https://ipc-sim.github.io/file/IPC-paper-fullRes.pdf);
the static waypoint operation does not reproduce IPC's general nonlinear machinery.
Unlike the global matrix construction in
[Fast Projection](https://www.cs.columbia.edu/cg/ESIC/esic.html), the changing static-contact
relation here is block diagonal by point. Materializing or prefactorizing a global system
would erase that structure; explicit batched factorization of the independent three-row
blocks is the smaller operation.

The current BSP domain is the union of compiled world brushes, server-solid BSP
submodels, and engine-tessellated quadratic-patch triangles. Waypoint serialization
uses its own spatial quantum as the projection tolerance so decimal output cannot round
a reconciled coordinate back into a surface.

Quadratic-patch collision realization evaluates each tensor-product Bernstein sample
tile as contiguous arrays, then applies the engine collision snap to the complete tile.
It is algebraically identical to evaluating each parameter pair separately but does not
run an interpreter loop over those pairs. Portal subtraction operates on both brushes
and patches. Intersecting odd patch grids are decomposed into their exact 3-by-3
quadratic blocks, shader and texture coordinates included, and only blocks whose
tessellated collision prisms intersect the convex aperture are removed. A curved wall
therefore cannot survive a brush-only doorway cut as an invisible collision surface.

Projection and energy minimization compose as alternating numerical operations:

1. Accumulate anchor, strain, bend, cusp, tangent-point, and thickness coordinates.
2. Propose a gradient displacement for movable vertices.
3. Project the displaced hulls into the continuous geometric domain.
4. Transduce all incident identities and edges through the same displacement map.
5. Continue until displacement and energy-change measures reach their declared
   numerical resolution.

A line search must account for the swept motion between embeddings. A finite endpoint
separation alone does not preserve a knot: two strands can exchange sides between
samples. Knot class is preserved when every realized embedding step has a collision-free
swept strand volume and positive nonneighbor thickness. For an open curve, endpoint
motion is part of that contract; for a closed curve, it realizes an ambient isotopy.

## Xonotic uses

An authored waypoint coordinate is a stable identity. Its reconciled coordinate replaces
the source coordinate, and every cache edge endpoint is mapped through the same relation.
A cache-implied vertex is materialized and reconciled rather than causing its incident
edge to disappear. Measures include input collision mass, displaced identity mass,
displacement integral and square integral, maximum displacement, sparse candidate-pair
mass, plane-evaluation mass, synchronous ray-front mass, directional-null mass,
world-boundary reconciliation mass, and residual projection mass.

Standing-location construction uses the same array boundary. Candidate hulls are
projected together. Center and footprint rays are expanded as one point-to-solid
relation; affine half-space intervals produce the first downward contact parameter for
every ray, and a segmented maximum produces one support height per hull. All requested
lift coordinates are evaluated in one hull-fit relation. There is no radial stencil,
angular sweep, or one-candidate-at-a-time floor trace. The output includes candidate,
ray, solid-pair, plane-evaluation, realized-support, and unrealized-support measures.

On the current fused BSP, the waypoint operation processes 5,048 point identities
against 418,984 convex collision atoms and 3,156,553 plane rows. It reconciles 424
colliding identities through 1,395 initial penetration pairs and eight synchronous
sweeps in about 1.1 seconds, with zero residual collision after decimal serialization.
For the 5,613-location spawn workload, the batched standing operation expands 50,517
footprint rays and about 0.78 million plane evaluations. The complete 256-team,
32-cart entity realization fell from 119.28 seconds to 39.99 seconds on the M4 mini
while preserving 32 realized carts, 32 continuous rider paths, and zero construction,
spawn-occupancy, or team-cart reachability residual mass.

A cart motion plan is a polyline or spline embedding with cart-and-rider swept hulls.
Its endpoints and declared semantic nodes may be pinned while the remaining vertices
minimize the energy coordinates. Floor activation and obstacle projection are domain
operations. Bending or tangent-point energy cannot substitute for floor support, and a
floor probe cannot substitute for a continuous swept-volume relation.

Map-to-map connectors use the same construction with portal rims as pinned boundary
curves. Strain retains aperture correspondence, bending removes accidental cusps, and
thickness separates connector strands when several joins occupy the same region.

## Unrelated examples

### Filling a debugging rabbit with untangled string

Take a closed triangle surface such as the familiar graphics-debugging rabbit and
compute its interior domain. Erode that domain by the desired string radius. A coarse
space-filling seed curve may be obtained from a voxel traversal, a medial-axis walk, or
a prescribed knot. Its sample count follows the requested length or volume-density
measure rather than a fixed application limit.

Anchor coordinates retain the desired fill distribution; strain keeps adjacent samples
from collapsing; bending and cusp coordinates remove foldbacks; tangent-point and
segment-thickness coordinates separate distant parts of the string; interior projection
reconciles clipping against ears, legs, and surface cusps. A closed seed retains its knot
class when every optimization step satisfies the swept-thickness condition. Useful output
measures are occupied-volume integral, uncovered-volume integral, curve length, minimum
surface clearance, minimum strand separation, cusp integral, tangent-point integral,
and source-to-realized displacement moments.

### Cable routing through machinery

Connector poses are pinned, the cable radius defines (R), and machine collision meshes
define the projection domain. Bend and cusp coordinates represent manufacturability;
tangent-point and thickness coordinates prevent self-contact. Additional strain weights
can vary by material segment without changing the kernel or inventing a machine class.

### Additive-manufacturing and inspection paths

A deposition or camera path supplies reference coverage coordinates. Tool clearance is
the swept hull, and view or deposition constraints are additional projection coordinates.
The same returned measures expose how much of the original coverage was displaced and
where path density, curvature, or strand separation changed.

## Relation to project specifications

This numerical layer realizes the continuous cart-curve obligations in
`NAV-SPEC.md`, the compiled collision domain in `FUSION-SPEC.md`, and the geometry data
flow in `ALGORITHM-CONTRACTS.md`. It does not redefine the stock navigation graph or add
policy semantics. Those layers supply identities, incidence, and constraints; this layer
transduces their coordinates.
