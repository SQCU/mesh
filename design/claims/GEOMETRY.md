# Geometry implementation claims

Controlling specifications: [`../NAV-SPEC.md`](../NAV-SPEC.md),
[`../FUSION-SPEC.md`](../FUSION-SPEC.md),
[`../GEOMETRY-RECONCILIATION.md`](../GEOMETRY-RECONCILIATION.md), and
[`../ALGORITHM-CONTRACTS.md`](../ALGORITHM-CONTRACTS.md).

Implementation surfaces:

- `xonotic/payload/tools/navmesh.py` constructs the canonical navigation/Voronoi object
  and the feasible cart-curve representation.
- `xonotic/payload/tools/mkentfile.py` assigns cart spans and emits only represented
  feasible curves.
- `xonotic/payload/tools/mapfuse.py` realizes requested stock/bridge tiles and computes
  graph-derived join properties.
- `xonotic/solver/strat/live_belief.py` consumes the canonical Voronoi object.

The negative-space source index unions player-solid brushes from the BSP world model with
the brush models whose QuakeC spawn functions realize server-side `SOLID_BSP`. Spawn,
cart, support, clearance, and join computations therefore consume one assembled-world
collision domain; trigger and explicitly non-solid models remain traversable. Serialized
negative-space caches carry the geometry-interface schema and are reconstructed when that
scope changes.
Compiled patch control grids run through the DarkPlaces collision tessellation equations,
collision tolerance, triangle winding, and vertex-snap quantum. Each realized triangle is
represented as a one-quantum convex prism in the same indexed half-space union, so curved
floors and walls participate in spawn hulls, cart sweeps, and support intervals rather
than existing only in renderer collision.

The map realization serializes reconciled stock waypoint positions, weighted edges, and
each waypoint's exact share of the metric graph's one-dimensional path-length measure.
Authored and cache-implied coordinates are projected into the player-hull erosion of the
compiled collision domain, and every cache endpoint is transduced through the same
source-to-realized coordinate map. The stock bot loader, cart planner, and Voronoi
realization therefore share the identical vertex and edge relations at runtime. Input
collision mass, displaced identity mass, displacement moments, and unresolved projection
mass are reported independently. Point hulls, swept hulls, and direction rays traverse one
Morton-ordered bounding hierarchy as compacted owner-node arrays. Plane normal and distance
coordinates occupy four independent contiguous streams. Point reconciliation
forms one fixed-size proximal system per penetrating identity, solves its six independent
coefficient streams with explicit batched Cholesky operations, and advances each identity
through the complete connected union of convex ray intervals. The domain-inward relation
is evaluated only for the compacted identities whose proximal direction reaches a boundary.
Ray horizons begin at the exact containing-solid exit and double while a connected interval
reaches the examined horizon; a reached solid contributes its planes once. It does not
enumerate crossed grid cells, materialize the unconnected remainder of a ray, or allocate
systems for nonpenetrating identities. Candidate-pair,
plane-coordinate, direction, boundary, displacement, and residual masses expose the work
without feeding a classification back into construction. The
cart-origin and connected-component sites induce one multi-source shortest-path Voronoi
owner for every navigation node. The realization carries that owner, its cell measures,
and a digest of the complete object; the curriculum passes it without reconstruction to
the responder. Belief cells, receptive support, explore instruments, cart routes, and
actor-to-instrument walking distances therefore derive from one stock-navigation graph
and one Voronoi measure rather than a second observed-position lattice. Compiled
solid-brush half-spaces separately represent
the continuous swept-volume and floor-support domain of the cart body and rider.
When the static realization is not yet present, streamed waypoint links and literal
participant transitions remain separately named topology sources. No nearest-neighbor
relation is synthesized from coordinate proximity.

Candidate generation may sample endpoints, curves, or policy actions. Such samples search
the represented feasible domain; they do not define feasibility. Segment membership is
the union of exact half-space intervals for both swept-body clearance and continuous floor
support. No candidate with incomplete interval coverage enters an emitted path population.
Each supporting face is shrunk by the complete horizontal cart hull before its
path-parameter interval is formed; bottom-center contact cannot admit a ledge overhang.
On a 128-unit square floor, the full 64-unit-wide cart hull produced one complete
support interval at the center and zero intervals when its center was shifted to leave
a sixteen-unit ledge overhang.
Portal realization groups opposing half-spaces by their exact plane identity, reuses each
cell cross-section only within that plane-pair group, and releases the group before the next
plane. Its memory extent therefore follows the currently intersecting plane relation rather
than accumulating every cross-section in the map while preserving the same portal polygons.
Source-map socket construction measures complete left, right, floor, and ceiling solid
support around the requested clear aperture. Each emitted socket has zero solid-support
residual atoms after complete convex decomposition. Its source approach is collision-free for
the complete coordinate-quantized cart-and-rider swept body, a generated approach floor
supplies continuous support to the aperture, and the outward lane has zero source-solid
incidence through the source extent. The generated sleeve supplies the exterior floor.
Extremal-waypoint adapters do not enter the realized source relation.
When a stock map has no supported wall socket, its measured internal transfer site
becomes a degree-one bidirectional teleporter leaf. The opposite endpoints are distinct
backbone tiles. Trigger volume, literal stock-nav attachment, and non-cart incidence are
reported. No incidence ceiling alters graph construction or turns that measure into a
release condition.
Observed socket capacity remains unchanged by graph construction.
Path positions carry literal outward-normal requirements. Bipartite source assignment and
direction-constrained socket selection realize those requirements without scoring an
inward-facing socket as if it were a connection. Exact height propagation on the corridor
tree gives each pair one floor elevation. The complete route from both stock navigation
endpoints through their sleeves and an axis-aligned hall is cut across every placed brush
owner before its computed floor and shell are realized. The compiled world-model solid
domain then measures swept-hull clearance and floor-support gaps over every
connector-chain segment, and these measures participate in cart navigability and release
residual mass. BSP inline models retain model-local coordinates and are excluded from
model-zero static feasibility rather than appearing as phantom world solids; their source
brush geometry remains part of corridor carving.
The server consumes emitted `plc_path` rows and does not synthesize a second path by
walking generic playerbot waypoints.
Curriculum and the reference demo realize the requested cart count through that offline
constructor before starting the server.

The claims are substantiated only when every requested tile has a realized artifact,
every emitted cart curve has continuous swept-volume and
floor-support measures, belief cells retain source Voronoi identity and path-length
measure, and join
prominence, connectivity, and cart-traversable aperture measures can be traced to the
realized graph.
Source translation is measured while each stock entity and brush enters the fused
coordinate system. Quoted and unquoted origin vectors share one parser, generated map
source contains no embedded prose, and any unrepresented origin contributes to the
artifact's source-translation residual measure.
Point entities without an explicit origin use the map-format origin of zero. Fusion
materializes that origin at the tile offset before compilation and reports the mass of
implicit origins, so an originless stock point cannot remain at the fused world's global
zero and open a leak line across tile space.
Every stock tile is realized from its authored `.map` member in the stock archive.
Generated bridge tiles retain their authored `.map`. The common parser consumes stock
brush primitives, generated Quake faces, Valve-220 faces, and patches, then emits one
brush-primitive dialect. Brush-primitive texture axes and offsets are transformed with
the same plane-derived basis used by q3map2, so placement keeps texture coordinates
locked. No BSP decompilation or q3map2 map-conversion path exists in fusion. The artifact
reports authored source bytes, incidence, and missing mass for every realized tile.
The compiler asset root contains the complete stock runtime archives, the matching
mapping-support source-image archive, and the paint/PBR overlay. Its shader list is the
union of concrete shader modules and names declared by installed shader lists. A declared
name with no module is realized as an empty compiler module rather than left as a failed
load. Archive-native symbolic image aliases are followed by the compiler VFS with the same
relative-target semantics as the runtime VFS.
When the compiler names a missing shader image, the asset relation is searched by exact
logical stem, then basename and basename-suffix relation; the source bytes are realized
under the compiler-requested logical name, compilation resumes, and that same alias is
bundled at its runtime logical path. Available MD3 default skins are selected through the
stock skin relation. Compiler-only copies of referenced ASE models remove unusable
exported normal tables and make each transform node name match its object while preserving
faces and texture coordinates. Compiler-only OBJ copies provide an object row and a
material library for every literal material name. Runtime archives retain the stock bytes.
Compiler records report archives, concrete and declared-empty
shader modules, dereferenced and compiler-requested asset aliases,
missing images, and missing files.
All non-world entities are retained, including brush entities, triggers, movers,
teleporters, target chains, sounds, models, items, weapons, and spawn points. Symbolic
`target`, numbered target, `targetname`, and `killtarget` values receive one tile-local
namespace so a common stock name cannot cross-wire two maps. The artifact reports
source/placed entity mass and the exact worldspawn properties displaced by the fused
world's single worldspawn.

Join exclusivity and prominence are the realized region graph's cut-edge or leaf-edge
incidence measure. Cart navigability is the realized stock-waypoint attachment,
bidirectional connector-chain incidence, and cart-hull aperture measure. The serialized
join interface copies those computed values and supplies no corridor-type defaults. The
resume path preserves the original source-realization measures rather than replacing
them with a smaller post-compile record. When an interrupted first pass did not publish
its final metrics row, resume reconstructs map, bridge, join, component, coordinate,
compiler, and BSP measures from the already-published source and compile artifacts.

Every socket-bearing stock map is a physical leaf on a generated bridge-tree backbone.
The bridge count is derived by increasing the direction-complete backbone relation until
a bipartite match assigns every stock leaf to a distinct compatible bridge direction and
the remaining physical nodes can host every transfer leaf distinctly. The placement cells
separate backbone turns, physical leaves, and the transfer-only region, and realize all
three translation coordinates on the portal lattice before source geometry, entities,
waypoints, and connector coordinates are transduced. Transfer-only tiles share that
compiler-stable coordinate contract rather than retaining fractional packing offsets.
Transfer incidence
is balanced by current host degree and distance. If several transfer edges share one host,
their trigger volume is partitioned into disjoint subvolumes instead of imposing a host-count
ceiling. Within non-overlap
slack, alternating exact site selection and cross-axis coordinate descent reduces lateral
portal displacement while preserving positive axial separation and reports its initial
and final length integrals and maxima. The BSP coordinate
extent and per-axis excess over the format boundary are emitted as measures, so a long
one-dimensional strip cannot be silently clipped at the compiler boundary. When q3map2
exhausts its draw-surface allocation, compilation doubles the exhausted value and
continues without an application-side maximum. The compiler record retains every
attempted capacity and return code. Visibility storage follows the realized portal-cluster
relation instead of a fixed compiler cluster ceiling. Directed-portal ordering, leaf
incidence, traversal vectors, merge vectors, and histogram storage all size from that
same realized relation. Portal-front storage is transient. Passage masks and worklist
states retain only nonzero 32-bit words; on the 190,730-directed-portal fused workload,
63,147,768 stored words occupy 737 MB instead of the 34,148 MB dense passage matrix.
Passage flow is heap scheduled rather than recursively represented on worker stacks.
Lighting retains every realized contribution in dynamic storage instead of stopping at a
fixed contribution count. Patch lightmap subdivision and linear-row compaction use
input-derived heap extents rather than fixed per-worker draw-vertex matrices. The release
PVS is the conservative portal-flood upper bound:
33,135,853 visible pairs over 33,382 clusters, or 2.974 percent of the complete cluster
relation, in the current fused artifact. Corridor sleeves extend past their cutter and overlap
source solid around the cutter's full transverse boundary. The
serialized clearance, depths, directional overlaps, and minimum overlap realize the
sealed join relation specified by [`FUSION-SPEC.md`](../FUSION-SPEC.md) instead of
relying on coplanar brush contact.
Generated bridge maps enter survey only after their four ground-tier arms, vertical
gallery, jump pad, teleporter, BSP, waypoints, and waypoint links have been realized.
`joinshot.py` reports frame support and luminance distributions without feeding a visual
classification back into geometry construction.
The portal-carve candidate relation uses oriented sleeve bounds only as a scheduling
index. Half-space vertices consume every plane triple through vectorized work slices;
convex subtraction retains every realized fragment without a piece-count ceiling. That
exact relation establishes solid-sleeve incidence before a source brush is split, and
exact nonempty fragments are the only replacements. The cutter and shell
share one oblique frame; the cutter is wider and taller than the playable shell interior
by a measured clearance smaller than shell thickness. A slanted brush with a broad bound
remains intact when its solid never meets the sleeve. Serialized clearance, carve depth,
embed depth, and directional shell overlaps expose the seal construction.
Map serialization retains ten significant digits for every face point. At the declared
BSP coordinate extent this preserves the orientation of the 64-unit derived-face basis;
the two fragments incident to a split therefore reconstruct one shared plane instead of
two rounded, diverging planes.
Generated bridge perimeter lights partition their edge spans without overlapping corner
brushes, so their triangles remain distinct compiler inputs.
`fusemeasure.py` consumes the literal serialized connector chain and canonical
cart-and-rider hull, then measures the same exact swept-clearance and floor-support gap
relations used by release construction. It defines no second corridor coordinate system
or fallback dimensions.
Generated map archives are completely assembled beside their destination and atomically
replace the preceding package, so a live client can observe only a complete old or complete
new geometry generation.

Spawn-pool realization ranks the complete finite candidate relation in one vectorized
pass by cart-access imbalance, total graph distance, and distance from the candidate
median. It emits the requested cardinality in localized order. It does not repeatedly
replace every selected row with every unselected row or recompute group percentiles for
each pair.
One spawn-access relation owns the cart-origin-to-waypoint distance rows, vectorized
nearest-node attachment, and point-to-node attachment distances. Spawn recovery,
balanced selection, per-cart access, and final path measures consume that same relation;
they do not rebuild identical Dijkstra rows at each call site. The artifact reports
origin-node, distance-row, nearest-cache, and attachment-distance measures.

Survey disintegrates each source negative-space decomposition into the retained portal-site
relation and then releases that source decomposition. The final evaluator constructs one
indexed fused solid-half-space domain directly from the compiled BSP. It does not expand
the BSP into a second convex free-cell graph because the stock waypoint graph is the
navigation object. Peak retained geometry therefore does not grow as the sum of all
already-surveyed tile decompositions.

Placement consumes every waypoint in each source's largest canonical walk component.
It does not replace that component with a fixed-size farthest-point sample before
computing extents or offsets, and no unused map-classification routine supplies a
fictional topology label.
Generated connector waypoints and stock waypoints without serialized flag rows remain in
the same navigation graph. Missing flag incidence is reported; it is not an edge rejection.
Cart-path components are admitted only when an exact weighted shortest-path horizon in
the component reaches the physical minimum, rather than by node count, bounding-box
extent, relative component size, or short-dangle deletion. Their traversal curves remain
constrained by swept-body clearance, floor support, and cart-incompatible semantics.
Cart-origin k-centering and the serialized Voronoi object use the ordinary stock-playerbot
walking graph across those physically feasible origin candidates; thus player walking
distance may pass through jumps and teleporters that a cart path itself cannot traverse.
Track orientation maximizes the minimum start-to-start walking distance after the
counterflow and direction-coupling measures, and emitted start points attach to that metric
with their literal point-to-waypoint distances rather than being replaced by waypoint IDs.
The artifact reports candidate-component, candidate-node, selected-component, metric-pool,
and disconnected-candidate masses.
