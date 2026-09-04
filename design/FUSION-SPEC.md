# Map-fusion specification

Map fusion constructs one runnable Xonotic world from authored stock map sources and
generated bridge sources. Its purpose is to make traversal and cart commitment span
larger-than-Quake-scale spaces while preserving the stock playerbot navigation relation.

The implementation claim for this specification is indexed by
[`claims/GEOMETRY.md`](claims/GEOMETRY.md). Continuous navigation and cart-path semantics
are further specified by [`NAV-SPEC.md`](NAV-SPEC.md).

## Inputs

The constructor receives:

- a stock map archive containing authored `.map` members and waypoint material;
- either every navigable stock source, an explicit source-name relation, or a sampled
  source relation;
- a requested generated-bridge count or a count derived from the measured socket-direction
  relation and transfer demand;
- requested team and cart cardinalities;
- an output directory, compiler work directory, and deterministic random seed;
- the q3map2 executable and runtime asset root used by both bridge-source and fused-world
  compilation.

`--maps=all` means the complete navigable stock source relation discovered in the input
archive. `--names` preserves the literal named relation. A source that cannot be realized
remains in `requested_map_names`, contributes to unfinished and missing-source measures,
and is never silently removed from the request's accounting.

Team and cart cardinalities define generated gameplay objects. They do not alter which
stock sources can be represented and are not transport limits.

## Construction flow

```text
authored stock .map + waypoints       generated bridge .map + ports
                  \                    /
                   source survey relation
                            |
             direction-constrained tree placement
                            |
             tile-local entity and target translation
                            |
                portal carve + connector chains
                      /                 \
             fused authored map      fused waypoint graph
                      |                 |
                   q3map2          weighted nav/Voronoi
                      |                 |
               compiled BSP ---- continuous solid domain
                      \                 /
                feasible cart curves + spawn pools
                            |
             entity overlay + measures + runtime package
```

Each arrow names a serialized or reconstructible relation. No stage replaces a missing
relation with a topology label, fixed score, or sampled collision guess.

The requested BSP model owns the brush range admitted to the compiled solid domain.
World-static feasibility uses model zero; inline mover models retain their own local
coordinate spaces and cannot be folded into world coordinates as phantom solids.

## Authored source realization

Fusion reads the authored `.map` member for every stock tile. It does not decompile a BSP
or ask q3map2 to invent source geometry. The common parser accepts stock brush primitives,
generated Quake faces, Valve-220 faces, and patches, then writes one brush-primitive
dialect.

Every source brush, patch, and semantically realized non-world entity is translated into
fused coordinates.
Point entities without an explicit origin carry map-format origin zero and are translated
from that value; they cannot remain at the fused world's global zero. Brush-primitive
texture coordinates use the same plane-derived basis as q3map2 so geometry placement does
not detach its material coordinates.

The fused world has one worldspawn. Displaced worldspawn properties and the source and
placed masses of all other entities are measured. `target`, numbered target, `targetname`,
and `killtarget` values receive the same tile-local namespace mapping so symbolic chains
remain inside their source tile.

Runtime launch derives `g_max_info_autoscreenshot` from the realized entity relation and
removes DarkPlaces' fixed VM-jump ceiling. Neither an authored camera nor a finite
world-construction traversal may be discarded because a stock-map multiplicity exceeded a
single-map default or an engine-global magic number.

An `_decal` row with no patch and no brush has no visual or runtime realization. q3map2
otherwise clears that row's keys and incorrectly introduces an anonymous flood occupant at
global origin. Fusion removes precisely that empty relation and reports
`compiler_inert_empty_decal_mass`; decals carrying geometry remain ordinary translated
entities.

## Placement and topology

Every stock source with at least one supported wall socket is a physical degree-one leaf.
Generated direction-complete bridge tiles form a rectilinear tree backbone. Its turn
sequence leaves a measured relation of unused directed bridge sockets. Bipartite matching
assigns every physical stock leaf to a distinct unused socket whose opposite direction is
literally present on that stock source. Backbone edges consume their literal bridge
sockets once. The generated bridge count is the first cardinality for which that exact
matching exists and the physical-node relation is large enough to give every transfer leaf
a distinct counterpart. Increasing the requested bridge count adds capacity; requesting
fewer cannot remove the computed relation.

Backbone cells are separated to leave leaf cells and connector corridors disjoint. Maps
without sockets occupy a separate placement region because their graph edges are
teleporters rather than physical halls. Every tile translation is realized on the same
four-unit lattice used by portal apertures, including transfer-only tiles and vertical
alignment, so q3map2 receives one translated plane representation instead of fractional
copies whose rounding can diverge across a large combined BSP. The output records tile
cells, offsets, placement
shape, source extents, absolute BSP coordinate extent, and per-axis excess beyond the
format extent. Observed socket count remains a measure; topology never pretends that a
socket exists or faces a direction it does not face.

Each generated bridge tile exposes four ground-tier physical arms around a traversable
hub. Its internal gallery is vertically separated from the cart tier and connected by a
stock jump-pad/teleporter pair. Bridge-source compilation and waypoint generation happen
before the tile enters the same survey relation as stock maps; a written `.map` without
its compiled BSP and navigation members is not a realized bridge source.

Portal endpoints derive from source negative-space and navigation sites. Portal width,
height, and floor offset derive from the canonical cart-and-rider swept hull. Rounding
outward to the map coordinate quantum supplies positive clearance and keeps the waypoint
origin exactly one player-bottom offset above the corridor floor. Each endpoint has a
collision-free source approach for the full cart-and-rider swept body. A generated floor
brush supplies continuous support from the source waypoint to the aperture. Past the wall,
the full cart-and-rider cross-section has zero solid incidence all the way to the source
extent; the connector sleeve supplies the new exterior floor. Its source wall has positive
shared-volume incidence with four measured support prisms around the clear aperture:
left, right, floor, and ceiling. Their missing-incidence mass is zero for every realized
endpoint; no inferred boundary, point sample, extremal waypoint, or general convex
subtraction search substitutes for that relation. Tree propagation aligns source-relative elevations exactly. Each endpoint
sleeve follows its measured outward normal from aperture to the outside of the source
extent. Fusion cuts the complete swept route from each stock navigation endpoint through
both sleeves and the hall. The cut is applied to every placed brush and patch owner before
generated floor and shell brushes enter the same owners. Convex brushes are differenced
by half-space incidence. Odd patch grids are decomposed into exact quadratic control
blocks, and blocks whose compiled collision triangles intersect the cutter are removed
with their shader and texture coordinates otherwise unchanged. Editor groups,
moving-brush entities, triggers, curved walls, and authored exterior seals cannot survive
as an unmeasured obstruction inside the route. Between direction-compatible tree cells,
an axis-aligned hall spans the two
sleeve mouths and has one literal cart-width opening at each end; its cross-axis extent is
computed from both mouths. Cutter cross-sections extend beyond the clear playable
cross-section by less than one wall thickness. Supported sleeves overlap
measured intact source solid laterally, above, and below while preserving the requested
playable width and height. Their shells embed one wall thickness beyond each cutter.
Fusion schedules candidate brushes and patch blocks by route bounds, then cuts only
geometry whose compiled solid intersects the swept route; bounds never substitute for
solid incidence. It adds the
connector waypoint chain and records direction residual, hall cross-span, horizontal span,
rise, grade, width, height, carve clearance, carve depth, embed depth, longitudinal
overlap, transverse overlap, and their positive minimum seal overlap.
Half-space realization enumerates the complete plane-triple relation in vectorized work
slices. Convex difference retains the complete dynamically sized fragment relation; no
plane prefix, fragment ceiling, or dropped-leaf path changes the realized geometry.
Planar coordinate descent aligns only the cross-axis coordinate of paired mouths. It
preserves the extent pack's positive axial separation rather than moving opposing mouths
onto one plane. Inside the hall, the waypoint chain travels straight beyond each end jamb
by one wall thickness, carve clearance, and cart-hull radius before changing its
cross-axis coordinate. Approach-floor longitudinal overlap is one wall thickness plus the
cart-hull radius, so support extends beyond the complete body at the source waypoint.
Generated bridge hub openings carry the complete clear arm width plus both arm walls;
hub jambs and arm shells meet without coplanar overlap.
Maps without a supported source-wall socket enter the topology through one bidirectional
teleporter edge anchored at a stock navigation node whose complete player hull occupies
source negative space. Such maps are degree-one leaves. Every counterpart is distinct,
so each transfer leaf and counterpart has one transfer incidence. Corridor topology consumes only
observed supported-socket capacity; it does not extend that capacity to fit a desired
graph. Teleporter trigger volume and per-map non-cart incidence are serialized measures;
no incidence limit changes construction or release state.
Supported-site count is the literal socket capacity. The physical tree consumes only its
incident directional sockets, without redefining maps containing additional sockets or
adding elevation cycles.
Derived split faces serialize enough significant digits to preserve their shared plane
at the full BSP coordinate extent. Opposing faces are reconstructed from the same plane
coefficients, so repeated convex splitting cannot turn numerical formatting into an
exterior seam.

Region-graph connected components, articulation tiles, cut edges, degrees, and hop
diameter derive from the realized edge relation. Join exclusivity is cut-edge or leaf-edge
incidence; prominence copies that computed property. These are graph measures, not author
labels.

## Compiler asset closure

q3map2 sees the complete stock runtime archive relation, the matching mapping-support
source-image archive, and the paint/PBR overlay. Its shader relation is the union of
concrete shader modules and shader-list declarations. A declared shader without a module
is represented by an empty compiler module so the name remains present.

Archive-native symbolic image aliases are followed by the compiler VFS using their
relative archive targets, matching runtime VFS semantics.
When q3map2 names a missing image, the asset relation is searched by logical stem,
basename, and basename suffix. A discovered source is materialized at the requested
logical path and bundled at that runtime path. Available default skins are materialized
through their stock skin relation. Referenced ASE models retain faces and texture
coordinates in compiler-only copies while unusable exported normal blocks are removed
and an object transform's node name is made identical to the object name it transforms;
the compiler derives normals from the face relation. Referenced OBJ compiler copies carry
an object row before mesh creation and a material library containing every literal
`usemtl` name. Runtime archives retain the stock model bytes.

The compile record contains every archive, concrete module, declaration-only module,
resolved alias, model skin, missing image, missing file, stage return code, leak-line
mass, and draw-surface capacity used. If q3map2 exhausts a draw-surface allocation, the
next compile doubles that resident compiler capacity. This is compile scheduling, not a
map or workload cardinality ceiling.

Visibility storage is sized from the realized portal-cluster relation. Directed-portal
ordering, leaf incidence, traversal vectors, cluster merge vectors, and histograms are
dynamic. Preliminary portal-front vectors exist only for the duration of their parallel
flood. Passage visibility is a sorted sparse relation of nonzero 32-bit words, and the
portal-flow state is an explicit heap worklist carrying that same sparse representation;
neither a dense passage matrix nor process-stack recursion represents causal reach.
Passage-memory accounting covers every
directed portal and stored word without narrowing its byte total to an integer. Grid-light
contributions are dynamically sized from the realized light relation rather than a fixed
per-worker stack array or a truncated prefix. Quadratic-patch lightmap tessellation derives
its working extent from the input grid and subdivision relation, stores that extent on the
heap, and compacts linear rows and columns into a separately sized result. It does not place
a fixed 128-by-128 draw-vertex matrix on every worker stack or stop subdivision at that
matrix boundary. The compiler does not
compare these allocations to legacy fixed cluster, portal, or portals-per-leaf counts.
The serialized BSP lump and the engine's dynamic PVS allocation are the actual
representation boundary.

Release visibility serializes the conservative portal-flood upper bound. It can retain an
occluded cluster but cannot omit a cluster from the exact portal-flow relation. On the
33,382-cluster fused world it realizes 33,135,853 visible cluster pairs, 2.974 percent of
the complete relation, in 20 seconds. Exact passage-portal refinement remains an offline
measurement over the same sparse heap worklist; it is not placed on the release build path.

## Canonical navigation object

Each source contributes every node in its largest stock walking component, its weighted
adjacency, link flags when present, and waypoint-cache links. Connector nodes and
bidirectional links join those translated stock graphs. Coordinate proximity never
creates an edge.

The realized graph records walking diameter, walking-distance distribution, unreachable
pair mass, per-tile reachable-node mass, and tile-to-tile walking distances. Join cart
navigability is the conjunction of:

- complete bidirectional incidence along the connector chain;
- attachment of both chain ends to their source regions;
- width and height aperture incidence for the complete cart-and-rider hull;
- zero exact swept-hull clearance gaps, support gaps, and portal-direction residual over
  every segment of the compiled connector chain.

The same weighted graph produces the shortest-path Voronoi relation consumed by cart
origins, spawn access, belief integration, and causal V-cell working sets. There is no
strategy-private second navigation graph.

## Continuous cart-curve domain

The compiled BSP supplies the indexed solid-brush half-space domain. That domain is the
union of the world model and every brush submodel whose server spawn contract realizes
`SOLID_BSP`; trigger, illusionary, ladder, particle, camera, and game-mode-deleted models
do not become fictional walls. Compiled quadratic patches use the renderer's collision
tessellation contract, including its curvature-derived subdivision, integer vertex snap,
and collision-triangle topology; thin triangle prisms add those surfaces to the same
half-space index used by swept hulls and support. Cache identity includes this
geometry-interface schema so a world-model-only or brush-only cache is reconstructed
rather than reused. Map measurement schema 11 publishes that interface schema and the
masses of compiled brush and tessellated patch-triangle collision atoms used by
placement. Cart paths are represented
as tangent-energy curves over stock-navmesh-derived origin components. A
component enters the origin relation when its exact weighted shortest-path horizon
reaches the required physical travel horizon. Node count, bounding-box span, relative
component size, and short-dangle pruning are not substitutes for that distance.

Every curve segment must have complete parameter-interval coverage for both:

- collision-free swept volume of the full cart-and-rider hull;
- continuous floor support under the cart's push surface after shrinking every support
  face by the horizontal cart hull.

These interval relations define the feasible domain for the entire segment. Candidate
sampling searches within that domain; it does not establish feasibility by checking a
finite set of points. Jump-pad, teleporter, and other cart-incompatible stock link
semantics do not become cart curves, though ordinary player walking distance may use
them when the stock graph permits it.

Cart-origin selection and spawn-pool construction consume shared precomputed
cart-to-waypoint distances, vectorized nearest-node attachment, and literal
point-to-waypoint attachment distance. Track orientation maximizes the minimum
start-to-start walking distance after direction-coupling measures. The entity overlay
emits only represented curves and records construction residuals for the requested cart
and team relation.

The generic spawn pool is constructed in the compiled negative-space domain eroded by
the stock player hull plus one complete player-body width on every horizontal side. Its
standing origins are lifted by one stock player lower-half height, and their coordinates
retain enough significant digits to preserve the constructed incidence relation when
parsed by the engine. This surrounding body-width realizes relocation and
simultaneous-occupancy clearance; it is not a sampled collision guess. Runtime engine
traces report spawn-origin solid incidence and intrusive entity-list errors as separate
measures of the realized map.

## Runtime artifacts

The fused package contains the realized members from this relation:

```text
maps/fused.map
maps/fused.bsp
maps/fused.ent
maps/fused.waypoints
maps/fused.waypoints.cache
maps/fused.mapinfo
maps/fused.joins.json
maps/fused.metrics.json
maps/fused.measurements.json
maps/fused.compile.json
```

Compiler-resolved asset aliases are included at their runtime logical paths. Stock game
assets and the PBR material overlay remain separately staged runtime archives, so the
client sees their union without duplicating every stock byte into `fused.pk3`.

`fused.joins.json` carries tiles, portals, joins, vantages, graph-derived prominence, and
cart-navigability measures. Each corridor records its portal carve clearance and depth,
solid-shell embed depth, longitudinal overlap, transverse overlap, and the minimum of
those positive overlaps. The shell and intact source solid have shared volume around
the entire opening, so a sloped connector cannot meet a source aperture at a merely
coplanar edge. `fused.measurements.json` carries cart construction, spawn
access, path, and team/cart reachability measures. `fused.metrics.json` joins source,
placement, compiler, BSP, navigation, geometry, artifact, and wall-time measures.
Resuming a geometry-only compile executes its missing VIS and light stages against the
already-realized fused source before reconstructing measurements and the release bundle.

## Residual measure

Incomplete work is represented as a nonnegative residual vector, not a judgment or a
filter. Its coordinates include unfinished sources, missing stock maps, compiler-stage
residuals, leak lines, unresolved compiler assets, missing BSP or bundle artifacts,
source-translation error, coordinate excess, negative-space error, overlay error,
region-component residual, non-cart-navigable joins, cart-construction residual, and
team/cart nonadvanceable pairs.

`release_residual_mass` is the sum of those coordinates. A finished realization has zero
mass while retaining every component measure that produced the sum. A nonzero coordinate
does not authorize dropping a source, team, cart, artifact, or compiler stage; it names
the remaining work.

Resume reuses the authored and compiled artifacts but reconstructs the complete measure
record, negative-space object, entity overlay, compiler overlays, and package. It does not
replace the original source relation with whatever files happened to finish first.

Join and region observation tools consume `fused.joins.json` and the runtime package.
They emit requested, observed, missing, and unreadable frame masses plus luminance-support
measures for the literal captured pixels. Those observations do not decide which geometry
is emitted and do not collapse image content into an acceptance label.
