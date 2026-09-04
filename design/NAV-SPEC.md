# NAV-SPEC — navmesh, reachability, cart-path synthesis, navigability

Quote index, per the provenance law in `SPECIFICATION.md`: every normative sentence
below is a verbatim user-authored transcript block quote. Everything else is a section
title, a pointer, or a clearly-labelled level-3 observation.

These requirements were stated across the transcript and **never joined into a spec**,
so the implementation drifted into point-sampling (`solid_brush_at` at arbitrary
operands) instead of the computed pipeline the quotes describe. This file joins them.

## 1. The pipeline is COMPUTED, not sampled

> procedural geometry computing navmeshes from each map and ensuring that at least 3
> different cart starting points, no matter the map or the level of tangling of the
> cart paths overall, are at least approximately equidistant from each other in
> map-navmesh-walking-distance from each other, no matter how many carts are sampled
> or teams are used.

Read the verbs: procedural geometry **computes navmeshes**; origins are placed by
**navmesh-walking-distance** equidistance. Nothing here is a probe. The navmesh is the
computed object every downstream decision consumes.

## 2. Cart-path synthesis: Voronoi cells, edge validity, tangent-energy curves

> it might be best to do some voronoi cell stuff and compute whether there are navmesh
> edges which let the carts travel through solid territory to get from place ot place,
> and use a different tangent energy curve optimization function to place cart paths
> which are mostly not intersecting terrain or floating substantially in the air in
> non-walking-navigable ways

Three separate obligations in one sentence: **Voronoi decomposition** over the navmesh;
an explicit **edge-validity computation** (does this navmesh edge let a cart pass
through solid?); and **tangent-energy curve optimization** as the path placer, whose
objective is paths that neither intersect terrain nor float in air that is not
walking-navigable.

## 3. The hard constraint: activation distance to negative space

> the carts don't need an 'unstick', they need to have motion plans which are always
> within activation distance of negative space

This is the load-bearing constraint and it is a property of the **motion plan**, not a
test applied afterwards. A synthesized path that satisfies it cannot burrow; an
"unstick" rule is the wrong shape by construction. "Negative space" is the user's own
term for the free volume — the thing the path must remain within pushing range of.

## 4. Semantic edge classification (why carts burrowed)

> current maps definitely still have carts that burrow into level geometry along very
> smooth waypoint following curves (beacuse some waypoints communicate jump pad paths)

A waypoint link is not a cart-traversable segment. Links encoding jump-pad or teleport
trajectories must be classified out before any geometric fitting.

## 5. One navigation definition — the stock navmesh

> even though we have a bunch of required features liek custom map objects and map-map
> procedural fusion, all of *that* code is required to be compliant with ordinary
> navmesh stuff as usedby normal playerbots, not require or use a second definition of
> navigation which is inlined inside of our playerbot strategy adapter code...

> finally, we should make sure that procedural remappings have playbot navmesh
> navigability allowing net playerbot transport between maps.

## 6. Coverage

> ideally we can render or represent a payload map for any map navigable by playerbots
> at all, whihc ideally is all default maps for the game

## 7. The cart-navigability budget across fused maps

> not all level-level connections (at least one maximum per map) need to even be
> cart-path-navigable

At most ONE non-cart-navigable connection per map — a budget, not a licence to refuse
joins.

## 8. V-cells fuse contiguous NAVIGABLE paths

> why v-cells" (you should have asked this): beacuse we can make v-cells then fuse
> together contiguous navigable paths until our distance decay term (think context
> masking, parallel operations, not recurrent operations) is limiting the scope of what
> nearby observations are affecting a playerbot's decisions to a bounded compute use
> and a bounded maximum total map area that could be described *and* a bounded minimum
> map area that could be described/affected by the integration over adjacent cells

The same Voronoi-over-navmesh structure serves both the cart-path placer (§2) and the
belief map-reduce — fusing along *navigable* adjacency, not geometric proximity.

The serialized stock-compatible waypoint relation is reconciled as specified by
`GEOMETRY-RECONCILIATION.md`. Authored coordinates retain stable identities; cache-implied
vertices are materialized; and every cache endpoint is transduced through the same
source-to-realized coordinate map. The stock bot loader, computed metric graph, and fused
serializer therefore see the same vertices and edges. Degenerate waypoints are projected
onto the player-hull erosion of the compiled brush-and-patch collision domain rather than
removed. Nondegenerate trigger boxes retain their authored volume. Input collision mass,
displacement moments, and unresolved projection mass remain distinct measures.

## 9. Inspection tooling

> also you might need to make a 3d viewer or renderer tool which renders the floorplans
> of procedural levels and their level fusions at each of their joins, with some
> fuzzing tools to quickly measure and demonstrate how contorted of a path a player
> agent has to wind through a level<->level edge to cross it, and actually rendering the
> egocentric player view through each side of the edge

## 10. Required data flow (derived from §§1–8; level-3)

```
map BSP
  ├─► COMPUTE stock navmesh metric graph             (§1; stock-compatible, §5)
  │     └─► shortest-path Voronoi cells              (§2, §8)
  │           ├─► fuse contiguous NAVIGABLE cells    (§8)
  │           └─► k-center origins, equidistant in
  │               navmesh-walking-distance (>=3)     (§1)
  └─► compiled solid half-space domain
        ├─► classify edges: cart-traversable?        (§2 edge validity, §4 semantic)
        └─► tangent-energy curve optimization
              placing cart paths, subject to:
                · not intersecting terrain           (§2)
                · not floating in non-walking-
                  navigable air                      (§2)
                · every path point within
                  activation distance of
                  negative space                     (§3)  ← by construction
```

Validation is therefore **by construction and by structural predicate**: a path emitted
by a solver constrained on §3 cannot burrow, and a spawn or node placed *in* computed
free space cannot be in solid. Any point-sampled `is-this-solid?` check is a symptom
that a stage of this pipeline is missing, not a validator to improve.
