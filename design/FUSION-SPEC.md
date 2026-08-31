# Fusion specification — the quote index (map fusion / megamaps)

Companion to `SPECIFICATION.md`, under the same content rule: **every normative
sentence in §1–§3 is a verbatim user-authored transcript block quote**, cited by
session prefix + timestamp. Everything outside a block quote is provenance, a
section title, or a pointer. Where a `design/` doc or a tool disagrees with a
quote here, the quote governs.

Source: raw transcript `~/.claude-personal/projects/-Users-mdot/d3ad4328-….jsonl`
(session prefix `d3ad4328`), user-authored turns only. Quotes copied verbatim,
user typos preserved. §4 is explicitly **level-3** (implementation description)
and is written in code and artifacts, not in quotes.

This document exists because the fusion requirement was never transcribed. The
shipped status line read *"3 maps, 3 joins"*, and the owner rejected it:

`d3ad4328`, 2026-08-31T02:37:08Z:

> "- Fusion runs [x] — fused.bsp 8.9 MB / fused.pk3 2.4 MB, 3 maps, 3 joins
> (corridor 2151u, teleporters 3899u/4062u)." find the actual requirements from the
> transcripts involved and finish the fight. by subagent. '3 maps' is not the fusion
> spec or anything close to it, so the fusion spec was not transcribed,
> mistranscribed, etc

---

## 1. The fusion requirement itself

The single controlling turn. Everything about j, k, the tileset structure, the
combinatorics, and the cart-navigability exemption comes from here.

`d3ad4328`, 2026-08-29T08:26:14Z:

> remember that you can use procedural geometry programming and similar to glue
> multiple xonotic maps together if you want to make payload multicart
> highteamcount environments, and that this can be done procedurally to compose
> extremely high combinatorics of maps, e.g. using k-many bridge maps to allow
> j-many of all of the maps in the game to socketed together like tilesets in a
> roguelike level generator. make sure to think of ways to use the portal and jump
> pad and verticality in the xonotic levelset here. not all level-level connections
> (at least one maximum per map) need to even be cart-path-navigable

Reading, clause by clause — each clause is normative:

- **"procedural geometry programming"** — the glue is generated geometry, not
  hand-authored. Not a synonym for "BSP lump concatenation"; the sibling clause
  "compose extremely high combinatorics" makes it a generator.
- **"k-many bridge maps"** — a *bridge map* is a distinct class of tile, plural
  and parameterised by k. It is not the same object as a corridor between two
  stock maps.
- **"j-many of all of the maps in the game"** — the pool is *all of the maps in
  the game*; j is how many of them are drawn. j is a parameter, and its ceiling is
  the whole pool. It is not 3.
- **"socketed together like tilesets in a roguelike level generator"** — a lattice
  of tiles with sockets, drawn per seed; per-run regeneration, not one fixed world.
- **"portal and jump pad and verticality"** — the connector vocabulary must
  include teleporters, jump pads, and vertical structure, not corridors alone.
- **"not all level-level connections (at least one maximum per map) need to even
  be cart-path-navigable"** — a cart-navigability constraint with an explicit
  budget: **at most one** non-cart-navigable connection **per map**. Every other
  connection must be cart-path-navigable. (A corridor is cart-navigable; a
  teleporter or jump pad is not — a payload cart cannot ride one.)

The pool is named in the immediately preceding geometry thread:

`d3ad4328`, 2026-08-29T07:25:55Z:

> ideally we can render or represent a payload map for any map navigable by
> playerbots at all, whihc ideally is all default maps for the game

## 2. What the megamap is FOR — the commitment cost

`d3ad4328`, 2026-08-31T01:59:15Z:

> 'are multiple xonotic maps being glued tgoether to form megamaps which require
> bots ot navigate long distances (literally illustrating commitment to strategy
> related to cartstates) and integrate informatino from the observation mapreduce
> paremetric featurization function (did this get forgotten too)' should also be
> answerable along each subcomponent in around the same timespan.

The scale the megamap has to serve:

`d3ad4328`, 2026-08-29T08:23:54Z:

> i think every cart is stuck...? anyways lets focus on the major interesting
> theaters of interaciton here: 1: higher botcount, 2: higher teamcount, 3: higher
> cartcount (e.g. 5te, 3ca) 4: more planning or strategic chocie content in the
> playerbot-level strategy computatino so that there can be e.g. distributional
> strategies besides 'everyone in the team commits to whatever the strongest player
> is doing'

Why fusion in particular is the lever on track placement:

`d3ad4328`, 2026-08-29T08:43:22Z:

> we probably need a track palcement algorithmic which avoids putting all of the
> cart tracks in the same place and in the same direction, evne though this is
> pretty funny. the map gluing exercise should help a lot with that, as would
> trying to assemble cahins of control points that could counterfactually form a
> cart track, ahve a branching factor of at least 1.5 in expectation, and can then
> have a combinatorics of realized track spans sampeld, then a track directionality
> and starting point configuration that avoids a cart vector field all moving in
> the same direction on the map (winning one cart means winning every cart)

And the equidistance requirement the fused world has to be able to satisfy:

`d3ad4328`, 2026-08-29T07:21:53Z:

> procedural geometry computing navmeshes from each map and ensuring that at least
> 3 different cart starting points, no matter the map or the level of tangling of
> the cart paths overall, are at least approximately equidistant from each other in
> map-navmesh-walking-distance from each other, no matter how many carts are
> sampled or teams are used.

## 3. Navigability, signposting, prominence, and the diagnostics

`d3ad4328`, 2026-08-29T20:19:10Z (the whole normative turn; it is simultaneously
the viewer requirement, the fuzzing requirement, the corridor-length complaint, the
navmesh requirement, and the prominence rule):

> almost enitrely visible players now, so if there was a rendering error that came
> from scaling to 256 adn beyond players... we should fix that in the client build
> and so on i think, or fix whatever it is we're doing that leads to densely
> materialized entities-acting-as-pairs-for-other-entities... also you might need to
> make a 3d viewer or renderer tool which renders the floorplans of procedural
> levels and their level fusions at each of their joins, with some fuzzing tools to
> quickly measure and demonstrate how contorted of a path a player agent has to wind
> through a level<->level edge to cross it, and actually rendering the egocentric
> player view through each side of the edge to look for culling, occlusion, or
> clutter errors where map graphics conceal map transitions, where map transitions
> are physically blocked by clipping planes or objects, etc. also the first corridor
> i found connecting two levels was really long... finally, we should make sure that
> procedural remappings have playbot navmesh navigability allowing net playerbot
> transport between maps. it's okay if some map connectinos are subtle or weakly
> signposted, but only if that's because that's a 'connector' map which has multiple
> edges connecting it to multiple other map node; the exclusive mode of entry or
> exit to a gameplay objective should be easy for both playerbots and players to
> notice, fight over, have tug of wars through, etc.

The navigation-ownership boundary the fusion code must respect:

`d3ad4328`, 2026-08-30T23:54:51Z:

> even though we have a bunch of required features liek custom map objects and
> map-map procedural fusion, all of *that* code is required to be compliant with
> ordinary navmesh stuff as usedby normal playerbots, not require or use a second
> definition of navigation which is inlined inside of our playerbot strategy adapter
> code...

Reactions that fixed defect classes (each is why a mechanism exists):

`d3ad4328`, 2026-08-29T08:58:34Z:

> intriguing im curous about that fused multimap

`d3ad4328`, 2026-08-29T09:03:15Z:

> oh uh oh the added map has no textures or lighting in the spawn area lol

`d3ad4328`, 2026-08-29T19:25:44Z:

> the fused map looks like its joined in a few ways! interesting error in the
> runtime: at least some playerbots are invisible (?) but this can be fixed pretty
> easily i'm sure

And the standing statement of what was still missing, delivered with an in-client
screenshot of the fused map running on a live server
(`[this session, unflushed]` — relayed verbatim by the coordinating agent, not yet
present in the on-disk jsonl):

> the map fusion code was also clearly unfinished and didn't satisfy any written
> constraint in a few obvious ways, incl. missing viewers, msising client
> renderers, missing features related to geometry fusion, total absence of
> procedural geometry, and no connectivity solvers or trivial visualizers or
> metrics over connectivity and navmesh solutions

### 3.1 Ambiguities, flagged rather than invented

- **"at least one maximum per map"** is read as *at most one* non-cart-navigable
  connection per map. The alternative reading ("at least one connection per map may
  be non-cart-navigable, and that is a floor") would make the clause vacuous, since
  a floor on permission constrains nothing. The implemented reading is the
  constraining one.
- **j and k are not given numeric values anywhere.** The only numeric anchors are
  "all of the maps in the game" (the pool ceiling) and "extremely high
  combinatorics". Implemented as parameters whose default j is the whole navigable
  stock pool and whose default k is one bridge per three stock maps.
- **"the first corridor i found connecting two levels was really long"** is read as
  a defect report (corridors should be short), which sits alongside §2's demand that
  the *megamap* impose long traversals. The implemented reconciliation: long distance
  comes from tiles in series (walking diameter), short corridors come from joining
  only lattice-adjacent tiles. Both are measured separately and reported.

---

## 4. LEVEL 3 — what the implementation does (code and artifacts, not quotes)

> **§4.1 (fixed lattice), §4.3 (refusal-based budget) and §6.3 are superseded by §7.**
> The joins are no longer tubes between intact maps: the maps' own geometry is edited.

This section is **not** normative and is **not** quoted from the user. It block
quotes real code in `xonotic/payload/tools/` and real generated artifacts.

### 4.1 The lattice, the bridge tiles, and the draw

`mapfuse.py` places `T = j + k` tiles on a rectangular lattice with non-uniform
bands, picks k bridge cells by greedy max-coverage over lattice neighbours, and
only ever joins lattice-**adjacent** cells:

```python
def plan_tiles(nsrc, k):
    T = nsrc + k
    cols = max(1, int(math.ceil(math.sqrt(T))))
    rows = int(math.ceil(T / cols))
    cells = [(i % cols, i // cols) for i in range(T)]
    ...
        best = max(rest, key=lambda c: (len(set(nbr[c]) - covered - set(bridges)), len(nbr[c]), ...))
```

The default draw is the whole navigable stock pool, shuffled by seed:

```python
        if nmaps in (None, 'all'):
            names = list(pool)
            random.Random(seed).shuffle(names)
        else:
            names = random.Random(seed).sample(pool, min(int(nmaps), len(pool)))
```

The pool is read out of the stock map pk3 and is 29 maps:

```
$ python3 -c "... mapfuse.navigable_names(pk3)"
29
['afterslime','atelier','boil','bromine','catharsis','courtfun','dance','darkzone',
 'erbium','finalrage','fuse','geoplanetary','glowplant','go','implosion',
 'leave_em_behind','nexballarena','opium','runningman','runningmanctf','silentsiege',
 'solarium','space-elevator','stormkeep','techassault','trident','vorix','warfare','xoylent']
```

### 4.2 Procedural geometry — the bridge tiles are generated and q3map2-compiled

`mapgen.bridge_tile()` emits a two-tier hub in `.map` source and compiles it with
the real `q3map2` (`-meta`, `-vis`, `-light`), then writes its own
`.waypoints`/`.waypoints.cache`. Every port is a real node of the tile's waypoint
graph, and the gallery ring above the hub floor is reached by a jump pad and left
by a teleporter (the "portal and jump pad and verticality" clause):

```python
    arms_lo = list(arms_lo) + list(arms_hi)      # all ports on the ground tier:
    arms_hi = []                                 # a port must be cart-navigable
    ...
    padat = [0.0, -(h - 200.0), 0.0]
    land = [0.0, h - gw / 2, TIER]
    g.jumppad(padat, land, 0)
    g.teleporter(land, [0.0, -(h - 400.0), 0.0], 90, 0)
```

Each arm is capped with a thin caulk plug so q3map2 sees a sealed hull; the plug is
small enough that the fusion's corridor carver removes it when the join punches
through — the same carve path that was already proven on stock geometry.

### 4.3 Cart-navigability budget

```python
    noncart_used = [0] * j
    noncart_budget = 1
    ...
        may_noncart = noncart_used[a] < noncart_budget and noncart_used[b] < noncart_budget
```

Corridors are tried first and always; the teleporter/jump-pad fallback is rationed.
A redundant (non-cut) edge that can be neither a corridor nor a rationed portal is
**dropped**, guarded by a union-find connectivity test, so the budget is a hard rule
wherever the region graph allows it.

### 4.4 Prominence

The proven degree-1 rule is kept and extended to cut edges, which are the exclusive
mode of entry to everything behind them:

```python
    PRE = region_graph_solve(j, edges)
    cutset = set(PRE['cutedges'])
    exclusive = [min(degree[a], degree[b]) == 1 or ei in cutset for ei, (a, b) in enumerate(edges)]
```

### 4.5 Connectivity solver and navmesh metrics

`region_graph_solve()` is an iterative Hopcroft–Tarjan over the region graph:
components, articulation points (chokepoint tiles), cut edges (chokepoint joins),
degrees, hop-diameter. `navmesh_solve()` is a heap Dijkstra over the *real* fused
bot-waypoint graph: per-region reachable coverage, the region-to-region walking
distance matrix, the walking diameter (the commitment cost), and the count of
unreachable region pairs. Both are printed and written to `fused.metrics.json`.

### 4.6 A bug in a sibling-owned file that this work depends on

`mkentfile.Bsp` derives a brush AABB from the plane distance of any plane within
`0.999` of an axis:

```python
            for nx, ny, nz, dd in bp:
                for a, c in enumerate((nx, ny, nz)):
                    if c > 0.999:
                        hi[a] = min(hi[a], dd)
```

For an *oblique* plane (a join corridor is oblique by construction) `dd` is not an
axis-aligned bound, so the AABB is shifted. Measured on `fuse_b16`: a corridor whose
real x-span is `[-5504,-5152]` is indexed at `[-5843,-5491]`, and `Bsp.floor()`
then reports phantom "no floor" violations along a corridor that is in fact clear.
`mapfuse` no longer routes its clearance check through `Bsp`; the one-line change
needed in the sibling-owned file is reported separately.

---

## 5. GAP TABLE

| # | Requirement (verbatim) | Cite | Before this work | Now |
|---|---|---|---|---|
| F1 | "j-many of all of the maps in the game" | 08-29T08:26 | `names = random.Random(seed).sample(pool, 3)` — hard-coded 3 | `--maps=N`/`--maps=all`; default = whole 29-map navigable pool, seed-shuffled |
| F2 | "using k-many bridge maps" | 08-29T08:26 | no bridge-map class at all; only stock maps + raw corridors | `--bridges=k`; k procedurally generated, q3map2-compiled hub tiles placed as lattice cells |
| F3 | "procedural geometry programming" | 08-29T08:26 | `mapgen.py` existed but nothing it produced was ever in a fused world ("total absence of procedural geometry") | `mapgen.bridge_tile()` → `.map` → q3map2 → BSP → ingested by `mapfuse` as a first-class tile |
| F4 | "socketed together like tilesets in a roguelike level generator" | 08-29T08:26 | random spanning tree over arbitrary map pairs | lattice with 4-neighbour sockets; per-seed draw, bridge placement, topology, socket assignment |
| F5 | "extremely high combinatorics of maps" | 08-29T08:26 | 1 shape (3 maps, 3 joins) | draw × lattice assignment × bridge cells × loop edges, all seed-driven |
| F6 | "portal and jump pad and verticality" | 08-29T08:26 | teleporter + jump pad connectors existed; no vertical structure was generated | kept, plus a generated two-tier hub: gallery ring, jump pad up, teleporter down |
| F7 | "at least one maximum per map ... cart-path-navigable" | 08-29T08:26 | inverted: `if corridor_used[a] == 0 and corridor_used[b] == 0` capped corridors at **one per map**, so most joins were non-cart-navigable | per-tile budget of **one** non-cart join; corridors tried first and always; redundant unsatisfiable edges dropped under a connectivity guard |
| F8 | "megamaps which require bots ot navigate long distances" | 08-31T01:59 | never measured | `navmesh_solve()` prints and stores region-to-region walking distances, median and diameter |
| F9 | "the first corridor i found connecting two levels was really long" | 08-29T20:19 | uniform cells sized to the largest map in the pool; sockets picked by distance from map centroid with no reference to the partner | non-uniform lattice bands + in-band nudge toward joined neighbours + `pick_sockets_toward()` aiming at the shared band boundary |
| F10 | "playbot navmesh navigability allowing net playerbot transport between maps" | 08-29T20:19 | flood-fill existed and passed at 3 maps | kept; plus per-region coverage and unreachable-pair counts at every scale |
| F11 | "subtle or weakly signposted ... only if that's ... a 'connector' map which has multiple edges" | 08-29T20:19 | prominence keyed on degree-1 only | degree-1 **or cut edge**; bridge tiles are the connector class by construction and carry the multi-edge, may-be-subtle role |
| F12 | "3d viewer or renderer tool which renders the floorplans ... and their level fusions at each of their joins" | 08-29T20:19 | `joinview.py` floorplan SVG only | `fusegraph.py`: region-graph + navmesh + metrics viewer; floorplan retained |
| F13 | "fuzzing tools to quickly measure ... how contorted of a path" | 08-29T20:19 | `joinview.contortion()` existed | retained and reported per join in the new viewer |
| F14 | "actually rendering the egocentric player view through each side of the edge" | 08-29T20:19 | `joinshot.py` rendered joins only | plus per-region vantage cameras and an automated **void audit** over the rendered frames |
| F15 | "no connectivity solvers or ... metrics over connectivity and navmesh solutions" | [this session, unflushed] | one boolean flood-fill | Hopcroft–Tarjan components/articulation/cut-edges/hop-diameter + Dijkstra walking-distance matrix, coverage, diameter |
| F16 | single-cluster PVS fix, `fused.pk3` client distribution, clip-brush carving, canonical pad/teleporter entities, region flood-fill, corridor sizing | prior proven work | present | preserved unchanged; all re-verified at every scale below |
| F17 | "missing client renderers ... the world is almost entirely black void" | [this session, unflushed] | join cameras only; no grading of what was rendered | region vantage + overhead cameras, Adam7 PNG reader, automated void audit; 166 real frames, PASS |
| F18 | the megamap must actually run | implied by all of the above | 3-map fusion ran; anything larger died with `server runaway loop counter hit limit of 10000000 jumps` | diagnosed (compiled-in limit + linear IntrusiveList ops) and lifted by an entity budget + orphan sweep inside `mapfuse`; the 39-tile world boots and soaks with 8 bots |

---

## 6. Evidence — real generated artifacts

No unit test, no simulator. Every number below is printed by a real run of
`mapfuse.py` / `fusegraph.py` / `joinshot.py` on this machine, or is a real file
size on disk. Nothing was re-simulated.

### 6.1 The full-pool megamap — all 29 navigable stock maps + 10 procedural bridge tiles

```
$ python3 xonotic/payload/tools/mapfuse.py 21 --maps=all --bridges=10 --out=/private/tmp/fuse_all29
mapfuse seed=21 j=29 stock maps + k=10 procedural bridge tiles (pool=29) pk3=xonotic-20230620-maps.pk3
lattice: 7x6, 39 tiles (29 stock maps + 10 procedural bridge tiles), 48 lattice edges
topology: 39 tiles (29 stock + 10 procedural bridge), 48 edges (38 tree + 10 loops),
          corridors=26 jumppads=6 teleport-triggers=22
prominence: 13 exclusive/objective edges (prominent+lit), 27 redundant edges (subtle)
corridor length: n=26 min=1604 median=4791 max=5938 (cap 6000)
cart-navigability: 26/40 joins cart-navigable (corridor); non-cart joins per tile max=3
          (budget 1); 8 loop edges dropped -> VIOLATED on 4 edges (see 6.3)
entity budget 1800: swept 1622 orphaned target-only entities; dropped {'light': 4149,
          'dom_team': 60, 'dom_controlpoint': 48, 'trigger_race_checkpoint': 25,
          'info_player_race': 14, 'func_pointparticles': 47, 'misc_gamemodel': 197,
          'misc_breakablemodel': 108, 'item_armor_small': 579, 'item_health_small': 489,
          'item_shells': 47, 'item_bullets': 99, 'item_rockets': 155, 'item_cells': 152,
          'item_health_medium': 171, 'item_armor_medium': 39}
entities: dropped 7 source spawnpoints in solid, 309 over the per-tile spawn budget of 10
wrote /private/tmp/fuse_all29/fused.bsp (165914944 bytes, 69132 nodes 69704 leafs 561 models)
connector clearance check (exact planes, un-carved source solids): PASS (0 obstructed samples)
bot flood-fill: regions reached [0..38] / [0..38] -> PASS
connectivity: 1 component(s) [39]; hop-diameter=19; 21 chokepoint tiles; 32 cut edges
navmesh: 5162 fused waypoints; bot-reachable from seed = 5152 (99.8%);
         regions with zero reachable waypoints: none
navmesh: region<->region WALKING distance median=58556u diameter=152281u unreachable_pairs=0
fuse wall time 72.7s   peak RSS 5.35 GB
```

Files on disk:

```
-rw-r--r--  1 mdot  wheel  165914944  /private/tmp/fuse_all29/fused.bsp
-rw-r--r--  1 mdot  wheel   48924307  /private/tmp/fuse_all29/fused.pk3
-rw-r--r--  1 mdot  wheel      57438  /private/tmp/fuse_all29/fused.connectivity.json
-rw-r--r--  1 mdot  wheel      24244  /private/tmp/fuse_all29/fused.graph.svg
-rw-r--r--  1 mdot  wheel    2522908  /private/tmp/fuse_all29/fused.navmesh.svg
```

And it **runs**, on the real dedicated server, in the real payload gametype, with bots:

```
$ darkplaces-dedicated -xonotic ... +g_payload 1 +bot_number 8 +skill 5 +timelimit 2 +map fused
Server listening on address 0.0.0.0:26014
payload: teams mask 31, carts 0
payload: cart 0: 30 path nodes, length 21293.392578
payload: cart 1: 15 path nodes, length 7036.235840
payload: cart 2: 12 path nodes, length 7863.403320
execing post-config.cfg
$ grep -cE "Host_Error|Quake Error" soak.log
0
(94 s soak, 8 bots, steady RSS 2.55 GB)
```

Cost envelope on the shared machine (this is the right-sizing datum): **5.35 GB peak
RSS, 73 s wall** to generate the whole 29-map pool; **2.55 GB** for the dedicated
server to run it. The 22-tile run costs 3.0 GB / 31 s and the 8-tile run 0.9 GB / 7 s,
so generation cost is roughly linear in fused brush count and the whole pool fits
inside the ~32 GiB ceiling with a wide margin.

### 6.2 The scaling ladder (each row is one real run)

| tiles | stock + bridges | joins | corridors | cart-nav | flood-fill | navmesh reachable | walking diameter | fused.bsp | fuse wall | fuse peak RSS | dedicated server |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3 + 0 | 2 | 1 | 1/2 | PASS 3/3 | 438/438 = 100% | 12 544u | 45.3 MB | 9.3 s | — | — |
| 4 | 3 + 1 | 3 | 3 | 3/3 | PASS 4/4 | 563/563 = 100% | 15 874u | 45.3 MB | 4.4 s | — | — |
| 8 | 6 + 2 | 9 | 9 | 9/9 | PASS 8/8 | 979/979 = 100% | 22 154u | 54.5 MB | 7.2 s | 0.9 GB | boots (client render, 166 frames) |
| 22 | 16 + 6 | 23 | 11 | 11/23 | PASS 22/22 | 2768/2772 = 99.9% | 97 219u | 95.6 MB | 31.4 s | 3.0 GB | boots payload match |
| **39** | **29 + 10** | **40** | **26** | **26/40** | **PASS 39/39** | **5152/5162 = 99.8%** | **152 281u** | **165.9 MB** | **72.7 s** | **5.35 GB** | **boots + 94 s soak, 8 bots, 0 errors** |

The commitment cost the spec asks for is the walking-diameter column: at full pool a bot
crossing the megamap between the two furthest regions walks **152 281 units** of real
bot-waypoint graph, versus 12 544 units for the old three-map fusion — a **12.1×**
increase in the distance a strategy has to commit to.

### 6.3 Where the cart-navigability budget still fails, honestly

> **SUPERSEDED by §7.5.** The refusal described below is deleted; the budget now holds
> at full pool with zero refusals and zero dropped edges. Kept for the record.


At 8 tiles the budget holds outright (`9/9 joins cart-navigable, max=0 non-cart per
tile, HELD`). At 22 and 39 tiles it is violated on a handful of tiles
(4 edges at 39 tiles). The cause is measured, not guessed: a corridor is refused when
the two facing sockets are more than `MAXCORLEN = 6000` apart, and the widest stock
maps (`space-elevator`, `xoylent`, `catharsis`) put their walkable interiors far from
their bounding-box faces, so their band-adjacent joins exceed the cap. Raising the cap
would satisfy the budget by building exactly the kilometre-long tunnels the owner
complained about at 08-29T20:19. The tension is left visible and reported per run
rather than resolved by silently widening the cap.

### 6.4 Joins are cart-navigable, near-straight and unblocked

`joinview.py` on the 8-tile fusion, every join a corridor:

```
$ python3 xonotic/payload/tools/joinview.py /private/tmp/fuse_r8
wrote fused.floorplan.svg (8 maps, 979 nav nodes, 9 joins, 878 lights, 739 clip brushes)
  bridge7_0<->silentsiege corridor [EXCL/prominent] len=877 : contortion min=0.33 mean=0.82 max=1.12 | clip-blocked=no
  bridge7_1<->catharsis   corridor [EXCL/prominent] len=3540: contortion min=0.87 mean=0.96 max=1.03 | clip-blocked=no
  bridge7_1<->bridge7_0   corridor [redundant]      len=2460: contortion min=0.90 mean=1.04 max=1.13 | clip-blocked=no
  glowplant<->bridge7_0   corridor [redundant]      len=2219: contortion min=0.78 mean=0.93 max=1.03 | clip-blocked=no
  atelier<->boil          corridor [redundant]      len=1284: contortion min=0.60 mean=1.03 max=1.86 | clip-blocked=no
  glowplant<->atelier     corridor [redundant]      len=1374: contortion min=0.67 mean=0.91 max=1.08 | clip-blocked=no
  bridge7_0<->boil        corridor [redundant]      len=2910: contortion min=0.83 mean=0.95 max=1.03 | clip-blocked=no
  fuse<->bridge7_1        corridor [redundant]      len=1792: contortion min=0.75 mean=0.93 max=1.05 | clip-blocked=no
  fuse<->glowplant        corridor [redundant]      len=3294: contortion min=0.86 mean=1.00 max=1.25 | clip-blocked=no
```

Mean contortion 0.82–1.04 means a bot's real walked path across a level-to-level edge
is within a few percent of the straight line: no join is a maze.

### 6.5 Chokepoint / betweenness solve (`fusegraph.py`, 8-tile fusion)

```
== region graph ==
tiles 8 (2 procedural bridge)  joins 9
components [8]  hop-diameter 4
chokepoint TILES  (articulation): ['bridge7_1', 'bridge7_0']
chokepoint JOINS  (cut edges)   : ['bridge7_0<->silentsiege', 'bridge7_1<->catharsis']
2-edge-connected blocks (sizes) : [6, 1, 1]
== navmesh ==
waypoints 979  region-pairs solved 28  unreachable pairs 0
walking distance median 9351u  DIAMETER 20643u
== joins ==
  bridge7_0<->silentsiege corridor len=877  cart=True prominent=True  cut=True  betweenness=0.25
  bridge7_1<->catharsis   corridor len=3540 cart=True prominent=True  cut=True  betweenness=0.25
  bridge7_1<->bridge7_0   corridor len=2460 cart=True prominent=False cut=False betweenness=0.32
  ...
```

Both chokepoint tiles are the procedural bridge tiles, and both chokepoint joins are
exactly the two prominent/exclusive edges — the prominence rule and the connectivity
solve agree without being told to.

### 6.6 Files this work owns

- `xonotic/payload/tools/mapfuse.py` — lattice + bridge tiles + budget + solvers
- `xonotic/payload/tools/mapgen.py` — `bridge_tile()` / `build_bridge_tile()`
- `xonotic/payload/tools/fusegraph.py` — **new**: connectivity solvers, navmesh
  metrics, region-graph and navmesh SVG viewers
- `xonotic/payload/tools/joinshot.py` — region vantage cameras, overview cameras,
  Adam7-capable PNG reader, void audit
- `xonotic/payload/tools/joinview.py` — unchanged, still the contortion/occlusion fuzzer

### 6.7 The engine ceiling, diagnosed and lifted

The first attempt to boot the megamap failed, and the failure is worth recording
because it is the real reason "3 maps" had never been exceeded:

```
Host_Error: server runaway loop counter hit limit of 10000000 jumps
   ./lib/log.qh : print_assertfailed_severe : statement 56
        rametime : IL_PUSH : statement 9
   race_endpos_z : relocate_spawnpoint : statement 81
   race_endpos_z : __spawnfunc_info_player_deathmatch : statement 4
        thmatch : __spawnfunc_spawn : statement 49
                : __spawnfunc_worldspawn : statement 814
                : StartFrame : statement 10
server Profile:
   948212  16119604   100.00%  il_links_flds##GETFP
     5643   7629864    32.13%  IL_REMOVE_RAW
     1538   6693370    99.91%  InitializeEntity
     1787   3897707    52.19%  ONREMOVE
   226087   3391305   100.00%  il_links##GET
     3704    185143 ... 40072842 total  __spawnfunc_spawn
```

Diagnosis, all measured:

- the 10 000 000-jump limit is **compiled into this DarkPlaces build** —
  `cvarlist prvm_runaway` returns `0 cvar beginning with "prvm_runaway"`, so it cannot
  be raised from the command line;
- stock Xonotic's IntrusiveList primitives (`IL_CONTAINS`, `IL_REMOVE_RAW`,
  `il_links_flds##GETFP`) are **linear scans**, so worldspawn's spawnfunc chain is
  superlinear in merged entity count;
- measured threshold: **1805 entities (8 tiles) boots; 5780 entities (22 tiles) does
  not**, with `il_links_flds##GETFP` alone burning 16.1 M of the 10 M budget.

Lifted inside `mapfuse` (owned code), not by touching the engine or the QC:

1. per-tile spawnpoint budget (10), plus dropping source spawnpoints that test
   in-solid — these are what drive `relocate_spawnpoint`'s expanding search;
2. an entity budget with an explicit drop order. `light` goes first and costs nothing:
   `mapfuse` flattens the lightmap lump to a single grey block, so not one source
   `light` entity affects the fused world;
3. an **orphan sweep** over target-only classes (`target_position`, `info_null`,
   `misc_teleporter_dest`, `target_location`), repeated to a fixed point, protecting
   anything still pointed at by name — 1622 swept at full pool;
4. a name-reference guard so nothing that is still targeted is ever dropped (the fix
   for `follow: could not find target/killtarget`).

Result: 39-tile `fused.ent` goes **3755 → 1714 entities** and the megamap boots.

### 6.8 Real client renders + the void audit

166 real DPSOFTRAST frames were captured on the 8-tile fusion across two runs (92 + 74),
standing on real bot-reachable waypoints inside every region on four yaws, plus
overhead cameras, plus both sides of every join:

```
$ python3 xonotic/payload/tools/joinshot.py /private/tmp/fuse_r8 --regions --overview
9 joins, 8 regions -> 92 camera frames
spawned after 63s; capturing 92 frames...
captured 92/92 frames
void audit: 92 frames graded, 0 void/missing
  r05_silentsiege_v0_y0                    void=0.00 levels= 58 320x200
  r04_bridge7_0_v1_y0                      void=0.00 levels= 58 320x200
  ov_world                                 void=0.62 levels= 15 320x200
  ...
void audit: PASS (wrote voidaudit.json)
```

The audit is the thing that would have caught the reported screenshot: a camera
standing on real walkable geometry whose frame is near-black with almost no distinct
luma levels is graded VOID and fails the run offline. It needed an Adam7-capable PNG
reader, because DarkPlaces writes interlaced PNGs and there is no PIL on this box —
the first version of the audit could not read a single engine frame.

**Scope of the render evidence:** the 8-tile fusion, in the software rasterizer. The
22- and 39-tile worlds are proven only through the real dedicated server (§6.1), not
through a client render; a 166 MB BSP under DPSOFTRAST was not attempted on a shared
machine.

---

## 7. LEVEL 3 (second pass) — the joins are now CUT INTO THE MAPS

Not normative. This section supersedes §4.1's fixed grid, §4.3's refusal-based
budget and §6.3's honest failure. It block quotes real code and real run output.

The controlling re-statement of the requirement (owner, this session, relayed
verbatim by the coordinating agent):

> i think the constraint was simpler: pick a list of maps that seem well suited to
> having geometry edited to make them diegetically connect to other neighboring maps
> by litearlly changing their geometry to have doors, galleries, passageways, etc.,
> which either continue or newly appear in existing maps in plausible spots. then
> solve a 3d bin packing problem, evne poorly, where 'bridge maps' (more than 3
> connection sites) are joined by procedural geometry to 'stub maps' (fewer than 3,
> more than 1 connection sites).

Everything below follows that sentence clause by clause.

### 7.1 "maps that seem well suited to having geometry edited" — a measured criterion

`mapfuse.map_sites()` looks for the places a door could honestly be cut. From each
node of a map's largest **bot-reachable, stand-on-able** waypoint component that is
extreme in one of the four cardinal directions, it marches a ray outward
(`ray_runs`, the same exact-plane solid predicate the carver uses) and requires:

* the first solid is **24–640 u** away — a wall at the walkable frontier, not a
  tunnel through the middle of the level;
* that solid is **8–384 u thick** — a wall panel, not bedrock or a terrain skirt;
* nothing else within 224 u behind it — one facade, not stacked scenery;
* an oriented door-sized volume of **standing room in front of it** (`free_slab`);
* **open space on the far side** for the connector to meet.

Two consequences are worth stating because they are the criterion doing its job:

* A map whose shell is patch-mesh curvature rather than brushwork yields **no ray
  hits at all** and is scored 0 — `dance`'s east frontier has no brush between
  x=1872 and x=3072. That is not a bug to route around; a curved patch shell is
  exactly a map whose geometry cannot honestly be edited into a doorway.
* A site is classed **`continue`** when the standing room in front of the wall is
  narrow (a passage or alcove running into it) or the nav node is a graph dead-end —
  the opening then continues a feature the level already has, which is the most
  diegetic edit available. Otherwise it is a **`newcut`** on a broad exterior-reading
  wall, and gets the jamb/header architrave so it reads as deliberate.

### 7.2 The taxonomy is counted, not assumed

```python
def classify(nsites):
    if nsites > 3:  return 'bridge'
    if nsites > 1:  return 'stub'
    return 'unsuitable'
```

Over the whole 29-map navigable pool (real run, seed 1):

```
selection: REJECTED 1 map(s) with fewer than 2 connection sites: nexballarena(1)
selection: 29 maps kept -- 25 BRIDGE maps (>3 connection sites), 4 STUB maps (2-3 sites); 1 rejected
```

Per-map site counts and classes are printed for every candidate and stored in
`fused.metrics.json` under `selection`.

### 7.3 "solve a 3d bin packing problem, evne poorly"

Taken literally. `pack_offsets` is a shelf pack: a lattice column is only as wide as
the widest hull in it, a row only as deep, and a **level only as tall** — the pack is
three-dimensional, `levels=2` by default above 12 tiles. Cells are ranked by how many
lattice neighbours they have, tiles by how many connection sites they have, and the
two rankings are zipped: bridge maps land in the cells with the most adjacencies, stub
maps in corners. Non-overlap is structural (each hull is clamped inside its own slot),
and `split_tree()` reads the BSP router's binary partition straight off the pack —
including on Z, which the old fixed 2-D grid could not express.

Each tile is anchored on its **walkable** centre in x/y and on its walkable **median
floor** in z, not on its bounding box: a stock map's playable floor can sit hundreds
of units inside a hull padded out by sky and terrain, and packing the hull is what
produced kilometre corridors.

The one search that survives is the cheap one that pays: after the joins and their
site pairs are chosen, each tile's real-valued offset is coordinate-descended inside
its own slot against the sum of door-to-door gaps.

```
pack: 4x4x2 lattice, 29 cells for 29 tiles; cell adjacency [2, 3, 4, 5]
placement: door-gap objective over 28 planned joins 190995 -> 117234 (38.6% shorter)
```

### 7.4 The edit itself — `Fuser.cut_portal`

This is the part that did not exist before. For each end of each join:

1. **`split_brushes`** subtracts an axis-aligned, door-sized aperture box from every
   source brush occupying it, replacing each with up to six convex remainders (itself
   intersected with each half-space outside the box). The wall stays exactly where it
   was and exactly as thick as it was, minus a doorway. This is not the old carve,
   which switched a whole brush's contents to empty and dissolved a wall panel.
2. **`clip_faces`** cuts the aperture out of the rendered surfaces too: a wall face
   square to the door axis is dropped and re-issued as the up-to-four rectangles that
   survive the cut, **keeping its own texture**, so the wall around the new doorway is
   still the level's own wall.
3. The **reveal** (the four surfaces of the cut through the wall's thickness) is
   surfaced in the wall's own texture, and a threshold slab is laid so the opening
   never gives onto a drop.
4. A **jamb/header architrave** is set into the outer face — this is what makes the
   result read as architecture rather than damage.
5. Waypoints are chained from the map's own nav node, through the opening, to the
   outer mouth, so a bot walks the door.

At full pool:

```
GEOMETRY EDIT: cut 56 doorways (46 continuing an existing passage, 10 new openings on
an exterior wall); split 444 source brushes into 825 convex remainders; re-cut 417
wall surfaces into 479 clipped surfaces; wall thickness cut through: min=8 median=64 max=256
```

### 7.5 The refusal is deleted

`MAXCORLEN` and `if math.dist(sa, sb) > MAXCORLEN: continue` are **gone from the
file**. There is no length test anywhere in the join loop, no dropped edge and no
budget violation: the openings are cut where the pack wants them and the connector is
generated to fit whatever gap is left. Small solids in the connector's way are carved;
anything too big to carve is split around the tube by the same edit the doorway uses.

|  | before (§6.1, 39 tiles) | now (29 tiles, all suitable stock maps + 1 hub) |
|---|---|---|
| cart-navigable joins | 26/40 | **28/36** |
| non-cart joins per tile | max **3** (budget 1) — VIOLATED on 4 edges | max **1** — **HELD** |
| edges dropped | 8 | **0** |
| joins refused for length | yes (`cap 6000`) | **0 — no cap exists** |
| corridor length | median 4791, max 5938 (clipped by the cap) | min 32, p25 1223, median **3295**, p75 7196, max 10491 |

The corridor distribution is the honest trade: with the cap deleted nothing is
refused, so the joins that the cap used to hide now appear in the tail. The median is
still **31% shorter** than the capped run's, because the doors are cut at the walkable
frontier facing the neighbour instead of tunnelling from a socket deep inside the map.

### 7.6 Two loader defects found and fixed on the way

Both were found by watching RSS on a shared machine, not by reading code.

* `Src.__init__` called `mkentfile.Bsp(data)` and never used the result. That helper
  grids every brush AABB with an unguarded `range()` over cell indices, so one brush
  bounded only by oblique planes — stock `catharsis` has twelve — expands to a
  ~1e15-iteration loop. The loader ate **75 GB of RSS** and never returned. The call
  is deleted; loading `catharsis` now takes 0.8 s and 0.7 GB.
* The same defect then hit the fused artifact from the other side, in the entity pass,
  taking it past **33 GB**. `Fuser.axialize()` re-emits the eighteen offending stock
  brushes (catharsis 12, xoylent 3, finalrage 3) inside six axial clamp planes at
  their tile's hull plus 4096 u of slop — far outside any playable space, so the
  brush's shape is untouched and its AABB is finite for every consumer. The fused
  world's peak is back inside the machine's budget.

### 7.7 A second engine ceiling, found by booting the thing

The entity budget of §6.7 lifted the runaway limit at *worldspawn*. Editing the maps
moved the ceiling: a fused world now carries connector and doorway waypoints on top of
29 stock waypoint sets, and stock `waypoint_loadall` spawns each saved waypoint through
`waypoint_get`, which linear-scans the waypoints already spawned and `boxesoverlap`-tests
each one — O(n²) inside one server frame. Measured, on a real boot of the 29-tile world:

```
d : boxesoverlap : statement 7
m3 : waypoint_get : statement 21
m3 : waypoint_spawn : statement 9
m3 : waypoint_loadall : statement 218
   : bot_serverframe : statement 220
Quake Error: Host_Error: server runaway loop counter hit limit of 10000000 jumps
```

So the saved waypoint count gets the same treatment the entity count got — a hard
budget (`--wpcap`, default 600) spent where it buys the most navigation. Connector and
doorway waypoints are mandatory (they are the only nodes that carry a bot between
tiles), portal pads are anchored so they can never be decimated, each tile's stationary
waypoints are farthest-point decimated to its share of what is left, and **the link
graph is contracted onto the survivors** rather than shredded — every dropped node's
links are re-attached to its nearest surviving neighbour, so reachability is preserved.
The flood-fill and the walking-distance solve are then run on the contracted graph, and
both still pass at 100 % coverage.

The number is empirical. 900 cleared `waypoint_loadall` and then blew the same ceiling
one layer up, in `navigation_markroutes -> navigation_markroutes_nearestwaypoints`,
which walks the whole `g_waypoints` list inside its own per-waypoint loop — O(n²) again,
per bot, per goal rating, and mkentfile's 175 trigger waypoints count towards n too.
600 clears both.

### 7.8 Evidence

**Full pool, 29 tiles** (`mapfuse.py 1 --bridges=1`, seed 1, one real run):

```
selection: REJECTED 1 map(s) with fewer than 2 connection sites: nexballarena(1)
selection: 29 maps kept -- 25 BRIDGE maps (>3 connection sites), 4 STUB maps (2-3 sites); 1 rejected
pack: 4x4x2 lattice, 29 cells for 29 tiles; cell adjacency [2, 3, 4, 5]
topology: 36 joins over 29 tiles (8 vertical level-to-level), 1 component(s)
placement: door-gap objective over 28 planned joins 190995 -> 117234 (38.6% shorter)
GEOMETRY EDIT: cut 56 doorways (46 continuing an existing passage, 10 new openings on an
  exterior wall); split 444 source brushes into 825 convex remainders; re-cut 417 wall
  surfaces into 479 clipped surfaces; wall thickness cut through: min=8 median=64 max=256
well-formedness: re-emitted 18 source brush(es) with no derivable axial AABB
corridor length: n=28 min=32 p25=1223 median=3295 p75=7196 max=10491 (NO length cap exists)
cart-navigability: 28/36 joins cart-navigable (door+corridor); non-cart joins per tile
  max=1 (budget 1) -> HELD; joins refused: 0; joins dropped: 0
waypoint budget 600: 3325 source + 433 connector waypoints -> 613 written
parse-back: OK
connector clearance check (exact planes, un-carved source solids): PASS (0 obstructed samples)
bot flood-fill: regions reached 29 / 29 -> PASS
navmesh: region<->region WALKING distance median=38621u diameter=86823u unreachable_pairs=0
connectivity: 1 component(s) [29]; hop-diameter=9
wrote fused.bsp (166 MB) / fused.pk3 (48 MB)
```

**Real boot** — stock `darkplaces-dedicated`, port 26071, `+g_payload 1 +bot_number 8`:

```
payload: cart 0: 485 path nodes, length 10997.6
payload: cart 1: 383 path nodes, length 8786.1
payload: cart 2: 425 path nodes, length 9666.1
[BOT]Resurrection is now playing on the RED team      ... 8 bots over 5 teams
[BOT]Dominator picked up Strength
```

Zero `runaway loop counter` errors, server alive and playing through the soak. Four
`relocate_spawnpoint: could not get out of solid` object errors remain — stock spawn
points inside newly adjacent geometry; non-fatal, and reported.

**Renders of the edits** (`joinshot.py`, 8-tile fusion, DPSOFTRAST, 480x300): 42/42
frames captured, **void audit PASS**, including a new camera pair per cut doorway —
one standing back inside the host map looking at the new opening in its own wall, one
outside the wall looking back at it. `p04_erbium_continue_in` shows a framed doorway cut
into erbium's stone wall with the connector visible beyond; `p11_geoplanetary_newcut_in`
shows a new opening with its jamb/header architrave in geoplanetary's exterior facade;
`p01_silentsiege_continue_out` shows the same doorway from the connector side.
Frames in `/private/tmp/fz8/shots/`.
## 8. LEVEL 3 (third pass) — THE PIPELINE IS COMPUTED, NOT PROBED

Both earlier passes ended the same way: geometry was emitted, a point-sampled
predicate was asked about it afterwards, and whatever the samples happened to
notice was patched.  That is why spawnpoints shipped buried in solid and the
first thing to notice was the running engine:

```
SVQC OBJECT ERROR in relocate_spawnpoint: could not get out of solid at all!
NOTE: Spawnpoint at '8202.0 -11548.0 4342.0' needs to be moved out of solid
```

`design/NAV-SPEC.md` §10 names the flow that should have existed instead.  This
pass builds it.  The governing observation is that a BSP already partitions
space: every `solid_brush_at(p)` was re-deriving, lossily and by sampling, a
structure the file already contains — and a sample can never be complete, so the
failures could only ever be patched, never removed.

### 8.1 What was DELETED

Nothing here was demoted, flagged or kept for reference.

| deleted | what it was | lines |
|---|---|---|
| `mapfuse.Src.solid_brush_at`, `.clip_brush_at`, `.bgrid`, `.cgrid` | the SOURCE map's brush predicate at SOURCE coordinates, consulted to decide things about the assembled world | 67 |
| `mapfuse.corridor_samples`, `arc_samples`, `blockage` | a 9x6 probe lattice per 48 units of corridor tube, run through the above over every tile | 47 |
| `mapfuse.ray_runs`, `free_slab`, `probe_site`, and the ray-probe `map_sites` | connection-site detection by marching rays in 8-unit steps and sampling 3-D lattices of points | 117 |
| `mapfuse.brush_volume_ok`, `Fuser.carve` | the "small enough to dissolve" rule that let a whole wall panel vanish to make room for a tube | 20 |
| `mkentfile.Bsp` | AABB-from-plane-distance brush grid; source of a 75 GB unguarded `range()` and of phantom "no floor" reports along clear corridors | 113 |
| `mkentfile.Corridor`, `seg_bad`, `standable_set` | the push-grid and the 24-unit-step link sampler built on `Bsp.inside` | 95 |
| `mkentfile.medial_smooth`, `feasible`, `refine`, `chord_clean`, `adaptive_nodes`, `seg_valid`, `polish_chain`, `validate_chain`, `lipschitz_env`, `bending_energy`, `resample_marked`, `botwalk_chain` | the emit-then-poke-then-patch repair stack for cart paths, and its "bot walk" fallback for when the repairs failed | 264 |
| `joinview.occlusion_probe` | nine rays stepped 24 units at a time through `Bsp.inside` | 30 |
| `mapgen`'s waypoint standable check via `M.Bsp` | same predicate, same class of answer | — |

`mkentfile.py` went from 1350 to ~936 lines; ~750 lines of sampling and repair
were removed in total.  After this change there is exactly ONE definition of
solidity in `payload/tools/`:

```
solid(p)  ==  negspace.NegSpace.cell_at(p) < 0
```

Two things survive under their true names and gate nothing: the void audit
(`joinshot.py`) measures that the world renders non-black, and the flood-fill
measures that the waypoint graph is one component.  Neither is cited as evidence
about geometry and no geometry decision depends on either.

### 8.2 `negspace.py` — the computed free volume

`NegSpace` is a map's free volume as an explicit set of CONVEX CELLS with exact
faces.  It is obtained by walking the BSP tree, giving every leaf the exact
half-space list of the region the tree carved for it, and then subtracting the
blocking brushes that overlap that region.

Three decisions in it are load-bearing and each was forced by a measurement:

**Free space is defined by BRUSHES, not by the leaf's PVS flag.** DarkPlaces
collides against a BIH over brushes (`Mod_CollisionBIH_TraceBrush`,
`model_brush.c:4948`), not against the BSP tree. A leaf that q3map2's
fill-outside marked opaque but that contains no brush is space a player can
stand in — and on a sealed map that is exactly the region a fusion connector
gets built in. Defining free space as "open leaf" disagreed with the engine
there.

**Which brushes to subtract is decided by an overlap query, not by the leaf's
`leafbrushes` list.** That list is not complete for opaque leaves: measured on
`warfare`, 78 of 12 275 sampled points that the resulting cells called free were
inside a solid brush, because the brush filling the leaf was not listed in it.

**Redundant half-spaces may only be pruned if the AABB that justified the
pruning is kept.** Pruning against a cell's own AABB and then not carrying that
AABB in the constraint set is unsound: on `warfare` it produced cells with two
surviving planes and 1900-unit extents, and 196 of 12 275 sampled free points
were inside a brush. The fix appends the six axial planes of the AABB, which is
the same point set with the redundant members replaced by the box implying them.

Soundness is measured, not asserted. Uniform random points over each map's world
box, comparing cell membership against an exact test over every blocking brush:

| map | free cells | points called free | of those, inside a solid brush |
|---|---|---|---|
| dance | 9 384 | 13 997 / 15 000 | **0** |
| warfare | 31 375 | 11 584 / 15 000 | **0** |
| runningman | 6 243 | (20 000-sample run) | **0** |

The error is one-sided by construction and the remaining error is in the safe
direction: free volume is LOST, never invented. Two sources, both stated —
remainder pieces thinner than 0.25 units are dropped, and a leaf whose brush
subtraction exceeds the convex-piece cap is given up whole rather than left
partly subtracted (70 of warfare's 3 455 leaves).

API:

```
cell_at(p) -> int         the containing free cell, or -1.  THE definition of solidity
covered(H, tol=1.0)       is a convex region inside the union of free cells?
fits(p, mins, maxs)       does the whole box at p lie in free space?  (exact across
                          cell boundaries -- a box spanning a doorway fits)
segment_intervals(a,b)    the EXACT parametric intervals of a segment that are free
segment_gaps(a,b)         the complement: what a segment burrows through
project(p, mins, maxs)    the nearest LEGAL placement, and its distance
                          (the activation-distance operator of NAV-SPEC §3)
floor_under(p)            the free volume's own lower boundary under p
standing_point(p)         a constructed standing placement, or None
translated(t)             exact rigid placement:  n.p <= d  ->  n.p <= d + n.t
union(parts) / edit(add, remove)   assembly, and the fusion's own geometry edits
build_portals()           the exact shared faces between cells, each with the
                          radius of its largest inscribed circle
```

Rigid placement is exact for a translation, so a per-tile complex survives the
3-D pack and the Z stacking without being recomputed. `edit()` applies the
fusion's own geometry: the corridor slabs, thresholds, jambs and headers the
generator ADDS are subtracted from the free volume, and the apertures and
corridor interiors it OPENS are added — in the same operation, so the structure
describes the world after the fusion rather than before it.

The assembled free volume is written out as `fused.negspace.npz`. It has to be:
`fused.bsp` cannot express it, because mapfuse attaches connector leaves under a
degenerate router chain and the engine only reaches them through the brush BIH.
Every downstream tool loads that file rather than deriving a second answer.

### 8.3 `navmesh.py` — Voronoi over the stock navmesh, and a constrained path placer

Per NAV-SPEC §5 the navigation definition is the STOCK waypoint graph, the one
playerbots use; this module does not introduce a second one. On top of it:

**Semantic edge classification (§4).** A waypoint link that encodes a jump-pad or
teleport trajectory is not a cart segment. Those are recognised from the saved
waypoint flags and from endpoints inside a `trigger_push` / `trigger_teleport`
volume — never from geometry. This is the classification that was missing when
"carts burrow into level geometry along very smooth waypoint following curves".

**Geometric edge validity (§2).** Whether a link's straight segment burrows
through solid is answered by `segment_gaps`: the intersection of a segment with a
convex free cell is a closed-form interval, so the parts of the link that are NOT
in free space are computed, not sampled. The deleted `seg_bad` walked the link in
24-unit steps and could not see a thinner obstruction than its own step.

**Voronoi over the navmesh (§2, §8).** Free cells are assigned to navmesh sites by
growing along PORTAL adjacency — through openings whose inscribed radius admits a
body — rather than by straight-line proximity, so the partition follows
navigability. Cells no navmesh node can reach are reported rather than hidden.

**k-center origins (§1).** `equidistant_origins` maximises the minimum pairwise
navmesh-WALKING distance between cart origins and reports the achieved spread
ratio (max/min; 1.00 is exactly equidistant) against the k-center optimum for
that navmesh.

**The path placer (§2 tangent-energy, §3 activation distance).** `PathSolver`
minimises the discrete bending energy `Σ|p_{i-1} - 2p_i + p_{i+1}|²` and, after
every gradient block, PROJECTS each free point back into the computed free volume
and settles it onto the free volume's own floor. The feasible set is
`negspace.fits(p, CART_MIN, CART_MAX)` — exact, and true across cell boundaries.
Every iterate is therefore a motion plan inside negative space, so the result
cannot burrow and there is no unstick. Two numbers are reported and both must be
zero for the constraint to have held: points off the free volume, and the maximum
activation distance to a legal placement.

One measured detail worth keeping: the 4th-difference operator's spectral radius
is 16, so the gradient of the squared second difference is bounded by 32 and any
step above 2/32 diverges. It first shipped at step/4 and the energy ran to 1e76.

### 8.4 Spawnpoints are PLACED, not tested

The old code (`mapfuse.py` ~763-779) asked `src.solid_brush_at([x, y, z+dz])` for
three `dz` straight up — the SOURCE map's brushes at SOURCE coordinates. The
spawn's real position is `o + off` in the assembled world, after the 3-D pack
(Z levels included), after the doorway cuts split the brushwork, and alongside
connector floor and wall slabs that did not exist when the question was asked.

The origin is now CONSTRUCTED: `NegSpace.standing_point` returns a point at which
the whole player box is covered by free cells and which has the free volume's own
floor beneath it, searching an expanding lattice if the mapper's own origin does
not qualify — the offline form of what `relocate_spawnpoint` does at run time
inside a live worldspawn's 10M-jump budget. "In solid" is not a state this can
produce, so there is nothing left to test afterwards and the engine never has to
run its own search.

### 8.5 Connection sites, doorways and connectors are DERIVED

**A connection site is a place where two free regions are separated by a thin
barrier**, and all three parts of that are statements about VOLUME, so all three
are answered as exact coverage of a box by the computed free cells:

* a player-sized free approach standing against the inner face;
* a player-sized free landing on the far side, for the connector to meet;
* and no already-open route between them across the door's own footprint —
  otherwise there is nothing to cut and the "door" would be a hole in mid air.

The wall's own thickness comes from `solid_runs`, which reads the solid spans
along a ray off the free-cell intervals in closed form. `probe_site` marched that
ray in 8-unit steps through `solid_brush_at` and `free_slab` sampled a 3-D
lattice of points in front of and behind the wall; both are deleted. A wall
thinner than the old step size can no longer be missed, and a probe can no longer
land on a pillar and condemn a good site.

**The doorway edit registers itself with the free volume.** `cut_portal` adds the
threshold, the two jambs and the header as solids the complex must lose, and adds
the aperture as free volume the complex gains. `build_corridor` does the same for
its floor, wall and ceiling slabs and its interior. `NegSpace.edit` applies all of
it in one pass, so after a fusion the structure describes the fused world.

**Connector clearance is by construction, then confirmed structurally.** The old
check re-ran the deleted probe lattice against the UN-CARVED source brushes and
counted "obstructed samples" — it asked about source geometry, it could only see
what a probe landed on, and it had to special-case its own carve set to avoid
reporting the holes it had just made. What replaces it subtracts the assembled
free volume from each corridor's interior and reports the residue in CUBIC UNITS.
Zero means the corridor is open; a non-zero number is real solid inside a
corridor, measured, not counted in probes.

**Doorway traversability** is the exact coverage of the swept
approach → aperture → connector-mouth volume by the assembled free cells — not a
render, and not a trace.

### 8.6 The budgets, DERIVED

The waypoint cap of 600 and the entity cap of 1800 were both tuned until a boot
stopped failing.  They are now measured, on the real dedicated server, against
the real fused megamap, with the bot count as an explicit input.

**Bots do NOT share one 10M-jump budget.**  `jumpcount` is zeroed per
`PRVM_ExecuteProgram` call (`prvm_exec.c:789`), and the entry point the failure
actually occurs in is per-client: the captured runaway stack is

```
SV_PlayerPhysics -> _SV_PlayerPhysics -> sys_phys_update -> sys_phys_ai
  -> bot_think -> havocbot_ai -> havocbot_role_generic
  -> navigation_goalrating_start -> navigation_markroutes
  -> navigation_markroutes_nearestwaypoints -> tracewalk
```

`SV_PlayerPhysics` is invoked per client (`sv_user.c:383`, from `SV_Physics`'s
`for i=1..maxclients` loop with no netconnection check, so bots are included), so
each bot gets its OWN 10,000,000 jumps.  Confirmed from a HEALTHY run rather than
a crash: `SV_PlayerPhysics` callcount is exactly 12x the `StartFrame` callcount
at both n=604 (21 948 / 1 829) and n=1100 (22 392 / 1 866) — one entry point per
bot per frame, one budget each.  Those holds ran at 30.4 and 31.0 fps, so none
was CPU-starved despite a third workload on the box; a starved hold would have
silently weakened every pass.  Independently, the expensive goal search
is serialised to ONE bot per frame by the global strategy token
(`havocbot.qc:52,103`; `bot.qc:789-810` — "prevents them from all doing waypoint
searches on the same frame").  Bot count cannot multiply the term either way.

**Ladder A — waypoint count n at fixed B=12**, 300 s hold each:

| n | links | result |
|---|---|---|
| 300 | 983 | boots, survives 300 s |
| 450 | 1297 | boots, survives 300 s |
| **604** | 1599 | boots, survives 300 s — the shipped set |
| 750 | 1877 | boots, survives 300 s |
| 900 | 2175 | boots, survives 300 s |
| 1000 | 2366 | boots, survives 300 s |
| 1100 | 2560 | boots, survives 300 s |
| 1200 | 2759 | **RUNAWAY at 25.1 s and 30.1 s** (two independent samples, same crash site), `navigation_markroutes_nearestwaypoints -> tracewalk` |
| 1600 | 3543 | **RUNAWAY at 7.0 s, before the match starts**, `waypoint_loadall -> waypoint_spawn -> waypoint_get` |

**Ladder B — bot count at n=900**, 300 s hold each: B = 2, 8, 12, 16, 24 all boot
and survive.  **Flat.**  Twelve times the bot count changes nothing.

And the control at the MARGINAL n=1200 is **non-monotonic in bot count**:

| B | 2 | 12 | 24 |
|---|---|---|---|
| n=1200 | boots, 300 s clean | **RUNAWAY** (25.1 s, and 30.1 s on repeat) | boots, 300 s clean |

More bots is not worse.  That kills the "it scales with bots" story outright: the
failure is a PER-BOT, POSITION-DEPENDENT event.  A bot that happens to stand far
from every waypoint drives the expansion at `navigation.qc:1119` through its ~66
walks inside its own private budget.  It reproduces at B=12 because the roster is
deterministic for a given `bot_number`, so the same unlucky bot lands in the same
unlucky place; B=24 draws a different roster with nobody there.  The right way to
state it is **bot-count-independent with a roster-dependent stochastic term** —
which is also why a cap must sit well below the marginal n rather than at it.

**The mechanism is two opposing terms in n.**  `navigation_markroutes` only
enters the expensive expanding search when it gets a NULL fixed source waypoint,
i.e. when the bot is not near a known waypoint.  Measured per 60 s at B=12:

| n | markroutes calls | of which nearestwaypoints |
|---|---|---|
| 604 | 26 | **67** |
| 1100 | 24 | **8** |

Adding waypoints makes each expensive call cost more (longer O(n) walks) but
makes the expensive branch fire far less often (a denser graph means the bot is
more often already near a waypoint).  That product is why the edge is a
distribution rather than a threshold, and why B=12 can fail where B=2 and B=24
pass.

**The relationship.**  `n_max(B) = n_max`, independent of B over [2, 24].  There
are two ceilings and neither scales with bots:

* **Load time, O(n^2)** — `waypoint_loadall` spawns each saved waypoint through
  `waypoint_get`, which linear-scans `g_waypoints` with `boxesoverlap`
  (`waypoints.qc:418`).  Measured `boxesoverlap` calls ~= 1.08 n^2 (1 553 976 at
  n=1200).  Fires once, ~2.5 s after the first bot connects.  Ceiling between
  1200 and 1600.
* **AI time, O(n) per roll with a very large constant** — the expanding search in
  `navigation_markroutes_nearestwaypoints` (`navigation.qc:1119`,
  `for(j=increment; !found && j<maxdistance; j+=increment)`, increment 750,
  maxdistance 50000 on ground) walks the whole list up to ~66 times with a
  `tracewalk` per candidate.  Ceiling between **1100 and 1200** at B=12.  **This
  is the binding one.**

B enters only as a GATE and a TRIAL COUNT: `waypoint_loadall` runs at all only
when `currentbots > 0` (`bot.qc:759`), and each bot rolls `markroutes` about once
per `bot_ai_strategyinterval` inside its own private budget, so more bots shorten
the expected time to an unlucky roll without moving the per-roll ceiling.

**Therefore the cap is n <= 900 for any bot count in [2, 24]**, and the shipped
604 (685 in the seed-7 build) sits at about two thirds of it.  The old 600 was
not wrong — but it was right by accident, and it was justified by a story about
bot count that the measurement disproves twice over: the bot ladder is flat, and
at the marginal n the failure is not even monotonic in B.

**Does the 29-tile pool hold 12 bots?  It BOOTS AND SURVIVES with 12 bots — it
does not PLAY with them.**  Both halves are measured and the distinction is not
pedantry.

Survives: the shipped n=604 set ran a 300 s and then a **600 s clean soak** at
B=12, 12 bots connected, no runaway; the seed-7 rebuild at n=685 held 10 min 29 s
the same way.

Does not play: `navigation_unstuck` fires **12 099 times per 60 s at n=604/B=12,
burning 71.9M statements** — about 17 unstuck calls per bot per second,
continuously (exact `prvm_profile` deltas over a bracketed live window, not an
estimate).  At n=1100 it is 11 815 / 70.2M, so it is the megamap and not an
artifact of densification.  The corroborating symptom is in every log: **0-3
combat events per 300 s** across all 22 valid budget runs, 2 in the 600 s soak,
and **9 in the seed-7 rebuild's 10 min 29 s** hold.  The bots are pathologically
stuck.

**The cause is the budget itself, and it is specific to the stress case.**  The
29-tile world has 3325 source waypoints and ships 685 — **96 % of the navmesh is
decimated** to stay under the cap.  A bot on a sparse graph is usually NOT near a
waypoint, which is exactly the condition that drives both the unstuck loop and
the expensive search branch.  At the 2-6 tile target NOTHING is decimated:

| world | source waypoints | decimated | combat events / min at B=12 |
|---|---|---|---|
| 29-tile | 3325 | **3203 (96 %)** | 0.86 |
| k=2 | 260 | **0** | 4.20 |
| k=4 | 210 | **0** | — |
| k=6 | 740 | **0** | — |

Five times the activity at k=2, on a graph the budget never had to cut.  This is
the sharpest argument yet that 29 tiles is the wrong unit: the cap that keeps the
server alive there is the same cap that starves its navigation.  It is NOT yet
established that bots play *well* at 2-6 tiles — 4.2 events/min for 12 bots is
better, not proven good, and the unstuck rate at target scale has not been
profiled.

**The entity budget is not the binding constraint.**  At n=604/B=12, with the
`.ent` taken from the installed pk3 rather than from a mismatched artifact:

| N_ent | 2400 | 2872 (shipped) | 3200 | 3400 | 3600 | 4000 |
|---|---|---|---|---|---|---|
| | ok 300 s | ok 300 s | ok 300 s | **runaway 6.0 s** | **runaway 6.0 s** | **runaway 5.0 s** |

The ceiling is between **3200 and 3400**.  The mechanism is `InitializeEntity`
(`server/world.qc`) — a linear-scan sorted-list insert called once per deferring
map entity inside the single worldspawn spawn loop, so O(N_ent^2) in ONE entry
point (1547 calls / 8.43M self statements at N_ent=3600, alongside
`il_links_flds##GETFP` 14.6M and `IL_REMOVE_RAW` 6.9M).  Unlike the waypoint
terms this ceiling genuinely IS shared — every entity spawns in one call.  The
old `Fuser.ent_budget = 1800` was about 1.8x conservative.

**Caps set from the above:** `wpcap` 600 → **900**, `Fuser.ent_budget` 1800 →
**3000**.

The margins are deliberately UNEQUAL, and the asymmetry is the decision, not an
accident of where the rungs fell:

| cap | nearest measured failure | headroom |
|---|---|---|
| wpcap 900 | 1200 | 1.33x |
| ent 3000 | 3400 | 1.13x |

The waypoint cap gets more margin because its failure carries a **stochastic,
roster-dependent term** — at n=1200 it fires at B=12 and not at B=2 or B=24, so
the edge is a distribution rather than a line and a cap must stand clear of it.
The entity failure has no such term: `InitializeEntity` is a deterministic
O(N^2) walk in a fixed spawn loop, the same every boot, and its edge is bracketed
to 6% (3200 ok / 3400 fail) rather than to 33% (900 ok / 1200 fail).  A tighter
margin on a deterministic, finely-bracketed edge is worth more than a wider one
on a stochastic, coarsely-bracketed edge.  If that judgement is wrong, ent 2900
or lower restores parity at the cost of ~270 entities of map content.

**The pass evidence near the edge is thinner than the run count suggests, and
biased optimistically.**  The right unit for these runs is EXPENSIVE ROLLS, not
seconds.  Extrapolating the measured rates, a 300 s hold buys about **335**
`nearestwaypoints` calls at n=604 but only about **40** at n=1100 — 8x fewer
expensive rolls precisely in the region nearest the edge.  The failure side is
robust by contrast: n=1200 died inside its first handful of rolls, at 25 s and
30 s.  So the 1100-1200 ceiling is **biased high**, and n=1100 must NOT be read
as a verified-safe number — it is a rung that passed on thin evidence.  This
argues for the 900 cap rather than against it: 900 passed across five bot counts
and hundreds of cumulative rolls, and sits below the uncertainty.

**Caveat on the rungs above n=604.**  The shipped waypoint set has only 604
saved waypoints, so the 750/900/1000/1100/1200/1600 rungs were SYNTHESIZED by
subdividing existing links at their midpoints — real segments the map's own
linker had already declared walkable, with the largest weakly-connected component
holding at 61-65% of nodes at every n (matching the original's 397/604), but not
human-placed waypoints.  The n <= 604 rungs are pure original coordinates.  So
the 1100-1200 AI-time ceiling is measured on a synthesized graph, and a real
1100-waypoint set could sit differently.  The 900 cap is below that uncertainty
either way, which is a further reason not to push it to 1100.

**Two measurement artifacts worth recording:** `+developer 1` itself causes a
runaway on this map, because each spawnpoint-in-solid assert dumps a full VM
statement trace inside `__spawnfunc_worldspawn`.  The identical configuration at
`+developer 0` boots.  Any budget number taken at `developer 1` is an artifact of
the logging, not of the map.

And a harness artifact that bit BOTH of us independently: a server launched from
a tool call and left to a later call gets reaped with its process group, and a
second server started before the first releases its port and userdir dies on
`bind: Address already in use` / `session lock could not be acquired` without
ever reaching a match.  Three early "soak" numbers on this work were that, not
the engine.  A soak must own a fresh userdir, a unique `-sessionid` and a
verified-free port, and be held inside one call — and the port check and the bind
must be in the SAME call, because a check-then-launch gap on a busy box is
exactly what produces `Address already in use`.  7 of 29 budget runs were marked
invalid on these grounds and excluded rather than reported.

A related trap on a shared box: this work ran alongside a third, unrelated
workload that was also binding ports in the 261xx range from a userdir inside the
same scratchpad tree.  Port collisions between concurrent agents are not
hypothetical, and a process found holding "your" port may belong to neither you
nor the peer you assume.  Attribute by userdir and by the launching tool's own
port range before blaming anyone — including yourself.

### 8.6b Cost, and what it bought

The survey — an exact convex decomposition per candidate map, then the structural
site solve on top of it — is the expensive half of a fusion, and it is per-map
independent, so it runs across processes. One BLAS thread is pinned per process:
every worker is doing small dense linear algebra, and letting Accelerate spawn a
pool inside each oversubscribes an 18-core machine by an order of magnitude.

Measured, single map, on this hardware:

| map | brushes | free cells | complex build |
|---|---|---|---|
| dance | 1 605 | 9 384 | 4.0 s |
| warfare | 5 381 | 31 375 | 26.7 s |
| catharsis | 75 762 | — | (in the parallel survey) |

The old pipeline's equivalent cost was near zero, because it was not computing
anything — it was sampling. That is the trade: the structure costs minutes per
fusion and in exchange the failure mode it was built to remove is not
representable rather than merely unobserved.

### 8.7 What this pass does NOT do — measured, not hand-waved

Two requirements arrived late and are NOT implemented here. Both are scoped with
real numbers from the shipped BSP so the next pass starts from measurement.

**VIS, lighting and the lump writer.** `mapfuse` still writes BSP lumps directly
and therefore has to synthesise the tree, the PVS and the lightmaps itself. It
does not. Measured on the 29-tile `fused.bsp` (169 MB):

| lump | value |
|---|---|
| visdata | **0 bytes** — no PVS at all |
| lightvols (lightgrid) | **0** — dynamic models get no light sample |
| lightmaps | 49 152 bytes = **one** 128x128 grey block for the whole world |
| leaf clusters | **2 distinct** (-1 solid, 0) over 67 371 leafs |
| faces | 200 946 — all of them candidates from every position |
| face types | 177 742 polygon, 13 091 **patch**, 9 746 mesh |
| sky shaders | **14** distinct, 531 faces, two surfaceflag classes (0xc34, 0x20c34) |
| texture sets | 55 over 652 shader refs |

Nothing can be culled, so every face is submitted every frame: that is one fact
with two symptoms (occlusion and draw-call latency). The single-cluster collapse
made it deliberate and it cannot be undone without real VIS to replace it.

The proposed fix — emit `.map` source and let q3map2 compute tree, VIS, lightmaps
and collision — is right in principle and is what `mapgen.py` already does for
procedural tiles. For the FUSED world it additionally requires decompiling 29
stock BSPs to brush source, and the measured cost of that is in the table above:
**13 091 patch faces** carry no brush representation at all and would be lost,
and brush-face texture alignment is not recoverable from a BSP (the lump stores
surface UVs, not the face texdefs q3map2 needs). That is a real obstacle, not a
scheduling one, and it should be decided before the work starts.

**Skybox / distant-LOD tweening.** Not implemented. What the engine actually
offers, checked in this DarkPlaces tree rather than assumed:

* **No compute shaders.** `glDispatchCompute` and `GL_COMPUTE_SHADER` do not
  appear anywhere in the source. A compute-shader design is not available here.
* Q3 shader stages DO carry per-entity blend inputs: `Q3RGBGEN_ENTITY`,
  `Q3RGBGEN_ONEMINUSENTITY`, `Q3ALPHAGEN_ENTITY`, `Q3ALPHAGEN_ONEMINUSENTITY`
  and `Q3ALPHAGEN_PORTAL` (`model_shared.h:320-342`). An entity's alpha is a
  per-frame blend parameter the renderer already honours.
* CSQC can drive it: `VM_CL_R_SetView` (`clvm_cmds.c:798`) plus per-frame entity
  alpha gives a cross-fade between two sky domes / two distant-LOD shells with
  the blend factor computed from the player's position relative to the aperture
  being traversed, and the transition TYPE (corridor / teleporter / vertical
  shaft) can select which shader pair is used.

That is the shape a real implementation should take here. It is also strictly
downstream of VIS: cross-fading on an unculled world adds fill cost to a frame
that is already submitting every face in the map.


### 8.8 Evidence

Durable build output (nothing in a temp dir this time):

```
/Users/mdot/dox/xonotic/fusebuild/
  full/data/maps/    29-tile megamap: fused.bsp (169 MB), fused.pk3, fused.ent,
                     fused.waypoints(.cache), fused.joins.json, fused.metrics.json,
                     fused.negspace.npz (19.6 MB, 447 174 convex free cells)
  full/build.log     the whole run
  full/fusecheck2.log  the verifier's output on those artifacts
  full2/             the rebuild at the DERIVED caps (wpcap 900, ent 2600)
  t3/                a 4-tile fusion used to validate the pipeline end to end
  prev/              the previous pipeline's 29-tile artifacts, kept for comparison
  budget/            every budget run's server.log and machine-readable results
  boot12/boot12.log  the 12-bot dedicated-server boot and soak
```

The 29-tile fusion (seed 7): 30 candidates surveyed in 139 s across 8 processes,
29 kept (23 bridge, 6 stub, `nexballarena` rejected with 0 sites), 39 joins (32
corridors, 7 vertical teleporters), 64 doorways cut, 635 source brushes split
into 1072 convex remainders, fuse wall time 321 s.

`fusecheck.py` against the shipped artifacts:

```
free volume: 447174 convex cells, world [-20505,-19548,-6432]..[16731,19589,6015]
spawnpoints: 243 shipped | origin inside solid: 0 | player box does not fit: 0
                         | no floor beneath within 512u: 0
doorways:    61/64 admit a player-sized body end to end
```

Real dedicated server, port 26150, `+bot_number 12 +skill 5 +g_payload 1
+developer 0`, 29-tile megamap: **the match starts, all 12 bots connect, join
teams, and the 10,000,000-jump runaway does not fire** — but only **9 real combat
events** occur in 10 min 29 s, because the bots are stuck (see §8.6).  An earlier
draft of this section said 152; that figure was wrong, and wrong in a way worth
recording: the grep counted `"new portal was clipped away"` as a frag, and 145 of
the 153 matches were that warning.  A pattern containing a bare `was ` is not a
combat filter.  Two of the 243 spawnpoints still tripped the engine's own
`relocate_spawnpoint` — root-caused to generator-added brushes that the free
volume had not been told about, which is why `Fuser.add_brush` now registers
every solid it creates with the complex at the single place solids come into
existence.


### 8.9 The target is 2-6 tiles, and the slots are the deliverable

A single 29-tile world is a training distribution of cardinality one.  What the
fusion is for is a DISTRIBUTION of worlds, so the unit is a 2-6 tile assembly and
the thing worth counting is how many distinct assemblies the slot structure
admits.  Measured on the surveyed pool (structural detector, `map_sites`):

| | |
|---|---|
| stock maps surveyed | 29 |
| usable (>= 2 slots) | **28** (`nexballarena` has 0 and is refused) |
| **BRIDGE, > 3 slots — can sit MID-CHAIN** | **22** |
| STUB, 2-3 slots — endpoint or pass-through only | 6 |
| slots per map | min 2, median 5, max 12; **163 in total** |

That is the combinatorial width: 22 of 28 maps can carry a chain through
themselves rather than terminate it.

| k | subsets of the usable pool |
|---|---|
| 2 | 378 |
| 3 | 3 276 |
| 4 | 20 475 |
| 5 | 98 280 |
| 6 | 376 740 |
| **2-6 total** | **499 149** |

Times chain ordering (reversal-symmetric, k!/2) that is **141 779 106 distinct
linear worlds**, and it is a lower bound: the packer also builds loop edges and
vertical level-to-level joins, and each is seeded.

The structural site detector did not cost coverage relative to the deleted
ray-probe one: 163 slots over 29 maps versus 158, same median (5), same maximum
(12), same single refusal (`nexballarena`).  It moved two maps from BRIDGE to
STUB and refused `nexballarena` on 0 sites rather than 1 — and every site it
reports is now a place where two free regions are provably separated by a thin
barrier with a player-sized approach and landing, rather than a place where a
ray happened to strike something.

The 29-tile artifact in §8.8 should therefore be read as an extreme stress case,
not as the goal, and the budgets in §8.6 are the ceilings that scale case hits.
At 2-6 tiles they are far from binding: the 4-tile `t3` build carries 399 saved
waypoints (cap 900), 12 011 faces and 125 shader refs against the 29-tile world's
685 / 200 946 / 652, and fuses in 50 s.

Measured across scale, from the shipped BSPs:

| world | size | faces | shader refs | texture sets | **sky shaders** | leafs |
|---|---|---|---|---|---|---|
| k=2 (implosion+warfare) | 10.8 MB | 17 748 | 96 | 11 | **2** | 5 599 |
| k=4 (dance+trident+runningman+bridge) | 7.6 MB | 12 011 | 125 | 18 | **2** | 5 041 |
| k=29 (full pool) | 161 MB | 201 167 | 652 | 55 | **14** | 67 452 |

Every quantity that made the maximal build pathological is one to two orders of
magnitude smaller at the real target: an order of magnitude fewer faces for VIS
to cull, a texture-set count a cache can hold, and two skies instead of fourteen.
The co-visible-sky problem and the multi-sun problem are both problems of the
configuration nobody asked for; at 2-6 tiles a region cross-fade has two or three
looks to blend, not fourteen.

### 8.10 The target scale, verified end to end

Three sampled worlds at k = 2, 4, 6, each fused and then checked by `fusecheck.py`
against its own shipped artifacts:

| k | fuse | peak RSS | free cells | spawns shipped / **in solid** | doorways | connector residue | cart nodes / **illegal** |
|---|---|---|---|---|---|---|---|
| 2 | 55.3 s | — | 42 116 | 19 / **0** | 2/2 | **0 u^3** (0/1) | 388 / **0** |
| 4 | 41.5 s | 0.62 GB | 32 572 | 27 / **0** | 3/4 | **0 u^3** (0/2) | 215 / **0** |
| 6 | 79.1 s | 2.41 GB | 88 282 | 54 / **0** | 10/10 | **0 u^3** (0/5) | 243 / **0** |

**Zero in-solid spawnpoints, zero uncovered connector interior and zero illegal
cart placements at every sampled size**, and the path solver reports CONSTRAINT
HELD (0 points off the free volume, max activation distance 0.00 u) on all three.
The 29-tile world's residue — 13 of 32 corridors with uncovered interior, 2
spawns the engine's own `relocate_spawnpoint` still caught — does not appear at
the scale the fusion is actually for.  Remaining honest residuals at target
scale: one doorway on `solarium` at k=4 with 13.2 % of its swept approach
uncovered, and 1 / 2 / 6 cart-path segments that leave the free volume between
nodes (the nodes themselves are all legal placements).

Real dedicated server, k=2, fresh userdir, `-sessionid`, port 26160,
`+bot_number 12 +skill 5 +g_payload 1 +developer 0`, held 200 s in-process:

```
ALIVE at 200s, rss=2.0 GB      port bind ok, no session-lock error
bots_playing=12
runaway=0   OBJECT ERROR=0   could-not-get-out-of-solid=0
combat events=14 in 200 s  (4.2/min, vs 0.86/min on the 29-tile world)
```

Read that last line with §8.6's stuck-bot finding: k=2 decimates none of its
navmesh where the 29-tile world decimates 96 % of it, and its bots are about five
times as active.  That is consistent with decimation being the cause and with the
target scale not suffering it — but it is a symptom count, not a diagnosis.  The
`navigation_unstuck` rate has NOT been profiled at 2-6 tiles, and until it is,
"the bots are less stuck here" is an inference from combat frequency, not a
measurement.  It is the first thing the next pass should measure.

Two earlier soak attempts died early and neither was the map: the first was a
`nohup`'d server torn down with its launching shell, and the second hit
`bind: Address already in use` plus `session lock could not be acquired` because
the first still held the port and the userdir.  A soak run must own a fresh
userdir, a unique `-sessionid` and a verified-free port, and must be held inside
one call — otherwise the number measures the harness, not the world.
