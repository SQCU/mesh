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
