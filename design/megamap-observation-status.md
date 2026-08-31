# Megamap + observation map-reduce — status audit (2026-08-30)

Scope: the two AGENDA discovery obligations opened by R18 — G1/G2/G3 (map fusion,
megamap use, long-distance traversal, commitment cost) and E3/E4/E5/E6 (the
observation map-reduce parametric featurization), plus E1/E2 (the observation
buffer).

Method under the provenance law (`SPECIFICATION.md`, final section): every claim
below is a block quote of code, of a real artifact, or of an arithmetic done on a
real artifact. No re-simulation was used; `cartsim` and everything derived from it
is inadmissible and is not cited. No unit test was run. No live process was
touched. Two artifact classes are used:

- files on disk under `/private/tmp` (the mapfuse outputs and the engine logs), and
- files recovered from git history with `git show 7ca8afb^:<path>` — the `runs/`
  telemetry was deleted from the worktree by commit `7ca8afb` but is intact in the
  parent tree. (`runs/` in the *current* worktree is untracked: `git status
  --porcelain` reports `?? xonotic/solver/strat/runs/`.)

The controlling spec text is `design/payload-spec.md` §2.2 (recovered the same
way), quoted in §5 below.

---

## 1. Map fusion exists and runs — **[x] full**

`tools/mapfuse.py` is 47 KB of live code and its outputs are on disk. The most
recent seed, `fuse_v7`, produced a complete artifact set:

```
$ ls -la /private/tmp/fuse_v7/data/maps/
-rw-r--r--@  1 mdot  wheel  8932396 Aug 29 13:52 fused.bsp
-rw-r--r--@  1 mdot  wheel    43657 Aug 29 13:52 fused.ent
-rw-r--r--@  1 mdot  wheel   139290 Aug 29 13:53 fused.floorplan.svg
-rw-r--r--@  1 mdot  wheel     1198 Aug 29 13:52 fused.joins.json
-rw-r--r--@  1 mdot  wheel      129 Aug 29 13:52 fused.mapinfo
-rw-r--r--@  1 mdot  wheel  2419777 Aug 29 13:52 fused.pk3
-rw-r--r--@  1 mdot  wheel    10682 Aug 29 13:52 fused.waypoints
-rw-r--r--@  1 mdot  wheel    87831 Aug 29 13:52 fused.waypoints.cache
```

`fused.joins.json` is a real three-map fusion with real join geometry:

```json
"maps": [
 {"name":"fuse",      "offset":[-1856,-1344,-384], "mins":[-4288,-3328,-960], "maxs":[-448,-448,960]},
 {"name":"warfare",   "offset":[ 2144,-1536,-416], "mins":[  984,-3272,-488], "maxs":[3752,-504,488]},
 {"name":"runningman","offset":[-2480, 1856,-544], "mins":[-3792,  576,-1120],"maxs":[-944,3200,1120]}],
"joins": [
 {"a":0,"b":2,"kind":"corridor",  "length":2151.6},
 {"a":0,"b":1,"kind":"teleporter","length":3899.5},
 {"a":1,"b":2,"kind":"teleporter","length":4062.3}]
```

The map loaded and ran under the real engine — from the same seed's smoke log:

```
$ tail /private/tmp/fuse_v7/smoke.log
Server using port 26013
Server listening on address 0.0.0.0:26013
Loaded maps/fused.ent
NOTE: this map needs FIXING. Spawnpoint at '-1568.0 -1064.0 -800.0' needs to be moved out of solid ...
```

`joinview.py` still runs against it today (re-run for this audit, read-only apart
from rewriting the SVG it owns):

```
$ python3 xonotic/payload/tools/joinview.py /private/tmp/fuse_v7/data/maps
wrote ./fused.floorplan.svg (3 maps, 256 nav nodes, 3 joins, 132 lights, 859 clip brushes)
```

A second, larger fusion (`fuse+catharsis+glowplant`) is also on disk and is the one
actually deployed (§2):

```
$ cat /private/tmp/fuse_fixed/fused.mapinfo
title Fused fuse+catharsis+glowplant
description procedurally fused mega-map
author mapfuse
gametype dm
gametype tdm
gametype plc
```

```
-rw-r--r--@ 1 mdot wheel  45308184 Aug 29 12:38 /private/tmp/fuse_fixed/fused.bsp
-rw-r--r--@ 1 mdot wheel  16417377 Aug 29 12:38 /private/tmp/fuse_fixed/fused.pk3
```

Next action: none required for this subcomponent. (Note only that the deployed
`fuse_fixed` build predates the `fused.joins.json` emitter and has no joins file;
the diagnosable seed and the deployed seed are different builds.)

---

## 2. Megamaps are actually USED by the running system — **[x] full**

The fused pk3 is mounted in the engine basedir, so it is on every server's search
path:

```
$ ls -la ~/dox/xonotic/Xonotic/data/ | grep fused
-rw-r--r--@ 1 mdot staff 16417377 Aug 29 12:38 zzzz-fused.pk3
```

The Game-2 cartserver run cited by AGENDA R14 loaded it. Recovered engine log:

```
$ git show 7ca8afb^:xonotic/solver/strat/runs/cartserver_engine.log | sed -n '103,107p'
Loaded maps/fused.ent
payload: teams mask 31, carts 0
payload: cart 0: 30 path nodes, length 10920.210938
payload: cart 1: 22 path nodes, length 8269.576172
payload: cart 2: 16 path nodes, length 5786.716797
```

The accompanying telemetry is 228 lines, every one tagged with the live server
environment:

```
$ git show 7ca8afb^:xonotic/solver/strat/runs/game2_train.jsonl | wc -l
     228
$ ... | python3 -c "...; print(set(x['environment'] for x in L))"
{'game2_server'}
```

A later, larger real run on the same fused world is on disk uncollected
(`/private/tmp/xonotic-server.log`, 14 MB, Aug 30 15:53) — 244 match starts
alternating `fused` and `runningmanctf`, with 50 bots across 5 teams and carts
actually banking progress:

```
Loaded maps/fused.ent
payload: mesh handle 0 peer 1
payload: teams mask 31, carts 0
payload: cart 0: 30 path nodes, length 10920.210938
...
payload: bank cart 0 team 5 point 10 s 3480.611816
```

**Caveat (this is the [~] half of G2 that survives):** the *curriculum* driver
cannot reach the fused map. It defaults to a stock map —

```python
# xonotic/solver/strat/curriculum.py:94
cfg["map"] = str(cfg.get("map", "runningmanctf"))
# xonotic/solver/strat/curriculum.py:529
ap.add_argument("--maps", default="runningmanctf")
```

— and its asset resolver only looks at loose `data/maps/` files and pk3s whose name
matches `*maps*`, which `zzzz-fused.pk3` does not:

```python
# xonotic/solver/strat/curriculum.py:228-236
loose = os.path.join(self.basedir, "data", "maps", mapname + suffix)
if os.path.exists(loose):
    return ("file", loose)
archives = sorted(glob.glob(os.path.join(self.basedir, "data", "*maps*.pk3")), reverse=True)
```

Next action: pass `--maps fused` through the curriculum and widen `locate_asset`'s
glob to `*.pk3` (or add an explicit `--map-pk3` mount), so the curriculum trains on
the same world Game-2 ran on.

---

## 3. Long-distance navigation is real — **[~] partial**

Real, but asymmetrically: **spawns are spread across all source maps while every
cart track is confined to one region.** Measured on the deployed entity file
(`/private/tmp/fused.ent`, byte-identical in size to `/private/tmp/fuse_fixed/
fused.ent`, and the one matching the engine log's 30/22/16 cart node counts), by
union-find clustering the 960 entity origins with a 700-unit XY link threshold —
which recovers exactly three disjoint regions:

```
n entities with origin 960   clusters 3
  size=611  x -4556..-1476  y  3973..6589   plc_nodes=0   spawns=17
  size=282  x  1523..4365   y -9326..-1347  plc_nodes=68  spawns=12
  size=67   x -4320..-1688  y -6293..-4253  plc_nodes=0   spawns=12
```

All 68 cart-path nodes sit in one region. 29 of 41 spawns sit in the other two.
Straight-line distance from each spawn to the nearest cart-track node:

```
plc nodes 73  bbox x 1523..4365 y -8728..-1347
spawns: 41
straight-line spawn->nearest cart-track node: min 156  median 5195  max 9857
spawns further than 4000 units: 26 / 41
```

So a bot that spawns in the `fuse` or `glowplant` region must cross a join and
travel a median 5195 (max 9857) units before it can touch *any* cart — that travel
cost is real. What is **not** real is inter-region *cart choice*: because all three
carts live in the same region, committing to cart A instead of cart B costs
in-region walking only, not a join traversal. The strategic "commitment to a cart"
lever the megamap was supposed to create is only half-created.

Join traversability is measured, not assumed (`joinview.py` on the diagnosable
seed):

```
fuse<->runningman corridor   [redundant] len=2152: contortion min=0.78 mean=0.93 max=1.03 (n=6) | clip-blocked=no | passage-occlusion 0/6 rays
fuse<->warfare  teleporter   [redundant] len=3900: contortion min=1.00 mean=1.29 max=1.53 (n=6) | endpoints-clear=YES (near=ok far=ok)
warfare<->runningman teleporter [redundant] len=4062: contortion min=1.00 mean=1.08 max=1.12 (n=6) | endpoints-clear=YES (near=ok far=ok)
```

and the flood-fill obligation is documented in `xonotic/payload/README.md:369-371`:

> A region flood-fill over (cache walk-links + modeled jumps) asserts all source
> maps land in one bot-reachable component and reports per-join traversability.

Next action: place cart *goals* in different source regions — `mkentfile.py` on a
fused BSP should be told to seed one track per region (or one track whose goal is
across a join), so that choosing between carts costs a join traversal instead of
in-region walking.

---

## 4. Commitment cost is represented to the strategy layer — **[~] partial**

The plumbing is complete end to end. Schema:

```python
# xonotic/payload/tools/strategy_io_schema.py
SC = dict(TARGET=0, GAIN=1, LANE=2, HUNT=3, EXPLORE=4, COMMIT=5, SPAWN=6, LEAD=7)
```

Scatter:

```c
// xonotic/payload/qcsrc/.../sv_payload_strategy_io.qc:232-233
mesh_scatter(plc_str_h_sc, PLC_SC_COMMIT,  plc_str_commit,  1, nclients);
mesh_scatter(plc_str_h_sc, PLC_SC_SPAWN,   plc_str_spawn,   1, nclients);
```

Enactment against the stock navigation clock:

```c
// xonotic/payload/qcsrc/.../sv_payload_strategy_io.qc:276
if (this.plc_str_commit > 0)
    this.bot_strategytime = max(this.bot_strategytime, time + this.plc_str_commit);
```

The responder fills both columns:

```python
# xonotic/solver/strat/strat_responder.py:377-380
local_response = decode_allocations(
    batch, actions, intensity=intensities,
    commitments=np.clip(intensities, 0.25, 3.0),
    spawn_delays=np.clip(intensities, 0.0, 3.0),
```

**But the columns are only written when the sampled instrument IS the commitment
instrument** — they are an *action*, not a per-assignment cost:

```python
# xonotic/solver/strat/instruments.py:255-260
elif inst.kind == InstrumentKind.SPAWN_TIMING:
    out[p, SC["TARGET"]] = encode_target("cell", max(0, actor.cell))
    out[p, SC["SPAWN"]] = max(0.0, 1.0 if np.isnan(spawn[p]) else spawn[p])
elif inst.kind == InstrumentKind.TRAVEL_COMMITMENT:
    out[p, SC["TARGET"]] = encode_target("cell", max(0, actor.cell))
    out[p, SC["COMMIT"]] = max(0.0, 1.0 if np.isnan(commit[p]) else commit[p])
```

Consequently the real Game-2 run almost never exercises it. Over all 228 telemetry
lines, 3150 per-player assignments:

```
assignments 3150  kinds {'explore_cell': 1346, 'push_cart': 1224, 'contest_post': 400,
                         'hunt_rival': 156, 'spawn_timing': 20, 'suppress_cart': 3,
                         'travel_commitment': 1}
commit>0 1        spawn>0 20
```

The single non-zero commitment row in the whole run:

```json
{"row": 11, "edict": 12, "team": 3, "controller": "bot", "behavior": "policy",
 "action": 20, "kind": "travel_commitment", "subject": -1, "target": 196870.0,
 "gain": 0.0, "lane": 0.0, "commit": 0.9605, "spawn": 0.0,
 "target_logp": -2.397955, "behavior_logp": -2.397955}
```

1 of 3150 (0.03%). The QC guard `if (this.plc_str_commit > 0)` is therefore a
no-op on 3149 of 3150 assignments: the travel commitment a bot incurs by choosing a
distant cart is not priced into that choice — it is a separate, almost-never-chosen
action.

Next action: emit `SC["COMMIT"]` on *every* assignment as a function of the chosen
target's travel distance (the `push_cart`/`contest_post`/`hunt_rival` branches of
`decode_allocations`), rather than only on the `TRAVEL_COMMITMENT` branch, so
commitment becomes the cost of an allocation instead of a rival to it.

---

## 5. The observation map-reduce parametric featurization — **[~] partial, and this
is the one that was silently forgotten**

The spec, `design/payload-spec.md` §2.2 (recovered from HEAD), stages 2–5:

> 2. **V-cell segmentation `[FIRM]`.** Partition the map into Voronoi cells over
>    item/waypoint nodes; fuse contiguous **navigable** paths until the
>    distance-decay context mask (stage 4) bounds each bot's receptive field,
>    two-sided, to no less than ~5% and no more than ~15% of map area.
> 3. **Temporal contraction `[FIRM]`.** `f_c^eff = rho(dt)*f_c^obs +
>    (1-rho(dt))*f_c^prior`, `rho(dt) = exp(-dt/T)`.
> 4. **Spatial mask `[FIRM]`.** ... bounded-support kernel `g(dist_graph(c(b), c))`.
> 5. **Egocentric integration `[FIRM]`.**
>    `beta_b = sum_c  g(dist_graph(c(b), c)) * Phi * f_c^eff        (Phi low-rank)`
>
> **The belief integration is the system's ONLY spatial mixing operator `[FIRM]`.**

### (a) Implemented? Only in `live_belief.py`. `featurize.py` has been gutted.

`xonotic/solver/strat/featurize.py` is now 27 lines, holding stage 4 alone:

```python
# featurize.py:1-27 (entire file)
SLOT_DIM = 7

class VCellMap:
    def __init__(self, centroids, areas, map_area, graph_dist, support_radius, node_positions, node_cell, band=(0.0, 1.0)):
        ...
    def spatial_mask(self, cell):
        distance = self.graph_dist[cell]
        radius = self.support_radius
        weights = np.exp(-4.0 * np.square(distance / radius)) if radius > 0 else (distance == 0).astype(np.float64)
        return np.where(np.isfinite(distance) & (distance <= radius + 1e-9), weights, 0.0)

__all__ = ["SLOT_DIM", "VCellMap"]
```

At HEAD the same file is 705 lines and contains the whole pipeline:

```
$ git show HEAD:xonotic/solver/strat/featurize.py | wc -l
     705
216:def segment_vcells(
402:def build_cell_slots(rows, vcmap, now)
473:def temporal_contraction(f_obs, obs_time, now, T, ...)
515:def _resolve_phi(Phi, F)
525:def egocentric_integration(vcmap, cell_idx, f_eff, Phi=None)
544:def belief(rows, vcmap, bot_position_or_cell, Phi=None, ...)
562:def beliefs_for_bots(rows, vcmap, bot_positions_or_cells, Phi=None, ...)
622:def assemble_features(x_b, beta_b, cartstate, ...)
189:    def receptive_fraction(self, cell_idx) -> float
```

HEAD's stage-5 map-reduce, verbatim:

```python
def egocentric_integration(vcmap: VCellMap, cell_idx: int, f_eff, Phi=None) -> np.ndarray:
    f_eff = np.asarray(f_eff, dtype=np.float64)
    Phi = _resolve_phi(Phi, f_eff.shape[1])
    g = vcmap.spatial_mask(cell_idx)     # (C,)
    projected = f_eff @ Phi.T            # (C, rank)
    return g @ projected                 # (rank,)
```

### (b) Called on the real server path? `featurize`'s pipeline: **no**. Exact grep:

```
$ grep -rn -e egocentric -e beliefs_for_bots -e temporal_contraction -e segment_vcells \
      xonotic/solver xonotic/payload | grep -v __pycache__
xonotic/payload/tools/joinshot.py:2:"""joinshot.py -- headless egocentric client renders at map-fusion join edges.
```

Zero code references. `featurize` has exactly one importer, and it takes only the
two surviving symbols:

```python
# xonotic/solver/strat/live_belief.py:9
from .featurize import SLOT_DIM, VCellMap
```

`live_belief.py` re-inlines every other stage. Stage 2 (kNN + all-pairs, replacing
`segment_vcells`):

```python
# live_belief.py:175-189
if n > 1:
    d2 = np.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1)
    for i in range(n):
        for j in np.argsort(d2[i], kind="stable")[1:min(n, 3)]:
            adjacency[i].add(int(j))
...
for mid in range(n):
    distance = np.minimum(distance, distance[:, mid, None] + distance[None, mid, :])
vcmap = VCellMap(positions, np.ones(n), float(n), distance, 2.0,
                 positions, np.arange(n), band=(0.0, 1.0))
```

Note `areas=np.ones(n)`, `map_area=float(n)`, `support_radius=2.0` hardcoded, and
`receptive_fraction` gone with the rest of HEAD's file: **the spec's 5–15%
receptive-field bound (stage 2) is not computed anywhere in the worktree.**

Stage 3 (replacing `temporal_contraction`), with `T` demoted to a fixed attribute:

```python
# live_belief.py:232-236
prior = np.asarray((0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)
observed = np.isfinite(times)
age = np.maximum(0.0, float(now) - np.where(observed, times, float(now)))
rho = np.where(observed, np.exp(-age / self.decay), 0.0)
return rho * features + (1.0 - rho) * prior[None]
```

Φ (the "parametric contraction") is a **constant integer literal**, not a parameter:

```python
# live_belief.py:238-250
@staticmethod
def _project(features):
    phi = np.asarray([
        (1, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 1),
        (0, 0, 1, 0, 0, 0, 0),
        (1, -1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 1, 0),
    ], dtype=np.float64)
    return features @ phi.T
```

Stage 5, the map-reduce itself, re-inlined and **not semantically identical** to the
spec function it replaced (it adds a `weights / total` normalization HEAD's
`egocentric_integration` does not have):

```python
# live_belief.py:263-273
for p, (_, team, cell, _) in enumerate(players):
    if team not in team_projection:
        team_projection[team] = self._project(self._team_features(team, ids, index, self.now))
    weights = vcmap.spatial_mask(index[cell])
    total = float(weights.sum())
    if total <= 0:
        weights[index[cell]] = 1.0
        total = 1.0
    out[p] = ((weights / total) @ team_projection[team]).astype(np.float32)
```

### (c) Does it feed the policy's query? Yes — via `live_belief`, not `featurize`.

The chain, verbatim:

```c
// sv_payload_strategy_io.qc:281-287
void payload_strategy_step(entity leadcart)
{
	payload_strategy_attach();
	float nclients = maxclients;
	FOREACH_CLIENT(IS_PLAYER(it), { payload_perceive(it); });
	payload_strategy_gather(nclients);
	payload_strategy_scatter(nclients);
}
```
```python
# strat_responder.py:14,165,254-265
from solver.strat.live_belief import LiveBelief
live_belief = LiveBelief()
belief_reset = live_belief.sync(episode_context, max(int(ch["tick"]), int(oh["tick"])))
    deposited = live_belief.ingest(event_rows, EVT)
beta, belief_diag = live_belief.beliefs(rows, OBS)
```
```python
# estimator.py:135-141
beta = mx.stop_gradient(mx.array(np.asarray(state.beta, dtype=np.float32)))
projected = est.qkv.query(x, beta)
# qkv.py:42-43
def query(self, x, beta):
    return mx.concatenate((mx.array(x), mx.array(beta)), axis=1) @ self.W_q.T
```

So: the map-reduce **runs** and **reaches** `W_q[x ; beta]`. What was forgotten is
that it runs as a hand-inlined copy with a hardcoded Φ, a hardcoded T, a hardcoded
support radius, no receptive-field bound, and different normalization from the
`[FIRM]` formula — while the file that implements the spec sits deleted and
uncalled. AGENDA E6 already reads `[ ]` for exactly this; the audit confirms it and
adds that E3 (5–15% receptive field) is unmeasured, not merely partial.

Next action: delete the inlined copy in `live_belief.py:162-273` and call
`featurize.segment_vcells` / `temporal_contraction` / `egocentric_integration`
(restored via `git checkout HEAD -- xonotic/solver/strat/featurize.py`), passing Φ,
T and the support radius in as parameters — then assert `receptive_fraction` lands
in [0.05, 0.15] against a real fused map's waypoint graph.

---

## 6. Observation buffer — **[~] partial (real, but very sparse)**

The gate is implemented in QuakeC and all three conditions are present and
necessary, matching §2.2 stage 1:

```c
// sv_payload_strategy_io.qc:82-104
void payload_perceive(entity bot)
{
	if (IS_DEAD(bot)) return;
	vector eye = bot.origin + bot.view_ofs;
	float bcell = payload_str_cell_of(bot.origin);
	...
	FOREACH_CLIENT(IS_PLAYER(it) && it != bot && !IS_DEAD(it) && DIFF_TEAM(it, bot),
	{
		if (!payload_str_within2(bot.origin, it.origin)) continue;   // 2-V-cell cap
		if (!checkpvs(eye, it))                          continue;   // frustum/PVS
		traceline(eye, it.origin, MOVE_NOMONSTERS, bot);             // LOS
		if (trace_fraction < 1)                          continue;
		payload_str_deposit(payload_str_cell_of(it.origin), 3, myteam,
			etof(it), GetResource(it, RES_HEALTH));
	});
```
```c
// sv_payload_strategy_io.qc:48-53
float payload_str_within2(vector a, vector b)
{
	float dx = fabs(floor(a.x / 256) - floor(b.x / 256));
	float dy = fabs(floor(a.y / 256) - floor(b.y / 256));
	return (max(dx, dy) <= 2);
}
```

It really produces events on the real server. From the recovered Game-2 telemetry
(`game2_train.jsonl`, 228 ticks over 18.4 s, k=5 teams, l=12 players):

```json
{"reset": true,  "event_tick": 88,  "deposited": 1, "cells": 12, "edges": 1,  "events": 1,  "accepted": 1,  "duplicates": 0,  "invalid": 0, "teams": 1, "mean_norm": 1.233395}
{"reset": false, "event_tick": 206, "deposited": 0, "cells": 71, "edges": 82, "events": 11, "accepted": 11, "duplicates": 57, "invalid": 0, "teams": 5, "mean_norm": 1.232762}
{"reset": false, "event_tick": 207, "deposited": 0, "cells": 80, "edges": 103,"events": 11, "accepted": 11, "duplicates": 58, "invalid": 0, "teams": 5, "mean_norm": 1.231814}
```

Aggregated over the run:

```
sum deposited 32   sum accepted 1096   sum duplicates 6430   invalid 0
max cells 80       max edges 119       ticks with deposited>0: 32 / 228
```

The V-cell graph really grows from observation (12 cells / 1 edge at reset to 80
cells / 103 edges by tick 207) and all five teams end up with buffers. But only
**32 events were deposited in 18.4 seconds across 12 bots** — roughly one new
observation every 0.6 s for the whole server; 6430 of the reads were duplicates.
That is a live pipe, not a saturated one.

The events do reach the featurization — `ingest` → `buffer.observe` →
`_team_features` → `beliefs()` → `beta` (chain quoted in §5c). One defect: the
Python mirror of the gate is dead code, because its only caller passes the three
gate arguments as literals:

```python
# buffers.py:49-51
def observe(self, observation):
    if not observation.in_frustum or not observation.los_clear or observation.vcell_dist > 2:
        return None
```
```python
# live_belief.py:92-95
event = self.buffer.observe(Observation(
    int(round(team)), -1, float(stamp), int(round(cell)), kind,
    int(round(subject)), True, True, 0.0, payload,
))
```

`in_frustum=True, los_clear=True, vcell_dist=0.0` — the Python gate can never
reject. (Defensible, since QuakeC is the authority, but it means `buffers.py:50` is
never exercised in production.)

Next action: instrument the QC side with a per-round deposit counter in the engine
log and re-run one fused match, to establish whether 32 deposits / 18.4 s is the
gate working as designed (genuine occlusion on a megamap) or the ring buffer /
dedupe latch at `sv_payload_strategy_io.qc:60-66` swallowing observations.

---

## Summary

| # | Subcomponent | State | Strongest evidence | Single next action |
|---|---|---|---|---|
| 1 | Map fusion exists and runs | **[x]** | `/private/tmp/fuse_v7/data/maps/` full artifact set (8.9 MB `fused.bsp`, 2.4 MB `fused.pk3`, 3-map `fused.joins.json`); `joinview.py` re-run today: "3 maps, 256 nav nodes, 3 joins" | — |
| 2 | Megamaps USED by the running system | **[x]** | `cartserver_engine.log:103` `Loaded maps/fused.ent`; `zzzz-fused.pk3` mounted in basedir; 228 `game2_server` telemetry lines | Teach `curriculum.py` to select `fused` (`--maps`, `*.pk3` glob) |
| 3 | Long-distance navigation is real | **[~]** | 3 disjoint regions; **all 68 cart nodes in ONE region**, 29/41 spawns in the other two; spawn→cart median 5195, max 9857 units | Seed one cart track per region so cart *choice* costs a join |
| 4 | Commitment cost reaches the policy | **[~]** | Full schema+QC path exists (`SC["COMMIT"]` → `bot_strategytime`), but **1 of 3150** real assignments had `commit>0` | Emit `COMMIT` on every allocation as a function of target distance, not as a rival instrument |
| 5 | Observation map-reduce featurization | **[~]** | Runs and reaches `W_q[x;beta]` — but via `live_belief.py:162-273`'s inlined copy with constant Φ; `featurize.py` gutted 705→27 lines, `egocentric_integration` referenced **nowhere** | Restore `featurize.py` from HEAD, call it, and assert the 5–15% receptive fraction |
| 6 | Observation buffer | **[~]** | Real: cells 12→80, edges 1→103, teams 1→5, `invalid 0`; but only **32 deposits / 18.4 s / 12 bots**, 6430 duplicates | Add a QC-side deposit counter and re-run one fused match to explain the sparsity |

**Most at risk of having been silently forgotten: #5.** It is the only subcomponent
where the artifact that names the spec (`featurize.py`, 705 lines, containing
`egocentric_integration`, `temporal_contraction`, `segment_vcells`,
`receptive_fraction`) has been deleted from the working tree while an unreviewed
hand-inlined substitute with hardcoded constants and different normalization runs
in its place on the live path — and the substitution is invisible from the outside
because `beta` still flows into `qkv.query`. Nothing fails; the `[FIRM]` formula
just quietly is not the one being computed.
