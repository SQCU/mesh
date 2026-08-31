# MAPGEN-ROADMAP — the generator, and the decoupling it demonstrates

## Why this exists

A control experiment: an agent with **no context whatsoever** was handed one query —

> i want a procedural geometry engine to make xonotic maps but they're like really
> like, spiraling, like they spiral in a certain tunneling direction like how
> highertower in tf2 just keeps going up. what's the procedural geometry tooling i want
> to make this scriptable through a cli or tui or randomizer?

In ~24 minutes it produced a working, validated, scriptable generator — and repaired a
toolchain bug that had been silently degrading the *existing* fusion work. The
comparison is the point: months of coupled effort produced a 166 MB artifact that
spawns players inside walls; one decoupled task produced a generator whose output is
correct by construction.

## What it found, level-3 (quoting its report)

- Do not write a CSG/brush library. **Emit `.map` text and let `q3map2` do the BSP
  tree, VIS, lightmaps and collision.** On the existing approach:

  > your existing `fusebuild/mapfuse.py` writes BSP lumps directly — great for fusing
  > prebuilt maps, but a dead end for authoring new architecture, because you'd have to
  > synthesize the tree, lightmaps and vis yourself (it currently fakes the lightmap
  > lump grey and ships empty visdata).

- The geometry kernel is one primitive, ~15 lines:

  > **swept surface → quad strip → triangle → extrude along a fixed vector → convex
  > prism**

  A triangle extruded along a constant vector is convex *for any triangle*, so an
  arbitrarily curved sweep becomes legal brushes with **zero failure modes** — no
  convex decomposition, no CSG. This matters concretely: a helical floor quad is
  **provably non-planar** (its two radial edges at different heights are skew lines),
  so the naive annular-sector prism is an invalid brush. Triangulating first is the fix.

- **Validation is by construction:** 20/20 random seeds compile leak-free with full VIS
  and lighting (604–3370 brushes, up to 12k units of climb, 0.9–10 s each), and
  `spiral1.pk3` loads in the real `darkplaces-dedicated`, confirmed by a `getstatus`
  returning `mapname\spiral1`, zero engine errors.

- A spiral is *unusually good* for idTech3: each turn occludes the next so VIS is cheap,
  and a compact XY footprint keeps the lightgrid small even at 12k units tall.

- On the TUI: skip it. `--randomize` plus an SVG top/elevation preview already provides
  the see-before-committing affordance, and is scriptable for sweeps.

## Two substrate repairs it made (they affect the fusion path too)

1. **q3map2 SIGBUS on every multithreaded stage.** `netradiant-custom/tools/quake3/
   common/threads.cpp:162` used `std::thread`, which on macOS gives worker threads a
   **512 KB** stack while only the main thread gets 8 MB; q3map2's vis/lightgrid
   recursion overflows it. The Windows branch of the same function already cranks its
   stack with the comment *"ydnar: cranking stack size to eliminate radiosity crash
   with 1MB stack on win32"* — the POSIX path never got the equivalent. Patched to
   pthreads with an 8 MB stack (`pthread_attr_setstacksize(&attr, 8 * 1024 * 1024)`,
   now at `threads.cpp:183`) and rebuilt `install/q3map2.arm64`. **vis 4.0 s → 0.35 s**,
   and `-threads 1` is no longer needed. In its words: *"This will have been silently
   costing you on the fusebuild side too."*
2. **Exact-abutment leaks.** Shell pieces offset along their own ring radial abut rather
   than overlap, leaving sub-unit slivers that leak at coarse segmentation. Bisected to
   an `--overlap` factor: 1.0 and 1.5 leak, ≥2.0 seals; default 3.0.

Also reported, unfixed: `Xonotic/data/zzz-mesh-payload.pk3` hijacks dedicated-server
boot (`Cvar_Set: variable g_payload not found`, the map command never fires) — it
smoke-tested against a clean symlinked basedir rather than touch the install.

## The staged goal (owner, this session)

The generator is not one goal but three, and they are **separately assignable**:

> 'make maps into big spirals' is one goal which can be rephrased into 'and also make
> them have apertures which let them connect to other maps in the default map pool'
> before finally 'and then can we parameterize the spiral down to sometimes spiraling,
> often not spiraling at all'

**Stage 1 — spiral.** `spiralgen.py`: `--turns --radius --radius-growth --rise --width
--height --thickness --segments --handed --wobble --seed --randomize`, emitting `.map`,
`.waypoints`, `.mapinfo`, `.meta.json` and an SVG preview; `build.sh` runs
generate → q3map2 `-meta`/`-vis`/`-light` → `.pk3` in ~3.5 s. **Done.**

**Stage 2 — apertures.** The generated map grows connection sites that let it socket
into the default map pool. Because the generator authored the geometry, an aperture is
a *parameter of the sweep* (a gap in the shell at a chosen arclength/azimuth with a
known facing and known free volume on both sides) — not something discovered by ray
marching afterwards. This is the same boundary-face object `FUSION-SPEC` needs, arrived
at by construction instead of by archaeology.

**Stage 3 — generalize the spine.** Parameterize the spiral down until it is *sometimes*
spiraling and often not: the sweep kernel already accepts any centerline, so the helix
becomes one setting of a general curve (straight runs, switchbacks, branches, a central
shaft, flat loops). The agent named this itself as the next lever — *"the sweep kernel
generalizes — any centerline works, not just a helix"*.

## What this proves about the coupling

Stages 1–3 are independently useful and independently verifiable, and none requires the
fusion, cart, strategy or demo work to be finished. That is the property the rest of the
project lost by treating map fusion, cart placement, navigability, validation and the
strategy game as one entangled problem: every task became a conjunction of six hard
problems, and an agent facing a conjunction it cannot win honestly reaches for a proxy,
a tuned constant, or a refusal. The generator is the counter-example — narrow scope,
honest scope, correct by construction, and it *also* fixed the toolchain the coupled work
depends on.

## Stage 2b — region transitions are tweened, not avoided (owner, this session)

Measured from the shipped 29-tile BSP: **14 distinct sky shaders**
(`skies/{calm_sea, distant_sunset, exosystem2, exosystem2_high_elevation,
extragalactic_asteroids, extragalactic_planets_intensity_256, heaven,
polluted_earth, purple_nebulae}` plus per-map skies from `glowplant`, `implosion`,
`space-elevator`, `warfare`, `xoylent`) and **49 texture sets** across 652 shader refs
(634 actually referenced by faces). Lateral movement changes which are active.

An earlier note here proposed designing so that sky-bearing volumes are never
co-visible. **That is superseded.** The owner:

> lets handle this by simply telling the agent working on this to handle an expanded
> requirement to make procedural tweening for skyboxes and distant lod geometry as
> players switch which area they're in. any compute shader the agent can come up with
> is probably fine, but you could use 5 different compute shaders within the same map
> for different intra-map-region transitions

So the inherited variety is the *material*, not the defect: a region change should
**tween** — sky and distant LOD silhouette cross-fading as the player traverses an
aperture — so the transition reads as travel between worlds. Per-transition variety is
explicitly wanted (~5 distinct treatments within one map), keyed to the transition type:
a corridor join, a teleporter join and a vertical shaft need not blend the same way.

Constraints carried with it:
- **Composes with VIS; does not replace it.** Real visdata is what stops all 14 skies
  and 200,946 faces being candidates every frame; tweening over an unculled world
  increases fill cost. VIS first, then tween between what VIS makes distinct.
- **Mechanism is latitude, not licence.** DarkPlaces is idTech3-derived; a compute-shader
  pipeline may not exist. Build on what the engine actually offers (Q3 `.shader`
  multi-stage blending with distance/portal-driven alpha, the DarkPlaces GLSL path,
  CSQC-driven blend parameters) and report honestly if a desired effect is unreachable.
- **Evidence is a rendered mid-blend frame**, not a description.

### Why this section exists at all: the rendering failure it comes from

The same BSP inspection that produced these counts also explains the owner's report of
"incredibly bad occlusion, texturing, and draw call latency", and the observation that
panning the camera at one spawnpoint retextures the whole scene with no movement:

    Visdata      len = 0            <- no PVS whatsoever
    Lightvols    len = 0            <- no light grid
    Lightmaps    len = 49152        <- ONE 128x128 lightmap for a 166 MB world
    LEAFS        67416 leafs, distinct cluster indices = 2 -> {-1 solid, 0}
    FACES        200946   (187338 reference lightmap 0; 11449 LIGHTMAP_BY_VERTEX; 2159 none)
    SHADERS      652 refs, 634 referenced by faces, across 49 texture sets

Face→texture indices are all in range, so this is not static corruption. With empty
visdata the visible set is the entire world, so 634 shaders are live at once and the
texture cache thrashes as the frustum sweeps — the "retexture on yaw" is the *visible
set* changing, not the position. Occlusion, draw calls, and the retexturing are one
bug with three faces, and its cause is that `mapfuse` wrote BSP lumps directly and
therefore had to synthesize the tree, VIS and lightmaps itself — faking the lightmap
grey and shipping empty visdata — compounded by an earlier deliberate single-cluster
PVS collapse of mine, introduced to stop `sv_cullentities` dropping bots.

## The traversing-object niche (owner, this session)

> so this world-traversing object which is paired with some supermassive skybox level
> object and visual effect has a niche which is missing something like it...?

It does, and the missing occupant is **the payload cart**. Verified in code, the cart is
not merely un-modelled — it is *borrowed*:

    mkentfile.py:881  visible = [m for m in models if not mclass[m].startswith('trigger_')] or models or ['*1']
    mkentfile.py:910  '"model" "%s"' % visible[c % len(visible)]

Each cart is assigned a **random brush model scavenged from the source map** — whatever
`func_door` / `func_wall` / platform happened to exist, indexed by cart number. Hence no
consistent silhouette, no size contract (a big door makes a big cart), the `view_ofs =
mins` offset that made every nearest-cart column wrong, and no team colour: control
state currently appears only on a HUD sprite and an untextured additive ribbon
(`cl_payload.qc:38`, `R_BeginPolygon("", DRAWFLAG_ADDITIVE, false)`).

The niche both objects occupy: *traverses the world, must be legible from anywhere, has
a skybox-scale counterpart, changes the world's appearance as it passes.* Consequences
now folded into the render brief:

- **Cart as a procedural traversing body**, tinted by controlling team, emissive keyed
  to control state (grey uncontrolled, team colour under plurality, streaked when
  contested), with a consistent bounding size.
- **Ink is cart territory.** Advancing under control lays that team's ink; contested
  lays muddied colour; being driven home has the deposited ink overpainted by the team
  pushing it back. This writes depth / control / contest / reversal onto the world
  surface, answering the standing requirement *"how should cart paths be diegetically
  communicated?"* without a HUD overlay — the world becomes the readout. The neutral
  match-long rubberier/wetter/off-color drift composes with it rather than competing
  for the same channel.
- **Skybox counterpart solves megamap legibility.** With a ~152,281-unit walking
  diameter you cannot see across the world; a supermassive per-cart sky presence
  (which cart, where, whose colour, how deep) is what makes it comprehensible, and it
  is the same skybox-scale rendering the artifact already requires.

Cart state to read rather than re-derive: `plc_s`/`plc_length`, `plc_ctrl`, per-team
cylinder presence — all live server-side and already networked in packed form
(`STAT(PAYLOAD_*)`, RADARLINK `clientcolors`).
