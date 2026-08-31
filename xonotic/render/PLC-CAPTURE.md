# Capturing and benchmarking a real payload match

`bench.sh` drives a stock deathmatch; `plc-run.sh` drives the `plc` gamemode with the
artifact, the procedural cart bodies and the world ink turned on, on a **stock map**
(`dance`) rather than on the fused world — the fused BSP still ships empty visdata and
one 128x128 lightmap for 200,946 faces, so any timing taken there measures the missing
vis, not the renderer.

## Setup (`plc-home/`)

    plc-home/data/{progs.dat,csprogs.dat}     from xonotic/payload-build
    plc-home/data/gamemodes-payload.cfg       the mode's cvar overlay
    plc-home/data/maps/dance.{ent,mapinfo}    tools/mkentfile.py output + gametype plc
    plc-home/data/autoexec.cfg                bots, no map end, sv_public -1

The `.ent` is generated with `tools/mkentfile.py <bsp> <out.ent> <teams> <carts>`. Note
that mkentfile currently emits `plc_goal "cnt"` values from a k>4 palette (32, 25); the
mode reads `cnt` with the `dom_team` convention (4 red / 13 blue / 12 yellow / 9 pink),
so the generated values were remapped in the `.ent` before use.

## Frames — `plc-run.sh <label> <seconds> <steps.cfg>`

Runs the GL client windowed at 1280x720 against `plc-home`, execing `steps.cfg`, and
kills it after `<seconds>`. `steps.cfg` is a list of `defer` lines; the deferred queue
survives the map load, so one file scripts a whole match.

An early/late pair of the *same* scene needs the camera pinned, which is what
`cl_artifact_camera 1` + `cl_artifact_cam_origin` / `cl_artifact_cam_angles` are for
(`cl_artifact_camera -1` prints the live view each second so a camera can be chosen).
`god; noclip` keeps the client alive so the death scoreboard never covers the frame, and
`r_drawviewmodel 0` takes the weapon out of it.

Only one client at a time: the userdir holds a session lock, and a second client exits
with "session lock could not be acquired".

## Cameras used for the shipped pair

    deck   origin  -200 0 330    angles 6 0 0      the central deck
    b      origin  0 -300 330    angles 6 180 0    the same deck, looking back

`track` (`-600 -1150 240`, `9 161 0`) was aimed at cart 0's path and looks off the
map edge on `dance`; it is kept in the script only as a negative result.

## Measured frame times

`timedemo` could not be used: `record` must run before connecting, and
`record <name> <map>` bypasses the gametype selection so the recorded match came up
as CTF with no carts at all. The measurement instead uses the engine's own
`timerefresh`, which renders 128 frames sweeping 360 degrees from the client's
current origin — reproducible for an A/B because the client is `noclip`ped and
stationary, and every variant renders from the same point in the same match.

`dance`, `plc`, 3 carts, 9 bots, 1280x720 windowed, GL on an Apple M5 Max,
`vid_vsync 0`:

| variant                                    | fps   | ms/frame |
|--------------------------------------------|-------|----------|
| everything on (warm-up)                     | 201.1 | 4.97     |
| everything on                               | 215.2 | 4.65     |
| everything on (repeat)                      | 196.9 | 5.08     |
| everything on (repeat)                      | 199.4 | 5.02     |
| `cl_artifact 0` (ink shading still on)      | 234.7 | 4.26     |
| `cl_artifact_sky 0` (world bodies only)     | 227.5 | 4.40     |
| `r_ink 0` (artifact bodies still on)        | 196.6 | 5.09     |
| `cl_artifact 0; r_ink 0` (stock renderer)   | 234.6 | 4.26     |

Read across the run-to-run spread (~10%, the match is live under the sweep):

* **The whole presentation costs ~0.75 ms/frame** (4.26 -> ~5.0 ms), ~15%.
* **Of that, ~0.6 ms is the four supermassive sky bodies** — they are fill-bound,
  not vertex-bound: `cl_artifact_sky 0` recovers most of the cost while leaving the
  drifting body and all three cart bodies drawn. `cl_artifact_sky_scale` and
  `..._thickness` are therefore the levers if it ever has to get cheaper.
* **The ink volume is free within the noise.** `cl_artifact 0` (234.7 fps, ink
  volume live and sampled by every world fragment) and `cl_artifact 0; r_ink 0`
  (234.6 fps, no volume at all) are the same number, and `r_ink 0` with the bodies
  on is the same as everything on. One trilinear 3D fetch per fragment plus the
  partial re-upload of the dirty box does not show up at 1280x720.

Volume cost, from `r_ink_stats` on the same match: 96x96x57 RGBA8 = **2.0 MB**,
64.7 units per voxel, and over a 5-minute match ~11.3M voxel-uploads, i.e. about
45 MB of texture traffic in 300 s (~150 KB/s) — the partial dirty-box upload path
doing its job against a 2 MB volume.

## Capture gotchas found the hard way

* `record <name> <map>` starts the map itself and comes up in the map's *first*
  supported gametype, ignoring `g_payload 1` — the recorded match was CTF with no
  carts. `g_payload 1; map dance` as a single deferred command is the form that
  actually starts `plc`.
* A `defer` queue survives the map load, so one steps file scripts a whole match;
  but two clients cannot share the userdir (session lock), so kill the previous
  client before starting the next.
* `cl_welcome 0` (welcome dialog), `hud_damage 0` (the full-screen red damage tint
  that otherwise washes every frame), `cl_deathscoreboard 0` (the scoreboard that
  covers the top half whenever the client dies) and `r_drawviewmodel 0` are all
  required for a legible evidence frame. `god` on its own is not enough — the
  client can still die between the join and the cheat taking effect.
* On `dance` the drift needs a wide glob (`cl_artifact_glob_radius 420`) to build
  visible coverage inside a five-minute capture: the level is a set of floating
  platforms, so the drift spends much of its Lissajous over open sky and only a
  fraction of its globs find a floor.

## Shipped frames (`shots/`)

All 1280x720, `dance`, `plc`, 3 carts, 3 teams, 9 bots, same match, same pinned
cameras, `r_ink_stats` at the moment of each.

| file | when | ink volume |
|---|---|---|
| `plc-dance-deck-early.png` | t=47  | 869 splats |
| `plc-dance-deck-late.png`  | t=302 | 10,039 splats |
| `plc-dance-b-early.png`    | t=51  | |
| `plc-dance-b-late.png`     | t=306 | |
| `plc-dance-c-early.png`    | t=55  | |
| `plc-dance-c-late.png`     | t=310 | 11,019 splats |
| `plc-ink-fill-proof.png`   | `r_ink_fill 1 0.15 0.65 1` | the material response with the volume saturated, used to prove the shader path independently of where the artifact happened to paint |

`deck` early vs late is the accumulation pair: a single green patch on the lower deck
becomes the whole deck and the walkway above it. `b` late carries the blue cart's
supermassive sky body with its travelling progress band; `deck` late carries two red
ones (a second cart) plus the drifting artifact's own green body and a red cart body
standing on the deck itself.
