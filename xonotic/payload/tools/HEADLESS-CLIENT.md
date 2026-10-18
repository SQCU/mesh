# Headless client renders of fused-map joins (`joinshot.py`)

`joinview.py` can only *probe* a join with a raycast: the dedicated server has no GL
context, so it cannot produce an image. `joinshot.py` fills that gap by driving the
**real Xonotic/DarkPlaces client** in an offscreen software rasterizer and writing a
PNG of what a player actually sees crossing each level-to-level join.

## What renders, and how (the honest version)

* **Backend: DarkPlaces Software Rasterizer (DPSOFTRAST), windowless.**
  The stock macOS client (`Xonotic.app/Contents/MacOS/xonotic-osx-sdl-bin`, an
  x86_64 binary that runs under Rosetta 2) already contains DPSOFTRAST and the
  `vid_soft` code path. Launched with

  ```
  SDL_VIDEODRIVER=dummy  xonotic-osx-sdl-bin ... +vid_soft 1
  ```

  SDL's **dummy** video driver creates an in-memory window (no Cocoa window, no
  screen, nothing on the display), and `vid_soft 1` selects the CPU rasterizer,
  which renders into a plain SDL surface. The engine's own `screenshot foo.png`
  (`scr_screenshot_png 1`) writes that surface to disk. The engine log confirms it:
  `Using DarkPlaces Software Rasterizer rendering path` and
  `Video Mode: window 320x200x32`.

* **Why not real GL offscreen?** On macOS there is no EGL/`vid_null` GL surface and
  no reliable windowless CGL/NSOpenGL path for this engine; a hidden Cocoa GL window
  is not "headless". DPSOFTRAST is genuine, correct rasterization of the same scene
  the GL path draws (BSP, lightmaps, models) entirely on the CPU, so it is the right
  headless backend here — not a degraded raycast probe. It is **slower** and it
  **cannot decode the compressed `.dds` textures** the game ships, so we force
  `r_texture_dds_load 0; gl_texturecompression 0` and the textures load from their
  full-res `.tga`/`.png` sources (then get downscaled hard with `gl_picmip 3;
  r_texture_max_size 128` to keep first-load time and RAM in check). Missing
  `particlefont`/some fonts are harmless.

## How the camera is placed

No engine or QuakeC edits. We reuse Xonotic's stock **`info_autoscreenshot`** entity
and the **`impulse 143`** cheat, which teleports a noclipping player onto the next
such entity (adopting its view angles) and deletes it — so repeated impulses walk an
ordered camera list. `joinshot.py`:

1. Reads `fused.joins.json` and, for each join, computes both-sides cameras
   (see below), turning each into an `info_autoscreenshot` `{origin, angles}` block.
2. Repacks `fused.pk3` under a unique map name (`joinshotmap`) with those markers
   appended to `maps/joinshotmap.ent`. The rename guarantees nothing in the base
   `data/` dir (e.g. a stray `zzzz-fused.pk3`) can shadow our entity override.
3. Boots the client, joins as a player, `god; noclip`, then fires
   `impulse 143` + `screenshot <name>.png` once per camera.

The driver speaks to the client through its console on stdin and reads the client's
console on stdout; every step waits on a line, never on a delay. The first line that
shows a connection sends `cmd join` once; `is now playing` sends `god; noclip` and the
first camera. Each camera is `impulse 143`, `--settle-frames` console `wait`s (each
defers the rest of the commands by one rendered frame), `screenshot <name>.png` and
`echo JOINSHOT_SHOT <name>`; that echo sends the next camera, and the last one sends
`quit`. The driver then waits for the client to exit, however long the first map load
takes.

### Camera geometry per join kind

* **corridor** — two frames: eye on side A looking through toward B, and eye on B
  looking back toward A (the walk-through sightline each way).
* **teleporter / jumppad** — transport is instant/ballistic, so there is no
  straight-line sightline. Two frames instead: an **approach** (backed ~176 qu off
  the near pad/portal, looking at it) and a **landing** (at the far endpoint looking
  outward into the destination map).

Frames are named `j<NN>_<kind>_<a|b>_<through|approach|landing>.png`.

## Usage

```
tools/joinshot.py <fused_map_dir> --settle-frames N [--out DIR] [--width W] [--height H]
                  [--xonotic DIR] [--keep]
```

* `<fused_map_dir>` — a `mapfuse.py` output dir containing `fused.pk3` and
  `fused.joins.json` (e.g. `/tmp/fuse_v7/data/maps`).
* `--out` — where PNGs land (default `<dir>/joinshots`).
* `--width/--height` — framebuffer size (default 320x200; keep it small, this is CPU
  rasterization).
* `--settle-frames` — rendered frames between a camera move and its screenshot (the
  impulse reaches the local server and the moved view is drawn); required whenever the
  engine runs.
* `--keep` — keep the temp run dir (with `run.log`) for debugging.

Example:

```
tools/joinshot.py /tmp/fuse_v7/data/maps --settle-frames 4 --out /tmp/fuse_v7/data/maps/joinshots
```

## Sample output

`joinshots-sample/` holds a committed 256x160 capture of the `fuse_v7` seed
(`fuse` + `warfare` + `runningman`, 1 corridor + 2 teleporter joins) — six frames,
both sides of each join. A faint residual HUD element (match timer, a chat line) can
survive in a corner; it does not obstruct the geometry the frames are for.

## Honest limits

* **Software raster, low res.** Lighting/shaders are the CPU approximation, textures
  are downscaled, and `.dds`-only assets (some particles/fonts) do not appear. This
  is for *reading geometry, sightlines, occlusion and clutter at a join* — not a
  pixel-accurate beauty shot.
* **Slow, and load-sensitive.** Each run boots a full client and loads every texture
  of every fused sub-map once. On a busy shared machine expect single-digit minutes
  per run; the driver waits for the spawn as long as the load takes.
* **Camera points are the join endpoints from `fused.joins.json`.** If mapfuse placed
  an endpoint awkwardly (e.g. a teleporter pad tight against a wall), the approach
  frame will show that wall — which is exactly the kind of thing this tool exists to
  surface.
* **A human still has to look.** The tool cannot tell you whether a crossing "reads"
  as navigable — that estimate (how contorted / occluded / cluttered each join is)
  is the point, and it is made by eye from the frames.
