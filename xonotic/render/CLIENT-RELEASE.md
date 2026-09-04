# Xonotic client release contract

## Provenance

The PBR and ink assignment first landed on `main` in a change whose title discussed
strategy hardcodes but whose diff also introduced the renderer PBR path, ink volume,
shaders, client build harness, and render captures. A follow-up on `main` described
the client render work as done and measured and repaired one dynamic-light PBR shader
permutation. Both changes name Claude Opus 4.7 as co-author. No feature branch or
authored `_pbr` texture exists in the named branches; the repository has only `main`
and `origin/main`.

The implementation therefore was not lost in a later branch merge. It was shipped
without completing its runtime and content contract.

## Work that was unfinished

- The custom client could not decode stock JPEG or PNG textures. It was built
  without a working JPEG ABI and could not locate PNG or Freetype dylibs. The live
  log contained hundreds of `file loaded but decode failed` errors, then CSQC
  arithmetic errors caused by zero-sized missing HUD images.
- No ORM assets, material manifest, or `_pbr` images were authored for any stock or
  custom environment material. The shader capability existed without content.
- Every model loader initialized the material metallic multiplier to zero, so the
  blue ORM channel could not make a material metallic unless every shader supplied
  an explicit override.
- Authored roughness was multiplied through a stock gloss heuristic instead of
  being interpreted as authored roughness. The stock heuristic itself ignored the
  gloss texture alpha channel.
- Vertex-blended terrain sampled only its foreground ORM texture.
- The demo staged only filename-selected stock archives. It excluded the mesh
  payload, fused map, and any other valid package while still looking provisioned.
- The client supervisor reused one userdir and session lock, launched another
  process every 15 seconds, truncated the only log on every attempt, and treated
  supervisor existence as client health.
- The release staged fresh server and client QuakeC but continued loading an older
  `menu.dat` from `zzz-mesh-payload.pk3`. The three programs claimed to be one build
  set but were not deployed as one.

## Completed release

`build-client.sh` links the discovered libjpeg ABI, bundles JPEG, the SDL2
compatibility ABI and its SDL3 implementation, PNG, and Freetype beside the
executables, rewrites their install names to loader-relative paths, and signs the
resulting relocatable build set. The SDL implementation is part of the package so
its initializer reaches the engine instead of opening a modal missing-library alert.

`pbr-materials.py` scans every staged archive and deterministically emits
`zzzzz-mesh-pbr.pk3`. It preserves 1,073 stock gloss maps, provides a low-specular
derived default for the other 2,418 materials, and supplies a derived ORM map and family
classification for all 3,491 discovered environment materials. The package embeds
its material and source manifests.

The renderer now defaults metallic modulation to one, distinguishes explicit ORM
roughness from stock-derived roughness, includes gloss alpha, and samples both ORM
textures on vertex-blended terrain. Wet paint continues to tint albedo and pull
roughness and F0 toward the runtime ink-film values; unpainted surfaces retain their
stock diffuse, normal, gloss, and classified base material response.

`runtime-package.py` emits `zzzzzz-mesh-runtime.pk3` with the matching
`progs.dat`, `csprogs.dat`, `menu.dat`, `effectinfo.txt`, payload config, and build
manifest. Both derived archives publish by atomic replacement. `demo.sh` merges every
available archive plus the PBR and runtime packages into the working content generation
without first deleting the preceding generation, records every source and checksum, and verifies the engine, complete
QuakeC set, content manifests, and package identities.
Distributed curriculum staging copies the dedicated server's loader-relative JPEG
library beside the remote executable as part of the same runtime generation.

`client-keep.sh` gives the client a stable session identity and one append-only log
per generation. It adopts only an exact matching process, reconciles lock owners,
uses TERM for executable replacement, detects executable generation changes by
inode, and separates process, connection, and renderer health. An explicit timeout,
disconnect, or host error replaces that generation once so the configured connection is
re-established. Each generation's connectivity state is read only from its own log, so a
prior disconnect cannot create a polling-period restart loop. A renderer failure is
reported as degraded instead of hidden by a restart loop. Launchers address `mesh-mini` by its stable SSH name;
address discovery is reporter data and does not replace SSH configuration.
Shutdown waits for the engine's orderly TERM path to retire and writes process-absent
health only after that retirement; an arbitrary supervisor deadline cannot leave an
unowned engine generation behind.
Renderer health covers image decoding, shader compilation, material fallback, and
image-size arithmetic. Unavailable optional cryptography, rigid-body, capture, or
codec plugins remain capability telemetry and cannot be conflated with texture
health.

## Measurements on August 31, 2026

The release engine and all three QuakeC programs built successfully. The deployed
asset manifest contains 11 archives: the complete stock data, maps, music,
compatibility, and font set; mesh payload; 29-tile fused world; PBR overlay; and
matching runtime package.

The live client connected to `127.0.0.1:26042` and remained healthy across repeated
15-second supervisor intervals. PNG and Freetype loaded, stock JPEG map imagery
decoded, and the log contained zero image decode failures, zero shader compile
failures, and zero CSQC division-by-zero failures. Nine PBR-plus-ink shader
permutations compiled on the live map.

The replacement measurement rebuilt the client while it was running. The supervisor saw
PID 39809 on obsolete inode 71685136, sent TERM, observed its exit, and started PID
44112 on inode 71686200 exactly once. The replacement connected and remained
healthy; the former 15-second collision loop did not return. The final staged
generation is recorded in `/tmp/mesh-joracle/dev.manifest`,
`/tmp/mesh-joracle/assets.manifest`, and
`/tmp/mesh-joracle/client-engine.health`.

## Relation to mesh and strategy optimization

The client is the human observability edge of the same live system the strategy
optimizer controls. A powered client that continually collides with its own lock,
or connects while rendering absent content, is reachable but not working and is an
outage under the repository contract.

Paint is a runtime presentation of authoritative cart ownership, contest,
regression, and neutral drift already carried by payload state. It does not inject
a second strategy state or alter optimizer inputs. Keeping that boundary matters:
the strategy model owns decisions and telemetry, while the renderer provides a
faithful, materially legible view of their consequences. The manifest-bound build
set prevents an observer from attributing stale client behavior to current strategy
code. If paint later becomes an optimizer input, it must enter `STRATEGY-IO.md` as
explicit measured state rather than being inferred from pixels.
