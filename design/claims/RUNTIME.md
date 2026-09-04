# Runtime and rendering implementation claims

Controlling specifications: [`../mesh-coprocessor-demo.md`](../mesh-coprocessor-demo.md),
[`../FUSION-SPEC.md`](../FUSION-SPEC.md),
[`../NAV-SPEC.md`](../NAV-SPEC.md), and
[`../ALGORITHM-CONTRACTS.md`](../ALGORITHM-CONTRACTS.md).

Implementation surfaces:

- `viz/serve.py` owns the one 8787 whole-fabric service and its reconnecting telemetry
  subscriptions.
- `viz/index.html` owns budgeted phase-space calculation and time-based visual tweening.
- `xonotic/render/` owns stock content inclusion and derived PBR material artifacts.
- `xonotic/darkplaces-work/` owns the runtime renderer, client protocol, and engine
  execution surfaces.
- `xonotic/payload/runtime-package.py` owns release artifact assembly.
- `xonotic/solver/strat/joracle/demo.sh`, `evaluate-distributed.sh`, and
  `client-keep.sh` own generation staging, cardinality-derived engine capacity, and client
  supervision.

The 8787 process is a replaceable presentation subscriber. It does not own telemetry
truth, persist a filesystem cache, open duplicate browser panes, or establish node
membership. Node telemetry remains in leased in-memory rings, and a restarted viewer
reconstructs current presentation by subscribing to every reachable node.
Each currently reachable leased node has its own exact phase anchor and magnified lens
trajectory, joined to the observing node by the measured telemetry-reachability edge.
Particle motion on that edge is driven separately by live bridge counters, so topology
does not imply invented RDMA traffic.

The client release contains stock Xonotic texture and model namespaces alongside the
derived paint/PBR material namespace. Missing material inputs, shader load failures,
renderer errors, content incidence, frame time, and GPU utilization are reported as
measures rather than hidden by a fallback presentation.
Staging merges every available archive into the live content namespace without first
removing the working generation. PBR and QuakeC runtime archives are completely assembled
beside their destination and atomically replace the preceding archive, so generation work
cannot expose a truncated package to a live client.
Server launch retains every authored map entity, derives the autoscreenshot capacity from
the entity files, and disables DarkPlaces' fixed ten-million-jump VM cutoff. Runtime
construction therefore scales with realized work instead of restarting on single-map
defaults or a fixed interpreter counter.
Generated generic spawns come from the player-hull-plus-body-width horizontal erosion,
lower-half-height standing lift, and full-precision origin realization specified by
`FUSION-SPEC.md`; runtime logs measure engine collision incidence rather than relying on
the source constructor's coordinate quantization.

The `USEPBR` shader permutation replaces the stock specular lobe with a metallic-roughness
GGX/Trowbridge-Reitz distribution, Smith-Schlick visibility, and Schlick Fresnel. Its
Fresnel-tinted specular color and remaining diffuse energy are the inputs selected by the
existing lighting permutations; permutations without a specular term keep the stock diffuse
albedo. These are renderer-contract claims here, not comments embedded in generated shader
source.
