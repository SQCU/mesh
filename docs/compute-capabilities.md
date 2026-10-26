# Compute inputs at preparation

The compiler needs both hardware facts and work rates. A GPU core count is not a CPU count. Neither is the rate of
a matrix operation. CPU SIMD, Accelerate, Metal/MPS and Core ML can have different costs at different shapes.
Work rates enter the assignment solver as operands; whole-function observations adjust the resulting shares.

`rdma/mesh_compute.py` reads hardware facts once per process, before numerical work. It keeps each CPU performance
level's reported name, physical and logical core counts, cores per L2, and L2 size. It reads GPU cores from
`AGXAccelerator` separately. Missing facts or inconsistent CPU totals raise an error. No substitute chip or count is
used. These are inventory facts, not a claim that the OS exposes placement on each core or die.

Read from the live nodes on 2026-10-26:

| Node | Reported CPU classes | GPU cores |
|---|---|---:|
| Studio 521750041 | 10 Super, 20 Performance | 64 |
| Studio 455770905 | 10 Super, 20 Performance | 64 |
| Mini 1974797816 | 8 Performance, 4 Efficiency | 16 |

The chain requires the `capabilities` directory in its configuration. `torch_mesh.compute.metal` loads the matching
`etc/compute/appleN.json` and reads the device's live resource limits. Matrix extent and memory alignment come from
the manifest. A missing file or field stops preparation. Launch constants, tile packing and memory sizing all consume
this record. No capability file is read while the prepared function runs.

The original chain already used these manifests before `1c0afa53`. Restoration `0cd9e915` omitted that input and
hardcoded the matrix extent and memory allowance. The eight in `K/8` was the matrix instruction's width, never a
CPU-count estimate. Both versions capped groups by columns alone. The current split includes independent row tiles;
register tiling cannot fold those rows back together if that would leave its chosen groups without work. The fused
kernel still joins resident groups between products. Queuing more groups than can reside is not a valid way to widen
that kernel. Its larger products also have the existing prepared linear geometry.

The engine declaration requires work rates for its configured operation and shape. It no longer silently seeds CPU,
GPU and ANE shares with equal rates, or removes an incompatible positive-rate engine. CPU Accelerate still owns an
aggregate CPU share. Distinct core classes are recorded, but independent SIMD executors for those classes have not
been implemented here. The GPU's launch geometry and the choice of compute engine remain separate compiler inputs.

In metal-microbench, `MetalGPUCores` used IORegistry but silently returned eight on discovery failure. Commit
`3ae9e2bc` restored that fallback while reverting the resident kernel. Its preceding error-returning function is now
restored verbatim, and the attention planner throws on a missing count during preparation. The build and compilation
checks establish that these inputs are connected. Performance
claims still require the committed-main hourly's whole-call duration and useful throughput.
