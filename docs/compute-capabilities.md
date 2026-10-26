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
| Laptop 1980260065 | 6 Super, 12 Performance | 40 |

The shared build operands are in `etc/compute/metal.mk`: macOS 26.0 and Metal 4.0. The bridge libraries,
Torch extension and native Swift build consume that file. Numerical shader frontends use `MetalDeviceCompilerOptions(device)`;
transport AIR uses the same language setting. Numerical floating-point options keep the language's existing default;
the transport build's explicit safe math setting remains separate. `METAL_CONFIGURATION` names a different file when
a host needs a different build. A missing file fails the build. The extension passes the deployment target explicitly
to both compilation and linking; Python's own macOS 11 build flags cannot silently set this library's API floor.

`etc/bridge/thunderbolt.mk` supplies the provider's reserved queue-pair count. `ibv_query_device` supplies each port's
total. The bridge derives usable pairs from those inputs before making a link. Frame capacity also has to fit the
bridge's completion-record table. The planner does not choose capacities from an OS version or a model name.

A prepared schedule now requires the execution context's queue as an operand. There is no schedule constructor that
creates a queue from a device. Native rank calls share their process queue; FX invocations share their stream context's
queue. Invocation slots do not multiply queues. Required queues and host-result events fail preparation if unavailable.
Core count does not establish a residency guarantee for a cooperative grid. The chain now compiles its existing
numerical stages as separate dispatches ordered by their X/Y accesses, removing its cross-group spin barrier.
The source changes and crash evidence are recorded in `design/metal-resource-failures-2026-10-26.md`.

The chain requires the `capabilities` directory in its configuration. `torch_mesh.compute.metal` loads the matching
`etc/compute/appleN.json` and reads the device's live resource limits. Matrix extent and memory alignment come from
the manifest. A missing file or field stops preparation. Launch constants, tile packing and memory sizing all consume
this record. No capability file is read while the prepared function runs.

The original chain already used these manifests before `1c0afa53`. Restoration `0cd9e915` omitted that input and
hardcoded the matrix extent and memory allowance. The eight in `K/8` was the matrix instruction's width, never a
CPU-count estimate. Both versions capped groups by columns alone. The current split includes independent row tiles;
register tiling cannot fold those rows back together if that would leave its chosen groups without work. The fused
kernels can run their groups in waves. The compiler checks each phase's pipeline limit, rather than treating a
core count as a guarantee that all groups fit at once. Larger products also have the existing prepared linear geometry.

The engine declaration requires work rates for its configured operation and shape. It no longer silently seeds CPU,
GPU and ANE shares with equal rates, or removes an incompatible positive-rate engine. CPU Accelerate still owns an
aggregate CPU share. Distinct core classes are recorded, but independent SIMD executors for those classes have not
been implemented here. The GPU's launch geometry and the choice of compute engine remain separate compiler inputs.

In metal-microbench, `MetalGPUCores` used IORegistry but silently returned eight on discovery failure. Commit
`3ae9e2bc` restored that fallback while reverting the resident kernel. Its preceding error-returning function is now
restored verbatim, and the attention planner throws on a missing count during preparation. The build and compilation
checks establish that these inputs are connected. Performance
claims still require the committed-main hourly's whole-call duration and useful throughput.

The shared `etc/compute/metal.def` manifest supplies binding and indirect-command limits, SIMD width, matrix tiles,
linear tiles and the initial launch policy. `metal-capabilities.h` exposes the same operands to C, Swift and the
Python extension; `MetalDeviceCompilerOptions(device)` and the transport source builder supply them to Metal. These consumers read one definition of each moved operand.
The linear compiler derives the grouped-product count from the binding capacity. Each row tile gets
its own grid work, and FX fusion includes wide products. The row crossover selects a compiled geometry; it does not
limit the number of rows in a function. Gather grids use the discovered GPU core count and configured work per thread.

`etc/bridge/layout.h` holds the shared-region and wire contract, including capacities and ABI checks.
`etc/bridge/thunderbolt.h` holds credit size, address-bank and registration limits, request size and provider codes.
The registration planner and DMA span compiler use the same extent. Changing a layout still requires its version to
change and its peers to be rebuilt; moving these values into the manifest does not change the existing layout.

`etc/compute/solver.json` supplies routing precision, tolerances, regularization, fit effort and observation policy.
The compiler loads it before numerical execution. These are initial numerical choices, with the existing empirical
updates still operating on the resulting page assignments.

The Swift engine consumes `metal-microbench/configs/structure.json`'s engine section, which its planner already reads.
Batch, KV capacity, page layout, classes and queue capacity now come from that operand. The eight-row MoE assertion
and silent KV-pool clamp have been removed. Page addressing in the engine's shaders consumes the same configured
page extent. Preparation still sizes storage from the requested shape and the device's available resource limits.

`etc/compute/streamed.json` is the shared Python/Swift input for streamed-weight kernel candidates and their cost
model: row tiles, K splits, tiles in flight, precision eligibility, fit priors and numerical settings. A missing
cooperative-tensor layout now stops Python preparation instead of substituting an ALU kernel. `MetalMeshFile` reads
a repository-relative preparation input; both shaders and this manifest use that same source loader.

The compiler now receives its Metal device explicitly. The cooperative kernel's thread and scratch limits are
compile operands from that device; the source no longer states 1,024 threads and 32 KiB as universal limits.
Panel tile count sets the shader's matrix shape, SIMD-group count, host grid, cost-model features and descriptor
span together. A changed capability must reach every one of those consumers; changing a launch count alone does
not configure the kernel.

Selection kernels share their radix, SIMD reductions and scratch sizing across Torch, Swift and LiteRT. Transport
launch sizes and command-storage geometry are also declared once. The engine supplies its model's hidden width,
expert count and active-expert count; shader defaults for one model have been removed. Its control capacities are
operands of the same engine configuration on the host and in the compiled shaders.
