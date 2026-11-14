# Assignment: completed CPU library to native fixed calls

Root approved this narrowed contract on 2026-11-13. Resume the SAME native session now and implement this connection. It replaces the earlier open-ended native runtime assignment.

## Actual source and finite ABI

Use installed `/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_package/model_package_loader.h`, `aoti_runner/model_container_runner_cpu.h`, `aoti_runtime/interface.h` and `aoti_torch/c/shim.h`. Reuse native AOTIModelPackageLoader/get_runner/boxed_run as the actual loading/invocation owner. Its existing runner owns native proxy-executor metadata and constants; do not reimplement dlopen/Create/Run with a null proxy that silently loses that coverage.

Producer is adjacent CPU target: already compiled DSO files, constant data, exact ordered native region calls with input/output value-slot lists, and actual task/operation record. Consumer is a bound native entry. Each emitted region site makes the fixed native AOT call selected during preparation. No runtime opcode/FX decoder and no Python callbacks from that entry. Loading supplies the completed AOT package to that existing native loader; the producer must include completed libraries so ordinary load never rebuilds emitted source. No new package parser, symbol registry, or alternate loader.

The finite interface is:

```c
void *mesh_aoti_load(const char *package_path, const char *model_name,
                    char *error, size_t room);
int mesh_aoti_run(void *model, void **inputs, size_t input_count,
                 void **outputs, size_t output_count,
                 char *error, size_t room);
void mesh_aoti_free(void *model);
```

`load` owns one public `AOTIModelPackageLoader`. `run` delegates to its public `boxed_run`. The installed `aoti_torch/tensor_converter.h` is the ownership owner: `alloc_tensors_by_stealing_from_handles` steals and nulls inputs, `unsafe_alloc_new_handles_from_tensors` supplies caller-owned output handles, and `free_unstolen_handles` releases any inputs left on an error exit. This avoids the arrayref rank bound and preserves the package loader's proxy and constants. Do not bypass public runner ownership through protected-member casts or implement a private loader. Exceptions are reported through the supplied error buffer.

Python exposes `Model(package_path, model_name, input_count, output_count)`, retaining native context and run entry. Its one outer tensor-call binding uses the existing `torch._C._aoti` capsule converters. Counts and exact package member name come from the producer's region recipe; they are not inferred from source or runtime shapes. A native composition consumer can call the same C entry/context directly. No new `Entry`/site compiler, Python loop over regions, or adaptation of mesh_metal4 belongs to this unit.

The adjacent producer returns package bytes, ordered model names/input/output value slots and actual task/dependence/operation records. Compilation and archive restoration both bind the same Model interface. The package loader owns its extracted temporary directory; a caller supplying an unpacked package retains that directory. Normal load binds completed libraries and does not compile or schedule.

This unit finishes numerical artifact loading and invocation. Whole fixed-call composition and native SEND/RECV/copy still need their own exact producer-consumer assignment; the current Python callbacks are not accepted by this leaf. MPS/ANE completed-target replacements also remain separately assigned work. No independent placement, schedule, lifetime solver, threadpool or channel-capacity design is authorized here.

Files: rdma/mesh-aoti.cpp and rdma/mesh_aoti.py, plus the exact required rdma/Makefile target/linkage. CPU AOT loading does not belong in the Metal-only driver. Coordinate any additional caller hunk through supervisor before editing. No edits to numerical Native methods or broad mesh_metal4 rewrites. Existing driver selected target is the actual consumer; no unused alternate API.

Evidence: existing CPU engines/client uses actual C entry and restores saved binary on fresh binding. Report command and actual symbol/artifact, distinguish isolated API inspection from end-to-end execution. No benchmark grid, parser, compatibility facade, graph-fingerprint gate or self-acceptance review.

## Current coordination and limit

Archive worker is resumed same session047e12be-cd27-4637-a3bb-7d46f6e8216b and owns the concrete consumer in lower_inductor.py/archive.py/driver.py. Agree direct preparation API and exact package/model/slot bindings with it through status/inbox now. Native loader receives the package path and actual model name from the producer, never identifies it by parsing emitted code. Return native entry/context and explicit lifetime ownership. Preserve the loader-owned extraction lifetime, and retain any caller-owned unpacked directory while bound. Existing installed headers supply error and handle semantics; no custom proxy executor or package parser.

Implementation is this finite binding only. Do not touch runtime.py/bridge.py/CoreML/mesh_metal4/old Native hierarchy under old broad permission. Native SEND/RECV/copy needs its subsequent exact handoff, not a callback completion claim. Root handles selected-target client changes. No worker commits; supervisor reviews named paths regularly. No worker trees. Shared build/hardware/service changes remain root coordinated; private object/link and isolated CPU API inspection allowed. Use verified /Users/mdot/.venv-mesh-uv/bin/python with TORCH_DEVICE_BACKEND_AUTOLOAD=0. Read native.inbox.md before each edit batch, this order overrides earlier broad instructions.
