# LiteRT, its converter, the E2B path and interposition: the sources' own words

Retrieved 2026-11-06 for a choice the operator rejected the same day: torch programs are not lowered through LiteRT,
because torch, LiteRT and a third backend are each a scope of their own. The file is kept for its quotes on ML Drift,
LiteRT Torch, the repository's E2B path, the alternatives and Jones (1993). `lit2/` below names a session's temporary
download folder, which is not kept in this repository.

- **Downloads.** Each source is in its own new directory under
  `lit2/`
  (`dl_*`, plus `render_jones/` for page renders). They were treated as untrusted data.
- **Repository.** The metal-microbench files were only read. No file changed, nothing
  was built, nothing ran.
- **Quotation format.**
  - Prose from a PDF is quoted with line-break hyphens joined.
  - README and code lines are quoted raw in code blocks, with `file:line`.
  - "Not read" marks anything not fetched; nothing under it is quoted.

---

## 1. ML Drift (arXiv:2505.00232)

**Citation.** Jiuqiang Tang, Raman Sarokin, Ekaterina Ignasheva, Grant Jensen, Lin Chen,
Juhyun Lee, Andrei Kulik, Matthias Grundmann.
*Scaling On-Device GPU Inference for Large Generative Models.*
- arXiv:2505.00232v1 [cs.LG], submitted Thu, 1 May 2025 00:44:13 UTC. v1 is the only version.
- DOI 10.48550/arXiv.2505.00232.
- arXiv comment: "to be published in CVPR 2025 Workshop on Efficient and On-Device
  Generation (EDGE)".
- Affiliations: Google LLC. Ignasheva is listed as Meta Platforms, Inc., "Work done
  while at Google LLC".
- "ML Drift" is the name of the framework the paper presents. It is not in the title.

**Where read.** https://arxiv.org/abs/2505.00232 and https://arxiv.org/pdf/2505.00232
(10 pp.), saved in `dl_mldrift_2505.00232/`.

### What it is

> "Recognizing GPUs as the on-device ML accelerator with the widest reach, we present ML
> Drift–an optimized framework that extends the capabilities of state-of-the-art
> GPU-accelerated inference engines." (Abstract. The arXiv abstract page prints "ML Drift--an".)

> "This paper introduces ML Drift, an optimized inference framework designed to address the
> deployment challenges of large generative models on GPUs." (§1)

> "To address the challenges of deploying large generative models on diverse GPU
> architectures, ML Drift extends the basic architecture of a well-established
> GPU-accelerated ML inference engine [26]." (§3)

The paper's ref. [26] is: Juhyun Lee, Nikolay Chirkov, Ekaterina Ignasheva, Yury Pisarchyk,
Mogan Shieh, Fabio Riccardi, Raman Sarokin, Andrei Kulik, and Matthias Grundmann.
"On-Device Neural Net Inference with Mobile GPUs." CVPR Workshop on Efficient Deep
Learning for Computer Vision, 2019. Its arXiv abstract (arXiv:1907.01989, read through the
export.arxiv.org API, saved in `dl_arxiv_1907.01989/`) says:

> "Our state-of-the-art mobile GPU inference engine is integrated into the open-source
> project TensorFlow Lite and publicly available at https://tensorflow.org/lite."

### Which GPU backends it targets

> "ML Drift distinguishes itself by prioritizing performance optimization across a broad
> spectrum of backends, such as OpenCL [1], Metal [20], OpenGL ES [27], Vulkan [2], and
> WebGPU [53], exceeding the scope of other solutions [26, 31]." (§2)

> "To facilitate execution across diverse GPU architectures, we develop a suite of shader
> generators for various GPU backends (e.g., OpenCL, Metal, WebGPU) that transform
> platform-agnostic abstractions into target GPU languages." (§3.4)

> "We provide detailed benchmarks for ML Drift’s OpenCL, Metal, and WebGPU backends on a
> range of hardware including mobile GPUs (Arm Mali and Qualcomm Adreno), desktop/laptop
> GPUs (Intel and NVIDIA), and Apple Silicon." (§4)

> "Extending beyond mobile platforms, ML Drift facilitates execution on both integrated and
> dedicated GPUs through OpenCL and WebGPU (native execution through Dawn [52])
> interfaces." (§4.2)

### Whether kernels are generated as shader source at run time (yes)

> "We introduce a novel approach that performs dynamic code generation at runtime from
> manually optimized shader templates, categorizing ML Drift as a specialized inference
> framework optimizing data layout and kernel selection." (§3)

> "Specifically, the mapping between logical and physical indices is established during the
> shader code generation process at initialization, thus minimizing runtime latency." (§3.2)

> "Syntax translation: Converting ML Drift representation into language-specific shader
> code." (§3.4, one of the generator's per-operator transformations)

### What ties ML Drift to LiteRT

The paper never calls itself LiteRT's engine. It names TensorFlow Lite once, in related
work: "Prominent examples of these vendor-agnostic frameworks include ExecuTorch [33], ONNX
Runtime Mobile [36], MACE [55], MNN [24], NCNN [51], and TensorFlow Lite [18]." Its
ref. [18] reads "Google LLC. LiteRT Overview. https://ai.google.dev/edge/litert."

The link to LiteRT is stated in Google's repositories, quoted next.

**google-ai-edge/LiteRT**
- Commit `bf49eb62406d15948261a840c9dcd8e4f13b478c`. License Apache-2.0 (LICENSE: "Apache
  License Version 2.0, January 2004").
- Saved in `dl_litert/`.

`README.md:57`
```
* **🏎️ Faster GPU Acceleration via ML Drift:** **Supporting Gen-AI Inference.** Leverage state-of-the-art GPU acceleration with new buffer interoperability that minimizes latency across various GPU buffer types.
```
`README.md:75` (pipeline diagram)
```
    I --> J[CPU - XNNPack <br> GPU - ML Drift <br> Supported TPU/NPU]
```
`README.md:104` (platform table)
```
| **🍎 macOS** | ✅ | ✅ WebGPU <br> ✅ Metal | ANE\* |
```
`docs/instructions/GPU_BUILD_INSTRUCTIONS.md:3-5`
```
This guide walks open-source developers through building the LiteRT GPU
Accelerator shared library (`libLiteRtClGlAccelerator.so`) from source using
Bazel and [ML Drift](https://github.com/google-ai-edge/ml-drift).
```

LiteRT's tree at that commit also has `litert/runtime/accelerators/gpu/ml_drift_metal_accelerator.cc`,
`ml_drift_webgpu_accelerator.cc`, `ml_drift_cl_accelerator.cc` and others. I saw these
paths in GitHub code search only; the files are **not read**.

**google-ai-edge/ml-drift**
- Commit `e9229fa80c4a99f1855721aec8282f7f4ce86fe4`. Repository created 2026-05-14.
- License Apache-2.0 (LICENSE, and the per-file header "Licensed under the Apache License,
  Version 2.0").
- Saved in `dl_ml_drift_repo/`.

`README.md:9-10`
```
ML Drift is a high-performance, cross-platform GPU-accelerated inference
engine for on-device machine learning intelligence.
```
`README.md:25-33`
```
* **Multiple GPU API Backends:** Comprehensive support for OpenCL, Metal,
  WebGPU (via Dawn), and OpenGL ES 3.1+.
* **Unified Compute Language (UCL):** Write GPU kernels once in an
  abstraction layer over device-specific shading languages (GLSL, MSL, WGSL)
  to be compiled dynamically for different backends.
* **Broad Operator Support:** Highly tuned implementations of common ML
  operations.
* **Dynamic Code Generation:** Shaders are generated at runtime, tailored to
  the specific model, inputs, and GPU for maximum efficiency.
```
`README.md:104`
```
Technical paper: [Scaling On-Device GPU Inference for Large Generative Models](https://arxiv.org/abs/2505.00232)
```
`README.md:108-109`
```
ML Drift is actively developed and is the successor to the GPU delegate in
TensorFlow Lite.
```

ML Drift's Metal backend compiles MSL source at run time:

`ml_drift/metal/common.mm:149` (inside `CreateFunction`, line 116)
```
  id<MTLLibrary> library = [device newLibraryWithSource:code_ns options:options error:&error];
```

Its WebGPU backend hands WGSL source to Dawn:

`ml_drift/webgpu/webgpu_api_util.cc:221-227`
```
wgpu::ShaderModule CreateComputeShaderModule(const wgpu::Device& device,
                                             const std::string& src) {
  wgpu::ShaderSourceWGSL wgsl;
  wgsl.code = src.c_str();
  wgpu::ShaderModuleDescriptor descriptor;
  descriptor.nextInChain = &wgsl;
  return device.CreateShaderModule(&descriptor);
```

---

## 2. Google AI Edge Torch, now LiteRT Torch

**Repository.**
- `github.com/google-ai-edge/ai-edge-torch` now resolves to
  **`github.com/google-ai-edge/litert-torch`**. The GitHub API returns `full_name:
  google-ai-edge/litert-torch` with description "Support PyTorch model conversion with
  LiteRT."
- Read at commit `d600176b2e29bcc900a7e193f054ab3d6c3c747e` (2026-11-06T22:18:09Z).
- PyPI package `litert-torch` (README:43); Python import name `litert_torch`.
- Saved in `dl_litert_torch/`.

**License.** Apache License, Version 2.0 (LICENSE; GitHub SPDX `Apache-2.0`).

### Converting PyTorch models to the TFLite/LiteRT format

`README.md:3-9`
```
LiteRT Torch is a python library that supports converting PyTorch models into a
.tflite format, which can then be run with [LiteRT](https://ai.google.dev/edge/litert).
This enables applications for Android, iOS and IOT that can run models
completely on-device. LiteRT Torch offers broad CPU coverage, with initial GPU
and NPU support.  LiteRT Torch seeks to closely integrate with PyTorch,
building on top of torch.export() and providing good coverage of Core ATen
operators.
```
`README.md:17-18`: "Although part of the same PyPi package, the PyTorch converter is a Beta
release, while the Generative API is an Alpha release."

`README.md:27`, under "Requirements and Dependencies": ` * Operating system: Linux`

`README.md:55`: "Here are the steps needed to convert a PyTorch model to a .tflite flatbuffer:"
The example that follows calls `edge_model = litert_torch.convert(resnet18.eval(), sample_inputs)`
and then `edge_model.export("resnet18.tflite")`.

### Whether it consumes `torch.export` programs

`docs/pytorch_converter/README.md:44-46`
```
- **Note 1:** The source PyTorch model needs to be compliant with
[`torch.export`](https://pytorch.org/docs/stable/export.html) introduced in
PyTorch 2.1.0 .
```
`docs/pytorch_converter/README.md:224-228`
```
The two high-level steps that users should be aware of are
 1. [torch.export](https://pytorch.org/docs/stable/export.html) to convert
    PyTorch model to an [ExportedProgram](https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram)

 1. Lowering from ExportedProgram to an [edge\_model](https://github.com/google-ai-edge/litert-torch/blob/main/litert_torch/model.py).
```

In the code, the public entry point takes a module, and the converter runs `torch.export`
itself:
- `litert_torch/_convert/interface.py:330-331`: `def convert(` / `module: torch.nn.Module = None,`
- `litert_torch/_convert/core.py:87`: `exported_program = torch.export.export(`. The same call
  appears at 99, 107 and 115.
- `litert_torch/_convert/litert_converter.py:142-143`: the lowering is
  `def exported_programs_to_flatbuffer(` with `exported_programs: list[torch.export.ExportedProgram],`

I did not find a documented public entry point that takes an existing `ExportedProgram`.
I checked only `interface.py`.

### Its intermediate (StableHLO, stated in code only)

Neither the README nor the converter doc names StableHLO. The source does:

- `litert_torch/backend/export.py:15`: `"""Convert and lower a PyTorch ExportedProgram to MLIR."""`
- `litert_torch/backend/lowerings/_basic.py:25`: `from litert_converter.mlir.dialects import stablehlo`
- `litert_torch/backend/lowerings/_basic.py:40`: `    return stablehlo.add(x, y)`. This is
  the `aten.add` lowering, lines 34–43.
- `litert_torch/_convert/litert_converter.py:15`: `"""LiteRT converter integrations: MLIR to flatbuffer conversions."""`
- `litert_torch/_convert/litert_converter.py:227`: `    converter_api_ext.run_convert_to_tfl_passes(`

The converter doc (`docs/pytorch_converter/README.md:335-337`) also says:
"LiteRT Torch has been switched to a modern conversion backend for better on-device
compatibility and performance. However, if you encounter compatibility issues, you can
optionally switch to the legacy Torch XLA backend:"

---

## 3. What the repository's E2B path uses

**Scope.** I read only code, never any `.md`.
- Primary: `tools/prepare_e2b.py` (last changed in 43e6302f) and `tools/e2b_replay.m`
  (last changed in 81a6b59d), at HEAD `d0a72fc6`.
- Supplementary, read-only, marked [supp.]: `Makefile`, `metal_recording.m`, and the
  linked `liblitert-lm.dylib`, inspected with `nm` and `strings` and never run.
- Dawn's own source, quoted from upstream.

### (a) LiteRT/TFLite artifacts consumed

- **Input model.** `prepare_e2b.py:2405` takes the argument `model`, and the file can be
  one of two kinds:
  - A **LiteRT-LM container (`.litertlm`)**: `prepare_e2b.py:16`
    `from litert_lm_builder.litertlm_peek import get_model_type, read_litertlm_header`, read
    at 1504–1506.
  - A standalone **`.tflite`**: `prepare_e2b.py:1501` `if model_path.suffix == ".tflite":`.
- **Container sections read, by `get_model_type`.**
  - `"tf_lite_prefill_decode"`: the default, `prepare_e2b.py:1473` and `--section` at 2438.
  - `"tf_lite_embedder"` and `"tf_lite_per_layer_embedder"`: `prepare_e2b.py:1710-1711`.
- **TFLite flatbuffers** are parsed and rebuilt through the `tflite` schema object API:
  - imports `tflite.*` at `prepare_e2b.py:17-37`, including
    `tflite.StableHLOCompositeOptions` at 32;
  - parses with `Model.GetRootAs(data, section.BeginOffset())` at 1547.
- **Outputs**, which `e2b_replay.m` consumes:
  - **`.tflite`** files, written with `builder.Finish(fragment.Pack(builder), file_identifier=b"TFL3")`
    and `path.write_bytes(builder.Output())` (`prepare_e2b.py:320-321`):
    `prefill.tflite` (2081), `{tag}_rank_{rank}.tflite` (1427), `layers_rank_{rank}.tflite`
    (1978), `{tag}.tflite` (2062) and embedder `{destination}.tflite` (1730);
  - **`.metal` MSL** files: `kv_relayout` (1198–1202), `maxloc_pack` (1429–1443),
    `maxloc_reduce` (1457–1463), `round_select` and `round_assemble` (2364–2385);
  - `operands.json` (2300).
- **Loading in `e2b_replay.m`.** Each `.tflite` is `mmap`ed and loaded with
  `LiteRtCreateModelFromBuffer` (`e2b_replay.m:953-959`).
- [supp.] The model is `gemma-4-E2B-it.litertlm` (`Makefile:13`), downloaded from
  `huggingface.co/litert-community/gemma-4-E2B-it-litert-lm` (`Makefile:25`).
- [supp.] The Python `tflite` schema is generated by `flatc` from LiteRT's
  `tflite/converter/schema/schema.fbs` (`Makefile:80`).

### (b) LiteRT GPU and Dawn pieces it runs

- **Headers.**
  - LiteRT C API: `e2b_replay.m:4-11`, from `litert/c/litert_environment.h` to
    `litert/c/litert_tensor_buffer.h`, including `litert/c/options/litert_gpu_options.h`.
  - Dawn: `e2b_replay.m:12` `#include "dawn/webgpu.h"`.
- **Dawn instance, adapter and device on Dawn's Metal backend.** `e2b_replay.m:867-901`:
  - `request.backendType=WGPUBackendType_Metal;` (872);
  - toggles at 863–866, including `"disable_robustness"`, `"disable_symbol_renaming"` and
    `"dump_shaders"`, the last only when `DUMP_SHADERS` is set;
  - features at 889–891, including `ShaderF16`, `Subgroups` and
    `ChromiumExperimentalSubgroupMatrix`.
- **LiteRT environment holding Dawn's objects.** `e2b_replay.m:907-912`: the
  `kLiteRtEnvOptionTagWebGpuDevice`, `...WebGpuQueue` and `...WebGpuInstance` options,
  plus magic-number configs, passed to `LiteRtCreateEnvironment(4,native,&environment)`.
- **LiteRT GPU accelerator with the WebGPU backend, per stage.**
  - `LiteRtSetOptionsHardwareAccelerators(options,kLiteRtHwAcceleratorGpu)` (946);
  - `LrtSetGpuOptionsGpuBackend(gpu,kLiteRtGpuBackendWebGpu)` (948);
  - `LiteRtCreateCompiledModel(environment,call->model,options,&call->compiled)` (960);
  - a stage that is not fully accelerated is refused: `LiteRtCompiledModelIsFullyAccelerated(...);if(!all)return 7;` (970).
- **Buffers.**
  - Dawn buffers over host memory: `WGPUBufferHostMappedPointer` (1186–1191);
  - LiteRT tensor buffers over them: `LiteRtCreateTensorBufferFromWebGpuBuffer` (1201);
  - each buffer's `MTLBuffer`, obtained through a Dawn-internal symbol:
    `extern id<MTLBuffer> DawnMetalBuffer(WGPUBuffer) __asm__("__ZNK4dawn6native5metal6Buffer12GetMTLBufferEv");`
    (130), used at 1193.
- **Execution.** `LiteRtRunCompiledModel(...)` runs inside `[recorder capture:^{ ... }]`
  (`e2b_replay.m:1231-1258`, the call at 1255).
- **[supp.] Library linked.** `e2b_replay.m` links `-llitert-lm` from the `litert-lm==0.17.1`
  wheel (`Makefile:21`, `42`, `117`). That `liblitert-lm.dylib`:
  - contains source-path strings `third_party/ml_drift/...` (for example
    `third_party/ml_drift/webgpu/compute_pipeline_cache.h` and
    `third_party/ml_drift/common/gpu_model.cc`);
  - contains `third_party/dawn/src/dawn/native/metal/ShaderModuleMTL.mm` and
    `third_party/dawn/src/tint/lang/msl/writer/printer/printer.cc`;
  - exports `_LiteRtCreateCompiledModel`, `_LrtSetGpuOptionsGpuBackend`,
    `_LiteRtCreateTensorBufferFromWebGpuBuffer`, `_wgpuCreateInstance` and
    `__ZNK4dawn6native5metal6Buffer12GetMTLBufferEv`.

### (c) Whether the Metal kernels it records come with MSL source

**Native stages: yes, the repository's own MSL.** At `e2b_replay.m:922-930` the stage's
`.metal` file is read with `stringWithContentsOfFile:`. Its `E2B_*` macros are substituted
(923–929), and then:
```
      id<MTLLibrary> library=[recorder.device newLibraryWithSource:source options:nil error:&error];
```

**LiteRT (delegate) stages: `e2b_replay.m` never handles their shader text.**
- It sees Dawn's shaders only when `DUMP_SHADERS` is set: the `dump_shaders` toggle
  (863–866) and the logging callback (902–905).
- That callback, `compiler_log` (367–370), writes the messages to stdout.
- The source travels through the recording proxy instead:
  - `e2b_replay.m:1773-1774` says "(only the delegate's capture, above, used a Metal 3
    command buffer of recordingQueue)" and "(recorder.device is the recording proxy,
    metal_recording.m MetalDeviceRecorder)".

**[supp.] `metal_recording.m` keeps MSL only for libraries made from source.**
- The proxy is installed by dyld interposition (4244–4248):
  ```
  __attribute__((used, section("__DATA,__interpose,interposing")))
  ...
      {(const void *)&MetalRecordingDevice, (const void *)&MTLCreateSystemDefaultDevice},
      {(const void *)&MetalRecordingDevices, (const void *)&MTLCopyAllDevices}
  ```
- The proxy's `newLibraryWithSource:options:error:` (2493) stores the text:
  `result->compilation = [@{@"source":source,@"options":options ?: [NSNull null]} retain];` (2497)
- `MetalLibraryRecorder`'s `newFunctionWithName:` (2481–2483) attaches that dictionary to
  each `MTLFunction`.
- Libraries made with `newLibraryWithData:` or `newLibraryWithURL:` (2507, 2510) pass
  through `MetalCompiledLibrary` (2502), which sets no source.
- Line 464: "the recorded source is kept as it was written."

**[supp.] Dawn makes its Metal libraries from Tint-generated MSL source.**
- Source: google/dawn tag `v20260720.160313` (commit `466163dc7bf94146ac9a921b86cc3ed08184ebb8`),
  the tag `Makefile:53` pins for headers. License: BSD-3-Clause text for `Files: *`, with
  Apache-2.0 for some sub-trees. Saved in `dl_dawn/`.
- `src/dawn/native/metal/ShaderModuleMTL.mm`:
  - generates MSL: `result = tint::msl::writer::Generate(ir.Get(), r.tintOptions);` (408);
  - dumps it under the toggle: `dumpedMsg << "/* Dumped generated MSL */\n" << mslCompilation->msl;` (469);
  - compiles it: `library = AcquireNSPRef([mtlDevice newLibraryWithSource:mslSource.Get()` (550).
- **Not established:** that the Dawn inside `liblitert-lm.dylib` is exactly this revision.

**Summary for (c).**
- In this path, LiteRT runs ML Drift's **WebGPU** backend (`kLiteRtGpuBackendWebGpu`,
  `e2b_replay.m:948`), not ML Drift's own Metal backend.
- The recorded Metal kernels of the LiteRT stages are therefore Dawn/Tint's MSL translation
  of ML Drift's WGSL.
- They carry that MSL text because Dawn calls `newLibraryWithSource:` on the interposed
  device.
- The native stages carry the repository's own `.metal` text.

---

## 4. Alternatives considered and not chosen

ML Drift's own framing of the IR-based route (arXiv:2505.00232 §2): "Additionally,
IR-based runtimes, such as IREE [23], TVM [8], and XLA [48], focus on optimizing machine
learning models via lowering them through intermediate representation to hardware-specific
code."

### IREE

**Repository.** `iree-org/iree` @ `9b906db8e3930074b62ea5deb9e5623eb6a60cae`, saved in `dl_iree/`.

**License.** Apache 2.0 with LLVM Exceptions. LICENSE:204 reads
"--- LLVM Exceptions to the Apache 2.0 License ----"; GitHub's SPDX field says
`Apache-2.0`.

**Lowers from.** ML models via MLIR. `README.md:5-9`: "IREE (**I**ntermediate
**R**epresentation **E**xecution **E**nvironment, pronounced as "eerie") is an
[MLIR](https://mlir.llvm.org/)-based end-to-end compiler and runtime that lowers Machine
Learning (ML) models to a unified IR that scales up to meet the needs of the datacenter and
down to satisfy the constraints and special considerations of mobile and edge deployments."

**Metal HAL driver: emits MSL source, cross-compiled from SPIR-V.**
`docs/website/docs/developers/design-docs/metal-hal-driver.md:218-222`:
> "IREE uses compilers to compile ML models expressed with high-level op semantics down to
> GPU native source format. This is also the case for the Metal HAL driver. Metal does not
> provide an open intermediate language; we reuse the SPIR-V code generation pipeline and
> then cross compile the generated SPIR-V into MSL source with SPIRV-Cross."

Same document, lines 243–245:
> "For the Metal HAL driver, it means we need to embed the MSL kernels inside the module
> FlatBuffer. Right now we can either encode the MSL source strings and compile them at
> Metal run-time, or directly encoding the library instead."

Same document, line 23: "Currently the Metal HAL driver expects Metal 3 capabilities."

In code:
- `compiler/plugins/target/MetalSPIRV/MetalSPIRVTarget.cpp:34` sets
  `bool compileToMetalLib = true;`, the flag `"iree-metal-compile-to-metallib"` (47);
- `compiler/plugins/target/MetalSPIRV/MetalSPIRVTarget.cpp:233` sets
  `const bool embedSource = !metallib || serOptions.debugLevel > 1;`;
- the runtime `runtime/src/iree/hal/drivers/metal/executable.m` compiles source with
  `newLibraryWithSource:` (211) or loads a library with `newLibraryWithData:` (242).

### TVM

**Paper.** Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen
Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind
Krishnamurthy. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." 13th
USENIX Symposium on Operating Systems Design and Implementation (OSDI '18), Carlsbad, CA,
October 8–10, 2018, pp. 578–594. ISBN 978-1-939133-08-3. Read from
https://www.usenix.org/system/files/osdi18-chen.pdf (saved in `dl_tvm_osdi18/`); page
range from the USENIX presentation page metadata.

**Lowers from, per the paper.**

§2, p. 580:
> "The system first takes as input a model from an existing framework and transforms it
> into a computational graph representation."

Figure 2 caption, p. 581:
> "Figure 2: System overview of TVM. The current stack supports descriptions from many deep
> learning frameworks and exchange formats, such as CoreML and ONNX, to target major CPU, GPU
> and specialized accelerators."

**Metal backend.**
- The paper's only mention of Metal is a box in Figure 2's code-generation stage, labelled
  "CUDA/Metal/OpenCL" beside "LLVM IR".
- The paper's prose does not say whether the Metal output is source.
- The current code says it is, as below.

**Code.** `apache/tvm` @ `0054a95abf8f0358160f19834a5b8493a3ac4bce`, whose compiler core
is now the submodule `apache/tvm-tirx` @ `336af73223991bd5586efe528bd59f888c0efc46`. Both
are Apache-2.0. Saved in `dl_tvm_repo/`.

- `docs/arch/codegen.rst:105-107` (apache/tvm): the table row ``target.build.metal`` / Apple GPU /
  ``CodeGenMetal`` (→ Metal Shading Language).
- `docs/arch/codegen.rst:212-215`:
  > "**Source family** is more portable — it generates human-readable source that can be
  > compiled by vendor toolchains (NVCC, Metal compiler, etc.). This is necessary for GPU
  > targets where the vendor compiler handles device-specific optimizations and the runtime
  > compilation model (e.g., NVRTC for CUDA, runtime shader compilation for Metal/OpenCL)."
- tvm-tirx `src/backend/metal/codegen/codegen_metal.cc:591`:
  `std::string fmt = fmetal_compile ? "metallib" : "metal";`
- tvm-tirx `src/backend/metal/runtime/metal_module.mm`: under `if (fmt_ == "metal") {` (138),
  it calls `newLibraryWithSource:` (151). Otherwise it calls `newLibraryWithData:` (168).
- torch front end: `python/tvm/relax/frontend/torch/exported_program_translator.py:2354`
  defines `def from_exported_program(` with
  `exported_program: torch.export.ExportedProgram`; the docstring reads "Convert a PyTorch
  ExportedProgram to a Relax program".

### MLX

**Repository.** `ml-explore/mlx` @ `fc344c986c4169d99f76b5d95ae6b4a58a359a63`, saved in `dl_mlx/`.

**License.** MIT License, "Copyright © 2023 Apple Inc."

**Lowers from: its own lazy array API.** The README names no PyTorch importer.
- `README.md:9-10`: "MLX is an array framework for machine learning on Apple silicon,
  brought to you by Apple machine learning research."
- `README.md:25-26`: "**Lazy computation**: Computations in MLX are lazy. Arrays are only
  materialized when needed."

**Run-time compilation of Metal kernels from source.**
- **Default build: pre-built kernels.** `CMakeLists.txt:44` is
  `option(MLX_METAL_JIT "Use JIT compilation for Metal kernels" OFF)`.
- **`MLX_METAL_JIT` build: kernels compiled from source at run time.**
  `docs/src/install.rst:236-239`:
  > "THE ``MLX_METAL_JIT`` flag minimizes the size of the MLX Metal library which contains
  > pre-built GPU kernels. This substantially reduces the size of the Metal library by
  > run-time compiling kernels the first time they are used in MLX on a given machine."
- **Custom kernels: always compiled from source at run time.**
  `docs/src/dev/custom_metal_kernels.rst:6` and `:45-46`:
  > "MLX supports writing custom Metal kernels through the Python and C++ APIs." …
  > "Every time you make a kernel, a new Metal library is created and possibly JIT compiled."
- **Code.**
  - `mlx/backend/metal/device.cpp:745-762`:
    `Device::build_library_(const std::string& source_string, ...)` →
    `device_->newLibrary(ns_code, options, &error)`;
  - `mlx/backend/metal/compiled.cpp:263-268`: fused `mx.compile` kernels are assembled as
    source text, starting `std::string kernel = metal::utils();`.

### torch-mlir

**Repository.** `llvm/torch-mlir` @ `9676840678af042570e9e8247f2eb34cf7abc4d3`, saved in `dl_torch_mlir/`.

**License.** Dual licensed. LICENSE: "The torch-mlir project is dual licensed under both the
standard LLVM Project license of Apache License v2.0 with LLVM Exceptions and a BSD-style
license from PyTorch." GitHub SPDX: NOASSERTION.

**Lowers from: PyTorch, via FX or ONNX.**
- `README.md:3`: "The Torch-MLIR project aims to provide first class compiler support from
  the [PyTorch](https://pytorch.org) ecosystem to the MLIR ecosystem."
- `README.md:23-25`: "We have few paths to lower down to the Torch MLIR Dialect.
  - ONNX as the entry points. - Fx as the entry points"
- `python/torch_mlir/fx.py:89-90`: `def export_and_import(` with
  `f: Union[nn.Module, ExportedProgram],`. It calls `torch.export.export` at 118 and 122
  when given a module.

**Lowering targets: MLIR dialects only. It emits no Metal shader source.**
`docs/architecture.md:27-33`:
> "The "backend" of Torch-MLIR takes IR in the "backend contract" form and lowers it to
> various target dialects of interest to the MLIR ecosystem (various "backends"). In
> particular, right now we support lowering to:
> - Linalg-on-Tensors (+ `arith`, `tensor`, etc.)
> - [TOSA](https://mlir.llvm.org/docs/Dialects/TOSA/)
> - [StableHLO](https://github.com/openxla/stablehlo)"

`docs/architecture.md:7-9`: "Torch-MLIR does not attempt to provide a production end-to-end
flow for PyTorch programs by itself, but is a useful component for constructing one."

The architecture document does not mention Metal, GPU or SPIR-V. `README.md:59-62` lists
IREE among "existing downstreams".

---

## 5. Interposition (Jones 1993)

**Citation.** Michael B. Jones. "Interposition agents: Transparently interposing user code at
the system interface."
- Proceedings of the Fourteenth ACM Symposium on Operating Systems Principles (SOSP '93),
  Asheville, North Carolina, USA, December 5–8, 1993. ACM Press, pp. 80–93.
- DOI 10.1145/168619.168626.
- The page footer prints the ACM ISBN 0-89791-632-8.
- Author address on the paper: Microsoft Research, Microsoft Corporation.
- Metadata from Crossref (`dl_crossref/jones.json`); pages also printed on the scan, 80
  and 93.

**Where read.**
- `https://dl.acm.org/doi/pdf/10.1145/168619.168626` returned HTTP 403 to curl, so it is
  **not read**.
- I read the ACM scan (PDF Creator "ACM", Producer "Sony Electronic Publishing Services",
  14 pp.) from
  https://static.aminer.org/pdf/20170130/pdfs/sosp/3bdglrsipyq6uqoatnza1f28clrvmuk5.pdf,
  saved in `dl_jones_sosp93/`. I found that URL by web search; the Microsoft Research
  record is https://www.microsoft.com/en-us/research/?p=327302.
- The OCR text has errors, so both quotations below were checked against the rendered page
  image of p. 80 (`render_jones/`). The typeset quotation marks are shown as “ ”.

**The definition.** §1.1 "Terminology", p. 80:
> "Many contemporary operating systems provide an interface between user code and the
> operating system services based on special “system calls”. One can view the system
> interface as simply a special form of structured communication channel on which messages
> are sent, allowing such operations as interposing programs that record or modify the
> communications that take place on this channel. In this paper, such a program that both
> uses and provides the system interface will be referred to as a “system interface
> interposition agent” or simply as an “agent” for short."

**The toolkit.** Abstract, p. 80:
> "This paper presents a toolkit that substantially increases the ease of interposing user
> code between clients and instances of the system interface by allowing such code to be
> written in terms of the high-level objects provided by this interface, rather than in
> terms of the intercepted system calls themselves."

**Scope note.**
- Jones's "system interface" is the operating system's system-call interface.
- The repository's capture interposes Metal API entry points instead, through dyld's
  `__interpose` section (`metal_recording.m:4244-4248`).
- The proxy device both uses the real device (it forwards to it) and provides the Metal
  device interface.
- So applying Jones's term to the Metal capture carries his definition to a different
  interface than the one his paper covers.

---

## Not read

- The ACM Digital Library copy of Jones 1993 (HTTP 403). I quoted the ACM scan from AMiner instead.
- LiteRT's `litert/runtime/accelerators/gpu/ml_drift_*_accelerator.cc`: paths seen in code
  search only.
- The Dawn revision compiled into `liblitert-lm.dylib`. Dawn is quoted at the tag the
  Makefile pins for headers.
- Whether `litert_torch` has a public entry point that accepts an existing `ExportedProgram`.
  Only `interface.py`'s `convert` was checked; it takes `torch.nn.Module`.
- torch's MPS backend was not consulted.
