#pragma once
#include <stddef.h>
#include <stdint.h>
#include "mesh-host.h"

/* Compiler operands for the Core ML API. Shapes and strides are in elements.
   Names, storage and types are bound during preparation, never inferred by run. */
typedef struct { void *base; size_t bytes, offset; } MeshModelPages;
typedef struct {
  const char *name;
  MeshModelPages pages;
  const int64_t *shape, *strides;
  int rank, dtype;
} MeshModelView;

/* Scalar codes shared by the compiled model contract and page operands. */
enum MeshModelType { MeshModelFloat16, MeshModelFloat32, MeshModelInt32, MeshModelFloat64,
                     MeshModelInt8, MeshModelTypeCount };
enum MeshModelUnits { MeshModelCPUAndANE, MeshModelAll, MeshModelCPU, MeshModelCPUAndGPU };

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration JSON selects the backend, function, units, devices, explicit
   state input/output pairs and implementation.resources. Resource rows name a
   JSON identity and positive integer capacity; prepared models retain them.
   Core ML units set device eligibility; its plan
   reports preferred placement. The direct ANE backend requires its own declared
   capabilities and symbols, and returns the compiled operand contract. */
long mesh_coreml_load(const char *compiled, const char *configuration);
void mesh_coreml_release(long model);
/* Every model input and output must have a view, including optional inputs.
   Binding does not request provider defaults or leave outputs unbound. The
   direct ANE backend retains bridge-page roots and their declared offsets.
   The contract must guarantee caller-owned output pages. Public Core ML
   metadata is inspectable, but its advisory output API cannot be bound.
   Keep all page roots alive until the binding is released. */
void *mesh_coreml_bind(long model, const MeshModelView *inputs, size_t inputCount,
                      const MeshModelView *outputs, size_t outputCount);
/* Bind one invocation's request, map, completion and resource requirements.
   This is preparation; only the returned task's submit is used during work. */
const char *mesh_coreml_prepare(void *binding, MeshHostCompletion, void *context, MeshHostTask *);
void mesh_coreml_retain(void *binding);
void mesh_coreml_unbind(void *binding);
const char *mesh_coreml_error(void);
int mesh_coreml_compile(const char *package, const char *compiled);
/* Borrowed JSON held until model release: allowed_devices from compute units,
   preferred/supported device counts, unresolved counts and the operation tree
   with each operation's outputs, preferred/supported devices and nested blocks.
   Unknown device usage is null, distinct from an empty support set. The plan
   reports anticipated placement, not actual execution or output-page guarantees. */
const char *mesh_coreml_placement(long model);
/* Prepared provider operand contracts, indexed by the declared feature name.
   Direct ANE contracts include the procedure, symbol, physical shape/strides,
   scalar type, storage footprint and address alignment from its compiled model.
   output_storage records caller-pages or advisory; only caller-pages can bind.
   The caller resolves these requirements through the shared layout compiler
   before allocation. The returned JSON lives until model release. */
const char *mesh_coreml_contract(long model);

/* CPU numerical operands and workspace are bound during preparation. */
typedef struct MeshCpuCommand {
  void (*run)(const void *);
  int rows, columns, inner, input_stride, weight_stride, output_stride;
  int input_column, weight_column, output_column;
  float *input, *weight, *output, *scratch;
  int input_transpose, weight_transpose;
} MeshCpuCommand;

/* Returns the required workspace bytes, or a negative preparation error.
   Supplying a non-null workspace keeps BNNS from allocating during execution. */
int64_t mesh_cpu_product_bind(MeshCpuCommand *command);
void mesh_cpu_product_workspace(MeshCpuCommand *command, void *workspace);
void mesh_cpu_command_release(void *command);
void mesh_cpu_gelu(const void *command);
/* Preparation-only compilation. Configuration declares argv and distinct
   source_argument/output_argument tokens, replaced by prepared file paths.
   The returned artifact path is cached by the complete source/configuration
   and remains valid for replay. NULL reports mesh_coreml_error(). */
const char *mesh_cpu_compile(const char *source, const char *configuration);
/* A prepared instruction's arguments are copied once. Static entries supply
   run; compiled entries supply library and symbol instead, retained by the
   program with dlopen/dlclose. Successful creation transfers argument-owned
   resources to release_arguments; failure leaves them with the caller. */
typedef struct {
  void (*run)(const void *);
  const void *arguments;
  size_t argument_bytes;
  void (*release_arguments)(void *);
  const char *library, *symbol;
} MeshCpuInstruction;
/* resources_json is the implementation's declared resource array, including
   an explicit [] when none are required. Each row names a JSON identity and a
   positive integer capacity. Identity lifetime is held by prepared programs. */
void *mesh_cpu_program_create(const MeshCpuInstruction *instructions, size_t count, const char *resources_json);
const char *mesh_cpu_program_prepare(void *program, MeshHostCompletion, void *context, MeshHostTask *);
void mesh_cpu_program_retain(void *program);
void mesh_cpu_program_release(void *program);

#ifdef __cplusplus
}
#endif
