#pragma once
#include <stddef.h>
#include <stdint.h>
#include "mesh-host.h"
#include "mesh-cpu-source.h"

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
/* Bind source already completed by the dependency compiler. The generated
   submit/completion entries replace the same native entries; no runtime effect
   interpreter or wrapper is installed. Arguments are copied during binding. */
int mesh_coreml_entry(void *binding, const char *source, const char *compiler,
                     const void *arguments, size_t argument_bytes);
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


#ifdef __cplusplus
}
#endif
