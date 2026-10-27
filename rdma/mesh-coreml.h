#pragma once
#include <stddef.h>
#include <stdint.h>

/* Compiler operands for the Core ML API. Shapes and strides are in elements.
   Names, storage and types are bound during preparation, never inferred by run. */
typedef struct {
  const char *name;
  void *data;
  const int64_t *shape, *strides;
  int rank, dtype;
} MeshModelView;

enum MeshModelType { MeshModelFloat16, MeshModelFloat32, MeshModelInt32, MeshModelFloat64 };
enum MeshModelUnits { MeshModelCPUAndANE, MeshModelAll, MeshModelCPU, MeshModelCPUAndGPU };

#ifdef __cplusplus
extern "C" {
#endif

long mesh_coreml_load(const char *compiled, int units);
void mesh_coreml_release(long model);
void *mesh_coreml_bind(long model, const MeshModelView *inputs, size_t inputCount,
                      const MeshModelView *outputs, size_t outputCount);
const char *mesh_coreml_run(void *binding);
void mesh_coreml_retain(void *binding);
void mesh_coreml_unbind(void *binding);
const char *mesh_coreml_error(void);
int mesh_coreml_compile(const char *package, const char *compiled);
long mesh_coreml_placement(long model, char *output, size_t size);

#ifdef __cplusplus
}
#endif
