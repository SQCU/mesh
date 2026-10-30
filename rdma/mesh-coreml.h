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

/* Configuration JSON declares units, function and allowed devices (cpu/gpu/ane).
   Loading inspects that function's complete compute plan during preparation. */
long mesh_coreml_load(const char *compiled, const char *configuration);
void mesh_coreml_release(long model);
void *mesh_coreml_bind(long model, const MeshModelView *inputs, size_t inputCount,
                      const MeshModelView *outputs, size_t outputCount);
const char *mesh_coreml_run(void *binding);
void mesh_coreml_retain(void *binding);
void mesh_coreml_unbind(void *binding);
const char *mesh_coreml_error(void);
int mesh_coreml_compile(const char *package, const char *compiled);
/* Borrowed JSON from the prepared plan, held until model release: preferred
   device counts and unresolved operator counts. Not an execution guarantee. */
const char *mesh_coreml_placement(long model);

/* CPU numerical operands and workspace are bound during preparation. */
typedef struct MeshCpuCommand {
  void (*run)(const struct MeshCpuCommand *);
  int rows, columns, inner, input_stride, weight_stride, output_stride;
  int input_column, weight_column, output_column;
  float *input, *weight, *output, *scratch;
  int input_transpose, weight_transpose;
} MeshCpuCommand;

/* Returns the required workspace bytes, or a negative preparation error.
   Supplying a non-null workspace keeps BNNS from allocating during execution. */
int64_t mesh_cpu_product_bind(MeshCpuCommand *command);
void mesh_cpu_product_workspace(MeshCpuCommand *command, void *workspace);
void mesh_cpu_command_release(MeshCpuCommand *command);
void mesh_cpu_gelu(const MeshCpuCommand *command);
void *mesh_cpu_program_create(const MeshCpuCommand *commands, size_t count);
const char *mesh_cpu_program_run(void *program);
void mesh_cpu_program_retain(void *program);
void mesh_cpu_program_release(void *program);

#ifdef __cplusplus
}
#endif
