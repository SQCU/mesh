#pragma once
#include <stddef.h>
#include <stdint.h>
#include "mesh-host.h"
#ifdef __cplusplus
extern "C" {
#endif
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
   and remains valid for replay. NULL reports mesh_cpu_error(). */
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
  const MeshHostAccess *accesses;
  size_t access_count;
} MeshCpuInstruction;
/* resources_json is the implementation's declared resource array, including
   an explicit [] when none are required. Each row names a JSON identity and a
   positive integer capacity. Identity lifetime is held by prepared programs. */
/* Each instruction supplies its own page accesses, including any workspace.
   Program preparation expands these into ordinary host dependency nodes; the
   instruction array does not impose serial execution. */
void *mesh_cpu_program_create(const MeshCpuInstruction *instructions, size_t count, const char *resources_json);
const char *mesh_cpu_program_prepare(void *program, MeshHostCompletion, void *context, MeshHostTask *);
void mesh_cpu_program_retain(void *program);
void mesh_cpu_program_release(void *program);

const char *mesh_cpu_error(void);
/* One preparation diagnostic for the shared host providers. */
extern _Thread_local char mesh_host_error_text[2048];
#ifdef __cplusplus
}
#endif
#ifdef __OBJC__
#import <Foundation/Foundation.h>
@class MeshDeclaredResource;
@interface MeshResourceBinding : NSObject { @public MeshHostResource *entries; size_t count; }
@property(strong) NSArray<MeshDeclaredResource *> *owners;
@end
MeshResourceBinding *mesh_host_resources(id declaration);
#endif
