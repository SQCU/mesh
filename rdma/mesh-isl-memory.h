#ifndef MESH_ISL_MEMORY_H
#define MESH_ISL_MEMORY_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* ISL preparation storage. All objects, including returned strings, belong to
   bridge pages. This allocation API is not used by recorded tensor work. */
void *mesh_isl_malloc(size_t bytes);
void *mesh_isl_calloc(size_t count, size_t bytes);
void *mesh_isl_realloc(void *memory, size_t bytes);
char *mesh_isl_strdup(const char *text);
void mesh_isl_free(void *memory);
#ifdef __cplusplus
}
#endif
#endif
