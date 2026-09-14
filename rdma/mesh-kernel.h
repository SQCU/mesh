#ifndef MESH_KERNEL_H
#define MESH_KERNEL_H
#include <stdint.h>

/* design/algorithm-sources.md#programkernel_call */
#define MESH_KERNEL_DECLARATIONS \
struct mesh_kernel_section { \
  uint64_t row_begin,row_end,column_begin,column_end; \
  uint32_t first,count; \
}; \
struct mesh_kernel_publication { \
  const struct mesh_kernel_section *sections; \
  uint64_t count; \
  void *context; \
  void (*publish)(void *,uint32_t,uint32_t); \
};

MESH_KERNEL_DECLARATIONS

#define MESH_KERNEL_STRINGIFY_INNER(...) #__VA_ARGS__
#define MESH_KERNEL_STRINGIFY(...) MESH_KERNEL_STRINGIFY_INNER(__VA_ARGS__)
#define MESH_KERNEL_SOURCE "#include <stdint.h>\n" MESH_KERNEL_STRINGIFY(MESH_KERNEL_DECLARATIONS) "\n"
#endif
