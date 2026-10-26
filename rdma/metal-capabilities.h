#pragma once
#include <stdint.h>
#include <limits.h>
// One manifest, also readable by Swift and the Python extension. No runtime I/O.
enum {
#define MESH_METAL_VALUE(name, value) MESH_##name = value,
#include "../etc/compute/metal.def"
#undef MESH_METAL_VALUE
};
_Static_assert(MESH_BUFFER_BINDINGS <= MESH_ACCESS_WRITE_SHIFT && MESH_ACCESS_WRITE_SHIFT + MESH_BUFFER_BINDINGS <= sizeof(uint64_t) * CHAR_BIT,
              "the access-word layout must hold every configured buffer binding");
_Static_assert(MESH_ACCESS_WRITE_SHIFT + MESH_MEMORY_BINDINGS <= sizeof(uint64_t) * CHAR_BIT,
               "the declaration word must hold every threadgroup-memory binding");
static inline const char *MetalCompilationPrelude(void) {
    return
#define MESH_METAL_VALUE(name, value) "#define MESH_" #name " " #value "\n"
#include "../etc/compute/metal.def"
#undef MESH_METAL_VALUE
    ;
}
