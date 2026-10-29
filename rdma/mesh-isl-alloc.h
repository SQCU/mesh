#ifndef MESH_ISL_ALLOC_H
#define MESH_ISL_ALLOC_H
/* Included by every ISL/imath translation unit, after the system declarations.
   Compile-time symbol binding covers allocation before an isl_ctx exists too.
   Never include this header in the host library or the allocator itself. */
#include <stdlib.h>
#include <string.h>
#include "mesh-isl-memory.h"
/* Bind external symbols, not source tokens: ISL's template macros also use
   `free` as a name fragment (isl_map_free etc.). Function addresses are bound
   here too. The compiler supplies its platform's external-symbol prefix. */
#define MESH_ISL_PRAGMA_(value) _Pragma(#value)
#define MESH_ISL_PRAGMA(value) MESH_ISL_PRAGMA_(value)
#define MESH_ISL_SYMBOL_(prefix, name) prefix##name
#define MESH_ISL_SYMBOL(prefix, name) MESH_ISL_SYMBOL_(prefix, name)
#define MESH_ISL_BIND(from, to) MESH_ISL_PRAGMA(redefine_extname from MESH_ISL_SYMBOL(__USER_LABEL_PREFIX__, to))
MESH_ISL_BIND(malloc, mesh_isl_malloc)
MESH_ISL_BIND(calloc, mesh_isl_calloc)
MESH_ISL_BIND(realloc, mesh_isl_realloc)
MESH_ISL_BIND(strdup, mesh_isl_strdup)
MESH_ISL_BIND(free, mesh_isl_free)
#undef MESH_ISL_BIND
#undef MESH_ISL_SYMBOL
#undef MESH_ISL_SYMBOL_
#undef MESH_ISL_PRAGMA
#undef MESH_ISL_PRAGMA_
#endif
