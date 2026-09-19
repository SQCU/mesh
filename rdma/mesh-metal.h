#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
/* design/prepared-machine.md#M07 */
struct mesh_metal_transport { void *publication,*publish,*stop; };
struct mesh_metal_input { void *completion,*stop; uint64_t offset,stride; };
/* design/prepared-machine.md#M10 */
struct mesh_metal_publication { void *records; uint32_t count; };
_Static_assert(sizeof(struct mesh_metal_publication)==16,"prepared publication binding");
_Static_assert(sizeof(struct mesh_metal_transport)==24,"prepared Metal transport");
_Static_assert(sizeof(struct mesh_metal_input)==32,"prepared numerical input");
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_transport_create(struct mesh_ctx *,void *,uint32_t,struct mesh_metal_transport *);
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,uint32_t,struct mesh_metal_input *);
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_publication_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,struct mesh_metal_publication *);
/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
#endif
