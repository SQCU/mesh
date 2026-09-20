#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
/* design/prepared-machine.md#M07 */
struct mesh_metal_transport { void *publication,*stop; };
struct mesh_metal_input { void *completion,*stop; uint64_t offset,stride; };
/* design/prepared-machine.md#M10 */
struct mesh_metal_publication { void *records,*completion; uint32_t count; };
/* design/prepared-machine.md#M28 */
struct mesh_output_status { _Alignas(32) uint32_t remaining; uint32_t padding[7]; };
_Static_assert(sizeof(struct mesh_output_status)==32 && _Alignof(struct mesh_output_status)==32,"M28");
_Static_assert(sizeof(struct mesh_metal_publication)==24,"prepared publication binding");
_Static_assert(sizeof(struct mesh_metal_transport)==16,"prepared Metal transport");
_Static_assert(sizeof(struct mesh_metal_input)==32,"prepared numerical input");
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_transport_create(struct mesh_ctx *,void *,struct mesh_metal_transport *);
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,uint32_t,struct mesh_metal_input *);
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_publication_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,uint32_t,struct mesh_metal_publication *);
/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
#endif
