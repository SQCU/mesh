#ifndef MESH_REDUCE_H
#define MESH_REDUCE_H
#include <stddef.h>
#include <stdint.h>

struct mesh_reduce_node { uint64_t owner, contributor; size_t input[2]; };
struct mesh_reduce_token { uint64_t program, invocation, node; };
typedef int (*mesh_reduce_index_fn)(void *, size_t, struct mesh_reduce_node *);
typedef struct mesh_reduce_function mesh_reduce_function;
typedef struct mesh_reduction mesh_reduction;

mesh_reduce_function *mesh_reduce_compile(size_t contributors, mesh_reduce_index_fn index, void *capture);
size_t mesh_reduce_count(const mesh_reduce_function *f);
const struct mesh_reduce_node *mesh_reduce_node(const mesh_reduce_function *f, size_t index);
void mesh_reduce_free(mesh_reduce_function *f);
mesh_reduction *mesh_reduce_bind(mesh_reduce_function *f, uint64_t program, uint64_t invocation, uint64_t owner);
struct mesh_reduce_token mesh_reduce_token(const mesh_reduction *r, size_t index);
int mesh_reduce_needed(const mesh_reduction *r, size_t index);
int mesh_reduce_ready(const mesh_reduction *r, size_t index);
int mesh_reduce_claim(mesh_reduction *r, struct mesh_reduce_token token);
int mesh_reduce_commit(mesh_reduction *r, struct mesh_reduce_token token, int error);
int mesh_reduce_status(const mesh_reduction *r);
void mesh_reduce_cancel(mesh_reduction *r, int error);
int mesh_reduce_retire(mesh_reduction *r);
#endif
