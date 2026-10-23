#ifndef MESH_PLAN_H
#define MESH_PLAN_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

struct mesh_reach { uint32_t node,cable; uint64_t bandwidth; };
int mesh_reached(struct mesh_reach *out,uint32_t capacity,uint32_t *node);

int mesh_region(uint64_t *bytes,uint64_t *unleased);

enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH };
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t (*link)[2]; uint64_t *bandwidth; uint32_t relays; uint32_t *next,*mark; };

int mesh_link_map_joined(const struct mesh_link_map *,uint32_t *apart);

int mesh_routes(const struct mesh_link_map *,uint32_t *next);

uint32_t mesh_route_path(const struct mesh_link_map *,uint32_t a,uint32_t b,uint32_t *path);

int mesh_map_members(const struct mesh_link_map *whole,const uint32_t *members,uint32_t n,uint32_t *whole_node,struct mesh_link_map *map);

int mesh_map(const uint32_t *members,uint32_t n,uint32_t *node,struct mesh_link_map *map);

int mesh_map_wait(uint64_t *serial,uint64_t bound_ns);

int mesh_link_between(const struct mesh_link_map *,uint32_t a,uint32_t b);

struct mesh_trees { uint32_t nodes,count; const uint32_t *root,*parent; const double *log_weight; const uint32_t *way; };

struct mesh_operand { uint32_t type,element_bytes; uint64_t elements; };

enum { MESH_STEP_SEND, MESH_STEP_REDUCE, MESH_STEP_COPY };
struct mesh_step { uint32_t op,peer,round,buffer; uint64_t first; struct mesh_operand piece; uint32_t way,padding; };

enum { MESH_ALLREDUCE, MESH_BROADCAST, MESH_REDUCE, MESH_REDUCE_SCATTER, MESH_ALLGATHER };
struct mesh_move { uint32_t op,peer,round,buffer,part,parts,way,padding; };
struct mesh_program { uint32_t what,root,nodes,parts,scratch,padding; const uint32_t *segment; const double *log_weight;
                      const uint32_t *first; const struct mesh_move *move; };

void mesh_compile_room(const struct mesh_trees *,uint32_t chunks,uint32_t *parts,uint32_t *moves);

int mesh_compile(const struct mesh_trees *,uint32_t what,uint32_t root,int whole,uint32_t chunks,const uint64_t *contributors,
                 struct mesh_program *program,uint32_t *segment,double *log_weight,uint32_t *first,struct mesh_move *move);

uint32_t mesh_program_steps_cut(const struct mesh_program *,uint32_t rank,struct mesh_operand,const uint64_t *segments,uint64_t page,
                                struct mesh_step *steps);

struct mesh_place { uint32_t step,send; uint64_t entry; };

uint64_t mesh_lay(const struct mesh_step *steps,uint32_t count,uint32_t peer,uint32_t way,int send,uint64_t at,struct mesh_place *made,uint32_t *placed);

int mesh_steps_before(const struct mesh_step *steps,uint32_t count,int32_t *before);

enum { MESH_GROUP_COLLECTIVE, MESH_GROUP_POINT, MESH_GROUP_HOSTED };

int mesh_group_before(uint32_t kind,const struct mesh_step *steps,uint32_t count,int32_t *before);
struct mesh_group_call { uint32_t kind,padding; const uint32_t *count; const struct mesh_step *const *steps; };
struct mesh_group { uint32_t nodes,calls; const struct mesh_group_call *call; uint64_t page; };
struct mesh_group_place { uint32_t call,padding; struct mesh_place place; };
uint64_t mesh_group_lay(const struct mesh_group *,uint32_t rank,uint32_t peer,uint32_t way,int send,uint64_t at,struct mesh_place *scratch,
                        struct mesh_group_place *made,uint32_t *placed);

struct mesh_routing { uint32_t nodes,routes; const uint32_t *pair,*start,*node; const double *share; const uint32_t *way; };

struct mesh_demand { uint32_t root,padding; uint64_t up,down; const uint64_t *leaves; };

uint32_t mesh_solve_room(const struct mesh_link_map *,double eps,uint32_t count);

uint32_t mesh_solve(const struct mesh_link_map *,double eps,double hop,const struct mesh_demand *demands,uint32_t count,uint32_t room,
                    uint32_t *demand,uint32_t *parent,uint32_t *way,double *share);

uint32_t mesh_flow_trees(uint32_t nodes,const struct mesh_demand *demands,const uint32_t *demand,const uint32_t *parent,const uint32_t *way,
                         const double *share,uint32_t count,uint32_t first,uint32_t last,uint32_t *root,uint32_t *tree,uint32_t *tree_way,
                         double *log_weight);

uint32_t mesh_flow_routing(uint32_t nodes,const struct mesh_demand *demands,const uint32_t *demand,const uint32_t *parent,const uint32_t *way,
                           const double *share,uint32_t count,uint32_t first,uint32_t last,struct mesh_routing *routing,uint32_t *pair,
                           uint32_t *start,uint32_t *node,uint32_t *hop_way,double *weight);

void mesh_cut_shares(uint64_t elements,uint32_t element_bytes,uint64_t page,const double *share,uint32_t count,uint64_t *first,uint64_t *length);

struct mesh_block { uint32_t from,to,source,target,type,element_bytes; uint64_t first,at,elements; };
struct mesh_part { uint32_t route,padding; uint64_t first,elements; };

uint32_t mesh_block_parts(const struct mesh_routing *,const struct mesh_block *,int shortest,uint64_t page,struct mesh_part *parts);

uint32_t mesh_exchange_steps(const struct mesh_routing *,uint32_t rank,const struct mesh_block *blocks,uint32_t count,int pairwise,uint32_t relay,
                             uint64_t page,struct mesh_step *steps,uint32_t *block,uint64_t *relayed);

uint32_t mesh_exchange_room(const struct mesh_routing *,const struct mesh_block *blocks,uint32_t count,uint32_t *parts);

struct mesh_health { uint64_t booted,available; };

#define MESH_MODEL_BYTES 32
struct mesh_node_health { uint32_t node; uint64_t region,memory; char model[MESH_MODEL_BYTES];
                          struct mesh_health health; };

int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity);

#ifdef __cplusplus
}
#endif
#endif
