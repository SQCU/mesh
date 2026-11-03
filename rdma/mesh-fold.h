/* Preparation-only association algebra. Leaves retain rank identity; internal
   nodes retain the exact left/right order consumed by source differentiation.
   Tree rotations change association without permuting any leaf. */
#ifndef MESH_FOLD_H
#define MESH_FOLD_H
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

struct mesh_fold_node { uint32_t left,right,owner,round; };
struct mesh_fold { uint32_t nodes,count,result,leaves; struct mesh_fold_node *node; };
static inline void mesh_fold_free(struct mesh_fold *fold) { free(fold->node);*fold=(struct mesh_fold){0}; }
struct mesh_fold_visit {
  const uint32_t *parent; uint32_t *state; struct mesh_fold *fold; int ordered;
};
struct mesh_fold_value { uint32_t id,round,first,last,leaves; };
static inline int mesh_fold_visit(struct mesh_fold_visit *visit,uint32_t rank,struct mesh_fold_value *out) {
  struct mesh_fold *fold=visit->fold;
  if(visit->state[rank])return EINVAL;
  visit->state[rank]=1;
  struct mesh_fold_value value={rank,0,rank,rank,1};
  for(uint32_t k=0;k<fold->nodes;k++) {
    const uint32_t child=visit->ordered?fold->nodes-1-k:k;
    if(child==rank||visit->parent[child]!=rank)continue;
    struct mesh_fold_value next;
    int error=mesh_fold_visit(visit,child,&next);if(error)return error;
    if(visit->ordered==1) {
      /* Missing contributor ranks are legal; order is checked by the parent
         relation and by disjoint intervals, not by adjacent rank numbers. */
      if(next.last>=value.first)return EINVAL;
      for(uint32_t r=next.last+1;r<value.first;r++)
        if(visit->parent[r]!=UINT32_MAX)return EINVAL;
    }
    const uint32_t round=(next.round>value.round?next.round:value.round)+1;
    fold->node[fold->count]=(struct mesh_fold_node){visit->ordered?next.id:value.id,
      visit->ordered?value.id:next.id,rank,round};
    value.id=fold->nodes+fold->count++;value.round=round;
    if(next.first<value.first)value.first=next.first;
    if(next.last>value.last)value.last=next.last;
    value.leaves+=next.leaves;
  }
  visit->state[rank]=2;*out=value;return 0;
}
static inline int mesh_fold_decode(uint32_t nodes,const uint32_t *parent,int ordered,struct mesh_fold *out) {
  *out=(struct mesh_fold){.nodes=nodes};
  out->node=calloc((size_t)nodes+1,sizeof *out->node);
  uint32_t *state=calloc((size_t)nodes+1,sizeof *state),root=UINT32_MAX,leaves=0;
  if(!out->node||!state){free(state);mesh_fold_free(out);return ENOMEM;}
  int error=0;
  for(uint32_t r=0;r<nodes;r++)if(parent[r]!=UINT32_MAX) {
    leaves++;
    if(parent[r]>=nodes||(ordered==1&&parent[r]<r)){error=EINVAL;break;}
    if(parent[r]==r){if(root!=UINT32_MAX){error=EINVAL;break;}root=r;}
  }
  struct mesh_fold_value value={0};
  if(!error&&root!=UINT32_MAX) {
    struct mesh_fold_visit visit={parent,state,out,ordered};
    error=mesh_fold_visit(&visit,root,&value);
    if(!error&&value.leaves!=leaves)error=EINVAL;
  } else if(!error)error=EINVAL;
  free(state);
  if(error){mesh_fold_free(out);return error;}
  out->result=value.id;out->leaves=leaves;return 0;
}
/* Encode a binary association by the rightmost leaf that owns each subtree.
   Each nonfinal leaf is the left root at exactly one combine. */
static inline uint32_t mesh_fold_encode_node(const struct mesh_fold *fold,uint32_t value,uint32_t *parent) {
  if(value<fold->nodes)return value;
  const struct mesh_fold_node node=fold->node[value-fold->nodes];
  const uint32_t left=mesh_fold_encode_node(fold,node.left,parent),right=mesh_fold_encode_node(fold,node.right,parent);
  parent[left]=right;return right;
}
static inline void mesh_fold_encode(const struct mesh_fold *fold,uint32_t *parent) {
  for(uint32_t r=0;r<fold->nodes;r++)parent[r]=UINT32_MAX;
  const uint32_t root=mesh_fold_encode_node(fold,fold->result,parent);parent[root]=root;
}
typedef int (*mesh_fold_emit)(void *,const uint32_t *);
/* One-step neighbors, polynomial in the retained tree size. Repeated
   preparation reaches every rank-preserving binary association by rotations;
   this is reachability, not a claim about convergence of empirical costs. */
static inline int mesh_fold_rotations(uint32_t nodes,const uint32_t *parent,mesh_fold_emit emit,void *context) {
  struct mesh_fold fold={0};int error=mesh_fold_decode(nodes,parent,1,&fold);
  if(error)return error;
  uint32_t *made=malloc(((size_t)nodes+1)*sizeof *made);
  if(!made){mesh_fold_free(&fold);return ENOMEM;}
  for(uint32_t i=0;!error&&i<fold.count;i++)for(int right=0;!error&&right<2;right++) {
    const struct mesh_fold_node saved=fold.node[i];
    const uint32_t child=right?saved.left:saved.right;
    if(child<nodes)continue;
    const uint32_t j=child-nodes;const struct mesh_fold_node nested=fold.node[j];
    if(right) {
      fold.node[i].left=nested.left;fold.node[i].right=child;
      fold.node[j].left=nested.right;fold.node[j].right=saved.right;
    } else {
      fold.node[i].left=child;fold.node[i].right=nested.right;
      fold.node[j].left=saved.left;fold.node[j].right=nested.left;
    }
    mesh_fold_encode(&fold,made);error=emit(context,made);
    fold.node[i]=saved;fold.node[j]=nested;
  }
  free(made);mesh_fold_free(&fold);return error;
}
#endif
