/* AOT ordering of streamed path fragments. No clocks or transport words are
   read here. Every task occupies one directed cable; a relay can begin after
   its parent's first page, while its last page still depends on that parent. */
#ifndef MESH_WRITE_ORDER_H
#define MESH_WRITE_ORDER_H
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

struct mesh_write_task {
  uint32_t arc, parent, order;
  uint64_t production;
  double bytes, release, start, finish, first_bytes;
};

/* These heaps belong to compilation. A candidate is inserted once when its
   parent has a predicted interval, then moved once from pending to ready.
   Numerical execution never allocates, visits, or changes these structures. */
struct mesh_write_heap { uint32_t *items, count; };
static int mesh_write_before(const struct mesh_write_task *tasks,uint32_t a,uint32_t b,int pending) {
  if(pending&&tasks[a].release!=tasks[b].release)return tasks[a].release<tasks[b].release;
  if(tasks[a].production!=tasks[b].production)return tasks[a].production<tasks[b].production;
  return a<b;
}
static void mesh_write_push(struct mesh_write_heap *heap,uint32_t item,const struct mesh_write_task *tasks,int pending) {
  uint32_t at=heap->count++;
  while(at&&mesh_write_before(tasks,item,heap->items[(at-1)/2],pending)) {
    heap->items[at]=heap->items[(at-1)/2];at=(at-1)/2;
  }
  heap->items[at]=item;
}
static uint32_t mesh_write_pop(struct mesh_write_heap *heap,const struct mesh_write_task *tasks,int pending) {
  const uint32_t first=heap->items[0],last=heap->items[--heap->count];
  uint32_t at=0;
  while(at<heap->count/2) {
    uint32_t child=2*at+1;
    if(child+1<heap->count&&mesh_write_before(tasks,heap->items[child+1],heap->items[child],pending))child++;
    if(!mesh_write_before(tasks,heap->items[child],last,pending))break;
    heap->items[at]=heap->items[child];at=child;
  }
  if(heap->count)heap->items[at]=last;
  return first;
}
static int mesh_write_order(struct mesh_write_task *tasks, uint32_t count,
                            const double *capacity, uint32_t arcs, double page) {
  if(!(page>0)||!isfinite(page))return EINVAL;
  for(uint32_t a=0;a<arcs;a++)if(!(capacity[a]>0)||!isfinite(capacity[a]))return EINVAL;
  for(uint32_t k=0;k<count;k++) {
    struct mesh_write_task *t=tasks+k;t->order=UINT32_MAX;
    if(t->arc>=arcs||(t->parent!=UINT32_MAX&&t->parent>=count)||!(t->bytes>0)||!isfinite(t->bytes)||
       !isfinite(t->release)||t->release<0||(t->release&&!isnormal(t->release)))return EINVAL;
    if(!t->first_bytes)t->first_bytes=fmin(page,t->bytes);
    if(!(t->first_bytes>0)||t->first_bytes>t->bytes||!isnormal(t->first_bytes/capacity[t->arc])||
       !isnormal(t->bytes/capacity[t->arc]))return EINVAL;
  }
  struct mesh_write_arc { struct mesh_write_heap pending,ready; uint32_t count; double free_at; };
  struct mesh_write_arc *links=calloc((size_t)arcs+1,sizeof *links);
  uint32_t *pending=malloc(((size_t)count+1)*sizeof *pending),*ready=malloc(((size_t)count+1)*sizeof *ready),
           *children=malloc(((size_t)count+1)*sizeof *children),*next=malloc(((size_t)count+1)*sizeof *next);
  int status=0;
  if(!links||!pending||!ready||!children||!next){status=ENOMEM;goto done;}
  for(uint32_t k=0;k<count;k++){links[tasks[k].arc].count++;children[k]=next[k]=UINT32_MAX;}
  for(uint32_t a=0,at=0;a<arcs;a++){links[a].pending.items=pending+at;links[a].ready.items=ready+at;at+=links[a].count;}
  for(uint32_t k=0;k<count;k++) {
    if(tasks[k].parent==UINT32_MAX)mesh_write_push(&links[tasks[k].arc].pending,k,tasks,1);
    else {next[k]=children[tasks[k].parent];children[tasks[k].parent]=k;}
  }
  double now=0;
  for(uint32_t order=0;order<count;order++) {
    double begin=INFINITY;
    for(uint32_t a=0;a<arcs;a++) {
      const struct mesh_write_arc *link=links+a;
      if(link->ready.count)begin=fmin(begin,fmax(now,link->free_at));
      else if(link->pending.count)begin=fmin(begin,fmax(fmax(now,link->free_at),tasks[link->pending.items[0]].release));
    }
    if(!isfinite(begin)){status=ELOOP;goto done;}
    uint32_t best=UINT32_MAX;
    for(uint32_t a=0;a<arcs;a++) {
      struct mesh_write_arc *link=links+a;
      if(link->free_at>begin)continue;
      while(link->pending.count&&tasks[link->pending.items[0]].release<=begin)
        mesh_write_push(&link->ready,mesh_write_pop(&link->pending,tasks,1),tasks,0);
      if(link->ready.count&&(best==UINT32_MAX||mesh_write_before(tasks,link->ready.items[0],best,0)))best=link->ready.items[0];
    }
    if(best==UINT32_MAX){status=ELOOP;goto done;}
    struct mesh_write_task *t=tasks+best;
    mesh_write_pop(&links[t->arc].ready,tasks,0);
    double end=begin+t->bytes/capacity[t->arc];
    if(t->parent!=UINT32_MAX)end=fmax(end,tasks[t->parent].finish+t->first_bytes/capacity[t->arc]);
    if(!isfinite(end)){status=ERANGE;goto done;}
    t->order=order;t->start=begin;t->finish=end;links[t->arc].free_at=end;now=begin;
    for(uint32_t k=children[best];k!=UINT32_MAX;k=next[k]) {
      tasks[k].release=fmax(tasks[k].release,begin+t->first_bytes/capacity[t->arc]);
      mesh_write_push(&links[tasks[k].arc].pending,k,tasks,1);
    }
  }
done:
  free(links);free(pending);free(ready);free(children);free(next);return status;
}
#endif
