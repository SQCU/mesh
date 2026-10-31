/* Preparation ordering of page transfers and their receive/combine events.
   One declared dependency graph and the directed cable capacities determine
   every interval. Execution consumes the resulting order, never this graph. */
#ifndef MESH_WRITE_ORDER_H
#define MESH_WRITE_ORDER_H
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

struct mesh_write_task {
  uint32_t arc, order; /* UINT32_MAX is a receive/combine event, not a cable use. */
  uint64_t production;
  double bytes, release, start, finish, first_bytes, first, tail;
};
enum mesh_write_dependency_kind { MESH_WRITE_PUBLISHED, MESH_WRITE_STREAMED, MESH_WRITE_COMPLETE };
struct mesh_write_dependency { uint32_t before, after; enum mesh_write_dependency_kind kind; };

/* These heaps belong to compilation. A task becomes eligible only when every
   declared predecessor has an interval. A combine can therefore have any
   number of upstream contributors, while a relay retains first-page release. */
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
static int mesh_write_order(struct mesh_write_task *tasks,uint32_t count,
                            const struct mesh_write_dependency *edges,size_t edge_count,
                            const double *capacity,uint32_t arcs) {
  for(uint32_t k=0;k<count;k++) {
    struct mesh_write_task *t=tasks+k;t->order=UINT32_MAX;t->tail=0;
    if((t->arc!=UINT32_MAX&&t->arc>=arcs)||!isfinite(t->release)||t->release<0)return EINVAL;
    if(t->arc!=UINT32_MAX&&(!(capacity[t->arc]>0)||!isfinite(capacity[t->arc])||
       !(t->bytes>0)||!isfinite(t->bytes)||!(t->first_bytes>0)||
       t->first_bytes>t->bytes||!isfinite(t->bytes/capacity[t->arc])))return EINVAL;
  }
  struct mesh_write_arc { struct mesh_write_heap pending,ready; uint32_t count; double free_at; };
  struct mesh_write_arc *links=calloc((size_t)arcs+1,sizeof *links);
  uint32_t *pending=malloc(((size_t)count+1)*sizeof *pending),*ready=malloc(((size_t)count+1)*sizeof *ready),
           *waiting=calloc((size_t)count+1,sizeof *waiting);
  size_t *first=calloc((size_t)count+1,sizeof *first),*at=calloc((size_t)count+1,sizeof *at),
         *children=malloc((edge_count+1)*sizeof *children);
  int status=0;
  if(!links||!pending||!ready||!waiting||!first||!at||!children){status=ENOMEM;goto done;}
  for(uint32_t k=0;k<count;k++)links[tasks[k].arc==UINT32_MAX?arcs:tasks[k].arc].count++;
  for(size_t k=0;k<edge_count;k++) {
    const struct mesh_write_dependency *e=edges+k;
    if(e->before>=count||e->after>=count||e->before==e->after||e->kind>MESH_WRITE_COMPLETE){status=EINVAL;goto done;}
    if(waiting[e->after]==UINT32_MAX){status=EOVERFLOW;goto done;}
    first[e->before+1]++;waiting[e->after]++;
  }
  for(uint32_t k=0;k<count;k++){first[k+1]+=first[k];at[k]=first[k];}
  for(size_t k=0;k<edge_count;k++)children[at[edges[k].before]++]=k;
  for(uint32_t a=0,offset=0;a<=arcs;a++) {
    links[a].pending.items=pending+offset;links[a].ready.items=ready+offset;offset+=links[a].count;
  }
  for(uint32_t k=0;k<count;k++)if(!waiting[k])
    mesh_write_push(&links[tasks[k].arc==UINT32_MAX?arcs:tasks[k].arc].pending,k,tasks,1);
  double now=0;
  for(uint32_t order=0;order<count;order++) {
    double begin=INFINITY;
    for(uint32_t a=0;a<=arcs;a++) {
      const struct mesh_write_arc *link=links+a;
      if(link->ready.count)begin=fmin(begin,fmax(now,link->free_at));
      else if(link->pending.count)begin=fmin(begin,fmax(fmax(now,link->free_at),tasks[link->pending.items[0]].release));
    }
    if(!isfinite(begin)){status=EDEADLK;goto done;}
    uint32_t best=UINT32_MAX;
    for(uint32_t a=0;a<=arcs;a++) {
      struct mesh_write_arc *link=links+a;
      if(link->free_at>begin)continue;
      while(link->pending.count&&tasks[link->pending.items[0]].release<=begin)
        mesh_write_push(&link->ready,mesh_write_pop(&link->pending,tasks,1),tasks,0);
      if(link->ready.count&&(best==UINT32_MAX||mesh_write_before(tasks,link->ready.items[0],best,0)))best=link->ready.items[0];
    }
    if(best==UINT32_MAX){status=EDEADLK;goto done;}
    struct mesh_write_task *t=tasks+best;
    const uint32_t arc=t->arc==UINT32_MAX?arcs:t->arc;
    mesh_write_pop(&links[arc].ready,tasks,0);
    const double duration=arc==arcs?0:t->bytes/capacity[arc];
    const double lead=arc==arcs?0:t->first_bytes/capacity[arc];
    const double end=fmax(begin+duration,t->tail);
    if(!isfinite(end)){status=ERANGE;goto done;}
    t->order=order;t->start=begin;t->first=begin+lead;t->finish=end;
    if(arc!=arcs)links[arc].free_at=end;
    now=begin;
    for(size_t k=first[best];k<first[best+1];k++) {
      const struct mesh_write_dependency *edge=edges+children[k];
      struct mesh_write_task *child=tasks+edge->after;
      const double release=edge->kind==MESH_WRITE_PUBLISHED?t->start:
                           edge->kind==MESH_WRITE_STREAMED?t->first:t->finish;
      child->release=fmax(child->release,release);
      if(edge->kind==MESH_WRITE_STREAMED)child->tail=fmax(child->tail,t->finish+
        (child->arc==UINT32_MAX?0:child->first_bytes/capacity[child->arc]));
      if(!--waiting[edge->after])mesh_write_push(
        &links[child->arc==UINT32_MAX?arcs:child->arc].pending,edge->after,tasks,1);
    }
  }
done:
  free(links);free(pending);free(ready);free(waiting);free(first);free(at);free(children);return status;
}
#endif
