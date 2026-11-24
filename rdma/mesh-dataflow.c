#include "mesh.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

static int mesh_open(int writable,struct hdr **out,size_t *length,int *fd){
  int file=shm_open(MESH_NAME,writable?O_RDWR:O_RDONLY,0);
  if(file<0)return errno;
  struct stat info;
  if(fstat(file,&info)){int error=errno;close(file);return error;}
  if(info.st_size<(off_t)sizeof(struct hdr)){close(file);return EINVAL;}
  struct hdr *m=mmap(NULL,(size_t)info.st_size,PROT_READ|(writable?PROT_WRITE:0),MAP_SHARED,file,0);
  int error=errno;
  if(m==MAP_FAILED){close(file);return error;}
  error=m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT || m->length>(uint64_t)info.st_size?EINVAL:0;
  if(error || !fd)close(file);
  if(error){munmap(m,(size_t)info.st_size);return error;}
  *out=m;*length=(size_t)info.st_size;
  if(fd)*fd=file;
  return 0;
}

int mesh_reached(struct mesh_reach *out,uint32_t capacity,uint32_t *node){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  struct mesh_reach all[MESH_NODES];
  const uint32_t count=mesh_nodes_reached(m,all,MESH_NODES);
  for(uint32_t k=0;out && k<count && k<capacity;k++)out[k]=all[k];
  if(node)*node=m->node;
  munmap(m,length);return (int)count;
}

int mesh_busy(uint64_t *busy,uint64_t *idle_since){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  *busy=atomic_load_explicit(&m->busy,memory_order_acquire);
  *idle_since=atomic_load_explicit(&m->idle_since,memory_order_acquire);
  if(!*busy && !*idle_since)*busy=1;
  munmap(m,length);return 0;
}

int mesh_region(uint64_t *bytes,uint64_t *unleased){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  const uint32_t pages=mesh_flat_pages(m);
  const struct mesh_bmp_desc *d=&mesh_bitmaps_of(m)->pages;
  const _Atomic uint64_t *l0=mesh_bmp_l0(m,d);
  uint64_t free_pages=0;
  for(uint32_t w=0;w<d->l0_words;w++)
    free_pages+=(uint64_t)__builtin_popcountll(atomic_load_explicit(&l0[w],memory_order_relaxed));
  *bytes=(uint64_t)pages*MESH_PAGE_BYTES;
  *unleased=free_pages*MESH_PAGE_BYTES;
  munmap(m,length);return 0;
}

static int mesh_records_read(struct hdr *m,struct mesh_record **out){
  const struct mesh_map_copy *map=mesh_map_published(m);
  const uint32_t count=mesh_map_nodes(map);
  struct mesh_record *copy=malloc((count?count:1)*sizeof *copy);
  if(!copy)return -ENOMEM;
  memcpy(copy,map->record,count*sizeof *copy);
  *out=copy;return (int)count;
}

static uint32_t mesh_paired(struct mesh_record *record,uint32_t node,uint64_t *slowest){
  uint32_t cables=0;
  for(uint32_t i=0;i<record->reached && i<MESH_STREAMS;i++)
    if(record->reach[i].node==node){
      const uint64_t bandwidth=record->reach[i].bandwidth;
      if(bandwidth && (!*slowest || bandwidth<*slowest))*slowest=bandwidth;
      cables++;
    }
  return cables;
}
static uint32_t mesh_join(struct mesh_record *a,struct mesh_record *b,uint64_t *bandwidth){
  uint64_t slowest=0;
  const uint32_t ends[2]={mesh_paired(a,b->node,&slowest),mesh_paired(b,a->node,&slowest)};
  if(!ends[0] || !ends[1])return 0;
  *bandwidth=slowest;
  return ends[0]<ends[1]?ends[0]:ends[1];
}
int mesh_joined(struct mesh_record *a,struct mesh_record *b,double *beta){
  uint64_t bandwidth=0;
  const uint32_t joined=mesh_join(a,b,&bandwidth);
  if(joined)*beta=bandwidth?8e9/(double)bandwidth:0;
  return (int)joined;
}

static int mesh_node_order(const void *a,const void *b){
  uint32_t x=((const struct mesh_record *)a)->node,y=((const struct mesh_record *)b)->node;
  return (x>y)-(x<y);
}
static int mesh_records_map(struct mesh_record *r,uint32_t n,uint32_t *node,struct mesh_link_map *map){
  qsort(r,n,sizeof *r,mesh_node_order);
  uint64_t bandwidth=0;
  uint32_t links=0,pairs=0;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++){
    const uint32_t cables=mesh_join(r+a,r+b,&bandwidth);
    links+=cables;pairs+=cables>0;
  }
  if(map->nodes<n || map->links<links){map->nodes=n;map->links=links;return -ENOSPC;}
  uint32_t l=0;
  for(uint32_t a=0;a<n;a++){
    if(node)node[a]=r[a].node;
    for(uint32_t b=a+1;b<n;b++)
      for(uint32_t k=mesh_join(r+a,r+b,&bandwidth);k;k--){
        map->link[l][0]=a;map->link[l][1]=b;
        if(map->bandwidth)map->bandwidth[l]=bandwidth;
        l++;
      }
  }
  map->nodes=n;map->links=links;
  map->kind=(uint64_t)pairs*2==(uint64_t)n*(n?n-1:0)?MESH_LINKS_MESH:MESH_LINKS_GRAPH;
  return (int)n;
}

int mesh_map(uint32_t *node,struct mesh_link_map *map){
  if(!map || (map->links && !map->link))return -EINVAL;
  struct hdr *m;size_t length;struct mesh_record *r=NULL;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  const int count=mesh_records_read(m,&r);
  munmap(m,length);
  if(count<0)return count;
  const int result=mesh_records_map(r,(uint32_t)count,node,map);
  free(r);
  return result;
}

int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity){
  struct hdr *m;size_t length;struct mesh_record *r=NULL;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  const int count=mesh_records_read(m,&r);
  munmap(m,length);
  if(count<0)return count;
  for(int i=0;i<count && (uint32_t)i<capacity;i++){
    out[i].node=r[i].node;
    out[i].region=r[i].region;out[i].memory=r[i].memory;
    memcpy(out[i].model,r[i].model,MESH_MODEL_BYTES);out[i].model[MESH_MODEL_BYTES-1]=0;
    out[i].health=r[i].health;
  }
  free(r);
  return count;
}

int mesh_map_serial(uint64_t *serial){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return error;
  const uint64_t seen=atomic_load_explicit(&m->map,memory_order_acquire);
  if(__atomic_load_n(&m->version,__ATOMIC_ACQUIRE)!=MESH_LAYOUT)error=ESRCH;
  else *serial=seen;
  munmap(m,length);
  return error;
}

int mesh_attach(struct mesh_ctx *c){
  if(!c)return EINVAL;
  if(c->M)return 0;
  struct hdr *m;size_t length;int file;
  const int error=mesh_open(1,&m,&length,&file);
  if(error)return error;
  *c=(struct mesh_ctx){.M=m,.len=length,.fd=file};
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c || !c->M)return 0;
  int error=(c->fd>=0 && munmap(c->M,c->len))?errno:0;
  if(c->fd>=0 && close(c->fd) && !error)error=errno;
  *c=(struct mesh_ctx){.M=NULL,.len=0,.fd=-1};
  return error;
}
