/* mesh-cluster: the whole cluster's fabric as this node's region holds it, one JSON object, read-only.
   The cluster part is the flooded link-state of every node with a bridge (mesh.h struct mesh_record: each bridge
   floods its own record, the newest of which every bridge keeps; a record's reach is an entry a cable, the peer node
   and the bandwidth that end reports), so it names nodes that are no communicator's member (a relay-only node) and is
   the same on every node once the floods settle: `records` (each node's boot, sequence, region, memory, model and
   cables) and `links` (every cable the two ends both report, each pair's count the fewer of the two ends' reports,
   its bandwidth the lesser, as mesh_map takes it; `one_sided` the reports the other end does not match).  The local
   part is this node's own: its ports (peer, phase, bandwidth, pairings) and its program slots (toward each node: live,
   generation, its own cables' bandwidth, its programs).  Run it on every node and compare `records` and `links`.
   usage: mesh-cluster   (MESH_REGION names another region, as the clients' attach) */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "mesh-observe.h"
static unsigned long long u(uint64_t v){ return (unsigned long long)v; }
int main(void){
  int fd=shm_open(mesh_name(),O_RDONLY,0);
  if(fd<0){printf("{\"up\":false,\"errno\":%d}\n",errno);return 0;}
  struct stat st;
  if(fstat(fd,&st) || st.st_size<(off_t)sizeof(struct hdr)){close(fd);printf("{\"up\":false,\"errno\":%d}\n",EINVAL);return 0;}
  struct hdr *m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0);close(fd);
  if(m==MAP_FAILED){printf("{\"up\":false,\"errno\":%d}\n",errno);return 0;}
  if(m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT){printf("{\"up\":false,\"layout\":%u}\n",m->version);return 0;}
  /* a whole copy of the records: the map's serial even before and unchanged after (mesh.h mesh_map_whole) */
  struct mesh_record *rec=NULL;uint64_t n=0;
  for(int tries=0;tries<1000;tries++){
    const uint64_t before=atomic_load_explicit(&m->map,memory_order_acquire);
    if(before&1)continue;
    n=atomic_load_explicit(&m->map_nodes,memory_order_relaxed);
    if(atomic_load_explicit(&m->map_off,memory_order_relaxed)+n*sizeof *rec>(uint64_t)st.st_size){n=0;break;}
    free(rec);rec=malloc((n+1)*sizeof *rec);
    if(!rec)return 1;
    memcpy(rec,mesh_records(m),n*sizeof *rec);
    if(mesh_map_whole(m,before))break;
    n=0;
  }
  printf("{\"up\":true,\"node\":%u,\"records\":[",m->node);
  for(uint64_t i=0;i<n;i++){
    const struct mesh_record *r=rec+i;
    printf("%s{\"node\":%u,\"boot\":%llu,\"sequence\":%llu,\"region\":%llu,\"memory\":%llu,\"model\":\"",i?",":"",r->node,u(r->boot),
           u(r->sequence),u(r->region),u(r->memory));
    for(const char *c=r->model;*c && c<r->model+MESH_MODEL_BYTES;c++)printf(*c=='"' || *c=='\\'?"\\%c":"%c",*c);
    printf("\",\"cables\":[");
    for(uint32_t k=0;k<r->reached && k<MESH_STREAMS;k++)
      printf("%s{\"peer\":%u,\"bandwidth\":%llu}",k?",":"",r->reach[k].node,u(r->reach[k].bandwidth));
    printf("]}");
  }
  /* the links: each unordered pair's cables as both ends report them */
  printf("],\"links\":[");
  int first=1;
  for(uint64_t i=0;i<n;i++)for(uint64_t j=i+1;j<n;j++){
    const struct mesh_record *a=rec+i,*b=rec+j;
    uint32_t na=0,nb=0;uint64_t bwa=UINT64_MAX,bwb=UINT64_MAX;
    for(uint32_t k=0;k<a->reached && k<MESH_STREAMS;k++)if(a->reach[k].node==b->node){na++;if(a->reach[k].bandwidth<bwa)bwa=a->reach[k].bandwidth;}
    for(uint32_t k=0;k<b->reached && k<MESH_STREAMS;k++)if(b->reach[k].node==a->node){nb++;if(b->reach[k].bandwidth<bwb)bwb=b->reach[k].bandwidth;}
    if(!na && !nb)continue;
    const uint32_t both=na<nb?na:nb;const uint64_t bw=bwa<bwb?bwa:bwb;
    printf("%s{\"nodes\":[%u,%u],\"cables\":%u,\"bandwidth\":%llu,\"one_sided\":%u}",first?"":",",a->node,b->node,both,
           u(both?bw:0),na>nb?na-nb:nb-na);
    first=0;
  }
  printf("],\"ports\":[");
  static const char *const phase_name[]={[MESH_UNKNOWN]="UNKNOWN",[MESH_PAIRING]="PAIRING",[MESH_PAIRED]="PAIRED",[MESH_STOPPED]="STOPPED"};
  for(uint32_t p=0;p<MESH_PORTS;p++){
    const struct mesh_link_info *l=mesh_links(m)+p;
    const uint32_t phase=atomic_load_explicit(&l->phase,memory_order_acquire);
    if(phase==MESH_UNKNOWN)break;
    printf("%s{\"port\":%u,\"peer\":%u,\"phase\":\"%s\",\"bandwidth\":%llu,\"pairings\":%llu}",p?",":"",p,atomic_load(&l->peer),
           phase<=MESH_STOPPED?phase_name[phase]:"?",u(atomic_load(&l->bandwidth)),u(atomic_load(&l->pairings)));
  }
  printf("],\"orders\":[");
  first=1;
  for(uint32_t k=0;k<MESH_NODES;k++){
    const struct mesh_node *x=mesh_node_at(m,k);
    const uint32_t node=atomic_load(&x->node);
    if(!node)continue;
    uint32_t programs=0;
    for(uint32_t p=0;p<MESH_PROGRAMS;p++)programs+=atomic_load(&x->program[p].owner)!=0;
    printf("%s{\"peer\":%u,\"live\":%u,\"generation\":%llu,\"bandwidth\":%llu,\"programs\":%u}",first?"":",",node,atomic_load(&x->live),
           u(atomic_load(&x->generation)),u(atomic_load(&x->bandwidth)),programs);
    first=0;
  }
  printf("]}\n");
  free(rec);munmap(m,(size_t)st.st_size);
  return 0;
}
