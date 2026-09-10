
#include "mesh.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <string.h>
// ../design/algorithm-sources.md#transport-page-addressing
int main(int argc,char**argv){
  static const char *N[NOWN]={"free","recv","send","app"};
  static const char *Q[NRING]={"sub","cmp","rel"};
  int readiness=argc>1 && !strcmp(argv[1],"--ready");
  const char *name=argc>1+readiness?argv[1+readiness]:getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int f=shm_open(name,O_RDONLY,MESH_MODE);
  struct stat info;
  struct hdr *h = f<0 || fstat(f,&info) || info.st_size<(off_t)sizeof(struct hdr)?MAP_FAILED:mmap(NULL,sizeof *h,PROT_READ,MAP_SHARED,f,0);
  if(h==MAP_FAILED||h->magic!=MESH_MAGIC||h->version!=MESH_VERSION){ printf("{\"up\":false}\n"); return readiness?1:0; }
  #define A(x) (unsigned long long)atomic_load_explicit(&h->x,memory_order_relaxed)
  uint64_t client=atomic_load_explicit(&h->client,memory_order_relaxed);
  int alive=client && (!kill((pid_t)client,0)||errno==EPERM);
  uint64_t bridge=A(bridge_pid),beat=A(heartbeat_ms),phase=A(phase),op=A(operation);
  int bridge_alive=bridge && (!kill((pid_t)bridge,0)||errno==EPERM);
  if(readiness){
    uint64_t count=A(port_count);
    size_t base=RINGS+NRING*MESH_RING*sizeof(struct desc);
    int ready=bridge_alive && count && h->data_off>=base && count<=(h->data_off-base)/sizeof(struct mesh_port_info);
    size_t length=base+count*sizeof(struct mesh_port_info);
    if(ready && length<=(size_t)info.st_size){
      struct hdr *mapped=mmap(NULL,length,PROT_READ,MAP_SHARED,f,0);
      ready=mapped!=MAP_FAILED;
      if(ready){
        struct mesh_port_info *ports=mesh_ports(mapped);
        for(uint64_t i=0;i<count;i++) ready &= atomic_load_explicit(&ports[i].phase,memory_order_acquire)==MESH_PAIRED;
        munmap(mapped,length);
      }
    } else ready=0;
    printf("{\"ready\":%s,\"version\":%u,\"node\":%u,\"pgsz\":%u}\n",ready?"true":"false",h->version,h->node,h->pgsz);
    munmap(h,sizeof *h); close(f); return ready?0:1;
  }
  struct timeval tv; gettimeofday(&tv,0);
  uint64_t stamp=(uint64_t)tv.tv_sec*1000+tv.tv_usec/1000;
  uint64_t age=stamp>beat?stamp-beat:0,op_time=A(operation_ms);
  static const char *phases[]={"unknown","pairing","paired","retiring","stopping","stopped"};
  printf("{\"up\":true,\"region\":\"%s\",\"node\":%u,\"pgsz\":%u,\"pool\":%u,\"arena\":%u,"
    "\"client\":%llu,\"client_alive\":%s,\"sent\":%llu,\"recvd\":%llu,\"bad\":%llu,\"up_ms\":%llu",
    name,h->node,h->pgsz,h->pool,h->arena,(unsigned long long)client,alive?"true":"false",A(sent),A(recvd),A(bad),A(up_ms));
  printf(",\"bridge_pid\":%llu,\"bridge_alive\":%s,\"phase\":\"%s\",\"heartbeat_age_ms\":%llu,\"responsive\":%s,\"paired\":%s,\"operation\":%llu,\"operation_age_ms\":%llu",
    (unsigned long long)bridge,bridge_alive?"true":"false",phase<=MESH_STOPPED?phases[phase]:"unknown",
    (unsigned long long)age,bridge?(bridge_alive&&age<3000?"true":"false"):"null",
    bridge?(bridge_alive&&age<3000&&phase==MESH_PAIRED?"true":"false"):"null",
    (unsigned long long)op,(unsigned long long)(op&&stamp>op_time?stamp-op_time:0));
  for(int i=0;i<NRING;i++){
    unsigned long long head=atomic_load_explicit(&h->r[i].head,memory_order_relaxed);
    unsigned long long tail=atomic_load_explicit(&h->r[i].tail,memory_order_relaxed);
    printf(",\"%s_head\":%llu,\"%s_tail\":%llu,\"%s_depth\":%llu",Q[i],head,Q[i],tail,Q[i],head-tail); }
  for(int i=0;i<NOWN;i++) printf(",\"%s\":%llu,\"sd_%s\":%llu",N[i],A(mean[i]),N[i],A(sd[i]));
  uint64_t count=A(port_count);
  if(count && count<=(h->data_off-(RINGS+NRING*MESH_RING*sizeof(struct desc)))/sizeof(struct mesh_port_info)){
    size_t length=(size_t)h->data_off;
    struct hdr *mapped=mmap(NULL,length,PROT_READ,MAP_SHARED,f,0);
    if(mapped!=MAP_FAILED){
      struct mesh_port_info *ports=mesh_ports(mapped); printf(",\"ports\":[");
      for(uint64_t i=0;i<count;i++){
        uint64_t state=atomic_load(&ports[i].phase);
        printf("%s{\"device\":\"%.31s\",\"peer\":%u,\"generation\":%llu,\"phase\":\"%s\",\"heartbeat_us\":%llu,\"operation\":%llu}",
          i?",":"",ports[i].device,ports[i].peer,(unsigned long long)atomic_load(&ports[i].generation),
          state<=MESH_STOPPED?phases[state]:"unknown",(unsigned long long)atomic_load(&ports[i].heartbeat_us),
          (unsigned long long)atomic_load(&ports[i].operation));
      }
      printf("]"); munmap(mapped,length);
    }
  }
  printf("}\n"); return 0; }
