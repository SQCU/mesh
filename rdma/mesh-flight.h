#ifndef MESH_FLIGHT_H
#define MESH_FLIGHT_H
#include "mesh.h"
#include <stdint.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define FLIGHT_CAPACITY 512
#define FLIGHT_OPS(X) X(NONE) X(OPEN_DEVICE) X(QUERY_PORT) X(ALLOC_PD) X(REG_MR) X(CREATE_CQ) X(CREATE_QP) X(INIT) X(QUERY_GID) X(RTR) X(RTS) X(POST_RECV) X(POST_SEND) X(COMPLETION_ERROR) X(DESTROY_QP) X(DESTROY_CQ) X(DEREG_MR) X(DEALLOC_PD) X(CLOSE_DEVICE) X(PAIR_UP) X(PAIR_DOWN) X(POLL_CQ)
#define FLIGHT_ENUM(x) F_##x,
enum { FLIGHT_OPS(FLIGHT_ENUM) };
#define FLIGHT_NAME(x) #x "\0"
static const char flight_names[]=FLIGHT_OPS(FLIGHT_NAME);
struct flight_event { uint64_t sequence, time_us, object, argument, bytes; int64_t result; int32_t error; uint32_t operation; };
struct flight_log { uint32_t magic, version, event_size, capacity; uint64_t pid, node, span, sequence, heartbeat_us, cursor[2]; char operations[1024]; struct flight_event events[2*FLIGHT_CAPACITY]; };
static _Thread_local struct flight_log *flight;
static _Thread_local struct hdr *flight_status;
static uint64_t flight_time(void){ struct timeval t; gettimeofday(&t,0); return (uint64_t)t.tv_sec*1000000+t.tv_usec; }
static void flight_open(const char *name, int node, uint64_t span){
  const char *dir=getenv("MESH_LOG_DIR"); char fallback[1024],path[2048];
  if(!dir){ snprintf(fallback,sizeof fallback,"%s/.mesh-logs",getenv("HOME")?getenv("HOME"):"/var/tmp"); dir=fallback; }
  if(mkdir(dir,0755) && errno!=EEXIST) fprintf(stderr,"flight directory %s: %s\n",dir,strerror(errno));
  snprintf(path,sizeof path,"%s/%s-%llu-%d.flight",dir,name[0]=='/'?name+1:name,(unsigned long long)flight_time(),getpid());
  int fd=open(path,O_CREAT|O_EXCL|O_RDWR,0644);
  if(fd>=0 && !ftruncate(fd,sizeof *flight)){
    void *p=mmap(0,sizeof *flight,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(p!=MAP_FAILED){ flight=p; flight->version=1; flight->event_size=sizeof(struct flight_event); flight->capacity=2*FLIGHT_CAPACITY;
      flight->pid=getpid(); flight->node=node; flight->span=span;
      memcpy(flight->operations,flight_names,sizeof flight_names); flight->magic=0x4d464c54;
      fprintf(stderr,"flight recording %s\n",path); } }
  if(!flight) fprintf(stderr,"flight recording unavailable %s: %s\n",path,strerror(errno));
  if(fd>=0) close(fd); }
static struct flight_event *flight_begin(uint32_t op, uintptr_t object, uint64_t argument, uint64_t bytes){
  uint64_t stamp=flight_time();
  if(flight_status){ atomic_store_explicit(&flight_status->operation_ms,stamp/1000,memory_order_relaxed); atomic_store_explicit(&flight_status->operation,op,memory_order_release); }
  if(!flight) return 0;
  uint64_t seq=flight->sequence+1;
  int bank=op==F_POST_SEND || op==F_POST_RECV || op==F_POLL_CQ;
  struct flight_event *e=&flight->events[bank*FLIGHT_CAPACITY+flight->cursor[bank]++%FLIGHT_CAPACITY];
  __atomic_store_n(&e->sequence,0,__ATOMIC_RELEASE);
  e->time_us=stamp; e->object=object; e->argument=argument; e->bytes=bytes;
  e->operation=op; e->error=0; e->result=INT64_MIN;
  __atomic_store_n(&e->sequence,seq,__ATOMIC_RELEASE);
  __atomic_store_n(&flight->sequence,seq,__ATOMIC_RELEASE);
  return e; }
static void flight_end(struct flight_event *e, int64_t result, int error){
  if(e){ e->error=error; __atomic_store_n(&e->result,result,__ATOMIC_RELEASE); }
  if(flight_status) atomic_store_explicit(&flight_status->operation,0,memory_order_release); }
static void flight_heartbeat(uint64_t phase){
  uint64_t stamp=flight_time();
  if(flight) __atomic_store_n(&flight->heartbeat_us,stamp,__ATOMIC_RELEASE);
  if(flight_status){ atomic_store_explicit(&flight_status->phase,phase,memory_order_relaxed);
    atomic_store_explicit(&flight_status->heartbeat_ms,stamp/1000,memory_order_release); }
}
#define TRACE(op,object,argument,bytes,call) ({ struct flight_event *event_=flight_begin(F_##op,(uintptr_t)(object),(argument),(bytes)); errno=0; __auto_type result_=(call); int error_=errno; flight_end(event_,(intptr_t)result_,error_); errno=error_; result_; })
#endif
