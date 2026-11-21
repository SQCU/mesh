/* mesh-switch-log: the switch's records for the routing policy's trainer (design/switched-multipath-dma.md;
   rdma/tests/route), drained off the bridge's path.  It turns the switch's logging on, takes every process's ring
   (mesh-switched.h mesh_switch_log_take: each port child's, the parent's; a ring full when the bridge writes drops the
   record and counts it, so no transfer ever waits on this tool) and appends the records to shard files in DIRECTORY:
   - draws.NNNNNN: struct mesh_switch_record of kind MESH_LOG_DRAW, raw: each its split.h struct route_draw (each
     (destination, out-lane)'s arrival estimate, validity, the allowed mask, the sampled split as its token) and its
     behaviour log-probability `logp`, with the node, program key, execution, source, transfer, span and bytes;
   - pieces.NNNNNN: kind MESH_LOG_PIECE, each piece delivered: its latency and queueing over its floor (ns);
   - executions.NNNNNN: kind MESH_LOG_EXECUTION, made here from the pieces: for each (program key, execution) whose
     pieces stopped arriving for a second, the span of this node's clock from its first piece delivered to its last
     (`latency`), its mean queueing, its floor, its pieces and bytes (the transaction's cost at this node).
   A shard is closed at SHARD bytes (default 64 MiB) and the next opened.  SIGTERM or SIGINT turns logging off, writes
   what is pending and ends.

   usage: mesh-switch-log DIRECTORY [SHARD_BYTES]   (MESH_REGION names another region, as the clients' attach) */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "mesh-switched.h"

static volatile sig_atomic_t stopping;
static void stop(int signal){ (void)signal; stopping=1; }
struct shard { const char *kind; FILE *f; uint64_t bytes,index; };
static const char *directory;static uint64_t limit=UINT64_C(64)<<20;
static void put(struct shard *s,const struct mesh_switch_record *r){
  if(!s->f || s->bytes>=limit){
    if(s->f)fclose(s->f);
    char path[4096];snprintf(path,sizeof path,"%s/%s.%06llu",directory,s->kind,(unsigned long long)s->index++);
    if(!(s->f=fopen(path,"wb"))){perror(path);exit(1);}
    s->bytes=0;
  }
  fwrite(r,sizeof *r,1,s->f);s->bytes+=sizeof *r;
}
/* the executions being gathered from their pieces */
struct gathering { uint64_t key,execution,first,last,seen,pieces,bytes; double queueing,floor; uint32_t node; };
static struct gathering open_[4096];static uint32_t opened;
static uint64_t clock_ns(void){ struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec; }
static void gathered(struct shard *s,const struct gathering *g){
  struct mesh_switch_record r;memset(&r,0,sizeof r);
  r.kind=MESH_LOG_EXECUTION;r.node=g->node;r.key=g->key;r.execution=g->execution;r.at=g->last;
  r.latency=(float)(g->last-g->first);r.queueing=(float)(g->queueing/(double)g->pieces);r.floor=(float)g->floor;
  r.span=(uint32_t)g->pieces;r.bytes=(uint32_t)(g->bytes>UINT32_MAX?UINT32_MAX:g->bytes);
  put(s,&r);
}
static void gather(struct shard *s,const struct mesh_switch_record *r,uint64_t now){
  uint32_t k=0;
  while(k<opened && (open_[k].key!=r->key || open_[k].execution!=r->execution))k++;
  if(k==opened){
    if(opened==sizeof open_/sizeof *open_){gathered(s,open_);memmove(open_,open_+1,--opened*sizeof *open_);k=opened;}
    open_[opened++]=(struct gathering){r->key,r->execution,r->at,r->at,now,0,0,0,0,r->node};
  }
  struct gathering *g=open_+k;
  if(r->at<g->first)g->first=r->at;
  if(r->at>g->last)g->last=r->at;
  g->seen=now;g->pieces++;g->bytes+=r->bytes;g->queueing+=r->queueing;if(r->floor>g->floor)g->floor=r->floor;
}
static void settle(struct shard *s,uint64_t now,int all){
  uint32_t kept=0;
  for(uint32_t k=0;k<opened;k++){
    if(all || now-open_[k].seen>UINT64_C(1000000000))gathered(s,open_+k);
    else open_[kept++]=open_[k];
  }
  opened=kept;
}
int main(int argc,char **argv){
  if(argc<2){fprintf(stderr,"usage: mesh-switch-log DIRECTORY [SHARD_BYTES]\n");return 2;}
  directory=argv[1];
  if(argc>2)limit=strtoull(argv[2],NULL,0);
  mkdir(directory,0755);
  int fd=shm_open(mesh_name(),O_RDWR,0);
  if(fd<0){perror("shm_open");return 1;}
  struct stat st;
  if(fstat(fd,&st) || st.st_size<(off_t)sizeof(struct hdr)){close(fd);return 2;}
  struct hdr *m=mmap(NULL,(size_t)st.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
  if(m==MAP_FAILED){perror("mmap");return 3;}
  if(m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT){fprintf(stderr,"mesh-switch-log: not a region of layout %u\n",MESH_LAYOUT);return 4;}
  struct mesh_switch *sw=mesh_switch_of(m);
  if(sw->magic!=MESH_SWITCH_MAGIC){fprintf(stderr,"mesh-switch-log: node %u has no switch\n",m->node);return 5;}
  signal(SIGTERM,stop);signal(SIGINT,stop);
  struct shard draws={"draws",NULL,0,0},pieces={"pieces",NULL,0,0},executions={"executions",NULL,0,0};
  atomic_store(&sw->logging,1);
  uint64_t taken=0,settled=clock_ns();
  while(!stopping){
    int any=0;struct mesh_switch_record r;
    const uint64_t now=clock_ns();
    for(uint32_t g=0;g<MESH_SWITCH_LOGS;g++)
      while(mesh_switch_log_take(sw,g,&r)){
        any=1;taken++;
        if(r.kind==MESH_LOG_DRAW)put(&draws,&r);
        else if(r.kind==MESH_LOG_PIECE){put(&pieces,&r);gather(&executions,&r,now);}
      }
    if(now-settled>UINT64_C(100000000)){settle(&executions,now,0);settled=now;}
    if(!any){struct timespec pause={0,1000000};nanosleep(&pause,NULL);}
  }
  atomic_store(&sw->logging,0);
  struct mesh_switch_record r;
  for(uint32_t g=0;g<MESH_SWITCH_LOGS;g++)while(mesh_switch_log_take(sw,g,&r)){
    taken++;if(r.kind==MESH_LOG_DRAW)put(&draws,&r);else if(r.kind==MESH_LOG_PIECE){put(&pieces,&r);gather(&executions,&r,clock_ns());}
  }
  settle(&executions,clock_ns(),1);
  uint64_t dropped=0;for(uint32_t g=0;g<MESH_SWITCH_LOGS;g++)dropped+=atomic_load(&sw->log[g].dropped);
  for(struct shard *s=&draws;s;s=s==&draws?&pieces:s==&pieces?&executions:NULL)if(s->f)fclose(s->f);
  fprintf(stderr,"mesh-switch-log: node %u: %llu records taken, %llu dropped by the bridge (rings full)\n",m->node,
          (unsigned long long)taken,(unsigned long long)dropped);
  return 0;
}
