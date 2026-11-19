#include "../mesh-peer.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

struct storage {
  struct mesh_peer_transfer first,second[2];
  struct mesh_dma_span spans[3];
  _Atomic uint64_t release_first,release_reply,complete[3];
  unsigned char payload[3][3];
};

static int wait_word(const _Atomic uint64_t *word){
  const uint64_t due=mesh_clock_ns()+MESH_PLATFORM_NS;
  for(;;){
    uint64_t value=atomic_load_explicit(word,memory_order_acquire);
    if(value==UINT64_MAX)return EIO;
    if(value==1)return 0;
    if(mesh_clock_ns()>=due)return ETIMEDOUT;
    __asm__ volatile("yield");
  }
}

static int sample(struct mesh_ctx *ctx,uint32_t peer,uint64_t group,int receiver){
  struct mesh_section lease;int status=mesh_section_create(ctx,sizeof(struct storage),1,&lease);
  if(status)return status;
  struct storage *s=mesh_section_address(ctx,lease,0);
  memset(s,0,sizeof *s);uint64_t base=(uint64_t)((char *)s-(char *)ctx->M);
  uint64_t handles[2]={0},start=0,elapsed=0;
  const unsigned char expected[]={0x9d,0xa7,0x31};
  for(unsigned i=0;i<3;i++){
    int land=i==2?!receiver:receiver;
    s->payload[i][0]=0xab;s->payload[i][1]=land?0:expected[i];s->payload[i][2]=0xcd;
    s->spans[i]=(struct mesh_dma_span){base+offsetof(struct storage,payload)+3*i+1,1};
    uint64_t gate=land || i==1?0:base+(i?offsetof(struct storage,release_reply):offsetof(struct storage,release_first));
    struct mesh_peer_transfer row={i,base+offsetof(struct storage,spans)+i*sizeof *s->spans,1,gate,1,
      base+offsetof(struct storage,complete)+8*i,1,(uint32_t)land,0};
    if(!i)s->first=row;else s->second[i-1]=row;
  }
  status=mesh_peer_install(ctx,peer,group,base+offsetof(struct storage,first),1,handles);
  if(!status)status=mesh_peer_wait(ctx,handles[0],MESH_PLATFORM_NS);
  if(!status)status=mesh_peer_install(ctx,peer,group+1,base+offsetof(struct storage,second),2,handles+1);
  if(!status)status=mesh_peer_wait(ctx,handles[1],MESH_PLATFORM_NS);
  if(!status)status=wait_word(s->complete+1);
  if(!status && (atomic_load(s->complete) || s->payload[1][1]!=expected[1]))status=EILSEQ;
  if(!status && receiver)atomic_store_explicit(&s->release_reply,1,memory_order_release);
  if(!status)status=wait_word(s->complete+2);
  if(!status && s->payload[2][1]!=expected[2])status=EILSEQ;
  if(!status && !receiver){
    start=mesh_clock_ns();atomic_store_explicit(&s->release_first,1,memory_order_release);
  }
  if(!status)status=wait_word(s->complete);
  if(!status && !receiver)elapsed=mesh_clock_ns()-start;
  for(unsigned i=0;!status && i<3;i++)if(s->payload[i][0]!=0xab || s->payload[i][1]!=expected[i] || s->payload[i][2]!=0xcd)status=EILSEQ;
  for(int i=1;i>=0;i--)if(handles[i]){
    int released=mesh_peer_release(ctx,handles[i]);if(!status)status=released;
  }
  int returned=mesh_lease_return(ctx,lease.first);if(!status)status=returned;
  if(!status)printf("{\"role\":\"%s\",\"group\":%" PRIu64 ",\"second_completed_before_first\":true,\"guards\":true,\"first_source_retire_ns\":%" PRIu64 "}\n",
                    receiver?"receive":"send",group,elapsed);
  return status;
}

int main(int argc,char **argv){
  if(argc!=4 || (strcmp(argv[1],"send") && strcmp(argv[1],"receive"))){
    fprintf(stderr,"usage: %s send|receive PEER_NODE GROUP\n",argv[0]);return 2;
  }
  struct mesh_ctx ctx={.fd=-1};int status=mesh_attach(&ctx);
  uint32_t peer=(uint32_t)strtoul(argv[2],NULL,0);uint64_t group=strtoull(argv[3],NULL,0);
  if(!status)status=mesh_peer_capable(&ctx,peer);
  if(!status)status=sample(&ctx,peer,group,!strcmp(argv[1],"receive"));
  if(ctx.M)mesh_detach(&ctx);
  if(status)fprintf(stderr,"%s: %s (%d)\n",argv[0],strerror(status),status);
  return status?1:0;
}
