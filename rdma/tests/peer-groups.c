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

static int sample(struct mesh_ctx *ctx,uint32_t peer,uint64_t group,int receiver,int receive_gate,size_t held_bytes){
  const size_t bytes[]={held_bytes,receive_gate?16384:1,1};
  struct mesh_section lease;int status=mesh_section_create(ctx,sizeof(struct storage)+bytes[0]+bytes[1]+4,1,&lease);
  if(status)return status;
  struct storage *s=mesh_section_address(ctx,lease,0);
  memset(s,0,sizeof *s);uint64_t base=(uint64_t)((char *)s-(char *)ctx->M);
  uint64_t handles[2]={0},start=0,elapsed=0;
  const unsigned char expected[]={0x9d,0xa7,0x31};
  unsigned char *first=(unsigned char *)(s+1);
  unsigned char *payload[]={first,first+bytes[0]+2,s->payload[2]};
  for(unsigned i=0;i<3;i++){
    int land=i==2?!receiver:receiver;
    payload[i][0]=0xab;memset(payload[i]+1,land?0:expected[i],bytes[i]);payload[i][bytes[i]+1]=0xcd;
    s->spans[i]=(struct mesh_dma_span){(uint64_t)(payload[i]+1-(unsigned char *)ctx->M),bytes[i]};
    uint64_t gate=0;
    if(i==0 && receiver==receive_gate)gate=base+offsetof(struct storage,release_first);
    if(i==2 && receiver)gate=base+offsetof(struct storage,release_reply);
    struct mesh_peer_transfer row={i,base+offsetof(struct storage,spans)+i*sizeof *s->spans,1,gate,1,
      base+offsetof(struct storage,complete)+8*i,1,(uint32_t)land,0};
    if(!i)s->first=row;else s->second[i-1]=row;
  }
  status=mesh_peer_install(ctx,peer,group,base+offsetof(struct storage,first),1,handles);
  if(!status)status=mesh_peer_wait(ctx,handles[0],MESH_PLATFORM_NS);
  if(!status)status=mesh_peer_install(ctx,peer,group+1,base+offsetof(struct storage,second),2,handles+1);
  if(!status)status=mesh_peer_wait(ctx,handles[1],MESH_PLATFORM_NS);
  if(!status)status=wait_word(s->complete+1);
  if(!status && ((receiver==receive_gate && atomic_load(s->complete)) || payload[1][1]!=expected[1]))status=EILSEQ;
  if(receiver && receive_gate)for(size_t i=1;!status && i<=held_bytes;i++)if(payload[0][i])status=EILSEQ;
  if(!status && receiver)atomic_store_explicit(&s->release_reply,1,memory_order_release);
  if(!status)status=wait_word(s->complete+2);
  if(!status && s->payload[2][1]!=expected[2])status=EILSEQ;
  if(!status && receiver==receive_gate){
    start=mesh_clock_ns();atomic_store_explicit(&s->release_first,1,memory_order_release);
  }
  if(!status)status=wait_word(s->complete);
  if(!status && receiver==receive_gate)elapsed=mesh_clock_ns()-start;
  for(unsigned i=0;!status && i<3;i++){
    if(payload[i][0]!=0xab || payload[i][bytes[i]+1]!=0xcd)status=EILSEQ;
    for(size_t k=1;!status && k<=bytes[i];k++)if(payload[i][k]!=expected[i])status=EILSEQ;
  }
  for(int i=1;i>=0;i--)if(handles[i]){
    int released=mesh_peer_release(ctx,handles[i]);if(!status)status=released;
  }
  int returned=mesh_lease_return(ctx,lease.first);if(!status)status=returned;
  if(!status)printf("{\"role\":\"%s\",\"group\":%" PRIu64 ",\"second_completed_before_first\":true,\"guards\":true,\"held\":\"%s\",\"held_bytes\":%zu,\"%s\":%" PRIu64 "}\n",
                    receiver?"receive":"send",group,receive_gate?"destination":"source",held_bytes,
                    receive_gate?"first_destination_landing_ns":"first_source_retire_ns",elapsed);
  return status;
}

static int separate_groups(struct mesh_ctx *ctx,uint32_t peer,uint64_t first,int receiver,int concurrent){
  enum { N=16 };
  struct mesh_section lease;int status=mesh_section_create(ctx,sizeof(struct storage)*N,1,&lease);
  if(status)return status;
  struct storage *s=mesh_section_address(ctx,lease,0);memset(s,0,sizeof *s*N);
  uint64_t handles[N]={0};unsigned sent=0;
  for(unsigned i=0;!status && i<N;i++){
    uint64_t off=(uint64_t)((char *)(s+i)-(char *)ctx->M);
    s[i].payload[0][0]=0x1d;s[i].payload[0][1]=receiver?0:0x70+i;s[i].payload[0][2]=0x28;
    s[i].spans[0]=(struct mesh_dma_span){off+offsetof(struct storage,payload)+1,1};
    s[i].first=(struct mesh_peer_transfer){i,off+offsetof(struct storage,spans),1,
      receiver?0:off+offsetof(struct storage,release_first),1,off+offsetof(struct storage,complete),1,(uint32_t)receiver,0};
    status=mesh_peer_install(ctx,peer,first+i,off,1,handles+i);
    if(!status)status=mesh_peer_wait(ctx,handles[i],MESH_PLATFORM_NS);
    if(!status && !concurrent){
      if(!receiver)atomic_store_explicit(&s[i].release_first,1,memory_order_release);
      status=wait_word(s[i].complete);
      if(!status)sent++;
      int r=mesh_peer_release(ctx,handles[i]);handles[i]=0;if(!status)status=r;
    }
  }
  if(!status && concurrent && !receiver)
    for(unsigned i=0;i<N;i++)atomic_store_explicit(&s[i].release_first,1,memory_order_release);
  for(unsigned i=0;i<N;i++){
    if(!status && concurrent){status=wait_word(s[i].complete);if(!status)sent++;}
    if(!status && (s[i].payload[0][0]!=0x1d || s[i].payload[0][1]!=0x70+i || s[i].payload[0][2]!=0x28))status=EILSEQ;
    if(handles[i]){int r=mesh_peer_release(ctx,handles[i]);if(!status)status=r;}
  }
  int r=mesh_lease_return(ctx,lease.first);if(!status)status=r;
  if(!status)printf("{\"role\":\"%s\",\"first_group\":%" PRIu64 ",\"groups_completed\":%u,\"concurrent\":%s,\"guards\":true}\n",
    receiver?"receive":"send",first,sent,concurrent?"true":"false");
  return status;
}

static int three_groups(struct mesh_ctx *ctx,uint32_t peer,uint64_t first,int receiver){
  struct mesh_section lease;int status=mesh_section_create(ctx,3*sizeof(struct storage),1,&lease);if(status)return status;
  struct storage *s=mesh_section_address(ctx,lease,0);memset(s,0,3*sizeof *s);
  uint64_t handles[3]={0};
  for(unsigned i=0;i<3 && !status;i++){
    uint64_t off=(uint64_t)((char *)(s+i)-(char *)ctx->M);unsigned count=i==2?2:1;
    for(unsigned k=0;k<count;k++){
      int land=k?!receiver:receiver;
      s[i].payload[k][0]=0x1c;s[i].payload[k][1]=land?0:0x39+i+k;s[i].payload[k][2]=0x21;
      s[i].spans[k]=(struct mesh_dma_span){off+offsetof(struct storage,payload)+k*3+1,1};
      uint64_t gate=land || (i==2 && !k)?0:off+offsetof(struct storage,release_first);
      s[i].second[k]=(struct mesh_peer_transfer){k,off+offsetof(struct storage,spans)+k*sizeof s[i].spans[0],1,
        gate,1,off+offsetof(struct storage,complete)+8*k,1,(uint32_t)land,0};
    }
    status=mesh_peer_install(ctx,peer,first+i,off+offsetof(struct storage,second),count,handles+i);
    if(!status)status=mesh_peer_wait(ctx,handles[i],MESH_PLATFORM_NS);
  }
  if(!status)status=wait_word(s[2].complete);
  if(!status && (atomic_load(s[0].complete) || atomic_load(s[1].complete) || s[2].payload[0][1]!=0x3b))status=EILSEQ;
  if(!status && receiver)atomic_store_explicit(&s[2].release_first,1,memory_order_release);
  if(!status)status=wait_word(s[2].complete+1);
  for(int i=1;i>=0 && !status;i--){
    if(!receiver)atomic_store_explicit(&s[i].release_first,1,memory_order_release);
    status=wait_word(s[i].complete);
  }
  for(unsigned i=0;i<3;i++)for(unsigned k=0;k<(i==2?2u:1u);k++)
    if(!status && (s[i].payload[k][0]!=0x1c || s[i].payload[k][1]!=0x39+i+k || s[i].payload[k][2]!=0x21))status=EILSEQ;
  for(unsigned i=0;i<3;i++)if(handles[i]){int r=mesh_peer_release(ctx,handles[i]);if(!status)status=r;}
  int closed=mesh_lease_return(ctx,lease.first);if(!status)status=closed;
  if(!status)printf("{\"two_held_while_third_completed\":true,\"guards\":true,\"role\":\"%s\"}\n",receiver?"receive":"send");
  return status;
}

static int packet(struct mesh_ctx *ctx,uint32_t peer,uint64_t group,int receiver){
  const uint64_t bytes=(UINT64_C(20)<<20)+37,gap=31;
  struct mesh_section lease;int status=mesh_section_create(ctx,sizeof(struct storage)+bytes+4*gap,1,&lease);
  if(status)return status;
  struct storage *s=mesh_section_address(ctx,lease,0);uint64_t off=(char *)s-(char *)ctx->M;
  unsigned char *data=(void *)(s+1);memset(s,0,sizeof *s);memset(data,0xbd,bytes+4*gap);
  const uint64_t lengths[]={bytes/3+1,bytes/3+9,bytes-2*(bytes/3)-10};
  uint32_t n=receiver?3:1;uint64_t cursor=0,handle=0;
  for(uint32_t k=0;k<n;k++){
    uint64_t length=receiver?lengths[k]:bytes;
    s->spans[k]=(struct mesh_dma_span){off+sizeof *s+cursor+(k+1)*gap,length};
    if(!receiver)for(uint64_t i=0;i<length;i++)data[gap+i]=(unsigned char)(i+13*(i>>8));
    cursor+=length;
  }
  s->first=(struct mesh_peer_transfer){1,off+offsetof(struct storage,spans),n,0,0,
    off+offsetof(struct storage,complete),1,(uint32_t)receiver,0};
  status=mesh_peer_install(ctx,peer,group,off,1,&handle);
  if(!status)status=mesh_peer_wait(ctx,handle,MESH_PLATFORM_NS);
  if(!status)status=wait_word(s->complete);
  cursor=0;
  for(uint32_t k=0;!status && k<n;k++){
    unsigned char *part=(unsigned char *)ctx->M+s->spans[k].offset;
    for(uint64_t i=0;i<gap;i++)if(part[(ptrdiff_t)i-(ptrdiff_t)gap]!=0xbd || part[s->spans[k].bytes+i]!=0xbd){status=EILSEQ;break;}
    for(uint64_t i=0;!status && i<s->spans[k].bytes;i++)if(part[i]!=(unsigned char)((cursor+i)+13*((cursor+i)>>8)))status=EILSEQ;
    cursor+=s->spans[k].bytes;
  }
  if(handle){int r=mesh_peer_release(ctx,handle);if(!status)status=r;}
  int r=mesh_lease_return(ctx,lease.first);if(!status)status=r;
  if(!status)printf("{\"role\":\"%s\",\"bytes\":%" PRIu64 ",\"spans\":%u,\"guards\":true}\n",receiver?"receive":"send",bytes,n);
  return status;
}

int main(int argc,char **argv){
  if((argc!=4 && argc!=5) || (strcmp(argv[1],"send") && strcmp(argv[1],"receive")) ||
     (argc==5 && strcmp(argv[4],"concurrent") && strcmp(argv[4],"sequential") && strcmp(argv[4],"blocked") && strcmp(argv[4],"packet") && strcmp(argv[4],"late-receive") && strcmp(argv[4],"late-receive-window"))){
    fprintf(stderr,"usage: %s send|receive PEER_NODE GROUP [concurrent|sequential|blocked|packet|late-receive|late-receive-window]\n",argv[0]);return 2;
  }
  struct mesh_ctx ctx={.fd=-1};int status=mesh_attach(&ctx);
  uint32_t peer=(uint32_t)strtoul(argv[2],NULL,0);uint64_t group=strtoull(argv[3],NULL,0);
  if(!status)status=mesh_peer_capable(&ctx,peer);
  if(!status){int receive=!strcmp(argv[1],"receive");
    if(argc==4 || !strcmp(argv[4],"late-receive"))status=sample(&ctx,peer,group,receive,argc==5,1);
    else if(!strcmp(argv[4],"late-receive-window"))status=sample(&ctx,peer,group,receive,1,64u*1024*1024+37);
    else if(!strcmp(argv[4],"packet"))status=packet(&ctx,peer,group,receive);
    else if(!strcmp(argv[4],"blocked"))status=three_groups(&ctx,peer,group,receive);
    else status=separate_groups(&ctx,peer,group,receive,!strcmp(argv[4],"concurrent"));
  }
  if(ctx.M)mesh_detach(&ctx);
  if(status)fprintf(stderr,"%s: %s (%d)\n",argv[0],strerror(status),status);
  return status?1:0;
}
