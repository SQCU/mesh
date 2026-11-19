/* Compile with the same private NCCL entry points used by plan capture and replay. */
#include "../nccl-mesh.c"
#include <assert.h>

enum { RANKS=4, PAGE=16384 };
static const uint32_t pairs[][2]={{0,1},{0,2},{0,3},{1,3},{2,3}};
static uint32_t cables[][2]={{0,1},{0,1},{0,2},{0,3},{1,3},{2,3}};
static uint64_t bandwidth[]={80000000000,80000000000,80000000000,80000000000,80000000000,80000000000};
static struct {
  uint64_t pad, ready[RANKS][RANKS], landed[RANKS][RANKS];
} words;
static uint64_t prefixes[RANKS][RANKS],published[RANKS][RANKS];
static unsigned gates;

static int recorded(void *unused,struct ncclComm *comm,uint32_t h,int list,struct mesh_entry entry,uint64_t end){
  (void)unused;
  const unsigned peer=comm->channels[h].peer;
  if(list){
    assert(end==prefixes[comm->rank][peer]+entry.units);
    prefixes[comm->rank][peer]=end;
    if(comm->rank==1 || comm->rank==2)assert(entry.units<=comm->schedule.piece/PAGE);
    return 0;
  }
  if(comm->rank==0)return 0;
  assert(entry.base==published[comm->rank][peer]);
  published[comm->rank][peer]+=entry.units;
  assert(comm->rank!=3 && comm->channels[h].peer==3);
  const uintptr_t word=entry.word+(uintptr_t)comm->attach->context.M;
  assert(word==(uintptr_t)&words.landed[comm->rank][0]);
  gates++;
  return 0;
}

int main(void){
  const uint64_t bytes=8*1024*1024+20,contributors=1,recipients=8;
  struct ncclComm comm[RANKS]={0};
  struct attach attach[RANKS]={0};
  struct call calls[RANKS]={0};
  struct channel channels[RANKS][RANKS]={0};
  struct schedule plans[RANKS][RANKS]={0};
  struct mesh_operand operand={ncclUint32+1,4,bytes/4};
  struct entry entry={.scheduled=1,.program={.what=MESH_BROADCAST,.root=0}};
  unsigned char *data[RANKS];
  ncclMeshLinkRate rates[2*sizeof pairs/sizeof *pairs];
  for(size_t e=0;e<sizeof pairs/sizeof *pairs;e++)for(int d=0;d<2;d++){
    rates[2*e+d]=(ncclMeshLinkRate){pairs[e][d],pairs[e][1-d],10e9,10e9,125,50};
  }
  for(uint32_t r=0;r<RANKS;r++){
    comm[r]=(struct ncclComm){.rank=(int)r,.nranks=RANKS,.page=PAGE,.channels=channels[r],.rates=rates,.nrates=sizeof rates/sizeof *rates,
      .map={.kind=MESH_LINKS_GRAPH,.nodes=RANKS,.links=sizeof cables/sizeof *cables,.link=cables,.bandwidth=bandwidth},.attach=attach+r};
    attach[r].context.M=(struct hdr *)&words;
    for(uint32_t e=0;e<sizeof pairs/sizeof *pairs;e++){
      uint32_t from=pairs[e][0],to=pairs[e][1];if(from!=r && to!=r)continue;
      unsigned peer=r==from?to:from;
      struct channel *ch=&channels[r][comm[r].count++];
      ch->cable=e;ch->peer=peer;ch->ready_at=(uintptr_t)&words.ready[r][peer]-(uintptr_t)attach[r].context.M;
      ch->view.landed=(_Atomic uint64_t *)&words.landed[r][peer];
    }
    calls[r]=(struct call){.comm=comm+r,.what=MESH_BROADCAST,.root=0,.count=operand.elements,.type=ncclUint32};
    ncclMeshCall roles={.contributors=&contributors,.recipients=&recipients};
    assert(!declared_members(calls+r,&roles));
    assert(call_contributes(calls+r)==(r==0) && call_receives(calls+r)==(r==3));
    assert(!collective_steps(calls+r,&entry,operand,NULL));
    assert(!schedule_group(comm+r,calls+r,1,plans[r]));
    schedule_laid(comm+r,plans[r]);assert(!group_entries(comm+r,plans[r],recorded,NULL));
    data[r]=malloc(bytes);assert(data[r]);
  }
  struct scatter_schedule *solved=calls[0].bound->scatter;
  uint64_t directed[RANKS][RANKS]={0};
  for(unsigned inv=0;inv<3;inv++){
    for(unsigned r=1;r<RANKS;r++)memset(data[r],0xf0+r,bytes);
    for(uint64_t i=0;i<bytes;i++)data[0][i]=(unsigned char)(i*151+(i>>15)*29+inv*73);
    size_t at[RANKS][RANKS]={0};
    uint64_t consumed[RANKS][RANKS]={0};
    /* Drain the independent direct and second-relay path before the first relay receives anything. */
    for(int held=0;held<=1;held++)for(uint32_t k=0;k<solved->count;k++){
      const struct mesh_scatter_send *hop=solved->send+k;
      const struct mesh_scatter_link *link=solved->link+hop->link;
      const uint32_t a=link->from,b=link->to;
      if((a==1 || b==1)!=held)continue;
      unsigned tx=0,rx=0;
      while(channels[a][tx].peer!=(int)b)tx++;
      while(channels[b][rx].peer!=(int)a)rx++;
      const size_t e=at[a][b];
      assert(e<plans[a][tx].outs && e<plans[b][rx].ins);
      const struct message *send=plans[a][tx].out+e,*land=plans[b][rx].in+e;
      assert(send->offset==land->offset && send->bytes==land->bytes);
      assert(send->offset+consumed[a][b]==hop->first && hop->bytes<=send->bytes-consumed[a][b]);
      assert(!memcmp(data[a]+hop->first,data[0]+hop->first,hop->bytes));
      memcpy(data[b]+hop->first,data[a]+hop->first,hop->bytes);
      consumed[a][b]+=hop->bytes;
      if(consumed[a][b]==send->bytes){at[a][b]++;consumed[a][b]=0;}
      if(!inv)directed[a][b]+=hop->bytes;
    }
    for(unsigned r=0;r<RANKS;r++)for(unsigned h=0;h<comm[r].count;h++)
      assert(at[r][channels[r][h].peer]==plans[r][h].outs);
    assert(!memcmp(data[3],data[0],bytes));
  }
  assert(directed[0][1] && directed[0][2] && directed[0][3] && directed[1][3] && directed[2][3]);
  assert(directed[0][1]==directed[1][3] && directed[0][2]==directed[2][3]);
  assert(directed[0][1]+directed[0][2]+directed[0][3]==bytes && gates && gates<calls[0].comm->schedule.via_relays);
  printf("source publications %u for %llu source pieces\n", calls[0].bound->count,
         calls[0].comm->schedule.sends-calls[0].comm->schedule.via_relays);
  for(unsigned r=0;r<RANKS;r++){
    struct steps *view=steps_copy(calls[r].bound);
    assert(view && view->scatter==calls[r].bound->scatter);steps_free(view);
    for(unsigned h=0;h<comm[r].count;h++){free(plans[r][h].out);free(plans[r][h].in);}
    steps_free(calls[r].bound);free(calls[r].contributors);free(calls[r].recipients);free(data[r]);
  }
  printf("scheduled broadcast: %llu direct, %llu and %llu forwarded bytes, %u relay gates, 3 generations verified\n",
         directed[0][3],directed[1][3],directed[2][3],gates);
}
