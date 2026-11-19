#include "../nccl-mesh.c"
#include "../mesh-order.h"
#include <assert.h>

enum { RANKS=4, PAGE=16384 };
static const uint32_t pairs[][2]={{0,1},{0,2},{0,3},{1,3},{2,3}};
static uint32_t cables[][2]={{0,1},{0,1},{0,2},{0,3},{1,3},{2,3}};
static uint64_t bandwidth[]={80000000000,80000000000,80000000000,80000000000,80000000000,80000000000};
static struct {
  uint64_t pad,ready[RANKS][RANKS],landed[RANKS][RANKS];
} words;

struct wire {
  const struct mesh_dma_span *span[2];
  uint64_t units[2],digest[2],at;
  struct mesh_gate_term *gate;
  uint64_t *landed;
};
static struct wire wires[RANKS][RANKS];
static const uint64_t bytes=8*1024*1024+20;

static int recorded(void *unused,struct ncclComm *comm,uint32_t h,int list,struct mesh_entry entry,uint64_t end){
  (void)unused;
  const int me=comm->rank,peer=comm->channels[h].peer;
  struct wire *w=list?&wires[peer][me]:&wires[me][peer];
  assert(end==w->units[list]+entry.units);
  w->digest[list]=order_hash(w->digest[list],entry.units);
  for(uint64_t k=0;k<entry.units;k++){
    uint64_t at=w->units[list]++;
    if(list) w->landed[at]=at+1==end?end:0;
    else w->gate[at]=(struct mesh_gate_term){entry.word,entry.base+k+1};
  }
  return 0;
}

static void check(uint64_t recipients){
  struct ncclComm comm[RANKS]={0};
  struct attach attach[RANKS]={0};
  struct call calls[RANKS]={0};
  struct channel channels[RANKS][RANKS]={0};
  struct schedule plans[RANKS][RANKS]={0};
  const struct mesh_operand operand={ncclUint32+1,4,bytes/4};
  const struct entry entry={.scheduled=1,.program={.what=MESH_BROADCAST,.root=0}};
  const struct hdr arena={.pgsz=PAGE,.block=1,.data_off=PAGE,.length=bytes+3*PAGE};
  const uint64_t contributors=1,room=pages_of(bytes,PAGE).n;
  unsigned char *data[RANKS];
  memset(wires,0,sizeof wires);
  for(unsigned a=0;a<RANKS;a++)for(unsigned b=0;b<RANKS;b++){
    struct wire *w=&wires[a][b];
    w->gate=calloc(room,sizeof *w->gate);w->landed=calloc(room,sizeof *w->landed);
    for(unsigned d=0;d<2;d++)w->span[d]=calloc(room,sizeof *w->span[d]);
  }
  for(unsigned r=0;r<RANKS;r++){
    comm[r]=(struct ncclComm){.rank=(int)r,.nranks=RANKS,.page=PAGE,.channels=channels[r],
      .map={.kind=MESH_LINKS_GRAPH,.nodes=RANKS,.links=sizeof cables/sizeof *cables,.link=cables,.bandwidth=bandwidth},.attach=attach+r};
    attach[r].context.M=(struct hdr *)&words;
    for(uint32_t e=0;e<sizeof pairs/sizeof *pairs;e++){
      uint32_t from=pairs[e][0],to=pairs[e][1];if(from!=r && to!=r)continue;
      unsigned peer=r==from?to:from;
      struct channel *ch=&channels[r][comm[r].count++];
      ch->cable=e;ch->peer=peer;ch->ready_at=(uintptr_t)&words.ready[r][peer]-(uintptr_t)&words;
      ch->view.landed=(_Atomic uint64_t *)&words.landed[r][peer];
    }
    calls[r]=(struct call){.comm=comm+r,.what=MESH_BROADCAST,.root=0,.count=operand.elements,.type=ncclUint32};
    ncclMeshCall roles={.contributors=&contributors,.recipients=&recipients};
    assert(!declared_members(calls+r,&roles));
    assert(call_contributes(calls+r)==(r==0) && call_receives(calls+r)==((recipients>>r)&1));
    assert(!collective_steps(calls+r,&entry,operand,NULL));
    assert(!schedule_group(comm+r,calls+r,1,plans[r]));
    schedule_laid(comm+r,plans[r]);
    for(unsigned h=0;h<comm[r].count;h++)for(int d=0;d<2;d++){
      const int peer=channels[r][h].peer;
      struct wire *w=d?&wires[peer][r]:&wires[r][peer];
      struct schedule *p=plans[r]+h;
      for(unsigned j=0;j<(d?p->ins:p->outs);j++){
        struct message *m=(d?p->in:p->out)+j;
        const uint32_t head=m->c->bound->steps[m->step].round;
        const uint64_t start=m->end.n-m->pages.n;
        assert(m->end.n<=room);
        m->spans=(struct mesh_dma_span *)w->span[d]+start;m->table=start*sizeof *m->spans;
        assert(!scatter_spans(&arena,m->c->bound->scatter,head,PAGE+m->offset,PAGE,m));
      }
    }
    assert(!group_entries(comm+r,plans[r],recorded,NULL));
    data[r]=malloc(bytes);assert(data[r]);
  }
  const struct scatter_schedule *solved=calls[0].bound->scatter;
  uint64_t seen[RANKS][RANKS]={0};
  for(uint32_t k=0;k<solved->count;k++){
    const struct mesh_scatter_send *hop=solved->send+k;
    const struct mesh_scatter_link *edge=solved->link+hop->link;
    struct wire *w=&wires[edge->from][edge->to];
    for(uint64_t i=0;i<hop->bytes;i+=PAGE){
      const uint64_t at=seen[edge->from][edge->to]++;
      assert(at<w->units[0] && at<w->units[1] && w->digest[0]==w->digest[1]);
      assert(w->span[0][at].offset==PAGE+hop->first+i && w->span[0][at].bytes==(hop->bytes-i<PAGE?hop->bytes-i:PAGE));
      assert(!memcmp(w->span[0]+at,w->span[1]+at,sizeof *w->span[0]));
    }
  }
  for(unsigned inv=0;inv<3;inv++){
    memset(&words,0,sizeof words);
    for(unsigned r=0;r<RANKS;r++)for(unsigned q=0;q<RANKS;q++){words.ready[r][q]=UINT64_MAX-1;wires[r][q].at=0;}
    for(unsigned r=1;r<RANKS;r++)memset(data[r],0xf0+r,bytes);
    for(uint64_t i=0;i<bytes;i++)data[0][i]=(unsigned char)(i*151+(i>>15)*29+inv*73);
    for(int hold=1;hold>=0;hold--){
      int moved;
      do{
        moved=0;
        for(unsigned a=0;a<RANKS;a++)for(unsigned b=0;b<RANKS;b++){
          struct wire *w=&wires[a][b];
          if((hold && (a==1 || b==1)) || w->at==w->units[0])continue;
          const struct mesh_gate_term gate=w->gate[w->at];
          if(order_threshold((const struct hdr *)&words,gate.word,gate.threshold)<=0)continue;
          const struct mesh_dma_span span=w->span[0][w->at];
          const uint64_t offset=span.offset-PAGE;
          assert(!memcmp(data[a]+offset,data[0]+offset,span.bytes));
          memcpy(data[b]+offset,data[a]+offset,span.bytes);
          if(w->landed[w->at])words.landed[b][a]=w->landed[w->at];
          w->at++;moved=1;
        }
      }while(moved);
    }
    for(unsigned r=0;r<RANKS;r++){
      if(recipients>>r&1)assert(!memcmp(data[r],data[0],bytes));
      for(unsigned q=0;q<RANKS;q++)assert(wires[r][q].at==seen[r][q]);
    }
  }
  assert(calls[0].bound->count<solved->count/4 && comm[0].schedule.via_relays);
  printf("recipient mask %llu: %u source entries, %u solved hops, three generations checked\n",recipients,calls[0].bound->count,solved->count);
  for(unsigned r=0;r<RANKS;r++){
    struct steps *view=steps_copy(calls[r].bound);
    assert(view && view->scatter==calls[r].bound->scatter && view->steps==calls[r].bound->steps && view->done!=calls[r].bound->done);
    steps_free(calls[r].bound);assert(view->demand[1] && view->count);steps_free(view);
    for(unsigned h=0;h<comm[r].count;h++){free(plans[r][h].out);free(plans[r][h].in);}
    free(calls[r].contributors);free(calls[r].recipients);free(data[r]);
    for(unsigned q=0;q<RANKS;q++){
      struct wire *w=&wires[r][q];free(w->gate);free(w->landed);free((void *)w->span[0]);free((void *)w->span[1]);
    }
  }
}
int main(void){check(8);check(15);}
