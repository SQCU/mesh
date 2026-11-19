"""Production consumer permutation and local DMA retirement, without NIC mocks of progress."""
from pathlib import Path
import unittest

from test_native_lifetimes import function
import test_native_byte_plans as native

ROOT = Path(__file__).resolve().parents[1]

BOUNDARIES = r'''
static unsigned records,spins,turns;
static uint64_t record_position[8],record_need[8],spin_target[8];
static uint32_t record_track[8],spin_channel[8];
static int64_t *protected_source,read0[2],read1[2];
static struct ncclComm *running;
static ncclResult_t gpu(struct metal_result result,const char *what){(void)what;return result.status?ncclSystemError:ncclSuccess;}
struct metal_result metal_record(struct metal_program *p,uint32_t track,int cursor,uint64_t position,uint64_t need){
  (void)p;assert(cursor==METAL_CONSUME&&records<8);
  record_track[records]=track;record_position[records]=position;record_need[records++]=need;return METAL_DONE;
}
struct metal_result metal_spin(struct metal_program *p,const struct metal_wait *wait){
  (void)p;assert(spins<8);spin_target[spins]=wait->target.n;spin_channel[spins++]=wait->channel;return METAL_DONE;
}
static int revoked(struct ncclComm *c){(void)c;return 0;}
static ncclResult_t revoked_fail(struct ncclComm *c){(void)c;return ncclRemoteError;}
static ncclResult_t held_fail(const struct ncclComm *c,const struct channel *ch){(void)c;(void)ch;return ncclRemoteError;}
struct attach;
static int hold_live(const struct attach *a,uint32_t s,uint64_t wire){(void)a;(void)s;(void)wire;return 1;}
static int ktype(const struct call *c){(void)c;return ncclInt64;}
static int kop(const struct call *c){(void)c;return ncclSum;}
static size_t kcount(const struct call *c,size_t bytes){(void)c;return bytes/8;}
static void combine_into(int type,int op,void *out,const void *input,size_t n){
  (void)type;(void)op;for(size_t k=0;k<n;k++)((int64_t *)out)[k]+=((const int64_t *)input)[k];
}
/* The NIC boundary reads the released source only on later turns, and each
   physical cable retires independently. The production host loop runs intact. */
static uint64_t tick(const struct mesh_deadline *d){
  (void)d;assert(++turns<8);
  if(turns==2){
    assert(protected_source[0]==1&&protected_source[1]==2);
    memcpy(read0,protected_source,sizeof read0);atomic_store(running->channels[0].view.sent,2);
  }
  if(turns==3){
    assert(protected_source[0]==1&&protected_source[1]==2);
    memcpy(read1,protected_source,sizeof read1);atomic_store(running->channels[1].view.sent,2);
  }
  return 1;
}
#define mesh_deadline_left tick
'''

CASES = r'''
static void aot_permutation(void){
  uint32_t order0[]={9,4},order1[]={3};
  struct steps st[2]={{.consume_order=order0},{.consume_order=order1}};
  struct call calls[2]={{.bound=st},{.bound=st+1}};
  struct message incoming[6];
  for(size_t k=0;k<6;k++)incoming[k]=(struct message){.c=calls+(k%3==2),.step=k%3==2?0:k%3,.entry=k,.end={11+k}};
  struct schedule p={.in=incoming,.ins=6};
  assert(!schedule_consumption(&p,2));
  const size_t want[]={1,0,2,4,3,5};
  for(size_t k=0;k<6;k++){
    assert(p.consume[k]==want[k]&&consumption_at(&p,k)==incoming+want[k]);
    assert(incoming[k].entry==k&&incoming[k].end.n==11+k);
  }
  free(p.consume);
}
static void lifetime_and_host(void){
  int64_t source[]={1,2},immutable[]={41,42},delta[]={10,20};
  struct where buffer[]={{NULL,(size_t)source}};
  struct mesh_step steps[]={
    {MESH_STEP_SEND,1,0,0,0,{7,8,2}}, {MESH_STEP_SEND,2,1,0,0,{7,8,2}},
    {MESH_STEP_REDUCE,1,2,0,0,{7,8,2}}};
  uint32_t priors[]={0,0,0,0};unsigned char done[3]={0};
  struct steps st={.steps=steps,.buffer=buffer,.prior_first=priors,.done=done};
  struct call c={.bound=&st};
  struct message outgoing[2]={
    {.c=&c,.step=0,.entry=0,.end={2},.bytes=16,.data={NULL,(size_t)source}},
    {.c=&c,.step=1,.entry=0,.end={2},.bytes=16,.data={NULL,(size_t)source}}};
  struct message incoming={.c=&c,.step=2,.entry=0,.end={2},.bytes=16,.data={NULL,(size_t)delta}};
  struct schedule p[2]={{.out=outgoing,.outs=1,.in=&incoming,.ins=1},{.out=outgoing+1,.outs=1}};
  _Atomic uint64_t ready[2]={0},sent[2]={0},landed[2]={2,0};uint64_t wire=1;
  struct attach attach={0};struct ncclComm comm={.count=2,.attach=&attach};
  for(unsigned q=0;q<2;q++)comm.channels[q]=(struct channel){.stream=q,.wire=&wire,
    .buffer=(void *)sent,.sent=q*8,.view={landed+q,ready+q,sent+q}};
  assert(!bind_retirements(&comm,p));assert(incoming.retire[0]==2&&incoming.retire[1]==2);
  assert(!consume_retirements(NULL,&comm,&incoming,7,0));
  assert(records==2&&spins==2);
  for(unsigned k=0;k<2;k++)assert(record_track[k]==7&&record_position[k]==UINT64_MAX&&record_need[k]==0&&spin_target[k]==2&&spin_channel[k]==k);
  assert(record_position[0]+1==0); /* CQ completion never claims the landing consumed. */
  protected_source=source;running=&comm;assert(!run(&comm,p));
  assert(turns==3&&source[0]==11&&source[1]==22&&done[2]);
  assert(read0[0]==1&&read0[1]==2&&read1[0]==1&&read1[1]==2);
  free(incoming.retire);incoming.retire=NULL;
  /* Actual immutable original-input bindings require no CQ retirement guard. */
  outgoing[0].data.offset=outgoing[1].data.offset=(size_t)immutable;
  assert(!bind_retirements(&comm,p)&&!incoming.retire);
  /* COPY DMA writes precede consumption, so it cannot use that guard. */
  outgoing[0].data.offset=(size_t)source;steps[2].op=MESH_STEP_COPY;incoming.data.offset=(size_t)source;
  assert(bind_retirements(&comm,p)==ncclInvalidUsage);
}
int main(void){(void)attach_lock;aot_permutation();lifetime_and_host();puts("native event order and delayed local DMA PASS");return 0;}
'''


class NativeEventOrder(unittest.TestCase):
    def test_production_permutation_host_loop_and_gpu_retirement_records(self):
        source = (ROOT/'nccl-mesh.c').read_text()
        names = ('named', 'shifted', 'address_of', 'bind_retirements',
                 'consume_retirements', 'ready', 'count_of', 'run')
        body = native.MOCKS + BOUNDARIES + '\n'.join(function(source, name) for name in names) + CASES
        native.NativeBytePlans.compile_and_run(self, body, 'native event order and delayed local DMA PASS')


if __name__ == '__main__':
    unittest.main()
