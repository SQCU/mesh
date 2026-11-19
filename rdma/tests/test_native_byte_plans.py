"""Run production native byte placement, binding and relay publication bodies.

Allocation/stream boundaries are mocked; packet cuts and address validation are
the real headers. This checks native integration without claiming NIC progress.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from test_native_lifetimes import function

ROOT = pathlib.Path(__file__).resolve().parents[1]

MOCKS = r'''
#include "nccl.h"
#include "mesh.h"
#include "mesh-plan.h"
#include "mesh-row-span-plan.h"
#include "mesh-lease-retire.h"
#include "mesh-byte-account.h"
#include "nccl-mesh-metal.h"
#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/mman.h>
struct where { void *buffer; size_t offset; };
struct ncclMeshWritePlan { struct mesh_span_plan packets; size_t stride, row_bytes, rows, pitch;
  ncclMeshSpan *view; uint64_t *view_first; size_t views,view_extent; };
struct steps {
  struct scatter_schedule *scatter;
  struct mesh_step *steps; struct where *buffer,*read;
  uint32_t *channel,*prior_first,*prior,*consume_order,*retired_first,*retired; uint64_t *entry,*consumption; unsigned char *done,*fresh;
};
struct call { struct steps *bound; struct ncclMeshWritePlan *write_plan; ncclMeshFused *fused; int target,root; };
struct message { struct call *c; uint32_t step; uint64_t entry; mesh_pages pages,end; uint64_t need; size_t offset,bytes;
  struct where data; struct mesh_dma_span *spans; uint64_t table; double priority; uint64_t *retire; struct mesh_byte_shape *shape; };
struct schedule { struct message *out,*in; size_t outs,ins; uint64_t first[2],word,base; mesh_pages each[2]; size_t *consume; };
struct compiled { struct call *calls; size_t n; uint32_t repeat; struct schedule *plans,*write_plans; };
struct plan { ncclMeshPlan view; struct compiled *group; uint64_t (*entries)[2]; int active; };
struct mesh_ctx { struct hdr *M; };
struct mesh_entry { uint64_t spans,units,word,base; };
/* an entry the bridge could hold declared: its span table the region's, each span valid */
static int mesh_entry_valid(const struct hdr *m,struct mesh_entry e){
  if(!e.units || e.base>=UINT64_MAX-e.units || !mesh_region_holds(m,e.spans,e.units*sizeof(struct mesh_dma_span),8))return 0;
  const struct mesh_dma_span *spans=(const struct mesh_dma_span *)((const char *)m+e.spans);
  for(uint64_t i=0;i<e.units;i++)if(!mesh_dma_span_valid(m,spans[i]))return 0;
  return 1;
}
typedef int (*entry_visit)(void *context, struct ncclComm *comm, uint32_t h, int list, struct mesh_entry entry, uint64_t end);
struct channel_view { _Atomic uint64_t *landed,*ready,*sent; };
struct hold { uint64_t wire,generation; };
struct attach { struct mesh_ctx context; struct hold holds[2]; };
struct channel { uint32_t stream,node,cable; struct channel_view view; void *buffer; size_t sent; uint64_t ready_at;
  uint64_t published,consumed,waited; const uint64_t *wire; struct mesh_byte_account payload[2]; unsigned trace_readers; };
struct ncclComm { struct attach *attach; uint64_t page; uint32_t count; struct plan *plan; const char *name;
  struct channel channels[2]; uint32_t endpoint[2]; int rank; void *violation; };
struct lease { void *address; size_t bytes; int returned; struct mesh_lease_retire reads; };
static struct lease leases[64]; static size_t nleases,pool;
static pthread_mutex_t attach_lock=PTHREAD_MUTEX_INITIALIZER;
static struct {char text[512];} last;
static struct mesh_entry observed[2][2][8]; static unsigned observed_count[2][2];
static ncclResult_t fail(ncclResult_t result,const char *format,...){
  va_list a;va_start(a,format);vsnprintf(last.text,sizeof last.text,format,a);va_end(a);return result;
}
static ncclResult_t comm_fail(struct ncclComm *c,ncclResult_t r,const char *format,...){(void)c;(void)format;return r;}
void *metal_contents(void *p){return p;}
static const char *call_name(const struct call *c){(void)c;return "fixture";}
static struct lease *lease_at(const void *p,void *unused){
  (void)unused;uintptr_t at=(uintptr_t)p;
  for(size_t i=0;i<nleases;i++)if(at>=(uintptr_t)leases[i].address && at-(uintptr_t)leases[i].address<leases[i].bytes)return leases+i;
  return NULL;
}
static int in_region(struct ncclComm *c,struct where w,size_t bytes){
  uintptr_t p=w.buffer?(uintptr_t)w.buffer+w.offset:w.offset,base=(uintptr_t)c->attach->context.M;
  return p>=base && p-base<=c->attach->context.M->length && bytes<=c->attach->context.M->length-(p-base);
}
static void *spare_take(struct ncclComm *c,size_t bytes,uint64_t holder){
  (void)holder;assert(nleases<64);pool=(pool+15)&~(size_t)15;
  void *p=(char *)c->attach->context.M+pool;pool+=bytes;
  leases[nleases++]=(struct lease){.address=p,.bytes=bytes};return p;
}
static struct where spare_where(struct metal_program *p,struct ncclComm *c,size_t bytes,uint64_t holder){
  (void)p;return(struct where){NULL,(size_t)spare_take(c,bytes,holder)};
}
static struct steps *steps_copy(const struct steps *s){return (struct steps *)s;}
/* This fixture exercises WRITE placement; membership belongs to the collective
   integration fixture and has no effect on explicit endpoint writes. */
static void call_bind_roles(struct call *c,const struct call *from){(void)c;(void)from;}
static ncclResult_t compile_group(struct ncclComm *c,struct call *l,size_t n,struct schedule **s){
  (void)c;(void)l;(void)n;(void)s;assert(!"unexpected compile");return ncclInternalError;
}
static int all_writes(const struct call *l,size_t n){(void)l;(void)n;return 1;}
static ncclResult_t order_writes(const struct ncclComm *c,const struct call *l,size_t n,struct schedule *s){
  (void)c;(void)l;(void)n;(void)s;return ncclSuccess;
}
/* an issued entry as the trace log takes it: each one as the bridge holds it declared */
static void channel_log(struct ncclComm *comm,const struct channel *ch,int land,struct mesh_entry e){
  const uint32_t stream=ch->stream;
  assert(stream<2 && observed_count[stream][land]<8 && mesh_entry_valid(comm->attach->context.M,e));
  observed[stream][land][observed_count[stream][land]++]=e;
}
'''

# Keep private schedule lowering in the same production helper used by indexed
# preparation fixtures; there is no test-only permutation implementation.
_native_source = (ROOT / 'nccl-mesh.c').read_text()
MOCKS += _native_source[_native_source.index('struct scatter_slice {'):_native_source.index('struct steps {')]
_begin = _native_source.index('struct consumption_order {')
MOCKS += _native_source[_begin:_native_source.index('};', _begin)+2]
MOCKS += '\n'.join(function(_native_source, name) for name in
                   ('consumption_at', 'consumption_order_compare', 'schedule_consumption'))

CASES = r'''
static struct hdr *region;
static struct attach attachment;
static struct ncclComm comm;
static void fixture(void){
  for(size_t i=0;i<nleases;i++)mesh_lease_retire_destroy(&leases[i].reads);
  for(unsigned h=0;h<2;h++)for(unsigned d=0;d<2;d++)mesh_byte_account_destroy(&comm.channels[h].payload[d]);
  memset(leases,0,sizeof leases);nleases=0;pool=region->data_off;
  memset(observed_count,0,sizeof observed_count);
  attachment=(struct attach){.context={region},.holds={{1,1},{2,1}}};
  comm=(struct ncclComm){.attach=&attachment,.page=4096,.count=1,.name="native test",.endpoint={0,1}};
  for(unsigned h=0;h<2;h++)comm.channels[h]=(struct channel){.stream=h,.cable=h,.view={&region->control,&region->control,&region->control}};
}
static struct where binding(uint64_t offset,size_t bytes){
  void *p=(char *)region+region->data_off+offset;
  assert(nleases<64);leases[nleases++]=(struct lease){.address=p,.bytes=bytes};return(struct where){NULL,(size_t)p};
}
static struct mesh_step step(uint32_t op,uint64_t first,uint64_t bytes){
  return(struct mesh_step){.op=op,.first=first,.piece={.element_bytes=1,.elements=bytes}};
}
static void check_message(const struct message *m,const struct ncclMeshWritePlan *p,uint64_t origin){
  uint64_t total=0;
  for(uint64_t u=0;u<m->pages.n;u++){
    assert(m->spans[u].offset==origin+write_displacement(p,m->offset+total));
    assert(m->spans[u].bytes && mesh_dma_span_valid(region,m->spans[u]));
    if(p->rows>1)assert((m->offset+total)%p->row_bytes+m->spans[u].bytes<=p->row_bytes);
    total+=m->spans[u].bytes;
  }
  assert(total==m->bytes);
}
static void exact_and_common_cuts(void){
  const uint64_t payloads[]={3,4097,8195};
  const uint64_t phases[]={MESH_DMA_EXTENT-3,MESH_DMA_EXTENT-7},strides[]={11,17};
  for(unsigned test=0;test<3;test++){
    struct ncclMeshWritePlan p={0};uint64_t bytes=payloads[test];
    assert(!mesh_span_plan_create(phases,strides,2,bytes,3,4096,&p.packets));
    for(unsigned rank=0;rank<2;rank++){
      fixture();p.stride=strides[rank];
      struct where buffer=binding(phases[rank],bytes+2*p.stride);
      struct mesh_step raw=step(MESH_STEP_SEND,0,bytes);
      struct steps st={.steps=&raw,.buffer=&buffer};struct call call={.bound=&st,.write_plan=&p};
      struct message messages[3];
      for(unsigned r=0;r<3;r++)messages[r]=(struct message){.c=&call,.offset=r*p.stride,.bytes=bytes,.pages={span_units(&p.packets,0,bytes)}};
      struct schedule schedule={.out=messages,.outs=3};
      assert(!place_group(NULL,&comm,&schedule,1));
      for(unsigned r=0;r<3;r++)check_message(messages+r,&p,region->data_off+phases[rank]);
      if(bytes==3)assert(messages[0].pages.n==1 && messages[0].spans[0].bytes==3);
      assert(span_units(&p.packets,0,0)==0);
      for(size_t cut=0;cut+1<p.packets.count;cut++)
        assert(span_units(&p.packets,p.packets.cuts[cut],p.packets.cuts[cut+1]-p.packets.cuts[cut])==1);
    }
    mesh_span_plan_free(&p.packets);
  }
}
static void affine_rows(void){
  const uint64_t phases[]={MESH_DMA_EXTENT-2,MESH_DMA_EXTENT-1},pitches[]={7,13};
  struct ncclMeshWritePlan p={.row_bytes=3,.rows=5003};
  assert(!mesh_row_span_plan_create(phases,pitches,2,3,p.rows,4096,&p.packets));
  assert(p.packets.count>4096);
  for(unsigned rank=0;rank<2;rank++){
    fixture();p.pitch=pitches[rank];
    struct where buffer=binding(phases[rank],(p.rows-1)*p.pitch+p.row_bytes);
    struct mesh_step raw=step(rank?MESH_STEP_COPY:MESH_STEP_SEND,0,p.rows*p.row_bytes);
    struct steps st={.steps=&raw,.buffer=&buffer};struct call call={.bound=&st,.write_plan=&p};
    struct message m={.c=&call,.bytes=p.rows*p.row_bytes,.pages={p.packets.count-1}};
    struct schedule schedule=rank?(struct schedule){.in=&m,.ins=1}:(struct schedule){.out=&m,.outs=1};
    assert(!place_group(NULL,&comm,&schedule,1));
    check_message(&m,&p,region->data_off+phases[rank]);
    for(size_t u=0;u<m.pages.n;u++)assert(m.spans[u].bytes==p.packets.cuts[u+1]-p.packets.cuts[u]);
  }
  mesh_span_plan_free(&p.packets);
}
static void recount_repeats(void){
  fixture();uint64_t phases[]={MESH_DMA_EXTENT-3},strides[]={19};
  struct ncclMeshWritePlan p={.stride=19};
  assert(!mesh_span_plan_create(phases,strides,1,8195,2,4096,&p.packets));
  struct mesh_step raw[]={step(MESH_STEP_SEND,0,4096),step(MESH_STEP_SEND,4096,4099)};
  struct steps st={.steps=raw};struct call old={.bound=&st},issued={.write_plan=&p};
  /* The compiled template predates the actual physical binding's extra MR cuts. */
  struct message template[]={ {.c=&old,.step=0,.bytes=4096,.pages={1},.end={1}},
    {.c=&old,.step=1,.offset=4096,.bytes=4099,.pages={2},.end={3},.entry=1} };
  struct schedule original={.out=template,.outs=2};
  struct compiled compiled={.calls=&old,.n=1,.repeat=1,.plans=&original};
  struct plan plan={.view={.first=1,.issues=1},.group=&compiled,.active=1};comm.plan=&plan;
  struct schedule *made=NULL;assert(!group_bound(&comm,&issued,1,1,2,999,&made));
  assert(made && made->outs==4);uint64_t end=0;
  for(unsigned k=0;k<4;k++){
    const struct message *m=made->out+k;
    uint64_t first=k%2?4096:0,n=k%2?4099:4096;
    end+=span_units(&p.packets,first,n);
    assert(m->end.n==end && m->pages.n==span_units(&p.packets,first,n));
    assert(m->offset==first+(k/2)*19 && m->c==&issued);
    if(k==1)assert(made->each[0].n==end);
  }
  assert(end==2*made->each[0].n);schedules_free(&comm,made);mesh_span_plan_free(&p.packets);
}
static void relay_prefix(void){
  fixture();comm.count=2;comm.rank=1;
  uint64_t phase=MESH_DMA_EXTENT-3,stride=0;struct ncclMeshWritePlan p={0};
  assert(!mesh_span_plan_create(&phase,&stride,1,8195,1,4096,&p.packets));
  struct where buffer=binding(phase,8195);
  struct mesh_step raw[]={step(MESH_STEP_COPY,0,8195),step(MESH_STEP_SEND,4096,4099)};
  uint32_t channel[]={0,1},prior_first[]={0,0,1},prior[]={0};uint64_t entries[]={0,0};
  struct steps st={.steps=raw,.buffer=&buffer,.channel=channel,.prior_first=prior_first,.prior=prior,.entry=entries};
  struct call call={.bound=&st,.write_plan=&p,.target=3,.root=0};
  struct message in={.c=&call,.step=0,.bytes=8195,.pages={span_units(&p.packets,0,8195)}};
  struct message out={.c=&call,.step=1,.offset=4096,.bytes=4099,.pages={span_units(&p.packets,4096,4099)}};
  in.end=in.pages;out.end=out.pages;
  struct schedule schedules[]={{.in=&in,.ins=1},{.out=&out,.outs=1}};
  assert(!place_group(NULL,&comm,schedules,1));
  assert(!mesh_byte_shape_make(&in.shape,&in.spans[0].bytes,in.pages.n,sizeof *in.spans));
  assert(!mesh_byte_shape_make(&out.shape,&out.spans[0].bytes,out.pages.n,sizeof *out.spans));
  comm.channels[0].trace_readers=comm.channels[1].trace_readers=1;
  assert(!issue_entries(&comm,schedules));
  mesh_byte_shape_release(in.shape);mesh_byte_shape_release(out.shape);
  const struct mesh_entry e=observed[1][0][0];
  assert(e.word==(uint64_t)((char *)&region->control-(char *)region));
  assert(e.base==2 && e.units==2); /* MR tail 3 + 4093, not one 4096-byte page. */
  assert(in.spans[e.base].bytes==out.spans[0].bytes);
  assert(in.spans[e.base+1].bytes==out.spans[1].bytes);
  uint64_t bytes=0;assert(!mesh_byte_account_at(&comm.channels[1].payload[0],e.units,&bytes) && bytes==4099);
  mesh_span_plan_free(&p.packets);
}
int main(void){
  const size_t length=3*MESH_DMA_EXTENT+4096;
  region=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);assert(region!=MAP_FAILED);
  *region=(struct hdr){.pgsz=4096,.block=1,.data_off=4096,.length=length};
  exact_and_common_cuts();affine_rows();recount_repeats();relay_prefix();fixture();
  assert(!munmap(region,length));puts("native exact byte plans PASS");return 0;
}
'''


class NativeBytePlans(unittest.TestCase):
    def compile_and_run(self, body, marker):
        cc = shutil.which('clang')
        if not cc:
            self.skipTest('clang required for native extraction')
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / 'test.c').write_text(body)
            build = subprocess.run([cc, '-std=c11', '-fblocks', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                                    '-Wno-unused-function', '-fsanitize=address,undefined', '-I', str(ROOT),
                                    str(path / 'test.c'), '-o', str(path / 'test')], text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(path / 'test')], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn(marker, run.stdout)

    def test_native_exact_placement_binding_and_relay(self):
        source = (ROOT / 'nccl-mesh.c').read_text()
        names = ('named', 'shifted', 'address_of', 'write_displacement', 'span_after', 'span_units',
                 'page_up', 'schedules_free', 'group_at', 'repeat_of', 'group_bound', 'scatter_spans', 'place_group',
                 'forwards', 'channel_account_prune', 'forwarding_gate', 'visit_crossings', 'group_entries', 'entry_logged',
                 'issue_entries')
        bodies = '\n\n'.join(function(source, name) for name in names)
        self.compile_and_run(MOCKS + bodies + CASES, 'native exact byte plans PASS')

    def test_child_bootstrap_owns_fresh_listener(self):
        source = (ROOT / 'nccl-mesh.c').read_text()
        mocks = r'''
#include "nccl.h"
#include <assert.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
struct ncclComm { pthread_mutex_t guard; int listener,*peers,nranks; };
static struct {int fd;char id[NCCL_UNIQUE_ID_BYTES];} root;
static const char *address="::1";
static ncclResult_t fabric_address(char *out,size_t size){snprintf(out,size,"%s",address);return ncclSuccess;}
'''
        bodies = '\n'.join(function(source, name) for name in
                           ('bootstrap_listen', 'bootstrap_child_id', 'bootstrap_close'))
        cases = r'''
struct child {struct ncclComm comm;ncclUniqueId id;};
static void *create(void *at){struct child *c=at;assert(!bootstrap_child_id(&c->comm,&c->id));return NULL;}
int main(void){
  uint16_t public_port=0;root.fd=bootstrap_listen("::1",&public_port);assert(root.fd>=0);
  snprintf(root.id,sizeof root.id,"::1:%u",public_port);int public_fd=root.fd;
  char public_id[NCCL_UNIQUE_ID_BYTES];memcpy(public_id,root.id,sizeof public_id);
  assert(!setenv("NCCL_COMM_ID",root.id,1));
  struct child children[2]={0};pthread_t threads[2];
  for(unsigned i=0;i<2;i++){
    children[i].comm.listener=-1;assert(!pthread_mutex_init(&children[i].comm.guard,NULL));
    assert(!pthread_create(threads+i,NULL,create,children+i));
  }
  for(unsigned i=0;i<2;i++)assert(!pthread_join(threads[i],NULL));
  assert(root.fd==public_fd && !strcmp(root.id,public_id) && fcntl(public_fd,F_GETFD)>=0);
  assert(strcmp(children[0].id.internal,children[1].id.internal));
  for(unsigned i=0;i<2;i++){
    assert(strcmp(children[i].id.internal,public_id));
    assert(fcntl(children[i].comm.listener,F_GETFD)>=0);
  }
  int fd=children[0].comm.listener;bootstrap_close(&children[0].comm);
  assert(children[0].comm.listener==-1 && fcntl(fd,F_GETFD)<0 && errno==EBADF);
  assert(fcntl(children[1].comm.listener,F_GETFD)>=0 && fcntl(public_fd,F_GETFD)>=0);
  /* A failed private bind also leaves the public bootstrap and other child alive. */
  address="2001:db8::1";assert(bootstrap_child_id(&children[0].comm,&children[0].id));
  assert(children[0].comm.listener==-1 && root.fd==public_fd);
  bootstrap_close(&children[1].comm);assert(fcntl(public_fd,F_GETFD)>=0);close(public_fd);
  for(unsigned i=0;i<2;i++)pthread_mutex_destroy(&children[i].comm.guard);
  puts("child bootstrap ownership PASS");return 0;
}
'''
        self.compile_and_run(mocks + bodies + cases, 'child bootstrap ownership PASS')


if __name__ == '__main__':
    unittest.main()
