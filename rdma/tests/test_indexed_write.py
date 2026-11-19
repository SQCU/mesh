"""Actual indexed WRITE packet/placement functions, with local lease boundaries.

No alternate tensor packer or planner appears in the test. The independent
oracle only maps logical indices and checks the emitted SGE addresses/lengths.
"""
from pathlib import Path
import unittest

import test_native_byte_plans as native
from test_native_lifetimes import function

ROOT = Path(__file__).resolve().parents[1]


def structure(source, name):
    start = source.index('struct '+name+' {')
    return source[start:source.index('};', start)+2]


INDEXED = r'''
static uint64_t reference(const ncclMeshSpan *view,size_t count,uint64_t logical){
  for(size_t k=0;k<count;k++){
    if(logical<view[k].bytes)return view[k].offset+logical;
    logical-=view[k].bytes;
  }
  assert(!"logical byte outside view");return 0;
}
static void packet_check(const struct write_binding_report *report,const ncclMeshSpan *views,
                         size_t stride,unsigned ranks,unsigned packet,const struct mesh_span_plan *plan){
  assert(plan->count && plan->cuts[0]==0 && plan->cuts[plan->count-1]==report[0].bytes);
  for(size_t k=0;k+1<plan->count;k++){
    uint64_t first=plan->cuts[k],last=plan->cuts[k+1];assert(last>first && last-first<=packet);
    assert(first/packet==(last-1)/packet);
    for(unsigned r=0;r<ranks;r++){
      const ncclMeshSpan *v=views+r*stride;
      uint64_t begin=0;size_t j=0;
      while(first>=begin+v[j].bytes){begin+=v[j].bytes;j++;assert(j<report[r].views);}
      assert(last<=begin+v[j].bytes);
      const uint64_t physical=report[r].phase+v[j].offset+first-begin;
      assert(physical/MESH_DMA_EXTENT==(physical+last-first-1)/MESH_DMA_EXTENT);
    }
  }
}
static void indexed_placement(void){
  enum { ranks=3, most=4, bytes=47 };
  ncclMeshSpan views[ranks][most]={{{20,7},{0,5},{20,7},{100,28}},
                                  {{37,11},{3,19},{100,17}},{{5,47}}};
  struct write_binding_report report[ranks]={{.phase=MESH_DMA_EXTENT-25,.views=4,.bytes=bytes},
    {.phase=MESH_DMA_EXTENT-41,.views=3,.bytes=bytes},{.phase=MESH_DMA_EXTENT-9,.views=1,.bytes=bytes}};
  struct mesh_span_plan packets={0};
  assert(!write_view_packets(report,&views[0][0],most,ranks,13,&packets));
  packet_check(report,&views[0][0],most,ranks,13,&packets);
  unsigned char payload[bytes];
  for(unsigned k=0;k<bytes;k++)payload[k]=(unsigned char)(reference(views[0],4,k)+1);
  for(unsigned r=0;r<ranks;r++){
    fixture();uint64_t total=0,extent=0;
    assert(!write_view_validate(views[r],report[r].views,r!=0,&total,&extent));assert(total==bytes);
    uint64_t first[most+1]={0};for(size_t k=0;k<report[r].views;k++)first[k+1]=first[k]+views[r][k].bytes;
    struct ncclMeshWritePlan plan={.packets=packets,.view=views[r],.views=report[r].views,
      .view_first=first,.view_extent=extent};
    struct where buffer=binding(report[r].phase,extent);
    struct mesh_step raw=step(r?MESH_STEP_COPY:MESH_STEP_SEND,0,bytes);
    struct steps st={.steps=&raw,.buffer=&buffer};struct call call={.bound=&st,.write_plan=&plan};
    struct message message={.c=&call,.bytes=bytes,.pages={packets.count-1}};
    struct schedule schedule=r?(struct schedule){.in=&message,.ins=1}:(struct schedule){.out=&message,.outs=1};
    assert(!place_group(NULL,&comm,&schedule,1));
    unsigned char *base=address_of(buffer);memset(base,0xE7,extent);
    unsigned char changed[128]={0};assert(extent<=sizeof changed);
    for(size_t k=0;k+1<packets.count;k++){
      const uint64_t at=packets.cuts[k],length=packets.cuts[k+1]-at;
      const uint64_t offset=reference(views[r],report[r].views,at);
      assert(message.spans[k].offset==region->data_off+report[r].phase+offset);
      assert(message.spans[k].bytes==length && mesh_dma_span_valid(region,message.spans[k]));
      if(r)memcpy((char *)region+message.spans[k].offset,payload+at,length);
      for(uint64_t j=0;j<length;j++)changed[offset+j]=1;
    }
    if(r){
      for(unsigned k=0;k<bytes;k++)assert(base[reference(views[r],report[r].views,k)]==payload[k]);
      for(uint64_t k=0;k<extent;k++)if(!changed[k])assert(base[k]==0xE7);
    }
    /* Execution checks the entire original lease, even when the first and
       last logical spans would misleadingly suggest a smaller extent. */
    leases[0].bytes=extent-1;
    assert(place_group(NULL,&comm,&schedule,1)==ncclInvalidUsage);
  }
  mesh_span_plan_free(&packets);
}
static void many_views_and_registration(void){
  const size_t n=5003;ncclMeshSpan *views=calloc(2*n,sizeof *views);assert(views);
  for(size_t k=0;k<n;k++)views[k]=(ncclMeshSpan){7*k,3};
  views[n]=(ncclMeshSpan){2,3*n};
  struct write_binding_report r[2]={{.phase=MESH_DMA_EXTENT-2,.views=n,.bytes=3*n},
                                   {.phase=MESH_DMA_EXTENT-1,.views=1,.bytes=3*n}};
  struct mesh_span_plan plan={0};assert(!write_view_packets(r,views,n,2,4096,&plan));
  assert(plan.count>4096 && plan.count<2*n);packet_check(r,views,n,2,4096,&plan);
  mesh_span_plan_free(&plan);free(views);
  ncclMeshSpan huge[2][2]={{{MESH_DMA_EXTENT-7,MESH_DMA_EXTENT+11},{19,MESH_DMA_EXTENT+11}},
                          {{23,2*MESH_DMA_EXTENT+22}}};
  r[0]=(struct write_binding_report){.phase=3,.views=2,.bytes=2*MESH_DMA_EXTENT+22};
  r[1]=(struct write_binding_report){.phase=7,.views=1,.bytes=2*MESH_DMA_EXTENT+22};
  assert(!write_view_packets(r,&huge[0][0],2,2,UINT32_MAX,&plan));
  packet_check(r,&huge[0][0],2,2,UINT32_MAX,&plan);assert(plan.count<16);mesh_span_plan_free(&plan);
}
static void invalid_views(void){
  uint64_t bytes=0,extent=0;ncclMeshSpan views[]={{5,7},{11,3}};
  assert(!write_view_validate(views,2,0,&bytes,&extent)&&bytes==10&&extent==14);
  assert(write_view_validate(views,2,1,&bytes,&extent)==EINVAL);
  views[1]=(ncclMeshSpan){12,3};assert(!write_view_validate(views,2,1,&bytes,&extent));
  views[0].bytes=0;assert(write_view_validate(views,2,0,&bytes,&extent)==EINVAL);
  views[0]=(ncclMeshSpan){UINT64_MAX,2};assert(write_view_validate(views,1,0,&bytes,&extent)==EOVERFLOW);
  views[0]=(ncclMeshSpan){0,UINT64_MAX};views[1]=(ncclMeshSpan){0,1};
  assert(write_view_validate(views,2,0,&bytes,&extent)==EOVERFLOW);
  assert(write_view_validate(NULL,1,0,&bytes,&extent)==EINVAL);
  struct write_binding_report r[2]={{0},{0}};struct mesh_span_plan plan={0};
  assert(!write_view_packets(r,NULL,0,2,3,&plan)&&plan.count==1&&!plan.cuts[0]);mesh_span_plan_free(&plan);
  r[0].views=1;r[0].bytes=1;views[0]=(ncclMeshSpan){0,1};views[1]=(ncclMeshSpan){2,1};
  assert(write_view_packets(r,views,1,2,3,&plan)==EINVAL&&!plan.cuts);
  r[1]=r[0];r[1].phase=MESH_DMA_EXTENT;
  assert(write_view_packets(r,views,1,2,3,&plan)==EINVAL&&!plan.cuts);
  r[1].phase=0;assert(write_view_packets(r,views,1,2,0,&plan)==EINVAL&&!plan.cuts);
  assert(write_view_packets(r,views,SIZE_MAX,2,3,&plan)==EOVERFLOW&&!plan.cuts);
}
static uint32_t random_state=12345;
static uint32_t sample(void){random_state=random_state*1664525u+1013904223u;return random_state;}
static void random_views(void){
  for(unsigned run=0;run<300;run++){
    enum{ranks=3,most=9};ncclMeshSpan views[ranks][most]={0};struct write_binding_report r[ranks]={0};
    const uint64_t total=1+sample()%123;const unsigned packet=1+sample()%31;
    for(unsigned rank=0;rank<ranks;rank++){
      r[rank]=(struct write_binding_report){.phase=MESH_DMA_EXTENT-1-sample()%73,.bytes=total};
      uint64_t left=total;
      while(left){
        const size_t k=r[rank].views++;const uint64_t n=k+1==most?left:1+sample()%left;
        views[rank][k]=(ncclMeshSpan){(most-1-k)*128+(sample()%3),n};left-=n;
      }
    }
    struct mesh_span_plan plan={0};assert(!write_view_packets(r,&views[0][0],most,ranks,packet,&plan));
    packet_check(r,&views[0][0],most,ranks,packet,&plan);mesh_span_plan_free(&plan);
  }
}
'''

PREPARE = r'''
static struct write_binding_report configured_reports[3];
static ncclMeshSpan configured_views[3][4];
static unsigned configuration_exchanges;
/* Configuration sockets are the test boundary. The actual preparer must hold
   ONE lock across its fixed and variable reports; no tensor address is read. */
static int prepare_exchange_locked(ncclComm_t comm,const void *local,size_t bytes,void *reports){
  assert(pthread_mutex_trylock(&comm->prepare)==EBUSY);
  if(configuration_exchanges++==0){
    assert(bytes==sizeof(struct write_binding_report));
    memcpy(reports,configured_reports,sizeof configured_reports);
    memcpy((char *)reports+comm->rank*bytes,local,bytes);
  }else{
    assert(configuration_exchanges==2 && bytes==sizeof configured_views[0]);
    memcpy(reports,configured_views,sizeof configured_views);
    memcpy((char *)reports+comm->rank*bytes,local,bytes);
  }
  return 0;
}
'''

PREPARE_CASES = r'''
static void preparation(void){
  ncclMeshSpan views[3][4]={{{20,7},{0,5},{20,7},{100,28}},{{37,11},{3,19},{100,17}},{{5,47}}};
  memcpy(configured_views,views,sizeof views);
  const unsigned count[3]={4,3,1};
  for(unsigned r=0;r<3;r++)configured_reports[r]=(struct write_binding_report){
    .phase=MESH_DMA_EXTENT-25-8*r,.bytes=47,.count=47,.packet=13,.views=count[r],
    .repeats=1,.type=ncclUint8,.source=0,.destination=2,.indexed=1};
  for(int metal=0;metal<2;metal++)for(unsigned r=0;r<3;r++){
    fixture();comm.rank=(int)r;comm.nranks=3;assert(!pthread_mutex_init(&comm.prepare,NULL));
    uint64_t bytes=0,extent=0;assert(!write_view_validate(views[r],count[r],r!=0,&bytes,&extent));
    struct where buffer=binding(configured_reports[r].phase,extent);
    ncclMeshBuffer wrapped={(void *)buffer.offset,0};
    const void *base=metal?(const void *)&wrapped:(const void *)buffer.offset;
    configuration_exchanges=0;ncclMeshWritePlan_t plan=NULL;
    ncclMeshSpan local[4];memcpy(local,views[r],sizeof local);
    /* The public native prepare call must not touch the source/landing values. */
    const size_t page=(size_t)sysconf(_SC_PAGESIZE);
    void *guard=(void *)(buffer.offset&~(page-1));
    assert(!mprotect(guard,2*page,PROT_NONE));
    assert(!ncclMeshWriteSpansPrepare(r?NULL:base,r?(void *)base:NULL,local,count[r],0,2,&comm,metal,13,&plan));
    assert(!mprotect(guard,2*page,PROT_READ|PROT_WRITE));
    assert(configuration_exchanges==2 && plan && plan->bytes==47 && plan->count==47 && plan->views==count[r]);
    assert(plan->view_extent==extent && plan->repeats==1 && !plan->stride && plan->metal==metal);
    assert(plan->source.offset==(r?0:metal?0:buffer.offset));
    for(unsigned k=0;k<47;k++)assert(write_displacement(plan,k)==reference(views[r],count[r],k));
    local[0].offset=99999;assert(plan->view[0].offset==views[r][0].offset);
    packet_check(configured_reports,&configured_views[0][0],4,3,13,&plan->packets);
    assert(!ncclMeshWritePlanDestroy(plan));
    /* Invalid local lease bounds propagate through configuration metadata,
       without creating a plan or a silent compact operand. */
    local[0]=views[r][0];local[0].offset=extent+1;configuration_exchanges=0;plan=NULL;
    assert(ncclMeshWriteSpansPrepare(r?NULL:base,r?(void *)base:NULL,local,count[r],0,2,&comm,metal,13,&plan)==ncclInvalidUsage);
    assert(!plan && configuration_exchanges==1);
    assert(!pthread_mutex_destroy(&comm.prepare));
  }
}
'''


class IndexedWrite(unittest.TestCase):
    def test_actual_indexed_packets_and_exact_native_placement(self):
        source = (ROOT/'nccl-mesh.c').read_text()
        names = ('named', 'shifted', 'address_of', 'write_displacement', 'span_after', 'span_units',
                 'page_up', 'schedules_free', 'group_at', 'repeat_of', 'group_bound', 'place_group',
                 'forwards', 'channel_account_prune', 'channel_account_append', 'present_group',
                 'write_view_compare', 'write_view_validate', 'write_view_packets')
        body = native.MOCKS + structure(source, 'write_binding_report') + '\n'.join(function(source, name) for name in names)
        cases = native.CASES.replace('int main(void){', INDEXED+'\nint main(void){').replace(
            'exact_and_common_cuts();affine_rows();recount_repeats();relay_prefix();fixture();',
            'indexed_placement();many_views_and_registration();invalid_views();random_views();fixture();')
        native.NativeBytePlans.compile_and_run(self, body+cases, 'native exact byte plans PASS')

    def test_public_prepare_copies_only_metadata_and_preserves_lease_bindings(self):
        source = (ROOT/'nccl-mesh.c').read_text()
        mocks = native.MOCKS.replace(structure(native.MOCKS, 'ncclMeshWritePlan'), structure(source, 'ncclMeshWritePlan'))
        mocks = mocks.replace('uint32_t endpoint[2]; int rank;',
            'uint32_t endpoint[2]; int rank,nranks; pthread_mutex_t prepare; _Atomic unsigned runs; struct mesh_deadline until;')
        size_start = source.index('static const size_t SIZE[')
        mocks += '\n#include <unistd.h>\n'+source[size_start:source.index(';', size_start)+1]
        names = ('named', 'shifted', 'address_of', 'write_displacement', 'write_plan_drop', 'span_after', 'span_units',
                 'page_up', 'schedules_free', 'group_at', 'repeat_of', 'group_bound', 'place_group',
                 'forwards', 'channel_account_prune', 'channel_account_append', 'present_group',
                 'write_view_compare', 'write_view_validate', 'write_view_packets', 'where_of')
        body = mocks+structure(source, 'write_binding_report')+'\n'.join(function(source, name) for name in names)
        body += PREPARE+'\n'.join(function(source, name) for name in
                                ('write_prepare', 'ncclMeshWriteSpansPrepare', 'ncclMeshWritePlanDestroy'))
        cases = native.CASES.replace('int main(void){', INDEXED+PREPARE_CASES+'\nint main(void){').replace(
            'exact_and_common_cuts();affine_rows();recount_repeats();relay_prefix();fixture();', 'preparation();fixture();')
        native.NativeBytePlans.compile_and_run(self, body+cases, 'native exact byte plans PASS')


if __name__ == '__main__':
    unittest.main()
