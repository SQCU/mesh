"""Actual bridge posting and pin functions with mocked verbs, no cluster changes."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from test_native_lifetimes import function

ROOT = pathlib.Path(__file__).resolve().parents[1]

PRELUDE = r'''
#include "mesh-bridge.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
struct hdr *M;
static struct port port;
static struct mesh_queue queues[MESH_LINK_LANES];
static struct ibv_mr registration;
static struct ibv_mr *registrations[]={&registration};
static int failure;
static void link_failed(struct port *p,int64_t error,uint32_t domain){(void)p;(void)domain;failure=(int)error;}
static void lane_lost(struct port *p,uint32_t lane,int64_t error){(void)p;(void)lane;failure=(int)error;}
struct posted { uint64_t address,id; uint32_t bytes; };
static struct posted sends[2][MESH_LANE_UNITS+2],receives[2][MESH_LANE_UNITS+2];
static unsigned ns[2],nr[2];
static int post_send(struct ibv_qp *qp,struct ibv_send_wr *wr,struct ibv_send_wr **bad){
  (void)bad;unsigned lane=(unsigned)(uintptr_t)qp-1;
  assert(wr->num_sge==1 && lane<2 && ns[lane]<MESH_LANE_UNITS+2);
  sends[lane][ns[lane]++]=(struct posted){wr->sg_list->addr,wr->wr_id,wr->sg_list->length};return 0;
}
static int post_receive(struct ibv_qp *qp,struct ibv_recv_wr *wr,struct ibv_recv_wr **bad){
  (void)bad;unsigned lane=(unsigned)(uintptr_t)qp-1;
  assert(wr->num_sge==1 && lane<2 && nr[lane]<MESH_LANE_UNITS+2);
  receives[lane][nr[lane]++]=(struct posted){wr->sg_list->addr,wr->wr_id,wr->sg_list->length};return 0;
}
'''

CASES = r'''
static const uint64_t owner=37;
static struct mesh_stream *stream;
static struct mesh_dma_span *out,*in;
static struct mesh_ctx context;
static void fixture(void){
  free(port.link.sending[0].ring);free(port.link.landing[0].ring);free(M);
  memset(&port,0,sizeof port);memset(ns,0,sizeof ns);memset(nr,0,sizeof nr);failure=0;
  struct hdr shape={0};uint64_t bytes=mesh_layout(&shape,4096,1,8192);
  M=calloc(1,bytes);assert(M);*M=shape;M->version=MESH_LAYOUT;
  stream=mesh_stream_at(M,0);stream->lists=M->data_off;stream->rows=M->data_off+(1u<<19);
  stream->owner=owner;stream->live=1;stream->child=getpid();
  mesh_client_at(M,owner)->owner=owner;
  out=(void *)((char *)M+M->data_off+(1u<<20));in=out+MESH_LANE_UNITS*2+4;
  context=(struct mesh_ctx){.M=M,.client=owner};
  port.link.hold[0]=HOLD_SERVE;port.link.held[0]=owner;
  port.link.sending[0].ring=calloc(MESH_STREAM_ENTRIES,sizeof(struct taken));
  port.link.landing[0].ring=calloc(MESH_STREAM_ENTRIES,sizeof(struct taken));
  port.device.wire=(char *)M+M->data_off;port.device.extent_mrs=registrations;
  port.link.provider.queues=queues;
  for(unsigned i=0;i<2;i++){
    queues[MESH_LANE_FIRST+i]=(struct mesh_queue){.pair=(void *)(uintptr_t)(i+1),.send=post_send,.receive=post_receive};
    port.link.gate[MESH_LANE_FIRST+i].capacity.n=100000;
    port.link.lane[MESH_LANE_FIRST+i].frame_capacity=100000;
  }
}
static void present(int landing,struct mesh_dma_span *spans,uint64_t count){
  struct mesh_entry entry={(uint64_t)((char *)spans-(char *)M),count,0,0};
  assert(!mesh_stream_present(&context,0,landing,entry));
  assert(walk_take(landing?port.link.landing:port.link.sending,stream,landing));
}
static void validate(void){
  fixture();uint64_t p=M->data_off+(2u<<20);
  out[0]=(struct mesh_dma_span){p+7,1};out[1]=(struct mesh_dma_span){p+4090,20};
  struct mesh_entry e={(uint64_t)((char *)out-(char *)M),2,0,0};assert(mesh_entry_valid(M,e));
  assert(!mesh_dma_span_valid(M,(struct mesh_dma_span){p,0}));
  assert(!mesh_dma_span_valid(M,(struct mesh_dma_span){M->data_off-1,1}));
  assert(!mesh_dma_span_valid(M,(struct mesh_dma_span){M->length-1,2}));
  assert(!mesh_dma_span_valid(M,(struct mesh_dma_span){p,UINT64_MAX}));
  struct hdr huge=*M;huge.length=huge.data_off+2*MESH_DMA_EXTENT;
  assert(mesh_dma_span_valid(&huge,(struct mesh_dma_span){huge.data_off+MESH_DMA_EXTENT-7,7}));
  assert(!mesh_dma_span_valid(&huge,(struct mesh_dma_span){huge.data_off+MESH_DMA_EXTENT-7,8}));
  struct mesh_entry bad=e;bad.spans++;assert(!mesh_entry_valid(M,bad));
  bad=e;bad.units=UINT64_MAX;assert(!mesh_entry_valid(M,bad));
  bad=e;bad.spans=M->length-8;assert(!mesh_entry_valid(M,bad));
  bad=e;bad.base=UINT64_MAX-1;assert(!mesh_entry_valid(M,bad));
  bad=e;bad.word=3;assert(!mesh_entry_valid(M,bad));
  out[1].bytes=0;assert(mesh_stream_present(&context,0,0,e)==EINVAL && stream->sends==0);
}
static void exact_io(void){
  fixture();uint64_t source=M->data_off+(2u<<20),dest=M->data_off+(3u<<20);
  const uint64_t sizes[]={1,3,4097,19,4096};
  for(unsigned i=0;i<5;i++){
    out[i]=(struct mesh_dma_span){source+31+8192*i,sizes[i]};
    in[i]=(struct mesh_dma_span){dest+119+8192*i,sizes[i]};
    memset((char *)M+in[i].offset-1,0xA7,(size_t)sizes[i]+2);
    memset((char *)M+out[i].offset,17+i,(size_t)sizes[i]);
  }
  present(0,out,5);present(1,in,5);
  stream_post(&port,0,12);stream_send(&port,0,12);
  assert(!ns[0] && !ns[1] && nr[0]==3 && nr[1]==2);
  stream->ready=1;stream_send(&port,0,12);assert(ns[0]==1 && ns[1]==0);
  stream->ready=5;stream_send(&port,0,12);assert(!failure);
  assert(ns[0]==3 && ns[1]==2 && nr[0]==3 && nr[1]==2);
  for(unsigned i=0;i<5;i++){
    unsigned lane=i%2,u=i/2;struct posted a=sends[lane][u],b=receives[lane][u];
    assert(a.bytes==sizes[i] && b.bytes==sizes[i]);
    assert(a.address==(uintptr_t)M+out[i].offset && b.address==(uintptr_t)M+in[i].offset);
    memcpy((void *)(uintptr_t)b.address,(void *)(uintptr_t)a.address,a.bytes);
    assert(*((unsigned char *)M+in[i].offset-1)==0xA7);
    assert(*((unsigned char *)M+in[i].offset+sizes[i])==0xA7);
    assert(!memcmp((char *)M+in[i].offset,(char *)M+out[i].offset,(size_t)sizes[i]));
    struct mesh_dma_span pin;assert(mesh_pin_load(mesh_unit_pin(M,stream->rows,0,lane,u),u,&pin));
    assert(pin.offset==out[i].offset && pin.bytes==sizes[i]);
  }
  /* Two lanes' completion prefix cannot skip a slower earlier unit. */
  walk_count(port.link.sending,(struct mesh_units){lanes_through((uint64_t[]){2,0})},&stream->sent);
  assert(stream->sent==1);
  walk_count(port.link.sending,(struct mesh_units){lanes_through((uint64_t[]){2,1})},&stream->sent);
  assert(stream->sent==3);
  walk_count(port.link.sending,(struct mesh_units){lanes_through((uint64_t[]){3,2})},&stream->sent);
  assert(stream->sent==5 && port.link.sending[0].done==1);
}
static void frame_credits(void){
  fixture();uint64_t p=M->data_off+(2u<<20);
  for(unsigned i=0;i<6;i++)out[i]=in[i]=(struct mesh_dma_span){p+8192*i,i<2?4097:1};
  for(unsigned i=0;i<2;i++){
    port.link.gate[MESH_LANE_FIRST+i].capacity.n=2;
    port.link.lane[MESH_LANE_FIRST+i].frame_capacity=2;
  }
  present(0,out,6);present(1,in,6);stream->ready=6;
  stream_post(&port,0,12);stream_send(&port,0,12);
  assert(ns[0]==1 && ns[1]==1 && nr[0]==1 && nr[1]==1);
  for(unsigned lane=0;lane<2;lane++){
    gate_retire(port.link.gate+MESH_LANE_FIRST+lane,sends[lane][0].id);
    port.link.lane[MESH_LANE_FIRST+lane].done++;
    lane_landed(&port,MESH_LANE_FIRST+lane,receives[lane][0].id);
  }
  stream_post(&port,0,12);stream_send(&port,0,12);
  assert(ns[0]==3 && ns[1]==3 && nr[0]==3 && nr[1]==3 && !failure);
  /* Even one-byte requests cannot overrun the pin ring. */
  fixture();p=M->data_off+(2u<<20);
  for(unsigned i=0;i<2*MESH_LANE_UNITS+2;i++)out[i]=in[i]=(struct mesh_dma_span){p,1};
  present(0,out,2*MESH_LANE_UNITS+2);present(1,in,2*MESH_LANE_UNITS+2);stream->ready=2*MESH_LANE_UNITS+2;
  stream_post(&port,0,12);stream_send(&port,0,12);
  assert(ns[0]==MESH_LANE_UNITS && ns[1]==MESH_LANE_UNITS);
  assert(nr[0]==MESH_LANE_UNITS && nr[1]==MESH_LANE_UNITS && !failure);
}
static void pinning(void){
  fixture();const uint64_t p=M->data_off;
  /* A one-byte tail of the next allocation pins it even though the DMA began elsewhere. */
  mesh_page(M)[50].mapping=100;mesh_page(M)[51].mapping=101;
  struct mesh_dma_span span={p+101*4096-1,2};
  assert(unit_told(stream,0,0,0,span,owner));
  assert(mesh_pinned(M,50,1) && mesh_pinned(M,51,1));
  stream->live=0;assert(mesh_pinned(M,51,1));stream->live=1;
  /* Logical alias/scatter table IDs have no bearing on the physical backing pin. */
  mesh_page(M)[90].mapping=100;mesh_page(M)[91].mapping=700;
  assert(mesh_pinned(M,90,2));
  mesh_page(M)[90].mapping=600;assert(!mesh_pinned(M,90,2));
  /* Exact interval end does not overlap the next backing page. */
  mesh_pin_store(mesh_unit_pin(M,stream->rows,0,0,0),0,(struct mesh_dma_span){p+100*4096,4096});
  assert(mesh_pinned(M,50,1) && !mesh_pinned(M,51,1));
  /* Loss or a mismatched/reused slot is not retirement. */
  mesh_pin_store(mesh_unit_pin(M,stream->rows,0,0,0),MESH_LANE_UNITS,span);
  assert(mesh_pinned(M,50,1));
  stream->posts[0].done[0]=1;assert(!mesh_pinned(M,50,2));
  /* A given holding can publish no future unit; its tentative posted increment rolls back. */
  stream->owner=owner|MESH_STREAM_GIVEN;
  assert(!unit_told(stream,0,0,1,span,owner) && stream->posts[0].posted[0]==1);
  /* Incoming writes retain destination backing too; process exit retires actual QPs. */
  stream->owner=owner;assert(unit_told(stream,1,1,0,span,owner));
  assert(mesh_pinned(M,51,1));stream->child=0;assert(!mesh_pinned(M,51,1));
}
static void cpu_metadata(void){
  fixture();
  const uint64_t table=(char *)out-(char *)M,word=M->data_off+(3u<<20);
  out[0]=(struct mesh_dma_span){M->data_off+(2u<<20),19};
  struct taken entry={.spans=out,.units={1},.word=(void *)((char *)M+word),.whole=1};
  /* Death after the reader's ownership check: no DMA has been posted, yet the
     exact table and external dependency word remain protected. */
  metadata_pin(stream,&entry);assert(holding_kept(stream,owner));
  mesh_client_at(M,owner)->owner=0;
  assert(mesh_interval_pinned(M,(struct mesh_dma_span){table,16}));
  assert(mesh_interval_pinned(M,(struct mesh_dma_span){word,8}));
  assert(!mesh_interval_pinned(M,(struct mesh_dma_span){table+16,1}));
  assert(!mesh_interval_pinned(M,(struct mesh_dma_span){word+8,1}));
  assert(!stream->posts[0].posted[0] && !stream->posts[1].posted[0]);
  metadata_pin(stream,NULL);
  assert(!mesh_interval_pinned(M,(struct mesh_dma_span){table,16}));
  assert(!mesh_interval_pinned(M,(struct mesh_dma_span){word,8}));
  /* A publication/clear snapshot in progress conservatively retains storage;
     child exit retires CPU scopes too, including a fault-abandoned odd scope. */
  stream->metadata.sequence++;
  assert(mesh_interval_pinned(M,(struct mesh_dma_span){word+4096,8}));
  stream->child=0;assert(!mesh_interval_pinned(M,(struct mesh_dma_span){word+4096,8}));
  const uint64_t old_sequence=stream->metadata.sequence;
  metadata_pin(stream,NULL);assert(!(stream->metadata.sequence&1) && stream->metadata.sequence>old_sequence);
  /* Death before publication: the table can already be reclaimed. Protect its
     page against CPU access, then execute the actual SEND and LAND paths. The
     post-publication ownership check must stop before even reading the table. */
  fixture();out[0]=(struct mesh_dma_span){M->data_off+(2u<<20),19};in[0]=out[0];
  present(0,out,1);present(1,in,1);stream->ready=1;
  mesh_client_at(M,owner)->owner=0;
  const size_t page=(size_t)getpagesize();
  void *out_page=(void *)((uintptr_t)out&~(uintptr_t)(page-1));
  void *in_page=(void *)((uintptr_t)in&~(uintptr_t)(page-1));
  assert(!mprotect(out_page,page,PROT_NONE));assert(!mprotect(in_page,page,PROT_NONE));
  const uint32_t served=UINT32_C(3)<<MESH_LANE_FIRST;
  stream_send(&port,0,served);stream_post(&port,0,served);
  assert(!ns[0] && !ns[1] && !nr[0] && !nr[1] && !failure);
  assert(!stream->metadata.bytes && !stream->metadata.word);
  assert(!mprotect(out_page,page,PROT_READ|PROT_WRITE));assert(!mprotect(in_page,page,PROT_READ|PROT_WRITE));
}
int main(void){validate();exact_io();frame_credits();pinning();cpu_metadata();
  free(port.link.sending[0].ring);free(port.link.landing[0].ring);free(M);
  puts("byte span validation, exact DMA, frame/unit capacities and physical pin lifetimes PASS");return 0;
}
'''


class ByteSpanTests(unittest.TestCase):
    def test_actual_bridge_with_mock_verbs(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        link = (ROOT / 'mesh-link.c').read_text()
        flow = (ROOT / 'mesh-dataflow.c').read_text()
        call = (ROOT / 'mesh-call.c').read_text()
        bodies = [function(link, name) for name in (
            'stream_of', 'stream_words', 'dma_span', 'region_word', 'walk_take', 'walk_unit', 'walk_count',
            'holding_kept', 'metadata_pin', 'unit_told', 'stream_send', 'stream_post', 'lane_landed')]
        bodies += [function(flow, name) for name in ('mesh_interval_pinned', 'mesh_pinned')]
        bodies += [function(call, name) for name in ('stream_held', 'stream_owned', 'mesh_stream_present')]
        with tempfile.TemporaryDirectory() as directory:
            source, binary = pathlib.Path(directory) / 'spans.c', pathlib.Path(directory) / 'spans'
            source.write_text(PRELUDE + '\n'.join(bodies) + CASES)
            built = subprocess.run([compiler, '-std=c11', '-fblocks', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                                    '-Wno-unused-function', '-fsanitize=address,undefined', '-I', str(ROOT),
                                    str(source), '-o', str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('lifetimes PASS', run.stdout)


if __name__ == '__main__':
    unittest.main()
