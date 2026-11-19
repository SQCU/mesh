"""Library diagnostic descriptor/lifetime checks; never attach a live bridge."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_native_lifetimes import function

ROOT = Path(__file__).resolve().parents[1]
MOCKS = r'''
#include "mesh-call.h"
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
static struct mesh_ctx context;
static struct mesh_crossing appended[2][8];
static int appends[2], reject_land, detached, full;
static _Atomic int returns;
static uint32_t returned[32];
static struct mesh_section data={.first=10,.pages=1,.bytes=4096,.count=4,.stride=2};
static struct mesh_section table={.first=50,.pages=1,.bytes=4096,.count=1,.stride=1};
void *mesh_section_address(struct mesh_ctx *c,struct mesh_section s,uint32_t slot){
  return (char *)c->M+c->M->data_off+(s.first==table.first?(1u<<20):(2u<<20)+slot*4096);
}
/* the node's order: every transfer valid, each way's appended in order; one ring momentarily full once */
int mesh_order_append(struct mesh_ctx *c,const struct mesh_order *o,int way,const struct mesh_crossing *t){
  assert(o->node==9 && mesh_crossing_valid(c->M,t) && appends[way]<8);
  if(full){full--;return EAGAIN;}
  if(way && reject_land)return EINVAL;
  appended[way][appends[way]++]=*t;
  return 0;
}
int mesh_lease_return(struct mesh_ctx *c,uint32_t row){
  (void)c;returned[atomic_fetch_add(&returns,1)]=row;return 0;
}
int mesh_detach(struct mesh_ctx *c){(void)c;detached++;return 0;}
struct probe { _Atomic uint64_t ready,sent,landed,padding; struct mesh_dma_span span[]; };
'''
CASES = r'''
static void fixture(void){
  free(context.M);memset(&context,0,sizeof context);
  context.M=calloc(1,8u<<20);assert(context.M);
  context.M->length=8u<<20;context.M->data_off=4096;context.M->rows=2048;
  context.M->block=1;context.M->pgsz=4096;
  context.client=37;
  appends[0]=appends[1]=reject_land=detached=full=0;returns=0;memset(appended,0,sizeof appended);
}
int main(void){
  const struct mesh_order order={.node=9};
  fixture();full=1;
  struct probe *p=mesh_section_address(&context,table,0);
  assert(!present(&context,&order,3,data,p));assert(appends[0]==3 && appends[1]==3);
  const uint64_t source=(char *)mesh_section_address(&context,data,0)-(char *)context.M,base=(char *)p-(char *)context.M;
  for(uint32_t k=0;k<3;k++){
    const struct mesh_crossing *send=&appended[0][k],*land=&appended[1][k];
    const struct mesh_dma_span *s=(const struct mesh_dma_span *)((char *)context.M+send->spans);
    const struct mesh_dma_span *r=(const struct mesh_dma_span *)((char *)context.M+land->spans);
    /* round k: one block each way, the SEND gated on ready reaching k+1, done words raised to k+1 in order */
    assert(send->units==1 && land->units==1 && s->offset==source && s->bytes==4096);
    assert(r->offset==source+(uint64_t)(k+1)*4096 && r->bytes==4096);
    assert(send->gate==base+offsetof(struct probe,ready) && send->base==k && !land->gate);
    assert(send->done==base+offsetof(struct probe,sent) && land->done==base+offsetof(struct probe,landed));
    assert(send->value==k+1 && land->value==k+1);
  }
  release_probe(&context,data,table);
  assert(returns==5 && detached==1 && returned[0]==50);
  for(int k=0;k<4;k++)assert(returned[k+1]==10+(uint32_t)k*2);

  fixture();reject_land=1;p=mesh_section_address(&context,table,0);
  assert(present(&context,&order,3,data,p)==EINVAL && !appends[0]);
  release_probe(&context,data,table);
  assert(returns==5 && detached==1);
  free(context.M);puts("diagnostic transfers and retirement PASS");return 0;
}
'''


class DiagnosticTests(unittest.TestCase):
    def test_library_probe_descriptors_and_retirement(self):
        compiler = shutil.which('cc')
        if compiler is None:
            self.skipTest('C compiler unavailable')
        source = (ROOT/'mesh-diagnostics.c').read_text()
        body = '\n'.join(function(source, name) for name in ('present', 'release_probe'))
        with tempfile.TemporaryDirectory() as folder:
            path, binary = Path(folder)/'diagnostic.c', Path(folder)/'diagnostic'
            path.write_text(MOCKS+body+CASES)
            compiled = subprocess.run([compiler, '-std=c11', '-g', '-Wall', '-Wextra', '-Werror', '-pthread',
                '-fsanitize=address,undefined', '-I', str(ROOT), str(path), '-o', str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout+compiled.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('retirement PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
