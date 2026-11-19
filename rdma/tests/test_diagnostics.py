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
static struct mesh_declaration declared;
static _Atomic uint64_t issued_word;
static uint64_t issued_steps;
static int declares, closes, reject, detached;
static _Atomic int returns;
static uint32_t returned[32];
static struct mesh_section data={.first=10,.pages=1,.bytes=4096,.count=4,.stride=2};
static struct mesh_section table={.first=50,.pages=1,.bytes=4096,.count=1,.stride=1};
void *mesh_section_address(struct mesh_ctx *c,struct mesh_section s,uint32_t slot){
  return (char *)c->M+c->M->data_off+(s.first==table.first?(1u<<20):(2u<<20)+slot*4096);
}
/* the node's declaration: the program kept, its counter the mock's */
int mesh_declare(struct mesh_ctx *c,const struct mesh_order *o,const struct mesh_declaration *p,struct mesh_run *r){
  (void)c;assert(o->node==9);
  if(reject)return EINVAL;
  declared=*p;declares++;
  *r=(struct mesh_run){.order=*o,.program=3,.issued=&issued_word};
  return 0;
}
int mesh_close(struct mesh_ctx *c,struct mesh_run *r){(void)c;assert(r->program==3);closes++;issued_steps=atomic_load(&issued_word);return 0;}
int mesh_lease_return(struct mesh_ctx *c,uint32_t row){
  (void)c;returned[atomic_fetch_add(&returns,1)]=row;return 0;
}
int mesh_detach(struct mesh_ctx *c){(void)c;detached++;return 0;}
struct probe { _Atomic uint64_t ready,sent,landed,padding; struct mesh_crossing table[]; };
'''
CASES = r'''
static void fixture(void){
  free(context.M);memset(&context,0,sizeof context);
  context.M=calloc(1,8u<<20);assert(context.M);
  context.M->length=8u<<20;context.M->data_off=4096;context.M->rows=2048;
  context.M->block=1;context.M->pgsz=4096;
  context.client=37;
  declares=closes=reject=detached=0;returns=0;memset(&declared,0,sizeof declared);atomic_store(&issued_word,0);
}
int main(void){
  const struct mesh_order order={.node=9};
  fixture();
  struct probe *p=mesh_section_address(&context,table,0);
  struct mesh_run run={0};
  assert(!present(&context,&order,3,data,p,&run));
  /* one program of three steps, one execution, all issued at once */
  assert(declares==1 && declared.period==3 && declared.count==1 && atomic_load(&issued_word)==3);
  assert(declared.transfers[0]==3 && declared.transfers[1]==3 && declared.pieces[0]==3 && declared.pieces[1]==3);
  const uint64_t source=(char *)mesh_section_address(&context,data,0)-(char *)context.M,base=(char *)p-(char *)context.M;
  const struct mesh_crossing *sends=(const struct mesh_crossing *)((char *)context.M+declared.table[0]);
  const struct mesh_crossing *lands=(const struct mesh_crossing *)((char *)context.M+declared.table[1]);
  for(uint32_t k=0;k<3;k++){
    const struct mesh_crossing *send=sends+k,*land=lands+k;
    const struct mesh_dma_span *s=(const struct mesh_dma_span *)((char *)context.M+send->spans);
    const struct mesh_dma_span *r=(const struct mesh_dma_span *)((char *)context.M+land->spans);
    /* step k: one block each way, the SEND gated on ready reaching k+1, done words raised to k+1 in order */
    assert(send->units==1 && land->units==1 && send->first==k && land->first==k && send->step==k && land->step==k);
    assert(s->offset==source && s->bytes==4096 && r->offset==source+(uint64_t)(k+1)*4096 && r->bytes==4096);
    assert(send->gate==base+offsetof(struct probe,ready) && send->gate_base==k && !land->gate);
    assert(send->done==base+offsetof(struct probe,sent) && land->done==base+offsetof(struct probe,landed));
    assert(send->done_value==k+1 && land->done_value==k+1);
  }
  release_probe(&context,&run,data,table);
  assert(closes==1 && issued_steps==3 && returns==5 && detached==1 && returned[0]==50);
  for(int k=0;k<4;k++)assert(returned[k+1]==10+(uint32_t)k*2);

  fixture();reject=1;p=mesh_section_address(&context,table,0);
  struct mesh_run refused={0};
  assert(present(&context,&order,3,data,p,&refused)==EINVAL && !atomic_load(&issued_word));
  release_probe(&context,&refused,data,table);
  assert(!closes && returns==5 && detached==1);
  free(context.M);puts("diagnostic program and retirement PASS");return 0;
}
'''


class DiagnosticTests(unittest.TestCase):
    def test_library_probe_program_and_retirement(self):
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
