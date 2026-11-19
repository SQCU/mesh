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
static struct mesh_entry entries[2];
static int presentations, reject_land, detached;
static _Atomic int returns;
static uint32_t returned[32];
static struct mesh_section data={.first=10,.pages=1,.bytes=4096,.count=4,.stride=2};
static struct mesh_section table={.first=50,.pages=1,.bytes=6*sizeof(struct mesh_dma_span),.count=1,.stride=1};
void *mesh_section_address(struct mesh_ctx *c,struct mesh_section s,uint32_t slot){
  return (char *)c->M+c->M->data_off+(s.first==table.first?(1u<<20):(2u<<20)+slot*4096);
}
int mesh_stream_present(struct mesh_ctx *c,uint32_t stream,int land,struct mesh_entry e){
  assert(stream==0 && mesh_entry_valid(c->M,e));
  presentations++;
  if(land && reject_land)return EINVAL;
  entries[land]=e;
  return 0;
}
void mesh_stream_give(struct mesh_ctx *c,uint32_t stream){
  assert(stream==0);mesh_stream_at(c->M,stream)->owner=c->client|MESH_STREAM_GIVEN;
}
int mesh_lease_return(struct mesh_ctx *c,uint32_t row){
  (void)c;returned[atomic_fetch_add(&returns,1)]=row;return 0;
}
int mesh_detach(struct mesh_ctx *c){(void)c;detached++;return 0;}
'''
CASES = r'''
static void fixture(void){
  free(context.M);memset(&context,0,sizeof context);
  context.M=calloc(1,8u<<20);assert(context.M);
  context.M->length=8u<<20;context.M->data_off=4096;
  context.M->block=1;context.M->pgsz=4096;context.M->stream_off=512;context.M->streams=1;
  context.client=37;mesh_stream_at(context.M,0)->owner=context.client;
  presentations=reject_land=detached=0;returns=0;memset(entries,0,sizeof entries);
}
static void *quiesce(void *ignored){
  (void)ignored;
  while(mesh_stream_at(context.M,0)->owner!=(context.client|MESH_STREAM_GIVEN)){}
  /* The client has given its holding; returning the metadata now would race a
     bridge CPU reader. release_probe must still retain every allocation. */
  assert(returns==0);
  mesh_stream_at(context.M,0)->owner=MESH_STREAM_GIVEN;
  return NULL;
}
int main(void){
  fixture();assert(!present(&context,0,3,data,table));assert(presentations==2);
  struct mesh_dma_span *spans=mesh_section_address(&context,table,0);
  const uint64_t source=(char *)mesh_section_address(&context,data,0)-(char *)context.M;
  assert(entries[0].units==3 && entries[1].units==3 && !entries[0].word && !entries[1].word);
  assert(entries[0].spans==(uint64_t)((char *)spans-(char *)context.M));
  assert(entries[1].spans==entries[0].spans+3*sizeof *spans);
  assert(entries[0].spans!=table.first && entries[0].spans!=data.first);
  for(int k=0;k<3;k++){
    assert(spans[k].offset==source && spans[k].bytes==4096);
    assert(spans[3+k].offset==source+(uint64_t)(k+1)*4096 && spans[3+k].bytes==4096);
  }
  release_probe(&context,0,1,data,table);
  assert(returns==5 && detached==1 && returned[0]==50);
  for(int k=0;k<4;k++)assert(returned[k+1]==10+(uint32_t)k*2);

  fixture();reject_land=1;
  assert(present(&context,0,3,data,table)==EINVAL && presentations==1);
  pthread_t thread;assert(!pthread_create(&thread,NULL,quiesce,NULL));
  release_probe(&context,0,0,data,table);
  pthread_join(thread,NULL);assert(returns==5 && detached==1);

  fixture();release_probe(&context,MESH_ABSENT,0,data,table);
  assert(returns==5 && detached==1);
  free(context.M);puts("diagnostic exact descriptors and CPU-reader retirement PASS");return 0;
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
