"""Production AOT byte ancestry distinguishes identity forwarding from overwrite."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

CASE = r'''
#include "mesh-routed.c"
#include <assert.h>
#include <stdio.h>
int main(void){
  /* An old eight-byte SEND is relayed as an interior four-byte slice. The
     returning COPY overwrites precisely that slice before the entire old
     SEND need finish. Different endpoint buffer offsets are intentional. */
  struct routed_piece p[6]={
    {.rank=0,.transfer={.step={MESH_STEP_SEND,1,0,0,0,{1,1,8}}}},
    {.rank=0,.transfer={.step={MESH_STEP_COPY,2,2,0,2,{1,1,4}}}},
    {.rank=1,.transfer={.step={MESH_STEP_COPY,0,0,0,10,{1,1,8}}}},
    {.rank=1,.transfer={.step={MESH_STEP_SEND,2,1,0,12,{1,1,4}}}},
    {.rank=2,.transfer={.step={MESH_STEP_COPY,1,1,0,20,{1,1,4}}}},
    {.rank=2,.transfer={.step={MESH_STEP_SEND,0,2,0,20,{1,1,4}}}}
  };
  const size_t send[]={0,3,5},land[]={2,4,1};int proved=0;
  assert(!byte_retired(p,send,land,1,0,1,&proved)&&proved);
  /* Publication of old byte 2 does not retire old byte 3. Returning byte 2
     to byte 3 can corrupt the pending read; a same-transaction shortcut
     would incorrectly certify this one-byte shifted overwrite. */
  p[1].transfer.step.first=3;
  assert(!byte_retired(p,send,land,1,0,1,&proved)&&!proved);
  p[1].transfer.step.first=2;
  p[3].transfer.step.buffer=1;
  assert(!byte_retired(p,send,land,1,0,1,&proved)&&!proved);
  puts("pointwise partial relay lifetime proofs PASS");return 0;
}
'''


class RoutedByteProofs(unittest.TestCase):
    def test_partial_forwarding_requires_corresponding_source_bytes(self):
        compiler = shutil.which('clang') or shutil.which('cc')
        if compiler is None:
            self.skipTest('C compiler unavailable')
        with tempfile.TemporaryDirectory(prefix='mesh-byte-proof-') as directory:
            source, executable = Path(directory)/'proof.c', Path(directory)/'proof'
            source.write_text(CASE)
            built = subprocess.run([compiler, '-std=gnu11', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-I', str(ROOT), str(source),
                str(ROOT/'mesh-collective.c'), str(ROOT/'mesh-fabric.c'), '-lm', '-o', str(executable)],
                capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(ran.returncode, 0, ran.stdout+ran.stderr)
            self.assertIn('pointwise partial relay lifetime proofs PASS', ran.stdout)


if __name__ == '__main__':
    unittest.main()
