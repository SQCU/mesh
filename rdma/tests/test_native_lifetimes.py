"""Exercise actual nccl-mesh ownership functions with mocked device boundaries.

The production function bodies are extracted, not reproduced. Hardware stream
ownership, completed prefixes and bridge pin handback are controlled here; this
does not claim to test concurrent provider/CQ progress on a real device.
"""
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def function(source, name):
    """Read a C definition, excluding prototypes and calls, preserving its body."""
    start = re.search(r'^([^\n;{}]*\b' + re.escape(name) + r'\([^;{}]*\))\s*\{', source, re.M)
    if start is None:
        raise AssertionError(f'production definition missing: {name}')
    # Ignore braces in comments and string literals while locating its end.
    token = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    depth = 0
    for match in token.finditer(source, start.end() - 1):
        if match.group() == '{':
            depth += 1
        elif match.group() == '}':
            depth -= 1
            if not depth:
                return source[start.start():match.end()]
    raise AssertionError(f'unclosed production definition: {name}')


MOCKS = r'''
#include "mesh-lease-retire.h"
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#define MESH_STREAMS 8
typedef int ncclResult_t;
enum { ncclSuccess, ncclInvalidArgument };
struct mesh_ctx { int unused; };
struct mesh_stream_view { _Atomic uint64_t *sent; };
struct lease { void *address; uint32_t row; size_t bytes; void *buffer; struct mesh_lease_retire reads; int returned; };
struct hold { uint32_t node, stream, users, padding; struct mesh_stream_view view; uint64_t wire, generation; };
struct attach { struct mesh_ctx context; struct lease *leases; size_t count; struct hold holds[MESH_STREAMS]; struct attach *next; };
struct ncclComm { int binding; uint32_t held; struct attach *attach; };
static struct attach *attaches;
static pthread_mutex_t attach_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t completed[MESH_STREAMS];
static uint64_t wire[MESH_STREAMS];
static int owned[MESH_STREAMS], returned, released, given, pending, pinned;
static int mesh_stream_held(struct mesh_ctx *c, uint32_t stream, uint64_t expected) {
  (void)c;
  return owned[stream] && wire[stream] == expected;
}
static void mesh_stream_give(struct mesh_ctx *c, uint32_t stream) {
  (void)c; owned[stream] = 0; given++;
}
static int mesh_lease_return(struct mesh_ctx *c, uint32_t row) {
  (void)c; (void)row; returned++; if (pending) pinned++; return 0;
}
static void metal_release(void *buffer) { (void)buffer; released++; }
static void attach_settle(void) {}
static int fail(int result, const char *format, ...) { (void)format; return result; }
'''


CASES = r'''
static unsigned char payload[2][64];
static struct attach attachment;
static void fixture(void) {
  for (size_t i = 0; i < attachment.count; i++) mesh_lease_retire_destroy(&attachment.leases[i].reads);
  free(attachment.leases);
  memset(&attachment, 0, sizeof attachment);
  attachment.leases = calloc(2, sizeof *attachment.leases);
  assert(attachment.leases);
  attachment.count = 1;
  attachment.leases[0] = (struct lease){.address=payload[0], .row=1, .bytes=64, .buffer=payload[0]};
  attaches = &attachment;
  memset(owned, 0, sizeof owned);
  returned = released = given = pending = pinned = 0;
  for (uint32_t i = 0; i < MESH_STREAMS; i++) { completed[i] = 0; wire[i] = 100 + i; }
  owned[0] = 1;
  attachment.holds[0] = (struct hold){.stream=0, .users=1, .wire=100, .generation=10, .view={&completed[0]}};
  assert(!mesh_lease_retire_mark(&attachment.leases[0].reads, 0, 100, 10, 1000));
}
static void give(void) {
  struct ncclComm comm = {.binding=1, .held=1, .attach=&attachment};
  pthread_mutex_lock(&attach_lock);
  comm_let_go(&comm);
  pthread_mutex_unlock(&attach_lock);
  assert(!comm.binding && !comm.held);
}
int main(void) {
  int reusable = -1;
  fixture();
  /* A thousand planned pages, none posted: neither logical free nor the pool
     can reuse the source merely because no current request pins it yet. */
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && !reusable);
  assert(ncclMemFree(payload[0]) == ncclSuccess);
  assert(attachment.count == 1 && attachment.leases[0].returned && !returned);
  completed[0] = 999;
  leases_reap(&attachment);
  assert(attachment.count == 1 && !returned);
  completed[0] = 1000;
  leases_reap(&attachment);
  assert(!attachment.count && returned == 1 && released == 1);

  /* Giving ends future presentation; posted bytes are transferred to driver
     pins, not declared reusable by the application. */
  fixture();
  pending = 1;
  assert(ncclMemFree(payload[0]) == ncclSuccess);
  assert(!returned);
  give();
  assert(given == 1 && !attachment.count && returned == 1 && pinned == 1);

  fixture();
  give();
  assert(!returned && attachment.count == 1); /* a live application allocation */
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && !reusable);
  /* Reacquiring the same physical wire with a larger new prefix proves nothing. */
  owned[0] = 1;
  completed[0] = 9000;
  attachment.holds[0] = (struct hold){.stream=0, .users=1, .wire=100, .generation=11, .view={&completed[0]}};
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && !reusable);
  assert(attachment.leases[0].reads.deps[0].invalid);

  /* Latch completed old reads before clearing a holding, so a pool lease does
     not lose a valid completion proof when communicator destruction resets it. */
  fixture();
  completed[0] = 1000;
  give();
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && reusable);
  assert(!returned && !attachment.leases[0].returned);

  /* ~0 is loss, never a numerical prefix beyond the final planned page. */
  fixture();
  completed[0] = UINT64_MAX;
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && !reusable);
  assert(ncclMemFree(payload[0]) == ncclSuccess);
  assert(!returned && attachment.count == 1);
  completed[0] = 9000;
  leases_reap(&attachment);
  assert(!returned); /* a poisoned proof never revives */
  pending = 1;
  give();
  assert(returned == 1 && pinned == 1);

  /* All physical channels remain obligations, including parallel cables. */
  fixture();
  owned[1] = 1;
  attachment.holds[1] = (struct hold){.stream=1, .users=1, .wire=101, .generation=20, .view={&completed[1]}};
  assert(!mesh_lease_retire_mark(&attachment.leases[0].reads, 1, 101, 20, 2000));
  completed[0] = 1000;
  completed[1] = 1999;
  assert(ncclMemFree(payload[0]) == ncclSuccess && !returned);
  completed[1] = 2000;
  leases_reap(&attachment);
  assert(returned == 1 && !attachment.count);

  fixture();
  for (size_t i = 0; i < attachment.count; i++) mesh_lease_retire_destroy(&attachment.leases[i].reads);
  free(attachment.leases);
  puts("actual nccl source lifetime functions: all mocked ownership checks passed");
  return 0;
}
'''


class NativeLifetimeTests(unittest.TestCase):
    def test_actual_native_ownership_functions(self):
        compiler = shutil.which('cc')
        if compiler is None:
            self.skipTest('C compiler unavailable')
        source = (ROOT / 'nccl-mesh.c').read_text()
        names = ['lease_read_sample', 'lease_returnable', 'leases_reap', 'comm_let_go',
                 'lease_at', 'ncclMeshMemReusable', 'ncclMemFree']
        program = MOCKS + '\n'.join(function(source, name) for name in names) + CASES
        with tempfile.TemporaryDirectory(prefix='mesh-native-lifetimes-') as tmp:
            src, executable = pathlib.Path(tmp) / 'test.c', pathlib.Path(tmp) / 'test'
            src.write_text(program)
            built = subprocess.run([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                                    '-fsanitize=address,undefined', '-g', '-pthread', '-I', str(ROOT),
                                    str(src), '-o', str(executable)], text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(executable)], text=True, capture_output=True)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn('all mocked ownership checks passed', ran.stdout)


if __name__ == '__main__':
    unittest.main()
