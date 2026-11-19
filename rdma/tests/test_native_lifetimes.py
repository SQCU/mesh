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
#define NCCL_MESH_HOLDS 8
#define MESH_STREAMS NCCL_MESH_HOLDS
typedef int ncclResult_t;
enum { ncclSuccess, ncclInvalidArgument };
struct hdr { int unused; };
struct mesh_ctx { struct hdr *M; };
struct mesh_order { uint32_t slot, node, generation, padding; };
struct mesh_entry { uint64_t spans, units, word, base; };
struct channel_view { struct mesh_entry *list[2]; _Atomic uint64_t *presented[2], *ready, *sent, *landed; };
struct lease { void *address; uint32_t row; size_t bytes; void *buffer; struct mesh_lease_retire reads; int returned; };
struct hold { uint32_t node, users; struct mesh_order order; struct channel_view view; uint64_t wire, generation, presented[2]; };
struct attach { struct mesh_ctx context; struct lease *leases; size_t count; struct hold holds[NCCL_MESH_HOLDS]; struct attach *next; };
struct fabric { uint32_t nodes, cables; };
struct ncclComm { int binding; uint32_t held; struct attach *attach; uint32_t *endpoint; struct fabric fabric; };
static struct attach *attaches;
static pthread_mutex_t attach_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t completed[MESH_STREAMS], landed[MESH_STREAMS];
static struct hdr region;
static uint64_t wire[MESH_STREAMS];
static int owned[MESH_STREAMS], returned, released, pending, pinned;
/* the order a hold opened still live (its generation the order's) */
static int hold_live(const struct attach *a, uint32_t k, uint64_t expected) {
  return k < NCCL_MESH_HOLDS && a->holds[k].users && owned[k] && wire[k] == expected && a->holds[k].wire == expected;
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
  memset(&region, 0, sizeof region);
  attachment.context.M = &region;
  attachment.leases[0] = (struct lease){.address=payload[0], .row=1, .bytes=64, .buffer=payload[0]};
  attaches = &attachment;
  memset(owned, 0, sizeof owned);
  returned = released = pending = pinned = 0;
  for (uint32_t i = 0; i < MESH_STREAMS; i++) { completed[i] = landed[i] = 0; wire[i] = 100 + i; }
  owned[0] = 1;
  attachment.holds[0] = (struct hold){.users=1, .wire=100, .generation=10, .view={.sent=&completed[0], .landed=&landed[0]}};
  assert(!mesh_lease_retire_mark(&attachment.leases[0].reads, 0, 100, 10, 1000));
}
/* the communicator on hold 0 ends: the hold vacant, nothing of the bridge's to give back */
static void give(void) {
  uint32_t endpoint[1] = {0};
  struct ncclComm comm = {.binding=1, .held=1, .attach=&attachment, .endpoint=endpoint, .fabric={2, 1}};
  pthread_mutex_lock(&attach_lock);
  comm_let_go(&comm);
  pthread_mutex_unlock(&attach_lock);
  assert(!comm.binding && !comm.held && !attachment.holds[0].users && endpoint[0] == UINT32_MAX);
}
int main(void) {
  int reusable = -1;
  fixture();
  /* A thousand planned units, none posted: neither logical free nor the pool
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
  assert(!attachment.count && returned == 1 && pinned == 1);

  fixture();
  give();
  assert(!returned && attachment.count == 1); /* a live application allocation */
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && !reusable);
  /* Reacquiring the same physical wire with a larger new prefix proves nothing. */
  owned[0] = 1;
  completed[0] = 9000;
  attachment.holds[0] = (struct hold){.users=1, .wire=100, .generation=11, .view={.sent=&completed[0], .landed=&landed[0]}};
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && !reusable);
  assert(attachment.leases[0].reads.deps[0].invalid);

  /* Latch completed old reads before clearing a holding, so a pool lease does
     not lose a valid completion proof when communicator destruction resets it. */
  fixture();
  completed[0] = 1000;
  give();
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && reusable);
  assert(!returned && !attachment.leases[0].returned);

  /* ~0 is loss, never a numerical prefix beyond the final planned unit. */
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
  attachment.holds[1] = (struct hold){.users=1, .wire=101, .generation=20, .view={.sent=&completed[1], .landed=&landed[1]}};
  assert(!mesh_lease_retire_mark(&attachment.leases[0].reads, 1, 101, 20, 2000));
  completed[0] = 1000;
  completed[1] = 1999;
  assert(ncclMemFree(payload[0]) == ncclSuccess && !returned);
  completed[1] = 2000;
  leases_reap(&attachment);
  assert(returned == 1 && !attachment.count);

  /* A shared immutable table has separate SEND and LAND obligations. A fast
     outgoing prefix cannot release a table whose receives are still future. */
  fixture();
  mesh_lease_retire_destroy(&attachment.leases[0].reads);
  assert(!mesh_lease_retire_mark_use(&attachment.leases[0].reads, 0, 100, 10, 3, MESH_LEASE_SEND, 1));
  assert(!mesh_lease_retire_mark_use(&attachment.leases[0].reads, 0, 100, 10, 7, MESH_LEASE_LAND, 1));
  completed[0] = 3000; landed[0] = 6;
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && !reusable);
  assert(ncclMemFree(payload[0]) == ncclSuccess && !returned);
  assert(attachment.leases[0].reads.deps[0].done && !attachment.leases[0].reads.deps[1].done);
  landed[0] = 7; leases_reap(&attachment);
  assert(!attachment.count && returned == 1 && released == 1);

  /* Table reads: the bridge reads a table until it retires the transfer, and holds a returned lease of a client
     with a transfer not retired (mesh-dataflow.c mesh_pinned), so the hold's end is the library's whole proof. */
  fixture();
  mesh_lease_retire_destroy(&attachment.leases[0].reads);
  assert(!mesh_lease_retire_mark_use(&attachment.leases[0].reads, 0, 100, 10, 7, MESH_LEASE_LAND, 1));
  assert(ncclMemFree(payload[0]) == ncclSuccess && !returned);
  give();
  assert(returned == 1 && released == 1 && !attachment.count);

  /* A lost incoming prefix does not contaminate an already proved SEND, but
     it is not a completed LAND and cannot revive from a later numeric value. */
  fixture();
  assert(!mesh_lease_retire_mark_use(&attachment.leases[0].reads, 0, 100, 10, 7, MESH_LEASE_LAND, 1));
  completed[0] = 1000; landed[0] = UINT64_MAX;
  assert(ncclMeshMemReusable(payload[0], &reusable) == ncclSuccess && !reusable);
  assert(attachment.leases[0].reads.deps[0].done && attachment.leases[0].reads.deps[1].invalid);
  landed[0] = 9000;
  assert(ncclMemFree(payload[0]) == ncclSuccess && !returned);
  give();
  assert(returned == 1 && !attachment.count);

  /* Metadata registration is itself retained even when the bridge is already
     quiescent: unknown future dependencies cannot disappear during teardown. */
  fixture();
  assert(!mesh_lease_retire_begin(&attachment.leases[0].reads));
  assert(ncclMemFree(payload[0]) == ncclSuccess && !returned);
  give(); leases_reap(&attachment);
  assert(!returned && attachment.count == 1);
  assert(!mesh_lease_retire_end(&attachment.leases[0].reads));
  leases_reap(&attachment); assert(returned == 1 && !attachment.count);

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
        names = ['lease_read_sample', 'lease_detachable', 'lease_returnable', 'leases_reap', 'comm_let_go',
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
