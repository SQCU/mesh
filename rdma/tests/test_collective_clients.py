"""Actual client adapters preserve payload/layout metadata at the native boundary.

These tests do not simulate a network or certify the native route compiler. The
NumPy module runs with a recorded CFFI boundary; the production Objective-C issue
function is compiled against the public header and exercised for CPU and Metal.
Set MESH_MICROBENCH_SOURCE when the matching microbench checkout is elsewhere.
"""
import importlib.util
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

import cffi
import numpy as np

from test_native_lifetimes import function

ROOT = Path(__file__).resolve().parents[1]
MICROBENCH = Path(os.environ.get('MESH_MICROBENCH_SOURCE',
    str(ROOT.parents[2]/'metal-microbench')))


class Boundary:
    def __init__(self, ffi):
        self.ffi = ffi
        self.calls = []
        self.constants = {}

    def __getattr__(self, name):
        if name.startswith(('ncclFloat', 'ncclBfloat', 'ncclInt', 'ncclUint')) or name in {
                'ncclSum', 'ncclProd', 'ncclMax', 'ncclMin', 'ncclAvg'}:
            return self.constants.setdefault(name, len(self.constants)+1)
        def record(*args):
            self.calls.append((name, args))
            if name in {'ncclMeshWritePrepare', 'ncclMeshWriteRowsPrepare', 'ncclMeshWriteSpansPrepare'}:
                args[-1][0] = self.ffi.cast('ncclMeshWritePlan_t', 2000+len(self.calls))
            return 0
        return record


class NumpyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.ffi = cffi.FFI()
        self.ffi.cdef('typedef void *ncclComm_t; typedef struct ncclMeshWritePlan *ncclMeshWritePlan_t; '
                     'typedef struct ncclMeshSpan { uint64_t offset, bytes; } ncclMeshSpan;')
        self.lib = Boundary(self.ffi)
        self.allocations = []
        def lease(length):
            block = bytearray(length)
            self.allocations.append(block)
            return block
        def check(result):
            self.assertEqual(result, 0)
        fake = types.SimpleNamespace(ffi=self.ffi, lib=self.lib, check=check, lease=lease)
        spec = importlib.util.spec_from_file_location('test_actual_mesh_mpi', ROOT/'mesh_mpi.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'mesh': fake}):
            spec.loader.exec_module(self.module)
        self.comm = self.module.Comm.__new__(self.module.Comm)
        self.comm.rank, self.comm.size = 1, 3
        self.comm.comm = self.ffi.cast('ncclComm_t', 1234)

    def call(self, name):
        self.assertEqual(len(self.lib.calls), 1)
        actual, args = self.lib.calls[0]
        self.assertEqual(actual, name)
        self.assertEqual(args[-2], self.comm.comm)
        self.assertEqual(args[-1], self.ffi.NULL)
        return args

    def address(self, pointer):
        return int(self.ffi.cast('size_t', pointer))

    def test_ragged_gather_preserves_zero_share_and_exact_byte_counts(self):
        # Structured elements use ncclUint8 and multiply counts by itemsize,
        # never allocation pages or DMA credit quanta.
        dtype = np.dtype([('payload', 'u1', 3)])
        receive = np.empty(22, dtype=dtype)
        source = np.empty(0, dtype=dtype)
        self.comm.Allgatherv(source, [receive, ([3, 0, 19], [0, 3, 3])])
        args = self.call('ncclMeshAllGatherV')
        self.assertEqual([args[2][k] for k in range(3)], [9, 0, 57])
        self.assertEqual(args[3], self.lib.ncclUint8)
        self.assertEqual(self.address(args[0]), source.ctypes.data)
        self.assertEqual(self.address(args[1]), receive.ctypes.data)
        self.assertEqual(self.allocations, [])

    def test_in_place_gather_uses_original_subrange_address(self):
        self.comm.rank = 2
        receive = np.empty(22, dtype=np.float32)
        self.comm.Allgatherv(self.module.IN_PLACE, [receive, [3, 0, 19]])
        args = self.call('ncclMeshAllGatherV')
        self.assertEqual(self.address(args[0]), receive.ctypes.data+3*4)
        self.assertEqual(self.address(args[1]), receive.ctypes.data)
        self.assertEqual(self.allocations, [])

    def test_ragged_reduce_scatter_keeps_zero_output_and_operation(self):
        source, receive = np.empty(22, np.float32), np.empty(0, np.float32)
        self.comm.Reduce_scatter(source, receive, [3, 0, 19], self.module.MAX)
        args = self.call('ncclMeshReduceScatterV')
        self.assertEqual([args[2][k] for k in range(3)], [3, 0, 19])
        self.assertEqual(args[3:5], (self.lib.ncclFloat32, self.module.MAX))
        self.assertEqual(self.address(args[0]), source.ctypes.data)

    def test_byte_broadcast_and_equal_exchange_do_not_round(self):
        a = np.empty(21, dtype=np.dtype('V3'))
        self.comm.Bcast(a, root=2)
        args = self.call('ncclBroadcast')
        self.assertEqual(args[2:5], (63, self.lib.ncclUint8, 2))
        self.lib.calls.clear()
        b = np.empty_like(a)
        self.comm.Alltoall(a, b)
        args = self.call('ncclAlltoAll')
        self.assertEqual(args[2:4], (21, self.lib.ncclUint8))

    def test_equal_exchange_cannot_silently_drop_tail_or_overrun_receive(self):
        for source, target in ((np.empty(22, 'u1'), np.empty(22, 'u1')),
                               (np.empty(21, 'u1'), np.empty(20, 'u1')),
                               (np.empty(21, 'f4'), np.empty(21, 'u1'))):
            with self.assertRaisesRegex(ValueError, 'equal typed'):
                self.comm.Alltoall(source, target)
        self.assertEqual(self.lib.calls, [])

    def test_whole_function_metadata_delegates_to_shared_controller(self):
        calls = [{'what': 0, 'type': 7, 'count': 22, 'counts': [3, 0, 19], 'repeat': 31}]
        demands = {'source': 'native', 'bytes': [0, 22, 8, 17, 0, 11, 7, 13, 0]}
        seen = []
        library = types.SimpleNamespace(
            declare=lambda handle, passed: seen.append((handle, passed)) or 'declared',
            function_demands=lambda handle: seen.append(handle) or demands)
        with patch.dict(sys.modules, {'controller': library}):
            self.assertEqual(self.comm.Plan(calls), 'declared')
            self.assertIs(self.comm.PlanDemands(), demands)
        self.assertEqual(seen, [(self.comm.comm, calls), self.comm.comm])
        self.assertIs(seen[0][1], calls)
        self.assertEqual(self.lib.calls, [])

    def test_directed_writes_bind_independent_exact_extents_once(self):
        a, b = self.module.empty(7, 'u1'), self.module.empty(13, 'f4')
        outgoing = self.comm.Write_prepare(a, None, 1, 0)
        incoming = self.comm.Write_prepare(None, b, 2, 1)
        self.assertEqual([name for name, _ in self.lib.calls], ['ncclMeshWritePrepare'] * 2)
        send, receive = [args for _, args in self.lib.calls]
        self.assertEqual(send[2:6], (7, self.lib.ncclUint8, 1, 0))
        self.assertEqual(receive[2:6], (13, self.lib.ncclFloat32, 2, 1))
        self.assertEqual(self.address(send[0]), a.ctypes.data)
        self.assertEqual(self.address(receive[1]), b.ctypes.data)
        self.assertEqual((send[1], receive[0]), (self.ffi.NULL, self.ffi.NULL))
        outgoing.Execute(); incoming.Execute()
        self.assertEqual(self.lib.calls[2][1], (outgoing.handle, self.ffi.NULL))
        self.assertEqual(self.lib.calls[3][1], (incoming.handle, self.ffi.NULL))
        outgoing.Free(); incoming.Free()

    def test_region_allocation_is_shared_with_numpy_views(self):
        a = self.module.empty((7, 3), np.uint8)
        self.assertEqual(len(self.allocations[0]), 21)
        a[2, 1] = 97
        self.assertEqual(self.allocations[0][7], 97)
        self.assertTrue(np.shares_memory(a, a[1:]))

    def test_prepared_write_retains_original_view_and_reuses_native_handle(self):
        self.comm.rank = 0
        owner = self.module.empty(22, np.uint8)
        view = owner[3:19]
        weak = weakref.ref(view)
        plan = self.comm.Write_prepare(view, None, 0, 2, max_packet=257)
        name, args = self.lib.calls[0]
        self.assertEqual(name, 'ncclMeshWritePrepare')
        self.assertEqual(self.address(args[0]), owner.ctypes.data+3)
        self.assertEqual(args[1], self.ffi.NULL)
        self.assertEqual(args[2:11], (16, self.lib.ncclUint8, 0, 2, self.comm.comm, 0, 1, 0, 257))
        del view
        gc.collect()
        self.assertIsNotNone(weak(), 'fixed native bindings must retain the view/lease owner')
        plan.Execute()
        plan()
        self.assertEqual([name for name, _ in self.lib.calls],
            ['ncclMeshWritePrepare', 'ncclMeshWriteExecute', 'ncclMeshWriteExecute'])
        self.assertEqual(self.lib.calls[1][1], (plan.handle, self.ffi.NULL))
        plan.Free()
        gc.collect()
        self.assertIsNone(weak())

    def test_indexed_write_preserves_order_offsets_and_original_owner(self):
        self.comm.rank = 0
        owner = self.module.empty(128, np.uint8)
        view = owner[3:]
        retained = weakref.ref(view)
        plan = self.comm.Write_spans_prepare(view, None, [(20, 7), (0, 5), (20, 7), (100, 8)], 0, 2, max_packet=13)
        name, args = self.lib.calls[0]
        self.assertEqual(name, 'ncclMeshWriteSpansPrepare')
        self.assertEqual(self.address(args[0]), owner.ctypes.data+3)
        self.assertEqual(args[1], self.ffi.NULL)
        self.assertEqual([(args[2][k].offset, args[2][k].bytes) for k in range(args[3])],
                         [(20, 7), (0, 5), (20, 7), (100, 8)])
        self.assertEqual(args[3:9], (4, 0, 2, self.comm.comm, 0, 13))
        del view
        gc.collect()
        self.assertIsNotNone(retained())
        plan.Execute(); plan.Execute()
        self.assertEqual([name for name, _ in self.lib.calls],
                         ['ncclMeshWriteSpansPrepare', 'ncclMeshWriteExecute', 'ncclMeshWriteExecute'])
        self.assertEqual(len(self.allocations), 1, 'indexed preparation cannot allocate a compact numerical operand')
        plan.Free()
        gc.collect()
        self.assertIsNone(retained())

    def test_reversed_view_uses_shared_metadata_mapper_without_packing(self):
        self.comm.rank = 0
        owner = self.module.empty((3, 17), np.uint8)
        view = owner[:, 9:1:-2]
        spec = importlib.util.spec_from_file_location('actual_byte_views', ROOT/'views.py')
        mapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mapper)
        with patch.dict(sys.modules, {'views': mapper}):
            plan = self.comm.Write_view_prepare(view, None, 0, 2)
        name, args = self.lib.calls[0]
        self.assertEqual(name, 'ncclMeshWriteSpansPrepare')
        self.assertEqual(self.address(args[0]), owner.ctypes.data+3)
        self.assertEqual([(args[2][k].offset, args[2][k].bytes) for k in range(args[3])],
                         [(r*17+x, 1) for r in range(3) for x in (6, 4, 2, 0)])
        self.assertEqual(len(self.allocations), 1)
        plan.Free()
        with self.assertRaisesRegex(ValueError, 'already freed'):
            plan.Execute()
        plan.Free()
        self.assertEqual(sum(name == 'ncclMeshWritePlanDestroy' for name, _ in self.lib.calls), 1)

    def test_affine_rows_keep_different_pitches_and_odd_exact_payloads(self):
        for rank, pitch in enumerate((263, 271, 283)):
            self.comm.rank = rank
            backing = self.module.empty(5003 * pitch, np.uint8)
            rows = np.ndarray((5003, 257), dtype=np.uint8, buffer=backing, strides=(pitch, 1))
            self.lib.calls.clear()
            plan = self.comm.Write_rows_prepare(rows if rank == 0 else None,
                rows if rank != 0 else None, 0, 2, row_bytes=257, rows=5003, max_packet=4096)
            name, args = self.lib.calls[0]
            self.assertEqual(name, 'ncclMeshWriteRowsPrepare')
            self.assertEqual(self.address(args[0 if rank == 0 else 1]), rows.ctypes.data)
            self.assertEqual(args[1 if rank == 0 else 0], self.ffi.NULL)
            self.assertEqual(args[2:10], (257, 5003, pitch, 0, 2, self.comm.comm, 0, 4096))
            self.assertEqual(plan.arrays, (rows,))
            plan.Free()

    def test_affine_ndim_validation_never_packs_or_touches_values(self):
        backing = self.module.empty(4096, np.uint8)
        view = np.ndarray((2, 3, 7), dtype=np.uint8, buffer=backing, strides=(33, 11, 1))
        self.assertEqual(self.module._row_pitch(view, 7, 6), 11)
        stepped = backing[3:39:3]
        self.assertEqual(self.module._row_pitch(stepped, 1, 12), 3)
        nonaffine = np.ndarray((2, 3, 7), dtype=np.uint8, buffer=backing, strides=(100, 11, 1))
        with self.assertRaisesRegex(ValueError, 'indexed descriptor'):
            self.module._row_pitch(nonaffine, 7, 6)
        with self.assertRaises(ValueError):
            self.module._row_pitch(view[..., ::-1], 7, 6)
        with self.assertRaises(ValueError):
            self.module._row_pitch(view, 6, 7)
        with self.assertRaises(ValueError):
            self.module._row_pitch(np.empty(3, object), 8, 3)
        self.assertEqual(len(self.allocations), 1)
        self.assertEqual(self.lib.calls, [])

    def test_communicator_closes_bound_plans_before_native_destroy(self):
        landing = self.module.empty(3, np.uint8)
        plan = self.comm.Write_prepare(None, landing, 0, 2)
        self.comm.Free()
        self.assertEqual([name for name, _ in self.lib.calls],
            ['ncclMeshWritePrepare', 'ncclMeshWritePlanDestroy', 'ncclCommDestroy'])
        self.assertEqual(plan.handle, self.ffi.NULL)


OBJC_MOCKS = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <stdatomic.h>
#include <errno.h>
#include "nccl.h"
struct MeshRank { atomic_int cancelled; };
enum Collective { CollectiveNone, CollectiveGather, CollectiveBroadcast, CollectiveSum, CollectiveMax, CollectiveReduce };
struct MeshCrossing {
    struct MeshCrossing *next; enum Collective collective; int refused,root;
    NSUInteger rooted,count; size_t element; ncclDataType_t type; ncclRedOp_t reduction;
    ncclComm_t comm; NSArray *members; NSString *op; BOOL counted; size_t pieces[];
};
static ncclComm_t communicator(struct MeshRank *rank,NSArray *members) { (void)rank;(void)members;return NULL; }
static struct { int kind,root; const void *source; void *target; size_t count,counts[3]; ncclDataType_t type;
    ncclRedOp_t op; ncclComm_t comm; cudaStream_t stream; } observed;
static ncclResult_t record(int kind,const void *a,void *b,size_t count,ncclDataType_t type,ncclRedOp_t op,int root,
                          ncclComm_t comm,cudaStream_t stream) {
    observed.kind=kind;observed.source=a;observed.target=b;observed.count=count;observed.type=type;observed.op=op;
    observed.root=root;observed.comm=comm;observed.stream=stream;return ncclSuccess;
}
ncclResult_t ncclMeshAllGatherV(const void *a,void *b,const size_t *counts,ncclDataType_t type,ncclComm_t comm,cudaStream_t s) {
    memcpy(observed.counts,counts,sizeof observed.counts);return record(1,a,b,0,type,ncclSum,0,comm,s);
}
ncclResult_t ncclBroadcast(const void *a,void *b,size_t n,ncclDataType_t type,int root,ncclComm_t comm,cudaStream_t s) {
    return record(2,a,b,n,type,ncclSum,root,comm,s);
}
ncclResult_t ncclAllReduce(const void *a,void *b,size_t n,ncclDataType_t type,ncclRedOp_t op,ncclComm_t comm,cudaStream_t s) {
    return record(3,a,b,n,type,op,0,comm,s);
}
ncclResult_t ncclReduce(const void *a,void *b,size_t n,ncclDataType_t type,ncclRedOp_t op,int root,ncclComm_t comm,cudaStream_t s) {
    return record(4,a,b,n,type,op,root,comm,s);
}
const char *ncclGetLastError(ncclComm_t comm) { (void)comm;return "mock error"; }
@interface TestBuffer : NSObject { @public unsigned char bytes[128]; }
- (void *)contents;
@end
@implementation TestBuffer
- (void *)contents { return bytes; }
@end
'''
OBJC_CASES = r'''
int main(void) { @autoreleasepool {
    struct MeshRank rank={0};struct MeshCrossing *c=calloc(1,sizeof *c+3*sizeof(size_t));assert(c);
    c->comm=(ncclComm_t)1234;c->count=3;c->members=@[@0,@1,@2];c->op=@"test";
    c->pieces[0]=3;c->pieces[1]=0;c->pieces[2]=19;c->collective=CollectiveGather;c->counted=YES;
    TestBuffer *a=[TestBuffer new],*b=[TestBuffer new];
    ncclMeshBuffer buffers[2]={{(void *)a,7},{(void *)b,11}};ncclMeshStream stream={0};
    for(int metal=0;metal<2;metal++) {
        ncclMeshStream *s=metal?&stream:NULL;
        assert(!issue(&rank,c,buffers,17,s));
        assert(observed.kind==1&&observed.type==ncclUint8&&observed.comm==c->comm&&observed.stream==s);
        assert(observed.counts[0]==3&&observed.counts[1]==0&&observed.counts[2]==19);
        assert(observed.source==(metal?(void *)&buffers[0]:(void *)(a->bytes+7)));
        assert(observed.target==(metal?(void *)&buffers[1]:(void *)(b->bytes+11)));
        c->counted=NO;c->element=2;assert(!issue(&rank,c,buffers,5,s));
        assert(observed.counts[0]==30&&observed.counts[1]==0&&observed.counts[2]==190);
        c->collective=CollectiveBroadcast;c->root=2;c->element=3;
        assert(!issue(&rank,c,buffers,7,s));assert(observed.kind==2&&observed.count==21&&observed.root==2);
        c->collective=CollectiveMax;c->type=ncclFloat16;c->reduction=ncclMax;
        assert(!issue(&rank,c,buffers,19,s));assert(observed.kind==3&&observed.count==19&&observed.op==ncclMax&&observed.type==ncclFloat16);
        c->collective=CollectiveReduce;assert(!issue(&rank,c,buffers,13,s));
        assert(observed.kind==4&&observed.count==13&&observed.op==ncclSum&&observed.root==2);
        c->collective=CollectiveGather;c->counted=YES;
    }
    c->refused=EOVERFLOW;assert(issue(&rank,c,buffers,3,NULL)==EOVERFLOW);
    c->refused=0;rank.cancelled=1;assert(issue(&rank,c,buffers,3,NULL)==ECANCELED);
    free(c);puts("MeshRank exact CPU/Metal adapter PASS");
} return 0; }
'''


class NativeCallerTests(unittest.TestCase):
    def test_actual_objc_crossing_keeps_host_and_metal_bindings(self):
        if sys.platform != 'darwin' or shutil.which('clang') is None:
            self.skipTest('Objective-C Foundation/Metal SDK unavailable')
        source = MICROBENCH/'mesh_rank.m'
        if not source.exists():
            self.skipTest('matching microbench source unavailable: set MESH_MICROBENCH_SOURCE')
        actual = function(source.read_text(), 'issue')
        with tempfile.TemporaryDirectory() as folder:
            path, binary = Path(folder)/'client.m', Path(folder)/'client'
            path.write_text(OBJC_MOCKS+actual+OBJC_CASES)
            compiled = subprocess.run(['clang','-std=gnu11','-g','-fsanitize=undefined',
                '-I',str(ROOT),'-framework','Foundation','-framework','Metal',str(path),'-o',str(binary)],
                capture_output=True,text=True)
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            result = subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('adapter PASS',result.stdout)


if __name__ == '__main__':
    unittest.main()
