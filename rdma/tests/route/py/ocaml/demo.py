"""We can use OCaml: fabric.ml reads struct route_fabric records where numpy holds them. Three ways in: a numpy array
(ffi.from_buffer, no copy), a memoryview of bytes (read only, no copy), and the result shared out through DLPack.
Checked against the same summary computed by numpy on the same records."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from route import fabric  # noqa: E402
from route.native import ffi  # noqa: E402

ffi.cdef('int route_ml_init(void); size_t route_ml_fabric_size(void); int route_ml_fabric_stats(const void *fabrics, int n, double *out);', override=True)
ml = ffi.dlopen(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libroute_ml.dylib'))
assert ml.route_ml_init() == 0
print('struct route_fabric:', ml.route_ml_fabric_size(), 'bytes to the C compiler,', fabric.FABRIC.itemsize, 'to numpy')

measured = fabric.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'fabric-20261119.txt'))
rng = np.random.default_rng(20261120)
fabrics = np.empty(4096, fabric.FABRIC); fabrics[:] = measured   # the measured cluster (at the C dtype), its lanes' rates and latencies jittered
assert fabrics.dtype.itemsize == ml.route_ml_fabric_size()
lanes = int(measured['lanes'])
fabrics['rate'][:, :lanes] *= np.exp(0.3 * rng.standard_normal((4096, lanes)))
fabrics['latency'][:, :lanes] *= np.exp(0.3 * rng.standard_normal((4096, lanes)))

out = np.empty((len(fabrics), 4))
assert ml.route_ml_fabric_stats(ffi.from_buffer(fabrics), len(fabrics), ffi.from_buffer('double[]', out, require_writable=True)) == 0

# numpy's own: each compute node's ingress (its in-lanes' rates), the least; the mean lane latency
ingress = np.zeros((len(fabrics), int(measured['nodes'])))
for l in range(lanes):
    ingress[:, measured['to'][l]] += fabrics['rate'][:, l]
least = ingress[:, measured['compute'][:measured['ncompute']]].min(axis=1)
expect = np.stack([np.full(len(fabrics), lanes), np.full(len(fabrics), measured['ncompute']), least, fabrics['latency'][:, :lanes].mean(axis=1)], axis=1)
print('OCaml against numpy, 4096 fabrics: largest difference', np.abs(out - expect).max(), 'of', np.abs(expect).max())

view = memoryview(fabrics.tobytes())                 # bytes, read only: OCaml reads them in place just the same
again = np.empty_like(out)
assert ml.route_ml_fabric_stats(ffi.from_buffer(view), len(fabrics), ffi.from_buffer('double[]', again, require_writable=True)) == 0
print('through a memoryview of bytes: identical' if np.array_equal(out, again) else 'through a memoryview: DIFFERS')

shared = np.from_dlpack(out)                         # the result out through DLPack, no copy
print('through DLPack: same memory' if np.shares_memory(shared, out) else 'DLPack copied')
print('measured cluster, jittered: least compute ingress %.1f GB/s median (%.1f..%.1f), mean lane latency %.2f us median'
      % (np.median(out[:, 2]), out[:, 2].min(), out[:, 2].max(), np.median(out[:, 3]) / 1e3))
