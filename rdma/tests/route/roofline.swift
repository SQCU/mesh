// swiftc -O roofline.swift -o roofline && ./roofline   (the int8 ceiling the trainer's kernels are measured against)
// The GPU's int8 x int8 -> int32 roofline through Metal 4's matmul2d: C [M][N] = A [M][K] . B [N][K]^T, swept over
// output tile sizes and SIMD groups a threadgroup, at a square and at the routing network's shapes
import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
func source(_ tm: Int, _ tn: Int, _ sg: Int) -> String { """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
kernel void mm(device int8_t *a [[buffer(0)]], device int8_t *b [[buffer(1)]], device int32_t *c [[buffer(2)]], constant uint3 &p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = p.x, N = p.y, K = p.z, m0 = tg.y * \(tm), n0 = tg.x * \(tn);
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(K, M));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(K, N));
  auto C = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>(N, M));
  constexpr auto d = matmul2d_descriptor(\(tm), \(tn), static_cast<int>(dynamic_extent), false, true);
  matmul2d<d, execution_simdgroups<\(sg)>> op;
  auto tA = A.slice(0, m0); auto tB = B.slice(0, n0);
  auto r = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
  for (uint16_t i = 0; i < r.get_capacity(); ++i) if (r.is_valid_element(i)) r[i] = 0;
  op.run(tA, tB, r);
  r.store(C.slice(n0, m0));
}
""" }
let shapes = [(4096, 4096, 4096), (8192, 8192, 8192), (262144, 64, 64), (262144, 192, 64), (262144, 64, 192), (262144, 128, 64), (262144, 192, 192)]
for (M, N, K) in shapes {
  let a = dev.makeBuffer(length: M * K, options: .storageModePrivate)!, b = dev.makeBuffer(length: N * K, options: .storageModePrivate)!, c = dev.makeBuffer(length: M * N * 4, options: .storageModePrivate)!
  var best = (0.0, "")
  for (tm, tn) in [(32, 32), (64, 32), (64, 64), (128, 64), (64, 128), (128, 128)] where tn <= N || N >= 64 {
    for sg in [4, 8] {
      guard let lib = try? dev.makeLibrary(source: source(tm, tn, sg), options: nil), let f = lib.makeFunction(name: "mm"), let p = try? dev.makeComputePipelineState(function: f) else { continue }
      var dims = SIMD3<UInt32>(UInt32(M), UInt32(N), UInt32(K)), t = 1e9
      for _ in 0..<5 {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(p); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1); e.setBuffer(c, offset: 0, index: 2); e.setBytes(&dims, length: 16, index: 3)
        e.dispatchThreadgroups(MTLSize(width: (N + tn - 1) / tn, height: (M + tm - 1) / tm, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * sg, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        t = min(t, cb.gpuEndTime - cb.gpuStartTime)
      }
      let tops = 2 * Double(M) * Double(N) * Double(K) / t / 1e12
      if tops > best.0 { best = (tops, "tile \(tm)x\(tn), \(sg) SIMD groups") }
    }
  }
  print(String(format: "M %6d N %5d K %5d: %6.1f TOP/s (%@)", M, N, K, best.0, best.1))
}
