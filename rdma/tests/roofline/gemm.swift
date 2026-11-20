// gemm TYPE: C = A . B^T on this Mac's GPU at large square sizes, for TYPE fp32, fp16 or int8 (int8 x int8 -> int32),
// through Metal 4's matmul2d (MetalPerformancePrimitives), every output tile and SIMD-group count swept; fp32 and fp16
// also through MPS's matrix multiply for reference. Prints each size's best, in TFLOP/s (int8: TOP/s).
//   swiftc -O gemm.swift -o gemm && ./gemm fp32 && ./gemm fp16 && ./gemm int8
import Foundation
import Metal
import MetalPerformanceShaders

let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let type = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "int8"
let (inT, accT, bytesIn, bytesOut) = type == "fp32" ? ("float", "float", 4, 4) : type == "fp16" ? ("half", "float", 2, 4) : ("int8_t", "int32_t", 1, 4)

func source(_ tm: Int, _ tn: Int, _ sg: Int) -> String { """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
kernel void mm(device \(inT) *a [[buffer(0)]], device \(inT) *b [[buffer(1)]], device \(accT) *c [[buffer(2)]], constant uint3 &p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = p.x, N = p.y, K = p.z;
  auto A = tensor<device \(inT), dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(K, M));
  auto B = tensor<device \(inT), dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(K, N));
  auto C = tensor<device \(accT), dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>(N, M));
  constexpr auto d = matmul2d_descriptor(\(tm), \(tn), static_cast<int>(dynamic_extent), false, true);
  matmul2d<d, execution_simdgroups<\(sg)>> op;
  auto tA = A.slice(0, tg.y * \(tm)); auto tB = B.slice(0, tg.x * \(tn));
  auto r = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), \(accT)>();
  op.run(tA, tB, r);
  r.store(C.slice(tg.x * \(tn), tg.y * \(tm)));
}
""" }

// the same product with each K-block of A's and B's tile staged once in threadgroup memory (every SIMD group reads it there)
func staged(_ tm: Int, _ tn: Int, _ sg: Int, _ kb: Int) -> String { """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
kernel void mm(device \(inT) *a [[buffer(0)]], device \(inT) *b [[buffer(1)]], device \(accT) *c [[buffer(2)]], constant uint3 &p [[buffer(3)]],
               uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const int M = p.x, N = p.y, K = p.z, m0 = tg.y * \(tm), n0 = tg.x * \(tn);
  threadgroup \(inT) sa[\(tm) * \(kb)], sb[\(tn) * \(kb)];
  auto SA = tensor<threadgroup \(inT), dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(\(kb), \(tm)));
  auto SB = tensor<threadgroup \(inT), dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(\(kb), \(tn)));
  auto C = tensor<device \(accT), dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>(N, M));
  constexpr auto d = matmul2d_descriptor(\(tm), \(tn), \(kb), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<\(sg)>> op;
  auto r = op.get_destination_cooperative_tensor<decltype(SA), decltype(SB), \(accT)>();
  for (uint16_t i = 0; i < r.get_capacity(); ++i) if (r.is_valid_element(i)) r[i] = 0;
  for (int k0 = 0; k0 < K; k0 += \(kb)) {
    for (uint i = tid; i < \(tm * kb); i += \(32 * sg)) sa[i] = a[size_t(m0 + i / \(kb)) * K + k0 + i % \(kb)];
    for (uint i = tid; i < \(tn * kb); i += \(32 * sg)) sb[i] = b[size_t(n0 + i / \(kb)) * K + k0 + i % \(kb)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(SA, SB, r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  r.store(C.slice(n0, m0));
}
""" }

func time(_ reps: Int, _ encode: (MTLCommandBuffer) -> Void) -> Double {
  var best = 1e9
  for _ in 0..<reps { let cb = queue.makeCommandBuffer()!; encode(cb); cb.commit(); cb.waitUntilCompleted(); best = min(best, cb.gpuEndTime - cb.gpuStartTime) }
  return best
}
for n in [4096, 8192, 16384] {
  let ops = 2 * Double(n) * Double(n) * Double(n)
  let a = dev.makeBuffer(length: n * n * bytesIn, options: .storageModePrivate)!, b = dev.makeBuffer(length: n * n * bytesIn, options: .storageModePrivate)!
  let c = dev.makeBuffer(length: n * n * bytesOut, options: .storageModePrivate)!
  var best = (0.0, "")
  for (tm, tn) in [(32, 32), (64, 32), (32, 64), (64, 64), (128, 64), (64, 128), (128, 128), (256, 128), (128, 256)] {
    for sg in [1, 2, 4, 8, 16] {
      guard let lib = try? dev.makeLibrary(source: source(tm, tn, sg), options: nil), let f = lib.makeFunction(name: "mm"),
            let p = try? dev.makeComputePipelineState(function: f), 32 * sg <= p.maxTotalThreadsPerThreadgroup else { continue }
      var dims = SIMD3<UInt32>(UInt32(n), UInt32(n), UInt32(n))
      let t = time(3) { cb in
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(p); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1); e.setBuffer(c, offset: 0, index: 2); e.setBytes(&dims, length: 16, index: 3)
        e.dispatchThreadgroups(MTLSize(width: n / tn, height: n / tm, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * sg, height: 1, depth: 1))
        e.endEncoding()
      }
      if ops / t / 1e12 > best.0 { best = (ops / t / 1e12, "matmul2d tile \(tm)x\(tn), \(sg) SIMD groups") }
    }
  }
  for (tm, tn) in [(64, 64), (128, 64), (64, 128), (128, 128)] {
    for kb in [16, 32, 64, 128] where (tm + tn) * kb * bytesIn <= 32768 {
      for sg in [4, 8, 16] {
        guard let lib = try? dev.makeLibrary(source: staged(tm, tn, sg, kb), options: nil), let f = lib.makeFunction(name: "mm"),
              let p = try? dev.makeComputePipelineState(function: f), 32 * sg <= p.maxTotalThreadsPerThreadgroup else { continue }
        var dims = SIMD3<UInt32>(UInt32(n), UInt32(n), UInt32(n))
        let t = time(3) { cb in
          let e = cb.makeComputeCommandEncoder()!
          e.setComputePipelineState(p); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1); e.setBuffer(c, offset: 0, index: 2); e.setBytes(&dims, length: 16, index: 3)
          e.dispatchThreadgroups(MTLSize(width: n / tn, height: n / tm, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * sg, height: 1, depth: 1))
          e.endEncoding()
        }
        if ops / t / 1e12 > best.0 { best = (ops / t / 1e12, "staged tile \(tm)x\(tn), K-block \(kb), \(sg) SIMD groups") }
      }
    }
  }
  print(String(format: "%@ %5d^3: %6.1f T/s  (%@)", type, n, best.0, best.1)); fflush(stdout)
  if type != "int8" {   // MPS's matrix multiply, the platform's own
    let dt: MPSDataType = type == "fp32" ? .float32 : .float16
    let rowA = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * bytesIn, dataType: dt)
    let rowC = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * bytesIn, dataType: dt)
    let ma = MPSMatrix(buffer: a, descriptor: rowA), mb = MPSMatrix(buffer: b, descriptor: rowA)
    let cc = dev.makeBuffer(length: n * n * bytesIn, options: .storageModePrivate)!, mc = MPSMatrix(buffer: cc, descriptor: rowC)
    let k = MPSMatrixMultiplication(device: dev, transposeLeft: false, transposeRight: true, resultRows: n, resultColumns: n, interiorColumns: n, alpha: 1, beta: 0)
    let t = time(3) { cb in k.encode(commandBuffer: cb, leftMatrix: ma, rightMatrix: mb, resultMatrix: mc) }
    print(String(format: "%@ %5d^3: %6.1f T/s  (MPS)", type, n, ops / t / 1e12)); fflush(stdout)
  }
}
