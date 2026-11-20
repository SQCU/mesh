// The backward and the optimizer (route.h), on the GPU. For a batch of decision points and the rows whose contexts were
// drawn from: the forward again (policy.swift's), each task's surrogate sum(w . dw_t) (dw from the environment's
// credit, already scaled by the advantages), each one's gradient in theta by the graph's autodiff, and their Gram
// matrix. Then Jacobian descent (Quinton & Rey 2024, UPGrad: each task's gradient projected onto the dual cone of the
// 3 x P Jacobian, the projections averaged) and Adam, in place on theta and its moments.
import Foundation
import Metal
import MetalPerformanceShadersGraph

final class Learn {
  let g = MPSGraph(), theta: MPSGraphTensor, lanes: MPSGraphTensor, mask: MPSGraphTensor, tokens: MPSGraphTensor, index: MPSGraphTensor, dw: MPSGraphTensor
  let grads: [MPSGraphTensor], gram: MPSGraphTensor
  let B: Int, R: Int
  init(_ lay: Layout, decisions B: Int, rows R: Int) {
    self.B = B; self.R = R
    theta = g.placeholder(shape: [NSNumber(value: lay.count)], dataType: .float32, name: "theta")
    lanes = g.placeholder(shape: [NSNumber(value: B), NSNumber(value: L), NSNumber(value: IN)], dataType: .float32, name: "lanes")
    mask = g.placeholder(shape: [NSNumber(value: B), NSNumber(value: L)], dataType: .float32, name: "mask")
    tokens = g.placeholder(shape: [NSNumber(value: R), NSNumber(value: IN)], dataType: .float32, name: "tokens")
    index = g.placeholder(shape: [NSNumber(value: R)], dataType: .int32, name: "index")
    dw = g.placeholder(shape: [NSNumber(value: TASKS), NSNumber(value: R), NSNumber(value: FS)], dataType: .float32, name: "dw")
    let w = forward(g, lay, theta: theta, lanes: lanes, mask: mask, tokens: tokens, index: index, decisions: B, rows: R)
    var gs: [MPSGraphTensor] = []
    for t in 0..<TASKS {
      let d = g.reshape(g.sliceTensor(dw, dimension: 0, start: t, length: 1, name: nil), shape: [NSNumber(value: R), NSNumber(value: FS)], name: nil)
      let loss = g.reductionSum(with: g.multiplication(w, d, name: nil), axes: [0, 1], name: nil)
      gs.append(g.gradients(of: loss, with: [theta], name: nil)[theta]!)
    }
    grads = gs
    var dots: [MPSGraphTensor] = []
    for a in 0..<TASKS { for b in 0..<TASKS { dots.append(g.reductionSum(with: g.multiplication(gs[a], gs[b], name: nil), axes: [0], name: nil)) } }
    gram = g.concatTensors(dots, dimension: 0, name: nil)
  }
}

// UPGrad's weights from the Gram matrix G = J J^T: for each task i, u_i = argmin_(u >= e_i) u^T G u (the dual of projecting
// g_i onto {x : J x >= 0}, whose projection is J^T u_i); the aggregate J^T (sum_i u_i) / 3. A 3-variable box QP, by
// projected gradient (step 1 / (3 max |G|), a bound on G's largest eigenvalue)
func upgrad(_ G: [[Double]]) -> [Double] {
  var top = 0.0
  for a in 0..<3 { for b in 0..<3 { top = max(top, abs(G[a][b])) } }
  let step = 1 / max(3 * top, 1e-300)
  var total: [Double] = [0, 0, 0]
  for i in 0..<3 {
    var u: [Double] = [0, 0, 0]; u[i] = 1
    for _ in 0..<500 {
      for t in 0..<3 {
        var gt = 0.0
        for s in 0..<3 { gt += G[t][s] * u[s] }
        u[t] = max(u[t] - step * 2 * gt, t == i ? 1 : 0)
      }
    }
    for t in 0..<3 { total[t] += u[t] / 3 }
  }
  return total
}

// Adam on the aggregate, in place: theta, m, v [P]; grads [TASKS][P]
let adamSource = """
#include <metal_stdlib>
using namespace metal;
kernel void adam(device float *theta [[buffer(0)]], device float *m [[buffer(1)]], device float *v [[buffer(2)]], device const float *g [[buffer(3)]],
                 constant float4 &a [[buffer(4)]], constant float4 &c [[buffer(5)]], constant uint &P [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= P) return;
  const float d = a.x * g[i] + a.y * g[P + i] + a.z * g[2 * P + i];   // a: the three weights, the step size
  m[i] = 0.9f * m[i] + 0.1f * d; v[i] = 0.999f * v[i] + 0.001f * d * d;
  theta[i] -= a.w * (m[i] / c.x) / (sqrt(v[i] / c.y) + 1e-8f);   // c: the bias corrections
}
"""

final class Optimizer {
  let device: MTLDevice, queue: MTLCommandQueue, pipe: MTLComputePipelineState, P: Int
  let theta: MTLBuffer, m: MTLBuffer, v: MTLBuffer, grads: MTLBuffer, gram: MTLBuffer
  var step = 0
  init(_ device: MTLDevice, _ queue: MTLCommandQueue, initial: [Float]) {
    self.device = device; self.queue = queue; P = initial.count
    let lib = try! device.makeLibrary(source: adamSource, options: nil)
    pipe = try! device.makeComputePipelineState(function: lib.makeFunction(name: "adam")!)
    theta = device.makeBuffer(bytes: initial, length: P * 4, options: .storageModeShared)!
    m = device.makeBuffer(length: P * 4, options: .storageModePrivate)!; v = device.makeBuffer(length: P * 4, options: .storageModePrivate)!
    grads = device.makeBuffer(length: TASKS * P * 4, options: .storageModeShared)!; gram = device.makeBuffer(length: TASKS * TASKS * 4, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!, blit = cb.makeBlitCommandEncoder()!
    blit.fill(buffer: m, range: 0..<(P * 4), value: 0); blit.fill(buffer: v, range: 0..<(P * 4), value: 0); blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  }
  // the gradients in `grads` and their Gram in `gram` (from Learn): the step
  func apply(rate: Double) -> [Double] {
    let gp = gram.contents().bindMemory(to: Float.self, capacity: 9)
    let G = (0..<3).map { a in (0..<3).map { b in Double(gp[3 * a + b]) } }
    let alpha = upgrad(G)
    step += 1
    var a = SIMD4<Float>(Float(alpha[0]), Float(alpha[1]), Float(alpha[2]), Float(rate))
    var c = SIMD4<Float>(Float(1 - pow(0.9, Double(step))), Float(1 - pow(0.999, Double(step))), 0, 0), n = UInt32(P)
    let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    for (i, b) in [theta, m, v, grads].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    enc.setBytes(&a, length: 16, index: 4); enc.setBytes(&c, length: 16, index: 5); enc.setBytes(&n, length: 4, index: 6)
    enc.dispatchThreads(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return alpha
  }
}
