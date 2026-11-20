// ./stages route.metallib: the GPU backend measured and checked, from the bare product up to the network.
//  1. mm.metal's kept products, each tile instance, bit for bit against a CPU mirror of the same arithmetic (int8mm.h's
//     i8_out and i8_acc in the same order): forward (gemm_nt), dgrad (gemm_nn), wgrad (gemm_tn, gemm_wg);
//  2. their rates at the network's shapes against the roofline (roofline.swift's kernel, `roof`), the best tile each;
//  3. the network's build-up through libroute's C API (kernels.h), random inputs at capacity, the forward's stages
//     (fwd.metal's STAGE: the input product with its fused codes, scales and bias; + the gates and a context's step
//     (the next product's codes fused in, two products chained on chip); + the feed-forward (a whole layer, four
//     products); + the second layer and the readout; the lanes' recurrence), then the learning step kernel by kernel.
// swiftc -O -import-objc-header stages.h stages.swift -L . -lroute -o stages && ./stages route.metallib
import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
let libPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "route.metallib"
let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: libPath))
func pipe(_ n: String) -> MTLComputePipelineState { try! device.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
var failures: [String] = []
func check(_ ok: Bool, _ what: String) { print((ok ? "  ok   " : "  FAIL ") + what); if !ok { failures.append(what) } }
var rng = SystemRandomNumberGenerator()
func codes(_ n: Int) -> [Int8] { (0..<n).map { _ in Int8.random(in: -127...127, using: &rng) } }
func floats(_ n: Int, _ lo: Float = 0.001, _ hi: Float = 0.02) -> [Float] { (0..<n).map { _ in Float.random(in: lo...hi, using: &rng) } }
func buf<T>(_ a: [T]) -> MTLBuffer { device.makeBuffer(bytes: a, length: max(16, a.count * MemoryLayout<T>.stride), options: .storageModeShared)! }
func run(_ p: MTLComputePipelineState, _ bufs: [Int: MTLBuffer], _ bytes: [Int: [UInt32]], _ grid: MTLSize, _ threads: Int, times: Int = 1) -> Double {
  var best = 1e9
  for _ in 0..<times {
    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    for (i, b) in bufs { e.setBuffer(b, offset: 0, index: i) }
    for (i, a) in bytes { a.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: i) } }
    e.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  return best
}
let tiles: [(Int, Int, Int)] = [(64, 32, 4), (64, 64, 4), (64, 128, 4), (128, 64, 8)]
let wtiles: [(Int, Int, Int)] = [(64, 32, 4), (64, 64, 4), (128, 64, 8)]
func name(_ k: String, _ t: (Int, Int, Int)) -> String { "\(k)_\(t.0)x\(t.1)_\(t.2)" }

// ---- 1. the kept products, bit for bit ----
print("1. the kept products against their CPU mirror (int8mm.h), bit for bit")
do {
  let M = 256, N = 192, K = 192
  let a = codes(M * K), bt = codes(N * K), bn = codes(K * N), sa = floats(M), sb = floats(N), bias = floats(N, -1, 1)
  var want = [Float](repeating: 0, count: M * N), wantNN = want
  for m in 0..<M { for n in 0..<N {
    var s: Int32 = 0, s2: Int32 = 0
    for k in 0..<K { s += Int32(a[m * K + k]) * Int32(bt[n * K + k]); s2 += Int32(a[m * K + k]) * Int32(bn[k * N + n]) }
    want[m * N + n] = i8_out(s, sa[m], sb[n], bias[n]); wantNN[m * N + n] = i8_out(s2, sa[m], sb[n], bias[n])
  } }
  let A = buf(a), BT = buf(bt), BN = buf(bn), SA = buf(sa), SB = buf(sb), BI = buf(bias), out = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  for t in tiles {
    for (k, B, w) in [("gemm_nt", BT, want), ("gemm_nn", BN, wantNN)] {
      _ = run(pipe(name(k, t)), [0: A, 1: B, 2: SA, 3: SB, 4: out, 5: BI], [6: [UInt32(M), UInt32(N), UInt32(K), 1]], MTLSize(width: (N + t.1 - 1) / t.1, height: (M + t.0 - 1) / t.0, depth: 1), 32 * t.2)
      let o = out.contents().bindMemory(to: Float.self, capacity: M * N)
      var bad = 0; for i in 0..<(M * N) where o[i].bitPattern != w[i].bitPattern { bad += 1 }
      check(bad == 0, "\(name(k, t)): \(bad) of \(M * N) outputs differ")
    }
  }
  // wgrad: K rows in blocks of I8_WBLOCK, M x N outputs, a block's scales a column
  let Kw = 512, nb = Kw / Int(I8_WBLOCK), Mw = 128, Nw = 64
  let x = codes(Kw * Mw), y = codes(Kw * Nw), swa = floats(nb * Mw), swb = floats(nb * Nw)
  var xt = [Int8](repeating: 0, count: Kw * Mw), yt = [Int8](repeating: 0, count: Kw * Nw)   // the transposed operands
  for k in 0..<Kw { for m in 0..<Mw { xt[m * Kw + k] = x[k * Mw + m] }; for n in 0..<Nw { yt[n * Kw + k] = y[k * Nw + n] } }
  var wantW = [Float](repeating: 0, count: Mw * Nw)
  for m in 0..<Mw { for n in 0..<Nw {
    var acc: Float = 0
    for b in 0..<nb { var s: Int32 = 0; for k in (b * Int(I8_WBLOCK))..<((b + 1) * Int(I8_WBLOCK)) { s += Int32(x[k * Mw + m]) * Int32(y[k * Nw + n]) }; acc = i8_acc(s, swa[b * Mw + m], swb[b * Nw + n], acc) }
    wantW[m * Nw + n] = acc
  } }
  let part = device.makeBuffer(length: Mw * Nw * 4, options: .storageModeShared)!
  for t in wtiles {
    for (k, X, Y) in [("gemm_tn", buf(x), buf(y)), ("gemm_wg", buf(xt), buf(yt))] {
      _ = run(pipe(name(k, t)), [0: X, 1: Y, 2: buf(swa), 3: buf(swb), 4: part], [5: [UInt32(Mw), UInt32(Nw), UInt32(Kw), UInt32(nb), 0, UInt32(nb), 0, 0]],
              MTLSize(width: Nw / t.1, height: Mw / t.0, depth: 1), 32 * t.2)
      let o = part.contents().bindMemory(to: Float.self, capacity: Mw * Nw)
      var bad = 0; for i in 0..<(Mw * Nw) where o[i].bitPattern != wantW[i].bitPattern { bad += 1 }
      check(bad == 0, "\(name(k, t)): \(bad) of \(Mw * Nw) outputs differ")
    }
  }
}

// ---- 2. the kept products' rates at the network's shapes, against the roofline ----
print("\n2. the kept products at the network's shapes (TOP/s, best of 5): the roofline kernel's best tile, then each product's")
let big = 786432 * 192
let bigA = device.makeBuffer(length: big, options: .storageModePrivate)!, bigB = device.makeBuffer(length: 192 * 786432, options: .storageModePrivate)!
let bigS = device.makeBuffer(length: 786432 * 4 * 8, options: .storageModePrivate)!, bigO = device.makeBuffer(length: big * 4, options: .storageModePrivate)!
func best(_ kernel: String, _ ts: [(Int, Int, Int)], _ M: Int, _ N: Int, _ K: Int, wgrad: Bool = false) -> (Double, String) {
  var b = (0.0, "")
  for t in ts {
    let s: Double
    if wgrad {
      let nb = K / Int(I8_WBLOCK), per = 64, nz = (nb + per - 1) / per
      s = run(pipe(name(kernel, t)), [0: bigA, 1: bigB, 2: bigS, 3: bigS, 4: bigO], [5: [UInt32(M), UInt32(N), UInt32(K), UInt32(per), 0, UInt32(nb), 0, 0]],
              MTLSize(width: (N + t.1 - 1) / t.1, height: (M + t.0 - 1) / t.0, depth: nz), 32 * t.2, times: 5)
    } else {
      s = run(pipe(name(kernel, t)), [0: bigA, 1: bigB, 2: bigS, 3: bigS, 4: bigO, 5: bigS], [6: [UInt32(M), UInt32(N), UInt32(K), 1]],
              MTLSize(width: (N + t.1 - 1) / t.1, height: (M + t.0 - 1) / t.0, depth: 1), 32 * t.2, times: 5)
    }
    let tops = 2 * Double(M) * Double(N) * Double(K) / s / 1e12
    if tops > b.0 { b = (tops, "\(t.0)x\(t.1)/\(t.2)") }
  }
  return b
}
for (what, M, N, K) in [("inp, out (forward)", 262144, 64, 64), ("zh", 262144, 128, 64), ("up", 262144, 192, 64), ("down", 262144, 64, 192)] {
  let r = best("roof", tiles, M, N, K), g = best("gemm_nt", tiles, M, N, K)
  print(String(format: "  %-22@ M %7d N %3d K %3d   roofline %5.1f (%@)   gemm_nt %5.1f (%@)", what, M, N, K, r.0, r.1, g.0, g.1))
}
for (what, M, N, K) in [("zh (dgrad, 3 tasks)", 786432, 64, 128), ("up", 786432, 64, 192), ("down", 786432, 192, 64), ("out", 786432, 64, 64)] {
  let r = best("roof", tiles, M, N, K), g = best("gemm_nn", tiles, M, N, K)
  print(String(format: "  %-22@ M %7d N %3d K %3d   roofline %5.1f (%@)   gemm_nn %5.1f (%@)", what, M, N, K, r.0, r.1, g.0, g.1))
}
for (what, M, N) in [("inp (wgrad, 3 tasks)", 192, 64), ("zh", 384, 64), ("up", 576, 64), ("down", 192, 192), ("out", 120, 64)] {
  let K = 262144
  let r = best("roof", tiles, M, N, K), tn = best("gemm_tn", wtiles, M, N, K, wgrad: true), wg = best("gemm_wg", wtiles, M, N, K, wgrad: true)
  print(String(format: "  %-22@ M %7d N %3d K %6d roofline %5.1f (%@)   gemm_tn %5.1f (%@)   gemm_wg %5.1f (%@)", what, M, N, K, r.0, r.1, tn.0, tn.1, wg.0, wg.1))
}

// ---- 3. the network's build-up (libroute, random inputs at capacity) ----
print("\n3. the network's build-up: the forward stage by stage, contexts' tiles (262,144 rows) and the lanes' (1,024 runs x 32)")
let D = Int(ROUTE_D), E = Int(ROUTE_E), IN = Int(ROUTE_IN), FS = Int(ROUTE_FS), L = Int(ROUTE_L)
let runs = 1024, R = 262144
// a forward call's time (seconds a call over 20 back to back, committed without waits), with R contexts
func forwardTime(stage: Int, contexts: Int) -> Double {
  setenv("ROUTE_STAGE", "\(stage)", 1)
  guard let net = route_net_new(libPath, Int32(runs), 1, Int32(R), 20261119, nil) else { fatalError(String(cString: route_net_failure())) }
  let b = route_net_batch(net, Int32(runs), 1, Int32(R))!
  var g = SystemRandomNumberGenerator()
  for i in 0..<(runs * L * IN) { b.pointee.lanes[i] = Float.random(in: -1...1, using: &g) }
  for i in 0..<(runs * L) { b.pointee.mask[i] = 1 }
  for i in 0..<(runs * 2 * D) { b.pointee.h0[i] = Float.random(in: -0.1...0.1, using: &g) }
  for i in 0..<(R * IN) { b.pointee.tokens[i] = Float.random(in: -1...1, using: &g) }
  for r in 0..<R { b.pointee.pos[r] = Int32((r % runs) * L + L - 1) }
  route_net_forward(net, b, Int32(runs), Int32(contexts)); route_net_wait(net, b)
  var t = 1e9
  for _ in 0..<3 {
    let t0 = Date()
    for _ in 0..<20 { route_net_forward(net, b, Int32(runs), Int32(contexts)) }
    route_net_wait(net, b); t = min(t, Date().timeIntervalSince(t0) / 20)
  }
  route_net_free(net)
  return t
}
let ctxMacs = [1: 64 * D, 2: 64 * D + 2 * D * D, 3: 64 * D + 2 * D * D + 2 * D * E, 0: 64 * D + 2 * (2 * D * D + 2 * D * E) + FS * D]
let laneMacs = 64 * D + 2 * D * D + 2 * D * E + 2 * D * D
let tLanes = forwardTime(stage: 0, contexts: 0)
print(String(format: "  the lanes alone (lanes_fwd: layer 0 whole, layer 1's gates, the recurrence)   %7.1f us %5.1f TOP/s", tLanes * 1e6, 2 * Double(runs * L) * Double(laneMacs) / tLanes / 1e12))
for (stage, what) in [(1, "the input product: its codes, scales and bias fused"), (2, "+ the gates, the context's step (2 products)"), (3, "+ the feed-forward: a whole layer (4 products)"), (0, "+ the second layer and the readout (8 products)")] {
  let t = forwardTime(stage: stage, contexts: R), ctx = t - tLanes
  print(String(format: "  stage %d %-52@ ctx_fwd %7.1f us %5.1f TOP/s", stage, what, ctx * 1e6, 2 * Double(R) * Double(ctxMacs[stage]!) / max(ctx, 1e-12) / 1e12))
  if stage == 0 {
    let ops = 2 * (Double(runs * L) * Double(laneMacs) + Double(R) * Double(ctxMacs[0]!))
    print(String(format: "          the forward call: %.1f us, %.1f TOP/s effective", t * 1e6, ops / t / 1e12))
  }
}
setenv("ROUTE_STAGE", "0", 1)
print(failures.isEmpty ? "\nall checks pass" : "\n\(failures.count) checks FAILED")
exit(failures.isEmpty ? 0 : 1)
