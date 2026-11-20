// int8.metal's check: each kernel against a CPU mirror of the same arithmetic (the same Walsh-Hadamard butterfly order,
// the same single-rounding and fma steps), bit for bit: the quantizers' codes and scales (to nearest and stochastic),
// the forward and dgrad products with each epilogue (bias, relu, mask, residual, row maps; the gates' sigmoid and tanh
// to 1e-6, the GPU's transcendentals not the CPU's), wgrad's block-split product; then each product against the fp32
// product it stands for, on Gaussian and on outlier-heavy data. Its rates at the network's shapes last.
// swiftc -O -import-objc-header int8mm.h int8check.swift -o int8check && ./int8check route.metallib
import Metal
import Foundation
let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
func pipe(_ n: String) -> MTLComputePipelineState { try! device.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let P0 = ["quant_rows", "quant_cols", "gemm_tn", "reduce_blocks"].reduce(into: [String: MTLComputePipelineState]()) { $0[$1] = pipe($1) }
var cache: [String: MTLComputePipelineState] = [:]
func pipe(_ n: String, flags: UInt32) -> MTLComputePipelineState {   // a product's pipeline for its epilogue flags (FLAGS)
  if let p = cache["\(n) \(flags)"] { return p }
  let c = MTLFunctionConstantValues(); var f = flags; c.setConstantValue(&f, type: .uint, index: 0)
  let p = try! device.makeComputePipelineState(function: try! lib.makeFunction(name: n, constantValues: c)); cache["\(n) \(flags)"] = p; return p
}
struct Pipes { subscript(_ n: String) -> MTLComputePipelineState? { P0[n] } }
let P = Pipes()
let NONE = UInt32.max
var rng = SystemRandomNumberGenerator()
func buf<T>(_ a: [T]) -> MTLBuffer { device.makeBuffer(bytes: a, length: max(16, a.count * MemoryLayout<T>.stride), options: .storageModeShared)! }
func zeros(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: max(16, bytes), options: .storageModeShared)! }
func read<T>(_ b: MTLBuffer, _ n: Int, _: T.Type) -> [T] { Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: T.self, capacity: n), count: n)) }
enum Grid { case threads(Int, Int), groups(Int, Int, Int, Int) }
func run(_ name: String, _ bufs: [MTLBuffer?], _ params: [[UInt32]], _ g: Grid) -> Double {
  let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(name.hasPrefix("gemm_n") || name.hasPrefix("gemmq") ? pipe(name, flags: params[0][3]) : P[name]!)
  for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
  for (k, a) in params.enumerated() { a.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: bufs.count + k) } }
  switch g {
  case let .threads(n, w): e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
  case let .groups(x, y, z, w): e.dispatchThreadgroups(MTLSize(width: x, height: y, depth: z), threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
  }
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  if let err = cb.error { print(err); exit(1) }
  return cb.gpuEndTime - cb.gpuStartTime
}
// heavy-tailed test data: Gaussian, optionally 0.5% of entries 30x
func data(_ n: Int, outliers: Bool) -> [Float] {
  (0..<n).map { _ in
    let u = Double.random(in: 1e-12..<1, using: &rng), v = Double.random(in: 0..<1, using: &rng)
    let g = Float(sqrt(-2 * log(u)) * cos(2 * .pi * v))
    return outliers && Int.random(in: 0..<200, using: &rng) == 0 ? 30 * g : g
  }
}
var failures = 0
func report(_ name: String, _ ok: Bool, _ detail: String) { print(String(format: "%-62@ %@  %@", name, ok ? "ok  " : "FAIL", detail)); if !ok { failures += 1 } }

// ---- the CPU mirror: int8mm.h's reference functions (the same header the kernels include) ----
func quant(_ v: Float, _ inv: Float, _ seed: UInt32, _ at: UInt32) -> Int8 { i8_code(v, inv, seed, at) }
func fwht64(_ v: inout [Float]) { v.withUnsafeMutableBufferPointer { i8_fwht64($0.baseAddress!) } }
func inverse(_ m: Float) -> Float { i8_inverse(m) }
func rmap(_ m: [UInt32], _ r: Int) -> Int { Int(UInt32(r) / m[0]) * Int(m[1]) + Int(m[2]) + Int(UInt32(r) % m[0]) }
func qrowsRef(_ x: [Float], rows: Int, cols: Int, ld: Int, valid: Int, map: [UInt32], seed: UInt32) -> ([Int8], [Float]) {
  var q = [Int8](repeating: 0, count: rows * cols), s = [Float](repeating: 0, count: rows)
  for r in 0..<rows {
    let src = rmap(map, r) * ld
    var v = (0..<cols).map { $0 < valid ? x[src + $0] : 0 }
    for b in 0..<(cols / 64) { var blk = Array(v[(b * 64)..<(b * 64 + 64)]); fwht64(&blk); for j in 0..<64 { v[b * 64 + j] = blk[j] } }
    let m = v.map(abs).max()!, inv = inverse(m)
    s[r] = i8_scale(m)
    for j in 0..<cols { q[r * cols + j] = quant(v[j], inv, seed, UInt32(r * cols + j)) }
  }
  return (q, s)
}
func qcolsRef(_ x: [Float], rows: Int, cols: Int, ld: Int, valid: Int, block: Int, seed: UInt32, ones: Int, off: Int, xcols: Int) -> ([Int8], [Float]) {
  let nb = rows / block
  var q = [Int8](repeating: 0, count: rows * cols), s = [Float](repeating: 0, count: cols * nb)
  for c in 0..<cols { for b in 0..<nb {
    var v = [Float](repeating: 0, count: block)
    for r in 0..<block { let row = b * block + r; v[r] = row >= valid ? 0 : c < xcols ? x[(off + row) * ld + c] : c == ones ? 1 : 0 }
    for sb in 0..<(block / 64) { var blk = Array(v[(sb * 64)..<(sb * 64 + 64)]); fwht64(&blk); for j in 0..<64 { v[sb * 64 + j] = blk[j] } }
    let m = v.map(abs).max()!, inv = inverse(m)
    s[c * nb + b] = i8_scale(m)
    for r in 0..<block { let row = b * block + r; q[row * cols + c] = quant(v[r], inv, seed, UInt32(row * cols + c)) }
  } }
  return (q, s)
}
// GPU calls
func qrows(_ x: [Float], rows: Int, cols: Int, ld: Int, valid: Int, map: [UInt32], seed: UInt32) -> ([Int8], [Float], Double) {
  let q = zeros(rows * cols), s = zeros(rows * 4)
  let perSimd = cols <= 64 ? 4 : cols <= 128 ? 2 : 1
  let t = run("quant_rows", [buf(x), q, s], [[UInt32(rows), UInt32(cols), UInt32(ld), UInt32(valid), 1, seed, 0, 0] + map], .threads((rows + perSimd - 1) / perSimd * 32, 256))
  return (read(q, rows * cols, Int8.self), read(s, rows, Float.self), t)
}
func qcols(_ x: [Float], rows: Int, cols: Int, ld: Int, valid: Int, block: Int, seed: UInt32, ones: Int, off: Int, xcols: Int) -> ([Int8], [Float], Double) {
  let nb = rows / block, q = zeros(rows * cols), s = zeros(cols * nb * 4)
  let t = run("quant_cols", [buf(x), q, s], [[UInt32(rows), UInt32(cols), UInt32(ld), UInt32(valid), UInt32(block), 1, seed, ones < 0 ? NONE : UInt32(ones), UInt32(off), UInt32(xcols), 0, 0]],
              .groups((cols + 31) / 32, nb, 1, 256))
  return (read(q, rows * cols, Int8.self), read(s, cols * nb, Float.self), t)
}
func same<T: Equatable>(_ a: [T], _ b: [T]) -> Int { zip(a, b).filter { $0 != $1 }.count }

// ---- quantizers ----
for outliers in [false, true] {
  let tag = outliers ? "outliers" : "Gaussian"
  for (rows, cols, ld, valid) in [(300, 64, 48, 48), (300, 64, 64, 64), (257, 128, 128, 128), (200, 192, 192, 192), (100, 64, 40, 40)] {
    for seed in [UInt32(0), 987654321] {
      let x = data(rows * 2 * ld + 64, outliers: outliers), map: [UInt32] = [UInt32(rows / 2 + 1), UInt32(rows), 7, 0]   // two "tasks", a stride, an offset
      let (q, s, _) = qrows(x, rows: rows, cols: cols, ld: ld, valid: valid, map: map, seed: seed)
      let (rq, rs) = qrowsRef(x, rows: rows, cols: cols, ld: ld, valid: valid, map: map, seed: seed)
      report("quant_rows \(rows)x\(cols) valid \(valid)\(seed != 0 ? " stochastic" : "") (\(tag))", same(q, rq) == 0 && same(s, rs) == 0, "codes differing \(same(q, rq)), scales \(same(s, rs))")
    }
  }
  for (rows, valid, cols, xcols, ones, block, off) in [(1024, 1000, 65, 64, 64, 256, 3), (1024, 1024, 193, 192, 192, 256, 0), (768, 700, 49, 48, 48, 256, 11), (192, 192, 64, 64, -1, 192, 0), (64, 40, 64, 64, -1, 64, 0), (128, 128, 64, 64, -1, 128, 0)] {
    for seed in [UInt32(0), 123456789] {
      let ld = xcols, x = data((rows + off) * ld, outliers: outliers)
      let (q, s, _) = qcols(x, rows: rows, cols: cols, ld: ld, valid: valid, block: block, seed: seed, ones: ones, off: off, xcols: xcols)
      let (rq, rs) = qcolsRef(x, rows: rows, cols: cols, ld: ld, valid: valid, block: block, seed: seed, ones: ones, off: off, xcols: xcols)
      report("quant_cols \(rows) (valid \(valid)) x \(cols), block \(block)\(ones >= 0 ? ", ones" : "")\(seed != 0 ? " stochastic" : "") (\(tag))", same(q, rq) == 0 && same(s, rs) == 0, "codes differing \(same(q, rq)), scales \(same(s, rs))")
    }
  }
}

// ---- forward and dgrad products ----
func sigmoid(_ y: Float) -> Float { 1 / (1 + exp(-y)) }
func fwdRef(nt: Bool, qa: [Int8], qb: [Int8], sa: [Float], sb: [Float], M: Int, N: Int, K: Int, flags: UInt32, bias: [Float], res: [Float], resMap: [UInt32], ldr: Int,
            aux: [Float], auxOff: Int, auxSeg: Int, lda: Int, outMap: [UInt32], ldo: Int, out0: [Float]) -> [Float] {
  var out = out0
  for m in 0..<M { for n in 0..<N {
    var s: Int32 = 0
    for k in 0..<K { s += Int32(qa[m * K + k]) * Int32(nt ? qb[n * K + k] : qb[k * N + n]) }
    var y = i8_out(s, sa[m], sb[n], flags & 1 != 0 ? bias[n] : 0)
    if flags & 2 != 0 { y = max(y, 0) }
    if flags & 4 != 0 { y = n < N / 2 ? sigmoid(y) : tanh(max(-20, min(20, y))) }
    if flags & 8 != 0 && !(aux[(auxOff + m % auxSeg) * lda + n] > 0) { y = 0 }
    if flags & 16 != 0 { y += res[rmap(resMap, m) * ldr + n] }
    out[rmap(outMap, m) * ldo + n] = y
  } }
  return out
}
func exact(_ a: [Float], _ b: [Float]) -> (Int, Float) {
  var d = 0, w: Float = 0, sc: Float = 0
  for i in 0..<a.count { if a[i] != b[i] { d += 1 }; w = max(w, abs(a[i] - b[i])); sc = max(sc, abs(b[i])) }
  return (d, w / max(sc, 1e-30))
}
func relErr(_ got: [Float], _ want: [Float]) -> Float {
  var e = 0.0, n = 0.0
  for i in 0..<got.count { e += Double((got[i] - want[i]) * (got[i] - want[i])); n += Double(want[i] * want[i]) }
  return Float(sqrt(e / max(n, 1e-300)))
}
for outliers in [false, true] {
  let tag = outliers ? "outliers" : "Gaussian"
  // forward NT: x [M][Kx] (valid Kv) against w [N][Kv], Kx padded to K
  for (M, N, Kv, flags) in [(300, 80, 48, UInt32(1)), (300, 128, 64, 1 | 4), (333, 192, 64, 1 | 2), (300, 64, 192, 1 | 16), (100, 40, 64, 0)] {
    let K = (Kv + 63) / 64 * 64
    let x = data(M * Kv, outliers: outliers), w = data(N * Kv, outliers: outliers).map { $0 / Float(Kv).squareRoot() }
    let (qa, sa, _) = qrows(x, rows: M, cols: K, ld: Kv, valid: Kv, map: [NONE, 0, 0, 0], seed: 0)
    let (qb, sb, _) = qrows(w, rows: N, cols: K, ld: Kv, valid: Kv, map: [NONE, 0, 0, 0], seed: 0)
    let bias = data(N, outliers: false), res = data(M * N, outliers: false), out = zeros(M * N * 4)
    let p: [UInt32] = [UInt32(M), UInt32(N), UInt32(K), flags, UInt32(N), UInt32(N), 0, 1] + [NONE, 0, 0, 0] + [NONE, 0, 0, 0] + [0, 0, 0, 0]
    _ = run("gemm_nt", [buf(qa), buf(qb), buf(sa), buf(sb), out, buf(bias), buf(res), nil], [p], .groups((N + 31) / 32, (M + 63) / 64, 1, 128))
    let got = read(out, M * N, Float.self)
    let want = fwdRef(nt: true, qa: qa, qb: qb, sa: sa, sb: sb, M: M, N: N, K: K, flags: flags, bias: bias, res: res, resMap: [NONE, 0, 0, 0], ldr: N, aux: [], auxOff: 0, auxSeg: 1, lda: 0, outMap: [NONE, 0, 0, 0], ldo: N, out0: [Float](repeating: 0, count: M * N))
    let (d, r) = exact(got, want), gates = flags & 4 != 0
    report("gemm_nt \(M)x\(N)x\(K) flags \(flags) (\(tag))", gates ? r < 1e-6 : d == 0, String(format: "differing %d, max diff / max %.1e%@", d, r, gates ? " (gates: transcendentals)" : ""))
    var plain = [Float](repeating: 0, count: M * N)
    for m in 0..<M { for n in 0..<N { var s = 0.0; for k in 0..<Kv { s += Double(x[m * Kv + k]) * Double(w[n * Kv + k]) }; var y = Float(s) + (flags & 1 != 0 ? bias[n] : 0)
      if flags & 2 != 0 { y = max(y, 0) }; if flags & 4 != 0 { y = n < N / 2 ? sigmoid(y) : tanh(y) }; if flags & 16 != 0 { y += res[m * N + n] }; plain[m * N + n] = y } }
    print(String(format: "    against the fp32 product: relative error %.4f", relErr(got, plain)))
  }
  // dgrad NN: dy [3 tasks x R][Kv] (stochastic) against w [Kv][N] (column codes); mask by aux, residual, out mapped into a
  // [3][Nall] buffer at an offset (as layer 1's feed-forward writes its context rows)
  for (R, Kv, N, flags) in [(150, 40, 64, UInt32(0)), (150, 64, 192, 8), (150, 192, 64, 16), (150, 128, 64, 16)] {
    let K = (Kv + 63) / 64 * 64, M = 3 * R, Nall = R + 37, off = 37
    let dy = data(M * Kv, outliers: outliers), w = data(Kv * N, outliers: outliers).map { $0 / 8 }
    let (qa, sa, _) = qrows(dy, rows: M, cols: K, ld: Kv, valid: Kv, map: [NONE, 0, 0, 0], seed: 55555)
    let (qb, sb, _) = qcols(w, rows: K, cols: N, ld: N, valid: Kv, block: K, seed: 0, ones: -1, off: 0, xcols: N)
    let aux = data((off + R) * N, outliers: false), res = data(M * N, outliers: false), out0 = data(3 * Nall * N, outliers: false), out = buf(out0)
    let outMap: [UInt32] = [UInt32(R), UInt32(Nall), UInt32(off), 0]
    let p: [UInt32] = [UInt32(M), UInt32(N), UInt32(K), flags, UInt32(N), UInt32(N), UInt32(N), UInt32(R)] + outMap + [NONE, 0, 0, 0] + [UInt32(off), 0, 0, 0]
    _ = run("gemm_nn", [buf(qa), buf(qb), buf(sa), buf(sb), out, nil, buf(res), buf(aux)], [p], .groups((N + 31) / 32, (M + 63) / 64, 1, 128))
    let got = read(out, 3 * Nall * N, Float.self)
    let want = fwdRef(nt: false, qa: qa, qb: qb, sa: sa, sb: sb, M: M, N: N, K: K, flags: flags, bias: [], res: res, resMap: [NONE, 0, 0, 0], ldr: N, aux: aux, auxOff: off, auxSeg: R, lda: N, outMap: outMap, ldo: N, out0: out0)
    let (d, r) = exact(got, want)
    report("gemm_nn \(M)x\(N)x\(K) flags \(flags), mapped (\(tag))", d == 0, String(format: "differing %d, max diff / max %.1e", d, r))
  }
  // wgrad TN: dY [rows][Mw] against [X, 1] [rows][Nw], 256-row blocks, split
  for (rows, Mw, Xc, ones) in [(1000, 40, 64, false), (5000, 128, 64, true), (5000, 192, 64, true), (3000, 64, 192, true), (4100, 64, 48, true)] {
    let Kp = (rows + 255) / 256 * 256, nb = Kp / 256, Nw = Xc + (ones ? 1 : 0)
    let dy = data(rows * Mw, outliers: outliers), x = data(rows * Xc, outliers: outliers)
    let (qa, sa, _) = qcols(dy, rows: Kp, cols: Mw, ld: Mw, valid: rows, block: 256, seed: 24680, ones: -1, off: 0, xcols: Mw)
    let (qb, sb, _) = qcols(x, rows: Kp, cols: Nw, ld: Xc, valid: rows, block: 256, seed: 0, ones: ones ? Xc : -1, off: 0, xcols: Xc)
    // a threadgroup a 256-row block, then the pairwise tree (net.swift's)
    let partial = zeros(nb * Mw * Nw * 4), out = zeros(Mw * Nw * 4)
    _ = run("gemm_tn", [buf(qa), buf(qb), buf(sa), buf(sb), partial], [[UInt32(Mw), UInt32(Nw), UInt32(Kp), 1, UInt32(nb)]], .groups((Nw + 31) / 32, (Mw + 63) / 64, nb, 128))
    _ = run("reduce_blocks", [partial, out], [[UInt32(Mw * Nw), UInt32(nb), 1]], .threads(Mw * Nw, 256))
    let got = read(out, Mw * Nw, Float.self)
    var ref = [Float](repeating: 0, count: Mw * Nw), plain = [Float](repeating: 0, count: Mw * Nw)
    for m in 0..<Mw { for n in 0..<Nw {
      var part = [Float](repeating: 0, count: nb)
      for kb in 0..<nb {
        var s: Int32 = 0
        for k in (kb * 256)..<(kb * 256 + 256) { s += Int32(qa[k * Mw + m]) * Int32(qb[k * Nw + n]) }
        part[kb] = i8_acc(s, sa[m * nb + kb], sb[n * nb + kb], 0)
      }
      var st = 1
      while st < nb { var z = 0; while z + st < nb { part[z] += part[z + st]; z += 2 * st }; st *= 2 }
      let tot = part[0]
      ref[m * Nw + n] = tot
      var e = 0.0; for k in 0..<rows { e += Double(dy[k * Mw + m]) * (n < Xc ? Double(x[k * Xc + n]) : 1) }; plain[m * Nw + n] = Float(e)
    } }
    let (d, r) = exact(got, ref)
    report("gemm_tn \(Mw)x\(Nw) over \(rows) rows, \(nb) blocks, tree (\(tag))", d == 0, String(format: "differing %d, max diff / max %.1e", d, r))
    print(String(format: "    against the fp32 product (stochastic dY): relative error %.4f", relErr(got, plain)))
  }
}

// ---- the fused product: its stored values and codes against the plain product then quant_rows (bit for bit) ----
for (nt, M, N, Kv, flags, seed) in [(true, 300, 64, 48, UInt32(0), UInt32(0)), (true, 333, 192, 64, 2, 0), (true, 300, 64, 192, 16, 0), (false, 450, 192, 64, 8, 777), (false, 450, 64, 128, 16, 999), (false, 450, 64, 192, 16, 31)] {
  let K = (Kv + 63) / 64 * 64, cols = (N + 63) / 64 * 64
  let x = data(M * Kv, outliers: true), w = data(N * Kv, outliers: true).map { $0 / 8 }
  let (qa, sa, _) = qrows(x, rows: M, cols: K, ld: Kv, valid: Kv, map: [NONE, 0, 0, 0], seed: 0)
  let (qb, sb, _) = nt ? qrows(w, rows: N, cols: K, ld: Kv, valid: Kv, map: [NONE, 0, 0, 0], seed: 0)
                       : qcols(w.map { $0 }, rows: K, cols: N, ld: N, valid: Kv, block: K, seed: 0, ones: -1, off: 0, xcols: N)
  let bias = data(N, outliers: false), res = data(M * N, outliers: false), aux = data(M * N, outliers: false)
  let plainOut = zeros(M * N * 4), fusedOut = zeros(M * N * 4), q = zeros(M * cols), qs = zeros(M * 4)
  let p: [UInt32] = [UInt32(M), UInt32(N), UInt32(K), flags | (nt ? 1 : 0), UInt32(N), UInt32(N), UInt32(N), UInt32(M)] + [NONE, 0, 0, 0] + [NONE, 0, 0, 0] + [0, seed, 0, 0]
  _ = run(nt ? "gemm_nt" : "gemm_nn", [buf(qa), buf(qb), buf(sa), buf(sb), plainOut, buf(bias), buf(res), buf(aux)], [p], .groups((N + 31) / 32, (M + 63) / 64, 1, 128))
  var pf = p; pf[3] |= 32 | 64
  let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(pipe(nt ? "gemmq_nt" : "gemmq_nn", flags: pf[3]))
  for (i, b) in [buf(qa), buf(qb), buf(sa), buf(sb), fusedOut, buf(bias), buf(res), buf(aux)].enumerated() { e.setBuffer(b, offset: 0, index: i) }
  pf.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 8) }
  e.setBuffer(q, offset: 0, index: 9); e.setBuffer(qs, offset: 0, index: 10); e.setBuffer(nil, offset: 0, index: 11); e.setThreadgroupMemoryLength(32 * (cols + 4) * 4, index: 0)
  e.dispatchThreadgroups(MTLSize(width: (M + 31) / 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  let a1 = read(plainOut, M * N, Float.self), a2 = read(fusedOut, M * N, Float.self)
  let (rq, rs, _) = qrows(a1, rows: M, cols: cols, ld: N, valid: N, map: [NONE, 0, 0, 0], seed: seed)
  let dq = same(read(q, M * cols, Int8.self), rq), ds = same(read(qs, M, Float.self), rs), dv = same(a1, a2)
  report("gemmq_\(nt ? "nt" : "nn") \(M)x\(N)x\(K) flags \(flags)\(seed != 0 ? " stochastic" : "") vs product + quant_rows", dq == 0 && ds == 0 && dv == 0, "stored differing \(dv), codes \(dq), scales \(ds)")
  let t1 = median { run(nt ? "gemm_nt" : "gemm_nn", [buf(qa), buf(qb), buf(sa), buf(sb), plainOut, buf(bias), buf(res), buf(aux)], [p], .groups((N + 31) / 32, (M + 63) / 64, 1, 128)) }
  _ = t1
}

// ---- rates at the network's shapes (GPU clock, median of 9) ----
func median(_ f: () -> Double) -> Double { (0..<9).map { _ in f() }.sorted()[4] }
for (M, N, K) in [(147456, 64, 64), (147456, 128, 64), (147456, 192, 64), (147456, 64, 192), (442368, 64, 128), (442368, 192, 64), (442368, 64, 192)] {
  let a = zeros(M * K), b = zeros(N * K), sa = zeros(M * 4), sb = zeros(N * 4), out = zeros(M * N * 4), x = zeros(M * K * 4)
  let p: [UInt32] = [UInt32(M), UInt32(N), UInt32(K), 1, UInt32(N), UInt32(N), 0, 1] + [NONE, 0, 0, 0] + [NONE, 0, 0, 0] + [0, 0, 0, 0]
  let t = median { run("gemm_nt", [a, b, sa, sb, out, sb, nil, nil], [p], .groups((N + 31) / 32, (M + 63) / 64, 1, 128)) }
  var pq = p; pq[3] |= 32 | 64
  let cq = zeros(M * ((N + 63) / 64 * 64)), cs = zeros(M * 4)
  let tf = median {
    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pipe("gemmq_nt", flags: pq[3]))
    for (i, bb) in [a, b, sa, sb, out, sb, nil, nil].enumerated() { e.setBuffer(bb, offset: 0, index: i) }
    pq.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 8) }
    e.setBuffer(cq, offset: 0, index: 9); e.setBuffer(cs, offset: 0, index: 10); e.setThreadgroupMemoryLength(32 * ((N + 63) / 64 * 64 + 4) * 4, index: 0)
    e.dispatchThreadgroups(MTLSize(width: (M + 31) / 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted(); return cb.gpuEndTime - cb.gpuStartTime
  }
  let perSimd = K <= 64 ? 4 : K <= 128 ? 2 : 1
  let pso = N <= 64 ? 4 : N <= 128 ? 2 : 1, oc = (N + 63) / 64 * 64, qo = zeros(M * oc)
  let tq = median { run("quant_rows", [out, qo, sa], [[UInt32(M), UInt32(oc), UInt32(N), UInt32(N), 1, 0, 0, 0, NONE, 0, 0, 0]], .threads((M + pso - 1) / pso * 32, 256)) }
  _ = perSimd
  let tc = median { run("quant_cols", [x, a, sa], [[UInt32(M / 256 * 256), UInt32(K), UInt32(K), UInt32(M), 256, 1, 7, NONE, 0, UInt32(K), 0, 0]], .groups((K + 31) / 32, M / 256, 1, 256)) }
  print(String(format: "M %6d N %3d K %3d: fused gemmq_nt (stores and codes) %7.1f us %5.1f TOP/s; apart:", M, N, K, tf * 1e6, 2 * Double(M * N * K) / tf / 1e12))
  print(String(format: "M %6d N %3d K %3d: gemm_nt %7.1f us %5.1f TOP/s | quant_rows of its output %6.1f us %4.0f GB/s | quant_cols %6.1f us %4.0f GB/s", M, N, K, t * 1e6, 2 * Double(M * N * K) / t / 1e12,
               tq * 1e6, Double(M * N * 5) / tq / 1e9, tc * 1e6, Double(M * K * 5) / tc / 1e9))
}
for (rows, Mw, Nw) in [(147456, 128, 65), (147456, 192, 65), (147456, 64, 193)] {
  let Kp = rows, nb = Kp / 256, a = zeros(Kp * Mw), b = zeros(Kp * Nw), sa = zeros(Mw * nb * 4), sb = zeros(Nw * nb * 4), partial = zeros(nb * Mw * Nw * 4)
  let t = median {
    var total = run("gemm_tn", [a, b, sa, sb, partial], [[UInt32(Mw), UInt32(Nw), UInt32(Kp), 1, UInt32(nb)]], .groups((Nw + 31) / 32, (Mw + 63) / 64, nb, 128))
    total += run("reduce_blocks", [partial, zeros(Mw * Nw * 4)], [[UInt32(Mw * Nw), UInt32(nb), 1]], .threads(Mw * Nw, 256))
    return total
  }
  print(String(format: "wgrad %3dx%3d over %6d rows (%d blocks, tree): %7.1f us %5.1f TOP/s", Mw, Nw, rows, nb, t * 1e6, 2 * Double(Mw * Nw * rows) / t / 1e12))
}
print(failures == 0 ? "all checks pass" : "\(failures) checks FAILED")
exit(failures == 0 ? 0 : 1)
