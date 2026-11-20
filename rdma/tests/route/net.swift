// The policy (route.h) on the GPU, every 2D product int8 x int8 -> int32 (int8.metal), the rest net.metal's; the
// optimizer's parameters (theta) and moments fp32. One command buffer a call: the forward of a refresh for the rollouts,
// or a learning batch's forward again (its windows' K refreshes, the states carried) and its three tasks' backward.
//
// The network (minGRU): x = inp(token); two layers, each z, h~ = sigmoid, tanh of zh(x) (one product, 2D wide), the
// recurrence h = (1 - m z) h + m z h~ along a run's lanes, refresh after refresh (a context: one step from its refresh's
// state), xm = x + h, x' = xm + down(relu(up(xm))); then w = out(rms(x')) for each context. Layer 1's lanes need only
// their states, so their feed-forward is not run.
// A matrix's weights: W = diag(g) V / |V| (rows), V pinned to unit rows after each step (so W's scale is g's alone; its
// int8 scale is a row's absmax, so g is exactly absorbed into it); a bias b beside it.
// Rows: lane rows first ([run][refresh][lane]), then context rows; a context's state is the lane row pos[r] (its
// refresh's last); the backward's buffers [task][rows][...], the three tasks stacked along the rows.
import Foundation
import Metal

struct NetMat { let name: String, rows: Int, cols: Int, bias: Bool; var v = 0, g = 0 }   // b at g + rows
struct NetLayout {
  let mats: [NetMat], count: Int
  init() {
    var m = [NetMat(name: "inp", rows: D, cols: IN, bias: true)]
    for l in 0..<2 { m += [NetMat(name: "zh\(l)", rows: 2 * D, cols: D, bias: true), NetMat(name: "up\(l)", rows: E, cols: D, bias: true), NetMat(name: "down\(l)", rows: D, cols: E, bias: true)] }
    m.append(NetMat(name: "out", rows: FS, cols: D, bias: false))
    var n = 0
    for i in m.indices { m[i].v = n; n += m[i].rows * m[i].cols; m[i].g = n; n += m[i].rows * (m[i].bias ? 2 : 1) }
    mats = m; count = n
  }
  func mat(_ name: String) -> NetMat { mats.first { $0.name == name }! }
  // V's rows Gaussian, normalized; g 1 (the earlier layout's 1 / sqrt(fan-in) weights had rows of norm ~1); b 0
  func initial(_ r: inout route_rng) -> [Float] {
    var t = [Float](repeating: 0, count: count)
    for m in mats {
      for i in 0..<m.rows {
        var s = 0.0, row = [Double](repeating: 0, count: m.cols)
        for j in 0..<m.cols { let u = max(route_uniform(&r), 1e-300), v = route_uniform(&r); row[j] = sqrt(-2 * log(u)) * cos(2 * Double.pi * v); s += row[j] * row[j] }
        for j in 0..<m.cols { t[m.v + i * m.cols + j] = Float(row[j] / s.squareRoot()) }
        t[m.g + i] = 1
      }
    }
    return t
  }
  // policy.swift's fp32 layout (Layout(full: false): plain W and b) to this one, exactly (V = W / |W|, g = |W|), and back
  func from(old o: [Float], _ lay: Layout) -> [Float] {
    var t = [Float](repeating: 0, count: count)
    func block(_ n: String) -> (Int, [Int]) { let b = lay.blocks.first { $0.name == n }!; return (b.offset, b.shape) }
    func put(_ m: NetMat, _ rows: [(String, String?)]) {   // the old blocks stacked as this matrix's rows
      var r = 0
      for (w, b) in rows {
        let (wo, shape) = block(w)
        for i in 0..<shape[0] {
          var s = 0.0; for j in 0..<shape[1] { s += Double(o[wo + i * shape[1] + j]) * Double(o[wo + i * shape[1] + j]) }
          let n = max(s.squareRoot(), 1e-30)
          for j in 0..<shape[1] { t[m.v + (r + i) * m.cols + j] = Float(Double(o[wo + i * shape[1] + j]) / n) }
          t[m.g + r + i] = Float(n)
          if let b = b { t[m.g + m.rows + r + i] = o[block(b).0 + i] }
        }
        r += shape[0]
      }
    }
    put(mat("inp"), [("inp", "binp")])
    for l in 0..<2 { put(mat("zh\(l)"), [("z\(l)", "bz\(l)"), ("h\(l)", "bh\(l)")]); put(mat("up\(l)"), [("up\(l)", "bu\(l)")]); put(mat("down\(l)"), [("down\(l)", "bd\(l)")]) }
    put(mat("out"), [("out", nil)])
    return t
  }
  // a gradient in the fp32 layout's W, b -> this layout's (dV, dg, db), by the chain rule the GPU's chain kernel applies
  func chain(old d: [Float], theta t: [Float], _ lay: Layout) -> [Float] {
    var g = [Float](repeating: 0, count: count)
    func block(_ n: String) -> (Int, [Int]) { let b = lay.blocks.first { $0.name == n }!; return (b.offset, b.shape) }
    func put(_ m: NetMat, _ rows: [(String, String?)]) {
      var r = 0
      for (w, b) in rows {
        let (wo, shape) = block(w)
        for i in 0..<shape[0] {
          let row = r + i
          var s = 0.0, dot = 0.0
          for j in 0..<m.cols { let v = Double(t[m.v + row * m.cols + j]); s += v * v; dot += Double(d[wo + i * shape[1] + j]) * v }
          let k = 1 / max(s.squareRoot(), 1e-30); dot *= k
          for j in 0..<m.cols { g[m.v + row * m.cols + j] = Float(Double(t[m.g + row]) * k * (Double(d[wo + i * shape[1] + j]) - dot * k * Double(t[m.v + row * m.cols + j]))) }
          g[m.g + row] = Float(dot)
          if let b = b { g[m.g + m.rows + row] = d[block(b).0 + i] }
        }
        r += shape[0]
      }
    }
    put(mat("inp"), [("inp", "binp")])
    for l in 0..<2 { put(mat("zh\(l)"), [("z\(l)", "bz\(l)"), ("h\(l)", "bh\(l)")]); put(mat("up\(l)"), [("up\(l)", "bu\(l)")]); put(mat("down\(l)"), [("down\(l)", "bd\(l)")]) }
    put(mat("out"), [("out", nil)])
    return g
  }
  func toOld(_ t: [Float], _ lay: Layout) -> [Float] {
    var o = [Float](repeating: 0, count: lay.count)
    func block(_ n: String) -> (Int, [Int]) { let b = lay.blocks.first { $0.name == n }!; return (b.offset, b.shape) }
    func get(_ m: NetMat, _ rows: [(String, String?)]) {
      var r = 0
      for (w, b) in rows {
        let (wo, shape) = block(w)
        for i in 0..<shape[0] {
          var s = 0.0; for j in 0..<m.cols { s += Double(t[m.v + (r + i) * m.cols + j]) * Double(t[m.v + (r + i) * m.cols + j]) }
          let k = Double(t[m.g + r + i]) / max(s.squareRoot(), 1e-30)
          for j in 0..<m.cols { o[wo + i * shape[1] + j] = Float(k * Double(t[m.v + (r + i) * m.cols + j])) }
          if let b = b { o[block(b).0 + i] = t[m.g + m.rows + r + i] }
        }
        r += shape[0]
      }
    }
    get(mat("inp"), [("inp", "binp")])
    for l in 0..<2 { get(mat("zh\(l)"), [("z\(l)", "bz\(l)"), ("h\(l)", "bh\(l)")]); get(mat("up\(l)"), [("up\(l)", "bu\(l)")]); get(mat("down\(l)"), [("down\(l)", "bd\(l)")]) }
    get(mat("out"), [("out", nil)])
    return o
  }
}

func pad64(_ n: Int) -> Int { (n + 63) / 64 * 64 }
struct Codes { let q: MTLBuffer, s: MTLBuffer }   // int8 codes [rows][cols], a scale a row
let NONE = UInt32.max
func ident(_ off: Int = 0) -> [UInt32] { [NONE, 0, UInt32(off), 0] }   // a RowMap: rows from `off`

final class Net {
  let device: MTLDevice, queue: MTLCommandQueue, lay: NetLayout
  var pipes: [String: MTLComputePipelineState] = [:], library: MTLLibrary? = nil
  // a product's pipeline, compiled for its epilogue flags (int8.metal's FLAGS function constant), once each
  func pipe(_ name: String, _ flags: UInt32) -> MTLComputePipelineState {
    let key = "\(name) \(flags)"
    if let p = pipes[key] { return p }
    let c = MTLFunctionConstantValues(); var f = flags; c.setConstantValue(&f, type: .uint, index: 0)
    let p = try! device.makeComputePipelineState(function: try! library!.makeFunction(name: name, constantValues: c)); pipes[key] = p; return p
  }
  let runs: Int, refreshes: Int, laneCap: Int, ctxCap: Int, cap: Int
  // parameters, optimizer state
  let theta: MTLBuffer, m: MTLBuffer, v: MTLBuffer, grads: MTLBuffer, gram: MTLBuffer, alphaBuf: MTLBuffer, matsBuf: MTLBuffer, pinRows: MTLBuffer, pinCount: Int
  var step = 0, seed: UInt32 = 1
  // each matrix: fp32 W, its forward codes (rows quantized along its columns) and its dgrad codes (columns, along its rows)
  var Woff: [String: Int] = [:], Wall: MTLBuffer! = nil, WoffBuf: MTLBuffer! = nil   // every matrix's W, one arena
  // every matrix's codes, four arenas: rows' (wf: codes [rows][pad64 cols], wfs: a scale a row) and columns' (wb: codes
  // [pad64 rows][cols], wbs: a scale a column); offsets in floats (bytes / 4 for codes, whole 64-byte blocks)
  var wfAll: MTLBuffer! = nil, wfsAll: MTLBuffer! = nil, wbAll: MTLBuffer! = nil, wbsAll: MTLBuffer! = nil
  var wfOff: [String: Int] = [:], wfsOff: [String: Int] = [:], wbOff: [String: Int] = [:], wbsOff: [String: Int] = [:]
  var rowsMats: MTLBuffer! = nil, rowsItems: MTLBuffer! = nil, rowsCount = 0, colsMats: MTLBuffer! = nil, colsItems: MTLBuffer! = nil, colsCount = 0
  var dWoff: [String: Int] = [:], dWall: MTLBuffer! = nil, dWinfo: MTLBuffer! = nil   // every matrix's weight gradient [3][rows][cols + 1], one arena; (offset, bias) a matrix
  // activations (fp32), the backward's (tasks stacked), the codes' scratch
  let T: MTLBuffer, X: [MTLBuffer], ZC: [MTLBuffer], H: [MTLBuffer], XM: [MTLBuffer], U: [MTLBuffer], Nn: MTLBuffer, inv: MTLBuffer, nextScratch: MTLBuffer
  let dX2: MTLBuffer, dU: MTLBuffer, dXM: MTLBuffer, dZC: MTLBuffer, dXa: MTLBuffer, dXb: MTLBuffer, dhp: MTLBuffer, inj: MTLBuffer
  var segBuf: MTLBuffer   // the learning batch's segments (its own buffer, set by encodeLearn)
  let cA: Codes, cB: Codes, qx: MTLBuffer, sx: MTLBuffer, qy: MTLBuffer, sy: MTLBuffer, partial: MTLBuffer
  // the call being encoded: one command buffer, its compute encoder (profiling: one encoder a kernel, timestamps at each
  // encoder's start and end, so a kernel's GPU time; the labels "phase kernel")
  var cb: MTLCommandBuffer!, e: MTLComputeCommandEncoder!, phase = ""
  var profiling = false, samples: MTLCounterSampleBuffer? = nil, labels: [String] = []
  func open() {
    if profiling, let sb = samples, labels.count < 2046 {
      let d = MTLComputePassDescriptor()
      d.sampleBufferAttachments[0]!.sampleBuffer = sb
      d.sampleBufferAttachments[0]!.startOfEncoderSampleIndex = 2 * labels.count; d.sampleBufferAttachments[0]!.endOfEncoderSampleIndex = 2 * labels.count + 1
      e = cb.makeComputeCommandEncoder(descriptor: d)!
    } else { e = cb.makeComputeCommandEncoder()! }
  }
  func begin() { cb = queue.makeCommandBuffer()!; labels = []; open() }
  func mark(_ kernel: String) { if profiling { e.endEncoding(); labels.append(phase + " " + kernel); open() } }
  func blit(_ f: (MTLBlitCommandEncoder) -> Void) { e.endEncoding(); let b = cb.makeBlitCommandEncoder()!; f(b); b.endEncoding(); if profiling { labels.append(phase + " (blit)") }; open() }
  var lastGPU = 0.0   // the last call's command buffer, GPU start to end, ns
  func finish(_ what: String) {
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { fatalError("\(what): \(err)") }
    lastGPU = (cb.gpuEndTime - cb.gpuStartTime) * 1e9
  }
  // the last call's GPU time by label (summed over its kernels of that label), in order of first appearance; and the total
  func profile() -> [(String, Int, Double)] {
    guard let sb = samples, profiling, !labels.isEmpty else { return [] }
    let n = labels.count + 1   // the last open() after the final mark: its encoder ends at finish
    guard let data = try? sb.resolveCounterRange(0..<(2 * min(n, 2048))) else { return [] }
    let ts = data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
    var out: [(String, Int, Double)] = [], at: [String: Int] = [:]
    for i in 1..<min(n, ts.count / 2) {   // encoder i ran label i - 1's kernel (encoder 0: the call's first, before any mark)
      let t0 = ts[2 * i], t1 = ts[2 * i + 1]
      if t0 == 0 || t1 < t0 || t0 == UInt64.max { continue }
      let name = labels[i - 1], dt = Double(t1 - t0) * tick
      if let k = at[name] { out[k].1 += 1; out[k].2 += dt } else { at[name] = out.count; out.append((name, 1, dt)) }
    }
    let sum = out.reduce(0) { $0 + $1.2 }   // timestamps' unit unknown: scaled so the kernels sum to the buffer's GPU time
    return out.map { ($0.0, $0.1, $0.2 / max(sum, 1e-30) * lastGPU) }
  }
  var tick = 1.0   // ns a GPU timestamp tick (calibrated in enableProfiling)
  func enableProfiling() {
    let d = MTLCounterSampleBufferDescriptor()
    d.counterSet = device.counterSets!.first { $0.name == MTLCommonCounterSet.timestamp.rawValue }
    d.storageMode = .shared; d.sampleCount = 4096
    samples = try! device.makeCounterSampleBuffer(descriptor: d); profiling = true
    let (c0, g0) = device.sampleTimestamps(); usleep(200000); let (c1, g1) = device.sampleTimestamps()
    var tb = mach_timebase_info_data_t(); mach_timebase_info(&tb)
    tick = Double(c1 - c0) * Double(tb.numer) / Double(tb.denom) / Double(g1 - g0)
  }

  init(_ device: MTLDevice, _ queue: MTLCommandQueue, library: String, runs: Int, refreshes: Int, contexts: Int, initial: [Float]) {
    self.device = device; self.queue = queue; lay = NetLayout(); precondition(initial.count == lay.count)
    let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: library))
    self.library = lib
    for n in ["credit_draws", "credit_rows", "seg_sum", "quant_rows", "quant_cols", "quant_rows_h", "quant_rows_b", "quant_cols_h", "quant_cols_b", "gemm_tn", "reduce_blocks", "quant_rows_mats", "quant_cols_mats", "scan_fwd", "ctx_fwd", "ctx_bwd", "scan_bwd", "weff", "pin", "chain", "gram", "adam", "upgrad_weights"] {
      pipes[n] = try! device.makeComputePipelineState(function: lib.makeFunction(name: n)!)
    }
    let laneCap = runs * refreshes * L, cap = laneCap + contexts
    self.runs = runs; self.refreshes = refreshes; self.laneCap = laneCap; ctxCap = contexts; self.cap = cap
    func f(_ n: Int) -> MTLBuffer { device.makeBuffer(length: max(16, n * 4), options: .storageModeShared)! }
    func p(_ n: Int) -> MTLBuffer { device.makeBuffer(length: max(16, n * 4), options: .storageModePrivate)! }
    func b8(_ n: Int) -> MTLBuffer { device.makeBuffer(length: max(16, n), options: .storageModePrivate)! }
    let P = lay.count
    theta = device.makeBuffer(bytes: initial, length: P * 4, options: .storageModeShared)!
    m = p(P); v = p(P); grads = f(3 * P); gram = f(9); alphaBuf = f(4)
    var mats: [UInt32] = [], rows: [UInt32] = [], info: [UInt32] = [], dWsize = 0, woffs: [UInt32] = [], Wsize = 0, wfSize = 0, wfsSize = 0, wbSize = 0, wbsSize = 0
    for (i, mt) in lay.mats.enumerated() {
      mats += [UInt32(mt.v), UInt32(mt.rows), UInt32(mt.cols), UInt32(mt.g)]
      for r in 0..<mt.rows { rows += [UInt32(i), UInt32(r)] }
      Woff[mt.name] = Wsize; woffs.append(UInt32(Wsize)); Wsize += mt.rows * mt.cols
      wfOff[mt.name] = wfSize / 4; wfSize += mt.rows * pad64(mt.cols); wfsOff[mt.name] = wfsSize; wfsSize += mt.rows
      wbOff[mt.name] = wbSize / 4; wbSize += pad64(mt.rows) * mt.cols; wbsOff[mt.name] = wbsSize; wbsSize += mt.cols
      dWoff[mt.name] = dWsize; info += [UInt32(dWsize), mt.bias ? 1 : 0]; dWsize += 3 * mt.rows * (mt.cols + 1)
    }
    wfAll = b8(wfSize); wfsAll = p(wfsSize); wbAll = b8(wbSize); wbsAll = p(wbsSize)
    do {   // the codes' work: rows (quant_rows' lanes a row by its width) and 32-column blocks, a matrix after another
      var rm: [UInt32] = [], ri: [UInt32] = [], cm: [UInt32] = [], ci: [UInt32] = []
      for (i, mt) in lay.mats.enumerated() {
        let cols = pad64(mt.cols), lpr = cols <= 64 ? 8 : cols <= 128 ? 16 : 32
        rm += [UInt32(mt.rows), UInt32(cols), UInt32(mt.cols), UInt32(mt.cols), UInt32(Woff[mt.name]!), UInt32(wfOff[mt.name]! * 4), UInt32(wfsOff[mt.name]!), 0]
        for g in 0..<((mt.rows + 32 / lpr - 1) / (32 / lpr)) { ri += [UInt32(i), UInt32(g)] }
        cm += [UInt32(pad64(mt.rows)), UInt32(mt.cols), UInt32(mt.cols), UInt32(mt.rows), UInt32(Woff[mt.name]!), UInt32(wbOff[mt.name]! * 4), UInt32(wbsOff[mt.name]!), UInt32(pad64(mt.rows))]
        for b in 0..<((mt.cols + 31) / 32) { ci += [UInt32(i), UInt32(b)] }
      }
      func sh(_ a: [UInt32]) -> MTLBuffer { device.makeBuffer(bytes: a, length: a.count * 4, options: .storageModeShared)! }
      rowsMats = sh(rm); rowsItems = sh(ri); rowsCount = ri.count / 2; colsMats = sh(cm); colsItems = sh(ci); colsCount = ci.count / 2
    }
    Wall = p(Wsize); WoffBuf = device.makeBuffer(bytes: woffs, length: woffs.count * 4, options: .storageModeShared)!; dWall = p(dWsize); dWinfo = device.makeBuffer(bytes: info, length: info.count * 4, options: .storageModeShared)!
    matsBuf = device.makeBuffer(bytes: mats, length: mats.count * 4, options: .storageModeShared)!
    pinRows = device.makeBuffer(bytes: rows, length: rows.count * 4, options: .storageModeShared)!; pinCount = rows.count / 2
    T = p(cap * IN); X = (0..<3).map { _ in p(cap * D) }; ZC = (0..<2).map { _ in p(cap * D) }   /* fp16 */; H = (0..<2).map { _ in p(laneCap * D) }
    XM = (0..<2).map { _ in p(cap * D) }; U = (0..<2).map { _ in p(cap * E / 2) } /* fp16 */; Nn = p(contexts * D); inv = p(contexts); nextScratch = p(runs * 2 * D)
    dX2 = p(3 * contexts * D); dU = p(3 * cap * E / 2) /* bf16 */; dXM = p(3 * cap * D); dZC = p(3 * cap * D) /* bf16 */; dXa = p(3 * cap * D); dXb = p(3 * cap * D)
    dhp = p(3 * contexts * D); inj = p(3 * runs * refreshes * D); segBuf = f(runs * refreshes * 2)
    let rowsPad = (cap + 255) / 256 * 256
    cA = Codes(q: b8(3 * cap * 256), s: p(3 * cap)); cB = Codes(q: b8(3 * cap * 256), s: p(3 * cap)); qx = b8(rowsPad * 256); sx = p(256 * rowsPad / 256); qy = b8(3 * rowsPad * 256); sy = p(3 * 256 * rowsPad / 256)
    partial = p(3 * (rowsPad / 256 + 1) * 2 * D * (E + 1))   // a 256-row block's partial sums each, three tasks, the widest wgrad (M N <= 128 x 193)
    let cb = queue.makeCommandBuffer()!, bl = cb.makeBlitCommandEncoder()!
    bl.fill(buffer: m, range: 0..<(P * 4), value: 0); bl.fill(buffer: v, range: 0..<(P * 4), value: 0); bl.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    refresh()
  }

  // ---- encoders ----
  func set(_ name: String, _ bufs: [(MTLBuffer?, Int)], _ params: [[UInt32]]) {
    mark(name)
    current = pipes[name]!; e.setComputePipelineState(current!)
    for (i, (b, off)) in bufs.enumerated() { e.setBuffer(b, offset: off * 4, index: i) }
    for (k, a) in params.enumerated() { a.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: bufs.count + k) } }
  }
  var current: MTLComputePipelineState? = nil   // the pipeline set last: a dispatch's threadgroup its own widest
  func threads(_ n: Int) { if n > 0, let p = current { e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: p.maxTotalThreadsPerThreadgroup, height: 1, depth: 1)) } }
  // the scans: a threadgroup a sequence (n of them), a thread a position (scan.h)
  func sequences(_ n: Int, _ steps: Int) {
    if n == 0 { return }
    e.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: (steps + 31) / 32 * 32, height: 1, depth: 1))
  }
  func nextSeed() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed | 1 }
  // rows of x (map, ld, the first `valid` columns) -> codes (cols wide) into c from code row `at`; returns the seed
  @discardableResult
  func qrows(_ x: MTLBuffer, _ xoff: Int, rows: Int, cols: Int, ld: Int, valid: Int, map: [UInt32], stochastic: Bool, into c: Codes, at: Int = 0, seed given: UInt32? = nil, type: String = "") -> UInt32 {
    let sd = given ?? (stochastic ? nextSeed() : 0)
    if rows == 0 { return sd }
    mark("quant_rows" + type)
    current = pipes["quant_rows" + type]!; e.setComputePipelineState(current!)
    e.setBuffer(x, offset: xoff * (type == "" ? 4 : 2), index: 0); e.setBuffer(c.q, offset: at * cols, index: 1); e.setBuffer(c.s, offset: at * 4, index: 2)
    let pr: [UInt32] = [UInt32(rows), UInt32(cols), UInt32(ld), UInt32(valid), 1, sd, 0, 0] + map
    pr.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 3) }
    let perSimd = cols <= 64 ? 4 : cols <= 128 ? 2 : 1   // rows a SIMD group (int8.metal's quant_rows)
    threads((rows + perSimd - 1) / perSimd * 32)
    return sd
  }
  // a product of codes `src` (rows M) against matrix `w`: forward (nt: N = its rows) or dgrad (nn: N = its columns). Its
  // epilogue's flags (int8.metal); out (buffer, map, ld) stored if given; codes of the result into `dst` if given (the
  // fused product, stochastic if `stochastic`; the next product's input), inv for RMS / RMSB. Returns the codes' seed
  @discardableResult
  func product(_ w: String, forward: Bool, M: Int, src: Codes, flags: UInt32, out: (MTLBuffer, [UInt32], Int)?, res: (MTLBuffer, [UInt32], Int)? = nil,
               aux: (MTLBuffer, Int, Int, Int)? = nil, dst: Codes? = nil, stochastic: Bool = false, inv: MTLBuffer? = nil) -> UInt32 {
    let mt = lay.mat(w), N = forward ? mt.rows : mt.cols, K = forward ? pad64(mt.cols) : pad64(mt.rows)
    let fused = dst != nil || flags & (128 | 256 | 512 | 1024) != 0, sd: UInt32 = dst != nil && stochastic ? nextSeed() : 0
    if M == 0 { return sd }
    let bias: (MTLBuffer?, Int) = mt.bias && forward ? (theta, mt.g + mt.rows) : (nil, 0)
    let f = flags | (bias.0 != nil ? 1 : 0) | (fused && out != nil ? 32 : 0) | (dst != nil ? 64 : 0)
    let params: [UInt32] = [UInt32(M), UInt32(N), UInt32(K), f, UInt32(out?.2 ?? 0), UInt32(res?.2 ?? 0), UInt32(aux?.3 ?? 0), UInt32(aux?.2 ?? 1)]
      + (out?.1 ?? ident()) + (res?.1 ?? ident()) + [UInt32(aux?.1 ?? 0), sd, 0, 0]
    var bufs: [(MTLBuffer?, Int)] = [(src.q, 0), forward ? (wfAll, wfOff[w]!) : (wbAll, wbOff[w]!), (src.s, 0), forward ? (wfsAll, wfsOff[w]!) : (wbsAll, wbsOff[w]!), (out?.0, 0), bias, (res?.0, 0), (aux?.0, 0)]
    if fused {
      precondition(N % 64 == 0, "a fused product's width is whole 64-blocks")
      mark(forward ? "gemmq_nt" : "gemmq_nn")
      e.setComputePipelineState(pipe(forward ? "gemmq_nt" : "gemmq_nn", f))
      bufs += [(dst?.q, 0), (dst?.s, 0), (inv, 0)]
      for (i, (b, off)) in bufs.enumerated() { e.setBuffer(b, offset: off * 4, index: i < 8 ? i : i + 1) }   // 8: the parameters
      params.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 8) }
      e.setThreadgroupMemoryLength(32 * (pad64(N) + 4) * 4, index: 0)   // the 32 rows' int32 sums
      e.dispatchThreadgroups(MTLSize(width: (M + 31) / 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    } else {
      mark(forward ? "gemm_nt" : "gemm_nn")
      e.setComputePipelineState(pipe(forward ? "gemm_nt" : "gemm_nn", f))
      for (i, (b, off)) in bufs.enumerated() { e.setBuffer(b, offset: off * 4, index: i) }
      params.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 8) }
      e.dispatchThreadgroups(MTLSize(width: (N + 31) / 32, height: (M + 63) / 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    return sd
  }
  // NETCHECK: a fused producer's codes against quant_rows of what it stored (same seed), compared after the call
  var checking = false, probes: [(String, MTLBuffer, MTLBuffer, MTLBuffer, MTLBuffer, Int)] = []
  func probe(_ label: String, _ c: Codes, at: Int, rows: Int, cols: Int, x: MTLBuffer, xoff: Int, ld: Int, valid: Int, seed: UInt32, type: String = "") {
    if !checking || rows == 0 { return }
    let got = device.makeBuffer(length: rows * cols, options: .storageModeShared)!, gs = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
    let want = device.makeBuffer(length: rows * cols, options: .storageModeShared)!, ws = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
    blit { $0.copy(from: c.q, sourceOffset: at * cols, to: got, destinationOffset: 0, size: rows * cols); $0.copy(from: c.s, sourceOffset: at * 4, to: gs, destinationOffset: 0, size: rows * 4) }
    // quant_rows indexes the stochastic hash by its own row: shift the seed's rows by giving it the same logical rows
    qrows(x, xoff, rows: rows, cols: cols, ld: ld, valid: valid, map: ident(), stochastic: false, into: Codes(q: want, s: ws), seed: seed, type: type)
    probes.append((label + (at != 0 && seed != 0 ? " (row offset: stochastic indices differ)" : ""), got, want, gs, ws, rows * cols))
  }
  func probeReport() -> [String] {
    defer { probes = [] }
    return probes.map { (label, got, want, gs, ws, n) in
      let a = got.contents().bindMemory(to: Int8.self, capacity: n), b = want.contents().bindMemory(to: Int8.self, capacity: n)
      var d = 0, big = 0; for i in 0..<n where a[i] != b[i] { d += 1; if abs(Int(a[i]) - Int(b[i])) > 1 { big += 1 } }
      let r = n / max(1, gs.length / 4), sa = gs.contents().bindMemory(to: Float.self, capacity: n / r), sb = ws.contents().bindMemory(to: Float.self, capacity: n / r)
      var ds = 0; for i in 0..<(n / r) where sa[i] != sb[i] { ds += 1 }
      return String(format: "  %-52@ codes differing %d of %d (by more than 1: %d), scales %d", label, d, n, big, ds)
    }
  }
  // dW[w][task] = dY_t^T [X, 1]: dY rows `rows` from row (t stride + yoff) of dy (width M = w's rows, ld ldy); X rows from
  // row xoff of x (w's columns, ld ldx); stochastic codes for dY; the bias's column if w has one
  func wgrad(_ w: String, dy: MTLBuffer, yoff: Int, stride: Int, ldy: Int, x: MTLBuffer, xoff: Int, ldx: Int, rows: Int, dyType: String = "", xType: String = "") {
    let mt = lay.mat(w), K = (rows + 255) / 256 * 256, nb = K / 256, Nw = mt.cols + (mt.bias ? 1 : 0), M = mt.rows
    if rows == 0 { return }
    set("quant_cols" + xType, [(x, 0), (qx, 0), (sx, 0)], [[UInt32(K), UInt32(Nw), UInt32(ldx), UInt32(rows), 256, 1, 0, mt.bias ? UInt32(mt.cols) : NONE, UInt32(xoff), UInt32(mt.cols), 0, 0]])
    e.dispatchThreadgroups(MTLSize(width: (Nw + 31) / 32, height: nb, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    // the three tasks in each dispatch: dy's columns quantized (task t's rows t stride apart), a threadgroup a 256-row
    // block a task (the exactness unit, its partial sums), then the blocks' pairwise tree a task
    set("quant_cols" + dyType, [(dy, 0), (qy, 0), (sy, 0)], [[UInt32(K), UInt32(M), UInt32(ldy), UInt32(rows), 256, 1, nextSeed(), NONE, UInt32(yoff), UInt32(M), UInt32(stride), 0]])
    e.dispatchThreadgroups(MTLSize(width: (M + 31) / 32, height: nb, depth: 3), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    set("gemm_tn", [(qy, 0), (qx, 0), (sy, 0), (sx, 0), (partial, 0)], [[UInt32(M), UInt32(Nw), UInt32(K), 1, UInt32(nb)]])
    e.dispatchThreadgroups(MTLSize(width: (Nw + 31) / 32, height: (M + 63) / 64, depth: 3 * nb), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    set("reduce_blocks", [(partial, 0), (dWall, dWoff[w]!)], [[UInt32(M * Nw), UInt32(nb), 3]]); threads(3 * M * Nw)
  }
  // each matrix's W, and its two sets of codes, from theta
  func encodeWeights() {
    var m = UInt32(pinCount)
    mark("weff")   // every matrix's W, one dispatch
    current = pipes["weff"]!; e.setComputePipelineState(current!)
    for (i, b) in [theta, Wall, matsBuf, pinRows, WoffBuf].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&m, length: 4, index: 5); threads(pinCount)
    var n = UInt32(rowsCount)   // every matrix's codes: rows' then columns', a dispatch each (int8.metal's cores)
    mark("quant_rows_mats")
    current = pipes["quant_rows_mats"]!; e.setComputePipelineState(current!)
    for (i, b) in [Wall, wfAll, wfsAll, rowsMats, rowsItems].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&n, length: 4, index: 5); threads(rowsCount * 32)
    mark("quant_cols_mats")
    current = pipes["quant_cols_mats"]!; e.setComputePipelineState(current!)
    for (i, b) in [Wall, wbAll, wbsAll, colsMats, colsItems].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.dispatchThreadgroups(MTLSize(width: colsCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  }


  // the forward: B runs' K refreshes (lane tokens [B K L][IN] and masks [B K L] in `lanes`, `mask`, their states h0
  // [B][2][D]), R contexts (tokens [R][IN], each its lane row pos[r]): w [R][FS] and each run's last states [B][2][D].
  // Each producer writes the next product's codes (cA / cB in turn); `keep`: store what the backward reads
  func encodeForward(lanes: MTLBuffer, mask: MTLBuffer, h0: MTLBuffer, tokens: MTLBuffer, pos: MTLBuffer, B: Int, K: Int, R: Int, w: MTLBuffer, next: MTLBuffer, keep: Bool) {
    let LR = B * K * L, N = LR + R
    precondition(B <= runs && K <= refreshes && R <= ctxCap, "a batch past the network's buffers")
    qrows(lanes, 0, rows: LR, cols: 64, ld: IN, valid: IN, map: ident(), stochastic: false, into: cA)
    qrows(tokens, 0, rows: R, cols: 64, ld: IN, valid: IN, map: ident(), stochastic: false, into: cA, at: LR)   // the contexts' after the lanes'
    product("inp", forward: true, M: N, src: cA, flags: 0, out: (X[0], ident(), D), dst: cB)
    probe("inp -> X0 codes", cB, at: 0, rows: N, cols: 64, x: X[0], xoff: 0, ld: D, valid: D, seed: 0)
    var xc = cB, other = cA   // X_l's codes; the free buffer
    for l in 0..<2 {
      product("zh\(l)", forward: true, M: N, src: xc, flags: 4 | 512, out: (ZC[l], ident(), 2 * D))   // the gates, fp16
      let r0 = l == 0 ? 0 : LR, n = N - r0   // layer 1's lanes: their states only
      set("scan_fwd", [(ZC[l], 0), (mask, 0), (h0, 0), (H[l], 0), (X[l], 0), (XM[l], 0), (next, 0)], [[UInt32(B), UInt32(K * L), UInt32(l), l == 0 ? 1 : 0]])
      sequences(B * D, K * L)
      if R > 0 {
        set("ctx_fwd", [(ZC[l], LR * D) /* fp16: LR 2D halves */, (H[l], 0), (pos, 0), (X[l], LR * D), (XM[l], LR * D)], [[UInt32(R), UInt32(l == 0 ? LR : 0)]])
        threads(R * 32)
      }
      qrows(XM[l], r0 * D, rows: n, cols: 64, ld: D, valid: D, map: ident(), stochastic: false, into: other)   // every row's XM codes, one pass
      probe("scan/ctx -> XM\(l) codes", other, at: 0, rows: n, cols: 64, x: XM[l], xoff: r0 * D, ld: D, valid: D, seed: 0)
      let xm = other; other = xc
      product("up\(l)", forward: true, M: n, src: xm, flags: 2 | 512, out: keep ? (U[l], ident(r0), E) : nil, dst: other)   // U fp16
      if keep { probe("up\(l) -> U\(l) codes (U fp16)", other, at: 0, rows: n, cols: 192, x: U[l], xoff: r0 * E, ld: E, valid: E, seed: 0, type: "_h") }
      let uc = other; other = xm
      if l == 0 {
        product("down0", forward: true, M: n, src: uc, flags: 16, out: (X[1], ident(), D), res: (XM[0], ident(), D), dst: other)
        probe("down0 -> X1 codes", other, at: 0, rows: N, cols: 64, x: X[1], xoff: 0, ld: D, valid: D, seed: 0)
      } else if R > 0 {   // the readout's rms fused: n = x2 / rms(x2), its codes
        product("down1", forward: true, M: R, src: uc, flags: 16 | 128, out: keep ? (Nn, ident(), D) : nil, res: (XM[1], ident(LR), D), dst: other, inv: inv)
        if keep { probe("down1 + rms -> Nn codes", other, at: 0, rows: R, cols: 64, x: Nn, xoff: 0, ld: D, valid: D, seed: 0) }
      }
      xc = other; other = uc
    }
    if R > 0 { product("out", forward: true, M: R, src: xc, flags: 0, out: (w, ident(), FS)) }
  }
  // the same, committed and not waited for: `done` runs when it completes, with its GPU time (s)
  // a forward, encoded (the caller adds its completion handler, commits, and waits where it reads w and next)
  func forwardEncode(lanes: MTLBuffer, mask: MTLBuffer, h0: MTLBuffer, tokens: MTLBuffer, pos: MTLBuffer, B: Int, R: Int, w: MTLBuffer, next: MTLBuffer) -> MTLCommandBuffer {
    begin(); phase = "forward"
    encodeForward(lanes: lanes, mask: mask, h0: h0, tokens: tokens, pos: pos, B: B, K: 1, R: R, w: w, next: next, keep: false)
    e.endEncoding()
    return cb
  }

  // a learning batch: the forward again, then the three tasks' gradients of sum(w . dw_t) (dw [3][R][FS]) into grads
  // [3][P], and their Gram matrix. seg [B K] (as pairs): each refresh's context rows (contiguous)
  // a learning batch's credit (route.h route_draw records; each row's draws first[row] .. first[row + 1], its behaviour
  // w, its task-0 scale (its run's throughput advantage over the batch), the tasks' common scale)
  struct Credit { let draws: MTLBuffer, count: Int, first: MTLBuffer, wrows: MTLBuffer, scale0: MTLBuffer, scale: Float, ll: MTLBuffer }
  var dwBuf: MTLBuffer? = nil, diffBuf: MTLBuffer? = nil
  func room(_ b: inout MTLBuffer?, _ bytes: Int, shared: Bool) -> MTLBuffer {   // grown when a batch needs more
    if b == nil || b!.length < bytes { b = device.makeBuffer(length: max(16, bytes), options: shared ? .storageModeShared : .storageModePrivate)! }
    return b!
  }
  // encoded into the open command buffer (route_net_step's): the gradients into grads [3][P]
  func encodeLearn(lanes: MTLBuffer, mask: MTLBuffer, h0: MTLBuffer, tokens: MTLBuffer, pos: MTLBuffer, seg: MTLBuffer, B: Int, K: Int, R: Int, credit c: Credit, w: MTLBuffer) {
    let dw = room(&dwBuf, 3 * R * FS * 4, shared: true), diff = room(&diffBuf, c.count * FS * 4, shared: false), ll = c.ll
    let LR = B * K * L, N = LR + R, P = lay.count
    segBuf = seg
    phase = "learn forward"
    blit { $0.fill(buffer: self.dXM, range: 0..<(3 * N * D * 4), value: 0); $0.fill(buffer: self.grads, range: 0..<(3 * P * 4), value: 0) }
    encodeForward(lanes: lanes, mask: mask, h0: h0, tokens: tokens, pos: pos, B: B, K: K, R: R, w: w, next: nextScratch, keep: true)
    phase = "credit"   // each draw's score function under this forward's w, times its importance ratio; each row's dw
    if c.count > 0 {
      set("credit_draws", [(c.draws, 0), (c.wrows, 0), (w, 0), (diff, 0), (ll, 0), (c.first, 0)], [[UInt32(c.count), UInt32(R)]]); threads(c.count * 32)
    }
    var sc = c.scale
    set("credit_rows", [(diff, 0), (c.draws, 0), (c.first, 0), (c.scale0, 0), (dw, 0)], [[UInt32(R)]]); withUnsafeBytes(of: &sc) { e.setBytes($0.baseAddress!, length: 4, index: 6) }
    threads(3 * R * FS)
    // readout: dX2 = rms's backward of dw out (fused), its codes; out's wgrad
    phase = "back readout"
    qrows(dw, 0, rows: 3 * R, cols: 64, ld: FS, valid: FS, map: ident(), stochastic: true, into: cA)
    var sd = product("out", forward: false, M: 3 * R, src: cA, flags: 256, out: (dX2, ident(), D), aux: (Nn, 0, R, D), dst: cB, stochastic: true, inv: inv)
    probe("out dgrad + rms bwd -> dX2 codes", cB, at: 0, rows: 3 * R, cols: 64, x: dX2, xoff: 0, ld: D, valid: D, seed: sd)
    wgrad("out", dy: dw, yoff: 0, stride: R, ldy: FS, x: Nn, xoff: 0, ldx: D, rows: R)
    phase = "back layer 1"
    // layer 1's feed-forward, its context rows: dX2 [3][R] -> dU [3][R] -> dXM's context rows ([3][N] from LR)
    sd = product("down1", forward: false, M: 3 * R, src: cB, flags: 8 | 1024 | 2048, out: (dU, ident(), E), aux: (U[1], LR, R, E), dst: cA, stochastic: true)
    probe("down1 dgrad (masked) -> dU codes (dU bf16)", cA, at: 0, rows: 3 * R, cols: 192, x: dU, xoff: 0, ld: E, valid: E, seed: sd, type: "_b")
    wgrad("down1", dy: dX2, yoff: 0, stride: R, ldy: D, x: U[1], xoff: LR, ldx: E, rows: R, xType: "_h")
    product("up1", forward: false, M: 3 * R, src: cA, flags: 16, out: (dXM, [UInt32(R), UInt32(N), UInt32(LR), 0], D), res: (dX2, ident(), D))
    wgrad("up1", dy: dU, yoff: 0, stride: R, ldy: E, x: XM[1], xoff: LR, ldx: D, rows: R, dyType: "_b")
    layerBack(1, B: B, K: K, R: R, h0: h0, mask: mask, pos: pos, upstreamDX: nil, out: dXa, codes: cB, next: cA)
    phase = "back layer 0"
    // layer 0, every row: dXa -> dU -> dXM -> (the recurrence) -> dZC -> dXb
    sd = product("down0", forward: false, M: 3 * N, src: cA, flags: 8 | 1024 | 2048, out: (dU, ident(), E), aux: (U[0], 0, N, E), dst: cB, stochastic: true)
    probe("down0 dgrad (masked) -> dU codes (dU bf16)", cB, at: 0, rows: 3 * N, cols: 192, x: dU, xoff: 0, ld: E, valid: E, seed: sd, type: "_b")
    wgrad("down0", dy: dXa, yoff: 0, stride: N, ldy: D, x: U[0], xoff: 0, ldx: E, rows: N, xType: "_h")
    product("up0", forward: false, M: 3 * N, src: cB, flags: 16, out: (dXM, ident(), D), res: (dXa, ident(), D))
    wgrad("up0", dy: dU, yoff: 0, stride: N, ldy: E, x: XM[0], xoff: 0, ldx: D, rows: N, dyType: "_b")
    layerBack(0, B: B, K: K, R: R, h0: h0, mask: mask, pos: pos, upstreamDX: dXM, out: dXb, codes: cA, next: nil)
    // the input projection's tokens: lanes' then contexts' (two sources; wgrad reads one buffer, so T holds both)
    phase = "back input"
    blit { b in b.copy(from: lanes, sourceOffset: 0, to: self.T, destinationOffset: 0, size: LR * IN * 4); if R > 0 { b.copy(from: tokens, sourceOffset: 0, to: self.T, destinationOffset: LR * IN * 4, size: R * IN * 4) } }
    wgrad("inp", dy: dXb, yoff: 0, stride: N, ldy: D, x: T, xoff: 0, ldx: IN, rows: N)
    phase = "back params"   // every matrix's chain rule, one dispatch
    set("chain", [(dWall, 0), (theta, 0), (grads, 0), (matsBuf, 0), (pinRows, 0), (dWinfo, 0)], [[UInt32(P), UInt32(pinCount), 3]])
    threads(3 * pinCount)
  }
  // a layer's recurrence and gates, backward: dXM [3][N] (its context rows' gradients; its lanes' too if upstreamDX, the
  // local path) -> dZC [3][N] (stored, and its codes into `codes`) -> out = dXM + dZC zh (dX, [3][N]; its codes into
  // `next`, the next layer's down dgrad input, if given); zh's wgrad
  func layerBack(_ l: Int, B: Int, K: Int, R: Int, h0: MTLBuffer, mask: MTLBuffer, pos: MTLBuffer, upstreamDX: MTLBuffer?, out: MTLBuffer, codes: Codes, next: Codes?) {
    let LR = B * K * L, N = LR + R, sd = nextSeed()
    if R > 0 {
      set("ctx_bwd", [(dXM, 0), (ZC[l], 0), (H[l], 0), (pos, 0), (dZC, 0), (dhp, 0)], [[UInt32(R), 3, UInt32(N), UInt32(LR), sd, 0, 0, 0]])
      threads(3 * R * 32)
    }
    set("seg_sum", [(dhp, 0), (segBuf, 0), (inj, 0)], [[UInt32(B * K), 3, UInt32(R)]]); threads(3 * B * K * D)
    set("scan_bwd", [(dXM, 0), (inj, 0), (ZC[l], 0), (mask, 0), (H[l], 0), (h0, 0), (dZC, 0)],
        [[UInt32(B), UInt32(K * L), UInt32(l), upstreamDX != nil ? 1 : 0, UInt32(K), UInt32(L), UInt32(N), 3, UInt32(R), sd, 0, 0]])
    sequences(3 * B * D, K * L)
    qrows(dZC, 0, rows: 3 * N, cols: 128, ld: 2 * D, valid: 2 * D, map: ident(), stochastic: true, into: codes, seed: sd, type: "_b")   // every row's dZC codes, one pass
    probe("scan/ctx bwd -> dZC\(l) codes (dZC bf16)", codes, at: 0, rows: 3 * N, cols: 128, x: dZC, xoff: 0, ld: 2 * D, valid: 2 * D, seed: sd, type: "_b")
    let s2 = product("zh\(l)", forward: false, M: 3 * N, src: codes, flags: 16, out: (out, ident(), D), res: (dXM, ident(), D), dst: next, stochastic: true)
    if let nx = next { probe("zh\(l) dgrad -> dX codes", nx, at: 0, rows: 3 * N, cols: 64, x: out, xoff: 0, ld: D, valid: D, seed: s2) }
    wgrad("zh\(l)", dy: dZC, yoff: 0, stride: N, ldy: 2 * D, x: X[l], xoff: 0, ldx: D, rows: N, dyType: "_b")
  }

  // ---- the optimizer: UPGrad's weights from the Gram matrix (learn.swift's upgrad), Adam, V back to unit rows, codes ----
  // the optimizer's step, encoded after the learning call (and the exchange): the Gram matrix of grads (the ranks'
  // summed, `inv` their count's inverse), UPGrad's weights (upgrad.h), Adam, V pinned, the codes rebuilt; nothing read back
  func encodeApply(rate: Double, inv: Float) {
    step += 1
    let P = lay.count
    var r = Float(rate), iv = inv, c = SIMD4<Float>(Float(1 - pow(0.9, Double(step))), Float(1 - pow(0.999, Double(step))), 0, 0), n = UInt32(P), pc = UInt32(pinCount)
    phase = "optimizer"
    set("gram", [(grads, 0), (gram, 0)], [[UInt32(P)]]); e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    mark("upgrad")
    current = pipes["upgrad_weights"]!; e.setComputePipelineState(current!)
    e.setBuffer(gram, offset: 0, index: 0); e.setBuffer(alphaBuf, offset: 0, index: 1); e.setBytes(&r, length: 4, index: 2); e.setBytes(&iv, length: 4, index: 3)
    threads(1)
    mark("adam")
    current = pipes["adam"]!; e.setComputePipelineState(current!)
    for (i, b) in [theta, m, v, grads, alphaBuf].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&c, length: 16, index: 5); e.setBytes(&n, length: 4, index: 6)
    threads(P)
    mark("pin")
    current = pipes["pin"]!; e.setComputePipelineState(current!)
    e.setBuffer(theta, offset: 0, index: 0); e.setBuffer(matsBuf, offset: 0, index: 1); e.setBuffer(pinRows, offset: 0, index: 2); e.setBytes(&pc, length: 4, index: 3)
    threads(pinCount)
    encodeWeights()
  }
  // the weights as stored for deployment (int8mm.h's format): the codes and scales the forward reads, as they are on the
  // GPU (copied out, not computed again), each matrix's bias
  func writeCodes(_ path: String) {
    var bytes = Data()
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
    u32(0x50384952); u32(1); u32(UInt32(lay.mats.count))
    let cb = queue.makeCommandBuffer()!, bl = cb.makeBlitCommandEncoder()!
    var staged: [(MTLBuffer, MTLBuffer)] = []
    for mt in lay.mats {
      let q = device.makeBuffer(length: mt.rows * pad64(mt.cols), options: .storageModeShared)!, sc = device.makeBuffer(length: mt.rows * 4, options: .storageModeShared)!
      bl.copy(from: wfAll, sourceOffset: wfOff[mt.name]! * 4, to: q, destinationOffset: 0, size: q.length); bl.copy(from: wfsAll, sourceOffset: wfsOff[mt.name]! * 4, to: sc, destinationOffset: 0, size: sc.length)
      staged.append((q, sc))
    }
    bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let th = theta.contents().bindMemory(to: Float.self, capacity: lay.count)
    for (i, mt) in lay.mats.enumerated() {
      u32(UInt32(mt.rows)); u32(UInt32(mt.cols)); u32(UInt32(pad64(mt.cols))); u32(mt.bias ? 1 : 0)
      bytes.append(Data(bytes: staged[i].0.contents(), count: staged[i].0.length)); bytes.append(Data(bytes: staged[i].1.contents(), count: staged[i].1.length))
      if mt.bias { bytes.append(Data(bytes: th + mt.g + mt.rows, count: mt.rows * 4)) }
    }
    try! bytes.write(to: URL(fileURLWithPath: path))
  }
  // after theta was written from the CPU
  func refresh() { begin(); phase = "weights"; encodeWeights(); finish("weights") }
}
