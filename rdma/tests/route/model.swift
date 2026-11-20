// The policy (route.h) on the GPU: its parameters' layout (NetLayout: theta, the optimizer's fp32 master copy) and
// the network (Net), whose kernels are fwd.metal, bwd.metal and opt.metal on mm.metal's int8 x int8 -> int32 products.
// A matrix's weights: W = diag(g) V / |V| (rows), V pinned to unit rows after each step (so W's scale is g's alone);
// a bias b beside it. Rows: lane rows first ([run][pad64(K L)]), then context rows; a context's state its refresh's.
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

// The network on the GPU (fwd.metal, bwd.metal, opt.metal on mm.metal's products): one command buffer a call, a few
// dispatches each: the forward (lanes_fwd: a run's lanes; ctx_fwd: a tile of contexts through every layer), a learning
// step (the forward storing what the backward reads, credit, ctx_bwd, lanes_bwd, the wgrad products, grads, then the
// optimizer and the codes). Buffers held from init at the capacity given; no CPU waits.
final class Net {
  let device: MTLDevice, queue: MTLCommandQueue, lay: NetLayout, library: MTLLibrary
  var pipes: [String: MTLComputePipelineState] = [:]
  // a pipeline; fwd.metal's kernels specialized by STORE (index 0) and STAGE (index 1)
  func pipe(_ name: String, store: Bool = false, stage: UInt32 = 0) -> MTLComputePipelineState {
    let key = "\(name) \(store) \(stage)"
    if let p = pipes[key] { return p }
    let f: MTLFunction
    if name == "lanes_fwd" || name == "ctx_fwd" {
      let c = MTLFunctionConstantValues(); var s = store, g = stage
      c.setConstantValue(&s, type: .bool, index: 0); c.setConstantValue(&g, type: .uint, index: 1)
      f = try! library.makeFunction(name: name, constantValues: c)
    } else { f = library.makeFunction(name: name)! }
    let p = try! device.makeComputePipelineState(function: f); pipes[key] = p; return p
  }
  let runs: Int, refreshes: Int, ctxCap: Int, klpCap: Int, RT: Int
  let theta: MTLBuffer, m: MTLBuffer, v: MTLBuffer, grads: MTLBuffer, gram: MTLBuffer
  var step = 0, seed: UInt32 = 1
  // the weights' codes (wq: bytes) and scales and biases (wsc: floats), their table (Net in tile.h), W's fp32 scratch
  let wq: MTLBuffer, wsc: MTLBuffer, Wf: MTLBuffer, netTable: MTLBuffer, wfOff: MTLBuffer, rowItems: MTLBuffer, colItems: MTLBuffer, rowCount: Int, colCount: Int
  var mats: [[UInt32]] = []   // each matrix's MatW
  // the forward's states; what the learning forward keeps; the wgrad operands (gq codes, gs scales and column sums)
  let st: MTLBuffer, lzc: MTLBuffer, lh: MTLBuffer, lu: MTLBuffer, czc: MTLBuffer, cu: MTLBuffer, cn: MTLBuffer, cinv: MTLBuffer, nextScratch: MTLBuffer
  let gq: MTLBuffer, gs: MTLBuffer, gradTable: MTLBuffer, dhp: MTLBuffer, partial: MTLBuffer
  var matg: [[UInt32]] = [], partialOff: [Int] = []
  var dwBuf: MTLBuffer? = nil, diffBuf: MTLBuffer? = nil
  // the call being encoded; profiling: an encoder a kernel, timestamps at each encoder's ends ("phase kernel")
  var cb: MTLCommandBuffer!, e: MTLComputeCommandEncoder!, phase = ""
  var profiling = false, samples: MTLCounterSampleBuffer? = nil, labels: [String] = [], tick = 1.0, lastGPU = 0.0
  var checking = false, probes: [String] = []
  var last: (B: Int, K: Int, R: Int, seed: UInt32, lanes: MTLBuffer, tokens: MTLBuffer)? = nil   // the last learning call's
  // the forward's build-up stage (fwd.metal's STAGE; stages.swift's and the checks' view of a partial network): 0 whole
  var forwardStage: UInt32 = UInt32(ProcessInfo.processInfo.environment["ROUTE_STAGE"] ?? "0") ?? 0

  init(_ device: MTLDevice, _ queue: MTLCommandQueue, library path: String, runs: Int, refreshes: Int, contexts: Int, initial: [Float]) {
    self.device = device; self.queue = queue; lay = NetLayout(); precondition(initial.count == lay.count)
    library = try! device.makeLibrary(URL: URL(fileURLWithPath: path))
    self.runs = runs; self.refreshes = refreshes; ctxCap = contexts; klpCap = pad64(refreshes * L)
    RT = runs * klpCap + pad64(contexts)
    func sh(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: max(16, bytes), options: .storageModeShared)! }
    func pv(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: max(16, bytes), options: .storageModePrivate)! }
    let P = lay.count
    theta = device.makeBuffer(bytes: initial, length: P * 4, options: .storageModeShared)!
    m = pv(P * 4); v = pv(P * 4); grads = sh(3 * P * 4); gram = sh(9 * 4)
    // the weights' layout: a matrix's forward codes [pad64(rows)][pad64(cols)], dgrad codes [pad64(rows)][pad64(cols)]
    var q = 0, f = 0, wf = 0, offs: [UInt32] = [], ri: [UInt32] = [], ci: [UInt32] = [], pr: [UInt32] = []
    for (i, mt) in lay.mats.enumerated() {
      let kb = pad64(mt.rows), kf = pad64(mt.cols)
      mats.append([UInt32(q), UInt32(f), UInt32(f + kb), UInt32(q + kb * kf), UInt32(f + 2 * kb), UInt32(mt.rows), UInt32(mt.cols), UInt32(kf), UInt32(kb), UInt32(kf), UInt32(mt.v), UInt32(mt.g)])
      q += 2 * kb * kf; f += 2 * kb + kf; offs.append(UInt32(wf)); wf += kb * kf
      for r in 0..<kb { ri += [UInt32(i), UInt32(r)] }
      for c in 0..<kf { ci += [UInt32(i), UInt32(c)] }
      for r in 0..<mt.rows { pr += [UInt32(i), UInt32(r)] }
    }
    wq = pv(q); wsc = pv(f * 4); Wf = pv(wf * 4)
    let nt: [UInt32] = Array(mats.joined())
    netTable = device.makeBuffer(bytes: nt, length: nt.count * 4, options: .storageModeShared)!
    wfOff = device.makeBuffer(bytes: offs, length: offs.count * 4, options: .storageModeShared)!
    rowItems = device.makeBuffer(bytes: ri, length: ri.count * 4, options: .storageModeShared)!; rowCount = ri.count / 2
    colItems = device.makeBuffer(bytes: ci, length: ci.count * 4, options: .storageModeShared)!; colCount = ci.count / 2
    pinItems = device.makeBuffer(bytes: pr, length: pr.count * 4, options: .storageModeShared)!; pinCount = pr.count / 2
    let laneCap = runs * klpCap
    st = pv(runs * refreshes * 2 * D * 4); nextScratch = pv(runs * 2 * D * 4)
    lzc = pv(laneCap * 2 * 2 * D * 2); lh = pv(laneCap * 2 * D * 4); lu = pv(laneCap * 6 * 4)
    czc = pv(contexts * 2 * 2 * D * 2); cu = pv(contexts * 12 * 4); cn = pv(contexts * D * 4); cinv = pv(contexts * 4)
    // the wgrad operands: a matrix's dY codes [pad64(3 rows)][RT], X codes [pad64(cols)][RT], their scales and dY's
    // column sums [RT / 64][..]; the products' partial sums [z][3 rows][pad64(cols)]
    var gb = 0, gf = 0, pf = 0
    let blocks = RT / 64
    for mt in lay.mats {
      let M = 3 * mt.rows, kf = pad64(mt.cols)
      matg.append([UInt32(gb), UInt32(gb + pad64(M) * RT), UInt32(gf), UInt32(gf + blocks * M), UInt32(gf + blocks * (M + kf)), 0, 0, 0])
      gb += (pad64(M) + kf) * RT; gf += blocks * (2 * M + kf)
      partialOff.append(pf); pf += Net.zMax(blocks) * M * kf
    }
    gq = pv(gb); gs = pv(gf * 4); partial = pv(pf * 4)
    let table: [UInt32] = Array(matg.joined()) + [UInt32(RT), 0, 0, 0]
    gradTable = device.makeBuffer(bytes: table, length: table.count * 4, options: .storageModeShared)!
    dhp = pv(3 * 2 * pad64(contexts) * D * 4)
    let c0 = queue.makeCommandBuffer()!, bl = c0.makeBlitCommandEncoder()!
    bl.fill(buffer: m, range: 0..<(P * 4), value: 0); bl.fill(buffer: v, range: 0..<(P * 4), value: 0); bl.endEncoding()
    c0.commit(); c0.waitUntilCompleted()
    refresh()
  }
  let pinItems: MTLBuffer, pinCount: Int
  // the wgrad products' split of a matrix's blocks: `per` blocks a threadgroup (64: measured 36-40 TOP/s from 64 to
  // 512 threadgroups a 64-row tile column; the partial sums then 1/64 of the operands' bytes)
  static let perBlocks = 64
  static func zMax(_ blocks: Int) -> Int { (blocks + perBlocks - 1) / perBlocks }

  // ---- encoding ----
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
  func finish(_ what: String) {
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { fatalError("\(what): \(err)") }
    lastGPU = (cb.gpuEndTime - cb.gpuStartTime) * 1e9
  }
  // the last call's GPU time by label, in order of first appearance (scaled to the buffer's GPU time)
  func profile() -> [(String, Int, Double)] {
    guard let sb = samples, profiling, !labels.isEmpty else { return [] }
    let n = labels.count + 1
    guard let data = try? sb.resolveCounterRange(0..<(2 * min(n, 2048))) else { return [] }
    let ts = data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
    var out: [(String, Int, Double)] = [], at: [String: Int] = [:]
    for i in 1..<min(n, ts.count / 2) {
      let t0 = ts[2 * i], t1 = ts[2 * i + 1]
      if t0 == 0 || t1 < t0 || t0 == UInt64.max { continue }
      let name = labels[i - 1], dt = Double(t1 - t0) * tick
      if let k = at[name] { out[k].1 += 1; out[k].2 += dt } else { at[name] = out.count; out.append((name, 1, dt)) }
    }
    let sum = out.reduce(0) { $0 + $1.2 }
    return out.map { ($0.0, $0.1, $0.2 / max(sum, 1e-30) * lastGPU) }
  }
  func enableProfiling() {
    let d = MTLCounterSampleBufferDescriptor()
    d.counterSet = device.counterSets!.first { $0.name == MTLCommonCounterSet.timestamp.rawValue }
    d.storageMode = .shared; d.sampleCount = 4096
    samples = try! device.makeCounterSampleBuffer(descriptor: d); profiling = true
    let (c0, g0) = device.sampleTimestamps(); usleep(200000); let (c1, g1) = device.sampleTimestamps()
    var tb = mach_timebase_info_data_t(); mach_timebase_info(&tb)
    tick = Double(c1 - c0) * Double(tb.numer) / Double(tb.denom) / Double(g1 - g0)
  }
  // a kernel's buffers (nil: unbound) and constant parameters (each at its index)
  func set(_ p: MTLComputePipelineState, _ label: String, _ bufs: [Int: (MTLBuffer?, Int)], _ bytes: [Int: [UInt32]] = [:]) {
    mark(label); e.setComputePipelineState(p)
    for (i, (b, off)) in bufs { e.setBuffer(b, offset: off, index: i) }
    for (i, a) in bytes { a.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: i) } }
  }
  func groups(_ x: Int, _ y: Int = 1, _ z: Int = 1, threads: Int) {
    if x > 0 && y > 0 && z > 0 { e.dispatchThreadgroups(MTLSize(width: x, height: y, depth: z), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)) }
  }
  func nextSeed() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed | 1 }
  static let tileThreads = 256   // tile.h's NSG SIMD groups

  // the weights' codes from theta: a SIMD group a row (W, forward codes, bias), then a column (dgrad codes)
  func encodeWeights() {
    var n = UInt32(rowCount)
    set(pipe("weights_rows"), "weights_rows", [0: (theta, 0), 1: (wq, 0), 2: (wsc, 0), 3: (Wf, 0), 4: (netTable, 0), 5: (rowItems, 0), 7: (wfOff, 0)])
    e.setBytes(&n, length: 4, index: 6); groups((rowCount + 7) / 8, threads: 256)
    n = UInt32(colCount)
    set(pipe("weights_cols"), "weights_cols", [1: (wq, 0), 2: (wsc, 0), 3: (Wf, 0), 4: (netTable, 0), 5: (colItems, 0), 7: (wfOff, 0)])
    e.setBytes(&n, length: 4, index: 6); groups((colCount + 7) / 8, threads: 256)
  }
  func refresh() { begin(); phase = "weights"; encodeWeights(); finish("weights") }

  // the lanes' layout: a run's rows K L padded to the tile, or for one refresh 32 (a tile two runs: half a tile padded
  // otherwise, the forward's lanes measured 2x slower); its threadgroups and their tiles
  static func klp(_ K: Int) -> Int { K == 1 ? L : pad64(K * L) }
  static func laneGroups(B: Int, K: Int) -> Int { K == 1 ? (B + 1) / 2 : B }
  func fp(B: Int, K: Int, R: Int, seed s: UInt32 = 0) -> [UInt32] {
    let k = Net.klp(K)
    return [UInt32(B), UInt32(K), UInt32(k), UInt32(R), UInt32(pad64(B * k)), s, UInt32(pad64(ctxCap)), UInt32(K == 1 ? 1 : k / 64)]
  }
  // the forward: B runs' K refreshes (lanes [B K][L][IN], mask, h0 [B][2][D]), R contexts (tokens [R][IN], pos):
  // w [R][FS], next [B][2][D]; store: what the backward reads, the wgrad codes of every X
  func encodeForward(lanes: MTLBuffer, mask: MTLBuffer, h0: MTLBuffer, tokens: MTLBuffer, pos: MTLBuffer, B: Int, K: Int, R: Int, w: MTLBuffer, next: MTLBuffer, store: Bool, stage: UInt32 = 0) {
    precondition(B <= runs && K <= refreshes && R <= ctxCap, "a batch past the network's buffers")
    let prm = fp(B: B, K: K, R: R)
    set(pipe("lanes_fwd", store: store, stage: stage), "lanes_fwd", [0: (lanes, 0), 1: (mask, 0), 2: (h0, 0), 3: (st, 0), 4: (next, 0), 5: (netTable, 0), 6: (wq, 0), 7: (wsc, 0),
        9: (gq, 0), 10: (gs, 0), 11: (gradTable, 0), 12: (lzc, 0), 13: (lh, 0), 14: (lu, 0)], [8: prm])
    groups(Net.laneGroups(B: B, K: K), threads: Net.tileThreads)
    set(pipe("ctx_fwd", store: store, stage: stage), "ctx_fwd", [0: (tokens, 0), 1: (pos, 0), 2: (st, 0), 3: (w, 0), 5: (netTable, 0), 6: (wq, 0), 7: (wsc, 0),
        9: (gq, 0), 10: (gs, 0), 11: (gradTable, 0), 12: (czc, 0), 13: (cu, 0), 14: (cn, 0), 15: (cinv, 0)], [8: prm])
    groups((R + 63) / 64, threads: Net.tileThreads)
  }
  func forwardEncode(lanes: MTLBuffer, mask: MTLBuffer, h0: MTLBuffer, tokens: MTLBuffer, pos: MTLBuffer, B: Int, R: Int, w: MTLBuffer, next: MTLBuffer) -> MTLCommandBuffer {
    begin(); phase = "forward"
    encodeForward(lanes: lanes, mask: mask, h0: h0, tokens: tokens, pos: pos, B: B, K: 1, R: R, w: w, next: next, store: false, stage: forwardStage)
    e.endEncoding()
    return cb
  }

  // a learning batch's credit inputs (route.h's route_draw records, each row's draws first[row] .. first[row + 1], its
  // behaviour w, its task-0 scale, the tasks' common scale)
  struct Credit { let draws: MTLBuffer, count: Int, first: MTLBuffer, wrows: MTLBuffer, scale0: MTLBuffer, scale: Float, ll: MTLBuffer }
  func room(_ b: inout MTLBuffer?, _ bytes: Int, shared: Bool) -> MTLBuffer {
    if b == nil || b!.length < bytes { b = device.makeBuffer(length: max(16, bytes), options: shared ? .storageModeShared : .storageModePrivate)! }
    return b!
  }
  // the learning step's gradients into grads [3][P]: the forward storing, the credit, the backward (contexts, then the
  // lanes: their states' gradients from the contexts' summed), the wgrad products, the chain rule
  func encodeLearn(lanes: MTLBuffer, mask: MTLBuffer, h0: MTLBuffer, tokens: MTLBuffer, pos: MTLBuffer, seg: MTLBuffer, B: Int, K: Int, R: Int, credit c: Credit, w: MTLBuffer) {
    let dw = room(&dwBuf, 3 * R * FS * 4, shared: true)
    phase = "learn forward"
    encodeForward(lanes: lanes, mask: mask, h0: h0, tokens: tokens, pos: pos, B: B, K: K, R: R, w: w, next: nextScratch, store: true)
    phase = "credit"   // a draw a SIMD group (credit.metal), then a row's draws summed in order
    let diff = room(&diffBuf, max(1, c.count) * FS * 4, shared: false)
    if c.count > 0 {
      set(pipe("credit_draws"), "credit_draws", [0: (c.draws, 0), 1: (c.wrows, 0), 2: (w, 0), 3: (diff, 0), 4: (c.ll, 0)], [5: [UInt32(c.count)]])
      e.dispatchThreads(MTLSize(width: c.count * 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    set(pipe("credit_rows"), "credit_rows", [0: (diff, 0), 1: (c.draws, 0), 2: (c.first, 0), 3: (c.scale0, 0), 4: (dw, 0)], [5: [UInt32(R), 0, c.scale.bitPattern, 0]])
    e.dispatchThreads(MTLSize(width: 3 * R * FS, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    phase = "backward"
    let prm = fp(B: B, K: K, R: R, seed: nextSeed())
    last = (B, K, R, prm[5], lanes, tokens)
    set(pipe("ctx_bwd"), "ctx_bwd", [0: (dw, 0), 1: (pos, 0), 2: (st, 0), 3: (czc, 0), 4: (cu, 0), 5: (netTable, 0), 6: (wq, 0), 7: (wsc, 0), 9: (gq, 0), 10: (gs, 0),
        11: (gradTable, 0), 12: (cn, 0), 13: (cinv, 0), 14: (dhp, 0)], [8: prm])
    groups((R + 63) / 64, 3, threads: Net.tileThreads)
    set(pipe("lanes_bwd"), "lanes_bwd", [0: (mask, 0), 1: (h0, 0), 2: (lzc, 0), 3: (lh, 0), 4: (lu, 0), 5: (netTable, 0), 6: (wq, 0), 7: (wsc, 0), 9: (gq, 0), 10: (gs, 0),
        11: (gradTable, 0), 12: (seg, 0), 13: (dhp, 0), 15: (pos, 0)], [8: prm])
    groups(Net.laneGroups(B: B, K: K), 3, threads: Net.tileThreads)
    // the wgrad products: a matrix's [3 rows] x [cols] over its blocks (lanes' and contexts', or the contexts' alone)
    phase = "wgrad"
    let RL = pad64(B * Net.klp(K)), lb = RL / 64, cbk = (R + 63) / 64
    let wgp = pipe("gemm_wg_64x64_4")
    var gitems: [UInt32] = []
    for (i, mt) in lay.mats.enumerated() {
      let both = i <= 4, b0 = both ? 0 : lb, b1 = lb + cbk, M = 3 * mt.rows, kf = pad64(mt.cols), nz = (b1 - b0 + Net.perBlocks - 1) / Net.perBlocks
      let g = matg[i]
      set(wgp, "gemm_wg", [0: (gq, Int(g[0])), 1: (gq, Int(g[1])), 2: (gs, Int(g[2]) * 4), 3: (gs, Int(g[3]) * 4), 4: (partial, partialOff[i] * 4)],
          [5: [UInt32(M), UInt32(kf), UInt32(RT), UInt32(Net.perBlocks), UInt32(b0), UInt32(b1), 0, 0]])
      groups(kf / 64, (M + 63) / 64, nz, threads: 128)
      for t in 0..<3 { for r in 0..<mt.rows { gitems += [UInt32(i), UInt32(t), UInt32(r), UInt32(partialOff[i]), UInt32(nz), g[4], UInt32(b0), UInt32(b1)] } }
    }
    phase = "grads"
    let gi = device.makeBuffer(bytes: gitems, length: gitems.count * 4, options: .storageModeShared)!
    set(pipe("grads"), "grads", [0: (theta, 0), 1: (partial, 0), 2: (gs, 0), 3: (grads, 0), 4: (netTable, 0), 5: (gi, 0)], [6: [UInt32(gitems.count / 8), UInt32(lay.count)]])
    groups((gitems.count / 8 + 7) / 8, threads: 256)
  }
  // the optimizer's step after the learning call (and the exchange): one threadgroup (opt.metal), then the codes
  func encodeApply(rate: Double, inv: Float) {
    step += 1
    phase = "optimizer"
    let c1 = Float(1 - pow(0.9, Double(step))), c2 = Float(1 - pow(0.999, Double(step)))
    let op: [UInt32] = [UInt32(lay.count), UInt32(pinCount), Float(rate).bitPattern, inv.bitPattern, c1.bitPattern, c2.bitPattern, 0, 0]
    set(pipe("optimizer"), "optimizer", [0: (theta, 0), 1: (m, 0), 2: (v, 0), 3: (grads, 0), 4: (netTable, 0), 5: (pinItems, 0), 7: (gram, 0)], [6: op])
    groups(1, threads: 1024)
    encodeWeights()
  }
  // the weights as stored for deployment (int8mm.h's format): the forward codes and scales as on the GPU, each bias
  func writeCodes(_ path: String) {
    var bytes = Data()
    func u32(_ x: UInt32) { withUnsafeBytes(of: x.littleEndian) { bytes.append(contentsOf: $0) } }
    u32(0x50384952); u32(1); u32(UInt32(lay.mats.count))
    let c0 = queue.makeCommandBuffer()!, bl = c0.makeBlitCommandEncoder()!
    var staged: [(MTLBuffer, MTLBuffer)] = []
    for (i, mt) in lay.mats.enumerated() {
      let kf = pad64(mt.cols), q = device.makeBuffer(length: mt.rows * kf, options: .storageModeShared)!, sc = device.makeBuffer(length: mt.rows * 4, options: .storageModeShared)!
      bl.copy(from: wq, sourceOffset: Int(mats[i][0]), to: q, destinationOffset: 0, size: q.length); bl.copy(from: wsc, sourceOffset: Int(mats[i][1]) * 4, to: sc, destinationOffset: 0, size: sc.length)
      staged.append((q, sc))
    }
    bl.endEncoding(); c0.commit(); c0.waitUntilCompleted()
    let th = theta.contents().bindMemory(to: Float.self, capacity: lay.count)
    for (i, mt) in lay.mats.enumerated() {
      u32(UInt32(mt.rows)); u32(UInt32(mt.cols)); u32(UInt32(pad64(mt.cols))); u32(mt.bias ? 1 : 0)
      bytes.append(Data(bytes: staged[i].0.contents(), count: staged[i].0.length)); bytes.append(Data(bytes: staged[i].1.contents(), count: staged[i].1.length))
      if mt.bias { bytes.append(Data(bytes: th + mt.g + mt.rows, count: mt.rows * 4)) }
    }
    try! bytes.write(to: URL(fileURLWithPath: path))
  }
  // the probes (route_net_probes, after a learning call): tiles' fused wgrad codes against int8mm.h's CPU definition of
  // the same column codes (i8_fwht64 along the rows, a scale a column, i8_code; the gradients' with their seeds): the
  // tokens' (inp's X) of the first lane tile and the first context tile, dw's (out's dY, stochastic) of the first
  // context tile, each task's
  func probeReport() -> [String] {
    guard let c = last else { return [] }
    func read(_ b: MTLBuffer, _ off: Int, _ n: Int) -> UnsafeMutableRawPointer {
      let st = device.makeBuffer(length: max(16, n), options: .storageModeShared)!, cb = queue.makeCommandBuffer()!, bl = cb.makeBlitCommandEncoder()!
      bl.copy(from: b, sourceOffset: off, to: st, destinationOffset: 0, size: n); bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      let out = UnsafeMutableRawPointer.allocate(byteCount: max(16, n), alignment: 16); memcpy(out, st.contents(), n); return out
    }
    let RL = pad64(c.B * Net.klp(c.K)), blocks = RT / 64
    // one tile's columns (x [64 rows][cols] at stride ld) against the GPU's codes (gq at qoff, stride RT) and scales
    func compare(_ label: String, _ x: [Float], cols: Int, ld: Int, qoff: Int, soff: Int, row0: Int, seed: UInt32) -> String {
      let q = read(gq, qoff, (cols - 1) * RT + row0 + 64).bindMemory(to: Int8.self, capacity: cols * RT), sc = read(gs, soff * 4, cols * 4).bindMemory(to: Float.self, capacity: cols)
      var bad = 0, big = 0, badS = 0
      for col in 0..<cols {
        var v = (0..<64).map { x[$0 * ld + col] }
        v.withUnsafeMutableBufferPointer { i8_fwht64($0.baseAddress!) }
        let m = v.reduce(0) { max($0, abs($1)) }, inv = i8_inverse(m)
        if i8_scale(m).bitPattern != sc[col].bitPattern { badS += 1 }
        for r in 0..<64 {
          let want = i8_code(v[r], inv, seed, UInt32(truncatingIfNeeded: col * RT + row0 + r)), got = q[col * RT + row0 + r]
          if want != got { bad += 1; if abs(Int(want) - Int(got)) > 1 { big += 1 } }
        }
      }
      return String(format: "  %-44@ codes differing %d of %d (by more than 1: %d), scales %d of %d", label, bad, 64 * cols, big, badS, cols)
    }
    var out: [String] = []
    let IN = Int(ROUTE_IN), inp = matg[0]
    let lanesP = read(c.lanes, 0, min(c.lanes.length, 64 * IN * 4)).bindMemory(to: Float.self, capacity: 64 * IN)
    var xl = [Float](repeating: 0, count: 64 * 64)
    for r in 0..<64 where r < c.K * L { for j in 0..<IN { xl[r * 64 + j] = lanesP[r * IN + j] } }
    out.append(compare("lanes_fwd tokens -> inp's X codes", xl, cols: 64, ld: 64, qoff: Int(inp[1]), soff: Int(inp[3]), row0: 0, seed: 0))
    let tokP = read(c.tokens, 0, min(c.tokens.length, 64 * IN * 4)).bindMemory(to: Float.self, capacity: 64 * IN)
    var xc = [Float](repeating: 0, count: 64 * 64)
    for r in 0..<min(64, c.R) { for j in 0..<IN { xc[r * 64 + j] = tokP[r * IN + j] } }
    out.append(compare("ctx_fwd tokens -> inp's X codes", xc, cols: 64, ld: 64, qoff: Int(inp[1]), soff: Int(inp[3]) + (RL / 64) * 64, row0: RL, seed: 0))
    let og = matg[7], dwP = dwBuf!.contents().bindMemory(to: Float.self, capacity: 3 * c.R * FS)
    for t in 0..<3 {
      var y = [Float](repeating: 0, count: 64 * FS)
      for r in 0..<min(64, c.R) { for f in 0..<FS { y[r * FS + f] = dwP[(t * c.R + r) * FS + f] } }
      let seed = c.seed == 0 ? 0 : (i8_mix(c.seed ^ (65 &* 0x9e3779b1) ^ (UInt32(t) &* 0x85ebca6b)) | 1)   // bwd.metal's seed_of(S_OUT + 64, task)
      out.append(compare("ctx_bwd dw (task \(t)) -> out's dY codes", y, cols: FS, ld: FS, qoff: Int(og[0]) + t * FS * RT, soff: Int(og[2]) + (RL / 64) * 3 * FS + t * FS, row0: RL, seed: seed))
    }
    _ = blocks
    return out
  }
  // a buffer by its name (route_net_read)
  func named(_ n: String) -> MTLBuffer? {
    let all: [String: MTLBuffer?] = ["theta": theta, "grads": grads, "gram": gram, "wq": wq, "wsc": wsc, "st": st, "lzc": lzc, "lh": lh, "lu": lu, "czc": czc, "cu": cu,
                                    "cn": cn, "cinv": cinv, "gq": gq, "gs": gs, "dhp": dhp, "partial": partial, "dw": dwBuf, "net": netTable, "grad": gradTable]
    return all[n] ?? nil
  }
}
