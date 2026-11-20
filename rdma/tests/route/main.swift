// The routing policy (route.metal) trained as in Kool, van Hoof & Welling 2019: REINFORCE on each rollout's completion
// over the floor, each rollout's baseline the mean of its scenario's other three rollouts; two tasks (the completion,
// and its spread about the scenario's mean), their gradients combined at their min-norm point, then Adam. Measured on
// the five-node cluster as measured, at decode sizes, against all traffic on the unloaded shortest-path tree and
// against lanes chosen uniformly. The metallib's arm (GRU or minGRU) and layout come from its `layout` kernel.
//
// usage: route METALLIB FABRIC [steps] [scenarios a step]      CHECK=1: the gradient against central differences
import Foundation
import Metal
import MetalPerformanceShaders

let N = 6, L = 32, J = 6, KS = 3, EP = 4, TMAX = N - 1, H = 12, D = 64, E = 192, CF = 8, IN = CF + D, LF = 4
let TR = EP * KS, JTN = J * TR * N, DEC = EP * KS * TMAX * J, KREC = 196, TAPE = IN + 2 * D + 4
let page = 16384.0, scenarioBytes = 548

struct Mix { var s: UInt64
  mutating func next() -> UInt64 { s &+= 0x9e3779b97f4a7c15; var z = s; z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9; z = (z ^ (z >> 27)) &* 0x94d049bb133111eb; return z ^ (z >> 31) }
  mutating func uniform() -> Double { Double(next() >> 11) / 9007199254740992.0 }
  mutating func normal() -> Double { sqrt(-2 * log(max(uniform(), 1e-300))) * cos(2 * Double.pi * uniform()) }
}
struct Fabric { var n: Int; var compute: [Int]; var src: [Int]; var dst: [Int]; var rate: [Double]; var lat: [Double] }

// a scenario, as route.metal's Scenario: an all-gather of S a rank among the compute nodes
func scenario(_ f: Fabric, _ S: Double, _ g: inout Mix, background: Bool) -> [UInt8] {
  let n = f.n, nl = f.src.count, q = min(S, page), nc = f.compute.count
  let bg = (0..<nl).map { background && g.uniform() < 0.3 ? f.rate[$0] * 50e3 * g.uniform() : 0 }
  var cut = 0.0
  for v in f.compute {
    let inr = (0..<nl).filter { f.dst[$0] == v }.map { f.rate[$0] }.reduce(0, +), outr = (0..<nl).filter { f.src[$0] == v }.map { f.rate[$0] }.reduce(0, +)
    cut = max(cut, Double(nc - 1) * S / inr, S / outr)
  }
  var dist = Array(repeating: Array(repeating: Double.infinity, count: n), count: n)
  for v in 0..<n { dist[v][v] = 0 }
  for i in 0..<nl { dist[f.src[i]][f.dst[i]] = min(dist[f.src[i]][f.dst[i]], f.lat[i] + q / f.rate[i]) }
  for m in 0..<n { for a in 0..<n { for b in 0..<n { dist[a][b] = min(dist[a][b], dist[a][m] + dist[m][b]) } } }
  var reach = 0.0
  for a in f.compute { for b in f.compute where a != b { reach = max(reach, dist[a][b]) } }
  let floor = max(cut, reach)
  let slowest = (0..<nl).map { f.lat[$0] + page / f.rate[$0] }.max()!
  let dt = max(floor * 6 / 64, slowest / Double(H - 2))
  var b = [UInt8](repeating: 0, count: scenarioBytes)
  func f32(_ at: Int, _ v: Double) { var x = Float(v); withUnsafeBytes(of: &x) { for i in 0..<4 { b[at + i] = $0[i] } } }
  func u32(_ at: Int, _ v: Int) { var x = UInt32(v); withUnsafeBytes(of: &x) { for i in 0..<4 { b[at + i] = $0[i] } } }
  for l in 0..<nl { f32(4 * l, f.rate[l]); f32(128 + 4 * l, f.lat[l]); f32(256 + 4 * l, bg[l]); b[384 + l] = UInt8(f.src[l]); b[416 + l] = UInt8(f.dst[l]) }
  for i in 0..<(J * N) { b[448 + i] = 255 }
  let w = (0..<nl).map { f.lat[$0] + q / f.rate[$0] }
  for (j, s) in f.compute.enumerated() {   // the unloaded shortest-path tree, pruned to what leads to a destination
    var at = Array(repeating: Double.infinity, count: n), by = Array(repeating: -1, count: n), done = Array(repeating: false, count: n)
    at[s] = 0
    for _ in 0..<n {
      var u = -1
      for v in 0..<n where !done[v] && at[v].isFinite && (u < 0 || at[v] < at[u]) { u = v }
      if u < 0 { break }
      done[u] = true
      for i in 0..<nl where f.src[i] == u && at[u] + w[i] < at[f.dst[i]] { at[f.dst[i]] = at[u] + w[i]; by[f.dst[i]] = i }
    }
    var keep = Array(repeating: false, count: n)
    for d in f.compute where d != s { var v = d; while v != s { keep[v] = true; v = f.src[by[v]] } }
    for v in 0..<n where keep[v] { b[448 + j * N + v] = UInt8(by[v]) }
    b[484 + j] = UInt8(s); var mask = 0; for d in f.compute where d != s { mask |= 1 << d }; b[490 + j] = UInt8(mask)
    f32(496 + 4 * j, S)
  }
  f32(520, floor); f32(524, dt); f32(528, S); u32(532, nc); u32(536, nl); u32(540, n)
  return b
}
func cabled(_ n: Int, _ compute: [Int], _ pairs: [(Int, Int)], _ g: inout Mix) -> Fabric {
  var f = Fabric(n: n, compute: compute, src: [], dst: [], rate: [], lat: [])
  for (u, v) in pairs {
    let c = Double(1 + Int(g.uniform() * 4))
    for (a, b) in [(u, v), (v, u)] { f.src.append(a); f.dst.append(b); f.rate.append(c * (4 + 8 * g.uniform())); f.lat.append(2e3 + 11e3 * g.uniform()) }
  }
  return f
}
func network(_ g: inout Mix, _ measured: Fabric) -> Fabric {   // one network of a random kind
  let kind = Int(g.uniform() * 6), n = 3 + Int(g.uniform() * 4)
  var pairs: [(Int, Int)] = []
  switch kind {
  case 0:   // a random connected graph, degree at most 4, up to two relays
    var joined = Set<Int>(), degree = Array(repeating: 0, count: n)
    for v in 1..<n { let u = Int(g.uniform() * Double(v)); pairs.append((u, v)); joined.insert(u * 8 + v); degree[u] += 1; degree[v] += 1 }
    for u in 0..<n { for v in (u + 1)..<n where !joined.contains(u * 8 + v) && degree[u] < 4 && degree[v] < 4 && g.uniform() < 0.35 { pairs.append((u, v)); degree[u] += 1; degree[v] += 1 } }
    return cabled(n, Array(0..<(n - min(Int(g.uniform() * 3), n - 2))), pairs, &g)
  case 1: for v in 0..<n { pairs.append((v, (v + 1) % n)) }; if n == 3 { pairs.removeLast() }; return cabled(n, Array(0..<n), pairs, &g)
  case 2: for v in 1..<n { pairs.append((0, v)) }; return cabled(n, g.uniform() < 0.5 ? Array(1..<n) : Array(0..<n), pairs, &g)
  case 3:   // a ring of compute nodes and a relay hub
    let m = min(max(n - 1, 3), N - 1)
    for v in 0..<m { pairs.append((v, (v + 1) % m)) }
    for v in 0..<min(3, m) { pairs.append((m, v)) }
    return cabled(m + 1, Array(0..<m), pairs, &g)
  case 4: for u in 0..<n { for v in (u + 1)..<n { pairs.append((u, v)) } }; return cabled(n, Array(0..<n), pairs, &g)
  default:   // the measured cluster, jittered
    var f = measured
    for i in 0..<f.rate.count { f.rate[i] *= exp(0.3 * g.normal()); f.lat[i] *= exp(0.3 * g.normal()) }
    return f
  }
}
func load(_ path: String) -> Fabric {
  var names: [String] = [], compute: [Int] = [], f = Fabric(n: 0, compute: [], src: [], dst: [], rate: [], lat: [])
  for line in try! String(contentsOfFile: path, encoding: .utf8).split(separator: "\n") where !line.hasPrefix("#") {
    let w = line.split(separator: " ").map(String.init)
    if w.first == "node" { names.append(w[1]); if w[2] == "compute" { compute.append(names.count - 1) } }
    if w.first == "link" { f.src.append(names.firstIndex(of: w[1])!); f.dst.append(names.firstIndex(of: w[2])!); f.rate.append(Double(w[3])!); f.lat.append(Double(w[4])! * 1e3) }
  }
  f.n = names.count; f.compute = compute
  return f
}

let args = CommandLine.arguments
let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
func pipe(_ name: String) -> MTLComputePipelineState { try! device.makeComputePipelineState(function: library.makeFunction(name: name)!) }
let rollPipe = pipe("rollout"), backPipe = pipe("backward"), reducePipe = pipe("reduce"), layoutPipe = pipe("layout")
let queue = device.makeCommandQueue()!

// the layout, from the kernel
let lay: [Int] = {
  let b = device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
  let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(layoutPipe); enc.setBuffer(b, offset: 0, index: 0)
  enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  let p = b.contents().bindMemory(to: Int32.self, capacity: 32)
  return (0..<19).map { Int(p[$0]) }
}()
let full = lay[0] == 1, P = lay[1], REC = lay[12], SCRATCH = lay[13], LS = lay[14], RLS = lay[11]
let L_IN = lay[2], L_DR = lay[3], L_RX = lay[4], L_DZ = lay[5], L_DHT = lay[6], L_XM = lay[7], L_DU = lay[8], L_U = lay[9], L_DY = lay[10]
let R_DX0 = 0, R_IN = 64, R_LAYER = 136, R_DQ = lay[15], R_XF = lay[16], R_DST = lay[17]
let GW = full ? 2 * D : D, G = D * GW

// parameters (as route.metal's Net) and the gradient jobs: (records: 0 decisions, 1 keys; delta offset; input offset or
// -1 for ones; rows; cols; parameter offset)
let O_LAYER = D * IN + 2 * D, O_OUT = O_LAYER + 2 * LS, O_K1 = O_OUT + D * D, O_BK1 = O_K1 + D * LF, O_K2 = O_BK1 + D, O_BK2 = O_K2 + D * D
precondition(O_BK2 + D == P, "layout")
var jobs: [(Int, Int, Int, Int, Int, Int)] = [(0, R_DX0, R_IN, D, IN, 0), (0, R_DX0, -1, D, 1, D * IN), (0, R_DST, -1, D, 1, D * IN + D)]
var blocks: [(String, Int, Int)] = [("inp", 0, D * IN), ("binp", D * IN, D), ("start", D * IN + D, D)]
for layer in 0..<2 {
  let B = R_LAYER + layer * RLS, P0 = O_LAYER + layer * LS
  let oz = P0, obz = oz + G, or = obz + D, obr = or + G, oh = full ? obr + D : obz + D, obh = oh + G, oup = obh + D, obu = oup + E * D, odn = obu + E, obd = odn + D * E
  precondition(obd + D == P0 + LS, "layer layout")
  jobs += [(0, B + L_DZ, B + L_IN, D, GW, oz), (0, B + L_DZ, -1, D, 1, obz),
           (0, B + L_DHT, B + (full ? L_RX : L_IN), D, GW, oh), (0, B + L_DHT, -1, D, 1, obh),
           (0, B + L_DU, B + L_XM, E, D, oup), (0, B + L_DU, -1, E, 1, obu), (0, B + L_DY, B + L_U, D, E, odn), (0, B + L_DY, -1, D, 1, obd)]
  if full { jobs += [(0, B + L_DR, B + L_IN, D, GW, or), (0, B + L_DR, -1, D, 1, obr)]; blocks += [("r\(layer)", or, G)] }
  blocks += [("z\(layer)", oz, G), ("h\(layer)", oh, G), ("bh\(layer)", obh, D), ("up\(layer)", oup, E * D), ("down\(layer)", odn, D * E), ("bd\(layer)", obd, D)]
}
jobs += [(0, R_DQ, R_XF, D, D, O_OUT), (1, 0, 64, D, D, O_K2), (1, 0, -1, D, 1, O_BK2), (1, 128, 192, D, LF, O_K1), (1, 128, -1, D, 1, O_BK1)]
blocks += [("out", O_OUT, D * D), ("k1", O_K1, D * LF), ("bk1", O_BK1, D), ("k2", O_K2, D * D), ("bk2", O_BK2, D)]
let covered = jobs.map { $0.3 * $0.4 }.reduce(0, +)
precondition(covered == P, "gradient jobs cover \(covered) of \(P) parameters")

let measured = load(args[2])
let steps = args.count > 3 ? Int(args[3])! : 200, B = args.count > 4 ? Int(args[4])! : 64
let rate = Double(ProcessInfo.processInfo.environment["RATE"] ?? "1e-3")!
let R = max(4 * B, 6 * 32)
var g = Mix(s: UInt64(ProcessInfo.processInfo.environment["SEED"] ?? "20261119")!)
let pool = 8192
var scen = [UInt8](); scen.reserveCapacity((pool + 8) * scenarioBytes)
for _ in 0..<pool { let f = network(&g, measured); let S = exp(log(6144.0) + g.uniform() * (log(12582912.0) - log(6144.0))).rounded(); scen += scenario(f, S, &g, background: true) }
let sizes = [6144.0, 24576, 98304, 393216, 3145728, 12582912]
for S in sizes { scen += scenario(measured, S, &g, background: false) }
var net = [Float](repeating: 0, count: P)
for (name, at, count) in blocks where !name.hasPrefix("b") {   // weights at 1/sqrt(fan-in), biases zero
  let fan = name == "start" ? 1 : name == "inp" ? IN : name.hasPrefix("down") ? E : name.hasPrefix("k1") ? LF : (name.hasPrefix("z") || name.hasPrefix("r") || name.hasPrefix("h")) ? GW : D
  for i in 0..<count { net[at + i] = Float(g.normal() / sqrt(Double(fan))) }
}

func buffer(_ bytes: Int, _ mode: MTLResourceOptions = .storageModeShared) -> MTLBuffer { device.makeBuffer(length: max(bytes, 16), options: mode)! }
let scenBuf = device.makeBuffer(bytes: scen, length: scen.count, options: .storageModeShared)!
let netBuf = device.makeBuffer(bytes: net, length: P * 4, options: .storageModeShared)!
let whichBuf = buffer(R * 4), drawBuf = buffer(R * 4), doneBuf = buffer(R * 4), llBuf = buffer(R * 4), rllBuf = buffer(R * 4)
let tapeBuf = buffer(R * DEC * TAPE * 4), phiBuf = buffer(R * EP * L * LF * 4)
let rsBuf = buffer(R * lay[18] * 4, .storageModePrivate), bsBuf = buffer(R * SCRATCH * 4, .storageModePrivate), ringBuf = buffer(R * H * JTN * 2, .storageModePrivate)
let recBuf = buffer(R * DEC * REC * 4, .storageModePrivate), krecBuf = buffer(R * EP * L * KREC * 4, .storageModePrivate)
let rec2Buf = buffer(R * DEC * REC * 4, .storageModePrivate), krec2Buf = buffer(R * EP * L * KREC * 4, .storageModePrivate)
let onesBuf = device.makeBuffer(bytes: [Float](repeating: 1, count: R * DEC), length: R * DEC * 4, options: .storageModeShared)!
var SPLIT = 64; let wBuf = buffer(R * 8), gBuf = buffer(2 * P * 4), partBuf = buffer(SPLIT * 2 * P * 4, .storageModePrivate)
let which = whichBuf.contents().bindMemory(to: UInt32.self, capacity: R), draw = drawBuf.contents().bindMemory(to: UInt32.self, capacity: R)
let done = doneBuf.contents().bindMemory(to: Float.self, capacity: R), ll = llBuf.contents().bindMemory(to: Float.self, capacity: R)
let w = wBuf.contents().bindMemory(to: Float.self, capacity: 2 * R), grad = gBuf.contents().bindMemory(to: Float.self, capacity: 2 * P)
let theta = netBuf.contents().bindMemory(to: Float.self, capacity: P)

// the GEMMs for `count` rollouts, made once: (batches, [(kernel, deltas, inputs, partial gradient)])
// each (task, job)'s partials, nb of them, contiguous at `base`, summed into the gradient at task P + its offset
var plans: [Int: (Int, [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)])] = [:]
func plan(_ count: Int) -> (Int, [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)]) {
  var nb = min(SPLIT, count); while count % nb != 0 { nb -= 1 }
  if let p = plans[count * 1000 + nb] { return p }
  var gemms: [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)] = []
  var base = 0
  for task in 0..<2 {
    for (kind, dOff, xOff, rows, cols, gOff) in jobs {
      let per = (kind == 0 ? DEC : EP * L) * (count / nb), stride = (kind == 0 ? REC : KREC) * 4
      let deltas = task == 0 ? (kind == 0 ? recBuf : krecBuf) : (kind == 0 ? rec2Buf : krec2Buf), inputs = kind == 0 ? recBuf : krecBuf
      let A = MPSMatrix(buffer: deltas, offset: dOff * 4, descriptor: MPSMatrixDescriptor(rows: per, columns: rows, matrices: nb, rowBytes: stride, matrixBytes: per * stride, dataType: .float32))
      let X = xOff < 0 ? MPSMatrix(buffer: onesBuf, offset: 0, descriptor: MPSMatrixDescriptor(rows: per, columns: 1, matrices: nb, rowBytes: 4, matrixBytes: per * 4, dataType: .float32))
                       : MPSMatrix(buffer: inputs, offset: xOff * 4, descriptor: MPSMatrixDescriptor(rows: per, columns: cols, matrices: nb, rowBytes: stride, matrixBytes: per * stride, dataType: .float32))
      let C = MPSMatrix(buffer: partBuf, offset: base * 4, descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, matrices: nb, rowBytes: cols * 4, matrixBytes: rows * cols * 4, dataType: .float32))
      let mm = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: rows, resultColumns: cols, interiorColumns: per, alpha: 1, beta: 0)
      mm.batchStart = 0; mm.batchSize = nb
      gemms.append((mm, A, X, C, base, rows * cols, task * P + gOff)); base += nb * rows * cols
    }
  }
  plans[count * 1000 + nb] = (nb, gemms)
  return (nb, gemms)
}
var gpuTime = 0.0
func run(_ count: Int, rollout mode: UInt32?, backward: Bool, gradient: Bool) {
  let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
  let tg = MTLSize(width: 256, height: 1, depth: 1), grid = MTLSize(width: count, height: 1, depth: 1)
  if var m = mode {
    enc.setComputePipelineState(rollPipe)
    for (i, b) in [scenBuf, netBuf, whichBuf, drawBuf, doneBuf, tapeBuf, phiBuf, rsBuf, ringBuf].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    enc.setBytes(&m, length: 4, index: 9); enc.setBuffer(rllBuf, offset: 0, index: 10)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
  }
  if backward {
    enc.setComputePipelineState(backPipe)
    for (i, b) in [netBuf, tapeBuf, phiBuf, recBuf, krecBuf, bsBuf, llBuf, wBuf, rec2Buf, krec2Buf].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
  }
  enc.endEncoding()
  if gradient {   // each weight block, each task: G = delta^T input over every record, a GEMM split over batches of rollouts
    let (nb, gemms) = plan(count)
    for (mm, A, X, C, _, _, _) in gemms { mm.encode(commandBuffer: cb, leftMatrix: A, rightMatrix: X, resultMatrix: C) }
    let red = cb.makeComputeCommandEncoder()!
    red.setComputePipelineState(reducePipe); red.setBuffer(partBuf, offset: 0, index: 0); red.setBuffer(gBuf, offset: 0, index: 1)
    for (_, _, _, _, base, size, at) in gemms {
      var q = SIMD4<UInt32>(UInt32(base), UInt32(size), UInt32(nb), UInt32(at)); red.setBytes(&q, length: 16, index: 2)
      red.dispatchThreads(MTLSize(width: size, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    red.endEncoding()
  }
  cb.commit(); cb.waitUntilCompleted(); gpuTime = cb.gpuEndTime - cb.gpuStartTime
  if let e = cb.error { print("error: \(e)"); exit(1) }
}
let arm = full ? "GRU" : "minGRU"

if let path = ProcessInfo.processInfo.environment["REPLAY"] {   // REPLAY=weights SCEN=n DRAW=k: one rollout's decisions
  let d = try! Data(contentsOf: URL(fileURLWithPath: path)); d.withUnsafeBytes { memcpy(theta, $0.baseAddress!, P * 4) }
  which[0] = UInt32(ProcessInfo.processInfo.environment["SCEN"]!)!; draw[0] = UInt32(ProcessInfo.processInfo.environment["DRAW"]!)!
  run(1, rollout: 0, backward: false, gradient: false)
  let ph = phiBuf.contents().bindMemory(to: Float.self, capacity: EP * L * LF)
  for ep in 0..<EP { print("decision point \(ep) lane features: " + (0..<8).map { l in "[" + (0..<LF).map { String(format: "%.3g", ph[(ep * L + l) * LF + $0]) }.joined(separator: " ") + "]" }.joined(separator: " ")) }
  let t = tapeBuf.contents().bindMemory(to: Float.self, capacity: DEC * TAPE)
  for n in 0..<DEC {
    let f = t + n * TAPE, legal = f.advanced(by: IN + 2 * D).withMemoryRebound(to: UInt32.self, capacity: 2) { ($0[0], Int32(bitPattern: $0[1])) }
    let hmax = (0..<(2 * D)).map { abs(f[IN + $0]) }.max()!, inmax = (0..<IN).map { abs(f[$0]) }.max()!
    let badIn = (0..<IN).filter { !f[$0].isFinite }, badH = (0..<(2 * D)).filter { !f[IN + $0].isFinite }.count
    if !f[TAPE - 2].isFinite || !badIn.isEmpty || badH > 0 { print("  non-finite: input entries \(badIn.prefix(8)) (of \(badIn.count)), state entries \(badH)") }
    if legal.0 != 0 || !f[TAPE - 2].isFinite { print(String(format: "decision %3d (ep %d step %d row %2d): legal %08x chose %2d, log p %9.4f sum %9.4g, |in| %.3g |h| %.3g", n, n / (TMAX * 18), n / 18 % TMAX, n % 18, legal.0, legal.1, f[TAPE - 2], f[TAPE - 1], inmax, hmax)) }
  }
  exit(0)
}
if let c = ProcessInfo.processInfo.environment["LATENCY"] {   // the rollout's time against how many run at once
  for n in c.split(separator: ",").map({ Int($0)! }) {
    for i in 0..<n { which[i] = UInt32(i % pool); draw[i] = UInt32(i + 1) }
    run(n, rollout: 0, backward: false, gradient: false)
    var ms = 0.0; for _ in 0..<5 { run(n, rollout: 0, backward: false, gradient: false); ms += gpuTime * 200 }
    print(String(format: "%@ %5d rollouts at once: %.2f ms, %.0f rollouts/s", arm, n, ms, Double(n) / ms * 1000))
  }
  exit(0)
}
if ProcessInfo.processInfo.environment["PROFILE"] != nil {   // each kernel's GPU time, one step's worth
  let B = args.count > 4 ? Int(args[4])! : 64
  for i in 0..<(4 * B) { which[i] = UInt32(i / 4); draw[i] = UInt32(i + 1) }
  for i in 0..<(8 * B) { w[i] = 1e-3 }
  for (name, m, b, gr) in [("rollout", UInt32(0) as UInt32?, false, false), ("backward", nil, true, false), ("backward+GEMM", nil, true, true)] {
    let t0 = Date(); run(4 * B, rollout: m, backward: b, gradient: gr); let g1 = gpuTime; run(4 * B, rollout: m, backward: b, gradient: gr)
    print(String(format: "%@ %@: %.1f ms wall, %.1f ms GPU, for %d rollouts", arm, name, Date().timeIntervalSince(t0) * 500, (g1 + gpuTime) * 500, 4 * B))
  }
  exit(0)
}

if ProcessInfo.processInfo.environment["CHECK"] != nil {
  do {   // the rollout's log-likelihood of its own choices (matrix units) against the backward pass's (rows), 64 rollouts
    for i in 0..<64 { which[i] = UInt32(i * 97 % pool); draw[i] = UInt32(i + 9) }
    run(64, rollout: 0, backward: true, gradient: false)
    let rl = rllBuf.contents().bindMemory(to: Float.self, capacity: R)
    let worst = (0..<64).map { abs(Double(rl[$0] - ll[$0])) / max(1, abs(Double(ll[$0]))) }.max()!
    for i in 0..<64 { w[2 * i] = Float(g.normal()); w[2 * i + 1] = Float(g.normal()) }
    run(64, rollout: nil, backward: true, gradient: true); let split = (0..<(2 * P)).map { Double(grad[$0]) }
    SPLIT = 1; run(64, rollout: nil, backward: true, gradient: true); SPLIT = 64
    let whole = (0..<(2 * P)).map { Double(grad[$0]) }
    let diff = zip(split, whole).map { abs($0 - $1) }.max()!, size = whole.map { abs($0) }.max()!
    print(String(format: "gradient split over 64 batches against one GEMM: largest difference %.2e of %.2e", diff, size))
    print(String(format: "rollout vs backward log-likelihood: largest relative difference %.1e over 64 rollouts (rollout 0: %.4f vs %.4f)", worst, rl[0], ll[0]))
  }
  // d(-sum log p of rollout 0's recorded choices)/d theta, the backward pass against central differences
  which[0] = UInt32(pool - 1); draw[0] = 12345
  for i in 0..<(2 * R) { w[i] = 0 }; w[0] = 1
  run(1, rollout: 0, backward: false, gradient: false)
  run(1, rollout: nil, backward: true, gradient: true)
  let analytic = (0..<P).map { Double(grad[$0]) }
  print(String(format: "%@, %d parameters; rollout 0: log-likelihood of its choices %.4f", arm, P, ll[0]))
  for (name, at, count) in blocks {
    var rows: [String] = [], worst = 0.0, scale = 0.0
    for _ in 0..<3 {
      let i = at + Int(g.uniform() * Double(count)), eps: Float = 1e-2, keep = theta[i]
      theta[i] = keep + eps; run(1, rollout: nil, backward: true, gradient: false); let up = Double(ll[0])
      theta[i] = keep - eps; run(1, rollout: nil, backward: true, gradient: false); let down = Double(ll[0])
      theta[i] = keep
      let numeric = -(up - down) / Double(2 * eps)
      worst = max(worst, abs(numeric - analytic[i])); scale = max(scale, abs(numeric), abs(analytic[i]))
      rows.append(String(format: "%+.4f/%+.4f", analytic[i], numeric))
    }
    print(String(format: "  %-6@ largest difference %.1e of %.1e   (analytic/numeric) %@", name, worst, scale, rows.joined(separator: " ")))
  }
  exit(0)
}

func measure() -> String {
  let draws = 32
  var out: [String] = []
  for (mode, name) in [(UInt32(0), "sampled"), (1, "greedy"), (2, "spt"), (3, "uniform")] {
    for i in 0..<sizes.count { for d in 0..<draws { which[i * draws + d] = UInt32(pool + i); draw[i * draws + d] = UInt32(77 + d) } }
    run(sizes.count * draws, rollout: mode, backward: false, gradient: false)
    out.append(name + " " + (0..<sizes.count).map { i in String(format: "%.2f", (0..<draws).map { done[i * draws + $0] }.reduce(0, +) / Float(draws)) }.joined(separator: " "))
  }
  return out.joined(separator: " | ")
}

print("\(arm), \(P) parameters; \(B) scenarios x 4 rollouts a step; T / floor on the measured cluster at \(sizes.map { Int($0) }) bytes a rank")
print("start: " + measure())
let meanOnly = ProcessInfo.processInfo.environment["MEAN_ONLY"] != nil   // the completion task alone (a diagnosis)
var m1 = [Double](repeating: 0, count: P), m2 = [Double](repeating: 0, count: P)
var elapsed = 0.0, since = 0
let wall = Date()
for k in 0..<steps {
  let t0 = Date()
  for s in 0..<B { let sc = UInt32(Int(g.uniform() * Double(pool))); for q in 0..<4 { which[4 * s + q] = sc; draw[4 * s + q] = UInt32(truncatingIfNeeded: g.next()) } }
  run(4 * B, rollout: 0, backward: false, gradient: false)
  if ProcessInfo.processInfo.environment["TRACE"] != nil {   // a rollout whose own log-likelihood is not finite: its scenario
    let rl = rllBuf.contents().bindMemory(to: Float.self, capacity: R)
    for i in 0..<(4 * B) where !rl[i].isFinite {
      let at = Int(which[i]) * scenarioBytes
      func f(_ o: Int) -> Float { scen.withUnsafeBytes { $0.load(fromByteOffset: at + o, as: Float.self) } }
      func u(_ o: Int) -> UInt32 { scen.withUnsafeBytes { $0.load(fromByteOffset: at + o, as: UInt32.self) } }
      let rates = (0..<Int(u(536))).map { f(4 * $0) }, lats = (0..<Int(u(536))).map { f(128 + 4 * $0) }
      print(String(format: "  step %d rollout %d scenario %d draw %u: log-likelihood %f, T %f, size %.0f floor %.1f dt %.1f, %d commodities %d lanes %d nodes, rates %.2f..%.2f lat %.0f..%.0f",
                   k + 1, i, which[i], draw[i], rl[i], done[i], f(528), f(520), f(524), u(532), u(536), u(540), rates.min()!, rates.max()!, lats.min()!, lats.max()!))
      if ProcessInfo.processInfo.environment["SAVE"] != nil { netBuf.contents().withMemoryRebound(to: UInt8.self, capacity: P * 4) { try! Data(bytes: $0, count: P * 4).write(to: URL(fileURLWithPath: ProcessInfo.processInfo.environment["SAVE"]!)) }; exit(0) }
    }
  }
  // Buy 4: each rollout's baseline the mean of its scenario's other three; d c / d theta = -sum (c - b) d(-log p)
  for s in 0..<B {
    let c = (0..<4).map { Double(done[4 * s + $0]) }, mu = c.reduce(0, +) / 4, v = c.map { ($0 - mu) * ($0 - mu) }
    for q in 0..<4 {
      let b1 = (c.reduce(0, +) - c[q]) / 3, b2 = (v.reduce(0, +) - v[q]) / 3
      w[2 * (4 * s + q)] = Float(-(c[q] - b1) / Double(4 * B)); w[2 * (4 * s + q) + 1] = Float(-(v[q] - b2) / Double(4 * B))
    }
  }
  run(4 * B, rollout: nil, backward: true, gradient: true)
  var g11 = 0.0, g22 = 0.0, g12 = 0.0
  for i in 0..<P { let x = Double(grad[i]), y = Double(grad[P + i]); g11 += x * x; g22 += y * y; g12 += x * y }
  let den = g11 - 2 * g12 + g22, lambda = meanOnly ? 1 : den > 0 ? min(max((g22 - g12) / den, 0), 1) : 0.5
  let c1 = 1 - pow(0.9, Double(k + 1)), c2 = 1 - pow(0.999, Double(k + 1))
  for i in 0..<P {
    let d = lambda * Double(grad[i]) + (1 - lambda) * Double(grad[P + i])
    m1[i] = 0.9 * m1[i] + 0.1 * d; m2[i] = 0.999 * m2[i] + 0.001 * d * d
    theta[i] -= Float(rate * (m1[i] / c1) / (sqrt(m2[i] / c2) + 1e-8))
  }
  elapsed += Date().timeIntervalSince(t0); since += 1
  if ProcessInfo.processInfo.environment["TRACE"] != nil && k < 60 {
    let nan = (0..<P).filter { !theta[$0].isFinite }.count, cs = (0..<(4 * B)).map { Double(done[$0]) }
    let lls = (0..<(4 * B)).map { Double(rllBuf.contents().bindMemory(to: Float.self, capacity: R)[$0]) }
    print(String(format: "  step %d: |g1| %.3e |g2| %.3e lambda %.2f, non-finite weights %d, T mean %.2f max %.2f, log-likelihood of choices mean %.1f", k + 1, sqrt(g11), sqrt(g22), lambda, nan,
                 cs.reduce(0, +) / Double(cs.count), cs.max()!, lls.reduce(0, +) / Double(lls.count)))
  }
  if k < 3 || (k + 1) % 50 == 0 || k + 1 == steps {
    let mean = (0..<(4 * B)).map { done[$0] }.reduce(0, +) / Float(4 * B)
    print(String(format: "step %d (%.0fs): train %.3f, lambda %.2f, %.1f steps/s | ", k + 1, Date().timeIntervalSince(wall), mean, lambda, Double(since) / elapsed) + measure())
    fflush(stdout); elapsed = 0; since = 0
  }
}
