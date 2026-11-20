// The routing policy (route.metal), a stochastic forwarding table, trained as in Kool, van Hoof & Welling 2019: every
// piece draws its split at every node it reaches from its context's distribution, and each draw is credited with its
// advantage: for throughput (completion over the floor) the transaction's against the mean of its scenario's other three
// transactions; for latency and queueing the piece's own against the mean of the other pieces of its commodity's group.
// Jacobian descent over the three gradients (Quinton & Rey 2024, the UPGrad aggregator), then Adam. Measured on the
// five-node cluster as measured, at decode sizes, against every piece on the shortest-path tree and against splits
// drawn uniformly.
//
// usage: route METALLIB FABRIC [steps] [scenarios a step]
//   CHECK=1     the rollout's log-likelihood of its draws against the backward pass's; the gradient against central
//               differences (EPS, default 1e-3)
//   SIMCHECK=1  every piece on the shortest-path tree, every group posted at 0, as tests/switch.c's fixed router
//   PROFILE=1   each kernel's GPU time
import Foundation
import Metal
import MetalPerformanceShaders

let N = 6, L = 32, J = 6, D = 64, E = 192
let page = 16384.0

struct Mix { var s: UInt64
  mutating func next() -> UInt64 { s &+= 0x9e3779b97f4a7c15; var z = s; z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9; z = (z ^ (z >> 27)) &* 0x94d049bb133111eb; return z ^ (z >> 31) }
  mutating func uniform() -> Double { Double(next() >> 11) / 9007199254740992.0 }
  mutating func normal() -> Double { sqrt(-2 * log(max(uniform(), 1e-300))) * cos(2 * Double.pi * uniform()) }
}
// a lane: from, to, rate (all its cables, bytes a ns), latency (ns), cables
struct Fabric { var n: Int; var compute: [Int]; var src: [Int]; var dst: [Int]; var rate: [Double]; var lat: [Double]; var cables: [Int]; var duplex: Double }

let device = MTLCreateSystemDefaultDevice()!
let args = CommandLine.arguments
let library = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
func pipe(_ name: String) -> MTLComputePipelineState { try! device.makeComputePipelineState(function: library.makeFunction(name: name)!) }
let rollPipe = pipe("rollout"), backPipe = pipe("backward"), reducePipe = pipe("reduce"), zeroPipe = pipe("zero_rows"), layoutPipe = pipe("layout")
let queue = device.makeCommandQueue()!
let lay: [Int] = {   // the layout, from the kernel
  let b = device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
  let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(layoutPipe); enc.setBuffer(b, offset: 0, index: 0)
  enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  let p = b.contents().bindMemory(to: Int32.self, capacity: 32)
  return (0..<24).map { Int(p[$0]) }
}()
let full = lay[0] == 1, P = lay[1], REC = lay[12], BSCRATCH = lay[13], LS = lay[14], RLS = lay[11]
let L_IN = lay[2], L_DR = lay[3], L_RX = lay[4], L_DZ = lay[5], L_DHT = lay[6], L_XM = lay[7], L_DU = lay[8], L_U = lay[9], L_DY = lay[10]
let R_DX0 = 0, R_IN = 64, R_DW = lay[15], R_N = lay[16], R_LAYER = lay[17], RSCRATCH = lay[18], SIM = lay[19], IN = lay[20], FS = lay[21]
let scenarioBytes = lay[22], SB_HDR = lay[23]
let GW = full ? 2 * D : D, G = D * GW
let arm = full ? "GRU" : "minGRU"

// a scenario, as route.metal's Scenario: an all-gather of S a rank among the compute nodes. The floor as tests/switch.c's:
// the cut bound, and the slowest pair's shortest delay for one piece (latency and the piece's time on one cable, each hop)
func scenario(_ f: Fabric, _ S: Double, _ g: inout Mix, background: Bool) -> [UInt8] {
  let n = f.n, nl = f.src.count, q = min(S, page), nc = f.compute.count
  let bg = (0..<nl).map { background && g.uniform() < 0.3 ? f.rate[$0] * 50e3 * g.uniform() : 0 }
  var cut = 0.0
  for v in f.compute {
    let inr = (0..<nl).filter { f.dst[$0] == v }.map { f.rate[$0] }.reduce(0, +), outr = (0..<nl).filter { f.src[$0] == v }.map { f.rate[$0] }.reduce(0, +)
    cut = max(cut, Double(nc - 1) * S / inr, S / outr)
  }
  let w = (0..<nl).map { f.lat[$0] + q / (f.rate[$0] / Double(f.cables[$0])) }
  var dist = Array(repeating: Array(repeating: Double.infinity, count: n), count: n)
  for v in 0..<n { dist[v][v] = 0 }
  for i in 0..<nl { dist[f.src[i]][f.dst[i]] = min(dist[f.src[i]][f.dst[i]], w[i]) }
  for m in 0..<n { for a in 0..<n { for b in 0..<n { dist[a][b] = min(dist[a][b], dist[a][m] + dist[m][b]) } } }
  var reach = 0.0
  for a in f.compute { for b in f.compute where a != b { reach = max(reach, dist[a][b]) } }
  var b = [UInt8](repeating: 0, count: scenarioBytes)
  func f32(_ at: Int, _ v: Double) { var x = Float(v); withUnsafeBytes(of: &x) { for i in 0..<4 { b[at + i] = $0[i] } } }
  func u32(_ at: Int, _ v: Int) { var x = UInt32(v); withUnsafeBytes(of: &x) { for i in 0..<4 { b[at + i] = $0[i] } } }
  for l in 0..<nl { f32(4 * l, f.rate[l]); f32(128 + 4 * l, f.lat[l]); f32(256 + 4 * l, bg[l]); b[384 + l] = UInt8(f.src[l]); b[416 + l] = UInt8(f.dst[l]); b[448 + l] = UInt8(f.cables[l]) }
  for i in 0..<(J * N) { b[480 + i] = 255 }
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
    for v in 0..<n where keep[v] { b[480 + j * N + v] = UInt8(by[v]) }
    b[516 + j] = UInt8(s); var mask = 0; for d in f.compute where d != s { mask |= 1 << d }; b[522 + j] = UInt8(mask)
    f32(528 + 4 * j, S)
  }
  f32(552, max(cut, reach)); f32(556, S); f32(560, f.duplex); u32(564, nc); u32(568, nl); u32(572, n); f32(576, cut); f32(580, reach)
  return b
}
func cabled(_ n: Int, _ compute: [Int], _ pairs: [(Int, Int)], _ g: inout Mix) -> Fabric {
  var f = Fabric(n: n, compute: compute, src: [], dst: [], rate: [], lat: [], cables: [], duplex: 0.85 + 0.15 * g.uniform())
  for (u, v) in pairs {
    let c = 1 + Int(g.uniform() * 4), each = 4 + 8 * g.uniform()
    for (a, b) in [(u, v), (v, u)] { f.src.append(a); f.dst.append(b); f.rate.append(Double(c) * each); f.lat.append(2e3 + 11e3 * g.uniform()); f.cables.append(c) }
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
func load(_ path: String) -> Fabric {   // tests/fabric-*.txt: node NAME compute|relay; link FROM TO GBps us cables; duplex SHARE
  var names: [String] = [], f = Fabric(n: 0, compute: [], src: [], dst: [], rate: [], lat: [], cables: [], duplex: 1)
  for line in try! String(contentsOfFile: path, encoding: .utf8).split(separator: "\n") where !line.hasPrefix("#") {
    let w = line.split(separator: " ").map(String.init)
    if w.first == "duplex" { f.duplex = Double(w[1])! }
    if w.first == "node" { names.append(w[1]); if w[2] == "compute" { f.compute.append(names.count - 1) } }
    if w.first == "link" {
      f.src.append(names.firstIndex(of: w[1])!); f.dst.append(names.firstIndex(of: w[2])!)
      f.rate.append(Double(w[3])!); f.lat.append(Double(w[4])! * 1e3); f.cables.append(Int(w[5])!)
    }
  }
  f.n = names.count
  return f
}

// parameters (as route.metal's Net) and the gradient jobs: (delta offset, input offset or -1 for ones, rows, cols,
// parameter offset)
let O_LAYER = D * IN + D, O_OUT = O_LAYER + 2 * LS
precondition(O_OUT + FS * D == P, "layout")
var jobs: [(Int, Int, Int, Int, Int)] = [(R_DX0, R_IN, D, IN, 0), (R_DX0, -1, D, 1, D * IN)]
var blocks: [(String, Int, Int)] = [("inp", 0, D * IN), ("binp", D * IN, D)]
for layer in 0..<2 {
  let B = R_LAYER + layer * RLS, P0 = O_LAYER + layer * LS
  let oz = P0, obz = oz + G, or = obz + D, obr = or + G, oh = full ? obr + D : obz + D, obh = oh + G, oup = obh + D, obu = oup + E * D, odn = obu + E, obd = odn + D * E
  precondition(obd + D == P0 + LS, "layer layout")
  jobs += [(B + L_DZ, B + L_IN, D, GW, oz), (B + L_DZ, -1, D, 1, obz), (B + L_DHT, B + (full ? L_RX : L_IN), D, GW, oh), (B + L_DHT, -1, D, 1, obh),
           (B + L_DU, B + L_XM, E, D, oup), (B + L_DU, -1, E, 1, obu), (B + L_DY, B + L_U, D, E, odn), (B + L_DY, -1, D, 1, obd)]
  if full { jobs += [(B + L_DR, B + L_IN, D, GW, or), (B + L_DR, -1, D, 1, obr)]; blocks += [("r\(layer)", or, G), ("br\(layer)", obr, D)] }
  blocks += [("z\(layer)", oz, G), ("bz\(layer)", obz, D), ("h\(layer)", oh, G), ("bh\(layer)", obh, D), ("up\(layer)", oup, E * D), ("bu\(layer)", obu, E),
             ("down\(layer)", odn, D * E), ("bd\(layer)", obd, D)]
}
jobs += [(R_DW, R_N, FS, D, O_OUT)]
blocks += [("out", O_OUT, FS * D)]
precondition(jobs.map { $0.2 * $0.3 }.reduce(0, +) == P, "the gradient jobs cover every parameter")

let measured = load(args[2])
let steps = args.count > 3 ? Int(args[3])! : 200, B = args.count > 4 ? Int(args[4])! : 32
let rate = Double(ProcessInfo.processInfo.environment["RATE"] ?? "1e-3")!
let R = max(4 * B, 6 * 16)
var g = Mix(s: UInt64(ProcessInfo.processInfo.environment["SEED"] ?? "20261119")!)
let pool = 8192
var scen = [UInt8](); scen.reserveCapacity((pool + 8) * scenarioBytes)
for _ in 0..<pool { let f = network(&g, measured); let S = exp(log(6144.0) + g.uniform() * (log(12582912.0) - log(6144.0))).rounded(); scen += scenario(f, S, &g, background: true) }
let sizes = [6144.0, 24576, 98304, 393216, 3145728, 12582912]
for S in sizes { scen += scenario(measured, S, &g, background: false) }
func field<T>(_ k: Int, _ at: Int, _ type: T.Type) -> T { scen.withUnsafeBytes { $0.load(fromByteOffset: k * scenarioBytes + at, as: T.self) } }
var net = [Float](repeating: 0, count: P)
for (name, at, count) in blocks where !name.hasPrefix("b") {   // weights at 1/sqrt(fan-in), biases zero
  let fan = name == "inp" ? IN : name.hasPrefix("down") ? E : (name.hasPrefix("z") || name.hasPrefix("r") || name.hasPrefix("h")) ? GW : D
  for i in 0..<count { net[at + i] = Float(g.normal() / sqrt(Double(fan))) }
}

let CHUNK = 2048, RCAP = R * 1536 / CHUNK * CHUNK + CHUNK
func buffer(_ bytes: Int, _ mode: MTLResourceOptions = .storageModeShared) -> MTLBuffer { device.makeBuffer(length: max(bytes, 16), options: mode)! }
let scenBuf = device.makeBuffer(bytes: scen, length: scen.count, options: .storageModeShared)!
let netBuf = device.makeBuffer(bytes: net, length: P * 4, options: .storageModeShared)!
let whichBuf = buffer(R * 4), drawBuf = buffer(R * 4), costBuf = buffer(R * 16), rllBuf = buffer(R * 4), llBuf = buffer(R * 8), rowsBuf = buffer(16)
let rsBuf = buffer(R * RSCRATCH * 4, .storageModePrivate), simBuf = buffer(R * SIM * 4), bsBuf = buffer(R * BSCRATCH * 4, .storageModePrivate)
let recBuf = buffer(RCAP * REC * 4, .storageModePrivate)
let onesBuf = device.makeBuffer(bytes: [Float](repeating: 1, count: RCAP), length: RCAP * 4, options: .storageModeShared)!
let wBuf = buffer(R * 12), gBuf = buffer(3 * P * 4), partBuf = buffer(RCAP / CHUNK * P * 4, .storageModePrivate)
let which = whichBuf.contents().bindMemory(to: UInt32.self, capacity: R), draw = drawBuf.contents().bindMemory(to: UInt32.self, capacity: R)
let cost = costBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: R), rll = rllBuf.contents().bindMemory(to: Float.self, capacity: R)
let ll = llBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: R), rowsCount = rowsBuf.contents().bindMemory(to: UInt32.self, capacity: 4)
let w = wBuf.contents().bindMemory(to: Float.self, capacity: 3 * R), grad = gBuf.contents().bindMemory(to: Float.self, capacity: 3 * P)
let theta = netBuf.contents().bindMemory(to: Float.self, capacity: P)
let simHeader = { (r: Int, k: Int) -> UInt32 in simBuf.contents().load(fromByteOffset: (r * SIM + SB_HDR + k) * 4, as: UInt32.self) }

// the GEMMs over `padded` rows in batches of CHUNK, made once a size: G_b = delta^T input over batch b's rows, then the
// batches summed into the task's gradient
var plans: [Int: [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)]] = [:]
func plan(_ padded: Int) -> [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)] {
  if let p = plans[padded] { return p }
  let nb = padded / CHUNK, stride = REC * 4
  var gemms: [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)] = [], base = 0
  for (dOff, xOff, rows, cols, gOff) in jobs {
    let A = MPSMatrix(buffer: recBuf, offset: dOff * 4, descriptor: MPSMatrixDescriptor(rows: CHUNK, columns: rows, matrices: nb, rowBytes: stride, matrixBytes: CHUNK * stride, dataType: .float32))
    let X = xOff < 0 ? MPSMatrix(buffer: onesBuf, offset: 0, descriptor: MPSMatrixDescriptor(rows: CHUNK, columns: 1, matrices: nb, rowBytes: 4, matrixBytes: CHUNK * 4, dataType: .float32))
                     : MPSMatrix(buffer: recBuf, offset: xOff * 4, descriptor: MPSMatrixDescriptor(rows: CHUNK, columns: cols, matrices: nb, rowBytes: stride, matrixBytes: CHUNK * stride, dataType: .float32))
    let C = MPSMatrix(buffer: partBuf, offset: base * 4, descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, matrices: nb, rowBytes: cols * 4, matrixBytes: rows * cols * 4, dataType: .float32))
    let mm = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: rows, resultColumns: cols, interiorColumns: CHUNK, alpha: 1, beta: 0)
    mm.batchStart = 0; mm.batchSize = nb
    gemms.append((mm, A, X, C, base, rows * cols, gOff)); base += nb * rows * cols
  }
  plans[padded] = gemms
  return gemms
}
var gpuTime = 0.0
let tg = MTLSize(width: 256, height: 1, depth: 1)
// mode: 0 drawn from the tables, 2 the shortest-path tree, 3 uniform, 4 the shortest-path tree all posted at 0
func rollout(_ count: Int, _ mode: UInt32) {
  let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(rollPipe)
  for (i, b) in [scenBuf, netBuf, whichBuf, drawBuf, costBuf, rsBuf, simBuf].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
  var m = mode; enc.setBytes(&m, length: 4, index: 7); enc.setBuffer(rllBuf, offset: 0, index: 8)
  enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: tg)
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted(); gpuTime = cb.gpuEndTime - cb.gpuStartTime
  if let e = cb.error { print("error: \(e)"); exit(1) }
  for r in 0..<count where cost[r].w < 0 { print("rollout \(r): a capacity (draws, groups or heap) overflowed; nothing was dropped silently, so this stops"); exit(1) }
}
// one task's gradient into grad[task P ...]: the backward pass, then the GEMMs over the rows it wrote
func gradient(_ count: Int, _ task: UInt32) {
  rowsCount[0] = 0
  var cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(backPipe)
  for (i, b) in [scenBuf, netBuf, whichBuf, bsBuf, simBuf, recBuf, rowsBuf].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
  var t = task, cap = UInt32(RCAP); enc.setBytes(&t, length: 4, index: 7); enc.setBuffer(wBuf, offset: 0, index: 8); enc.setBuffer(llBuf, offset: 0, index: 9); enc.setBytes(&cap, length: 4, index: 10)
  enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: tg)
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted(); var time = cb.gpuEndTime - cb.gpuStartTime
  if let e = cb.error { print("error: \(e)"); exit(1) }
  let used = Int(rowsCount[0])
  if used > RCAP { print("the rows (\(used)) passed their capacity (\(RCAP)); this stops"); exit(1) }
  let padded = max(CHUNK, (used + CHUNK - 1) / CHUNK * CHUNK)
  cb = queue.makeCommandBuffer()!
  enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(zeroPipe); enc.setBuffer(recBuf, offset: 0, index: 0)
  var span = SIMD2<UInt32>(UInt32(used), UInt32(padded - used)); enc.setBytes(&span, length: 8, index: 1)
  enc.dispatchThreads(MTLSize(width: max(1, (padded - used) * REC), height: 1, depth: 1), threadsPerThreadgroup: tg)
  enc.endEncoding()
  let gemms = plan(padded)
  for (mm, A, X, C, _, _, _) in gemms { mm.encode(commandBuffer: cb, leftMatrix: A, rightMatrix: X, resultMatrix: C) }
  enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(reducePipe); enc.setBuffer(partBuf, offset: 0, index: 0); enc.setBuffer(gBuf, offset: 0, index: 1)
  for (_, _, _, _, base, size, at) in gemms {
    var q = SIMD4<UInt32>(UInt32(base), UInt32(size), UInt32(padded / CHUNK), UInt32(Int(task % 3) * P + at)); enc.setBytes(&q, length: 16, index: 2)
    enc.dispatchThreads(MTLSize(width: size, height: 1, depth: 1), threadsPerThreadgroup: tg)
  }
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted(); time += cb.gpuEndTime - cb.gpuStartTime; gpuTime = time
  if let e = cb.error { print("error: \(e)"); exit(1) }
}
func pieces(_ k: Int) -> (Int, Int) { (Int(field(k, 564, UInt32.self)), max(1, Int((Double(field(k, 556, Float.self)) / page).rounded(.up)))) }

if ProcessInfo.processInfo.environment["SIMCHECK"] != nil {
  print("every piece on the shortest-path tree, all posted at 0 (tests/switch.c's fixed router): completion, floor")
  for (i, S) in sizes.enumerated() {
    which[0] = UInt32(pool + i); draw[0] = 1
    rollout(1, 4)
    let fl = field(pool + i, 552, Float.self)
    print(String(format: "all-gather %9.0f  %9.2fus  floor %8.2fus", S, cost[0].x * fl / 1000, fl / 1000))
  }
  exit(0)
}

if ProcessInfo.processInfo.environment["CHECK"] != nil {
  let n = 32
  for i in 0..<n { which[i] = UInt32(i * 97 % pool); draw[i] = UInt32(i + 9) }
  rollout(n, 0); gradient(n, 3)
  let worst = (0..<n).map { abs(Double(rll[$0] - ll[$0].y)) / max(1, abs(Double(ll[$0].y))) }.max()!
  print(String(format: "rollout vs backward log-likelihood of the draws (unmasked tables): largest relative difference %.1e over %d rollouts (rollout 0: %.4f vs %.4f, %d draws in %d groups)",
               worst, n, rll[0], ll[0].y, simHeader(0, 0), simHeader(0, 1)))
  which[0] = UInt32(pool - 1); draw[0] = 12345
  rollout(1, 0); gradient(1, 3)
  let analytic = (0..<P).map { Double(grad[$0]) }
  print(String(format: "%@, %d parameters; rollout 0: %d draws in %d groups, masked log-likelihood %.4f", arm, P, simHeader(0, 0), simHeader(0, 1), ll[0].x))
  let eps = Float(ProcessInfo.processInfo.environment["EPS"] ?? "1e-3")!
  for (name, at, count) in blocks {
    var rows: [String] = [], worst = 0.0, scale = 0.0
    for _ in 0..<3 {
      let i = at + Int(g.uniform() * Double(count)), keep = theta[i]
      theta[i] = keep + eps; gradient(1, 3); let up = Double(ll[0].x)
      theta[i] = keep - eps; gradient(1, 3); let down = Double(ll[0].x)
      theta[i] = keep
      let numeric = (up - down) / Double(2 * eps)
      worst = max(worst, abs(numeric - analytic[i])); scale = max(scale, abs(numeric), abs(analytic[i]))
      rows.append(String(format: "%+.4f/%+.4f", analytic[i], numeric))
    }
    print(String(format: "  %-6@ largest difference %.1e of %.1e   (analytic/numeric) %@", name, worst, scale, rows.joined(separator: " ")))
  }
  exit(0)
}

func measure() -> String {   // T / floor at each decode size on the measured cluster, the mean over 16 transactions
  let draws = 16
  var out: [String] = []
  for (mode, name) in [(UInt32(0), "policy"), (2, "spt"), (3, "uniform")] {
    for i in 0..<sizes.count { for d in 0..<draws { which[i * draws + d] = UInt32(pool + i); draw[i * draws + d] = UInt32(77 + d) } }
    rollout(sizes.count * draws, mode)
    out.append(name + " " + (0..<sizes.count).map { i in String(format: "%.2f", (0..<draws).map { cost[i * draws + $0].x }.reduce(0, +) / Float(draws)) }.joined(separator: " "))
  }
  return out.joined(separator: " | ")
}

// UPGrad's weights from the Jacobian's Gram matrix G = J J^T: for each task i, u_i = argmin_(u >= e_i) u^T G u (the dual
// of projecting g_i onto {x : J x >= 0}, whose projection is J^T u_i); the aggregate J^T (sum_i u_i) / 3. The 3-variable
// box QP by projected gradient, its step 1 / (3 max |G|), a bound on G's largest eigenvalue
func upgrad(_ G: [[Double]]) -> [Double] {
  var top = 0.0
  for t in 0..<3 { for s in 0..<3 { top = max(top, abs(G[t][s])) } }
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

if ProcessInfo.processInfo.environment["PROFILE"] != nil {
  for s in 0..<B { for q in 0..<4 { which[4 * s + q] = UInt32(s); draw[4 * s + q] = UInt32(4 * s + q + 1) } }
  for i in 0..<(3 * R) { w[i] = 1e-3 }
  rollout(4 * B, 0); rollout(4 * B, 0); print(String(format: "%@ rollout: %.1f ms GPU for %d transactions", arm, gpuTime * 1000, 4 * B))
  for t in 0..<3 { gradient(4 * B, UInt32(t)); print(String(format: "%@ task %d backward + GEMMs: %.1f ms GPU, %d rows", arm, t, gpuTime * 1000, rowsCount[0])) }
  exit(0)
}

print("\(arm), \(P) parameters; \(B) scenarios x 4 transactions a step; tasks: throughput, latency, queueing; T / floor on the measured cluster at \(sizes.map { Int($0) }) bytes a rank")
print("start: " + measure())
var m1 = [Double](repeating: 0, count: P), m2 = [Double](repeating: 0, count: P)
var elapsed = 0.0, since = 0
let wall = Date()
for k in 0..<steps {
  let t0 = Date()
  for s in 0..<B { let sc = UInt32(Int(g.uniform() * Double(pool))); for q in 0..<4 { which[4 * s + q] = sc; draw[4 * s + q] = UInt32(truncatingIfNeeded: g.next()) } }
  rollout(4 * B, 0)
  for s in 0..<B {   // throughput: each transaction against its scenario's other three; the pieces' tasks' scale
    let c = (0..<4).map { Double(cost[4 * s + $0].x) }, total = c.reduce(0, +), (nj, np) = pieces(Int(which[4 * s]))
    for q in 0..<4 {
      w[3 * (4 * s + q)] = Float((c[q] - (total - c[q]) / 3) / Double(4 * B))
      w[3 * (4 * s + q) + 1] = Float(1 / Double(4 * B * nj * np)); w[3 * (4 * s + q) + 2] = w[3 * (4 * s + q) + 1]
    }
  }
  for t in 0..<3 { gradient(4 * B, UInt32(t)) }
  var Gm = [[Double]](repeating: [0, 0, 0], count: 3)
  for i in 0..<P { let v = [Double(grad[i]), Double(grad[P + i]), Double(grad[2 * P + i])]; for a in 0..<3 { for b in 0..<3 { Gm[a][b] += v[a] * v[b] } } }
  let alpha = upgrad(Gm)
  let c1 = 1 - pow(0.9, Double(k + 1)), c2 = 1 - pow(0.999, Double(k + 1))
  for i in 0..<P {
    let d = alpha[0] * Double(grad[i]) + alpha[1] * Double(grad[P + i]) + alpha[2] * Double(grad[2 * P + i])
    m1[i] = 0.9 * m1[i] + 0.1 * d; m2[i] = 0.999 * m2[i] + 0.001 * d * d
    theta[i] -= Float(rate * (m1[i] / c1) / (sqrt(m2[i] / c2) + 1e-8))
  }
  elapsed += Date().timeIntervalSince(t0); since += 1
  if k < 3 || (k + 1) % 50 == 0 || k + 1 == steps {
    let mean = (0..<3).map { t in (0..<(4 * B)).map { Double(t == 0 ? cost[$0].x : t == 1 ? cost[$0].y : cost[$0].z) }.reduce(0, +) / Double(4 * B) }
    let nonfinite = (0..<P).filter { !theta[$0].isFinite }.count
    print(String(format: "step %d (%.0fs): train throughput %.3f latency %.3f queueing %.3f, weights %.2f %.2f %.2f, %.2f steps/s%@ | ", k + 1, Date().timeIntervalSince(wall),
                 mean[0], mean[1], mean[2], alpha[0], alpha[1], alpha[2], Double(since) / elapsed, nonfinite > 0 ? ", \(nonfinite) NON-FINITE WEIGHTS" : "") + measure())
    fflush(stdout); elapsed = 0; since = 0
  }
}
