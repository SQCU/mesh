// The routing policy (route.metal) trained as in Kool, van Hoof & Welling 2019: REINFORCE, each rollout's baseline the
// mean of its scenario's other three rollouts, on three costs a rollout, each its own task: throughput (completion over
// the floor), latency (pieces' arrival less posting, over the floor), congestion (queueing wait over sending time).
// Jacobian descent over the three (Quinton & Rey 2024, the UPGrad aggregator): each task's gradient projected onto
// the dual cone of the 3 x P Jacobian, the projections averaged, so the step conflicts with none of them; then Adam. Measured on the five-node cluster as measured, at decode sizes, against every piece
// on the unloaded shortest-path tree and against lanes chosen uniformly.
//
// usage: route METALLIB FABRIC [steps] [scenarios a step]
//   CHECK=1     the rollout's log-likelihood against the backward pass's; per-rollout GEMMs against one GEMM; the
//               gradient against central differences
//   SIMCHECK=1  every piece on the shortest-path tree, every group posted at 0, as tests/switch.c's fixed router
import Foundation
import Metal
import MetalPerformanceShaders

let N = 6, L = 32, J = 6, KS = 3, TMAX = N - 1, D = 64, E = 192, CF = 8, IN = CF + D, LF = 4, KREC = 196, PR = KS * J
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
let rollPipe = pipe("rollout"), backPipe = pipe("backward"), reducePipe = pipe("reduce"), layoutPipe = pipe("layout")
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
let full = lay[0] == 1, P = lay[1], REC = lay[12], SCRATCH = lay[13], LS = lay[14], RLS = lay[11]
let L_IN = lay[2], L_DR = lay[3], L_RX = lay[4], L_DZ = lay[5], L_DHT = lay[6], L_XM = lay[7], L_DU = lay[8], L_U = lay[9], L_DY = lay[10]
let R_DX0 = 0, R_IN = 64, R_LAYER = 136, R_DQ = lay[15], R_XF = lay[16], R_DST = lay[17]
let RSCRATCH = lay[18], SIM = lay[19], DEC = lay[20], EPMAX = lay[21], scenarioBytes = lay[22], TAPE = lay[23]
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
  f32(552, max(cut, reach)); f32(556, S); f32(560, f.duplex); u32(564, nc); u32(568, nl); u32(572, n)
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
precondition(jobs.map { $0.3 * $0.4 }.reduce(0, +) == P, "the gradient jobs cover every parameter")

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
let whichBuf = buffer(R * 4), drawBuf = buffer(R * 4), costBuf = buffer(R * 16), llBuf = buffer(R * 4), rllBuf = buffer(R * 4), nbBuf = buffer(R * 4)
let tapeBuf = buffer(R * DEC * TAPE * 4), phiBuf = buffer(R * EPMAX * L * LF * 4), countBuf = buffer(R * EPMAX * PR * 4)
let rsBuf = buffer(R * RSCRATCH * 4, .storageModePrivate), simBuf = buffer(R * SIM * 4, .storageModePrivate), bsBuf = buffer(R * SCRATCH * 4, .storageModePrivate)
let recBuf = buffer(R * DEC * REC * 4, .storageModePrivate), krecBuf = buffer(R * EPMAX * L * KREC * 4, .storageModePrivate)
let onesBuf = device.makeBuffer(bytes: [Float](repeating: 1, count: R * DEC), length: R * DEC * 4, options: .storageModeShared)!
let wBuf = buffer(R * 12), gBuf = buffer(3 * P * 4), partBuf = buffer(R * P * 4, .storageModePrivate)
let which = whichBuf.contents().bindMemory(to: UInt32.self, capacity: R), draw = drawBuf.contents().bindMemory(to: UInt32.self, capacity: R)
let cost = costBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: R)
let ll = llBuf.contents().bindMemory(to: Float.self, capacity: R), rll = rllBuf.contents().bindMemory(to: Float.self, capacity: R)
let w = wBuf.contents().bindMemory(to: Float.self, capacity: 3 * R), grad = gBuf.contents().bindMemory(to: Float.self, capacity: 3 * P)
let theta = netBuf.contents().bindMemory(to: Float.self, capacity: P)

// the GEMMs for `count` rollouts in `nb` batches, made once: G_b = delta^T input over batch b's records, contiguous per
// (job, batch) at `base`; then each task's gradient the batches' sum weighted by w
var plans: [Int: [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)]] = [:]
func plan(_ count: Int, _ nb: Int) -> [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)] {
  if let p = plans[count * 1000 + nb] { return p }
  var gemms: [(MPSMatrixMultiplication, MPSMatrix, MPSMatrix, MPSMatrix, Int, Int, Int)] = [], base = 0
  for (kind, dOff, xOff, rows, cols, gOff) in jobs {
    let per = (kind == 0 ? DEC : EPMAX * L) * (count / nb), stride = (kind == 0 ? REC : KREC) * 4, recs = kind == 0 ? recBuf : krecBuf
    let A = MPSMatrix(buffer: recs, offset: dOff * 4, descriptor: MPSMatrixDescriptor(rows: per, columns: rows, matrices: nb, rowBytes: stride, matrixBytes: per * stride, dataType: .float32))
    let X = xOff < 0 ? MPSMatrix(buffer: onesBuf, offset: 0, descriptor: MPSMatrixDescriptor(rows: per, columns: 1, matrices: nb, rowBytes: 4, matrixBytes: per * 4, dataType: .float32))
                     : MPSMatrix(buffer: recs, offset: xOff * 4, descriptor: MPSMatrixDescriptor(rows: per, columns: cols, matrices: nb, rowBytes: stride, matrixBytes: per * stride, dataType: .float32))
    let C = MPSMatrix(buffer: partBuf, offset: base * 4, descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, matrices: nb, rowBytes: cols * 4, matrixBytes: rows * cols * 4, dataType: .float32))
    let mm = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: rows, resultColumns: cols, interiorColumns: per, alpha: 1, beta: 0)
    mm.batchStart = 0; mm.batchSize = nb
    gemms.append((mm, A, X, C, base, rows * cols, gOff)); base += nb * rows * cols
  }
  plans[count * 1000 + nb] = gemms
  return gemms
}
var gpuTime = 0.0
// mode: 0 sampled, 1 most likely, 2 shortest-path tree, 3 uniform, 4 shortest-path tree all posted at 0; batches: the
// GEMMs' (one a rollout, so each rollout's gradient carries its own weights)
func run(_ count: Int, rollout mode: UInt32?, backward: Bool, gradient: Bool, batches: Int? = nil) {
  let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
  let tg = MTLSize(width: 256, height: 1, depth: 1), grid = MTLSize(width: count, height: 1, depth: 1)
  if var m = mode {
    enc.setComputePipelineState(rollPipe)
    for (i, b) in [scenBuf, netBuf, whichBuf, drawBuf, costBuf, tapeBuf, phiBuf, rsBuf, simBuf].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    enc.setBytes(&m, length: 4, index: 9)
    for (i, b) in [rllBuf, countBuf, nbBuf].enumerated() { enc.setBuffer(b, offset: 0, index: 10 + i) }
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
  }
  if backward {
    enc.setComputePipelineState(backPipe)
    for (i, b) in [netBuf, tapeBuf, phiBuf, recBuf, krecBuf, bsBuf, llBuf, countBuf, nbBuf].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
  }
  enc.endEncoding()
  if gradient {
    let nb = batches ?? count, gemms = plan(count, nb)
    for (mm, A, X, C, _, _, _) in gemms { mm.encode(commandBuffer: cb, leftMatrix: A, rightMatrix: X, resultMatrix: C) }
    let red = cb.makeComputeCommandEncoder()!
    red.setComputePipelineState(reducePipe); red.setBuffer(partBuf, offset: 0, index: 0); red.setBuffer(gBuf, offset: 0, index: 1); red.setBuffer(wBuf, offset: 0, index: 3)
    var p = UInt32(P); red.setBytes(&p, length: 4, index: 4)
    for (_, _, _, _, base, size, at) in gemms {
      var q = SIMD4<UInt32>(UInt32(base), UInt32(size), UInt32(nb), UInt32(at)); red.setBytes(&q, length: 16, index: 2)
      red.dispatchThreads(MTLSize(width: size, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    red.endEncoding()
  }
  cb.commit(); cb.waitUntilCompleted(); gpuTime = cb.gpuEndTime - cb.gpuStartTime
  if let e = cb.error { print("error: \(e)"); exit(1) }
}

if ProcessInfo.processInfo.environment["SIMCHECK"] != nil {
  print("every piece on the shortest-path tree, all posted at 0 (tests/switch.c's fixed router): completion, floor")
  for (i, S) in sizes.enumerated() {
    which[0] = UInt32(pool + i); draw[0] = 1
    run(1, rollout: 4, backward: false, gradient: false)
    let fl = scen.withUnsafeBytes { $0.load(fromByteOffset: (pool + i) * scenarioBytes + 552, as: Float.self) }
    print(String(format: "all-gather %9.0f  %9.2fus  floor %8.2fus  (%d arrivals)", S, cost[0].x * fl / 1000, fl / 1000, Int(cost[0].w)))
  }
  exit(0)
}

if ProcessInfo.processInfo.environment["CHECK"] != nil {
  let n = 64
  for i in 0..<n { which[i] = UInt32(i * 97 % pool); draw[i] = UInt32(i + 9) }
  run(n, rollout: 0, backward: true, gradient: false)
  let worst = (0..<n).map { abs(Double(rll[$0] - ll[$0])) / max(1, abs(Double(ll[$0]))) }.max()!
  print(String(format: "rollout vs backward log-likelihood of the action: largest relative difference %.1e over %d rollouts (rollout 0: %.4f vs %.4f)", worst, n, rll[0], ll[0]))
  for i in 0..<(3 * n) { w[i] = 1 }
  run(n, rollout: nil, backward: true, gradient: true); let each = (0..<P).map { Double(grad[$0]) }
  run(n, rollout: nil, backward: true, gradient: true, batches: 1); let one = (0..<P).map { Double(grad[$0]) }
  print(String(format: "per-rollout GEMMs summed against one GEMM: largest difference %.2e of %.2e", zip(each, one).map { abs($0 - $1) }.max()!, one.map { abs($0) }.max()!))
  // d(-log p of rollout 0's action)/d theta against central differences
  which[0] = UInt32(pool - 1); draw[0] = 12345
  for i in 0..<(3 * R) { w[i] = 0 }; w[0] = 1
  run(1, rollout: 0, backward: false, gradient: false)
  run(1, rollout: nil, backward: true, gradient: true)
  let analytic = (0..<P).map { Double(grad[$0]) }
  print(String(format: "%@, %d parameters; rollout 0: %d decision points, log-likelihood of its action %.4f", arm, P, nbBuf.contents().load(as: UInt32.self), ll[0]))
  for (name, at, count) in blocks {
    var rows: [String] = [], worst = 0.0, scale = 0.0
    for _ in 0..<3 {
      let i = at + Int(g.uniform() * Double(count)), eps = Float(ProcessInfo.processInfo.environment["EPS"] ?? "1e-2")!, keep = theta[i]
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

if let c = ProcessInfo.processInfo.environment["LATENCY"] {   // the rollout's GPU time against how many run at once
  for n in c.split(separator: ",").map({ Int($0)! }) where n <= R {
    for i in 0..<n { which[i] = UInt32(i % pool); draw[i] = UInt32(i + 1) }
    run(n, rollout: 0, backward: false, gradient: false)
    var ms = 0.0; for _ in 0..<5 { run(n, rollout: 0, backward: false, gradient: false); ms += gpuTime * 200 }
    print(String(format: "%@ %5d rollouts at once: %.2f ms, %.0f rollouts/s", arm, n, ms, Double(n) / ms * 1000))
  }
  exit(0)
}

func measure() -> String {   // T / floor at each decode size on the measured cluster, the mean over 32 draws
  let draws = 32
  var out: [String] = []
  for (mode, name) in [(UInt32(0), "sampled"), (1, "greedy"), (2, "spt"), (3, "uniform")] {
    for i in 0..<sizes.count { for d in 0..<draws { which[i * draws + d] = UInt32(pool + i); draw[i * draws + d] = UInt32(77 + d) } }
    run(sizes.count * draws, rollout: mode, backward: false, gradient: false)
    out.append(name + " " + (0..<sizes.count).map { i in String(format: "%.2f", (0..<draws).map { cost[i * draws + $0].x }.reduce(0, +) / Float(draws)) }.joined(separator: " "))
  }
  return out.joined(separator: " | ")
}

// UPGrad's weights from the Jacobian's Gram matrix G = J J^T: for each task i, u_i = argmin_(u >= e_i) u^T G u (the dual
// of projecting g_i onto {x : J x >= 0}, whose projection is J^T u_i); the aggregate J^T (sum_i u_i) / 3. The 3-variable
// box QP by projected gradient, its step 1 / (G's largest diagonal sum, a bound on its largest eigenvalue)
func upgrad(_ G: [[Double]]) -> [Double] {
  var trace = 0.0
  for t in 0..<3 { for s in 0..<3 { trace = max(trace, abs(G[t][s])) } }
  let step = 1 / max(3 * trace, 1e-300)
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

print("\(arm), \(P) parameters; \(B) scenarios x 4 rollouts a step; tasks: throughput, latency, congestion; T / floor on the measured cluster at \(sizes.map { Int($0) }) bytes a rank")
print("start: " + measure())
var m1 = [Double](repeating: 0, count: P), m2 = [Double](repeating: 0, count: P)
var elapsed = 0.0, since = 0
let wall = Date()
for k in 0..<steps {
  let t0 = Date()
  for s in 0..<B { let sc = UInt32(Int(g.uniform() * Double(pool))); for q in 0..<4 { which[4 * s + q] = sc; draw[4 * s + q] = UInt32(truncatingIfNeeded: g.next()) } }
  run(4 * B, rollout: 0, backward: false, gradient: false)
  // Buy 4: each rollout's baseline its scenario's other three; d c / d theta = -sum (c - b) d(-log p)
  for s in 0..<B {
    for task in 0..<3 {
      let c = (0..<4).map { q -> Double in let v = cost[4 * s + q]; return Double(task == 0 ? v.x : task == 1 ? v.y : v.z) }, total = c.reduce(0, +)
      for q in 0..<4 { w[3 * (4 * s + q) + task] = Float(-(c[q] - (total - c[q]) / 3) / Double(4 * B)) }
    }
  }
  run(4 * B, rollout: nil, backward: true, gradient: true)
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
    print(String(format: "step %d (%.0fs): train throughput %.3f latency %.3f congestion %.3f, weights %.2f %.2f %.2f, %.1f steps/s%@ | ", k + 1, Date().timeIntervalSince(wall),
                 mean[0], mean[1], mean[2], alpha[0], alpha[1], alpha[2], Double(since) / elapsed, nonfinite > 0 ? ", \(nonfinite) NON-FINITE WEIGHTS" : "") + measure())
    fflush(stdout); elapsed = 0; since = 0
  }
}
