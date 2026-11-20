// The routing policy trained on the GPU (route.metal) by evolution strategies and two-task Jacobian descent, over
// randomised network kinds, and measured on the five-node cluster as measured (tests/fabric-20261119.txt).
//
// Each step: M antithetic pairs, each pair one scenario drawn from the pool and run at theta + sigma eps and
// theta - sigma eps (eps a sign a parameter, from the pair's key); two tasks, the smooth area objective and the true
// completion over the floor; each task's gradient from each pair's two sides' difference over their mean (route.metal
// gradient);
// the two combined at their min-norm point (MGDA for two) and applied by Adam.
//
// The pool's network kinds: random connected graphs, rings, stars, rings with a relay hub, full meshes, and the measured
// five-node cluster with its rates and latencies jittered; 3 to 6 nodes, relays only where the kind has them; 1 to 4
// cables a pair, each direction 4 to 12 GB/s a cable and 2 to 13 us; background bytes on some lanes; an all-gather of
// 6 KB to 12.6 MB a rank.
//
// usage: route METALLIB FABRIC [steps] [pairs a step]
import Foundation
import Metal

let N = 6, L = 32, J = 6, K = 8, H = 12, D = 64, E = 192, CTX = 12, VOCAB = (N + 2) * 8 * 16
let page = 16384.0, scenarioBytes = 1444

struct Mix { var s: UInt64
  mutating func next() -> UInt64 { s &+= 0x9e3779b97f4a7c15; var z = s; z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9; z = (z ^ (z >> 27)) &* 0x94d049bb133111eb; return z ^ (z >> 31) }
  mutating func uniform() -> Double { Double(next() >> 11) / 9007199254740992.0 }
  mutating func normal() -> Double { sqrt(-2 * log(max(uniform(), 1e-300))) * cos(2 * Double.pi * uniform()) }
}

struct Fabric { var n: Int; var compute: [Int]; var src: [Int]; var dst: [Int]; var rate: [Double]; var lat: [Double] }

// a scenario, laid out as route.metal's Scenario: the fabric, background bytes, an all-gather of S a rank among the
// compute nodes, the floor, the slot width, and K candidate trees a commodity (shortest-path trees under unloaded
// delays, k = 0, and under randomly perturbed ones), pruned to the nodes that lead to a destination
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
  var lane = Array(repeating: Array(repeating: Array(repeating: UInt8(255), count: N), count: K), count: J)
  for (j, s) in f.compute.enumerated() {
    for k in 0..<K {
      let w = (0..<nl).map { (f.lat[$0] + q / f.rate[$0]) * (k == 0 ? 1 : exp(0.6 * g.normal())) }
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
      for v in 0..<n where keep[v] { lane[j][k][v] = UInt8(by[v]) }
    }
  }
  var b = [UInt8](repeating: 0, count: scenarioBytes)
  func f32(_ at: Int, _ v: Double) { var x = Float(v); withUnsafeBytes(of: &x) { for i in 0..<4 { b[at + i] = $0[i] } } }
  func u32(_ at: Int, _ v: Int) { var x = UInt32(v); withUnsafeBytes(of: &x) { for i in 0..<4 { b[at + i] = $0[i] } } }
  func u16(_ at: Int, _ v: Int) { b[at] = UInt8(v & 255); b[at + 1] = UInt8(v >> 8) }
  for l in 0..<nl { f32(4 * l, f.rate[l]); f32(128 + 4 * l, f.lat[l]); f32(256 + 4 * l, bg[l]); b[384 + l] = UInt8(f.src[l]); b[416 + l] = UInt8(f.dst[l]) }
  for j in 0..<J { for k in 0..<K { for v in 0..<N { b[448 + (j * K + k) * N + v] = lane[j][k][v] } } }
  for (j, s) in f.compute.enumerated() {
    b[736 + j] = UInt8(s); var mask = 0; for d in f.compute where d != s { mask |= 1 << d }; b[742 + j] = UInt8(mask)
    f32(748 + 4 * j, S)
  }
  f32(772, floor); f32(776, dt); f32(780, S); u32(784, nc); u32(788, nl); u32(792, n)
  var edges: [[Int]] = Array(repeating: [], count: L)
  for j in 0..<J { for k in 0..<K { for v in 0..<N where lane[j][k][v] != 255 { edges[Int(lane[j][k][v])].append((j * K + k) * N + v) } } }
  var at = 0
  for l in 0..<L { u16(800 + 2 * l, at); for e in edges[l] { u16(866 + 2 * at, e); at += 1 } }
  u16(800 + 2 * L, at)
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
// one network of a random kind
func network(_ g: inout Mix, _ measured: Fabric) -> Fabric {
  let kind = Int(g.uniform() * 6), n = 3 + Int(g.uniform() * 4)
  var pairs: [(Int, Int)] = []
  switch kind {
  case 0:   // a random connected graph, up to 2 relays
    var joined = Set<Int>(), degree = Array(repeating: 0, count: n)
    for v in 1..<n { let u = Int(g.uniform() * Double(v)); pairs.append((u, v)); joined.insert(u * 8 + v); degree[u] += 1; degree[v] += 1 }
    for u in 0..<n { for v in (u + 1)..<n where !joined.contains(u * 8 + v) && degree[u] < 4 && degree[v] < 4 && g.uniform() < 0.35 { pairs.append((u, v)); degree[u] += 1; degree[v] += 1 } }
    let relays = min(Int(g.uniform() * 3), n - 2)
    return cabled(n, Array(0..<(n - relays)), pairs, &g)
  case 1:   // a ring
    for v in 0..<n { pairs.append((v, (v + 1) % n)) }
    if n == 3 { pairs.removeLast() }
    return cabled(n, Array(0..<n), pairs, &g)
  case 2:   // a star, its hub a relay or a compute node
    for v in 1..<n { pairs.append((0, v)) }
    return cabled(n, g.uniform() < 0.5 ? Array(1..<n) : Array(0..<n), pairs, &g)
  case 3:   // a ring of compute nodes with a relay hub joined to up to three of them
    let m = min(max(n - 1, 3), N - 1)
    for v in 0..<m { pairs.append((v, (v + 1) % m)) }
    for v in 0..<min(3, m) { pairs.append((m, v)) }
    return cabled(m + 1, Array(0..<m), pairs, &g)
  case 4:   // a full mesh
    for u in 0..<n { for v in (u + 1)..<n { pairs.append((u, v)) } }
    return cabled(n, Array(0..<n), pairs, &g)
  default:  // the measured cluster, its rates and latencies jittered
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
    if w.first == "link" {
      let a = names.firstIndex(of: w[1])!, b = names.firstIndex(of: w[2])!
      f.src.append(a); f.dst.append(b); f.rate.append(Double(w[3])!); f.lat.append(Double(w[4])! * 1e3)
    }
  }
  f.n = names.count; f.compute = compute
  return f
}

let args = CommandLine.arguments
let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
let rolloutPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "rollout")!)
let gradientPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "gradient")!)
let measured = load(args[2])
let steps = args.count > 3 ? Int(args[3])! : 300, M = args.count > 4 ? Int(args[4])! : 2048
let sigma: Float = 0.02, rate = 3e-3

var g = Mix(s: 20261119)
let pool = 16384
var scen = [UInt8](); scen.reserveCapacity((pool + 64) * scenarioBytes)
for _ in 0..<pool { let f = network(&g, measured); let S = exp(log(6144.0) + g.uniform() * (log(12582912.0) - log(6144.0))).rounded(); scen += scenario(f, S, &g, background: true) }
let sizes = [2, 8, 32, 128, 1024, 4096].map { Double($0 * 1536 * 2) }
for S in sizes { scen += scenario(measured, S, &g, background: false) }   // the measured cluster, held out, at decode sizes
var net = [Float]()
func fill(_ count: Int, _ fan: Int) { for _ in 0..<count { net.append(Float(g.normal() / sqrt(Double(fan)))) } }
func zero(_ count: Int) { net += [Float](repeating: 0, count: count) }
fill(D * (CTX + D), CTX + D); zero(D)
for _ in 0..<2 { fill(D * 2 * D, 2 * D); fill(D * 2 * D, 2 * D); fill(D * 2 * D, 2 * D); zero(3 * D); fill(E * D, D); zero(E); fill(D * E, E); zero(D) }
fill(VOCAB * D, D)
let P = net.count

let scenBuf = device.makeBuffer(bytes: scen, length: scen.count, options: .storageModeShared)!
let netBuf = device.makeBuffer(bytes: net, length: P * 4, options: .storageModeShared)!
let R = 2 * M
let whichBuf = device.makeBuffer(length: R * 4, options: .storageModeShared)!, perturbBuf = device.makeBuffer(length: R * 8, options: .storageModeShared)!
let objBuf = device.makeBuffer(length: R * 4, options: .storageModeShared)!, doneBuf = device.makeBuffer(length: R * 4, options: .storageModeShared)!
let weightBuf = device.makeBuffer(length: M * 8, options: .storageModeShared)!, keyBuf = device.makeBuffer(length: M * 4, options: .storageModeShared)!
let gradBuf = device.makeBuffer(length: 2 * P * 4, options: .storageModeShared)!
let queue = device.makeCommandQueue()!
let which = whichBuf.contents().bindMemory(to: UInt32.self, capacity: R), perturb = perturbBuf.contents().bindMemory(to: UInt32.self, capacity: 2 * R)
let obj = objBuf.contents().bindMemory(to: Float.self, capacity: R), done = doneBuf.contents().bindMemory(to: Float.self, capacity: R)
let weights = weightBuf.contents().bindMemory(to: Float.self, capacity: 2 * M), keys = keyBuf.contents().bindMemory(to: UInt32.self, capacity: M)
let grad = gradBuf.contents().bindMemory(to: Float.self, capacity: 2 * P), theta = netBuf.contents().bindMemory(to: Float.self, capacity: P)

func rollouts(_ count: Int, amplitude: Float, mode: UInt32) {
  let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(rolloutPipeline)
  enc.setBuffer(scenBuf, offset: 0, index: 0); enc.setBuffer(netBuf, offset: 0, index: 1); enc.setBuffer(whichBuf, offset: 0, index: 2)
  enc.setBuffer(objBuf, offset: 0, index: 3); enc.setBuffer(doneBuf, offset: 0, index: 4); enc.setBuffer(perturbBuf, offset: 0, index: 5)
  var a = amplitude, m = mode
  enc.setBytes(&a, length: 4, index: 6); enc.setBytes(&m, length: 4, index: 7)
  enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  if let e = cb.error { print("error: \(e)"); exit(1) }
}
// the measured cluster at the decode sizes: the policy, all on the shortest-path tree, a uniform split
func measure() -> String {
  var out: [String] = []
  for (mode, name) in [(UInt32(0), "policy"), (1, "spt"), (2, "uniform")] {
    for i in 0..<sizes.count { which[i] = UInt32(pool + i); perturb[2 * i] = 0; perturb[2 * i + 1] = 0 }
    rollouts(sizes.count, amplitude: 0, mode: mode)
    out.append(name + " " + (0..<sizes.count).map { String(format: "%.2f", done[$0]) }.joined(separator: " "))
  }
  return out.joined(separator: " | ")
}
func ranks(_ v: [Float]) -> [Float] {
  let order = v.indices.sorted { v[$0] < v[$1] }; var r = [Float](repeating: 0, count: v.count)
  for (i, j) in order.enumerated() { r[j] = Float(i) / Float(v.count - 1) - 0.5 }
  return r
}

print("network \(P) parameters; \(M) pairs a step; T / floor on the measured cluster at \(sizes.map { Int($0) }) bytes a rank")
print("start: " + measure())
var m1 = [Double](repeating: 0, count: P), m2 = [Double](repeating: 0, count: P)
var rolloutTime = 0.0, stepTime = 0.0
for k in 0..<steps {
  let t0 = Date()
  for m in 0..<M {
    keys[m] = UInt32(truncatingIfNeeded: g.next()); let sc = UInt32(Int(g.uniform() * Double(pool)))
    which[2 * m] = sc; which[2 * m + 1] = sc
    perturb[4 * m] = keys[m]; perturb[4 * m + 1] = 0; perturb[4 * m + 2] = keys[m]; perturb[4 * m + 3] = 1
  }
  let t1 = Date()
  rollouts(R, amplitude: sigma, mode: 0)
  rolloutTime += Date().timeIntervalSince(t1)
  // each pair scored on its own scenario: its two sides' difference over their mean, for each task
  for m in 0..<M {
    let oa = obj[2 * m], ob = obj[2 * m + 1], da = done[2 * m], db = done[2 * m + 1]
    weights[2 * m] = (oa - ob) / max((oa + ob) / 2, 1e-6) / (2 * sigma * Float(M))
    weights[2 * m + 1] = (da - db) / max((da + db) / 2, 1e-6) / (2 * sigma * Float(M))
  }
  let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(gradientPipeline)
  enc.setBuffer(weightBuf, offset: 0, index: 0); enc.setBuffer(keyBuf, offset: 0, index: 1); enc.setBuffer(gradBuf, offset: 0, index: 2)
  var pm = UInt32(M), pp = UInt32(P)
  enc.setBytes(&pm, length: 4, index: 3); enc.setBytes(&pp, length: 4, index: 4)
  enc.dispatchThreads(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  var g11 = 0.0, g22 = 0.0, g12 = 0.0
  for i in 0..<P { let x = Double(grad[i]), y = Double(grad[P + i]); g11 += x * x; g22 += y * y; g12 += x * y }
  let den = g11 - 2 * g12 + g22
  let lambda = den > 0 ? min(max((g22 - g12) / den, 0), 1) : 0.5
  for i in 0..<P {   // Adam on the min-norm combination
    let d = lambda * Double(grad[i]) + (1 - lambda) * Double(grad[P + i])
    m1[i] = 0.9 * m1[i] + 0.1 * d; m2[i] = 0.999 * m2[i] + 0.001 * d * d
    theta[i] -= Float(rate * (m1[i] / (1 - pow(0.9, Double(k + 1)))) / (sqrt(m2[i] / (1 - pow(0.999, Double(k + 1)))) + 1e-8))
  }
  stepTime += Date().timeIntervalSince(t0)
  if k < 10 || k % 25 == 24 || k + 1 == steps {
    let meanDone = (0..<R).map { done[$0] }.reduce(0, +) / Float(R)
    print(String(format: "step %d: train T/floor %.3f, lambda %.2f, %.1f steps/s, %.0f rollouts/s | ", k + 1, meanDone, lambda,
                 Double(k < 10 ? 1 : 25) / stepTime, Double((k < 10 ? 1 : 25) * R) / rolloutTime) + measure())
    fflush(stdout)
    rolloutTime = 0; stepTime = 0
  }
}
