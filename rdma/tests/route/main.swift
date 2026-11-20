// The routing policy's rollouts on the GPU (route.metal), v0: random all-gather scenarios, a randomly initialised
// network, and the forward rollout's time. usage: route METALLIB [rollouts a dispatch] [dispatches a command buffer]
import Foundation
import Metal

let N = 6, L = 32, J = 6, K = 8, H = 16, D = 64, E = 192, CTX = 12, VOCAB = (N + 2) * 64
let page = 16384.0

struct Mix { var s: UInt64
  mutating func next() -> UInt64 { s &+= 0x9e3779b97f4a7c15; var z = s; z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9; z = (z ^ (z >> 27)) &* 0x94d049bb133111eb; return z ^ (z >> 31) }
  mutating func uniform() -> Double { Double(next() >> 11) / 9007199254740992.0 }
  mutating func normal() -> Double { sqrt(-2 * log(max(uniform(), 1e-300))) * cos(2 * Double.pi * uniform()) }
}

// one scenario, laid out as route.metal's Scenario (800 bytes)
func scenario(_ g: inout Mix) -> [UInt8] {
  let n = 3 + Int(g.uniform() * 4)
  var joined = Array(repeating: Array(repeating: false, count: n), count: n), degree = Array(repeating: 0, count: n)
  for v in 1..<n { let u = Int(g.uniform() * Double(v)); joined[u][v] = true; joined[v][u] = true; degree[u] += 1; degree[v] += 1 }
  for u in 0..<n { for v in (u + 1)..<n where !joined[u][v] && degree[u] < 4 && degree[v] < 4 && g.uniform() < 0.35 {
    joined[u][v] = true; joined[v][u] = true; degree[u] += 1; degree[v] += 1 } }
  var src: [Int] = [], dst: [Int] = [], rate: [Double] = [], lat: [Double] = []
  for u in 0..<n { for v in (u + 1)..<n where joined[u][v] {
    let c = Double(1 + Int(g.uniform() * 4))
    for (a, b) in [(u, v), (v, u)] { src.append(a); dst.append(b); rate.append(c * (4 + 8 * g.uniform())); lat.append(2e3 + 11e3 * g.uniform()) } } }
  let nl = src.count
  let bg = (0..<nl).map { g.uniform() < 0.3 ? rate[$0] * 50e3 * g.uniform() : 0 }
  let S = exp(log(6144.0) + g.uniform() * (log(3145728.0) - log(6144.0))).rounded(), q = min(S, page)
  // the floor: the cut bound and the slowest pair's shortest delay for one piece
  var cut = 0.0
  for v in 0..<n {
    let inr = (0..<nl).filter { dst[$0] == v }.map { rate[$0] }.reduce(0, +), outr = (0..<nl).filter { src[$0] == v }.map { rate[$0] }.reduce(0, +)
    cut = max(cut, Double(n - 1) * S / inr, S / outr)
  }
  var dist = Array(repeating: Array(repeating: Double.infinity, count: n), count: n)
  for v in 0..<n { dist[v][v] = 0 }
  for i in 0..<nl { dist[src[i]][dst[i]] = min(dist[src[i]][dst[i]], lat[i] + q / rate[i]) }
  for m in 0..<n { for a in 0..<n { for b in 0..<n { dist[a][b] = min(dist[a][b], dist[a][m] + dist[m][b]) } } }
  var reach = 0.0
  for a in 0..<n { for b in 0..<n where a != b { reach = max(reach, dist[a][b]) } }
  let floor = max(cut, reach)
  let dt = max(floor * 6 / 64, (lat.max() ?? 0 + q / (rate.min() ?? 1)) / Double(H - 2))
  // candidate trees: shortest-path trees under unloaded delays (k = 0) and perturbed ones
  var lane = Array(repeating: Array(repeating: Array(repeating: UInt8(255), count: N), count: K), count: J)
  for s in 0..<n {
    for k in 0..<K {
      let w = (0..<nl).map { (lat[$0] + q / rate[$0]) * (k == 0 ? 1 : exp(0.6 * g.normal())) }
      var at = Array(repeating: Double.infinity, count: n), by = Array(repeating: -1, count: n), done = Array(repeating: false, count: n)
      at[s] = 0
      for _ in 0..<n {
        var u = -1
        for v in 0..<n where !done[v] && at[v].isFinite && (u < 0 || at[v] < at[u]) { u = v }
        if u < 0 { break }
        done[u] = true
        for i in 0..<nl where src[i] == u && at[u] + w[i] < at[dst[i]] { at[dst[i]] = at[u] + w[i]; by[dst[i]] = i }
      }
      for v in 0..<n where v != s && by[v] >= 0 { lane[s][k][v] = UInt8(by[v]) }
    }
  }
  var b = [UInt8](repeating: 0, count: 800)
  func f32(_ at: Int, _ v: Double) { var x = Float(v); withUnsafeBytes(of: &x) { for i in 0..<4 { b[at + i] = $0[i] } } }
  func u32(_ at: Int, _ v: Int) { var x = UInt32(v); withUnsafeBytes(of: &x) { for i in 0..<4 { b[at + i] = $0[i] } } }
  for l in 0..<nl { f32(4 * l, rate[l]); f32(128 + 4 * l, lat[l]); f32(256 + 4 * l, bg[l]); b[384 + l] = UInt8(src[l]); b[416 + l] = UInt8(dst[l]) }
  for j in 0..<J { for k in 0..<K { for v in 0..<N { b[448 + (j * K + k) * N + v] = lane[j][k][v] } } }
  for j in 0..<n {
    b[736 + j] = UInt8(j); var mask = 0; for d in 0..<n where d != j { mask |= 1 << d }; b[742 + j] = UInt8(mask)
    f32(748 + 4 * j, S)
  }
  f32(772, floor); f32(776, dt); f32(780, S); u32(784, n); u32(788, nl); u32(792, n)
  return b
}

let args = CommandLine.arguments
let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "rollout")!)
let rollouts = args.count > 2 ? Int(args[2])! : 1024, per = args.count > 3 ? Int(args[3])! : 100

var g = Mix(s: 20261119)
let pool = 4096
var scen = [UInt8](); scen.reserveCapacity(pool * 800)
for _ in 0..<pool { scen += scenario(&g) }
// the network: Gaussian weights scaled by fan-in, biases zero (route.metal's Net)
var net = [Float]()
func fill(_ count: Int, _ fan: Int) { for _ in 0..<count { net.append(Float(g.normal() / sqrt(Double(fan)))) } }
func zero(_ count: Int) { net += [Float](repeating: 0, count: count) }
fill(D * (CTX + D), CTX + D); zero(D)
for _ in 0..<2 { fill(D * 2 * D, 2 * D); fill(D * 2 * D, 2 * D); fill(D * 2 * D, 2 * D); zero(3 * D); fill(E * D, D); zero(E); fill(D * E, E); zero(D) }
fill(VOCAB * D, D)
let which = (0..<rollouts).map { UInt32($0 % pool) }

let scenBuf = device.makeBuffer(bytes: scen, length: scen.count, options: .storageModeShared)!
let netBuf = device.makeBuffer(bytes: net, length: net.count * 4, options: .storageModeShared)!
let whichBuf = device.makeBuffer(bytes: which, length: which.count * 4, options: .storageModeShared)!
let objBuf = device.makeBuffer(length: rollouts * 4, options: .storageModeShared)!
let doneBuf = device.makeBuffer(length: rollouts * 4, options: .storageModeShared)!
let ringBuf = device.makeBuffer(length: rollouts * H * J * K * N * 4, options: .storageModePrivate)!
let queue = device.makeCommandQueue()!
print("network: \(net.count) parameters; \(rollouts) rollouts a dispatch, \(per) dispatches a command buffer; threadgroup memory \(pipeline.staticThreadgroupMemoryLength) bytes")

func run(_ dispatches: Int) -> Double {
  let cb = queue.makeCommandBuffer()!
  let enc = cb.makeComputeCommandEncoder()!
  enc.setComputePipelineState(pipeline)
  enc.setBuffer(scenBuf, offset: 0, index: 0); enc.setBuffer(netBuf, offset: 0, index: 1); enc.setBuffer(whichBuf, offset: 0, index: 2)
  enc.setBuffer(objBuf, offset: 0, index: 3); enc.setBuffer(doneBuf, offset: 0, index: 4); enc.setBuffer(ringBuf, offset: 0, index: 5)
  for _ in 0..<dispatches { enc.dispatchThreadgroups(MTLSize(width: rollouts, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  if let e = cb.error { print("error: \(e)"); exit(1) }
  return cb.gpuEndTime - cb.gpuStartTime
}
_ = run(1)
let one = run(1), many = run(per)
let done = Array(UnsafeBufferPointer(start: doneBuf.contents().bindMemory(to: Float.self, capacity: rollouts), count: rollouts))
let obj = Array(UnsafeBufferPointer(start: objBuf.contents().bindMemory(to: Float.self, capacity: rollouts), count: rollouts))
print(String(format: "one dispatch: %.1f us; %d dispatches: %.1f us each, %.0f rollouts/s", one * 1e6, per, many / Double(per) * 1e6, Double(rollouts * per) / many))
print(String(format: "T / floor: mean %.3f (min %.3f, max %.3f); objective mean %.3f", done.reduce(0, +) / Float(rollouts), done.min()!, done.max()!, obj.reduce(0, +) / Float(rollouts)))
