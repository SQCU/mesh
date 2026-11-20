// The runtime (route.h; main.swift for Swift's top-level code). Each step: W windows, each on a random fabric's random
// stream of collectives, starting at a random point midstream (before it, a random warm router has routed), K refreshes
// `dt` apart with each table active `lag` after its refresh, the policy's state at the window's start drawn at random
// (zero, a scaled Gaussian, or a state some earlier window ended with); four rollouts a window. The environment's
// transactions run on the CPU's cores; every pending refresh across the windows is one forward batch on the GPU, each
// window's state carried; then one learning batch through every window's refreshes. Throughput's advantage is each
// rollout's against the mean of its window's other three; the credit carries the pieces' latency and queueing.
//
// usage: route FABRIC [steps] [windows a step]   (RATE Adam's step, default 1e-3; THETA=path writes the trained
//        parameters there (fp32, the optimizer's master copy), THETA_INT8=path the int8 weights as deployed; THETA_IN=path continues from them, NetLayout's or the earlier fp32 layout's; ROUTE_METALLIB
//        the Metal library, default route.metallib beside the binary)
//   The policy is net.swift's int8 network (minGRU): every 2D product int8 x int8 -> int32 on the GPU's matrix units.
//   SIMCHECK=1  every piece on the shortest-path tree posted at its release, as tests/switch.c's fixed router
//   CHECK=1     the environment's dw against central differences of its log-likelihood in w; GRAPH=1 also the int8
//               network against the fp32 reference (policy.swift, learn.swift: the same function): w, each task's gradient
//   DEPLOY=1    the C forward (forward.c, int8) against the GPU's; its time a refresh
//   BENCH=1     each call's time, kernel by kernel; the environment's
import Foundation
import Metal
import MetalPerformanceShadersGraph

let env = ProcessInfo.processInfo.environment, args = CommandLine.arguments
let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
precondition(env["GRU"] == nil, "the GRU arm has no int8 path (its gates read h: one product a lane); minGRU only")
let lay = Layout(full: false)   // the fp32 reference's layout (CHECK, earlier parameter files)
let nlay = NetLayout()
let arm = "minGRU (int8 products)"
let CONTEXTS = Int(ROUTE_CONTEXTS), KMAX = 4
// data-parallel across machines: RANK of WORLD, rank 0 at LEADER (host:port) summing every rank's three gradients each
// step and returning their mean, so every rank takes the same step from the same theta (its initial seed the same on
// every rank; each rank's windows its own, SEED + RANK)
let rank = Int(env["RANK"] ?? "0")!, world = Int(env["WORLD"] ?? "1")!
var rng = route_rng(s: UInt64(env["SEED"] ?? "20261119")! &+ UInt64(rank) &* 0x9e3779b97f4a7c15)
var thetaRng = route_rng(s: 20261119)
func normal() -> Double { let u = max(route_uniform(&rng), 1e-300), v = route_uniform(&rng); return sqrt(-2 * log(u)) * cos(2 * Double.pi * v) }

var measured = route_fabric()
guard route_fabric_load(args[1], &measured) == 0 else { print("cannot read \(args[1])"); exit(1) }
let steps = args.count > 2 ? Int(args[2])! : 200, windows = args.count > 3 ? Int(args[3])! : 32
let rate = Double(env["RATE"] ?? "1e-3")!
let fabrics: [route_fabric] = (0..<2048).map { _ in var f = route_fabric(); route_network(&rng, &measured, &f); return f }
let kinds = ["all-gather", "all-reduce", "scatter"], sizes: [Double] = [6144, 98304, 3145728, 12582912]

// ---- the runs in flight: a window's stream, its four rollouts ----
let TX = max(4 * windows, kinds.count * sizes.count * 4)
let sims = (0..<TX).map { _ in route_sim_new()! }
let decisions = UnsafeMutablePointer<route_decision>.allocate(capacity: TX)
let tokens = (0..<TX).map { _ in UnsafeMutablePointer<Float>.allocate(capacity: CONTEXTS * IN) }
let laneOffset = MemoryLayout<route_decision>.offset(of: \route_decision.lane)!
func laneTokens(_ t: Int) -> UnsafePointer<Float> { UnsafeRawPointer(decisions + t).advanced(by: laneOffset).assumingMemoryBound(to: Float.self) }
var streams = [route_stream](repeating: route_stream(), count: TX)
var states = [[Float]](repeating: [Float](repeating: 0, count: 2 * D), count: TX)   // each run's carried state
struct Kept { var lanes: [Float]; var lanesUsed: Int; var tokens: [Float]; var contexts: Int }
var kept = [[Kept]](repeating: [], count: TX), start = [[Float]](repeating: [], count: TX)
var pool: [[Float]] = []   // states windows ended with, to start later windows from

// ---- batches, in shared buffers (exact row counts: the int8 network's kernels take any) ----
let ROWS = 1 << 18, BATCH = 1024
func shared(_ floats: Int) -> MTLBuffer { device.makeBuffer(length: floats * 4, options: .storageModeShared)! }
func floats(_ b: MTLBuffer, _ n: Int) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: n) }
let laneBuf = shared(KMAX * BATCH * L * IN), maskBuf = shared(KMAX * BATCH * L), stateBuf = shared(BATCH * 2 * D), nextBuf = shared(BATCH * 2 * D)
let tokBuf = shared(ROWS * IN), idxBuf = shared(ROWS), wBuf = shared(ROWS * FS), dwBuf = shared(TASKS * ROWS * FS)
let laneP = floats(laneBuf, KMAX * BATCH * L * IN), maskP = floats(maskBuf, KMAX * BATCH * L), stateP = floats(stateBuf, BATCH * 2 * D), nextP = floats(nextBuf, BATCH * 2 * D)
let tokP = floats(tokBuf, ROWS * IN), idxP = idxBuf.contents().bindMemory(to: Int32.self, capacity: ROWS), wP = floats(wBuf, ROWS * FS), dwP = floats(dwBuf, TASKS * ROWS * FS)
let library = env["ROUTE_METALLIB"] ?? URL(fileURLWithPath: args[0]).deletingLastPathComponent().appendingPathComponent("route.metallib").path
let net = Net(device, queue, library: library, runs: TX, refreshes: KMAX, contexts: ROWS, initial: nlay.initial(&thetaRng))
func td(_ b: MTLBuffer, _ shape: [Int], _ type: MPSDataType = .float32) -> MPSGraphTensorData { MPSGraphTensorData(b, shape: dims(shape), dataType: type) }

// one refresh's forward for `bd` runs (lanes, masks, states written) and `rows` context rows: w and the next states
var forwardTime = 0.0, forwards = 0, decided = 0, rowsRun = 0, exchangeTime = 0.0
func runForward(_ bd: Int, _ rows: Int) {
  let t0 = Date()
  net.forward(lanes: laneBuf, mask: maskBuf, h0: stateBuf, tokens: tokBuf, pos: idxBuf, B: bd, R: rows, w: wBuf, next: nextBuf)
  forwardTime += Date().timeIntervalSince(t0); forwards += 1; decided += bd; rowsRun += rows
}

// runs `ids` to the end: mode 0 the policy (each pending refresh one batch), else a reference router
var simTime = 0.0
func rollout(_ ids: [Int], window: (Int) -> route_window, mode: Int32, key0: (Int) -> UInt32, key: (Int) -> UInt32, keep: Bool) {
  for t in ids { var s = streams[t], w = window(t); route_sim_begin(sims[t], &s, &w, mode, key0(t), key(t)); kept[t] = []; start[t] = states[t] }
  var active = ids
  while !active.isEmpty {
    let t0 = Date()
    let got = UnsafeMutablePointer<Int32>.allocate(capacity: active.count); defer { got.deallocate() }
    let now = active
    DispatchQueue.concurrentPerform(iterations: now.count) { k in got[k] = route_sim_decide(sims[now[k]], decisions + now[k], tokens[now[k]]) }
    active = (0..<now.count).filter { got[$0] == 1 }.map { now[$0] }
    simTime += Date().timeIntervalSince(t0)
    if active.isEmpty { break }
    var rowOf: [Int] = [], rows = 0
    for (b, t) in active.enumerated() {
      let nl = Int(decisions[t].lanes), nc = Int(decisions[t].contexts)
      memcpy(laneP + b * L * IN, laneTokens(t), L * IN * 4)
      for q in 0..<L { maskP[b * L + q] = q < nl ? 1 : 0 }
      for i in 0..<(2 * D) { stateP[b * 2 * D + i] = states[t][i] }
      memcpy(tokP + rows * IN, tokens[t], nc * IN * 4)
      for c in 0..<nc { idxP[rows + c] = Int32(b * L + L - 1) }   // its state: the refresh's last lane row
      rowOf.append(rows); rows += nc
      if keep { kept[t].append(Kept(lanes: Array(UnsafeBufferPointer(start: laneTokens(t), count: L * IN)), lanesUsed: nl, tokens: Array(UnsafeBufferPointer(start: tokens[t], count: nc * IN)), contexts: nc)) }
    }
    runForward(active.count, rows)
    for (b, t) in active.enumerated() { for i in 0..<(2 * D) { states[t][i] = nextP[b * 2 * D + i] } }
    let t1 = Date(), order = active, offsets = rowOf
    DispatchQueue.concurrentPerform(iterations: order.count) { k in route_sim_answer(sims[order[k]], wP + offsets[k] * FS) }
    simTime += Date().timeIntervalSince(t1)
  }
}
func costs(_ t: Int) -> route_costs { var c = route_costs(); route_sim_costs(sims[t], &c); return c }

// the learning batch over runs `ids` (kept), throughput advantages `adv`: lanes [run][refresh][lane]; drawn-from rows,
// each reading its refresh's state (lane row (b K + k) L + L - 1), contiguous a refresh (seg); each task's dw packed
// [TASKS][R][FS]. Returns (B, K, R, packed, seg)
var learnTime = 0.0, creditTime = 0.0
func gather(_ ids: [Int], _ adv: [Double]) -> (Int, Int, Int, MTLBuffer, [UInt32]) {
  let t0 = Date()
  var credit = [[Float]](repeating: [], count: ids.count)
  DispatchQueue.concurrentPerform(iterations: ids.count) { k in
    let t = ids[k]; var all: [Float] = []
    for (p, d) in kept[t].enumerated() { var part = [Float](repeating: 0, count: TASKS * d.contexts * FS), ll = 0.0; route_sim_credit(sims[t], Int32(p), &part, &ll); all += part }
    credit[k] = all
  }
  creditTime += Date().timeIntervalSince(t0)
  let B = ids.count, K = KMAX
  memset(laneP, 0, B * K * L * IN * 4); memset(maskP, 0, B * K * L * 4); memset(stateP, 0, B * 2 * D * 4)
  var rows = 0, seg = [UInt32](repeating: 0, count: 2 * B * K)
  let scale = 1 / Double(ids.count)
  for (b, t) in ids.enumerated() {
    precondition(kept[t].count <= K, "a window past the learning batch's refreshes")
    for i in 0..<(2 * D) { stateP[b * 2 * D + i] = start[t][i] }
    var at = 0
    for k in 0..<K {
      seg[2 * (b * K + k)] = UInt32(rows)
      if k < kept[t].count {
        let d = kept[t][k]
        memcpy(laneP + (b * K + k) * L * IN, d.lanes, L * IN * 4)
        for q in 0..<L { maskP[(b * K + k) * L + q] = q < d.lanesUsed ? 1 : 0 }
        for c in 0..<d.contexts {
          var used = false
          for task in 0..<TASKS { for f in 0..<FS where credit[b][at + (task * d.contexts + c) * FS + f] != 0 { used = true } }
          if !used { continue }
          precondition(rows < ROWS, "learning rows past the buffers")
          for i in 0..<IN { tokP[rows * IN + i] = d.tokens[c * IN + i] }
          idxP[rows] = Int32((b * K + k) * L + L - 1)
          for task in 0..<TASKS {
            let s = Float((task == 0 ? adv[b] : 1) * scale)
            for f in 0..<FS { dwP[(task * ROWS + rows) * FS + f] = credit[b][at + (task * d.contexts + c) * FS + f] * s }
          }
          rows += 1
        }
        at += TASKS * d.contexts * FS
      }
      seg[2 * (b * K + k) + 1] = UInt32(rows)
    }
  }
  let R = rows
  let packed = shared(TASKS * max(R, 1) * FS), pp = floats(packed, TASKS * max(R, 1) * FS)
  for task in 0..<TASKS { memcpy(pp + task * R * FS, dwP + task * ROWS * FS, R * FS * 4) }
  learnTime += Date().timeIntervalSince(t0)
  return (B, K, R, packed, seg)
}
// ---- the gradients' sum across machines (a star through rank 0, over TCP) ----
var peers: [Int32] = []
func fullSend(_ fd: Int32, _ p: UnsafeRawPointer, _ n: Int) { var at = 0; while at < n { let r = Darwin.send(fd, p + at, n - at, 0); precondition(r > 0, "send to a rank failed"); at += r } }
func fullRecv(_ fd: Int32, _ p: UnsafeMutableRawPointer, _ n: Int) { var at = 0; while at < n { let r = Darwin.recv(fd, p + at, n - at, 0); precondition(r > 0, "a rank went away"); at += r } }
func join() {
  guard world > 1 else { return }
  let parts = (env["LEADER"] ?? "").split(separator: ":"), port = UInt16(parts.count > 1 ? String(parts[1]) : "47611")!
  if rank == 0 {
    let fd = socket(AF_INET, SOCK_STREAM, 0); var one: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, 4)
    var a = sockaddr_in(); a.sin_family = sa_family_t(AF_INET); a.sin_port = port.bigEndian; a.sin_addr.s_addr = INADDR_ANY
    precondition(withUnsafePointer(to: &a) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } } == 0, "bind failed")
    listen(fd, 16)
    for _ in 1..<world { let c = accept(fd, nil, nil); precondition(c >= 0); setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, 4); peers.append(c) }
  } else {
    var hints = addrinfo(); hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM
    var res: UnsafeMutablePointer<addrinfo>? = nil
    precondition(getaddrinfo(String(parts[0]), String(port), &hints, &res) == 0, "cannot resolve the leader")
    var fd: Int32 = -1
    for _ in 0..<600 { fd = socket(AF_INET, SOCK_STREAM, 0); if connect(fd, res!.pointee.ai_addr, res!.pointee.ai_addrlen) == 0 { break }; close(fd); fd = -1; usleep(500_000) }
    precondition(fd >= 0, "cannot reach the leader"); var one: Int32 = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, 4); peers = [fd]
  }
  print("rank \(rank) of \(world): joined"); fflush(stdout)
}
func allMean(_ p: UnsafeMutablePointer<Float>, _ n: Int) {
  guard world > 1 else { return }
  if rank == 0 {
    let part = UnsafeMutablePointer<Float>.allocate(capacity: n); defer { part.deallocate() }
    for fd in peers { fullRecv(fd, part, n * 4); for i in 0..<n { p[i] += part[i] } }
    for i in 0..<n { p[i] /= Float(world) }
    for fd in peers { fullSend(fd, p, n * 4) }
  } else { fullSend(peers[0], p, n * 4); fullRecv(peers[0], p, n * 4) }
}
// the mean gradients' Gram matrix, for UPGrad (each rank's own Gram is of its own gradients)
func gramOfMean() {
  let P = nlay.count, g = floats(net.grads, TASKS * P), G = floats(net.gram, 9)
  for a in 0..<3 { for b in 0..<3 { var d = 0.0; for i in 0..<P { d += Double(g[a * P + i]) * Double(g[b * P + i]) }; G[3 * a + b] = Float(d) } }
}
join()

func gradients(_ B: Int, _ K: Int, _ R: Int, _ packed: MTLBuffer, _ seg: [UInt32]) {
  let t0 = Date()
  net.learn(lanes: laneBuf, mask: maskBuf, h0: stateBuf, tokens: tokBuf, pos: idxBuf, seg: seg, B: B, K: K, R: R, dw: packed, w: wBuf)
  learnTime += Date().timeIntervalSince(t0)
}

// ---- a window: a random fabric's random stream, its start, refreshes, lag, warm router, starting state ----
// the deployment's distribution (ONLINE=1): the measured cluster, one collective at a time of a random kind and size,
// every page posted at its release (tests/switch.c's), refreshed every 100 us and active 100 us late, the state carried
// from the transactions before
let online = env["ONLINE"] != nil
func deploymentWindow() -> (route_stream, route_window, [Float]) {
  var s = route_stream()
  let kind = Int32(Int(route_uniform(&rng) * 2) % 2), size = exp(log(6144.0) + route_uniform(&rng) * (log(12582912.0) - log(6144.0))).rounded()
  route_stream_one(&measured, kind, size, measured.compute.0, &s)
  let w = route_window(t0: 0, dt: 100e3, lag: 100e3, points: Int32(KMAX), warm: 2, burst: 1)
  return (s, w, pool.isEmpty ? [Float](repeating: 0, count: 2 * D) : pool[Int(route_uniform(&rng) * Double(pool.count)) % pool.count])
}
func newWindow() -> (route_stream, route_window, [Float]) {
  if online { return deploymentWindow() }
  var f = fabrics[Int(route_uniform(&rng) * Double(fabrics.count))], s = route_stream()
  let duration = 1e6 * (1 + 4 * route_uniform(&rng)), load = 0.3 + 1.2 * route_uniform(&rng)
  route_stream_make(&f, &rng, duration, load, 1, &s)
  let dt = exp(log(20e3) + route_uniform(&rng) * (log(300e3) - log(20e3)))
  // from 0: the stream's midstream state drawn with it (backlogs, collectives part done), not a replayed warm-up
  let w = route_window(t0: 0, dt: dt, lag: route_uniform(&rng) * dt, points: Int32(1 + Int(route_uniform(&rng) * Double(KMAX)) % KMAX),
                       warm: route_uniform(&rng) < 0.5 ? 2 : 3, burst: route_uniform(&rng) < 0.5 ? 1 : 0)
  var h = [Float](repeating: 0, count: 2 * D)
  let pick = route_uniform(&rng)
  if pick < 1.0 / 3 { let sigma = route_uniform(&rng); for i in 0..<(2 * D) { h[i] = Float(sigma * normal()) } }
  else if pick < 2.0 / 3 && !pool.isEmpty { h = pool[Int(route_uniform(&rng) * Double(pool.count)) % pool.count] }
  return (s, w, h)
}

// ---- evaluation on the measured cluster: one collective at a time, the policy refreshed every 100 us and active 100 us
// after (forward.c's measured refresh on one core), against the shortest-path tree, uniform splits, and switch.c's fixed
// router ----
let deployed = route_window(t0: 0, dt: 100e3, lag: 100e3, points: 0, warm: 2, burst: 0), deployedBurst = route_window(t0: 0, dt: 100e3, lag: 100e3, points: 0, warm: 2, burst: 1)
func measure() -> String {
  var out: [String] = []
  let cases = kinds.indices.flatMap { k in sizes.map { (k, $0) } }
  for t in 0..<(cases.count * 4) { route_stream_one(&measured, Int32(cases[t / 4].0), cases[t / 4].1, measured.compute.0, &streams[t]); states[t] = [Float](repeating: 0, count: 2 * D) }
  var rows: [String: [Double]] = [:]
  for (mode, name, w) in [(Int32(0), "policy", deployed), (0, "policy (burst)", deployedBurst), (2, "spt", deployed), (3, "uniform", deployed), (4, "switch.c fixed", deployed)] {
    rollout(Array(0..<(cases.count * 4)), window: { _ in w }, mode: mode, key0: { UInt32(77 + $0 % 4) }, key: { UInt32(91 + $0 % 4) }, keep: false)
    rows[name] = (0..<cases.count).map { c in (0..<4).map { costs(c * 4 + $0).throughput }.reduce(0, +) / 4 }
  }
  for (k, kind) in kinds.enumerated() {
    var line = "\(kind):"
    for name in ["policy", "policy (burst)", "spt", "uniform", "switch.c fixed"] { line += " \(name) " + (0..<sizes.count).map { String(format: "%.2f", rows[name]![k * sizes.count + $0]) }.joined(separator: " ") + ";" }
    out.append(line)
  }
  return out.joined(separator: " | ")
}

if let path = env["THETA_IN"] {   // continue from these parameters
  let d = try! Data(contentsOf: URL(fileURLWithPath: path))
  if d.count == lay.count * 4 {   // the earlier fp32 layout: reparametrized exactly
    let o = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, t = nlay.from(old: o, lay)
    memcpy(net.theta.contents(), t, t.count * 4)
  } else {
    precondition(d.count == nlay.count * 4, "parameters of another layout")
    d.withUnsafeBytes { memcpy(net.theta.contents(), $0.baseAddress!, nlay.count * 4) }
  }
  net.refresh()
}
if env["SIMCHECK"] != nil {
  print("every piece on the shortest-path tree posted at its release (tests/switch.c's fixed router), the measured cluster, us:")
  for (k, kind) in kinds.enumerated() {
    var line = String(format: "%-10@", kind)
    for (i, S) in sizes.enumerated() {
      route_stream_one(&measured, Int32(k), S, measured.compute.0, &streams[i])
      rollout([i], window: { _ in deployed }, mode: 4, key0: { _ in 1 }, key: { _ in 1 }, keep: false)
      line += String(format: " %9.2f", costs(i).throughput * streams[i].op.0.floor / 1000)
    }
    print(line)
  }
  exit(0)
}

if env["CHECK"] != nil {
  // the environment: channel 0 of its credit is d(log-likelihood of the draws)/dw; against central differences
  // a window whose draws had a choice (retried until one has), its first such refresh
  var nc = 0, dw: [Float] = [], ll = 0.0, point = 0, tries = 0, found = false
  let eps = Float(env["EPS"] ?? "1e-3")!
  while !found {
    var (s, w, h) = newWindow()
    w.points = 3; streams[0] = s; states[0] = h; tries += 1
    rollout([0], window: { _ in w }, mode: 0, key0: { _ in 3 }, key: { _ in UInt32(5 + tries) }, keep: true)
    nc = Int(route_sim_contexts(sims[0])); dw = [Float](repeating: 0, count: TASKS * nc * FS)
    for p in 0..<kept[0].count { route_sim_credit(sims[0], Int32(p), &dw, &ll); point = p; if dw.contains(where: { $0 != 0 }) { found = true; break } }
    let last = kept[0].last.map { $0.contexts } ?? 0
    var big: Float = 0, bad = 0; for i in 0..<(last * FS) { if !wP[i].isFinite { bad += 1 } else { big = max(big, abs(wP[i])) } }
    print(String(format: "window %d: %d contexts, %d refreshes; the last forward's w: largest |w| %.3f, %d not finite; %@", tries, nc, kept[0].count, big, bad, found ? "a choice" : "no choice, another"))
    precondition(tries < 50, "no window with a choice in 50")
  }
  print(String(format: "environment: %d contexts, %d refreshes, log-likelihood of refresh %d's draws %.4f", nc, kept[0].count, point, ll))
  let wv = route_sim_w(sims[0], Int32(point))!
  var shown = 0
  for i in 0..<(nc * FS) where dw[i] != 0 && shown < 6 {
    let keep = wv[i]; var up = 0.0, down = 0.0, scratch = [Float](repeating: 0, count: TASKS * nc * FS)
    wv[i] = keep + eps; route_sim_credit(sims[0], Int32(point), &scratch, &up); wv[i] = keep - eps; route_sim_credit(sims[0], Int32(point), &scratch, &down); wv[i] = keep
    print(String(format: "  w[context %d][%d]: credit %+.5f, central difference %+.5f", i / FS, i % FS, dw[i], (up - down) / Double(2 * eps)))
    shown += 1
  }
  if env["GRAPH"] == nil { exit(0) }
  // the int8 network against the fp32 reference (policy.swift and learn.swift's graph: the same function, plain fp32
  // weights W = g V / |V|), through the window's refreshes: w, and each task's gradient (the reference's dW mapped to
  // (dV, dg, db) by the same chain rule)
  let (B, K, R, packed, seg) = gather([0], [1])
  net.checking = true
  gradients(B, K, R, packed, seg)
  net.checking = false
  print("fused producers' codes against quant_rows of what they stored (bit for bit):"); for line in net.probeReport() { print(line) }
  let P = nlay.count, wInt = Array(UnsafeBufferPointer(start: wP, count: R * FS)), gInt = Array(UnsafeBufferPointer(start: floats(net.grads, TASKS * P), count: TASKS * P))
  let old = nlay.toOld(Array(UnsafeBufferPointer(start: floats(net.theta, P), count: P)), lay), oldBuf = device.makeBuffer(bytes: old, length: old.count * 4, options: .storageModeShared)!
  let refIdx = shared(max(R, 1)), ri = refIdx.contents().bindMemory(to: Int32.self, capacity: max(R, 1))
  for r in 0..<R { ri[r] = idxP[r] / Int32(L) }   // one window: the reference's state index k B + b is k
  let ref = Learn(lay, windows: B, refreshes: K, rows: R), wRef = shared(R * FS), gRef = (0..<TASKS).map { _ in shared(lay.count) }
  var res: [MPSGraphTensor: MPSGraphTensorData] = [ref.w: td(wRef, [R, FS])]
  for t in 0..<TASKS { res[ref.grads[t]] = td(gRef[t], [lay.count]) }
  ref.g.run(with: queue, feeds: [ref.theta: td(oldBuf, [lay.count]), ref.lanes: td(laneBuf, [K, B, L, IN]), ref.mask: td(maskBuf, [K, B, L]), ref.state: td(stateBuf, [B, 2, D]),
                                 ref.tokens: td(tokBuf, [R, IN]), ref.index: td(refIdx, [R], .int32), ref.dw: td(packed, [TASKS, R, FS])], targetOperations: nil, resultsDictionary: res)
  var e2 = 0.0, n2 = 0.0
  let wr = floats(wRef, R * FS)
  for i in 0..<(R * FS) { e2 += Double((wInt[i] - wr[i]) * (wInt[i] - wr[i])); n2 += Double(wr[i] * wr[i]) }
  print(String(format: "int8 network against the fp32 reference: %d refreshes, %d rows; w relative error %.4f", K, R, sqrt(e2 / max(n2, 1e-30))))
  for t in 0..<TASKS {
    let gr = nlay.chain(old: Array(UnsafeBufferPointer(start: floats(gRef[t], lay.count), count: lay.count)), theta: Array(UnsafeBufferPointer(start: floats(net.theta, P), count: P)), lay)
    var line = "  task \(t) gradient, cosine (relative error) by matrix:"
    for mt in nlay.mats {
      let span = mt.v..<(mt.g + mt.rows * (mt.bias ? 2 : 1))
      var dot = 0.0, a2 = 0.0, b2 = 0.0, d2 = 0.0
      for i in span { let a = Double(gInt[t * P + i]), b = Double(gr[i]); dot += a * b; a2 += a * a; b2 += b * b; d2 += (a - b) * (a - b) }
      line += String(format: " %@ %.4f (%.3f)", mt.name, dot / max(sqrt(a2 * b2), 1e-300), sqrt(d2 / max(b2, 1e-300)))
    }
    print(line)
  }
  exit(0)
}

if env["DEPLOY"] != nil {   // the C forward (int8, forward.c) against the GPU's, and its time a refresh
  var params = route_params()
  let stored = NSTemporaryDirectory() + "route-deploy-\(getpid()).i8"   // the weights as deployed: written, then read by forward.c
  net.writeCodes(stored)
  precondition(route_params_read(&params, stored) == 0, "forward.c cannot read the stored weights"); unlink(stored)
  route_stream_one(&measured, 1, 12582912, measured.compute.0, &streams[0])
  var w = deployed; var s = streams[0]
  route_sim_begin(sims[0], &s, &w, 0, 1, 1); _ = route_sim_decide(sims[0], decisions, tokens[0])
  let nl = Int(decisions[0].lanes), nc = Int(decisions[0].contexts)
  memcpy(laneP, laneTokens(0), L * IN * 4); for q in 0..<L { maskP[q] = q < nl ? 1 : 0 }
  memset(stateP, 0, 2 * D * 4); memcpy(tokP, tokens[0], nc * IN * 4); for c in 0..<nc { idxP[c] = Int32(L - 1) }
  runForward(1, nc)
  let scratch = UnsafeMutablePointer<Float>.allocate(capacity: route_forward_scratch(Int32(CONTEXTS))), wc = UnsafeMutablePointer<Float>.allocate(capacity: CONTEXTS * FS)
  var h = [Float](repeating: 0, count: 2 * D)
  route_forward(&params, laneTokens(0), Int32(nl), tokens[0], Int32(nc), &h, wc, scratch)
  var worst: Float = 0, size: Float = 0
  for i in 0..<(nc * FS) { worst = max(worst, abs(wc[i] - wP[i])); size = max(size, abs(wP[i])) }
  var hw: Float = 0; for i in 0..<(2 * D) { hw = max(hw, abs(h[i] - nextP[i])) }
  print(String(format: "C forward (int8) against the GPU's, %d lanes, %d contexts: w largest difference %.1e of %.1e, next state %.1e", nl, nc, worst, size, hw))
  let reps = 2000, t0 = Date()
  for _ in 0..<reps { route_forward(&params, laneTokens(0), Int32(nl), tokens[0], Int32(nc), &h, wc, scratch) }
  print(String(format: "C forward, a refresh of the measured cluster (%d lanes, %d contexts): %.1f us on one core", nl, nc, Date().timeIntervalSince(t0) / Double(reps) * 1e6))
  exit(0)
}

print(online ? "online: continuing on the deployment's distribution" : "offline: random fabrics, streams, windows")
print("\(arm), \(nlay.count) parameters; \(windows) windows x 4 rollouts a step, up to \(KMAX) refreshes a window; T / floor on the measured cluster at \(sizes.map { Int($0) }) bytes, refreshed every 100 us, 100 us late")
if rank == 0 { print("start: " + measure()) }
if env["BENCH"] != nil {   // each part's latency alone, beside the arithmetic it is
  let P = nlay.count, B = 4 * windows
  // int8 operations a row: forward, lane rows (layer 0: gates and feed-forward; layer 1: gates) and context rows (both
  // layers, the readout); the learning call's backward twice the forward's again (dgrad and wgrad)
  let laneOps = 2 * (D * 64 + 2 * D * D + 2 * D * E + 2 * D * D), ctxOps = 2 * (D * 64 + 2 * (2 * D * D + 2 * D * E) + FS * D)
  memset(laneP, 0, B * KMAX * L * IN * 4); memset(maskP, 0, B * KMAX * L * 4); memset(stateP, 0, B * 2 * D * 4)
  for r in [8192, 32768, 131072] {
    for i in 0..<r { idxP[i] = Int32((i % B) * L + L - 1) }
    runForward(B, r)   // first use apart
    var t = 0.0; for _ in 0..<10 { let t0 = Date(); runForward(B, r); t += Date().timeIntervalSince(t0) }
    let ops = Double(B * L * laneOps + r * ctxOps)
    print(String(format: "forward call, %d runs x %6d rows: %7.0f us (one command buffer, int8 products); %.2f G int8 ops, %.1f TOP/s effective", B, r, t / 10 * 1e6, ops / 1e9, ops / (t / 10) / 1e12))
  }
  let packed = shared(TASKS * 65536 * FS); memset(packed.contents(), 0, TASKS * 65536 * FS * 4)
  for r in [4096, 16384, 65536] {
    var seg = [UInt32](repeating: 0, count: 2 * B * KMAX)
    for i in 0..<r { idxP[i] = Int32((i * B * KMAX / r) * L + L - 1) }
    for s in 0..<(B * KMAX) { seg[2 * s] = UInt32(s * r / (B * KMAX)); seg[2 * s + 1] = UInt32((s + 1) * r / (B * KMAX)) }
    gradients(B, KMAX, r, packed, seg)   // first use apart
    var t = 0.0; for _ in 0..<5 { let t0 = Date(); gradients(B, KMAX, r, packed, seg); t += Date().timeIntervalSince(t0) }
    let ops = 3 * Double(B * KMAX * L * laneOps + r * ctxOps)
    print(String(format: "learning call, %d runs x %d refreshes x %6d rows: %7.0f us (forward again, three tasks' backward, Gram); ~%.1f G int8 ops, %.1f TOP/s effective", B, KMAX, r, t / 5 * 1e6, ops / 1e9, ops / (t / 5) / 1e12))
  }
  // one call of each, kernel by kernel (an encoder a kernel; GPU timestamps)
  net.enableProfiling()
  func show(_ what: String) {
    let p = net.profile(), total = p.reduce(0) { $0 + $1.2 }
    print(String(format: "  %@, kernel by kernel on the GPU clock: %.0f us in %d kernels", what, total / 1000, p.reduce(0) { $0 + $1.1 }))
    for (name, n, t) in p.sorted(by: { $0.2 > $1.2 }).prefix(24) { print(String(format: "    %-34@ %3d x %8.1f us = %8.1f us (%4.1f%%)", name, n, t / Double(n) / 1000, t / 1000, 100 * t / total)) }
  }
  do {
    for i in 0..<131072 { idxP[i] = Int32((i % B) * L + L - 1) }
    runForward(B, 131072); show("forward call, 131072 rows")
    var seg = [UInt32](repeating: 0, count: 2 * B * KMAX)
    for i in 0..<65536 { idxP[i] = Int32((i * B * KMAX / 65536) * L + L - 1) }
    for s in 0..<(B * KMAX) { seg[2 * s] = UInt32(s * 65536 / (B * KMAX)); seg[2 * s + 1] = UInt32((s + 1) * 65536 / (B * KMAX)) }
    gradients(B, KMAX, 65536, packed, seg); show("learning call, 65536 rows")
    _ = net.apply(rate: 0); show("optimizer step")
    net.profiling = false
  }
  do { let t0 = Date(); for _ in 0..<20 { _ = net.apply(rate: 0) }; print(String(format: "optimizer step (UPGrad, Adam on %d parameters, V pinned, codes rebuilt): %.0f us", P, Date().timeIntervalSince(t0) / 20 * 1e6)) }
  // the environment on one thread: windows of the training distribution, simulated and credited
  var events: UInt64 = 0, draws: UInt64 = 0, entries: UInt64 = 0, sends: UInt64 = 0, simT = 0.0, credT = 0.0, wins = 0
  let w0 = [Float](repeating: 0.01, count: CONTEXTS * FS)
  for k in 0..<200 {
    let (st, win, _) = newWindow(); var s = st, w = win
    let t0 = Date()
    route_sim_begin(sims[0], &s, &w, 0, 1, UInt32(k))
    while route_sim_decide(sims[0], decisions, tokens[0]) == 1 { route_sim_answer(sims[0], w0) }
    let t1 = Date()
    var part = [Float](repeating: 0, count: TASKS * Int(route_sim_contexts(sims[0])) * FS), ll = 0.0
    for p in 0..<Int(route_sim_points(sims[0])) { route_sim_credit(sims[0], Int32(p), &part, &ll) }
    credT += Date().timeIntervalSince(t1); simT += t1.timeIntervalSince(t0); wins += 1
    var c = [UInt64](repeating: 0, count: 4); route_sim_counts(sims[0], &c); events += c[0]; draws += c[1]; entries += c[2]; sends += c[3]
  }
  print(String(format: "environment, one thread, %d training windows: %.0f us a window simulated (%.0f events, %.0f draws, %.0f table entries each), %.0f us credited", wins, simT / Double(wins) * 1e6,
               Double(events) / Double(wins), Double(draws) / Double(wins), Double(entries) / Double(wins), credT / Double(wins) * 1e6))
  print(String(format: "  per event %.0f ns, per draw %.0f ns, per table entry %.1f ns (simulation); credit %.0f ns a draw", simT / Double(events) * 1e9, simT / Double(max(draws, 1)) * 1e9, simT / Double(max(entries, 1)) * 1e9, credT / Double(max(draws, 1)) * 1e9))
  print(String(format: "  so %d threads: %.0f windows a second", ProcessInfo.processInfo.activeProcessorCount, Double(ProcessInfo.processInfo.activeProcessorCount) * Double(wins) / (simT + credT)))
  exit(0)
}
var counted = [UInt64](repeating: 0, count: 4), transactions = 0
let wall = Date()
var stepTime = 0.0, since = 0
for k in 0..<steps {
  let t0 = Date()
  var keys0: [UInt32] = [], windowsOf: [route_window] = []
  for b in 0..<windows {
    let (s, w, h) = newWindow()
    windowsOf.append(w); keys0.append(UInt32(truncatingIfNeeded: route_next(&rng)))
    for q in 0..<4 { streams[4 * b + q] = s; states[4 * b + q] = h }
  }
  let ids = Array(0..<(4 * windows)), keys = ids.map { _ in UInt32(truncatingIfNeeded: route_next(&rng)) }
  rollout(ids, window: { windowsOf[$0 / 4] }, mode: 0, key0: { keys0[$0 / 4] }, key: { keys[$0] }, keep: true)
  for t in ids where pool.count < 4096 || route_uniform(&rng) < 0.05 { if pool.count < 4096 { pool.append(states[t]) } else { pool[Int(route_uniform(&rng) * 4096) % 4096] = states[t] } }
  let T = ids.map { costs($0).throughput }
  let adv = ids.map { t -> Double in let s = t / 4 * 4, total = T[s] + T[s + 1] + T[s + 2] + T[s + 3]; return T[t] - (total - T[t]) / 3 }
  let (B, K, R, packed, seg) = gather(ids, adv)
  gradients(B, K, R, packed, seg)
  let tx0 = Date()
  if world > 1 { allMean(floats(net.grads, TASKS * nlay.count), TASKS * nlay.count); gramOfMean() }
  exchangeTime += Date().timeIntervalSince(tx0)
  var n4 = [UInt64](repeating: 0, count: 4)
  for t in ids { var c = [UInt64](repeating: 0, count: 4); route_sim_counts(sims[t], &c); for q in 0..<4 { n4[q] += c[q] } }
  for q in 0..<4 { counted[q] += n4[q] }; transactions += ids.count
  let alpha = net.apply(rate: rate)
  stepTime += Date().timeIntervalSince(t0); since += 1
  if k < 3 || (k + 1) % 50 == 0 || k + 1 == steps {
    let c = ids.map { costs($0) }
    var mean = [0.0, 0.0, 0.0]
    for x in c { mean[0] += x.throughput / Double(c.count); mean[1] += x.latency / Double(c.count); mean[2] += x.queueing / Double(c.count) }
    print(String(format: "step %d (%.0fs): train throughput %.3f latency %.3f queueing %.3f, weights %.2f %.2f %.2f, %.2f steps/s (simulate %.0f%%, forward %.0f%%, credit %.0f%%, learn %.0f%%)",
                 k + 1, Date().timeIntervalSince(wall), mean[0], mean[1], mean[2], alpha[0], alpha[1], alpha[2], Double(since) / stepTime,
                 100 * simTime / stepTime, 100 * forwardTime / stepTime, 100 * creditTime / stepTime, 100 * (learnTime - creditTime) / stepTime))
    let sec = stepTime
    print(String(format: "  this rank, per second: %.0f transactions, %.0f policy evaluations in %.1f forward batches (%.0f context rows), %.0f events, %.0f sends, %.0f draws, %.0f table entries (40-wide dot products); exchange %.0f%%",
                 Double(transactions) / sec, Double(decided) / sec, Double(forwards) / sec, Double(rowsRun) / sec, Double(counted[0]) / sec, Double(counted[3]) / sec, Double(counted[1]) / sec, Double(counted[2]) / sec, 100 * exchangeTime / sec))
    counted = [0, 0, 0, 0]; transactions = 0; forwards = 0; decided = 0; rowsRun = 0; exchangeTime = 0
    if rank == 0 && ((k + 1) % 100 == 0 || k + 1 == steps) { print("  measured cluster: " + measure()) }
    fflush(stdout); stepTime = 0; since = 0; simTime = 0; forwardTime = 0; learnTime = 0; creditTime = 0
  }
}
if let path = env["THETA"], rank == 0 { try! Data(bytes: net.theta.contents(), count: nlay.count * 4).write(to: URL(fileURLWithPath: path)); print("parameters written to \(path)") }
if let path = env["THETA_INT8"], rank == 0 { net.writeCodes(path); print("int8 weights (for deployment: tests/switch.c's ROUTE_THETA) written to \(path)") }
