// The runtime (route.h; main.swift for Swift's top-level code). Each step: W windows, each on a random fabric's random
// stream of collectives, starting at a random point midstream (before it, a random warm router has routed), K refreshes
// `dt` apart with each table active `lag` after its refresh, the policy's state at the window's start drawn at random
// (zero, a scaled Gaussian, or a state some earlier window ended with); four rollouts a window. The environment's
// transactions run on the CPU's cores; every pending refresh across the windows is one forward batch on the GPU, each
// window's state carried; then one learning batch through every window's refreshes. Throughput's advantage is each
// rollout's against the mean of its window's other three; the credit carries the pieces' latency and queueing.
//
// usage: route FABRIC [steps] [windows a step]   (GRU=1 the GRU arm, else minGRU; RATE Adam's step, default 1e-3;
//        THETA=path writes the trained parameters there)
//   SIMCHECK=1  every piece on the shortest-path tree posted at its release, as tests/switch.c's fixed router
//   CHECK=1     the environment's dw against central differences of its log-likelihood in w; the graph's gradient
//               against central differences of sum(w . dw) in theta, through a window's refreshes
//   DEPLOY=1    the C forward against the graph's; its time a refresh
import Foundation
import Metal
import MetalPerformanceShadersGraph

let env = ProcessInfo.processInfo.environment, args = CommandLine.arguments
let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
let lay = Layout(full: env["GRU"] != nil)
let arm = lay.full ? "GRU" : "minGRU"
let CONTEXTS = Int(ROUTE_CONTEXTS), KMAX = 4
var rng = route_rng(s: UInt64(env["SEED"] ?? "20261119")!)
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

// ---- batches, in shared buffers; graphs built once a shape ----
func bucket(_ n: Int, _ least: Int) -> Int { var b = least; while b < n { b *= 2 }; return b }
let ROWS = 1 << 18, BATCH = 1024
func shared(_ floats: Int) -> MTLBuffer { device.makeBuffer(length: floats * 4, options: .storageModeShared)! }
func floats(_ b: MTLBuffer, _ n: Int) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: n) }
let laneBuf = shared(KMAX * BATCH * L * IN), maskBuf = shared(KMAX * BATCH * L), stateBuf = shared(BATCH * 2 * D), nextBuf = shared(BATCH * 2 * D)
let tokBuf = shared(ROWS * IN), idxBuf = shared(ROWS), wBuf = shared(ROWS * FS), dwBuf = shared(TASKS * ROWS * FS)
let laneP = floats(laneBuf, KMAX * BATCH * L * IN), maskP = floats(maskBuf, KMAX * BATCH * L), stateP = floats(stateBuf, BATCH * 2 * D), nextP = floats(nextBuf, BATCH * 2 * D)
let tokP = floats(tokBuf, ROWS * IN), idxP = idxBuf.contents().bindMemory(to: Int32.self, capacity: ROWS), wP = floats(wBuf, ROWS * FS), dwP = floats(dwBuf, TASKS * ROWS * FS)
var policies: [Int: Policy] = [:], learners: [Int: Learn] = [:]
let opt = Optimizer(device, queue, initial: lay.initial(&rng))
func td(_ b: MTLBuffer, _ shape: [Int], _ type: MPSDataType = .float32) -> MPSGraphTensorData { MPSGraphTensorData(b, shape: dims(shape), dataType: type) }

// one refresh's forward for `bd` runs (lanes, masks, states written) and `rows` context rows: w and the next states
var forwardTime = 0.0
func runForward(_ bd: Int, _ rows: Int) {
  let B = bucket(bd, 16), R = bucket(rows, 1024)
  precondition(B <= BATCH && R <= ROWS, "a batch past its buffers")
  for i in rows..<R { idxP[i] = 0 }; memset(tokP + rows * IN, 0, (R - rows) * IN * 4); memset(maskP + bd * L, 0, (B - bd) * L * 4); memset(stateP + bd * 2 * D, 0, (B - bd) * 2 * D * 4)
  let key = B << 32 | R
  if policies[key] == nil { policies[key] = Policy(lay, decisions: B, rows: R) }
  let p = policies[key]!, t0 = Date()
  p.g.run(with: queue, feeds: [p.theta: td(opt.theta, [lay.count]), p.lanes: td(laneBuf, [B, L, IN]), p.mask: td(maskBuf, [B, L]), p.state: td(stateBuf, [B, 2, D]),
                               p.tokens: td(tokBuf, [R, IN]), p.index: td(idxBuf, [R], .int32)], targetOperations: nil,
          resultsDictionary: [p.w: td(wBuf, [R, FS]), p.next: td(nextBuf, [B, 2, D])])
  forwardTime += Date().timeIntervalSince(t0)
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
      for c in 0..<nc { idxP[rows + c] = Int32(b) }
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

// the learning batch over runs `ids` (kept), throughput advantages `adv`: drawn-from rows, each reading its refresh's
// state (k B + b); each task's dw packed [TASKS][R][FS]. Returns (B, K, R, packed, rows)
var learnTime = 0.0, creditTime = 0.0
func gather(_ ids: [Int], _ adv: [Double]) -> (Int, Int, Int, MTLBuffer, Int) {
  let t0 = Date()
  var credit = [[Float]](repeating: [], count: ids.count)
  DispatchQueue.concurrentPerform(iterations: ids.count) { k in
    let t = ids[k]; var all: [Float] = []
    for (p, d) in kept[t].enumerated() { var part = [Float](repeating: 0, count: TASKS * d.contexts * FS), ll = 0.0; route_sim_credit(sims[t], Int32(p), &part, &ll); all += part }
    credit[k] = all
  }
  creditTime += Date().timeIntervalSince(t0)
  let B = bucket(ids.count, 16), K = max(1, ids.map { kept[$0].count }.max()!)
  precondition(B <= BATCH && K <= KMAX, "a learning batch past its buffers")
  memset(laneP, 0, K * B * L * IN * 4); memset(maskP, 0, K * B * L * 4); memset(stateP, 0, B * 2 * D * 4)
  var rows = 0
  let scale = 1 / Double(ids.count)
  for (b, t) in ids.enumerated() {
    for i in 0..<(2 * D) { stateP[b * 2 * D + i] = start[t][i] }
    var at = 0
    for (k, d) in kept[t].enumerated() {
      memcpy(laneP + (k * B + b) * L * IN, d.lanes, L * IN * 4)
      for q in 0..<L { maskP[(k * B + b) * L + q] = q < d.lanesUsed ? 1 : 0 }
      for c in 0..<d.contexts {
        var used = false
        for task in 0..<TASKS { for f in 0..<FS where credit[b][at + (task * d.contexts + c) * FS + f] != 0 { used = true } }
        if !used { continue }
        precondition(rows < ROWS, "learning rows past their buffer")
        for i in 0..<IN { tokP[rows * IN + i] = d.tokens[c * IN + i] }
        idxP[rows] = Int32(k * B + b)
        for task in 0..<TASKS {
          let s = Float((task == 0 ? adv[b] : 1) * scale)
          for f in 0..<FS { dwP[(task * ROWS + rows) * FS + f] = credit[b][at + (task * d.contexts + c) * FS + f] * s }
        }
        rows += 1
      }
      at += TASKS * d.contexts * FS
    }
  }
  let R = bucket(rows, 1024)
  for i in rows..<R { idxP[i] = 0 }; memset(tokP + rows * IN, 0, (R - rows) * IN * 4)
  let packed = shared(TASKS * R * FS), pp = floats(packed, TASKS * R * FS)
  for task in 0..<TASKS { memcpy(pp + task * R * FS, dwP + task * ROWS * FS, rows * FS * 4); memset(pp + task * R * FS + rows * FS, 0, (R - rows) * FS * 4) }
  learnTime += Date().timeIntervalSince(t0)
  return (B, K, R, packed, rows)
}
func learner(_ B: Int, _ K: Int, _ R: Int) -> Learn {
  let key = (B << 40) | (K << 32) | R
  if learners[key] == nil { learners[key] = Learn(lay, windows: B, refreshes: K, rows: R) }
  return learners[key]!
}
func learnFeeds(_ l: Learn, _ B: Int, _ K: Int, _ R: Int, _ packed: MTLBuffer) -> [MPSGraphTensor: MPSGraphTensorData] {
  [l.theta: td(opt.theta, [lay.count]), l.lanes: td(laneBuf, [K, B, L, IN]), l.mask: td(maskBuf, [K, B, L]), l.state: td(stateBuf, [B, 2, D]),
   l.tokens: td(tokBuf, [R, IN]), l.index: td(idxBuf, [R], .int32), l.dw: td(packed, [TASKS, R, FS])]
}
func gradients(_ B: Int, _ K: Int, _ R: Int, _ packed: MTLBuffer) {
  let t0 = Date(), l = learner(B, K, R)
  var results: [MPSGraphTensor: MPSGraphTensorData] = [l.gram: td(opt.gram, [TASKS * TASKS])]
  let gradBufs = (0..<TASKS).map { _ in shared(lay.count) }
  for t in 0..<TASKS { results[l.grads[t]] = td(gradBufs[t], [lay.count]) }
  l.g.run(with: queue, feeds: learnFeeds(l, B, K, R, packed), targetOperations: nil, resultsDictionary: results)
  let gp = floats(opt.grads, TASKS * lay.count)
  for t in 0..<TASKS { memcpy(gp + t * lay.count, gradBufs[t].contents(), lay.count * 4) }
  learnTime += Date().timeIntervalSince(t0)
}

// ---- a window: a random fabric's random stream, its start, refreshes, lag, warm router, starting state ----
func newWindow() -> (route_stream, route_window, [Float]) {
  var f = fabrics[Int(route_uniform(&rng) * Double(fabrics.count))], s = route_stream()
  let duration = 1e6 * (1 + 4 * route_uniform(&rng)), load = 0.3 + 1.2 * route_uniform(&rng)
  route_stream_make(&f, &rng, duration, load, 1, &s)
  let dt = exp(log(20e3) + route_uniform(&rng) * (log(300e3) - log(20e3)))
  let w = route_window(t0: route_uniform(&rng) * 0.6 * duration, dt: dt, lag: route_uniform(&rng) * dt, points: Int32(1 + Int(route_uniform(&rng) * Double(KMAX)) % KMAX), warm: route_uniform(&rng) < 0.5 ? 2 : 3)
  var h = [Float](repeating: 0, count: 2 * D)
  let pick = route_uniform(&rng)
  if pick < 1.0 / 3 { let sigma = route_uniform(&rng); for i in 0..<(2 * D) { h[i] = Float(sigma * normal()) } }
  else if pick < 2.0 / 3 && !pool.isEmpty { h = pool[Int(route_uniform(&rng) * Double(pool.count)) % pool.count] }
  return (s, w, h)
}

// ---- evaluation on the measured cluster: one collective at a time, the policy refreshed every 100 us and active 100 us
// after (forward.c's measured refresh on one core), against the shortest-path tree, uniform splits, and switch.c's fixed
// router ----
let deployed = route_window(t0: 0, dt: 100e3, lag: 100e3, points: 0, warm: 2)
func measure() -> String {
  var out: [String] = []
  let cases = kinds.indices.flatMap { k in sizes.map { (k, $0) } }
  for t in 0..<(cases.count * 4) { route_stream_one(&measured, Int32(cases[t / 4].0), cases[t / 4].1, measured.compute.0, &streams[t]); states[t] = [Float](repeating: 0, count: 2 * D) }
  var rows: [String: [Double]] = [:]
  for (mode, name) in [(Int32(0), "policy"), (2, "spt"), (3, "uniform"), (4, "switch.c fixed")] {
    rollout(Array(0..<(cases.count * 4)), window: { _ in deployed }, mode: mode, key0: { UInt32(77 + $0 % 4) }, key: { UInt32(91 + $0 % 4) }, keep: false)
    rows[name] = (0..<cases.count).map { c in (0..<4).map { costs(c * 4 + $0).throughput }.reduce(0, +) / 4 }
  }
  for (k, kind) in kinds.enumerated() {
    var line = "\(kind):"
    for name in ["policy", "spt", "uniform", "switch.c fixed"] { line += " \(name) " + (0..<sizes.count).map { String(format: "%.2f", rows[name]![k * sizes.count + $0]) }.joined(separator: " ") + ";" }
    out.append(line)
  }
  return out.joined(separator: " | ")
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
  var (s, w, h) = newWindow()
  w.points = 3; streams[0] = s; states[0] = h
  rollout([0], window: { _ in w }, mode: 0, key0: { _ in 3 }, key: { _ in 5 }, keep: true)
  let nc = Int(route_sim_contexts(sims[0])), eps = Float(env["EPS"] ?? "1e-3")!
  var dw = [Float](repeating: 0, count: TASKS * nc * FS), ll = 0.0, point = 0
  for p in 0..<kept[0].count { route_sim_credit(sims[0], Int32(p), &dw, &ll); point = p; if dw.contains(where: { $0 != 0 }) { break } }   // the first refresh whose draws had a choice
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
  // the graph, through the window's refreshes: d sum(w . dw_0)/d theta against central differences
  let (B, K, R, packed, rows) = gather([0], [1])
  gradients(B, K, R, packed)
  let pk = floats(packed, TASKS * R * FS), th = floats(opt.theta, lay.count), gp = floats(opt.grads, TASKS * lay.count)
  let l = learner(B, K, R)
  let wOut = shared(R * FS)
  func surrogate() -> Double {
    // the same graph's w, through the same refreshes (its forward alone)
    l.g.run(with: queue, feeds: learnFeeds(l, B, K, R, packed), targetOperations: nil, resultsDictionary: [l.w: td(wOut, [R, FS])])
    let wp = floats(wOut, R * FS); var a = 0.0
    for i in 0..<(rows * FS) { a += Double(wp[i]) * Double(pk[i]) }
    return a
  }
  print(String(format: "graph: %d refreshes, %d rows drawn from", K, rows))
  let geps = Float(env["GEPS"] ?? "1e-3")!
  for (name, shape, offset) in lay.blocks {
    var line: [String] = [], worst = 0.0, size = 0.0
    for _ in 0..<2 {
      let i = offset + Int(route_uniform(&rng) * Double(shape.reduce(1, *))), keep = th[i]
      th[i] = keep + geps; let up = surrogate(); th[i] = keep - geps; let down = surrogate(); th[i] = keep
      let numeric = (up - down) / Double(2 * geps), analytic = Double(gp[i])
      worst = max(worst, abs(numeric - analytic)); size = max(size, abs(numeric), abs(analytic))
      line.append(String(format: "%+.5f/%+.5f", analytic, numeric))
    }
    print(String(format: "  %-5@ largest difference %.1e of %.1e   (analytic/numeric) %@", name, worst, size, line.joined(separator: " ")))
  }
  exit(0)
}

if env["DEPLOY"] != nil {   // the C forward against the graph's, and its time a refresh
  var params = route_params()
  precondition(Int(route_params_count(lay.full ? 1 : 0)) == lay.count, "forward.c and policy.swift disagree on the layout")
  route_params_bind(&params, floats(opt.theta, lay.count), lay.full ? 1 : 0)
  route_stream_one(&measured, 1, 12582912, measured.compute.0, &streams[0])
  var w = deployed; var s = streams[0]
  route_sim_begin(sims[0], &s, &w, 0, 1, 1); _ = route_sim_decide(sims[0], decisions, tokens[0])
  let nl = Int(decisions[0].lanes), nc = Int(decisions[0].contexts)
  memcpy(laneP, laneTokens(0), L * IN * 4); for q in 0..<L { maskP[q] = q < nl ? 1 : 0 }
  memset(stateP, 0, 2 * D * 4); memcpy(tokP, tokens[0], nc * IN * 4); for c in 0..<nc { idxP[c] = 0 }
  runForward(1, nc)
  let scratch = UnsafeMutablePointer<Float>.allocate(capacity: route_forward_scratch(Int32(CONTEXTS))), wc = UnsafeMutablePointer<Float>.allocate(capacity: CONTEXTS * FS)
  var h = [Float](repeating: 0, count: 2 * D)
  route_forward(&params, laneTokens(0), Int32(nl), tokens[0], Int32(nc), &h, wc, scratch)
  var worst: Float = 0, size: Float = 0
  for i in 0..<(nc * FS) { worst = max(worst, abs(wc[i] - wP[i])); size = max(size, abs(wP[i])) }
  var hw: Float = 0; for i in 0..<(2 * D) { hw = max(hw, abs(h[i] - nextP[i])) }
  print(String(format: "C forward against the graph's, %d lanes, %d contexts: w largest difference %.1e of %.1e, next state %.1e", nl, nc, worst, size, hw))
  let reps = 2000, t0 = Date()
  for _ in 0..<reps { route_forward(&params, laneTokens(0), Int32(nl), tokens[0], Int32(nc), &h, wc, scratch) }
  print(String(format: "C forward, a refresh of the measured cluster (%d lanes, %d contexts): %.1f us on one core", nl, nc, Date().timeIntervalSince(t0) / Double(reps) * 1e6))
  exit(0)
}

print("\(arm), \(lay.count) parameters; \(windows) windows x 4 rollouts a step, up to \(KMAX) refreshes a window; T / floor on the measured cluster at \(sizes.map { Int($0) }) bytes, refreshed every 100 us, 100 us late")
print("start: " + measure())
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
  let (B, K, R, packed, _) = gather(ids, adv)
  gradients(B, K, R, packed)
  let alpha = opt.apply(rate: rate)
  stepTime += Date().timeIntervalSince(t0); since += 1
  if k < 3 || (k + 1) % 50 == 0 || k + 1 == steps {
    let c = ids.map { costs($0) }
    var mean = [0.0, 0.0, 0.0]
    for x in c { mean[0] += x.throughput / Double(c.count); mean[1] += x.latency / Double(c.count); mean[2] += x.queueing / Double(c.count) }
    print(String(format: "step %d (%.0fs): train throughput %.3f latency %.3f queueing %.3f, weights %.2f %.2f %.2f, %.2f steps/s (simulate %.0f%%, forward %.0f%%, credit %.0f%%, learn %.0f%%)",
                 k + 1, Date().timeIntervalSince(wall), mean[0], mean[1], mean[2], alpha[0], alpha[1], alpha[2], Double(since) / stepTime,
                 100 * simTime / stepTime, 100 * forwardTime / stepTime, 100 * creditTime / stepTime, 100 * (learnTime - creditTime) / stepTime))
    if (k + 1) % 100 == 0 || k + 1 == steps { print("  measured cluster: " + measure()) }
    fflush(stdout); stepTime = 0; since = 0; simTime = 0; forwardTime = 0; learnTime = 0; creditTime = 0
  }
}
if let path = env["THETA"] { try! Data(bytes: opt.theta.contents(), count: lay.count * 4).write(to: URL(fileURLWithPath: path)); print("parameters written to \(path)") }
