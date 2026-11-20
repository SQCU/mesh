// The runtime (route.h), main.swift for Swift's top-level code: keeps the environment's transactions in flight on the CPU's cores, gathers every pending
// decision point into one forward batch on the GPU, and after each step's transactions gathers the drawn-from contexts
// into one learning batch. Throughput's advantage is each transaction's completion against the mean of its scenario's
// other three; the environment's credit carries the pieces' latency and queueing advantages.
//
// usage: route FABRIC [steps] [scenarios a step]   (GRU=1 the GRU arm, else minGRU; RATE Adam's step, default 1e-3)
//   CHECK=1     the environment's dw against central differences of its log-likelihood in w; the graph's gradient
//               against central differences of sum(w . dw) in theta
//   SIMCHECK=1  every piece on the shortest-path tree, every group posted at 0, as tests/switch.c's fixed router
import Foundation
import Metal
import MetalPerformanceShadersGraph

let env = ProcessInfo.processInfo.environment, args = CommandLine.arguments
let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
let lay = Layout(full: env["GRU"] != nil)
let arm = lay.full ? "GRU" : "minGRU"
let CONTEXTS = Int(ROUTE_CONTEXTS)
var rng = route_rng(s: UInt64(env["SEED"] ?? "20261119")!)

var measured = route_fabric()
guard route_fabric_load(args[1], &measured) == 0 else { print("cannot read \(args[1])"); exit(1) }
let steps = args.count > 2 ? Int(args[2])! : 200, scenarios = args.count > 3 ? Int(args[3])! : 32
let rate = Double(env["RATE"] ?? "1e-3")!
let pool = 8192
var scen = [route_scenario](repeating: route_scenario(), count: pool)
for k in 0..<pool {
  var f = route_fabric(); route_network(&rng, &measured, &f)
  route_scenario_make(&f, exp(log(6144.0) + route_uniform(&rng) * (log(12582912.0) - log(6144.0))).rounded(), &rng, 1, &scen[k])
}
let sizes: [Double] = [6144, 24576, 98304, 393216, 3145728, 12582912]
let decode = sizes.map { S -> route_scenario in var x = route_scenario(); route_scenario_make(&measured, S, &rng, 0, &x); return x }

// ---- the transactions in flight ----
let TX = max(4 * scenarios, sizes.count * 16)
let sims = (0..<TX).map { _ in route_sim_new()! }
let decisions = UnsafeMutablePointer<route_decision>.allocate(capacity: TX)
let tokens = (0..<TX).map { _ in UnsafeMutablePointer<Float>.allocate(capacity: CONTEXTS * IN) }
let laneOffset = MemoryLayout<route_decision>.offset(of: \route_decision.lane)!
func laneTokens(_ t: Int) -> UnsafePointer<Float> { UnsafeRawPointer(decisions + t).advanced(by: laneOffset).assumingMemoryBound(to: Float.self) }

// a decision point kept for learning: its lane tokens and its contexts' tokens
struct Kept { var lanes: [Float]; var lanesUsed: Int; var tokens: [Float]; var contexts: Int }
var kept: [[Kept]] = Array(repeating: [], count: TX)

// ---- batches, in shared buffers; graphs built once a shape ----
func bucket(_ n: Int, _ least: Int) -> Int { var b = least; while b < n { b *= 2 }; return b }
let ROWS = 1 << 18, BATCH = 4096
func shared(_ floats: Int) -> MTLBuffer { device.makeBuffer(length: floats * 4, options: .storageModeShared)! }
let laneBuf = shared(BATCH * L * IN), maskBuf = shared(BATCH * L), tokBuf = shared(ROWS * IN), idxBuf = shared(ROWS), wBuf = shared(ROWS * FS), dwBuf = shared(TASKS * ROWS * FS)
let laneP = laneBuf.contents().bindMemory(to: Float.self, capacity: BATCH * L * IN), maskP = maskBuf.contents().bindMemory(to: Float.self, capacity: BATCH * L)
let tokP = tokBuf.contents().bindMemory(to: Float.self, capacity: ROWS * IN), idxP = idxBuf.contents().bindMemory(to: Int32.self, capacity: ROWS)
let wP = wBuf.contents().bindMemory(to: Float.self, capacity: ROWS * FS), dwP = dwBuf.contents().bindMemory(to: Float.self, capacity: TASKS * ROWS * FS)
var policies: [Int: Policy] = [:], learners: [Int: Learn] = [:]
var opt = Optimizer(device, queue, initial: lay.initial(&rng))
func td(_ b: MTLBuffer, _ shape: [Int], _ type: MPSDataType = .float32) -> MPSGraphTensorData { MPSGraphTensorData(b, shape: shape.map { NSNumber(value: $0) }, dataType: type) }

// the forward over decision points `bd` (their lanes and masks written) and `rows` context rows: w into wBuf
var forwardTime = 0.0
func runForward(_ bd: Int, _ rows: Int) {
  let B = bucket(bd, 16), R = bucket(rows, 1024)
  precondition(B <= BATCH && R <= ROWS, "a batch past its buffers")
  for i in rows..<R { idxP[i] = 0 }; memset(tokP + rows * IN, 0, (R - rows) * IN * 4); memset(maskP + bd * L, 0, (B - bd) * L * 4)
  let key = B << 32 | R
  if policies[key] == nil { policies[key] = Policy(lay, decisions: B, rows: R) }
  let p = policies[key]!, t0 = Date()
  p.g.run(with: queue, feeds: [p.theta: td(opt.theta, [lay.count]), p.lanes: td(laneBuf, [B, L, IN]), p.mask: td(maskBuf, [B, L]), p.tokens: td(tokBuf, [R, IN]),
                               p.index: td(idxBuf, [R], .int32)], targetOperations: nil, resultsDictionary: [p.w: td(wBuf, [R, FS])])
  forwardTime += Date().timeIntervalSince(t0)
}

// every transaction in `ids` run to the end; mode 0 the policy (decision points batched), else a reference router
var simTime = 0.0
func rollout(_ ids: [Int], _ scenarioOf: (Int) -> route_scenario, mode: Int32, keys: [UInt32], keep: Bool) {
  for (k, t) in ids.enumerated() { var s = scenarioOf(t); route_sim_begin(sims[t], &s, mode, keys[k]); kept[t] = [] }
  var active = ids
  while !active.isEmpty {
    let t0 = Date()
    let got = UnsafeMutablePointer<Int32>.allocate(capacity: active.count); defer { got.deallocate() }
    DispatchQueue.concurrentPerform(iterations: active.count) { k in got[k] = route_sim_decide(sims[active[k]], decisions + active[k], tokens[active[k]]) }
    active = (0..<active.count).filter { got[$0] == 1 }.map { active[$0] }
    simTime += Date().timeIntervalSince(t0)
    if active.isEmpty { break }
    var rowOf: [Int] = [], rows = 0
    if mode == 0 {   // one forward for every pending decision point
      for (b, t) in active.enumerated() {
        let nl = Int(decisions[t].lanes), nc = Int(decisions[t].contexts)
        memcpy(laneP + b * L * IN, laneTokens(t), L * IN * 4)
        for q in 0..<L { maskP[b * L + q] = q < nl ? 1 : 0 }
        memcpy(tokP + rows * IN, tokens[t], nc * IN * 4)
        for c in 0..<nc { idxP[rows + c] = Int32(b) }
        rowOf.append(rows); rows += nc
        if keep { kept[t].append(Kept(lanes: Array(UnsafeBufferPointer(start: laneTokens(t), count: L * IN)), lanesUsed: nl, tokens: Array(UnsafeBufferPointer(start: tokens[t], count: nc * IN)), contexts: nc)) }
      }
      runForward(active.count, rows)
    }
    let t1 = Date()
    let order = active, offsets = rowOf
    DispatchQueue.concurrentPerform(iterations: order.count) { k in route_sim_answer(sims[order[k]], mode == 0 ? wP + offsets[k] * FS : nil) }
    simTime += Date().timeIntervalSince(t1)
  }
}
func costs(_ t: Int) -> route_costs { var c = route_costs(); route_sim_costs(sims[t], &c); return c }

// the learning batch from transactions `ids` (kept), throughput advantages `adv`: the drawn-from rows, their decision
// points, each task's dw packed [TASKS][R][FS]. Returns (B, R, packed, the decision points and rows used)
var learnTime = 0.0, creditTime = 0.0
func gather(_ ids: [Int], _ adv: [Double]) -> (Int, Int, MTLBuffer, Int, Int) {
  let t0 = Date()
  var credit = [[Float]](repeating: [], count: ids.count)   // each transaction's dw, its points' [TASKS][contexts][FS] in turn
  DispatchQueue.concurrentPerform(iterations: ids.count) { k in
    let t = ids[k]; var all: [Float] = []
    for (p, d) in kept[t].enumerated() {
      var part = [Float](repeating: 0, count: TASKS * d.contexts * FS), ll = 0.0
      route_sim_credit(sims[t], Int32(p), &part, &ll)
      all += part
    }
    credit[k] = all
  }
  creditTime += Date().timeIntervalSince(t0)
  // the rows drawn from (any task's dw not zero), each with its decision point
  var bd = 0, rows = 0
  let scale = 1 / Double(ids.count)
  for (k, t) in ids.enumerated() {
    var at = 0
    for d in kept[t] {
      memcpy(laneP + bd * L * IN, d.lanes, L * IN * 4)
      for q in 0..<L { maskP[bd * L + q] = q < d.lanesUsed ? 1 : 0 }
      for c in 0..<d.contexts {
        var used = false
        for task in 0..<TASKS { for f in 0..<FS where credit[k][at + (task * d.contexts + c) * FS + f] != 0 { used = true } }
        if !used { continue }
        precondition(rows < ROWS, "learning rows past their buffer")
        for i in 0..<IN { tokP[rows * IN + i] = d.tokens[c * IN + i] }
        idxP[rows] = Int32(bd)
        for task in 0..<TASKS {
          let s = Float((task == 0 ? adv[k] : 1) * scale)
          for f in 0..<FS { dwP[(task * ROWS + rows) * FS + f] = credit[k][at + (task * d.contexts + c) * FS + f] * s }
        }
        rows += 1
      }
      at += TASKS * d.contexts * FS
      bd += 1
    }
  }
  let B = bucket(bd, 16), R = bucket(rows, 1024)
  precondition(B <= BATCH && R <= ROWS, "a batch past its buffers")
  for i in rows..<R { idxP[i] = 0 }; memset(tokP + rows * IN, 0, (R - rows) * IN * 4); memset(maskP + bd * L, 0, (B - bd) * L * 4)
  // the graph reads dw as [TASKS][R][FS]: pack the three channels at stride R
  let packed = shared(TASKS * R * FS), pp = packed.contents().bindMemory(to: Float.self, capacity: TASKS * R * FS)
  for task in 0..<TASKS { memcpy(pp + task * R * FS, dwP + task * ROWS * FS, rows * FS * 4); memset(pp + task * R * FS + rows * FS, 0, (R - rows) * FS * 4) }
  learnTime += Date().timeIntervalSince(t0)
  return (B, R, packed, bd, rows)
}
// each task's gradient at theta into opt.grads, their Gram into opt.gram
func gradients(_ B: Int, _ R: Int, _ packed: MTLBuffer) {
  let t0 = Date()
  let key = B << 32 | R
  if learners[key] == nil { learners[key] = Learn(lay, decisions: B, rows: R) }
  let l = learners[key]!
  var results: [MPSGraphTensor: MPSGraphTensorData] = [l.gram: td(opt.gram, [TASKS * TASKS])]
  let gradBufs = (0..<TASKS).map { _ in shared(lay.count) }
  for t in 0..<TASKS { results[l.grads[t]] = td(gradBufs[t], [lay.count]) }
  l.g.run(with: queue, feeds: [l.theta: td(opt.theta, [lay.count]), l.lanes: td(laneBuf, [B, L, IN]), l.mask: td(maskBuf, [B, L]), l.tokens: td(tokBuf, [R, IN]),
                               l.index: td(idxBuf, [R], .int32), l.dw: td(packed, [TASKS, R, FS])], targetOperations: nil, resultsDictionary: results)
  let gp = opt.grads.contents().bindMemory(to: Float.self, capacity: TASKS * lay.count)
  for t in 0..<TASKS { memcpy(gp + t * lay.count, gradBufs[t].contents(), lay.count * 4) }
  learnTime += Date().timeIntervalSince(t0)
}
func learn(_ ids: [Int], _ adv: [Double]) -> [Double] {
  let (B, R, packed, _, _) = gather(ids, adv)
  gradients(B, R, packed)
  let t0 = Date(); let alpha = opt.apply(rate: rate); learnTime += Date().timeIntervalSince(t0)
  return alpha
}

if env["SIMCHECK"] != nil {
  print("every piece on the shortest-path tree, all posted at 0 (tests/switch.c's fixed router): completion, floor")
  rollout(Array(0..<sizes.count), { decode[$0] }, mode: 4, keys: Array(repeating: 1, count: sizes.count), keep: false)
  for (i, S) in sizes.enumerated() { print(String(format: "all-gather %9.0f  %9.2fus  floor %8.2fus", S, costs(i).throughput * decode[i].floor / 1000, decode[i].floor / 1000)) }
  exit(0)
}

if env["COUNT"] != nil {   // what one transaction at each decode size does, and the forward's latency at small batches
  print("measured cluster, the policy at initialization: per transaction")
  for (i, S) in sizes.enumerated() {
    rollout([0], { _ in decode[i] }, mode: 0, keys: [3], keep: false)
    var n = [UInt64](repeating: 0, count: 4); route_sim_counts(sims[0], &n)
    let c = costs(0)
    print(String(format: "  %9.0f bytes a rank: %d decision points x %d contexts (one forward each), %llu pieces sent as %llu sends, %llu events, %llu draws, %llu table entries; completion %.2f us",
                 S, route_sim_points(sims[0]), route_sim_contexts(sims[0]), UInt64(decode[i].commodities) * UInt64(max(1, Int((S / ROUTE_PAGE).rounded(.up)))), n[3], n[0], n[1], n[2], c.throughput * decode[i].floor / 1000))
  }
  // the forward alone, B decision points of the measured cluster's 92 contexts each, after its graph is built
  for b in [1, 4, 16, 64, 128] {
    let ids = Array(0..<b)
    for t in ids { var s = decode[5]; route_sim_begin(sims[t], &s, 0, UInt32(t)); _ = route_sim_decide(sims[t], decisions + t, tokens[t]) }
    var rows = 0
    for (k, t) in ids.enumerated() {
      memcpy(laneP + k * L * IN, laneTokens(t), L * IN * 4); for q in 0..<L { maskP[k * L + q] = q < Int(decisions[t].lanes) ? 1 : 0 }
      let nc = Int(decisions[t].contexts); memcpy(tokP + rows * IN, tokens[t], nc * IN * 4); for c in 0..<nc { idxP[rows + c] = Int32(k) }; rows += nc
    }
    runForward(b, rows); forwardTime = 0
    let reps = 50; for _ in 0..<reps { runForward(b, rows) }
    print(String(format: "  forward, %3d decision points (%5d context rows): %.0f us a call (synchronous MPSGraph run, wall clock)", b, rows, forwardTime / Double(reps) * 1e6))
  }
  exit(0)
}
if env["DEPLOY"] != nil {   // the deployed policy's latency: the C forward on the CPU, the graph encoded on the GPU; and their agreement
  var params = route_params()
  let thetaP = opt.theta.contents().bindMemory(to: Float.self, capacity: lay.count)
  precondition(Int(route_params_count(lay.full ? 1 : 0)) == lay.count, "forward.c and policy.swift disagree on the layout")
  route_params_bind(&params, thetaP, lay.full ? 1 : 0)
  func one(_ sc: route_scenario) -> (Int, Int) { var s = sc; route_sim_begin(sims[0], &s, 0, 9); _ = route_sim_decide(sims[0], decisions, tokens[0]); return (Int(decisions[0].lanes), Int(decisions[0].contexts)) }
  // agreement, on the measured cluster's first decision point
  let (nl, nc) = one(decode[5])
  memcpy(laneP, laneTokens(0), L * IN * 4); for q in 0..<L { maskP[q] = q < nl ? 1 : 0 }
  memcpy(tokP, tokens[0], nc * IN * 4); for c in 0..<nc { idxP[c] = 0 }
  runForward(1, nc)
  let scratch = UnsafeMutablePointer<Float>.allocate(capacity: route_forward_scratch(Int32(CONTEXTS))), wc = UnsafeMutablePointer<Float>.allocate(capacity: CONTEXTS * FS)
  var state = [Float](repeating: 0, count: 2 * D)
  route_forward(&params, laneTokens(0), Int32(nl), tokens[0], Int32(nc), &state, wc, scratch)
  var worst: Float = 0, size: Float = 0
  for i in 0..<(nc * FS) { worst = max(worst, abs(wc[i] - wP[i])); size = max(size, abs(wP[i])) }
  print(String(format: "C forward against the graph's, %d lanes, %d contexts: largest difference %.1e of %.1e", nl, nc, worst, size))
  // the C forward's time on one core, by context count
  var seen = Set<Int>()
  for sc in [decode[5]] + scen.prefix(400) {
    let (nl, nc) = one(sc)
    let bucket = nc <= 100 ? 92 : nc <= 300 ? 290 : 1000
    if seen.contains(bucket) { continue }; seen.insert(bucket)
    let reps = 2000; let t0 = Date()
    for _ in 0..<reps { for d in 0..<(2 * D) { state[d] = 0 }; route_forward(&params, laneTokens(0), Int32(nl), tokens[0], Int32(nc), &state, wc, scratch) }
    print(String(format: "  C forward, %2d lanes %4d contexts: %.1f us a decision point (one core)", nl, nc, Date().timeIntervalSince(t0) / Double(reps) * 1e6))
  }
  // the graph, encoded into a command buffer (no synchronous run), on the GPU clock
  let (nl1, nc1) = one(decode[5])
  memcpy(laneP, laneTokens(0), L * IN * 4); for q in 0..<L { maskP[q] = q < nl1 ? 1 : 0 }; memcpy(tokP, tokens[0], nc1 * IN * 4)
  let B = 16, R = bucket(nc1, 1024), p = policies[B << 32 | R] ?? Policy(lay, decisions: B, rows: R)
  let feeds: [MPSGraphTensor: MPSGraphTensorData] = [p.theta: td(opt.theta, [lay.count]), p.lanes: td(laneBuf, [B, L, IN]), p.mask: td(maskBuf, [B, L]), p.tokens: td(tokBuf, [R, IN]), p.index: td(idxBuf, [R], .int32)]
  var gpu: [Double] = [], wall: [Double] = []
  for k in 0..<60 {
    let t0 = Date(), cb = MPSCommandBuffer(from: queue)
    p.g.encode(to: cb, feeds: feeds, targetOperations: nil, resultsDictionary: [p.w: td(wBuf, [R, FS])], executionDescriptor: nil)
    cb.commit(); cb.waitUntilCompleted()
    if k >= 10 { gpu.append(cb.commandBuffer.gpuEndTime - cb.commandBuffer.gpuStartTime); wall.append(Date().timeIntervalSince(t0)) }
  }
  gpu.sort(); wall.sort()
  print(String(format: "  graph encoded, 1 decision point (padded to %d x %d rows): GPU %.0f us median (%.0f min), commit to completion %.0f us median", B, R, gpu[gpu.count / 2] * 1e6, gpu[0] * 1e6, wall[wall.count / 2] * 1e6))
  exit(0)
}
if env["CHECK"] != nil {
  // the environment: channel 0 of its credit is d(log-likelihood of the draws)/dw; against central differences
  rollout([0], { _ in scen[17] }, mode: 0, keys: [5], keep: true)
  let nc = Int(route_sim_contexts(sims[0])), eps = Float(env["EPS"] ?? "1e-3")!
  var dw = [Float](repeating: 0, count: TASKS * nc * FS), ll = 0.0
  route_sim_credit(sims[0], 0, &dw, &ll)
  print(String(format: "environment: %d contexts, log-likelihood of decision point 0's draws %.4f", nc, ll))
  let w = route_sim_w(sims[0], 0)!
  var shown = 0
  for i in 0..<(nc * FS) where dw[i] != 0 && shown < 8 {
    let keep = w[i]; var up = 0.0, down = 0.0, scratch = [Float](repeating: 0, count: TASKS * nc * FS)
    w[i] = keep + eps; route_sim_credit(sims[0], 0, &scratch, &up)
    w[i] = keep - eps; route_sim_credit(sims[0], 0, &scratch, &down)
    w[i] = keep
    print(String(format: "  w[context %d][%d]: credit %+.5f, central difference %+.5f", i / FS, i % FS, dw[i], (up - down) / Double(2 * eps)))
    shown += 1
  }
  // the graph: d sum(w . dw_0)/d theta against central differences, on this transaction's learning batch
  let (B, R, packed, bd, rows) = gather([0], [1])
  gradients(B, R, packed)
  let pk = packed.contents().bindMemory(to: Float.self, capacity: TASKS * R * FS), th = opt.theta.contents().bindMemory(to: Float.self, capacity: lay.count)
  let gp = opt.grads.contents().bindMemory(to: Float.self, capacity: TASKS * lay.count)
  func surrogate() -> Double { runForward(bd, rows); var a = 0.0; for i in 0..<(rows * FS) { a += Double(wP[i]) * Double(pk[i]) }; return a }
  // runForward pads to the same buckets as gather's batch, so the forward sees the same rows
  print(String(format: "graph: %d decision points, %d rows drawn from", bd, rows))
  let geps = Float(env["GEPS"] ?? "1e-2")!
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

func measure() -> String {   // T / floor at each decode size on the measured cluster, the mean over 16 transactions
  var out: [String] = []
  for (mode, name) in [(Int32(0), "policy"), (2, "spt"), (3, "uniform")] {
    let ids = Array(0..<(sizes.count * 16))
    rollout(ids, { decode[$0 / 16] }, mode: mode, keys: ids.map { UInt32(77 + $0 % 16) }, keep: false)
    var cells: [String] = []
    for i in 0..<sizes.count { var a = 0.0; for d in 0..<16 { a += costs(i * 16 + d).throughput }; cells.append(String(format: "%.2f", a / 16)) }
    out.append(name + " " + cells.joined(separator: " "))
  }
  return out.joined(separator: " | ")
}

print("\(arm), \(lay.count) parameters; \(scenarios) scenarios x 4 transactions a step; \(ProcessInfo.processInfo.activeProcessorCount) CPU threads; T / floor on the measured cluster at \(sizes.map { Int($0) }) bytes a rank")
print("start: " + measure())
let wall = Date()
var stepTime = 0.0, since = 0
for k in 0..<steps {
  let t0 = Date()
  let picks = (0..<scenarios).map { _ in Int(route_uniform(&rng) * Double(pool)) }
  let ids = Array(0..<(4 * scenarios))
  rollout(ids, { scen[picks[$0 / 4]] }, mode: 0, keys: ids.map { _ in UInt32(truncatingIfNeeded: route_next(&rng)) }, keep: true)
  let T = ids.map { costs($0).throughput }
  let adv = ids.map { t -> Double in let s = t / 4 * 4, total = T[s] + T[s + 1] + T[s + 2] + T[s + 3]; return T[t] - (total - T[t]) / 3 }
  let alpha = learn(ids, adv)
  stepTime += Date().timeIntervalSince(t0); since += 1
  if k < 3 || (k + 1) % 50 == 0 || k + 1 == steps {
    let mean = (0..<3).map { t in ids.map { c -> Double in let x = costs(c); return t == 0 ? x.throughput : t == 1 ? x.latency : x.queueing }.reduce(0, +) / Double(ids.count) }
    print(String(format: "step %d (%.0fs): train throughput %.3f latency %.3f queueing %.3f, weights %.2f %.2f %.2f, %.2f steps/s (simulate %.0f%%, forward %.0f%%, credit %.0f%%, learn %.0f%%) | ",
                 k + 1, Date().timeIntervalSince(wall), mean[0], mean[1], mean[2], alpha[0], alpha[1], alpha[2], Double(since) / stepTime,
                 100 * simTime / stepTime, 100 * forwardTime / stepTime, 100 * creditTime / stepTime, 100 * (learnTime - creditTime) / stepTime) + measure())
    fflush(stdout); stepTime = 0; since = 0; simTime = 0; forwardTime = 0; learnTime = 0; creditTime = 0
  }
}
