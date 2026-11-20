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
import Accelerate
import Foundation
import Metal
import MetalPerformanceShadersGraph

let env = ProcessInfo.processInfo.environment, args = CommandLine.arguments
func tel(_ k: route_tel) -> Int32 { Int32(k.rawValue) }
route_tel_phase(tel(RT_PHASE_START))
let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
precondition(env["GRU"] == nil, "the GRU arm has no int8 path (its gates read h: one product a lane); minGRU only")
let lay = Layout(full: false)   // the fp32 reference's layout (CHECK, earlier parameter files)
let nlay = NetLayout()
let arm = "minGRU (int8 products)"
let CONTEXTS = Int(ROUTE_CONTEXTS), KMAX = 4
// data-parallel across machines: RANK of WORLD, MESH_ID the bootstrap root's host:port: the three gradients' mean an
// all-reduce through the mesh's bridges, overlapped with the next step (Exchange), so every rank takes the same steps from the same theta
// (its initial seed the same on every rank; each rank's windows its own, SEED + RANK)
let rank = Int(env["RANK"] ?? "0")!, world = Int(env["WORLD"] ?? "1")!
// the telemetry table (telemetry.h; read with route-stat): ROUTE_TELEMETRY, else /tmp/route-telemetry-RANK
if route_tel_open(env["ROUTE_TELEMETRY"] ?? "/tmp/route-telemetry-\(rank)", Int32(rank), Int32(world)) != 0 { print("telemetry: cannot map the table") }
var rng = route_rng(s: UInt64(env["SEED"] ?? "20261119")! &+ UInt64(rank) &* 0x9e3779b97f4a7c15)
var thetaRng = route_rng(s: 20261119)
func normal() -> Double { let u = max(route_uniform(&rng), 1e-300), v = route_uniform(&rng); return sqrt(-2 * log(u)) * cos(2 * Double.pi * v) }

var measured = route_fabric()
guard route_fabric_load(args[1], &measured) == 0 else { print("cannot read \(args[1])"); exit(1) }
let steps = args.count > 2 ? Int(args[2])! : 200, windows = args.count > 3 ? Int(args[3])! : 32
let rate = Double(env["RATE"] ?? "1e-3")!
let fabrics: [route_fabric] = (0..<2048).map { _ in var f = route_fabric(); route_network(&rng, &measured, &f); return f }
let kinds = ["all-gather", "all-reduce", "scatter"], sizes: [Double] = [6144, 98304, 3145728, 12582912]

// ---- the runs in flight (runs.swift): as many as one forward holds (its rows over a run's most contexts) ----
let ROWS = 1 << 18, BATCH = 1024
let TX = ROWS / CONTEXTS
var pool: [[Float]] = []   // states windows ended with, to start later windows from

// ---- the learning batch, in shared buffers (exact row counts: the int8 network's kernels take any) ----
func shared(_ floats: Int) -> MTLBuffer { device.makeBuffer(length: floats * 4, options: .storageModeShared)! }
func floats(_ b: MTLBuffer, _ n: Int) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: n) }
let laneBuf = shared(KMAX * BATCH * L * IN), maskBuf = shared(KMAX * BATCH * L), stateBuf = shared(BATCH * 2 * D), nextBuf = shared(BATCH * 2 * D)
let tokBuf = shared(ROWS * IN), idxBuf = shared(ROWS), wBuf = shared(ROWS * FS), dwBuf = shared(TASKS * ROWS * FS)
let laneP = floats(laneBuf, KMAX * BATCH * L * IN), maskP = floats(maskBuf, KMAX * BATCH * L), stateP = floats(stateBuf, BATCH * 2 * D), nextP = floats(nextBuf, BATCH * 2 * D)
let tokP = floats(tokBuf, ROWS * IN), idxP = idxBuf.contents().bindMemory(to: Int32.self, capacity: ROWS), wP = floats(wBuf, ROWS * FS), dwP = floats(dwBuf, TASKS * ROWS * FS)
let library = env["ROUTE_METALLIB"] ?? URL(fileURLWithPath: args[0]).deletingLastPathComponent().appendingPathComponent("route.metallib").path
// the learner's network (theta, the optimizer; a learning batch of up to BATCH runs) and the producers' (the forwards;
// theta copied in from the learner's as it publishes)
let net = Net(device, queue, library: library, runs: BATCH, refreshes: KMAX, contexts: ROWS, initial: nlay.initial(&thetaRng))
let fnet = Net(device, device.makeCommandQueue()!, library: library, runs: TX, refreshes: 1, contexts: ROWS, initial: Array(UnsafeBufferPointer(start: net.theta.contents().bindMemory(to: Float.self, capacity: nlay.count), count: nlay.count)))
func syncForward() { memcpy(fnet.theta.contents(), net.theta.contents(), nlay.count * 4); fnet.refresh() }
let R0 = Runner(device, net: fnet, slots: TX, rows: ROWS)   // the producers' runs (and the checks')
route_tel_phase(tel(RT_PHASE_NETWORKS))
func td(_ b: MTLBuffer, _ shape: [Int], _ type: MPSDataType = .float32) -> MPSGraphTensorData { MPSGraphTensorData(b, shape: dims(shape), dataType: type) }


// a finished run as the learning batch reads it (copied out of its slot, which the pool then reuses)
struct DoneRun {
  let kept: [Kept], start: [Float], credit: [Float], costs: route_costs, counts: [UInt64]
  init(_ s: Slot) { kept = s.kept; start = s.start; credit = s.credit; costs = s.costs; counts = s.counts }
}
// the learning batch over finished runs, throughput advantages `adv`: lanes [run][refresh][lane]; drawn-from rows, each
// reading its refresh's state (lane row (b K + k) L + L - 1), contiguous a refresh (seg); each task's dw packed
// [TASKS][R][FS]. Returns (B, K, R, packed, seg)
var learnTime = 0.0
func gather(_ runs: [DoneRun], _ adv: [Double]) -> (Int, Int, Int, MTLBuffer, [UInt32]) {
  let t0 = Date()
  let B = runs.count, K = KMAX
  precondition(B <= BATCH, "a learning batch past its buffers")
  memset(laneP, 0, B * K * L * IN * 4); memset(maskP, 0, B * K * L * 4); memset(stateP, 0, B * 2 * D * 4)
  var rows = 0, seg = [UInt32](repeating: 0, count: 2 * B * K)
  let scale = 1 / Double(B)
  for (b, r) in runs.enumerated() {
    precondition(r.kept.count <= K, "a window past the learning batch's refreshes")
    for i in 0..<(2 * D) { stateP[b * 2 * D + i] = r.start[i] }
    var at = 0
    for k in 0..<K {
      seg[2 * (b * K + k)] = UInt32(rows)
      if k < r.kept.count {
        let d = r.kept[k]
        memcpy(laneP + (b * K + k) * L * IN, d.lanes, L * IN * 4)
        for q in 0..<L { maskP[(b * K + k) * L + q] = q < d.lanesUsed ? 1 : 0 }
        for c in 0..<d.contexts {
          var used = false
          for task in 0..<TASKS { for f in 0..<FS where r.credit[at + (task * d.contexts + c) * FS + f] != 0 { used = true } }
          if !used { continue }
          precondition(rows < ROWS, "learning rows past the buffers")
          for i in 0..<IN { tokP[rows * IN + i] = d.tokens[c * IN + i] }
          idxP[rows] = Int32((b * K + k) * L + L - 1)
          for task in 0..<TASKS {
            let s = Float((task == 0 ? adv[b] : 1) * scale)
            for f in 0..<FS { dwP[(task * ROWS + rows) * FS + f] = r.credit[at + (task * d.contexts + c) * FS + f] * s }
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
// ---- the gradients' mean across machines: an all-reduce through the mesh (exchange.c: libnccl-mesh over the nodes'
// bridges, the Thunderbolt fabric as the mesh routes it), on its own thread ----
// MESH_ID: the bootstrap root's host:port (rank 0's). Every rank's sum is the same bits (the library's reduction is one
// program every rank runs). The exchange of step k's gradients runs while step k + 1 computes; step k + 1 applies step
// k's mean (a declared one-step delay: every rank applies the same mean in the same order).
func join() {
  guard world > 1 else { return }
  let id = env["MESH_ID"] ?? ""
  precondition(!id.isEmpty, "MESH_ID: the bootstrap root's host:port")
  if route_exchange_open(Int32(rank), Int32(world), id, size_t(TASKS * nlay.count + 2)) != 0 { print("exchange: " + String(cString: route_exchange_failure())); route_exchange_close(); exit(1) }
  atexit { route_exchange_close() }   // every exit: no program left behind with pieces outstanding
  print("rank \(rank) of \(world): joined through the mesh (bootstrap \(id))"); fflush(stdout)
  route_tel_phase(tel(RT_PHASE_JOINED))
}
// p [n]: every rank's mean, in place (through the lease)
// p [n]: every rank's sum (through the lease), in place
func meshSum(_ p: UnsafeMutablePointer<Float>, _ n: Int) {
  guard world > 1 else { return }
  let lease = route_exchange_buffer()!
  memcpy(lease, p, n * 4)
  if route_exchange_sum() != 0 { print("exchange: " + String(cString: route_exchange_failure())); fflush(stdout); exit(1) }   // atexit closes
  memcpy(p, lease, n * 4)
}
func meshMean(_ p: UnsafeMutablePointer<Float>, _ n: Int) { meshSum(p, n); let inv = 1 / Float(world); for i in 0..<n { p[i] *= inv } }
// the exchange: the learner adds each step's gradients into an accumulator (add, at once); the exchange thread runs a
// round whenever this rank has accumulated a step (or has finished), back to back: its payload the accumulated sum,
// its step count and its done flag, summed across ranks (the rendezvous: every rank's k-th round is one), so a round's
// mean is over every learning step in it, every rank's. ready() returns the means completed since (in round order, with
// their Gram matrices), never waiting: every rank applies the same means in the same order. finish() and drain(): the
// run's end, once every rank's done flag is summed in.
final class Exchange {
  let n: Int, P: Int, cond = NSCondition()
  var acc: [Float], steps = 0, finished = false, over = false, done: [([Float], [Float])] = [], thread: Thread? = nil
  init(P: Int) { self.P = P; n = TASKS * P; acc = [Float](repeating: 0, count: TASKS * P) }
  func add(_ g: UnsafePointer<Float>) {
    cond.lock()
    acc.withUnsafeMutableBufferPointer { a in vDSP_vadd(a.baseAddress!, 1, g, 1, a.baseAddress!, 1, vDSP_Length(n)) }
    steps += 1; route_tel_set(tel(RT_EXCHANGE_IN_FLIGHT), UInt64(steps))
    cond.broadcast(); cond.unlock()
  }
  func start() {
    let th = Thread { [self] in
      var payload = [Float](repeating: 0, count: n + 2)
      while true {
        cond.lock()
        while steps == 0 && !finished { cond.wait() }
        payload.withUnsafeMutableBufferPointer { p in acc.withUnsafeBufferPointer { memcpy(p.baseAddress!, $0.baseAddress!, n * 4) } }
        payload[n] = Float(steps); payload[n + 1] = finished ? 1 : 0
        for i in 0..<n { acc[i] = 0 }
        steps = 0; route_tel_set(tel(RT_EXCHANGE_IN_FLIGHT), 0)
        cond.unlock()
        let t0 = route_tel_now()
        payload.withUnsafeMutableBufferPointer { meshSum($0.baseAddress!, n + 2) }
        let total = payload[n], allDone = Int(payload[n + 1]) == world
        if total > 0 {
          var mean = Array(payload[0..<n]); let inv = 1 / total
          for i in 0..<n { mean[i] *= inv }
          var gram = [Float](repeating: 0, count: 9)
          for a in 0..<3 { for b in a..<3 { var d = 0.0; for i in 0..<P { d += Double(mean[a * P + i]) * Double(mean[b * P + i]) }; gram[3 * a + b] = Float(d); gram[3 * b + a] = Float(d) } }
          cond.lock(); done.append((mean, gram)); cond.broadcast(); cond.unlock()
        }
        route_tel_add(tel(RT_EXCHANGE_ROUNDS), 1); route_tel_add(tel(RT_EXCHANGE_NS), route_tel_now() - t0)
        if allDone { cond.lock(); over = true; cond.broadcast(); cond.unlock(); return }
      }
    }
    th.qualityOfService = .userInteractive; thread = th; th.start()
  }
  func ready() -> [([Float], [Float])] { cond.lock(); defer { cond.unlock() }; let r = done; done = []; return r }
  func finish() { cond.lock(); finished = true; cond.broadcast(); cond.unlock() }
  func drain() -> [([Float], [Float])] { cond.lock(); while !over { cond.wait() }; let r = done; done = []; cond.unlock(); return r }
}
let exchange = Exchange(P: nlay.count)
join()
if let n = env["XBENCH"].flatMap({ Int($0) }) {   // the exchange alone: n rounds of the three gradients' mean, no compute between
  // n 0: declare and close, no step issued: leaves no program with pieces outstanding (a flush of a dead run's)
  if n == 0 { route_exchange_close(); print("rank \(rank): declared and closed, no step"); exit(0) }
  let P = TASKS * nlay.count
  var t: [Double] = []
  let g = floats(net.grads, P)
  var wrong = 0
  for r in 0..<n {   // each round's values the rank's own, so its mean is known: (rank + 1 + round + i mod 7) averaged
    for i in 0..<P { g[i] = Float(rank + 1 + r % 5 + i % 7) }
    let t0 = Date(); meshMean(g, P); t.append(Date().timeIntervalSince(t0))
    let base = Float(world + 1) / 2 + Float(r % 5)
    for i in 0..<P where g[i] != base + Float(i % 7) { wrong += 1 }
  }
  print("rank \(rank): \(wrong) of \(n * P) means wrong")
  t.sort()
  print(String(format: "rank %d: the exchange alone (%d floats, %.0f KB), %d rounds: median %.2f ms, p90 %.2f ms, max %.2f ms", rank, P, Double(P * 4) / 1024, n, t[n / 2] * 1e3, t[n * 9 / 10] * 1e3, t[n - 1] * 1e3))
  exit(0)
}

func gradients(_ B: Int, _ K: Int, _ R: Int, _ packed: MTLBuffer, _ seg: [UInt32]) {
  let t0 = Date()
  net.learn(lanes: laneBuf, mask: maskBuf, h0: stateBuf, tokens: tokBuf, pos: idxBuf, seg: seg, B: B, K: K, R: R, dw: packed, w: wBuf)
  learnTime += Date().timeIntervalSince(t0); route_tel_add(tel(RT_LEARN_NS), UInt64(Date().timeIntervalSince(t0) * 1e9))
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
func measure(_ r: Runner, _ fabric: inout route_fabric) -> String {
  var out: [String] = []
  let cases = kinds.indices.flatMap { k in sizes.map { (k, $0) } }
  for t in 0..<(cases.count * 4) { route_stream_one(&fabric, Int32(cases[t / 4].0), cases[t / 4].1, fabric.compute.0, &r.slots[t].stream); r.slots[t].state = [Float](repeating: 0, count: 2 * D) }
  var rows: [String: [Double]] = [:]
  for (mode, name, w) in [(Int32(0), "policy", deployed), (0, "policy (burst)", deployedBurst), (2, "spt", deployed), (3, "uniform", deployed), (4, "switch.c fixed", deployed)] {
    for t in 0..<(cases.count * 4) { r.slots[t].state = [Float](repeating: 0, count: 2 * D) }
    r.rollout(Array(0..<(cases.count * 4)), window: { _ in w }, mode: mode, key0: { UInt32(77 + $0 % 4) }, key: { UInt32(91 + $0 % 4) }, keep: false)
    rows[name] = (0..<cases.count).map { c in (0..<4).map { r.costs(c * 4 + $0).throughput }.reduce(0, +) / 4 }
  }
  for (k, kind) in kinds.enumerated() {
    var line = "\(kind):"
    for name in ["policy", "policy (burst)", "spt", "uniform", "switch.c fixed"] { line += " \(name) " + (0..<sizes.count).map { String(format: "%.2f", rows[name]![k * sizes.count + $0]) }.joined(separator: " ") + ";" }
    out.append(line)
  }
  return out.joined(separator: " | ")
}
// rank 0's measurement on a thread of its own (its runner, its network: a snapshot of theta), so no rank waits for it:
// at the start and the end alone
let measuredRuns = kinds.count * sizes.count * 4   // measure()'s: each kind and size, four rollouts
let evaluator: Runner? = rank == 0 ? Runner(device, net: Net(device, device.makeCommandQueue()!, library: library, runs: measuredRuns, refreshes: 1, contexts: measuredRuns * CONTEXTS,
                                                              initial: [Float](repeating: 0, count: nlay.count)), slots: measuredRuns, rows: measuredRuns * CONTEXTS) : nil
let evaluating = DispatchSemaphore(value: 1), evalQueue = DispatchQueue(label: "evaluation")
func evaluate(_ label: String, wait: Bool) {
  guard let ev = evaluator, evaluating.wait(timeout: wait ? .distantFuture : .now()) == .success else { return }   // one at a time; a busy one skipped
  memcpy(ev.net.theta.contents(), net.theta.contents(), nlay.count * 4); ev.net.refresh()
  var f = measured   // copied here, on the main thread (which may pass measured inout meanwhile)
  let run = { print("  measured cluster (\(label)): " + measure(ev, &f)); fflush(stdout); evaluating.signal() }
  if wait { run() } else { evalQueue.async(execute: run) }
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
  net.refresh(); syncForward()
}
if env["SIMCHECK"] != nil {
  print("every piece on the shortest-path tree posted at its release (tests/switch.c's fixed router), the measured cluster, us:")
  for (k, kind) in kinds.enumerated() {
    var line = String(format: "%-10@", kind)
    for (i, S) in sizes.enumerated() {
      route_stream_one(&measured, Int32(k), S, measured.compute.0, &R0.slots[i].stream)
      R0.rollout([i], window: { _ in deployed }, mode: 4, key0: { _ in 1 }, key: { _ in 1 }, keep: false)
      line += String(format: " %9.2f", R0.costs(i).throughput * R0.slots[i].stream.op.0.floor / 1000)
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
    w.points = 3; R0.slots[0].stream = s; R0.slots[0].state = h; tries += 1
    R0.rollout([0], window: { _ in w }, mode: 0, key0: { _ in 3 }, key: { _ in UInt32(5 + tries) }, keep: true)
    nc = Int(route_sim_contexts(R0.sims[0])); dw = [Float](repeating: 0, count: TASKS * nc * FS)
    for p in 0..<R0.slots[0].kept.count { route_sim_credit(R0.sims[0], Int32(p), &dw, &ll); point = p; if dw.contains(where: { $0 != 0 }) { found = true; break } }
    let last = R0.slots[0].kept.last.map { $0.contexts } ?? 0
    var big: Float = 0, bad = 0; for i in 0..<(last * FS) { if !R0.wP[i].isFinite { bad += 1 } else { big = max(big, abs(R0.wP[i])) } }
    print(String(format: "window %d: %d contexts, %d refreshes; the last forward's w: largest |w| %.3f, %d not finite; %@", tries, nc, R0.slots[0].kept.count, big, bad, found ? "a choice" : "no choice, another"))
    precondition(tries < 50, "no window with a choice in 50")
  }
  print(String(format: "environment: %d contexts, %d refreshes, log-likelihood of refresh %d's draws %.4f", nc, R0.slots[0].kept.count, point, ll))
  let wv = route_sim_w(R0.sims[0], Int32(point))!
  var shown = 0
  for i in 0..<(nc * FS) where dw[i] != 0 && shown < 6 {
    let keep = wv[i]; var up = 0.0, down = 0.0, scratch = [Float](repeating: 0, count: TASKS * nc * FS)
    wv[i] = keep + eps; route_sim_credit(R0.sims[0], Int32(point), &scratch, &up); wv[i] = keep - eps; route_sim_credit(R0.sims[0], Int32(point), &scratch, &down); wv[i] = keep
    print(String(format: "  w[context %d][%d]: credit %+.5f, central difference %+.5f", i / FS, i % FS, dw[i], (up - down) / Double(2 * eps)))
    shown += 1
  }
  if env["GRAPH"] == nil { exit(0) }
  // the int8 network against the fp32 reference (policy.swift and learn.swift's graph: the same function, plain fp32
  // weights W = g V / |V|), through the window's refreshes: w, and each task's gradient (the reference's dW mapped to
  // (dV, dg, db) by the same chain rule)
  let (B, K, R, packed, seg) = gather([DoneRun(R0.slots[0])], [1])
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
  route_stream_one(&measured, 1, 12582912, measured.compute.0, &R0.slots[0].stream)
  var w = deployed; var s = R0.slots[0].stream
  route_sim_begin(R0.sims[0], &s, &w, 0, 1, 1); _ = route_sim_decide(R0.sims[0], R0.decisions, R0.tokens[0])
  let nl = Int(R0.decisions[0].lanes), nc = Int(R0.decisions[0].contexts)
  memcpy(R0.laneP, R0.laneTokens(0), L * IN * 4); for q in 0..<L { R0.maskP[q] = q < nl ? 1 : 0 }
  memset(R0.stateP, 0, 2 * D * 4); memcpy(R0.tokP, R0.tokens[0], nc * IN * 4); for c in 0..<nc { R0.idxP[c] = Int32(L - 1) }
  R0.forward(1, nc)
  let scratch = UnsafeMutablePointer<Float>.allocate(capacity: route_forward_scratch(Int32(CONTEXTS))), wc = UnsafeMutablePointer<Float>.allocate(capacity: CONTEXTS * FS)
  var h = [Float](repeating: 0, count: 2 * D)
  route_forward(&params, R0.laneTokens(0), Int32(nl), R0.tokens[0], Int32(nc), &h, wc, scratch)
  var worst: Float = 0, size: Float = 0
  for i in 0..<(nc * FS) { worst = max(worst, abs(wc[i] - R0.wP[i])); size = max(size, abs(R0.wP[i])) }
  var hw: Float = 0; for i in 0..<(2 * D) { hw = max(hw, abs(h[i] - R0.nextP[i])) }
  print(String(format: "C forward (int8) against the GPU's, %d lanes, %d contexts: w largest difference %.1e of %.1e, next state %.1e", nl, nc, worst, size, hw))
  let reps = 2000, t0 = Date()
  for _ in 0..<reps { route_forward(&params, R0.laneTokens(0), Int32(nl), R0.tokens[0], Int32(nc), &h, wc, scratch) }
  print(String(format: "C forward, a refresh of the measured cluster (%d lanes, %d contexts): %.1f us on one core", nl, nc, Date().timeIntervalSince(t0) / Double(reps) * 1e6))
  exit(0)
}

print(online ? "online: continuing on the deployment's distribution" : "offline: random fabrics, streams, windows")
print("\(arm), \(nlay.count) parameters; \(windows) windows x 4 rollouts a step, up to \(KMAX) refreshes a window; T / floor on the measured cluster at \(sizes.map { Int($0) }) bytes, refreshed every 100 us, 100 us late")
evaluate("start", wait: false)
if env["BENCH"] != nil {   // each part's latency alone, beside the arithmetic it is
  let P = nlay.count, B = 4 * windows
  // int8 operations a row: forward, lane rows (layer 0: gates and feed-forward; layer 1: gates) and context rows (both
  // layers, the readout); the learning call's backward twice the forward's again (dgrad and wgrad)
  let laneOps = 2 * (D * 64 + 2 * D * D + 2 * D * E + 2 * D * D), ctxOps = 2 * (D * 64 + 2 * (2 * D * D + 2 * D * E) + FS * D)
  memset(laneP, 0, B * KMAX * L * IN * 4); memset(maskP, 0, B * KMAX * L * 4); memset(stateP, 0, B * 2 * D * 4)
  for r in [8192, 32768, 131072] {
    for i in 0..<r { R0.idxP[i] = Int32((i % B) * L + L - 1) }
    R0.forward(B, r)   // first use apart
    var t = 0.0; for _ in 0..<10 { let t0 = Date(); R0.forward(B, r); t += Date().timeIntervalSince(t0) }
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
    for i in 0..<131072 { R0.idxP[i] = Int32((i % B) * L + L - 1) }
    R0.forward(B, 131072); show("forward call, 131072 rows")
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
    route_sim_begin(R0.sims[0], &s, &w, 0, 1, UInt32(k))
    while route_sim_decide(R0.sims[0], R0.decisions, R0.tokens[0]) == 1 { route_sim_answer(R0.sims[0], w0) }
    let t1 = Date()
    var part = [Float](repeating: 0, count: TASKS * Int(route_sim_contexts(R0.sims[0])) * FS), ll = 0.0
    for p in 0..<Int(route_sim_points(R0.sims[0])) { route_sim_credit(R0.sims[0], Int32(p), &part, &ll) }
    credT += Date().timeIntervalSince(t1); simT += t1.timeIntervalSince(t0); wins += 1
    var c = [UInt64](repeating: 0, count: 4); route_sim_counts(R0.sims[0], &c); events += c[0]; draws += c[1]; entries += c[2]; sends += c[3]
  }
  print(String(format: "environment, one thread, %d training windows: %.0f us a window simulated (%.0f events, %.0f draws, %.0f table entries each), %.0f us credited", wins, simT / Double(wins) * 1e6,
               Double(events) / Double(wins), Double(draws) / Double(wins), Double(entries) / Double(wins), credT / Double(wins) * 1e6))
  print(String(format: "  per event %.0f ns, per draw %.0f ns, per table entry %.1f ns (simulation); credit %.0f ns a draw", simT / Double(events) * 1e9, simT / Double(max(draws, 1)) * 1e9, simT / Double(max(entries, 1)) * 1e9, credT / Double(max(draws, 1)) * 1e9))
  print(String(format: "  so %d threads: %.0f windows a second", ProcessInfo.processInfo.activeProcessorCount, Double(ProcessInfo.processInfo.activeProcessorCount) * Double(wins) / (simT + credT)))
  exit(0)
}
// ---- producers and the learner. The producers (runs.swift's workers, every hardware thread) run window groups of four
// rollouts; the main thread serves their forwards, and starts a group whenever fewer runs are runnable than there are
// workers (a free slot group permitting). The learner thread takes every ready group at each step (as many as the
// learning batch's buffers hold), learns, applies every exchanged mean completed so far in order (or its own, alone),
// posts its gradients, and publishes theta, which the main thread copies into the producers' network between forwards.
// A group's version: the optimizer steps theta had taken when it began; its lag at its step, the steps since ----
var counted = [UInt64](repeating: 0, count: 4), transactions = 0, learned = 0, lagSum = 0, lagMax = 0, groupsLearned = 0, applied = 0
let shareLock = NSCondition()
var ready: [(runs: [DoneRun], version: Int)] = [], published = 0, finishedLearning = false
var groupVersion = [Int](repeating: 0, count: TX / 4), groupLeft = [Int](repeating: 0, count: TX / 4), freeGroups = Array((0..<(TX / 4)).reversed())
var thetaVersion = 0   // the learner's optimizer steps, as the producers last copied them
func startGroup(_ g: Int) {
  let (s, w, h) = newWindow(), k0 = UInt32(truncatingIfNeeded: route_next(&rng))
  for q in 0..<4 {
    let t = 4 * g + q
    R0.slots[t].stream = s; R0.slots[t].state = h
    R0.begin(t, window: w, mode: 0, key0: k0, key: UInt32(truncatingIfNeeded: route_next(&rng)))
  }
  groupVersion[g] = thetaVersion; groupLeft[g] = 4
  R0.submit((0..<4).map { 4 * g + $0 })
}
var alpha = [0.0, 0.0, 0.0], lastCosts: [route_costs] = []
func apply(_ mean: [Float], _ gram: [Float]) {
  let P3 = TASKS * nlay.count
  mean.withUnsafeBufferPointer { memcpy(net.grads.contents(), $0.baseAddress!, P3 * 4) }
  for i in 0..<9 { floats(net.gram, 9)[i] = gram[i] }
  alpha = net.apply(rate: rate); route_tel_add(tel(RT_OPTIMIZER_STEPS), 1)
}
func learnStep(_ batch: [(runs: [DoneRun], version: Int)]) {
  let runs = batch.flatMap { $0.runs }, T = runs.map { $0.costs.throughput }
  let adv = (0..<runs.count).map { t -> Double in let s = t / 4 * 4, total = T[s] + T[s + 1] + T[s + 2] + T[s + 3]; return T[t] - (total - T[t]) / 3 }
  for g in batch { let lag = net.step - g.version; route_tel_add(tel(RT_LAG_SUM), UInt64(max(lag, 0))); route_tel_max(tel(RT_LAG_MAX), UInt64(max(lag, 0))) }
  route_tel_add(tel(RT_LAG_GROUPS), UInt64(batch.count))
  for r in runs { for (q, k) in [RT_EVENTS, RT_DRAWS, RT_ENTRIES, RT_SENDS].enumerated() { route_tel_add(tel(k), r.counts[q]) } }
  route_tel_add(tel(RT_TRANSACTIONS), UInt64(runs.count)); route_tel_add(tel(RT_GROUPS), UInt64(batch.count)); lastCosts = runs.map { $0.costs }
  let (B, K, R, packed, seg) = gather(runs, adv)
  gradients(B, K, R, packed, seg)
  if world == 1 { alpha = net.apply(rate: rate); route_tel_add(tel(RT_OPTIMIZER_STEPS), 1); return }
  exchange.add(floats(net.grads, TASKS * nlay.count))
  for (mean, gram) in exchange.ready() { apply(mean, gram) }
}
let learner = Thread {
  while true {
    shareLock.lock()
    while ready.isEmpty && !finishedLearning { shareLock.wait() }
    if finishedLearning { shareLock.unlock(); break }
    var take = 0   // every ready group, as many as the learning batch's buffers hold
    while take < ready.count && 4 * (take + 1) <= BATCH { take += 1 }
    let batch = Array(ready.prefix(take)); ready.removeFirst(take)
    shareLock.unlock()
    learnStep(batch)
    learned += 1; route_tel_add(tel(RT_STEPS), 1); route_tel_phase(tel(RT_PHASE_FIRST_STEP))
    shareLock.lock(); published = net.step; shareLock.unlock(); R0.wake()
    if learned >= steps { shareLock.lock(); finishedLearning = true; shareLock.broadcast(); shareLock.unlock(); R0.wake(); break }
  }
}
learner.qualityOfService = .userInitiated; learner.stackSize = 16 << 20
R0.startWorkers()
if world > 1 { exchange.start() }
learner.start()
while true {
  shareLock.lock(); let over = finishedLearning, version = published; shareLock.unlock()
  if over { break }
  if version != thetaVersion { syncForward(); thetaVersion = version }   // the learner's theta, between forwards
  while R0.runnable < R0.workers.count, let g = freeGroups.popLast() { startGroup(g) }
  route_tel_set(tel(RT_RUNS_IN_FLIGHT), UInt64(TX - 4 * freeGroups.count))
  let ended = R0.service()
  for t in ended {
    let st = R0.slots[t].state
    if pool.count < 4096 { pool.append(st) } else if route_uniform(&rng) < 0.05 { pool[Int(route_uniform(&rng) * 4096) % 4096] = st }
    let g = t / 4; groupLeft[g] -= 1
    if groupLeft[g] == 0 {
      let done = (runs: (0..<4).map { DoneRun(R0.slots[4 * g + $0]) }, version: groupVersion[g])
      freeGroups.append(g)
      shareLock.lock(); ready.append(done); shareLock.signal(); shareLock.unlock()
    }
  }
}
if world > 1 { exchange.finish(); for (mean, gram) in exchange.ready() + exchange.drain() { apply(mean, gram) } }   // the means still in flight, in order
route_tel_phase(tel(RT_PHASE_DONE))
evaluate("theta at the end, \(net.step) optimizer steps", wait: true)
if let path = env["THETA"], rank == 0 { try! Data(bytes: net.theta.contents(), count: nlay.count * 4).write(to: URL(fileURLWithPath: path)); print("parameters written to \(path)") }
if let path = env["THETA_INT8"], rank == 0 { net.writeCodes(path); print("int8 weights (for deployment: tests/switch.c's ROUTE_THETA) written to \(path)") }
route_exchange_close()
