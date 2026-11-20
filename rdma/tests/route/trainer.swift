// The trainer (trainer.h): the int8 network's learning (net.swift), the producers (runs.swift), the gradient exchange
// (exchange.c), behind the C interface the Python driver (py/route) calls. Every window it runs is a case the driver
// pushed; it draws no data of its own.
import Accelerate
import Foundation
import Metal
import MetalPerformanceShadersGraph

func tel(_ k: route_tel) -> Int32 { Int32(k.rawValue) }
var failureText: UnsafeMutablePointer<CChar>? = nil
func fail(_ s: String) { free(failureText); failureText = strdup(s) }

// a case's four rollouts as slots `base ..< base + 4` of a runner: stream, state, window, mode, keys
func place(_ r: Runner, _ c: UnsafePointer<route_case>, base: Int) {
  var s = route_stream(); route_case_stream(c, &s)
  let h = withUnsafeBytes(of: c.pointee.h) { Array($0.bindMemory(to: Float.self)) }
  let keys = withUnsafeBytes(of: c.pointee.key) { Array($0.bindMemory(to: UInt32.self)) }, keys0 = withUnsafeBytes(of: c.pointee.key0) { Array($0.bindMemory(to: UInt32.self)) }
  for q in 0..<4 {
    r.slots[base + q].stream = s; r.slots[base + q].state = h
    r.begin(base + q, window: c.pointee.w, mode: Int32(c.pointee.mode), key0: keys0[q], key: keys[q])
  }
}

// a finished run as the learning batch reads it (copied out of its slot, which the producers then reuse)
struct DoneRun {
  let kept: [Kept], start: [Float], draws: [PointDraws], costs: route_costs, counts: [UInt64]
  init(_ s: Slot) { kept = s.kept; start = s.start; draws = s.draws; costs = s.costs; counts = s.counts }
}

// the gradients' mean across machines: the learner adds each step's gradients into an accumulator; the exchange's
// thread runs a round whenever this rank has accumulated a step (or has finished), back to back: its payload the
// accumulated sum, its step count and its done flag, summed across ranks (every rank's k-th round is one), so a round's
// mean is over every learning step in it, every rank's. ready() returns the means completed since, in round order, with
// their Gram matrices, never waiting: every rank applies the same means in the same order.
final class Exchange {
  let n: Int, P: Int, world: Int, cond = NSCondition(), sum: (UnsafeMutablePointer<Float>, Int) -> Bool
  var acc: [Float], steps = 0, finished = false, over = false, failed = false, done: [([Float], [Float])] = []
  init(P: Int, world: Int, sum: @escaping (UnsafeMutablePointer<Float>, Int) -> Bool) { self.P = P; self.world = world; self.sum = sum; n = TASKS * P; acc = [Float](repeating: 0, count: TASKS * P) }
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
        if !payload.withUnsafeMutableBufferPointer({ sum($0.baseAddress!, n + 2) }) { cond.lock(); failed = true; over = true; cond.broadcast(); cond.unlock(); return }
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
    th.qualityOfService = .userInteractive; th.start()
  }
  func ready() -> [([Float], [Float])] { cond.lock(); defer { cond.unlock() }; let r = done; done = []; return r }
  func finish() { cond.lock(); finished = true; cond.broadcast(); cond.unlock() }
  func drain() -> [([Float], [Float])] { cond.lock(); while !over { cond.wait() }; let r = done; done = []; cond.unlock(); return r }
}

final class Trainer {
  let rank: Int, world: Int, steps: Int, rate: Double, ROWS: Int, BATCH: Int, KMAX: Int, TX: Int, library: String
  let device = MTLCreateSystemDefaultDevice()!, queue: MTLCommandQueue, exchangeQueue: MTLCommandQueue
  let lay = Layout(full: false), nlay = NetLayout()
  let net: Net, fnet: Net, R0: Runner
  // the learning batch, in shared buffers (exact row counts: the int8 network's kernels take any)
  let laneBuf: MTLBuffer, maskBuf: MTLBuffer, stateBuf: MTLBuffer, tokBuf: MTLBuffer, idxBuf: MTLBuffer, wBuf: MTLBuffer
  let laneP: UnsafeMutablePointer<Float>, maskP: UnsafeMutablePointer<Float>, stateP: UnsafeMutablePointer<Float>, tokP: UnsafeMutablePointer<Float>, idxP: UnsafeMutablePointer<Int32>, wP: UnsafeMutablePointer<Float>
  // the batch's credit buffers (grown when a batch needs more): the draws copied whole a refresh, each row's first draw,
  // behaviour w and task-0 scale
  var creditDraws: MTLBuffer? = nil, creditFirst: MTLBuffer? = nil, creditW: MTLBuffer? = nil, creditScale0: MTLBuffer? = nil
  lazy var exchange = Exchange(P: nlay.count, world: world) { [unowned self] p, n in self.meshSum(p, n) }

  func shared(_ floats: Int) -> MTLBuffer { device.makeBuffer(length: max(16, floats * 4), options: .storageModeShared)! }
  func floats(_ b: MTLBuffer, _ n: Int) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: n) }

  init?(_ c: route_config) {
    rank = Int(c.rank); world = Int(c.world); steps = Int(c.steps); rate = c.rate
    ROWS = Int(c.forward_rows); BATCH = Int(c.learn_runs); KMAX = Int(c.refreshes); TX = ROWS / Int(ROUTE_CONTEXTS) / 4 * 4
    guard KMAX >= 1 && KMAX <= Int(ROUTE_POINTS), TX >= 4, BATCH >= 4 else { fail("configuration: refreshes 1..\(ROUTE_POINTS), forward_rows at least 4 runs' contexts, learn_runs at least 4"); return nil }
    library = String(cString: c.metallib)
    queue = device.makeCommandQueue()!; exchangeQueue = device.makeCommandQueue()!
    route_tel_phase(tel(RT_PHASE_START))
    var thetaRng = route_rng(s: c.seed)
    let learning = Net(device, queue, library: library, runs: BATCH, refreshes: KMAX, contexts: ROWS, initial: nlay.initial(&thetaRng))
    if let p = c.theta_in {   // continue from these parameters: NetLayout's, or the earlier fp32 layout's (reparametrized exactly)
      guard let d = FileManager.default.contents(atPath: String(cString: p)) else { fail("theta_in: cannot read \(String(cString: p))"); return nil }
      let count = nlay.count
      if d.count == lay.count * 4 { let t = nlay.from(old: d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, lay); memcpy(learning.theta.contents(), t, t.count * 4) }
      else if d.count == count * 4 { d.withUnsafeBytes { memcpy(learning.theta.contents(), $0.baseAddress!, count * 4) } }
      else { fail("theta_in: parameters of another layout (\(d.count) bytes)"); return nil }
      learning.refresh()
    }
    net = learning
    fnet = Net(device, device.makeCommandQueue()!, library: library, runs: TX, refreshes: 1, contexts: ROWS, initial: Array(UnsafeBufferPointer(start: net.theta.contents().bindMemory(to: Float.self, capacity: nlay.count), count: nlay.count)))
    R0 = Runner(device, net: fnet, slots: TX, rows: ROWS)
    func sh(_ n: Int) -> MTLBuffer { MTLCreateSystemDefaultDevice()!.makeBuffer(length: max(16, n * 4), options: .storageModeShared)! }
    laneBuf = sh(KMAX * BATCH * L * IN); maskBuf = sh(KMAX * BATCH * L); stateBuf = sh(BATCH * 2 * D); tokBuf = sh(ROWS * IN); idxBuf = sh(ROWS); wBuf = sh(ROWS * FS)
    laneP = laneBuf.contents().bindMemory(to: Float.self, capacity: KMAX * BATCH * L * IN); maskP = maskBuf.contents().bindMemory(to: Float.self, capacity: KMAX * BATCH * L)
    stateP = stateBuf.contents().bindMemory(to: Float.self, capacity: BATCH * 2 * D); tokP = tokBuf.contents().bindMemory(to: Float.self, capacity: ROWS * IN)
    idxP = idxBuf.contents().bindMemory(to: Int32.self, capacity: ROWS); wP = wBuf.contents().bindMemory(to: Float.self, capacity: ROWS * FS)
    thetaSnapshot = sh(nlay.count); thetaStaging = sh(nlay.count)
    groupVersion = [Int](repeating: 0, count: TX / 4); groupLeft = [Int](repeating: 0, count: TX / 4); freeGroups = Array((0..<(TX / 4)).reversed())
    route_tel_phase(tel(RT_PHASE_NETWORKS))
    if world > 1 {
      guard let id = c.mesh_id else { fail("mesh_id: the bootstrap root's host:port"); return nil }
      let declaring = exchangeQueue.makeCommandBuffer()!
      if route_exchange_open(c.rank, c.world, id, size_t(TASKS * nlay.count + 2), Unmanaged.passUnretained(declaring).toOpaque()) != 0 { fail("exchange: " + String(cString: route_exchange_failure())); route_exchange_close(); return nil }
      route_tel_phase(tel(RT_PHASE_JOINED))
    }
  }

  // ---- the gradients' sum across ranks: the step in a command buffer of the exchange's own queue, in place through
  // the lease; this thread parks until it completes ----
  func meshSum(_ p: UnsafeMutablePointer<Float>, _ n: Int) -> Bool {
    if world == 1 { return true }
    let lease = route_exchange_buffer()!
    memcpy(lease, p, n * 4)
    let cb = exchangeQueue.makeCommandBuffer()!
    if route_exchange_encode(Unmanaged.passUnretained(cb).toOpaque()) != 0 { fail("exchange: " + String(cString: route_exchange_failure())); return false }
    cb.commit(); cb.waitUntilCompleted()
    if cb.status != .completed || route_exchange_status() != 0 { fail("exchange: " + String(cString: route_exchange_failure()) + " \(cb.error.map { "\($0)" } ?? "")"); return false }
    memcpy(p, lease, n * 4)
    return true
  }

  // ---- the learning batch over finished runs, throughput advantages `adv`: lanes [run][refresh][lane]; a row each
  // context a refresh drew from (its draws in it), reading its refresh's state (lane row (b K + k) L + L - 1),
  // contiguous a refresh (seg); the rows' draws for the GPU's credit. Returns (B, K, R, credit, seg) ----
  func grow(_ b: inout MTLBuffer?, _ bytes: Int) -> MTLBuffer {
    if b == nil || b!.length < bytes { b = device.makeBuffer(length: max(16, bytes * 2), options: .storageModeShared)! }
    return b!
  }
  func gather(_ runs: [DoneRun], _ adv: [Double]) -> (Int, Int, Int, Net.Credit, [UInt32]) {
    let B = runs.count, K = KMAX
    precondition(B <= BATCH, "a learning batch past its buffers")
    memset(laneP, 0, B * K * L * IN * 4); memset(maskP, 0, B * K * L * 4); memset(stateP, 0, B * 2 * D * 4)
    var nrows = 0, ndraws = 0
    for r in runs { for pd in r.draws { nrows += pd.ctx.count; ndraws += pd.draws.count } }
    precondition(nrows <= ROWS, "learning rows past the buffers")
    let dbuf = grow(&creditDraws, ndraws * MemoryLayout<route_draw>.stride), fbuf = grow(&creditFirst, (nrows + 1) * 4)
    let wbuf = grow(&creditW, nrows * FS * 4), sbuf = grow(&creditScale0, nrows * 4)
    let dp = dbuf.contents(), fp = fbuf.contents().bindMemory(to: UInt32.self, capacity: nrows + 1)
    let wp = wbuf.contents().bindMemory(to: Float.self, capacity: nrows * FS), s0 = sbuf.contents().bindMemory(to: Float.self, capacity: nrows)
    var rows = 0, at = 0, seg = [UInt32](repeating: 0, count: 2 * B * K)
    let scale = 1 / Double(B)
    for (b, r) in runs.enumerated() {
      precondition(r.kept.count <= K, "a window past the learning batch's refreshes")
      for i in 0..<(2 * D) { stateP[b * 2 * D + i] = r.start[i] }
      for k in 0..<K {
        seg[2 * (b * K + k)] = UInt32(rows)
        if k < r.kept.count {
          let d = r.kept[k], pd = r.draws[k]
          memcpy(laneP + (b * K + k) * L * IN, d.lanes, L * IN * 4)
          for q in 0..<L { maskP[(b * K + k) * L + q] = q < d.lanesUsed ? 1 : 0 }
          d.tokens.withUnsafeBufferPointer { tp in for (j, c) in pd.ctx.enumerated() { memcpy(tokP + (rows + j) * IN, tp.baseAddress! + Int(c) * IN, IN * 4) } }
          for j in 0..<pd.ctx.count { idxP[rows + j] = Int32((b * K + k) * L + L - 1); fp[rows + j] = UInt32(at + Int(pd.first[j])); s0[rows + j] = Float(adv[b] * scale) }
          pd.draws.withUnsafeBytes { memcpy(dp + at * MemoryLayout<route_draw>.stride, $0.baseAddress!, $0.count) }
          pd.w.withUnsafeBufferPointer { memcpy(wp + rows * FS, $0.baseAddress!, $0.count * 4) }
          rows += pd.ctx.count; at += pd.draws.count
        }
        seg[2 * (b * K + k) + 1] = UInt32(rows)
      }
    }
    fp[rows] = UInt32(at)
    return (B, K, rows, Net.Credit(draws: dbuf, count: at, first: fbuf, wrows: wbuf, scale0: sbuf, scale: Float(scale)), seg)
  }
  func gradients(_ B: Int, _ K: Int, _ R: Int, _ credit: Net.Credit, _ seg: [UInt32]) {
    let t0 = route_tel_now()
    net.learn(lanes: laneBuf, mask: maskBuf, h0: stateBuf, tokens: tokBuf, pos: idxBuf, seg: seg, B: B, K: K, R: R, credit: credit, w: wBuf)
    route_tel_add(tel(RT_LEARN_NS), route_tel_now() - t0)
  }

  // ---- producers and the learner. The producers run window groups of four rollouts; the service thread forwards
  // for them, and starts a pushed group whenever fewer runs are runnable than there are workers. The learner takes
  // every ready group at each step (as many as the learning batch holds), learns, applies every exchanged mean
  // completed so far in order (or its own, alone), posts its gradients, and publishes theta, which the service thread
  // copies into the producers' network between forwards. A group's version: the optimizer steps theta had taken when
  // it began; its lag at its step, the steps since ----
  let shareLock = NSCondition(), caseLock = NSCondition()
  var ready: [(runs: [DoneRun], version: Int)] = [], published = 0, finishedLearning = false, learned = 0
  var groupVersion: [Int], groupLeft: [Int], freeGroups: [Int], cases: [route_case] = [], endedStates: [Float] = []
  var thetaVersion = 0
  let thetaSnapshot: MTLBuffer, thetaStaging: MTLBuffer
  let served = DispatchSemaphore(value: 0)

  func applyMean(_ mean: [Float], _ gram: [Float]) {
    mean.withUnsafeBufferPointer { memcpy(net.grads.contents(), $0.baseAddress!, TASKS * nlay.count * 4) }
    for i in 0..<9 { floats(net.gram, 9)[i] = gram[i] }
    _ = net.apply(rate: rate); route_tel_add(tel(RT_OPTIMIZER_STEPS), 1)
  }
  func learnStep(_ batch: [(runs: [DoneRun], version: Int)]) {
    let runs = batch.flatMap { $0.runs }, T = runs.map { $0.costs.throughput }
    let adv = (0..<runs.count).map { t -> Double in let s = t / 4 * 4, total = T[s] + T[s + 1] + T[s + 2] + T[s + 3]; return T[t] - (total - T[t]) / 3 }
    for g in batch { let lag = net.step - g.version; route_tel_add(tel(RT_LAG_SUM), UInt64(max(lag, 0))); route_tel_max(tel(RT_LAG_MAX), UInt64(max(lag, 0))) }
    route_tel_add(tel(RT_LAG_GROUPS), UInt64(batch.count))
    for r in runs { for (q, k) in [RT_EVENTS, RT_DRAWS, RT_ENTRIES, RT_SENDS].enumerated() { route_tel_add(tel(k), r.counts[q]) } }
    route_tel_add(tel(RT_TRANSACTIONS), UInt64(runs.count)); route_tel_add(tel(RT_GROUPS), UInt64(batch.count))
    let (B, K, R, credit, seg) = gather(runs, adv)
    gradients(B, K, R, credit, seg)
    if credit.count > 0, let lr = net.llBuf {   // the draws' importance ratios (current over behaviour policy)
      let p = lr.contents().bindMemory(to: Float.self, capacity: credit.count)
      var sum = 0.0, most: Float = 0
      for i in 0..<credit.count { sum += Double(exp(p[i])); most = max(most, abs(p[i])) }
      route_tel_add(tel(RT_RATIO_DRAWS), UInt64(credit.count)); route_tel_add(tel(RT_RATIO_SUM_E6), UInt64(sum * 1e6)); route_tel_max(tel(RT_LOGRATIO_MAX_E3), UInt64(most * 1e3))
    }
    if world == 1 { _ = net.apply(rate: rate); route_tel_add(tel(RT_OPTIMIZER_STEPS), 1); return }
    exchange.add(floats(net.grads, TASKS * nlay.count))
    for (mean, gram) in exchange.ready() { applyMean(mean, gram) }
  }
  func over() { shareLock.lock(); finishedLearning = true; shareLock.broadcast(); shareLock.unlock(); R0.wake(); caseLock.lock(); caseLock.broadcast(); caseLock.unlock() }
  func start() {
    if steps == 0 { over(); served.signal(); return }
    let learner = Thread { [unowned self] in
      while true {
        shareLock.lock()
        while ready.isEmpty && !finishedLearning { shareLock.wait() }
        if finishedLearning { shareLock.unlock(); break }
        var take = 0   // every ready group, as many as the learning batch's buffers hold
        while take < ready.count && 4 * (take + 1) <= BATCH { take += 1 }
        let batch = Array(ready.prefix(take)); ready.removeFirst(take)
        shareLock.unlock()
        learnStep(batch)
        if world > 1 && exchange.failed { over(); break }
        shareLock.lock(); learned += 1; memcpy(thetaSnapshot.contents(), net.theta.contents(), nlay.count * 4); published = net.step; shareLock.unlock()
        route_tel_add(tel(RT_STEPS), 1); route_tel_phase(tel(RT_PHASE_FIRST_STEP)); R0.wake()
        if learned >= steps { over(); break }
      }
    }
    learner.qualityOfService = .userInitiated; learner.stackSize = 16 << 20
    let service = Thread { [unowned self] in
      while true {
        shareLock.lock(); let done = finishedLearning, version = published; shareLock.unlock()
        if done { break }
        if version != thetaVersion {   // the learner's theta: its snapshot staged, copied in the forwards' queue order
          shareLock.lock(); memcpy(thetaStaging.contents(), thetaSnapshot.contents(), nlay.count * 4); shareLock.unlock()
          fnet.refresh(from: thetaStaging); thetaVersion = version
        }
        while R0.runnable < R0.workers.count {   // a pushed group started on demand (its theta the freshest)
          caseLock.lock()
          guard !cases.isEmpty, let g = freeGroups.popLast() else { caseLock.unlock(); break }
          var c = cases.removeFirst(); caseLock.broadcast(); caseLock.unlock()
          place(R0, &c, base: 4 * g)
          groupVersion[g] = thetaVersion; groupLeft[g] = 4
          R0.submit((0..<4).map { 4 * g + $0 })
        }
        caseLock.lock(); route_tel_set(tel(RT_RUNS_IN_FLIGHT), UInt64(TX - 4 * freeGroups.count)); caseLock.unlock()
        let ended = R0.service()
        var states: [Float] = []
        for t in ended {
          states += R0.slots[t].state
          let g = t / 4; groupLeft[g] -= 1
          if groupLeft[g] == 0 {
            let done = (runs: (0..<4).map { DoneRun(R0.slots[4 * g + $0]) }, version: groupVersion[g])
            caseLock.lock(); freeGroups.append(g); caseLock.broadcast(); caseLock.unlock()
            shareLock.lock(); ready.append(done); shareLock.signal(); shareLock.unlock()
          }
        }
        if !states.isEmpty { caseLock.lock(); endedStates += states; caseLock.broadcast(); caseLock.unlock() }
      }
      served.signal()
    }
    service.qualityOfService = .userInitiated; service.stackSize = 16 << 20
    R0.startWorkers()
    if world > 1 { exchange.start() }
    learner.start(); service.start()
  }
  func room(_ timeout: Double) -> Int {
    let until = Date(timeIntervalSinceNow: timeout)
    caseLock.lock(); defer { caseLock.unlock() }
    while true {
      shareLock.lock(); let done = finishedLearning; shareLock.unlock()
      if done { return -1 }
      // the queue holds a group's case for every group: whatever the driver falls behind by, its next draw is that
      // much larger (each draw's fixed cost spread over more cases), so the supply settles above the demand
      let n = TX / 4 - cases.count
      if n > 0 || !caseLock.wait(until: until) { return max(n, 0) }
    }
  }
  func push(_ c: UnsafePointer<route_case>, _ n: Int) { caseLock.lock(); for i in 0..<n { cases.append(c[i]) }; caseLock.unlock(); R0.wake() }
  func takeEnded(_ out: UnsafeMutablePointer<Float>, _ room: Int) -> Int {
    caseLock.lock(); defer { caseLock.unlock() }
    let n = min(room, endedStates.count / (2 * D))
    endedStates.withUnsafeBufferPointer { memcpy(out, $0.baseAddress!, n * 2 * D * 4) }
    endedStates.removeFirst(n * 2 * D)
    return n
  }
  func finish() {
    served.wait()
    if world > 1 && steps > 0 { exchange.finish(); for (mean, gram) in exchange.ready() + exchange.drain() { applyMean(mean, gram) } }   // the means in flight, in order
    route_tel_phase(tel(RT_PHASE_DONE))
  }

  // ---- evaluation: a runner and network of its own, theta a snapshot, one at a time ----
  let evalLock = NSLock()
  var evaluator: Runner? = nil
  func evaluate(_ c: UnsafePointer<route_case>, _ n: Int, _ out: UnsafeMutablePointer<Double>) {
    evalLock.lock(); defer { evalLock.unlock() }
    if evaluator == nil || evaluator!.n < 4 * n {
      evaluator = Runner(device, net: Net(device, device.makeCommandQueue()!, library: library, runs: 4 * n, refreshes: 1, contexts: 4 * n * Int(ROUTE_CONTEXTS), initial: [Float](repeating: 0, count: nlay.count)), slots: 4 * n, rows: 4 * n * Int(ROUTE_CONTEXTS))
    }
    let ev = evaluator!
    shareLock.lock(); memcpy(ev.net.theta.contents(), net.theta.contents(), nlay.count * 4); shareLock.unlock()
    ev.net.refresh()
    for i in 0..<n { place(ev, c + i, base: 4 * i) }
    var active = Array(0..<(4 * n))
    while !active.isEmpty { let ended = Set(ev.tick(active, keep: false)); active = active.filter { !ended.contains($0) } }
    for t in 0..<(4 * n) { out[t] = ev.costs(t).throughput }
  }
  func save(_ fp32: UnsafePointer<CChar>?, _ int8: UnsafePointer<CChar>?) -> Bool {
    if let p = fp32 { guard FileManager.default.createFile(atPath: String(cString: p), contents: Data(bytes: net.theta.contents(), count: nlay.count * 4)) else { fail("cannot write \(String(cString: p))"); return false } }
    if let p = int8 { net.writeCodes(String(cString: p)) }
    return true
  }
  func forwardOne(_ lanes: UnsafePointer<Float>, _ nl: Int, _ tokens: UnsafePointer<Float>, _ nc: Int, _ state: UnsafeMutablePointer<Float>, _ w: UnsafeMutablePointer<Float>) {
    memcpy(R0.laneP, lanes, L * IN * 4); for q in 0..<L { R0.maskP[q] = q < nl ? 1 : 0 }
    memcpy(R0.stateP, state, 2 * D * 4); memcpy(R0.tokP, tokens, nc * IN * 4); for c in 0..<nc { R0.idxP[c] = Int32(L - 1) }
    R0.forward(1, nc)
    memcpy(w, R0.wP, nc * FS * 4); memcpy(state, R0.nextP, 2 * D * 4)
  }
  func xbench(_ rounds: Int, _ seconds: UnsafeMutablePointer<Double>, _ wrong: UnsafeMutablePointer<Int>) -> Bool {   // each round's values the rank's own, so its mean is known
    let P = TASKS * nlay.count, g = floats(net.grads, P)
    wrong.pointee = 0
    for r in 0..<rounds {
      for i in 0..<P { g[i] = Float(rank + 1 + r % 5 + i % 7) }
      let t0 = route_tel_now()
      if !meshSum(g, P) { return false }
      let inv = 1 / Float(world); for i in 0..<P { g[i] *= inv }
      seconds[r] = Double(route_tel_now() - t0) / 1e9
      let base = Float(world + 1) / 2 + Float(r % 5)
      for i in 0..<P where g[i] != base + Float(i % 7) { wrong.pointee += 1 }
    }
    return true
  }

  // ---- check: the environment's credit (channel 0: d log-likelihood / dw) against central differences on the first
  // case whose draws had a choice; graph, also the GPU's credit against the CPU's and the int8 network against the
  // fp32 reference (policy.swift and learn.swift's graph: the same function, plain fp32 weights) ----
  func td(_ b: MTLBuffer, _ shape: [Int], _ type: MPSDataType = .float32) -> MPSGraphTensorData { MPSGraphTensorData(b, shape: dims(shape), dataType: type) }
  func check(_ cs: UnsafePointer<route_case>, _ n: Int, eps: Float, graph: Bool) -> [String: Any] {
    var report: [String: Any] = [:], nc = 0, dw: [Float] = [], ll = 0.0, point = -1, tried = 0
    for i in 0..<n where point < 0 {
      tried += 1
      place(R0, cs + i, base: 0)
      var active = [0]; while !active.isEmpty { active = R0.tick(active, keep: true).isEmpty ? active : [] }
      nc = Int(route_sim_contexts(R0.sims[0])); dw = [Float](repeating: 0, count: TASKS * nc * FS)
      for p in 0..<R0.slots[0].kept.count { route_sim_credit(R0.sims[0], Int32(p), &dw, &ll); if dw.contains(where: { $0 != 0 }) { point = p; break } }
    }
    report["cases_tried"] = tried
    guard point >= 0 else { report["error"] = "no case's draws had a choice"; return report }
    report["contexts"] = nc; report["refreshes"] = R0.slots[0].kept.count; report["refresh"] = point; report["loglik"] = ll
    let wv = route_sim_w(R0.sims[0], Int32(point))!
    var pairs: [[String: Any]] = []
    for i in 0..<(nc * FS) where dw[i] != 0 && pairs.count < 6 {
      let keep = wv[i]; var up = 0.0, down = 0.0, scratch = [Float](repeating: 0, count: TASKS * nc * FS)
      wv[i] = keep + eps; route_sim_credit(R0.sims[0], Int32(point), &scratch, &up); wv[i] = keep - eps; route_sim_credit(R0.sims[0], Int32(point), &scratch, &down); wv[i] = keep
      pairs.append(["context": i / FS, "feature": i % FS, "credit": Double(dw[i]), "central_difference": (up - down) / Double(2 * eps)])
    }
    report["credit_vs_central_difference"] = pairs
    if !graph { return report }
    let (B, K, R, credit, seg) = gather([DoneRun(R0.slots[0])], [1])
    net.checking = true; gradients(B, K, R, credit, seg); net.checking = false
    let packed = net.dwBuf!
    do {   // the GPU's credit (net.metal) against the CPU's (sim.c route_sim_credit), each row's dw
      var worst: Float = 0, size: Float = 0, row = 0
      let dp = floats(packed, 3 * R * FS)
      for (k, pd) in R0.slots[0].draws.enumerated() {
        var part = [Float](repeating: 0, count: TASKS * nc * FS), l2 = 0.0
        route_sim_credit(R0.sims[0], Int32(k), &part, &l2)
        for c in pd.ctx {
          for t in 0..<TASKS { for f in 0..<FS { let a = dp[(t * R + row) * FS + f], b = part[(t * nc + Int(c)) * FS + f]; worst = max(worst, abs(a - b)); size = max(size, abs(b)) } }
          row += 1
        }
      }
      report["gpu_credit"] = ["rows": R, "draws": credit.count, "largest_difference": Double(worst), "largest": Double(size)]
    }
    report["fused_codes_vs_quant_rows"] = net.probeReport()
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
    report["w_relative_error"] = sqrt(e2 / max(n2, 1e-30))
    var tasks: [[String: Any]] = []
    for t in 0..<TASKS {
      let gr = nlay.chain(old: Array(UnsafeBufferPointer(start: floats(gRef[t], lay.count), count: lay.count)), theta: Array(UnsafeBufferPointer(start: floats(net.theta, P), count: P)), lay)
      var mats: [String: Any] = [:]
      for mt in nlay.mats {
        var dot = 0.0, a2 = 0.0, b2 = 0.0, d2 = 0.0
        for i in mt.v..<(mt.g + mt.rows * (mt.bias ? 2 : 1)) { let a = Double(gInt[t * P + i]), b = Double(gr[i]); dot += a * b; a2 += a * a; b2 += b * b; d2 += (a - b) * (a - b) }
        mats[mt.name] = ["cosine": dot / max(sqrt(a2 * b2), 1e-300), "relative_error": sqrt(d2 / max(b2, 1e-300))]
      }
      tasks.append(mats)
    }
    report["gradients_by_task"] = tasks
    return report
  }

  // ---- bench: each call's time beside the arithmetic it is, kernel by kernel, and the environment's on the cases ----
  func bench(_ cs: UnsafePointer<route_case>, _ n: Int) -> [String: Any] {
    var report: [String: Any] = [:]
    let P = nlay.count, B = min(BATCH, TX)
    let laneOps = 2 * (D * 64 + 2 * D * D + 2 * D * E + 2 * D * D), ctxOps = 2 * (D * 64 + 2 * (2 * D * D + 2 * D * E) + FS * D)
    memset(laneP, 0, B * KMAX * L * IN * 4); memset(maskP, 0, B * KMAX * L * 4); memset(stateP, 0, B * 2 * D * 4)
    func now() -> Double { Double(route_tel_now()) / 1e9 }
    var fw: [[String: Any]] = []
    for r in [ROWS / 32, ROWS / 8, ROWS / 2] {
      for i in 0..<r { R0.idxP[i] = Int32((i % B) * L + L - 1) }
      R0.forward(B, r)
      let t0 = now(); for _ in 0..<10 { R0.forward(B, r) }; let t = (now() - t0) / 10
      let ops = Double(B * L * laneOps + r * ctxOps)
      fw.append(["runs": B, "rows": r, "seconds": t, "int8_tops": ops / t / 1e12])
    }
    report["forward"] = fw
    func noCredit(_ r: Int) -> Net.Credit {   // r rows, no draws: the learning call's cost apart from credit
      func z(_ n: Int) -> MTLBuffer { let b = shared(max(n, 4)); memset(b.contents(), 0, max(n, 4) * 4); return b }
      return Net.Credit(draws: z(40), count: 0, first: z(r + 1), wrows: z(r * FS), scale0: z(r), scale: 0)
    }
    func segs(_ r: Int) -> [UInt32] {
      for i in 0..<r { idxP[i] = Int32((i * B * KMAX / r) * L + L - 1) }
      var seg = [UInt32](repeating: 0, count: 2 * B * KMAX)
      for s in 0..<(B * KMAX) { seg[2 * s] = UInt32(s * r / (B * KMAX)); seg[2 * s + 1] = UInt32((s + 1) * r / (B * KMAX)) }
      return seg
    }
    var lc: [[String: Any]] = []
    for r in [ROWS / 64, ROWS / 16, ROWS / 4] {
      let seg = segs(r), packed = noCredit(r)
      gradients(B, KMAX, r, packed, seg)
      let t0 = now(); for _ in 0..<5 { gradients(B, KMAX, r, packed, seg) }; let t = (now() - t0) / 5
      let ops = 3 * Double(B * KMAX * L * laneOps + r * ctxOps)
      lc.append(["runs": B, "refreshes": KMAX, "rows": r, "seconds": t, "int8_tops": ops / t / 1e12])
    }
    report["learn"] = lc
    net.enableProfiling()
    func kernels() -> [[String: Any]] { net.profile().sorted { $0.2 > $1.2 }.map { ["kernel": $0.0, "calls": $0.1, "ns": $0.2] } }
    for i in 0..<(ROWS / 2) { R0.idxP[i] = Int32((i % B) * L + L - 1) }
    R0.forward(B, ROWS / 2); report["forward_kernels"] = kernels()
    gradients(B, KMAX, ROWS / 4, noCredit(ROWS / 4), segs(ROWS / 4)); report["learn_kernels"] = kernels()
    _ = net.apply(rate: 0); report["optimizer_kernels"] = kernels()
    net.profiling = false
    do { let t0 = now(); for _ in 0..<20 { _ = net.apply(rate: 0) }; report["optimizer_step"] = ["parameters": P, "seconds": (now() - t0) / 20] }
    // the environment on one thread: the cases' first rollouts, simulated and credited
    var events: UInt64 = 0, draws: UInt64 = 0, entries: UInt64 = 0, simT = 0.0, credT = 0.0
    let w0 = [Float](repeating: 0.01, count: Int(ROUTE_CONTEXTS) * FS)
    for i in 0..<n {
      var s = route_stream(); route_case_stream(cs + i, &s); var w = cs[i].w
      let t0 = now()
      route_sim_begin(R0.sims[0], &s, &w, 0, cs[i].key0.0, cs[i].key.0)
      while route_sim_decide(R0.sims[0], R0.decisions, R0.tokens[0]) == 1 { route_sim_answer(R0.sims[0], w0) }
      let t1 = now()
      var part = [Float](repeating: 0, count: TASKS * Int(route_sim_contexts(R0.sims[0])) * FS), ll = 0.0
      for p in 0..<Int(route_sim_points(R0.sims[0])) { route_sim_credit(R0.sims[0], Int32(p), &part, &ll) }
      credT += now() - t1; simT += t1 - t0
      var c = [UInt64](repeating: 0, count: 4); route_sim_counts(R0.sims[0], &c); events += c[0]; draws += c[1]; entries += c[2]
    }
    report["environment"] = ["windows": n, "seconds_simulated": simT, "seconds_credited": credT, "events": events, "draws": draws, "entries": entries, "threads": ProcessInfo.processInfo.activeProcessorCount]
    return report
  }
}

// ---- the C interface (trainer.h) ----
func trainer(_ t: UnsafeMutableRawPointer) -> Trainer { Unmanaged<Trainer>.fromOpaque(t).takeUnretainedValue() }
func json(_ r: [String: Any], _ out: UnsafeMutablePointer<CChar>, _ room: Int) -> Int32 {
  guard let d = try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys]), d.count < room else { return -1 }
  d.withUnsafeBytes { memcpy(out, $0.baseAddress!, d.count) }; out[d.count] = 0
  return 0
}
@_cdecl("route_trainer_open") public func route_trainer_open(_ c: UnsafePointer<route_config>) -> UnsafeMutableRawPointer? {
  if let p = c.pointee.telemetry, route_tel_open(p, c.pointee.rank, c.pointee.world) != 0 { fail("telemetry: cannot map \(String(cString: p))"); return nil }
  guard let t = Trainer(c.pointee) else { return nil }
  return Unmanaged.passRetained(t).toOpaque()
}
@_cdecl("route_trainer_failure") public func route_trainer_failure() -> UnsafePointer<CChar>? { UnsafePointer(failureText) }
@_cdecl("route_trainer_groups") public func route_trainer_groups(_ t: UnsafeMutableRawPointer) -> Int32 { Int32(trainer(t).TX / 4) }
@_cdecl("route_trainer_start") public func route_trainer_start(_ t: UnsafeMutableRawPointer) { trainer(t).start() }
@_cdecl("route_trainer_room") public func route_trainer_room(_ t: UnsafeMutableRawPointer, _ timeout: Double) -> Int32 { Int32(trainer(t).room(timeout)) }
@_cdecl("route_trainer_push") public func route_trainer_push(_ t: UnsafeMutableRawPointer, _ c: UnsafePointer<route_case>, _ n: Int32) { trainer(t).push(c, Int(n)) }
@_cdecl("route_trainer_ended") public func route_trainer_ended(_ t: UnsafeMutableRawPointer, _ s: UnsafeMutablePointer<Float>, _ room: Int32) -> Int32 { Int32(trainer(t).takeEnded(s, Int(room))) }
@_cdecl("route_trainer_learned") public func route_trainer_learned(_ t: UnsafeMutableRawPointer) -> Int32 { let x = trainer(t); x.shareLock.lock(); defer { x.shareLock.unlock() }; return Int32(x.learned) }
@_cdecl("route_trainer_finish") public func route_trainer_finish(_ t: UnsafeMutableRawPointer) { trainer(t).finish() }
@_cdecl("route_trainer_evaluate") public func route_trainer_evaluate(_ t: UnsafeMutableRawPointer, _ c: UnsafePointer<route_case>, _ n: Int32, _ out: UnsafeMutablePointer<Double>) -> Int32 { trainer(t).evaluate(c, Int(n), out); return 0 }
@_cdecl("route_trainer_save") public func route_trainer_save(_ t: UnsafeMutableRawPointer, _ a: UnsafePointer<CChar>?, _ b: UnsafePointer<CChar>?) -> Int32 { trainer(t).save(a, b) ? 0 : -1 }
@_cdecl("route_trainer_forward") public func route_trainer_forward(_ t: UnsafeMutableRawPointer, _ lanes: UnsafePointer<Float>, _ nl: Int32, _ tokens: UnsafePointer<Float>, _ nc: Int32, _ state: UnsafeMutablePointer<Float>, _ w: UnsafeMutablePointer<Float>) {
  trainer(t).forwardOne(lanes, Int(nl), tokens, Int(nc), state, w)
}
@_cdecl("route_trainer_xbench") public func route_trainer_xbench(_ t: UnsafeMutableRawPointer, _ rounds: Int32, _ s: UnsafeMutablePointer<Double>, _ wrong: UnsafeMutablePointer<Int>) -> Int32 { trainer(t).xbench(Int(rounds), s, wrong) ? 0 : -1 }
@_cdecl("route_trainer_check") public func route_trainer_check(_ t: UnsafeMutableRawPointer, _ c: UnsafePointer<route_case>, _ n: Int32, _ eps: Double, _ graph: Int32, _ out: UnsafeMutablePointer<CChar>, _ room: Int) -> Int32 {
  json(trainer(t).check(c, Int(n), eps: Float(eps), graph: graph != 0), out, room)
}
@_cdecl("route_trainer_bench") public func route_trainer_bench(_ t: UnsafeMutableRawPointer, _ c: UnsafePointer<route_case>, _ n: Int32, _ out: UnsafeMutablePointer<CChar>, _ room: Int) -> Int32 {
  json(trainer(t).bench(c, Int(n)), out, room)
}
@_cdecl("route_trainer_close") public func route_trainer_close(_ t: UnsafeMutableRawPointer) { route_exchange_close(); Unmanaged<Trainer>.fromOpaque(t).release() }
