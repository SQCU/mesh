// The runs in flight (route.h): a Runner's slots, each one transaction's simulator, its decision and tokens, its carried
// state, and what the learning batch keeps of it; its forward buffers and network. A tick advances every listed run to
// its next refresh (or its end: a finished run's credit is computed in the same parallel pass), runs one forward for the
// runs at a refresh, and answers them. The trainer's pool ticks its window groups continuously; a batch rollout (the
// measurement, the checks) is ticks until its runs end.
import Foundation
import Metal

struct Kept { var lanes: [Float]; var lanesUsed: Int; var tokens: [Float]; var contexts: Int }
// one slot: a class each, so the parallel pass writes each slot's own object
final class Slot {
  var stream = route_stream(), state = [Float](repeating: 0, count: 2 * D), start: [Float] = [], kept: [Kept] = [], credit: [Float] = []
  var costs = route_costs(), counts = [UInt64](repeating: 0, count: 4)
  var answer: [Float] = [], answered = false   // the streaming pool's: its refresh's w, copied out of the batch's
}

final class Runner {
  let n: Int, rows: Int, net: Net
  let sims: [OpaquePointer], decisions: UnsafeMutablePointer<route_decision>, tokens: [UnsafeMutablePointer<Float>], slots: [Slot]
  let laneBuf: MTLBuffer, maskBuf: MTLBuffer, stateBuf: MTLBuffer, nextBuf: MTLBuffer, tokBuf: MTLBuffer, idxBuf: MTLBuffer, wBuf: MTLBuffer
  let laneP: UnsafeMutablePointer<Float>, maskP: UnsafeMutablePointer<Float>, stateP: UnsafeMutablePointer<Float>, nextP: UnsafeMutablePointer<Float>
  let tokP: UnsafeMutablePointer<Float>, idxP: UnsafeMutablePointer<Int32>, wP: UnsafeMutablePointer<Float>
  let laneOffset = MemoryLayout<route_decision>.offset(of: \route_decision.lane)!
  // the time each part took (wall), and the threads' busy time inside the parallel passes (their efficiency: busy over
  // threads x wall)
  var simTime = 0.0, creditBusy = 0.0, busy = 0.0, parallelWall = 0.0, forwardTime = 0.0, forwards = 0, decided = 0, rowsRun = 0, ticks = 0
  let threads = ProcessInfo.processInfo.activeProcessorCount

  init(_ device: MTLDevice, net: Net, slots n: Int, rows: Int) {
    self.n = n; self.rows = rows; self.net = net
    sims = (0..<n).map { _ in route_sim_new()! }
    decisions = .allocate(capacity: n)
    tokens = (0..<n).map { _ in .allocate(capacity: Int(ROUTE_CONTEXTS) * IN) }
    slots = (0..<n).map { _ in Slot() }
    func shared(_ floats: Int) -> MTLBuffer { device.makeBuffer(length: max(16, floats * 4), options: .storageModeShared)! }
    laneBuf = shared(n * L * IN); maskBuf = shared(n * L); stateBuf = shared(n * 2 * D); nextBuf = shared(n * 2 * D)
    tokBuf = shared(rows * IN); idxBuf = shared(rows); wBuf = shared(rows * FS)
    laneP = laneBuf.contents().bindMemory(to: Float.self, capacity: n * L * IN); maskP = maskBuf.contents().bindMemory(to: Float.self, capacity: n * L)
    stateP = stateBuf.contents().bindMemory(to: Float.self, capacity: n * 2 * D); nextP = nextBuf.contents().bindMemory(to: Float.self, capacity: n * 2 * D)
    tokP = tokBuf.contents().bindMemory(to: Float.self, capacity: rows * IN); idxP = idxBuf.contents().bindMemory(to: Int32.self, capacity: rows)
    wP = wBuf.contents().bindMemory(to: Float.self, capacity: rows * FS)
  }
  func laneTokens(_ t: Int) -> UnsafePointer<Float> { UnsafeRawPointer(decisions + t).advanced(by: laneOffset).assumingMemoryBound(to: Float.self) }
  func costs(_ t: Int) -> route_costs { var c = route_costs(); route_sim_costs(sims[t], &c); return c }

  // slot t begins a run: its stream (slots[t].stream), window, router mode, keys; its state slots[t].state
  func begin(_ t: Int, window: route_window, mode: Int32, key0: UInt32, key: UInt32) {
    var s = slots[t].stream, w = window
    route_sim_begin(sims[t], &s, &w, mode, key0, key)
    slots[t].kept = []; slots[t].credit = []; slots[t].start = slots[t].state
  }
  // a forward for `bd` runs (lanes, masks, states in the buffers) and `rows` context rows: w and the next states
  func forward(_ bd: Int, _ rowsUsed: Int) {
    let t0 = Date()
    net.forward(lanes: laneBuf, mask: maskBuf, h0: stateBuf, tokens: tokBuf, pos: idxBuf, B: bd, R: rowsUsed, w: wBuf, next: nextBuf)
    let dt = Date().timeIntervalSince(t0)
    forwardTime += dt; forwards += 1; decided += bd; rowsRun += rowsUsed
    route_tel_add(Int32(RT_FORWARDS.rawValue), 1); route_tel_add(Int32(RT_EVALUATIONS.rawValue), UInt64(bd)); route_tel_add(Int32(RT_FORWARD_ROWS.rawValue), UInt64(rowsUsed))
    route_tel_add(Int32(RT_FORWARD_NS.rawValue), UInt64(dt * 1e9)); route_tel_phase(Int32(RT_PHASE_FIRST_FORWARD.rawValue))
  }
  // one tick over the running slots `active`: returns those that ended (their costs, counts and, if keep, credit made)
  func tick(_ active: [Int], keep: Bool) -> [Int] {
    if active.isEmpty { return [] }
    ticks += 1
    let t0 = Date(), m = active.count
    let got = UnsafeMutablePointer<Int32>.allocate(capacity: m), spent = UnsafeMutablePointer<Double>.allocate(capacity: m), cred = UnsafeMutablePointer<Double>.allocate(capacity: m)
    defer { got.deallocate(); spent.deallocate(); cred.deallocate() }
    DispatchQueue.concurrentPerform(iterations: m) { k in
      let a = Date(), t = active[k]
      got[k] = route_sim_decide(sims[t], decisions + t, tokens[t])
      cred[k] = 0
      if got[k] == 0 {   // ended: its costs and counts; its credit, each refresh's (what the learning batch reads)
        let c0 = Date(), s = slots[t]
        route_sim_costs(sims[t], &s.costs); route_sim_counts(sims[t], &s.counts)
        if keep {
          var all: [Float] = []
          for (p, d) in s.kept.enumerated() { var part = [Float](repeating: 0, count: TASKS * d.contexts * FS), ll = 0.0; route_sim_credit(sims[t], Int32(p), &part, &ll); all += part }
          s.credit = all
        }
        cred[k] = Date().timeIntervalSince(c0)
      }
      spent[k] = Date().timeIntervalSince(a)
    }
    let w0 = Date().timeIntervalSince(t0)
    simTime += w0; parallelWall += w0
    for k in 0..<m { busy += spent[k]; creditBusy += cred[k] }
    let pending = (0..<m).filter { got[$0] == 1 }.map { active[$0] }, ended = (0..<m).filter { got[$0] == 0 }.map { active[$0] }
    if pending.isEmpty { return ended }
    var rowOf: [Int] = [], used = 0
    for (b, t) in pending.enumerated() {
      let nl = Int(decisions[t].lanes), nc = Int(decisions[t].contexts)
      precondition(used + nc <= rows, "a forward past the runner's rows")
      memcpy(laneP + b * L * IN, laneTokens(t), L * IN * 4)
      for q in 0..<L { maskP[b * L + q] = q < nl ? 1 : 0 }
      for i in 0..<(2 * D) { stateP[b * 2 * D + i] = slots[t].state[i] }
      memcpy(tokP + used * IN, tokens[t], nc * IN * 4)
      for c in 0..<nc { idxP[used + c] = Int32(b * L + L - 1) }   // its state: the refresh's last lane row
      rowOf.append(used); used += nc
      if keep { slots[t].kept.append(Kept(lanes: Array(UnsafeBufferPointer(start: laneTokens(t), count: L * IN)), lanesUsed: nl, tokens: Array(UnsafeBufferPointer(start: tokens[t], count: nc * IN)), contexts: nc)) }
    }
    forward(pending.count, used)
    for (b, t) in pending.enumerated() { for i in 0..<(2 * D) { slots[t].state[i] = nextP[b * 2 * D + i] } }
    let t1 = Date(), offsets = rowOf, wp = wP
    let spent2 = UnsafeMutablePointer<Double>.allocate(capacity: pending.count); defer { spent2.deallocate() }
    DispatchQueue.concurrentPerform(iterations: pending.count) { k in let a = Date(); route_sim_answer(sims[pending[k]], wp + offsets[k] * FS); spent2[k] = Date().timeIntervalSince(a) }
    let w1 = Date().timeIntervalSince(t1)
    simTime += w1; parallelWall += w1
    for k in 0..<pending.count { busy += spent2[k] }
    return ended
  }
  // runs `ids` (their streams and states set) to the end, together
  func rollout(_ ids: [Int], window: (Int) -> route_window, mode: Int32, key0: (Int) -> UInt32, key: (Int) -> UInt32, keep: Bool) {
    for t in ids { begin(t, window: window(t), mode: mode, key0: key0(t), key: key(t)) }
    var active = ids
    while !active.isEmpty { let ended = Set(tick(active, keep: keep)); active = active.filter { !ended.contains($0) } }
  }
  func resetCounters() { simTime = 0; creditBusy = 0; busy = 0; parallelWall = 0; forwardTime = 0; forwards = 0; decided = 0; rowsRun = 0; ticks = 0 }

  // ---- the streaming pool: every run on its own. Workers (all hardware threads but two: the main thread's, the
  // exchange's) take runs from the queue: answer the run's last refresh (if it has one), simulate to its next refresh
  // or its end (its costs, counts and credit then made); the run goes to `awaiting` (a refresh) or `finished`. The main
  // thread forwards whatever awaits as one batch and queues those runs again; no run waits on another run. ----
  let lock = NSCondition()
  var queue: [Int] = [], awaiting: [Int] = [], finished: [Int] = [], workers: [Thread] = [], workerBusy = 0.0, streamSince = Date()
  // a worker on every hardware thread, below the learner's and the exchange's scheduling class: the OS gives those the
  // CPU when they need it, the environment every cycle left
  func startWorkers() {
    route_tel_set(Int32(RT_WORKERS.rawValue), UInt64(threads))
    for _ in 0..<threads {
      let th = Thread { [unowned self] in self.work() }
      th.stackSize = 8 << 20; th.qualityOfService = .utility; workers.append(th); th.start()
    }
  }
  var runnable: Int { lock.lock(); defer { lock.unlock() }; return queue.count }
  func submit(_ ts: [Int]) { lock.lock(); queue += ts; lock.broadcast(); lock.unlock() }
  private func work() {
    while true {
      lock.lock()
      while queue.isEmpty { lock.wait() }
      let t = queue.removeLast()
      lock.unlock()
      let a = Date(), s = slots[t]
      if s.answered { s.answer.withUnsafeBufferPointer { route_sim_answer(sims[t], $0.baseAddress!) }; s.answered = false }
      let got = route_sim_decide(sims[t], decisions + t, tokens[t])
      var c = 0.0
      if got == 0 {
        let c0 = Date()
        route_sim_costs(sims[t], &s.costs); route_sim_counts(sims[t], &s.counts)
        var all: [Float] = []
        for (p, d) in s.kept.enumerated() { var part = [Float](repeating: 0, count: TASKS * d.contexts * FS), ll = 0.0; route_sim_credit(sims[t], Int32(p), &part, &ll); all += part }
        s.credit = all
        c = Date().timeIntervalSince(c0)
      }
      let spent = Date().timeIntervalSince(a)
      lock.lock()
      if got == 0 { finished.append(t) } else { awaiting.append(t) }
      workerBusy += spent; creditBusy += c
      lock.broadcast(); lock.unlock()
      route_tel_add(Int32(RT_WORKER_NS.rawValue), UInt64(spent * 1e9)); route_tel_add(Int32(RT_CREDIT_NS.rawValue), UInt64(c * 1e9))
    }
  }
  // the main thread's turn: wait until a run awaits or has finished (at most `timeout`); forward every awaiting run as
  // one batch (kept for the learning batch) and queue them again. Returns the runs that finished
  func service(timeout: Double) -> [Int] {
    lock.lock()
    if awaiting.isEmpty && finished.isEmpty { _ = lock.wait(until: Date(timeIntervalSinceNow: timeout)) }
    let batch = awaiting, done = finished
    awaiting = []; finished = []
    lock.unlock()
    if batch.isEmpty { return done }
    ticks += 1
    var rowOf: [Int] = [], used = 0, take = 0
    for t in batch {   // a forward's rows at most `rows`: the rest wait for the next
      let nc = Int(decisions[t].contexts)
      if used + nc > rows { break }
      take += 1
      let b = take - 1, nl = Int(decisions[t].lanes)
      memcpy(laneP + b * L * IN, laneTokens(t), L * IN * 4)
      for q in 0..<L { maskP[b * L + q] = q < nl ? 1 : 0 }
      for i in 0..<(2 * D) { stateP[b * 2 * D + i] = slots[t].state[i] }
      memcpy(tokP + used * IN, tokens[t], nc * IN * 4)
      for c in 0..<nc { idxP[used + c] = Int32(b * L + L - 1) }
      rowOf.append(used); used += nc
      slots[t].kept.append(Kept(lanes: Array(UnsafeBufferPointer(start: laneTokens(t), count: L * IN)), lanesUsed: nl, tokens: Array(UnsafeBufferPointer(start: tokens[t], count: nc * IN)), contexts: nc))
    }
    if take < batch.count { lock.lock(); awaiting += batch[take...]; lock.unlock() }
    let sent = Array(batch[0..<take])
    forward(take, used)
    for (b, t) in sent.enumerated() {
      for i in 0..<(2 * D) { slots[t].state[i] = nextP[b * 2 * D + i] }
      slots[t].answer = Array(UnsafeBufferPointer(start: wP + rowOf[b] * FS, count: Int(decisions[t].contexts) * FS)); slots[t].answered = true
    }
    submit(sent)
    return done
  }
}
