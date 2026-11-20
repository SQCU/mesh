// The network's C interface (kernels.h): batches are shared buffers the driver writes in place; forward, step and
// publish are committed in the GPU's queue order; route_net_wait is the one CPU wait, where the driver reads a batch's
// results. No threads, no locks: the schedule is the driver's.
import Foundation
import Metal
import MetalPerformanceShadersGraph

func tel(_ k: route_tel) -> Int32 { Int32(k.rawValue) }
var netFailure: UnsafeMutablePointer<CChar>? = nil
func fail(_ s: String) { free(netFailure); netFailure = strdup(s) }

final class Batch {
  let s = UnsafeMutablePointer<route_batch>.allocate(capacity: 1)
  let lanes: MTLBuffer, mask: MTLBuffer, h0: MTLBuffer, next: MTLBuffer, tokens: MTLBuffer, pos: MTLBuffer, w: MTLBuffer, wcur: MTLBuffer
  let scale0: MTLBuffer, first: MTLBuffer, seg: MTLBuffer
  var draws: MTLBuffer, ll: MTLBuffer, last: MTLCommandBuffer? = nil
  init(_ device: MTLDevice, runs: Int, refreshes: Int, rows: Int) {
    func sh(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: max(16, bytes), options: .storageModeShared)! }
    lanes = sh(runs * refreshes * L * IN * 4); mask = sh(runs * refreshes * L * 4); h0 = sh(runs * 2 * D * 4); next = sh(runs * 2 * D * 4)
    tokens = sh(rows * IN * 4); pos = sh(rows * 4); w = sh(rows * FS * 4); wcur = sh(rows * FS * 4); scale0 = sh(rows * 4); first = sh((rows + 1) * 4)
    seg = sh(runs * refreshes * 2 * 4); draws = sh(MemoryLayout<route_draw>.stride); ll = sh(4)
    func f(_ b: MTLBuffer) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: b.length / 4) }
    s.pointee = route_batch(lanes: f(lanes), mask: f(mask), h0: f(h0), next: f(next), tokens: f(tokens), w: f(w), wcur: f(wcur), scale0: f(scale0), ll: f(ll),
                            pos: pos.contents().bindMemory(to: Int32.self, capacity: rows), first: first.contents().bindMemory(to: UInt32.self, capacity: rows + 1),
                            seg: seg.contents().bindMemory(to: UInt32.self, capacity: runs * refreshes * 2), draws: draws.contents().bindMemory(to: route_draw.self, capacity: 1),
                            runs: Int32(runs), refreshes: Int32(refreshes), rows: Int32(rows), draws_room: 1)
  }
  func reserve(_ device: MTLDevice, _ n: Int) {   // the draws' room: grown (geometrically) when a batch needs more
    if n <= Int(s.pointee.draws_room) { return }
    let room = 2 * n
    draws = device.makeBuffer(length: room * MemoryLayout<route_draw>.stride, options: .storageModeShared)!; ll = device.makeBuffer(length: room * 4, options: .storageModeShared)!
    s.pointee.draws = draws.contents().bindMemory(to: route_draw.self, capacity: room); s.pointee.ll = ll.contents().bindMemory(to: Float.self, capacity: room); s.pointee.draws_room = Int32(room)
  }
}

// a network and what another publishes into it: two staging copies of theta, alternating, and two events (a staging
// copy ready; a staging copy consumed), so publishing never waits on the CPU
final class Handle {
  let device: MTLDevice, net: Net, ready: MTLSharedEvent, consumed: MTLSharedEvent, staging: [MTLBuffer]
  var published: UInt64 = 0, batches: [Batch] = []
  init(_ device: MTLDevice, _ net: Net) {
    self.device = device; self.net = net; ready = device.makeSharedEvent()!; consumed = device.makeSharedEvent()!
    staging = (0..<2).map { _ in device.makeBuffer(length: net.lay.count * 4, options: .storageModePrivate)! }
  }
}
func handle(_ p: UnsafeMutableRawPointer) -> Handle { Unmanaged<Handle>.fromOpaque(p).takeUnretainedValue() }
func batch(_ h: Handle, _ b: UnsafeMutablePointer<route_batch>) -> Batch { h.batches.first { $0.s == b }! }

@_cdecl("route_net_new") public func route_net_new(_ metallib: UnsafePointer<CChar>, _ runs: Int32, _ refreshes: Int32, _ rows: Int32, _ seed: UInt64, _ thetaIn: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
  let device = MTLCreateSystemDefaultDevice()!, nlay = NetLayout()
  var g = route_rng(s: seed)
  var initial = nlay.initial(&g)
  if let p = thetaIn {   // NetLayout's parameters, or the earlier fp32 layout's (reparametrized exactly)
    guard let d = FileManager.default.contents(atPath: String(cString: p)) else { fail("theta_in: cannot read \(String(cString: p))"); return nil }
    let lay = Layout(full: false), t = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    if d.count == lay.count * 4 { initial = nlay.from(old: t, lay) } else if d.count == nlay.count * 4 { initial = t } else { fail("theta_in: parameters of another layout (\(d.count) bytes)"); return nil }
  }
  let net = Net(device, device.makeCommandQueue()!, library: String(cString: metallib), runs: Int(runs), refreshes: Int(refreshes), contexts: Int(rows), initial: initial)
  return Unmanaged.passRetained(Handle(device, net)).toOpaque()
}
@_cdecl("route_net_failure") public func route_net_failure() -> UnsafePointer<CChar>? { UnsafePointer(netFailure) }
@_cdecl("route_net_batch") public func route_net_batch(_ p: UnsafeMutableRawPointer, _ runs: Int32, _ refreshes: Int32, _ rows: Int32) -> UnsafeMutablePointer<route_batch> {
  let h = handle(p), b = Batch(h.device, runs: Int(runs), refreshes: Int(refreshes), rows: Int(rows))
  h.batches.append(b); return b.s
}
@_cdecl("route_net_reserve") public func route_net_reserve(_ p: UnsafeMutableRawPointer, _ b: UnsafeMutablePointer<route_batch>, _ n: Int32) { let h = handle(p); batch(h, b).reserve(h.device, Int(n)) }
@_cdecl("route_net_forward") public func route_net_forward(_ p: UnsafeMutableRawPointer, _ bp: UnsafeMutablePointer<route_batch>, _ B: Int32, _ R: Int32) {
  let h = handle(p), b = batch(h, bp)
  let cb = h.net.forwardEncode(lanes: b.lanes, mask: b.mask, h0: b.h0, tokens: b.tokens, pos: b.pos, B: Int(B), R: Int(R), w: b.w, next: b.next)
  cb.addCompletedHandler { c in route_tel_add(tel(RT_FORWARD_NS), UInt64((c.gpuEndTime - c.gpuStartTime) * 1e9)); route_wake() }
  cb.commit(); b.last = cb
}
@_cdecl("route_net_done") public func route_net_done(_ p: UnsafeMutableRawPointer, _ bp: UnsafeMutablePointer<route_batch>) -> Int32 {
  guard let cb = batch(handle(p), bp).last else { return 1 }
  return cb.status == .completed || cb.status == .error ? 1 : 0
}
@_cdecl("route_net_wait") public func route_net_wait(_ p: UnsafeMutableRawPointer, _ bp: UnsafeMutablePointer<route_batch>) {
  let h = handle(p), b = batch(h, bp)
  b.last?.waitUntilCompleted()
  if let err = b.last?.error { fail("\(err)") }
  if let cb = b.last { h.net.lastGPU = (cb.gpuEndTime - cb.gpuStartTime) * 1e9 }   // the profile's scale: this buffer's GPU time
}
// the ranks' sum of grads through the lease (its Metal buffer), into the open command buffer
func encodeExchange(_ n: Net) {
  var off = 0
  let lease = Unmanaged<MTLBuffer>.fromOpaque(route_exchange_metal(&off)).takeUnretainedValue(), bytes = 3 * n.lay.count * 4
  n.blit { $0.copy(from: n.grads, sourceOffset: 0, to: lease, destinationOffset: off, size: bytes) }
  n.e.endEncoding()
  if route_exchange_encode(Unmanaged.passUnretained(n.cb).toOpaque()) != 0 { fail("exchange: " + String(cString: route_exchange_failure())) }
  n.open()
  n.blit { $0.copy(from: lease, sourceOffset: off, to: n.grads, destinationOffset: 0, size: bytes) }
}
@_cdecl("route_net_exchange") public func route_net_exchange(_ p: UnsafeMutableRawPointer) {
  let n = handle(p).net
  n.begin(); encodeExchange(n); n.e.endEncoding(); n.cb.commit()
}
@_cdecl("route_net_dw") public func route_net_dw(_ p: UnsafeMutableRawPointer) -> UnsafeMutablePointer<Float> {
  let n = handle(p).net; idle(n); return n.dwBuf!.contents().bindMemory(to: Float.self, capacity: n.dwBuf!.length / 4)
}
@_cdecl("route_net_step") public func route_net_step(_ p: UnsafeMutableRawPointer, _ bp: UnsafeMutablePointer<route_batch>, _ B: Int32, _ K: Int32, _ R: Int32, _ ndraws: Int32, _ scale: Float, _ rate: Double, _ world: Int32) {
  let h = handle(p), b = batch(h, bp), n = h.net, P = n.lay.count
  let credit = Net.Credit(draws: b.draws, count: Int(ndraws), first: b.first, wrows: b.w, scale0: b.scale0, scale: scale, ll: b.ll)
  n.begin()
  n.encodeLearn(lanes: b.lanes, mask: b.mask, h0: b.h0, tokens: b.tokens, pos: b.pos, seg: b.seg, B: Int(B), K: Int(K), R: Int(R), credit: credit, w: b.wcur)
  if world > 1 { encodeExchange(n) }
  n.encodeApply(rate: rate, inv: 1 / Float(world))
  n.e.endEncoding()
  let ll = b.ll, count = Int(ndraws)
  n.cb.addCompletedHandler { cb in   // where its results are: the telemetry's counts (no one waits for them)
    if let err = cb.error { fail("step: \(err)"); route_wake(); return }
    let r = ll.contents().bindMemory(to: Float.self, capacity: max(count, 1))
    var sum = 0.0, most: Float = 0
    for i in 0..<count { sum += Double(exp(r[i])); most = max(most, abs(r[i])) }
    route_tel_add(tel(RT_OPTIMIZER_STEPS), 1); route_tel_add(tel(RT_LEARN_NS), UInt64((cb.gpuEndTime - cb.gpuStartTime) * 1e9))
    route_tel_add(tel(RT_RATIO_DRAWS), UInt64(count)); route_tel_add(tel(RT_RATIO_SUM_E6), UInt64(sum * 1e6)); route_tel_max(tel(RT_LOGRATIO_MAX_E3), UInt64(most * 1e3))
    route_wake()
  }
  n.cb.commit(); b.last = n.cb
}
@_cdecl("route_net_publish") public func route_net_publish(_ fp: UnsafeMutableRawPointer, _ tp: UnsafeMutableRawPointer) {
  let from = handle(fp), to = handle(tp), n = to.published + 1, st = to.staging[Int(n % 2)], bytes = to.net.lay.count * 4
  let a = from.net.queue.makeCommandBuffer()!   // after `from`'s steps: theta staged, once `to` has read that staging copy's last
  if n > 2 { a.encodeWaitForEvent(to.consumed, value: n - 2) }
  let ba = a.makeBlitCommandEncoder()!; ba.copy(from: from.net.theta, sourceOffset: 0, to: st, destinationOffset: 0, size: bytes); ba.endEncoding()
  a.encodeSignalEvent(to.ready, value: n); a.commit()
  let t = to.net, b = t.queue.makeCommandBuffer()!   // after `to`'s forwards: theta from the staging copy, the codes rebuilt
  b.encodeWaitForEvent(to.ready, value: n)
  let bb = b.makeBlitCommandEncoder()!; bb.copy(from: st, sourceOffset: 0, to: t.theta, destinationOffset: 0, size: bytes); bb.endEncoding()
  t.cb = b; t.labels = []; t.open(); t.encodeWeights(); t.e.endEncoding()
  b.encodeSignalEvent(to.consumed, value: n); b.commit()
  to.published = n
}
@_cdecl("route_net_steps") public func route_net_steps(_ p: UnsafeMutableRawPointer) -> Int32 { Int32(handle(p).net.step) }
@_cdecl("route_net_join") public func route_net_join(_ p: UnsafeMutableRawPointer, _ rank: Int32, _ world: Int32, _ id: UnsafePointer<CChar>) -> Int32 {
  let n = handle(p).net, declaring = n.queue.makeCommandBuffer()!
  return route_exchange_open(rank, world, id, size_t(3 * n.lay.count), Unmanaged.passUnretained(declaring).toOpaque())
}
func idle(_ n: Net) { let cb = n.queue.makeCommandBuffer()!; cb.commit(); cb.waitUntilCompleted() }
@_cdecl("route_net_save") public func route_net_save(_ p: UnsafeMutableRawPointer, _ fp32: UnsafePointer<CChar>?, _ int8: UnsafePointer<CChar>?) -> Int32 {
  let n = handle(p).net
  idle(n)
  if let f = fp32, !FileManager.default.createFile(atPath: String(cString: f), contents: Data(bytes: n.theta.contents(), count: n.lay.count * 4)) { fail("cannot write \(String(cString: f))"); return -1 }
  if let f = int8 { n.writeCodes(String(cString: f)) }
  return 0
}
@_cdecl("route_net_theta") public func route_net_theta(_ p: UnsafeMutableRawPointer, _ count: UnsafeMutablePointer<Int32>) -> UnsafeMutablePointer<Float> {
  let n = handle(p).net; idle(n); count.pointee = Int32(n.lay.count)
  return n.theta.contents().bindMemory(to: Float.self, capacity: n.lay.count)
}
@_cdecl("route_net_grads") public func route_net_grads(_ p: UnsafeMutableRawPointer) -> UnsafeMutablePointer<Float> {
  let n = handle(p).net; idle(n); return n.grads.contents().bindMemory(to: Float.self, capacity: 3 * n.lay.count)
}
func json(_ r: Any, _ out: UnsafeMutablePointer<CChar>, _ room: Int) -> Int32 {
  guard let d = try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys]), d.count < room else { return -1 }
  d.withUnsafeBytes { memcpy(out, $0.baseAddress!, d.count) }; out[d.count] = 0
  return 0
}
@_cdecl("route_net_profile") public func route_net_profile(_ p: UnsafeMutableRawPointer, _ on: Int32, _ out: UnsafeMutablePointer<CChar>, _ room: Int) -> Int32 {
  let n = handle(p).net
  if on != 0 { n.enableProfiling(); return 0 }
  let r = n.profile().map { ["kernel": $0.0, "calls": $0.1, "ns": $0.2] as [String: Any] }
  n.profiling = false
  return json(r, out, room)
}
@_cdecl("route_net_probes") public func route_net_probes(_ p: UnsafeMutableRawPointer, _ on: Int32, _ out: UnsafeMutablePointer<CChar>, _ room: Int) -> Int32 {
  let n = handle(p).net
  if on != 0 { n.checking = true; return 0 }
  n.checking = false
  return json(n.probeReport(), out, room)
}
// the fp32 reference (policy.swift and learn.swift's graph: the same function, plain fp32 weights W = g V / |V|) on a
// stepped batch of one window: its w, and each task's gradient mapped to NetLayout's (dV, dg, db) by the same chain rule
@_cdecl("route_net_reference") public func route_net_reference(_ p: UnsafeMutableRawPointer, _ bp: UnsafeMutablePointer<route_batch>, _ B: Int32, _ K: Int32, _ R: Int32, _ wOut: UnsafeMutablePointer<Float>, _ gOut: UnsafeMutablePointer<Float>) {
  let h = handle(p), b = batch(h, bp), n = h.net, lay = Layout(full: false), P = n.lay.count, dev = h.device
  idle(n)
  func td(_ x: MTLBuffer, _ shape: [Int], _ type: MPSDataType = .float32) -> MPSGraphTensorData { MPSGraphTensorData(x, shape: dims(shape), dataType: type) }
  func sh(_ c: Int) -> MTLBuffer { dev.makeBuffer(length: max(16, c * 4), options: .storageModeShared)! }
  let theta = Array(UnsafeBufferPointer(start: n.theta.contents().bindMemory(to: Float.self, capacity: P), count: P))
  let old = n.lay.toOld(theta, lay), oldBuf = dev.makeBuffer(bytes: old, length: old.count * 4, options: .storageModeShared)!
  let refIdx = sh(Int(R)), ri = refIdx.contents().bindMemory(to: Int32.self, capacity: max(Int(R), 1)), pos = b.pos.contents().bindMemory(to: Int32.self, capacity: max(Int(R), 1))
  for r in 0..<Int(R) { let q = Int(pos[r]) / L; ri[r] = Int32((q % Int(K)) * Int(B) + q / Int(K)) }   // the batch's (b K + k) as the reference's k B + b
  // the batch's lanes and masks [b][k] as the reference's [k][b]
  let lanesT = sh(Int(B * K) * L * IN), maskT = sh(Int(B * K) * L)
  let lp = b.lanes.contents().bindMemory(to: Float.self, capacity: Int(B * K) * L * IN), mp = b.mask.contents().bindMemory(to: Float.self, capacity: Int(B * K) * L)
  let ltp = lanesT.contents().bindMemory(to: Float.self, capacity: Int(B * K) * L * IN), mtp = maskT.contents().bindMemory(to: Float.self, capacity: Int(B * K) * L)
  for bb in 0..<Int(B) { for k in 0..<Int(K) {
    memcpy(ltp + (k * Int(B) + bb) * L * IN, lp + (bb * Int(K) + k) * L * IN, L * IN * 4); memcpy(mtp + (k * Int(B) + bb) * L, mp + (bb * Int(K) + k) * L, L * 4)
  } }
  let ref = Learn(lay, windows: Int(B), refreshes: Int(K), rows: Int(R)), wRef = sh(Int(R) * FS), gRef = (0..<TASKS).map { _ in sh(lay.count) }
  var res: [MPSGraphTensor: MPSGraphTensorData] = [ref.w: td(wRef, [Int(R), FS])]
  for t in 0..<TASKS { res[ref.grads[t]] = td(gRef[t], [lay.count]) }
  ref.g.run(with: n.queue, feeds: [ref.theta: td(oldBuf, [lay.count]), ref.lanes: td(lanesT, [Int(K), Int(B), L, IN]), ref.mask: td(maskT, [Int(K), Int(B), L]), ref.state: td(b.h0, [Int(B), 2, D]),
                                   ref.tokens: td(b.tokens, [Int(R), IN]), ref.index: td(refIdx, [Int(R)], .int32), ref.dw: td(n.dwBuf!, [TASKS, Int(R), FS])], targetOperations: nil, resultsDictionary: res)
  memcpy(wOut, wRef.contents(), Int(R) * FS * 4)
  for t in 0..<TASKS {
    let g = n.lay.chain(old: Array(UnsafeBufferPointer(start: gRef[t].contents().bindMemory(to: Float.self, capacity: lay.count), count: lay.count)), theta: theta, lay)
    memcpy(gOut + t * P, g, P * 4)
  }
}
@_cdecl("route_net_read") public func route_net_read(_ p: UnsafeMutableRawPointer, _ name: UnsafePointer<CChar>, _ offset: Int, _ bytes: Int, _ out: UnsafeMutableRawPointer) -> Int32 {
  let n = handle(p).net
  guard let b = n.named(String(cString: name)), offset + bytes <= b.length else { return -1 }
  let stage = n.device.makeBuffer(length: max(16, bytes), options: .storageModeShared)!
  let cb = n.queue.makeCommandBuffer()!, bl = cb.makeBlitCommandEncoder()!
  bl.copy(from: b, sourceOffset: offset, to: stage, destinationOffset: 0, size: bytes); bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  memcpy(out, stage.contents(), bytes)
  return 0
}
@_cdecl("route_net_free") public func route_net_free(_ p: UnsafeMutableRawPointer) { idle(handle(p).net); Unmanaged<Handle>.fromOpaque(p).release() }
