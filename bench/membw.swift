// Usage: xcrun swiftc -O -target arm64-apple-macos14 bench/membw.swift -o /tmp/membw && /tmp/membw [MiB per buffer=512] [min GPU seconds=0.5]
// 2026-11-16 baseline (512 MiB x2, 0.5 s): M5 Ultra Studios 1095-1104 GB/s, M4 Pro Mini 231-232 GB/s, M5 Max laptop 468-506 GB/s.
// GPU memory-bus healthcheck: float4 FMA stream a->b over buffers far larger than the SLC,
// repeated until >= minSeconds of GPU time; bandwidth from GPU timestamps; full CPU verify.
import Metal
import Foundation

let mib = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 512
let minSeconds = CommandLine.arguments.count > 2 ? Double(CommandLine.arguments[2])! : 0.5
let n4 = mib * 1024 * 1024 / 16            // float4 elements per buffer

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void fma4(device const float4* a [[buffer(0)]], device float4* b [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) { b[i] = fma(a[i], float4(1.5f), float4(0.25f)); }
"""
guard let dev = MTLCreateSystemDefaultDevice() else { print("FAIL no metal device"); exit(1) }
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "fma4")!)
let q = dev.makeCommandQueue()!
let a = dev.makeBuffer(length: n4 * 16, options: .storageModeShared)!
let b = dev.makeBuffer(length: n4 * 16, options: .storageModeShared)!
let pa = a.contents().bindMemory(to: Float.self, capacity: n4 * 4)
for i in 0..<(n4 * 4) { pa[i] = Float(i & 0xFFFF) }   // exact in fp32; 1.5x+0.25 stays exact

let tg = MTLSize(width: pso.maxTotalThreadsPerThreadgroup, height: 1, depth: 1)
let grid = MTLSize(width: n4, height: 1, depth: 1)
func run(_ passes: Int) -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
    for _ in 0..<passes { e.dispatchThreads(grid, threadsPerThreadgroup: tg) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.status != .completed { print("FAIL cb status \(cb.status.rawValue) \(String(describing: cb.error))"); exit(1) }
    return cb.gpuEndTime - cb.gpuStartTime
}
_ = run(2)                                      // warm-up
var gpu = 0.0, passes = 0, best = 0.0
let wall0 = Date()
while gpu < minSeconds {
    let t = run(8); gpu += t; passes += 8
    best = max(best, 8.0 * 2.0 * Double(n4 * 16) / t / 1e9)
}
let wall = Date().timeIntervalSince(wall0)
let pb = b.contents().bindMemory(to: Float.self, capacity: n4 * 4)
var bad = 0
for i in 0..<(n4 * 4) where pb[i] != pa[i] * 1.5 + 0.25 { bad += 1 }
let avg = Double(passes) * 2.0 * Double(n4 * 16) / gpu / 1e9
let thermal = ["nominal", "fair", "serious", "critical"][ProcessInfo.processInfo.thermalState.rawValue]
print(String(format: "%@ | %@ | buf %d MiB x2 | %d passes | gpu %.3f s (wall %.3f) | avg %.0f GB/s best-CB %.0f GB/s | verify %@ | thermal %@",
             ProcessInfo.processInfo.hostName, dev.name, mib, passes, gpu, wall, avg, best,
             bad == 0 ? "OK" : "FAIL(\(bad) mismatches)", thermal))
