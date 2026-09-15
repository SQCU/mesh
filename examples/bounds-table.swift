import Foundation
import Mesh

@main
struct BoundsTable {
    static func main() {
        for (n27, n23) in [(1, 3), (2, 17), (1, 1)] {
            let totalRate = Double(n27) + 0.67 * Double(n23)
            let totalBandwidth = Double(n27) + 0.54 * Double(n23)
            var nodes: [Int: Topology.Node] = [:]
            var work: [Int: Double] = [:]
            var bytes: [Int: Double] = [:]
            for node in 0..<(n27 + n23) {
                let rate = node < n27 ? 1.0 : 0.67
                let bandwidth = node < n27 ? 1.0 : 0.54
                nodes[node] = Topology.Node(rate: rate, bandwidth: bandwidth)
                work[node] = 600 * rate / totalRate
                bytes[node] = 600 * bandwidth / totalBandwidth
            }
            let topology = Topology(nodes: nodes, links: [:])
            let compute = bounds(Program(work: 600, bytes: 0),
                                 Placement(work: work, bytes: [:], cuts: [:], path: []), topology)
            let memory = bounds(Program(work: 0, bytes: 600),
                                Placement(work: [:], bytes: bytes, cuts: [:], path: []), topology)
            print("\(n27) capability 27 + \(n23) capability 23    "
                  + String(format: "%.2f%9.2f", compute.max, memory.max))
        }
    }
}
