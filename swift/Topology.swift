import CMesh
import Foundation

public struct Link {
    public let port: Int
    public let bandwidth: UInt64
    public let latency: Double?

    // design/algorithm-sources.md#link
    public init(port: Int, bandwidth: UInt64, latency: Double? = nil) {
        self.port = port; self.bandwidth = bandwidth; self.latency = latency
    }
}

public struct Topology {
    public struct Pair: Hashable {
        public let a: Int, b: Int

        // design/algorithm-sources.md#topologypair
        public init(_ a: Int, _ b: Int) { self.a = min(a, b); self.b = max(a, b) }
    }

    public let nodes: Set<Int>
    public let links: [Pair: [Link]]

    // design/algorithm-sources.md#topology
    public init(nodes: Set<Int>, links: [Pair: [Link]]) { self.nodes = nodes; self.links = links }

    // design/algorithm-sources.md#topologylinks
    public func links(between a: Int, _ b: Int) -> [Link] { links[Pair(a, b)] ?? [] }

    // design/algorithm-sources.md#topologymerging
    public func merging(_ other: Topology) -> Topology {
        Topology(nodes: nodes.union(other.nodes), links: links.merging(other.links, uniquingKeysWith: +))
    }
}

extension Mesh {
    // design/algorithm-sources.md#meshobserve
    // the bridges' whole map (mesh-plan.h mesh_map, n 0): its nodes by their bridges' numbers, a link a cable, its
    // number in the map as its port, its bandwidth 8e9 / beta bits a second and its latency alpha (seconds)
    public static func observe() throws -> Topology {
        var room = (nodes: 16, links: 64)
        while true {
            var node = [UInt32](repeating: 0, count: room.nodes)
            var link = [(UInt32, UInt32)](repeating: (0, 0), count: room.links)
            var cost = [(Double, Double)](repeating: (0, 0), count: room.links)
            var map = mesh_link_map()
            map.nodes = UInt32(room.nodes); map.links = UInt32(room.links)
            let count = link.withUnsafeMutableBufferPointer { links in
                cost.withUnsafeMutableBufferPointer { costs in
                    map.link = links.baseAddress; map.cost = costs.baseAddress
                    return mesh_map(nil, 0, &node, nil, &map)
                }
            }
            if count == -ENOSPC {
                room = (max(room.nodes, Int(map.nodes)), max(room.links, Int(map.links)))
                continue
            }
            if count < 0 { throw POSIXError(POSIXErrorCode(rawValue: -count) ?? .EIO) }
            let nodes = node.prefix(Int(count)).map(Int.init)
            var links: [Topology.Pair: [Link]] = [:]
            for i in 0..<Int(map.links) {
                let (alpha, beta) = cost[i]
                links[Topology.Pair(nodes[Int(link[i].0)], nodes[Int(link[i].1)]), default: []]
                    .append(Link(port: i, bandwidth: beta > 0 ? UInt64(8e9 / beta) : 0, latency: alpha * 1e-6))
            }
            return Topology(nodes: Set(nodes), links: links)
        }
    }
}
