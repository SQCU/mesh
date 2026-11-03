import CMesh
import Foundation

public struct Link {
    public let device: String
    public let bandwidth: UInt64
    public let latency: Double?

    // design/algorithm-sources.md#link
    public init(device: String, bandwidth: UInt64, latency: Double? = nil) {
        self.device = device; self.bandwidth = bandwidth; self.latency = latency
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
    public static func observe(region: String = "/mesh0") throws -> Topology {
        var node: UInt32 = 0
        var views: [mesh_link_view] = []
        while true {
            let count = mesh_observe(region, &views, UInt32(views.count), &node)
            if count < 0 { throw POSIXError(POSIXErrorCode(rawValue: -count) ?? .EIO) }
            if count > views.count {
                views = Array(repeating: mesh_link_view(), count: Int(count))
                continue
            }
            var nodes: Set<Int> = [Int(node)]
            var links: [Topology.Pair: [Link]] = [:]
            for var view in views.prefix(Int(count)) where view.phase == MESH_PAIRED {
                nodes.insert(Int(view.peer))
                let device = withUnsafeBytes(of: &view.device) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
                links[Topology.Pair(Int(node), Int(view.peer)), default: []].append(Link(device: device, bandwidth: view.bandwidth))
            }
            return Topology(nodes: nodes, links: links)
        }
    }
}
