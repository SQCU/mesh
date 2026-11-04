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
