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
    // number in the map as its port and its bandwidth in bits a second (0 where none is reported)
    public static func observe() throws -> Topology {
        try withMap { map, node, count in
            let nodes = node.prefix(count).map(Int.init)
            var links: [Topology.Pair: [Link]] = [:]
            for i in 0..<Int(map.pointee.links) {
                links[Topology.Pair(nodes[Int(map.pointee.link[i].0)], nodes[Int(map.pointee.link[i].1)]), default: []]
                    .append(Link(port: i, bandwidth: map.pointee.bandwidth[i]))
            }
            return Topology(nodes: Set(nodes), links: links)
        }
    }

    // design/algorithm-sources.md#meshobserve
    // the bridges' map of the nodes `members` names by their bridges' numbers (none: the whole map; else mesh_map's
    // members' map, its other nodes relays with its table), its storage grown until it fits, handed to `body` with each
    // map node's bridge and the map's node count while it lives
    static func withMap<Value>(_ members: [UInt32] = [],
                           _ body: (UnsafeMutablePointer<mesh_link_map>, [UInt32], Int) throws -> Value) throws -> Value {
        var room = (nodes: max(16, members.count), links: 64)
        while true {
            var node = [UInt32](repeating: 0, count: room.nodes), mark = [UInt32](repeating: 0, count: room.nodes)
            var link = [(UInt32, UInt32)](repeating: (0, 0), count: room.links), bandwidth = [UInt64](repeating: 0, count: room.links)
            var next = [UInt32](repeating: 0, count: room.nodes * room.nodes), map = mesh_link_map()
            map.nodes = UInt32(room.nodes); map.links = UInt32(room.links)
            let made: Value? = try link.withUnsafeMutableBufferPointer { l in try bandwidth.withUnsafeMutableBufferPointer { b in
                try next.withUnsafeMutableBufferPointer { x in try mark.withUnsafeMutableBufferPointer { k in
                    map.link = l.baseAddress; map.bandwidth = b.baseAddress
                    if !members.isEmpty { map.next = x.baseAddress; map.mark = k.baseAddress }
                    let count = mesh_map(members, UInt32(members.count), &node, nil, &map)
                    if count == -ENOSPC { return nil }
                    if count < 0 { throw POSIXError(POSIXErrorCode(rawValue: -count) ?? .EIO) }
                    return try body(&map, node, Int(count))
                } }
            } }
            if let made { return made }
            room = (max(room.nodes, Int(map.nodes)), max(room.links, Int(map.links)))
        }
    }
}
