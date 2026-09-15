public typealias CutKey = String

public struct LinkKey: Hashable {
    public let a: Int
    public let b: Int

    public init(_ a: Int, _ b: Int) {
        self.a = Swift.min(a, b)
        self.b = Swift.max(a, b)
    }
}

public struct Topology {
    public struct Node {
        public let rate: Double
        public let bandwidth: Double

        public init(rate: Double, bandwidth: Double) {
            self.rate = rate
            self.bandwidth = bandwidth
        }
    }

    public struct Link {
        public let bandwidth: Double
        public let latency: Double

        public init(bandwidth: Double, latency: Double) {
            self.bandwidth = bandwidth
            self.latency = latency
        }
    }

    public let nodes: [Int: Node]
    public let links: [LinkKey: Link]

    public init(nodes: [Int: Node], links: [LinkKey: Link]) {
        self.nodes = nodes
        self.links = links
    }
}

public struct Program {
    public let work: Double
    public let bytes: Double

    public init(work: Double, bytes: Double) {
        self.work = work
        self.bytes = bytes
    }
}

public struct Placement {
    public struct Cut {
        public let links: Set<LinkKey>
        public let bytes: Double

        public init(links: Set<LinkKey>, bytes: Double) {
            self.links = links
            self.bytes = bytes
        }
    }

    public struct Hop {
        public let link: LinkKey
        public let bytes: Double

        public init(link: LinkKey, bytes: Double) {
            self.link = link
            self.bytes = bytes
        }
    }

    public let work: [Int: Double]
    public let bytes: [Int: Double]
    public let cuts: [CutKey: Cut]
    public let path: [Hop]

    public init(work: [Int: Double], bytes: [Int: Double], cuts: [CutKey: Cut], path: [Hop]) {
        self.work = work
        self.bytes = bytes
        self.cuts = cuts
        self.path = path
    }
}

public struct Bounds {
    public let compute: Double
    public let memory: Double
    public let cut: [CutKey: Double]
    public let path: Double
    public let max: Double
}

/// Their maximum is a lower bound, not an exact execution-time formula.
public func bounds(_ program: Program, _ placement: Placement, _ topology: Topology) -> Bounds {
    var rate = 0.0
    var bandwidth = 0.0
    for node in topology.nodes.values {
        rate += node.rate
        bandwidth += node.bandwidth
    }
    var compute = program.work == 0 ? 0 : program.work / rate
    var memory = program.bytes == 0 ? 0 : program.bytes / bandwidth
    for (node, work) in placement.work {
        compute = Swift.max(compute, work / topology.nodes[node]!.rate)
    }
    for (node, bytes) in placement.bytes {
        memory = Swift.max(memory, bytes / topology.nodes[node]!.bandwidth)
    }
    var cut: [CutKey: Double] = [:]
    cut.reserveCapacity(placement.cuts.count)
    var maximum = Swift.max(compute, memory)
    for (key, crossing) in placement.cuts {
        var capacity = 0.0
        for link in crossing.links {
            capacity += topology.links[link]!.bandwidth
        }
        let seconds = crossing.bytes == 0 ? 0 : crossing.bytes / capacity
        cut[key] = seconds
        maximum = Swift.max(maximum, seconds)
    }
    var path = 0.0
    for hop in placement.path {
        let link = topology.links[hop.link]!
        path += link.latency + hop.bytes / link.bandwidth
    }
    return Bounds(compute: compute, memory: memory, cut: cut, path: path, max: Swift.max(maximum, path))
}
