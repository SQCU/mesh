import CMesh
import CoreML
import Foundation
import Metal

public typealias MeshOperands = UnsafeBufferPointer<mesh_operand>

fileprivate enum MeshSubmission {
    case cpu((MeshOperands, MeshOperands) -> Void)
    case metal(MTLDevice, (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void)
    case prediction(MLModel, [MeshFeatures], [MLPredictionOptions])
}

fileprivate final class MeshFeatures: NSObject, MLFeatureProvider {
    let featureNames: Set<String>
    let bindings: [String: (Int, MeshBindings<MLFeatureValue>)]
    var operands = MeshOperands(start: nil, count: 0)

    // design/algorithm-sources.md#programkernel_call
    init(names: Set<String>, bindings: [String: (Int, MeshBindings<MLFeatureValue>)]) {
        featureNames = names; self.bindings = bindings
        super.init()
    }

    // design/algorithm-sources.md#programkernel_call
    func featureValue(for featureName: String) -> MLFeatureValue? {
        bindings[featureName].map { index, values in values[operands[index]] }
    }
}

public struct TensorFunction {
    fileprivate let prepare: (Mesh, [TensorPart], [TensorPart]) throws -> MeshSubmission

    // design/algorithm-sources.md#programkernel_call
    private init(_ prepare: @escaping (Mesh, [TensorPart], [TensorPart]) throws -> MeshSubmission) { self.prepare = prepare }

    // design/algorithm-sources.md#programkernel_call
    public static func cpu(_ function: @escaping (MeshOperands, MeshOperands) -> Void) -> Self {
        Self { _, _, _ in .cpu(function) }
    }

    // design/algorithm-sources.md#programkernel_call
    public static func metal(_ device: MTLDevice, _ function: @escaping (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void) -> Self {
        Self { _, _, _ in .metal(device, function) }
    }

    // design/algorithm-sources.md#programkernel_call
    public static func prediction(_ model: MLModel, inputs: [(String, (MeshSpan) throws -> MLMultiArray)],
                                  outputs: [(String, (MeshSpan) throws -> MLMultiArray)]) -> Self {
        Self { mesh, parts, results in
            precondition(parts.count == inputs.count && results.count == outputs.count)
            let bindings = try Dictionary(uniqueKeysWithValues: inputs.indices.map { i in
                (inputs[i].0, (i, try mesh.bindings(parts[i]) { MLFeatureValue(multiArray: try inputs[i].1($0)) }))
            })
            let names = Set(bindings.keys)
            let features = (0..<mesh.count).map { _ in MeshFeatures(names: names, bindings: bindings) }
            let arrays = try outputs.indices.map { try mesh.bindings(results[$0], using: outputs[$0].1) }
            let options = (0..<mesh.count).map { index in
                let option = MLPredictionOptions()
                option.outputBackings = Dictionary(uniqueKeysWithValues: outputs.indices.map { (outputs[$0].0, arrays[$0].values[index]) })
                return option
            }
            return .prediction(model, features, options)
        }
    }

    // design/algorithm-sources.md#programkernel_call
    public init<Input, Output>(inputViews: [(MeshSpan) throws -> Input], outputViews: [(MeshSpan) throws -> Output],
                              _ function: @escaping ([MeshBindings<Input>], [MeshBindings<Output>]) throws -> TensorFunction) {
        prepare = { mesh, inputs, outputs in
            precondition(inputs.count == inputViews.count && outputs.count == outputViews.count)
            let x = try zip(inputs, inputViews).map { try mesh.bindings($0.0, using: $0.1) }
            let y = try zip(outputs, outputViews).map { try mesh.bindings($0.0, using: $0.1) }
            return try function(x, y).prepare(mesh, inputs, outputs)
        }
    }
}

public struct MeshBindings<Value> {
    public let values: [Value]
    public let index: (mesh_operand) -> Int

    // design/algorithm-sources.md#programtensor
    public subscript(_ operand: mesh_operand) -> Value { values[index(operand)] }
}

public struct MeshSpan {
    public let data: UnsafeMutableRawBufferPointer
    fileprivate let memory: MeshMemory

    // design/algorithm-sources.md#programtensor
    public func metal(device: MTLDevice) -> MTLBuffer { memory.metal(device, data: data) }

    // design/algorithm-sources.md#programkernel_call
    public func multiArray(shape: [Int], strides: [Int], type: MLMultiArrayDataType) throws -> MLMultiArray {
        let bytes = Int(type.rawValue & 0xff) / 8
        precondition(shape.count == strides.count && shape.allSatisfy { $0 > 0 } && strides.allSatisfy { $0 >= 0 })
        let last = zip(shape, strides).reduce(0) { $0 + ($1.0 - 1) * $1.1 }
        precondition((last + 1) * bytes <= data.count)
        return try MLMultiArray(dataPointer: data.baseAddress!, shape: shape.map(NSNumber.init), dataType: type,
                               strides: strides.map(NSNumber.init), deallocator: { [memory] _ in _ = memory })
    }
}

public struct TensorPart {
    public let rank: Int
    public let bytes: Int
    // design/algorithm-sources.md#tensorpartpartial
    public let partial: Bool
    fileprivate let section: mesh_section?
    fileprivate let shared: Bool
    fileprivate var identity: Int

    // design/algorithm-sources.md#tensorpartpartial
    fileprivate func withPartial(_ partial: Bool) -> TensorPart {
        TensorPart(rank: rank, bytes: bytes, partial: partial, section: section, shared: shared, identity: identity)
    }
}

// design/algorithm-sources.md#mesherror
public enum MeshError: Error {
    case partialOperand(TensorPart)
    case busy
    case link(peer: Int, code: Int32)
    case function(call: Int, code: Int32)
}

private struct MeshDelivery: Hashable {
    let value, source, destination, queue: Int
}

private final class MeshMemory {
    let context: UnsafeMutablePointer<mesh_ctx>
    var sections: [mesh_section] = []
    private let buffers = NSMapTable<NSString, AnyObject>(keyOptions: .strongMemory, valueOptions: .weakMemory)

    // design/algorithm-sources.md#programtensor
    init(_ name: String) throws {
        context = .allocate(capacity: 1)
        context.initialize(to: mesh_ctx())
        let error = mesh_attach(context, name)
        if error != 0 {
            context.deinitialize(count: 1); context.deallocate()
            throw POSIXError(POSIXErrorCode(rawValue: error)!)
        }
    }

    // design/algorithm-sources.md#programtensor
    deinit {
        releaseSections()
        mesh_detach(context)
        context.deinitialize(count: 1); context.deallocate()
    }

    // design/algorithm-sources.md#programtensor
    func releaseSections() {
        for section in sections { mesh_section_release(context, section) }
        sections.removeAll()
    }

    // design/algorithm-sources.md#programtensor
    func metal(_ device: MTLDevice, data: UnsafeMutableRawBufferPointer) -> MTLBuffer {
        let pageSize = Int(context.pointee.M.pointee.pgsz)
        let address = data.baseAddress!, bytes = (data.count + pageSize - 1) / pageSize * pageSize
        let key = "\(device.registryID):\(UInt(bitPattern: address)):\(bytes)" as NSString
        if let buffer = buffers.object(forKey: key) as? MTLBuffer { return buffer }
        let buffer = device.makeBuffer(bytesNoCopy: address, length: bytes, options: .storageModeShared,
                                       deallocator: { [self] _, _ in _ = self })!
        buffers.setObject(buffer, forKey: key)
        return buffer
    }
}

private final class MeshInvocation {
    let inputCount: Int, outputCount: Int
    let memory: MeshMemory
    let submit: (OpaquePointer, UInt32, MeshOperands, MeshOperands) -> Void

    // design/algorithm-sources.md#programkernel_call
    init(_ function: MeshSubmission, memory: MeshMemory, inputs: Int, outputs: Int, count: Int,
         copies: [(input: Int, source: mesh_section, target: mesh_section)]) {
        self.memory = memory; inputCount = inputs; outputCount = outputs
        let context = memory.context, block = mesh_block_pages(context)
        let quantum = Int(block * context.pointee.M.pointee.pgsz)
        let chunks = copies.map { copy in
            stride(from: 0, to: copy.source.bytes, by: quantum).map { min(quantum, copy.source.bytes - $0) }
        }
        let launch: (OpaquePointer, UInt32, MeshOperands, MeshOperands) -> Void
        let copyOnCPU: Bool
        switch function {
        case .cpu(let function):
            copyOnCPU = true
            launch = { call, _, inputs, outputs in
                function(inputs, outputs)
                mesh_call_complete(call)
            }
        case .metal(let device, let function):
            copyOnCPU = false
            let encode: (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void
            if copies.isEmpty {
                encode = function
            } else {
                let sources = copies.map { copy -> (UInt32, [MTLBuffer]) in
                    let range = mesh_receive_range(context, copy.source.channel)
                    let buffers = stride(from: range.first, to: range.first + range.count, by: Int(block)).map { page in
                        memory.metal(device, data: UnsafeMutableRawBufferPointer(start: mesh_page_address(context, page), count: quantum))
                    }
                    return (range.first, buffers)
                }
                let targets = copies.map { copy in
                    (0..<copy.target.count).map { index in
                        memory.metal(device, data: UnsafeMutableRawBufferPointer(start: mesh_section_address(context, copy.target, index), count: copy.target.bytes))
                    }
                }
                encode = { command, inputs, outputs in
                    let blit = command.makeBlitCommandEncoder()!
                    for i in copies.indices {
                        let input = inputs[copies[i].input]
                        for (chunk, bytes) in chunks[i].enumerated() {
                            let page = mesh_row_page(context, input.row + UInt32(chunk) * block)
                            blit.copy(from: sources[i].1[Int((page - sources[i].0) / block)], sourceOffset: 0,
                                      to: targets[i][Int(input.index)], destinationOffset: chunk * quantum, size: (bytes + 3) / 4 * 4)
                        }
                    }
                    blit.endEncoding()
                    function(command, inputs, outputs)
                }
            }
            let queue = device.makeCommandQueue(maxCommandBufferCount: count)!
            let commands = (0..<count).map { _ in queue.makeCommandBuffer()! }
            launch = { call, index, inputs, outputs in
                let command = commands[Int(index)]
                encode(command, inputs, outputs)
                command.addCompletedHandler { command in
                    if command.status == .completed { mesh_call_complete(call) }
                    else { mesh_call_fail(call, Int32((command.error as NSError?)?.code ?? -1)) }
                }
                command.commit()
            }
        case .prediction(let model, let features, let options):
            copyOnCPU = true
            launch = { call, index, inputs, _ in
                let provider = features[Int(index)]
                provider.operands = inputs
                model.__prediction(fromFeatures: provider, options: options[Int(index)]) { _, error in
                    if let error { mesh_call_fail(call, Int32((error as NSError).code)) }
                    else { mesh_call_complete(call) }
                }
            }
        }
        if copyOnCPU && !copies.isEmpty {
            submit = { call, index, inputs, outputs in
                for i in copies.indices {
                    let input = inputs[copies[i].input]
                    for (chunk, bytes) in chunks[i].enumerated() {
                        let page = mesh_row_page(context, input.row + UInt32(chunk) * block)
                        memcpy(input.data!.advanced(by: chunk * quantum), mesh_page_address(context, page), bytes)
                    }
                }
                launch(call, index, inputs, outputs)
            }
        } else {
            submit = launch
        }
    }

}

public final class Mesh {
    public let rank: Int, size: Int, count: Int
    private let memory: MeshMemory
    private let calls: OpaquePointer
    private let peers: [Int]
    private var transfer: UInt32 = 0
    private var value = 0
    private var deliveries: [MeshDelivery: TensorPart] = [:]
    private var preparations: [() throws -> Void] = []
    private var partialContributions: Set<Int> = []
    private var routes: [Placement.Edge: [Int]]

    // design/algorithm-sources.md#program
    public init(region: String, rank: Int, size: Int, workers: Int, count: Int = 1, placement: Placement = Placement()) throws {
        precondition(size > 0 && (0..<size).contains(rank) && count > 0)
        let memory = try MeshMemory(region)
        let owner = Unmanaged.passRetained(memory).toOpaque()
        guard let calls = mesh_calls_create(memory.context, UInt32(workers), UInt32(count), owner,
            { owner in Unmanaged<MeshMemory>.fromOpaque(owner!).release() }) else {
            Unmanaged<MeshMemory>.fromOpaque(owner).release()
            throw POSIXError(POSIXErrorCode(rawValue: errno)!)
        }
        self.memory = memory; self.calls = calls; self.rank = rank; self.size = size; self.count = count
        routes = placement.routes
        peers = (0..<Int(memory.context.pointee.M.pointee.links)).map { Int(mesh_links(memory.context.pointee.M)[$0].peer) }
    }

    // design/algorithm-sources.md#programtensor
    deinit { mesh_calls_destroy(calls) }

    // design/algorithm-sources.md#programtensor
    private func part(on owner: Int, bytes: Int, partial: Bool = false, shared: Bool = false, queue: UInt32? = nil) throws -> TensorPart {
        let identity = value; value += 1
        var section: mesh_section?
        if owner == rank {
            var local = mesh_section()
            let error = mesh_section_create(memory.context, bytes, UInt32(shared ? 1 : count), queue ?? MESH_ABSENT, &local)
            if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
            if shared { local.stride = 0 }
            memory.sections.append(local); section = local
        }
        return TensorPart(rank: owner, bytes: bytes, partial: partial, section: section, shared: shared, identity: identity)
    }

    // design/algorithm-sources.md#programtensor
    public func tensor(on owner: Int, sections: [Int]) throws -> [TensorPart] {
        precondition((0..<size).contains(owner))
        return try sections.map { try part(on: owner, bytes: $0) }
    }

    // design/algorithm-sources.md#programkernel_call
    public func call(_ function: TensorFunction, inputs: [TensorPart], outputs: [TensorPart],
                     on owner: Int, worker: Int) throws {
        preparations.append { [unowned self] in
            if let part = inputs.first(where: { $0.partial || partialContributions.contains($0.identity) }) {
                throw MeshError.partialOperand(part.withPartial(true))
            }
        }
        try bind(function, inputs: inputs, outputs: outputs, on: owner, worker: worker)
    }

    // design/algorithm-sources.md#programkernel_call
    private func bind(_ function: TensorFunction, inputs: [TensorPart], outputs: [TensorPart],
                      on owner: Int, worker: Int) throws {
        precondition(inputs.allSatisfy { $0.rank == owner } && outputs.allSatisfy { $0.rank == owner && !$0.shared })
        if owner != rank { return }
        preparations.append { [unowned self] in
            precondition(outputs.allSatisfy { $0.section!.channel == MESH_ABSENT })
            var views = inputs, storage: [TensorPart] = []
            var copies: [(input: Int, source: mesh_section, target: mesh_section)] = []
            for (i, input) in inputs.enumerated() {
                let section = input.section!
                if section.channel != MESH_ABSENT && section.pages > mesh_block_pages(memory.context) {
                    if let existing = copies.first(where: { $0.source.first == section.first }) {
                        views[i] = views[existing.input]
                        continue
                    }
                    let target = try part(on: owner, bytes: input.bytes, partial: input.partial)
                    views[i] = target; storage.append(target)
                    copies.append((i, section, target.section!))
                }
            }
            let invocation = MeshInvocation(try function.prepare(self, views, outputs), memory: memory,
                                            inputs: inputs.count, outputs: outputs.count, count: count, copies: copies)
            try bind(invocation, inputs: inputs, views: views, outputs: outputs + storage, worker: worker)
        }
    }

    // design/algorithm-sources.md#programkernel_call
    private func bind(_ invocation: MeshInvocation, inputs: [TensorPart], views: [TensorPart], outputs: [TensorPart], worker: Int) throws {
        let argument = Unmanaged.passRetained(invocation).toOpaque()
        let inputRows = inputs.map { $0.section! }, viewRows = views.map { $0.section! }, outputRows = outputs.map { $0.section! }
        let call = mesh_call_bind(calls, UInt32(worker), inputRows, viewRows, inputRows.count, outputRows, outputRows.count,
            { call, index, argument, inputs, outputs in
                let invocation = Unmanaged<MeshInvocation>.fromOpaque(argument!).takeUnretainedValue()
                invocation.submit(call!, index, MeshOperands(start: inputs, count: invocation.inputCount),
                                  MeshOperands(start: outputs, count: invocation.outputCount))
            }, argument,
            { argument in Unmanaged<MeshInvocation>.fromOpaque(argument!).release() })
        if call == nil {
            Unmanaged<MeshInvocation>.fromOpaque(argument).release()
            throw POSIXError(POSIXErrorCode(rawValue: errno)!)
        }
    }

    // design/algorithm-sources.md#programtensor
    fileprivate func bindings<Value>(_ part: TensorPart, using make: (MeshSpan) throws -> Value) throws -> MeshBindings<Value> {
        let source = part.section!, context = memory.context
        let pages: [UInt32], index: (mesh_operand) -> Int
        if source.channel != MESH_ABSENT {
            let range = mesh_receive_range(context, source.channel)
            let first = Int(range.first), quantum = Int(mesh_block_pages(context))
            pages = stride(from: first, through: first + Int(range.count - source.pages), by: quantum).map(UInt32.init)
            index = { (Int($0.page) - first) / quantum }
        } else {
            pages = (0..<source.count).map { mesh_row_page(context, source.first + $0 * source.stride) }
            let stride = part.shared ? 0 : 1
            index = { Int($0.index) * stride }
        }
        let values = try pages.map { page in
            try make(MeshSpan(data: UnsafeMutableRawBufferPointer(start: mesh_page_address(context, page),
                                                                 count: source.bytes), memory: memory))
        }
        return MeshBindings(values: values, index: index)
    }

    // design/algorithm-sources.md#programtensor
    public func constant(on owner: Int, bytes: Int, initialize: (MeshSpan) throws -> Void) throws -> TensorPart {
        let part = try part(on: owner, bytes: bytes, shared: true)
        if let section = part.section {
            let span = MeshSpan(data: UnsafeMutableRawBufferPointer(start: mesh_section_address(memory.context, section, 0),
                                                                   count: bytes), memory: memory)
            try initialize(span)
            mesh_section_constant(memory.context, section)
        }
        return part
    }

    // design/algorithm-sources.md#program
    public func map(_ function: TensorFunction, inputs: [[TensorPart]], outputs: [TensorPart],
                    workers: [Int]) throws {
        precondition(inputs.allSatisfy { $0.count == outputs.count } && workers.count == outputs.count)
        for i in outputs.indices {
            try call(function, inputs: inputs.map { $0[i] }, outputs: [outputs[i]],
                     on: outputs[i].rank, worker: workers[i])
        }
    }

    // design/algorithm-sources.md#programcopy
    public func send(_ part: TensorPart, to destination: Int, queue: Int = 0) throws -> TensorPart {
        if part.rank == destination { return part }
        let path = routes[Placement.Edge(part.rank, destination)] ?? [destination]
        precondition(path.last == destination && path.allSatisfy { (0..<size).contains($0) })
        var source = part
        for next in path {
            if source.rank == next { continue }
            let delivery = MeshDelivery(value: source.identity, source: source.rank, destination: next, queue: queue)
            if let existing = deliveries[delivery] { source = existing.withPartial(source.partial); continue }
            let identity = transfer; transfer += 1
            let participating = rank == source.rank || rank == next
            let channel = participating ? mesh_peer_channel(memory.context, UInt32(rank == next ? source.rank : next), UInt32(queue)) : 0
            if channel == MESH_ABSENT { throw POSIXError(.ENETUNREACH) }
            var output = try self.part(on: next, bytes: source.bytes, partial: source.partial, shared: source.shared, queue: channel)
            output.identity = source.identity
            if participating {
                let local = rank == next ? output : source
                let error = mesh_transfer_bind(memory.context, channel, rank == next ? 1 : 0,
                                               identity, local.section!)
                if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
            }
            deliveries[delivery] = output
            source = output
        }
        return source
    }

    // design/algorithm-sources.md#collective-movement
    public func broadcast(_ parts: [TensorPart], to destinations: [Int], queue: Int = 0) throws -> [[TensorPart]] {
        try destinations.map { destination in try parts.map { try send($0, to: destination, queue: queue) } }
    }

    // design/algorithm-sources.md#collective-movement
    public func scatter(_ parts: [TensorPart], to destinations: [Int], queue: Int = 0) throws -> [TensorPart] {
        precondition(parts.count == destinations.count)
        return try zip(parts, destinations).map { try send($0.0, to: $0.1, queue: queue) }
    }

    // design/algorithm-sources.md#collective-movement
    public func allScatter(_ parts: [[TensorPart]], to destinations: [Int], queue: Int = 0) throws -> [[TensorPart]] {
        let scattered = try parts.map { try scatter($0, to: destinations, queue: queue) }
        return destinations.indices.map { i in scattered.map { $0[i] } }
    }

    // design/algorithm-sources.md#collective-movement
    public func gather(_ parts: [TensorPart], to destination: Int, queue: Int = 0) throws -> [TensorPart] {
        try parts.map { try send($0, to: destination, queue: queue) }
    }

    // design/algorithm-sources.md#collective-movement
    public func allGather(_ parts: [TensorPart], queue: Int = 0) throws -> [[TensorPart]] {
        try (0..<size).map { try gather(parts, to: $0, queue: queue) }
    }

    // design/algorithm-sources.md#collective-movement
    public func allToAll(_ parts: [[TensorPart]], queue: Int = 0) throws -> [[TensorPart]] {
        try allScatter(parts, to: Array(0..<size), queue: queue)
    }

    // design/algorithm-sources.md#collectivereduce_scatter
    public func reduce(_ parts: [TensorPart], to destination: Int, using combine: TensorFunction,
                       worker: Int, queue: Int = 0) throws -> TensorPart {
        precondition(!parts.isEmpty && parts.allSatisfy { $0.bytes == parts[0].bytes })
        partialContributions.formUnion(parts.map(\.identity))
        var level = parts.map { $0.withPartial(true) }
        while level.count > 1 {
            var next: [TensorPart] = []
            for i in stride(from: 0, to: level.count, by: 2) {
                if i + 1 == level.count { next.append(level[i]); continue }
                let owner = level.count == 2 ? destination : level[i].rank
                let pair = try gather([level[i], level[i + 1]], to: owner, queue: queue)
                let output = [try part(on: owner, bytes: pair[0].bytes, partial: level.count != 2)]
                try bind(combine, inputs: pair, outputs: output, on: owner, worker: worker)
                next.append(output[0])
            }
            level = next
        }
        var result = try send(level[0].withPartial(false), to: destination, queue: queue)
        result.identity = value; value += 1
        return result
    }

    // design/algorithm-sources.md#collectivereduce_scatter
    public func reduceScatter(_ contributions: [[TensorPart]], to owners: [Int], using combine: TensorFunction,
                              workers: [Int], queue: Int = 0) throws -> [TensorPart] {
        precondition(contributions.allSatisfy { $0.count == owners.count } && workers.count == owners.count)
        return try owners.indices.map { i in
            try reduce(contributions.map { $0[i] }, to: owners[i], using: combine, worker: workers[i], queue: queue)
        }
    }

    // design/algorithm-sources.md#collectivereduce_scatter
    public func allReduce(_ contributions: [[TensorPart]], owners: [Int], using combine: TensorFunction,
                          workers: [Int], queue: Int = 0) throws -> [[TensorPart]] {
        try allGather(reduceScatter(contributions, to: owners, using: combine, workers: workers, queue: queue), queue: queue)
    }

    // design/algorithm-sources.md#programkernel_call
    public func start() throws {
        let storageError = mesh_transfers_prepare(memory.context)
        if storageError != 0 { throw POSIXError(POSIXErrorCode(rawValue: storageError)!) }
        for prepare in preparations { try prepare() }
        preparations.removeAll()
        deliveries.removeAll()
        partialContributions.removeAll()
        routes.removeAll()
        memory.releaseSections()
        let error = mesh_calls_start(calls)
        if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
        mesh_transfers_start(memory.context)
    }

    // design/algorithm-sources.md#program
    public func submit(_ index: Int) {
        mesh_calls_submit(calls, UInt32(index))
    }

    // design/algorithm-sources.md#meshresult
    public func result(_ index: Int) -> Result<Void, MeshError> {
        let status = mesh_calls_result(calls, UInt32(index))
        let identity = Int((status >> 32) & 0x3fffffff), code = Int32(truncatingIfNeeded: status)
        switch status >> 62 {
        case 0: return .success(())
        case 1: return .failure(.link(peer: peers[identity], code: code))
        case 2: return .failure(.function(call: identity, code: code))
        default: return .failure(.busy)
        }
    }

    // design/algorithm-sources.md#collectivesync_on_remote_fill
    public func syncOnRemoteFill(_ parts: [TensorPart], index: Int = 0) {
        let sections = parts.map { $0.section! }
        mesh_sync_on_remote_fill(memory.context, sections, sections.count, UInt32(index))
    }
}
