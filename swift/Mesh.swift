import CMesh
import CoreML
import Foundation
import Metal

public typealias MeshOperands = UnsafeBufferPointer<mesh_operand>

fileprivate enum MeshSubmission {
    case cpu((MeshOperands, MeshOperands) -> Void)
    case metal(MTLDevice, (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void)
    case prediction(MLModel, (MeshOperands, MeshOperands) -> (MLFeatureProvider, MLPredictionOptions))
}

public struct TensorFunction {
    fileprivate let prepare: (Mesh, [TensorPart], [TensorPart]) throws -> MeshSubmission

    // design/algorithm-sources.md#programkernel_call
    private init(_ submission: MeshSubmission) { prepare = { _, _, _ in submission } }

    // design/algorithm-sources.md#programkernel_call
    public static func cpu(_ function: @escaping (MeshOperands, MeshOperands) -> Void) -> Self {
        Self(.cpu(function))
    }

    // design/algorithm-sources.md#programkernel_call
    public static func metal(_ device: MTLDevice, _ function: @escaping (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void) -> Self {
        Self(.metal(device, function))
    }

    // design/algorithm-sources.md#programkernel_call
    public static func prediction(_ model: MLModel, _ function: @escaping (MeshOperands, MeshOperands) -> (MLFeatureProvider, MLPredictionOptions)) -> Self {
        Self(.prediction(model, function))
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
    fileprivate let storage: MeshSection?
    fileprivate let shared: Bool
}

private final class MeshMemory {
    let context: UnsafeMutablePointer<mesh_ctx>
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
        mesh_detach(context)
        context.deinitialize(count: 1); context.deallocate()
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

private final class MeshSection {
    let memory: MeshMemory
    let section: mesh_section
    let receiveQueue: UInt32?

    // design/algorithm-sources.md#programtensor
    init(_ memory: MeshMemory, bytes: Int, count: Int, shared: Bool = false, receiveQueue: UInt32? = nil) throws {
        var section = mesh_section()
        let error = mesh_section_create(memory.context, bytes, UInt32(count), receiveQueue == nil ? 0 : 1, &section)
        if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
        if shared { section.stride = 0 }
        self.memory = memory; self.section = section
        self.receiveQueue = receiveQueue
    }

    // design/algorithm-sources.md#programtensor
    deinit { mesh_section_release(memory.context, section) }
}

private final class MeshInvocation {
    let inputCount: Int, outputCount: Int
    let memory: MeshMemory
    let submit: (OpaquePointer, UInt32, MeshOperands, MeshOperands) -> Void

    // design/algorithm-sources.md#programkernel_call
    init(_ function: MeshSubmission, memory: MeshMemory, inputs: Int, outputs: Int, count: Int) {
        self.memory = memory; inputCount = inputs; outputCount = outputs
        switch function {
        case .cpu(let function):
            submit = { call, _, inputs, outputs in
                function(inputs, outputs)
                mesh_call_complete(call)
            }
        case .metal(let device, let function):
            let queue = device.makeCommandQueue(maxCommandBufferCount: count)!
            let commands = (0..<count).map { _ in queue.makeCommandBuffer()! }
            submit = { call, index, inputs, outputs in
                let command = commands[Int(index)]
                function(command, inputs, outputs)
                command.addCompletedHandler { command in
                    if command.status == .completed { mesh_call_complete(call) }
                    else { mesh_call_fail(call, Int32((command.error as NSError?)?.code ?? -1)) }
                }
                command.commit()
            }
        case .prediction(let model, let function):
            submit = { call, _, inputs, outputs in
                let (features, options) = function(inputs, outputs)
                model.__prediction(fromFeatures: features, options: options) { _, error in
                    if let error { mesh_call_fail(call, Int32((error as NSError).code)) }
                    else { mesh_call_complete(call) }
                }
            }
        }
    }

}

public final class Mesh {
    public let rank: Int, size: Int, count: Int
    private let memory: MeshMemory
    private let calls: OpaquePointer
    private var transfer: UInt32 = 0
    private var preparations: [() throws -> Void] = []

    // design/algorithm-sources.md#program
    public init(region: String, rank: Int, size: Int, workers: Int, count: Int = 1) throws {
        precondition(size > 0 && (0..<size).contains(rank) && count > 0)
        let memory = try MeshMemory(region)
        let owner = Unmanaged.passRetained(memory).toOpaque()
        guard let calls = mesh_calls_create(memory.context, UInt32(workers), UInt32(count), owner,
            { owner in Unmanaged<MeshMemory>.fromOpaque(owner!).release() }) else {
            Unmanaged<MeshMemory>.fromOpaque(owner).release()
            throw POSIXError(POSIXErrorCode(rawValue: errno)!)
        }
        self.memory = memory; self.calls = calls; self.rank = rank; self.size = size; self.count = count
    }

    // design/algorithm-sources.md#programtensor
    deinit { mesh_calls_destroy(calls) }

    // design/algorithm-sources.md#programtensor
    public func tensor(on owner: Int, sections: [Int]) throws -> [TensorPart] {
        precondition((0..<size).contains(owner))
        return try sections.map { bytes in
            TensorPart(rank: owner, bytes: bytes,
                       storage: owner == rank ? try MeshSection(memory, bytes: bytes, count: count) : nil, shared: false)
        }
    }

    // design/algorithm-sources.md#programkernel_call
    public func call(_ function: TensorFunction, inputs: [TensorPart], outputs: [TensorPart],
                     on owner: Int, worker: Int) throws {
        precondition(inputs.allSatisfy { $0.rank == owner } && outputs.allSatisfy { $0.rank == owner && !$0.shared })
        if owner != rank { return }
        preparations.append { [unowned self] in
            precondition(outputs.allSatisfy { $0.storage!.receiveQueue == nil })
            let invocation = MeshInvocation(try function.prepare(self, inputs, outputs), memory: memory,
                                            inputs: inputs.count, outputs: outputs.count, count: count)
            try bind(invocation, inputs: inputs, outputs: outputs, worker: worker)
        }
    }

    // design/algorithm-sources.md#programkernel_call
    private func bind(_ invocation: MeshInvocation, inputs: [TensorPart], outputs: [TensorPart], worker: Int) throws {
        let argument = Unmanaged.passRetained(invocation).toOpaque()
        let inputRows = inputs.map { $0.storage!.section }, outputRows = outputs.map { $0.storage!.section }
        let call = mesh_call_bind(calls, UInt32(worker), inputRows, inputRows.count, outputRows, outputRows.count,
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
        let source = part.storage!, context = memory.context
        let pages: [UInt32], index: (mesh_operand) -> Int
        if let queue = source.receiveQueue {
            var received = [UInt32](repeating: 0, count: mesh_receive_pages(context, queue, source.section.pages, nil))
            mesh_receive_pages(context, queue, source.section.pages, &received)
            pages = received
            let quantum = Int(mesh_block_pages(context))
            var slots = [Int](repeating: 0, count: Int(mesh_arena_pages(context)) / quantum)
            for (slot, page) in pages.enumerated() { slots[Int(page) / quantum] = slot }
            index = { slots[Int($0.page) / quantum] }
        } else {
            pages = (0..<source.section.count).map { mesh_section_page(context, source.section, $0) }
            let stride = part.shared ? 0 : 1
            index = { Int($0.index) * stride }
        }
        let values = try pages.map { page in
            try make(MeshSpan(data: UnsafeMutableRawBufferPointer(start: mesh_page_address(context, page),
                                                                 count: source.section.bytes), memory: memory))
        }
        return MeshBindings(values: values, index: index)
    }

    // design/algorithm-sources.md#programtensor
    public func constant(on owner: Int, bytes: Int, initialize: (MeshSpan) throws -> Void) throws -> TensorPart {
        let part = TensorPart(rank: owner, bytes: bytes,
                              storage: owner == rank ? try MeshSection(memory, bytes: bytes, count: 1, shared: true) : nil, shared: true)
        if let storage = part.storage {
            let span = MeshSpan(data: UnsafeMutableRawBufferPointer(start: mesh_section_address(memory.context, storage.section, 0),
                                                                   count: bytes), memory: memory)
            try initialize(span)
            mesh_section_constant(memory.context, storage.section)
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
        precondition((0..<size).contains(destination))
        let identity = transfer; transfer += 1
        let participating = rank == part.rank || rank == destination
        let channel = participating ? mesh_peer_channel(memory.context, UInt32(rank == destination ? part.rank : destination), UInt32(queue)) : 0
        if channel == MESH_ABSENT { throw POSIXError(.ENETUNREACH) }
        let output = TensorPart(rank: destination, bytes: part.bytes,
                                storage: destination == rank ? try MeshSection(memory, bytes: part.bytes, count: part.shared ? 1 : count,
                                                                               shared: part.shared, receiveQueue: channel) : nil, shared: part.shared)
        if participating {
            let local = rank == destination ? output : part
            let error = mesh_transfer_bind(memory.context, channel, rank == destination ? 1 : 0,
                                           identity, local.storage!.section)
            if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
        }
        return output
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
        var level = parts
        while level.count > 1 {
            var next: [TensorPart] = []
            for i in stride(from: 0, to: level.count, by: 2) {
                if i + 1 == level.count { next.append(level[i]); continue }
                let owner = level.count == 2 ? destination : level[i].rank
                let pair = try gather([level[i], level[i + 1]], to: owner, queue: queue)
                let output = try tensor(on: owner, sections: [pair[0].bytes])
                try call(combine, inputs: pair, outputs: output, on: owner, worker: worker)
                next.append(output[0])
            }
            level = next
        }
        return try send(level[0], to: destination, queue: queue)
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
        mesh_transfers_start(memory.context)
        let error = mesh_calls_start(calls)
        if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
    }

    // design/algorithm-sources.md#program
    public func submit(_ index: Int) {
        mesh_calls_submit(calls, UInt32(index))
    }

    // design/algorithm-sources.md#collectivesync_on_remote_fill
    public func syncOnRemoteFill(_ parts: [TensorPart], index: Int = 0) {
        let sections = parts.map { $0.storage!.section }
        mesh_sync_on_remote_fill(memory.context, sections, sections.count, UInt32(index))
    }
}
