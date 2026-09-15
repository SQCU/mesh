import CMesh
import CoreML
import Foundation
import Metal

public typealias MeshOperands = UnsafeBufferPointer<mesh_operand>

public enum TensorFunction {
    case cpu((MeshOperands, MeshOperands) -> Void)
    case metal(MTLCommandQueue, (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void)
    case prediction(MLModel, MLFeatureProvider, MLPredictionOptions)
}

public struct MetalOperand {
    public let buffer: MTLBuffer
    public let offset: Int, bytes: Int
}

public struct MeshSpan {
    public let data: UnsafeMutableRawBufferPointer
    fileprivate let memory: MeshMemory

    // design/algorithm-sources.md#programtensor
    public func metal(device: MTLDevice) -> MetalOperand {
        let base = mesh_page_address(memory.context, 0)!
        return MetalOperand(buffer: memory.metal(device), offset: base.distance(to: data.baseAddress!), bytes: data.count)
    }

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
}

private final class MeshMemory {
    let context: UnsafeMutablePointer<mesh_ctx>
    private let buffers = NSMapTable<NSNumber, AnyObject>(keyOptions: .strongMemory, valueOptions: .weakMemory)

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
    func metal(_ device: MTLDevice) -> MTLBuffer {
        let key = NSNumber(value: device.registryID)
        if let buffer = buffers.object(forKey: key) as? MTLBuffer { return buffer }
        let address = mesh_page_address(context, 0)!
        let bytes = Int(mesh_length(context) - mesh_data_offset(context))
        let buffer = device.makeBuffer(bytesNoCopy: address, length: bytes, options: .storageModeShared,
                                       deallocator: { [self] _, _ in _ = self })!
        buffers.setObject(buffer, forKey: key)
        return buffer
    }
}

private final class MeshSection {
    let memory: MeshMemory
    let section: mesh_section
    var receiveQueue: UInt32?

    // design/algorithm-sources.md#programtensor
    init(_ memory: MeshMemory, bytes: Int) throws {
        var section = mesh_section()
        let error = mesh_section_create(memory.context, bytes, &section)
        if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
        self.memory = memory; self.section = section
    }

    // design/algorithm-sources.md#programtensor
    deinit { mesh_section_release(memory.context, section) }
}

private final class MeshInvocation {
    let inputCount: Int, outputCount: Int
    let memory: MeshMemory
    let submit: (OpaquePointer, MeshOperands, MeshOperands) -> Void

    // design/algorithm-sources.md#programkernel_call
    init(_ function: TensorFunction, memory: MeshMemory, inputs: Int, outputs: Int) {
        self.memory = memory; inputCount = inputs; outputCount = outputs
        switch function {
        case .cpu(let function):
            submit = { call, inputs, outputs in
                function(inputs, outputs)
                mesh_call_complete(call)
            }
        case .metal(let queue, let function):
            submit = { call, inputs, outputs in
                let command = queue.makeCommandBuffer()!
                function(command, inputs, outputs)
                command.addCompletedHandler { command in
                    if command.status == .completed { mesh_call_complete(call) }
                    else { mesh_call_fail(call, Int32((command.error as NSError?)?.code ?? -1)) }
                }
                command.commit()
            }
        case .prediction(let model, let features, let options):
            submit = { call, _, _ in
                model.__prediction(fromFeatures: features, options: options) { _, error in
                    if let error { mesh_call_fail(call, Int32((error as NSError).code)) }
                    else { mesh_call_complete(call) }
                }
            }
        }
    }

    // design/algorithm-sources.md#programkernel_call
    init(memory: MeshMemory, inputs: Int, outputs: Int,
         submit: @escaping (OpaquePointer, MeshOperands, MeshOperands) -> Void) {
        self.memory = memory; inputCount = inputs; outputCount = outputs; self.submit = submit
    }
}

public final class Mesh {
    public let rank: Int, size: Int
    private let memory: MeshMemory
    private let calls: OpaquePointer
    private var transfer: UInt32 = 0
    private var preparations: [() throws -> Void] = []

    public var sectionCapacity: Int { mesh_section_capacity(memory.context) }

    // design/algorithm-sources.md#program
    public init(region: String, rank: Int, size: Int, workers: Int) throws {
        precondition((1...2).contains(size) && (0..<size).contains(rank))
        let memory = try MeshMemory(region)
        let owner = Unmanaged.passRetained(memory).toOpaque()
        guard let calls = mesh_calls_create(memory.context, UInt32(workers), owner,
            { owner in Unmanaged<MeshMemory>.fromOpaque(owner!).release() }) else {
            Unmanaged<MeshMemory>.fromOpaque(owner).release()
            throw POSIXError(POSIXErrorCode(rawValue: errno)!)
        }
        self.memory = memory; self.calls = calls; self.rank = rank; self.size = size
    }

    // design/algorithm-sources.md#programtensor
    deinit { mesh_calls_destroy(calls) }

    // design/algorithm-sources.md#programtensor
    public func tensor(on owner: Int, sections: [Int]) throws -> [TensorPart] {
        precondition((0..<size).contains(owner))
        return try sections.map { bytes in
            TensorPart(rank: owner, bytes: bytes,
                       storage: owner == rank ? try MeshSection(memory, bytes: bytes) : nil)
        }
    }

    // design/algorithm-sources.md#programkernel_call
    public func call(_ function: TensorFunction, inputs: [TensorPart], outputs: [TensorPart],
                     on owner: Int, worker: Int) throws {
        precondition(inputs.allSatisfy { $0.rank == owner } && outputs.allSatisfy { $0.rank == owner })
        if owner != rank { return }
        let invocation = MeshInvocation(function, memory: memory, inputs: inputs.count, outputs: outputs.count)
        try bind(invocation, inputs: inputs, outputs: outputs, worker: worker)
    }

    // design/algorithm-sources.md#programkernel_call
    private func bind(_ invocation: MeshInvocation, inputs: [TensorPart], outputs: [TensorPart], worker: Int) throws {
        let argument = Unmanaged.passRetained(invocation).toOpaque()
        let inputRows = inputs.map { $0.storage!.section }, outputRows = outputs.map { $0.storage!.section }
        let call = mesh_call_bind(calls, UInt32(worker), inputRows, inputRows.count, outputRows, outputRows.count,
            { call, argument, inputs, outputs in
                let invocation = Unmanaged<MeshInvocation>.fromOpaque(argument!).takeUnretainedValue()
                invocation.submit(call!, MeshOperands(start: inputs, count: invocation.inputCount),
                                  MeshOperands(start: outputs, count: invocation.outputCount))
            }, argument,
            { argument in Unmanaged<MeshInvocation>.fromOpaque(argument!).release() })
        if call == nil {
            Unmanaged<MeshInvocation>.fromOpaque(argument).release()
            throw POSIXError(POSIXErrorCode(rawValue: errno)!)
        }
    }

    // design/algorithm-sources.md#program
    public func map(_ function: @escaping (MeshSpan, MeshSpan) throws -> TensorFunction,
                    input: TensorPart, output: TensorPart, constants: [TensorPart] = [], worker: Int) {
        precondition(input.rank == output.rank && constants.allSatisfy { $0.rank == input.rank })
        if input.rank != rank { return }
        preparations.append { [unowned self] in
            let source = input.storage!, destination = output.storage!
            precondition(destination.receiveQueue == nil)
            let context = memory.context
            let quantum = Int(mesh_block_pages(context))
            let pageBytes = sectionCapacity / quantum
            var pages: [UInt32]
            if let queue = source.receiveQueue {
                pages = [UInt32](repeating: 0, count: mesh_receive_pages(context, queue, nil))
                mesh_receive_pages(context, queue, &pages)
            } else {
                let offset = mesh_page_address(context, 0)!.distance(to: mesh_section_address(context, source.section)!)
                pages = [UInt32(offset / pageBytes)]
            }
            let target = MeshSpan(data: UnsafeMutableRawBufferPointer(start: mesh_section_address(context, destination.section),
                                                                      count: destination.section.bytes), memory: memory)
            var functions = [MeshInvocation?](repeating: nil, count: Int(mesh_arena_pages(context)) / quantum)
            for page in pages {
                let operand = MeshSpan(data: UnsafeMutableRawBufferPointer(start: mesh_page_address(context, page),
                                                                           count: source.section.bytes), memory: memory)
                functions[Int(page) / quantum] = MeshInvocation(try function(operand, target), memory: memory, inputs: 1 + constants.count, outputs: 1)
            }
            let invocation = MeshInvocation(memory: memory, inputs: 1 + constants.count, outputs: 1) { call, inputs, outputs in
                functions[Int(inputs[0].page) / quantum]!.submit(call, inputs, outputs)
            }
            try bind(invocation, inputs: [input] + constants, outputs: [output], worker: worker)
        }
    }

    // design/algorithm-sources.md#programtensor
    public func constant(on owner: Int, bytes: Int, initialize: (MeshSpan) throws -> Void) throws -> TensorPart {
        let part = try tensor(on: owner, sections: [bytes])[0]
        if let storage = part.storage {
            let span = MeshSpan(data: UnsafeMutableRawBufferPointer(start: mesh_section_address(memory.context, storage.section),
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
        let output = try tensor(on: destination, sections: [part.bytes])[0]
        output.storage?.receiveQueue = UInt32(queue)
        let identity = transfer; transfer += 1
        let local = rank == destination ? output : part
        let error = mesh_transfer_bind(memory.context, UInt32(queue), rank == destination ? 1 : 0,
                                       identity, local.storage!.section)
        if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
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

    // design/algorithm-sources.md#programtensor
    public func metalStorage(device: MTLDevice) -> MTLBuffer {
        memory.metal(device)
    }

    // design/algorithm-sources.md#programkernel_call
    public func start() throws {
        for prepare in preparations { try prepare() }
        preparations.removeAll()
        if size > 1 { mesh_transfers_start(memory.context) }
        let error = mesh_calls_start(calls)
        if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
    }

    // design/algorithm-sources.md#collectivesync_on_remote_fill
    public func syncOnRemoteFill(_ parts: [TensorPart]) {
        let sections = parts.map { $0.storage!.section }
        mesh_sync_on_remote_fill(memory.context, sections, sections.count)
    }
}
