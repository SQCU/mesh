import CMesh
import CoreML
import Foundation
import Metal

public typealias MeshOperands = UnsafeBufferPointer<mesh_operand>

extension mesh_operand {
    // design/algorithm-sources.md#programtensor
    @inlinable public var invocation: UInt32 { sequence.pointee }
    // design/algorithm-sources.md#programtensor
    @inlinable public func load<Value>(at index: Int, as type: Value.Type = Value.self) -> Value {
        data!.load(fromByteOffset: index * MemoryLayout<Value>.stride, as: type)
    }
}

fileprivate enum MeshSubmission {
    case cpu((MeshOperands, MeshOperands) -> Void)
    case metal(MTLDevice, (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void, [MTLBuffer])
    case prediction(MLModel, [MeshFeatures], [MLPredictionOptions])
    case resident(MTLDevice, [[MeshMetalOperand]], [[MeshMetalOperand]], [mesh_section], (MeshMetalFrame) throws -> Void)
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
    private init(_ prepare: @escaping (Mesh, [TensorPart], [TensorPart]) throws -> MeshSubmission) {
        self.prepare = prepare
    }

    // design/algorithm-sources.md#programkernel_call
    public static func cpu(_ function: @escaping (MeshOperands, MeshOperands) -> Void) -> Self {
        Self { _, _, _ in .cpu(function) }
    }

    // design/algorithm-sources.md#programkernel_call
    public static func metal(_ device: MTLDevice, _ function: @escaping (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void) -> Self {
        Self { _, _, _ in .metal(device, function, []) }
    }

    // design/algorithm-sources.md#device-operands
    public static func metal(_ device: MTLDevice,
        operands: @escaping ([ContiguousArray<MeshMetalOperand>], [ContiguousArray<MeshMetalOperand>]) throws
            -> (MTLCommandBuffer, MeshOperands, MeshOperands) -> Void) -> Self {
        Self { mesh, inputs, outputs in
            let x = inputs.map { mesh.metal($0, device: device) }, y = outputs.map { mesh.metal($0, device: device) }
            let resources = (x + y).flatMap { $0.flatMap { $0.resources } }
            return .metal(device, try operands(x, y), resources)
        }
    }

    // design/algorithm-sources.md#resident-metal
    public static func resident(_ device: MTLDevice,
        _ body: @escaping (MeshMetalFrame) throws -> Void) -> Self {
        Self { mesh, inputs, outputs in
            .resident(device, inputs.map { Array(mesh.metal($0, device: device)) },
                      outputs.map { Array(mesh.metal($0, device: device)) },
                      outputs.map { $0.section! }, body)
        }
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
            let features = (0..<mesh.inFlight).map { _ in MeshFeatures(names: names, bindings: bindings) }
            let arrays = try outputs.indices.map { try mesh.bindings(results[$0], using: outputs[$0].1) }
            let options = (0..<mesh.inFlight).map { index in
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
    public let values: ContiguousArray<Value>

    // design/algorithm-sources.md#programtensor
    @inlinable public func index(_ operand: mesh_operand) -> Int { Int(operand.index) }

    // design/algorithm-sources.md#programtensor
    @inlinable public subscript(_ operand: mesh_operand) -> Value { values[index(operand)] }
}

public struct MeshMetalOperand {
    public let table: MTLBuffer
    public let offset: Int
    fileprivate let resources: ContiguousArray<MTLBuffer>
    public let bytes: Int
    fileprivate let quantum: Int
    public let data: MTLBuffer?
    public let availability: UInt64

    // design/algorithm-sources.md#device-operands
    // design/algorithm-sources.md#native-metal-program
    public func pipeline(library: MTLLibrary, function: String,
                         constants values: MTLFunctionConstantValues = MTLFunctionConstantValues(), indirect: Bool = false) throws -> MTLComputePipelineState {
        var quantum = UInt64(quantum)
        values.setConstantValue(&quantum, type: .ulong, index: 37)
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = try library.makeFunction(name: function, constantValues: values)
        descriptor.supportIndirectCommandBuffers = indirect
        return try library.device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
    }

    // design/algorithm-sources.md#device-operands
    public static let source = #"""
    #include <metal_stdlib>
    #pragma METAL internals : enable
    constant ulong mesh_quantum [[function_constant(37)]];
    namespace mesh {
    struct page { ulong mapping, host, address, stamp; };
    // design/prepared-machine.md#M10
    static_assert(sizeof(page) == 32, "mesh_page_entry");
    template<typename T> struct span { volatile coherent(system) device T* data; uint count; };
    // design/algorithm-sources.md#device-operands
    template<typename T> inline span<T> contiguous(volatile coherent(system) device const page* pages, ulong offset, uint count) {
        ulong byte = offset * sizeof(T), at = byte % mesh_quantum;
        return {reinterpret_cast<volatile coherent(system) device T*>(pages[byte / mesh_quantum].address + at),
                metal::min(count, uint((mesh_quantum - at) / sizeof(T)))};
    }
    }
    """#
}

public struct MeshSpan {
    public let data: UnsafeMutableRawBufferPointer
    fileprivate let memory: MeshMemory

    // design/algorithm-sources.md#programtensor
    public func metal(device: MTLDevice) -> MTLBuffer { memory.metal(device, data: data) }

    // design/algorithm-sources.md#device-operands
    public var metalOffset: Int { Int(bitPattern: data.baseAddress!) % Int(memory.context.pointee.M.pointee.pgsz) }

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
    fileprivate let section: mesh_section?
    fileprivate let shared: Bool
    fileprivate var identity: Int
}

// design/algorithm-sources.md#mesherror
public enum MeshError: Error {
    case busy
    case link(peer: Int, code: Int32)
    case function(call: Int, code: Int32)
}

private struct MeshDelivery: Hashable {
    let value, source, destination, queue: Int
}

private final class MeshMemory {
    let context: UnsafeMutablePointer<mesh_ctx>
    var functions: [MeshLaunch] = []
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
        let offset = Int(bitPattern: data.baseAddress!) % pageSize
        let address = data.baseAddress!.advanced(by: -offset)
        let bytes = (offset + data.count + pageSize - 1) / pageSize * pageSize
        let key = "\(device.registryID):\(UInt(bitPattern: address)):\(bytes)" as NSString
        if let buffer = buffers.object(forKey: key) as? MTLBuffer { return buffer }
        let buffer = device.makeBuffer(bytesNoCopy: address, length: bytes, options: .storageModeShared,
                                       deallocator: { [self] _, _ in _ = self })!
        buffers.setObject(buffer, forKey: key)
        return buffer
    }
}

private typealias MeshBody = (OpaquePointer) -> Void
private struct MeshLaunch {
    let function: MeshBody?
    let resources: [MTLBuffer]
    var dependencies: Int? = nil
    var prepare: ((OpaquePointer) throws -> Void)? = nil
}

// design/prepared-machine.md#M02
// design/algorithm-sources.md#programkernel_call
private func meshInvocation(_ function: MeshSubmission, memory: MeshMemory, inputs: Int, outputs: Int, count: Int) -> MeshLaunch {
    let launch: MeshBody
    let retained: [MTLBuffer]
    switch function {
    case .resident(let device, let x, let y, let results, let body):
        return meshResident(device, memory: memory, inputs: x, outputs: y, results: results,
                            body: body, count: count)
    case .cpu(let function):
        retained = []
        launch = { call in
            let operands = mesh_call_operands(call)
            let inputs = MeshOperands(start: operands, count: inputs), outputs = MeshOperands(start: operands.advanced(by: inputs.count), count: outputs)
            function(inputs, outputs)
            mesh_call_complete(call, 0)
        }
    case .metal(let device, let function, let resources):
        retained = resources
        let queue = device.makeCommandQueue(maxCommandBufferCount: count)!
        if !resources.isEmpty {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.initialCapacity = resources.count
            let residency = try! device.makeResidencySet(descriptor: descriptor)
            residency.addAllocations(resources)
            residency.commit()
            queue.addResidencySet(residency)
        }
        launch = { call in
            let operands = mesh_call_operands(call)
            let inputs = MeshOperands(start: operands, count: inputs), outputs = MeshOperands(start: operands.advanced(by: inputs.count), count: outputs)
            let command = queue.makeCommandBuffer()!
            function(command, inputs, outputs)
            command.addCompletedHandler { [memory] command in
                withExtendedLifetime(memory) {
                    if command.status == .completed { mesh_call_complete(call, 0) }
                    else { mesh_call_fail(call, Int32((command.error as NSError?)?.code ?? -1)) }
                }
            }
            command.commit()
        }
    case .prediction(let model, let features, let options):
        retained = []
        launch = { call in
            let inputs = MeshOperands(start: mesh_call_operands(call), count: inputs)
            let index = Int(mesh_call_index(call)), provider = features[index]
            provider.operands = inputs
            model.__prediction(fromFeatures: provider, options: options[index]) { [memory] _, error in
                withExtendedLifetime(memory) {
                    if let error { mesh_call_fail(call, Int32((error as NSError).code)) }
                    else { mesh_call_complete(call, 0) }
                }
            }
        }
    }
    return MeshLaunch(function: launch, resources: retained)
}


// design/prepared-machine.md#M41
// design/prepared-machine.md#M42
public struct MeshInvocation {
    @usableFromInline let owner: Mesh
    @usableFromInline let publication: OpaquePointer
    @usableFromInline let instance: OpaquePointer

    // design/algorithm-sources.md#program
    // design/prepared-machine.md#M41
    fileprivate init(_ owner: Mesh, _ publication: OpaquePointer, _ instance: OpaquePointer) {
        self.owner = owner; self.publication = publication; self.instance = instance
    }

    // design/algorithm-sources.md#program
    // design/prepared-machine.md#M41
    @inlinable @discardableResult
    public func submit() -> UInt32 { mesh_submit(publication) }

    // design/algorithm-sources.md#meshresult
    // design/prepared-machine.md#M42
    @inlinable
    public func result(_ generation: UInt32) -> Result<Void, MeshError> {
        let status = mesh_result(instance, generation)
        if status == 0 { return .success(()) }
        if status >> 62 == 3 { return .failure(.busy) }
        return owner.outcome(status)
    }
}

public final class Mesh {
    public let rank: Int, size: Int, inFlight: Int
    private let memory: MeshMemory
    private let calls: OpaquePointer
    private let peers: [Int]
    private var transfer: UInt32 = 0
    private var value = 0
    private var deliveries: [MeshDelivery: TensorPart] = [:]
    private var preparations: [() throws -> Void] = []
    private var finalizations: [() throws -> Void] = []
    private var routes: [Placement.Edge: [Int]]

    // design/algorithm-sources.md#program
    public init(region: String, rank: Int, size: Int, workers: Int, inFlight: Int = 1, placement: Placement = Placement()) throws {
        precondition(size > 0 && (0..<size).contains(rank) && inFlight > 0)
        let memory = try MeshMemory(region)
        let owner = Unmanaged.passRetained(memory).toOpaque()
        guard let calls = mesh_calls_create(memory.context, UInt32(workers), UInt32(inFlight), owner,
            { owner in
                let memory = Unmanaged<MeshMemory>.fromOpaque(owner!).takeRetainedValue()
                memory.functions.removeAll()
            }) else {
            Unmanaged<MeshMemory>.fromOpaque(owner).release()
            throw POSIXError(POSIXErrorCode(rawValue: errno)!)
        }
        self.memory = memory; self.calls = calls; self.rank = rank; self.size = size; self.inFlight = inFlight
        routes = placement.routes
        peers = (0..<Int(memory.context.pointee.M.pointee.links)).map { Int(mesh_links(memory.context.pointee.M)[$0].peer) }
    }

    // design/algorithm-sources.md#programtensor
    deinit { mesh_calls_destroy(calls) }

    // design/algorithm-sources.md#programtensor
    private func part(on owner: Int, bytes: Int, shared: Bool = false, queue: UInt32? = nil) throws -> TensorPart {
        let identity = value; value += 1
        var section: mesh_section?
        if owner == rank {
            var local = mesh_section()
            let error = mesh_section_create(memory.context, bytes, UInt32(shared ? 1 : inFlight), queue ?? MESH_ABSENT, &local)
            if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
            if shared { local.stride = 0; memory.context.pointee.shared_pages += local.pages }
            section = local
        }
        return TensorPart(rank: owner, bytes: bytes, section: section, shared: shared, identity: identity)
    }

    // design/algorithm-sources.md#programtensor
    public func tensor(on owner: Int, sections: [Int]) throws -> [TensorPart] {
        precondition((0..<size).contains(owner))
        return try sections.map { try part(on: owner, bytes: $0) }
    }

    // design/algorithm-sources.md#programkernel_call
    public func call(_ function: TensorFunction, inputs: [TensorPart], outputs: [TensorPart],
                     on owner: Int, worker: Int) throws {
        precondition(inputs.allSatisfy { $0.rank == owner } && outputs.allSatisfy { $0.rank == owner && !$0.shared })
        if owner != rank { return }
        preparations.append { [unowned self] in
            precondition(outputs.allSatisfy { $0.section!.channel == MESH_ABSENT })
            let invocation = meshInvocation(try function.prepare(self, inputs, outputs), memory: memory,
                                            inputs: inputs.count, outputs: outputs.count, count: inFlight)
            let inputRows = inputs.map { $0.section! }, outputRows = outputs.map { $0.section! }
            memory.functions.append(invocation)
            let native = withUnsafeBytes(of: invocation) {
                $0.load(as: (UnsafeRawPointer?, UnsafeMutableRawPointer?).self)
            }
            guard let binding = mesh_call_bind(calls, UInt32(worker), inputRows, inputRows.count,
                              invocation.dependencies ?? inputRows.count, outputRows, outputRows.count,
                              native.0, native.1) else {
                memory.functions.removeLast()
                throw POSIXError(POSIXErrorCode(rawValue: errno)!)
            }
            if let prepare = invocation.prepare { finalizations.append { try prepare(binding) } }
        }
    }

    // design/algorithm-sources.md#programtensor
    public func bindings<Value>(_ part: TensorPart, using make: (MeshSpan) throws -> Value) throws -> MeshBindings<Value> {
        let source = part.section!, context = memory.context
        let pages = (0..<source.count).map { mesh_row_page(context, source.first + $0 * source.stride, 0) }
        let values = try pages.map { page in
            try make(MeshSpan(data: UnsafeMutableRawBufferPointer(start: mesh_page_address(context, page),
                                                                 count: source.bytes), memory: memory))
        }
        return MeshBindings(values: ContiguousArray(values))
    }

    // design/algorithm-sources.md#device-operands
    fileprivate func metal(_ part: TensorPart, device: MTLDevice) -> ContiguousArray<MeshMetalOperand> {
        let section = part.section!, context = memory.context, block = mesh_block_pages(context)
        let quantum = Int(block * context.pointee.M.pointee.pgsz)
        let count = Int(section.pages / block)
        return ContiguousArray((0..<section.count).map { instance in
            let row = section.first + instance * section.stride
            var backing: ContiguousArray<MTLBuffer> = []
            do {
                for chunk in 0..<count {
                    let page = mesh_row_page(context, row, UInt32(chunk))
                    let buffer = memory.metal(device, data: UnsafeMutableRawBufferPointer(start: mesh_page_address(context, page), count: quantum))
                    mesh_device_bind(context, row + UInt32(chunk), buffer.gpuAddress)
                    backing.append(buffer)
                }
            }
            let entries = UnsafeMutableRawBufferPointer(start: mesh_page(context.pointee.M)!.advanced(by: Int(row)),
                                                     count: count * MemoryLayout<mesh_page_entry>.stride)
            let span = MeshSpan(data: entries, memory: memory)
            let data = section.bytes <= device.maxBufferLength
                ? memory.metal(device, data: UnsafeMutableRawBufferPointer(start: mesh_section_address(context, section, instance), count: section.bytes)) : nil
            if let data { backing.append(data) }
            let table = span.metal(device: device)
            let available = table.gpuAddress + UInt64(span.metalOffset + (section.channel == MESH_ABSENT ? 0 : count - 1) * 32 + 24)
            return MeshMetalOperand(table: table, offset: span.metalOffset,
                                    resources: backing, bytes: section.bytes, quantum: quantum, data: data, availability: available)
        })
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
            if let existing = deliveries[delivery] { source = existing; continue }
            let identity = transfer; transfer += 1
            let participating = rank == source.rank || rank == next
            let channel = participating ? mesh_peer_channel(memory.context, UInt32(rank == next ? source.rank : next), UInt32(queue)) : 0
            if channel == MESH_ABSENT { throw POSIXError(.ENETUNREACH) }
            var output = try self.part(on: next, bytes: source.bytes, shared: source.shared, queue: channel)
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
        var level = parts
        while level.count > 1 {
            var next: [TensorPart] = []
            for i in stride(from: 0, to: level.count, by: 2) {
                if i + 1 == level.count { next.append(level[i]); continue }
                let owner = level.count == 2 ? destination : level[i].rank
                let pair = try gather([level[i], level[i + 1]], to: owner, queue: queue)
                let output = [try part(on: owner, bytes: pair[0].bytes)]
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
        deliveries.removeAll()
        routes.removeAll()
        // design/prepared-machine.md#M18
        let error = mesh_calls_prepare(calls)
        if error != 0 { throw POSIXError(POSIXErrorCode(rawValue: error)!) }
        for finalize in finalizations { try finalize() }
        finalizations.removeAll()
        let workerError = mesh_calls_start(calls)
        if workerError != 0 { throw POSIXError(POSIXErrorCode(rawValue: workerError)!) }
        let transportError = mesh_transfers_start(memory.context)
        if transportError != 0 { throw POSIXError(POSIXErrorCode(rawValue: transportError) ?? .EIO) }
    }

    // design/algorithm-sources.md#program
    // design/prepared-machine.md#M41
    public func invocation(_ slot: Int) -> MeshInvocation {
        var instance: OpaquePointer?
        let publication = mesh_submission_at(calls, UInt32(slot), &instance)!
        return MeshInvocation(self, publication, instance!)
    }

    // design/algorithm-sources.md#meshresult
    @usableFromInline func outcome(_ status: UInt64) -> Result<Void, MeshError> {
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
        mesh_sync_on_remote_fill(memory.context, sections, sections.count, UInt32(truncatingIfNeeded: index))
    }
}

public final class MeshMetalFrame {
    public let index: Int
    public let inputs, outputs: [MeshMetalOperand]
    public let completion: (buffer: MTLBuffer, offset: Int)
    public let publications: [MTLBuffer]
    public let resources: [MTLBuffer]

    // design/prepared-machine.md#M37
    fileprivate init(memory: MeshMemory, device: MTLDevice, function: OpaquePointer, index: Int,
        inputs: [MeshMetalOperand], outputs: [MeshMetalOperand], results: [mesh_section]) {
        self.index = index; self.inputs = inputs; self.outputs = outputs
        let context = memory.context, m = context.pointee.M!
        // design/prepared-machine.md#M37
        let pointer = mesh_function_completion(function, UInt32(index))
        let state = MeshSpan(data: UnsafeMutableRawBufferPointer(start: pointer, count: 8), memory: memory)
        completion = (state.metal(device: device), state.metalOffset)
        var backing = (inputs + outputs).flatMap { $0.resources } + (inputs + outputs).map { $0.table }
        // design/prepared-machine.md#M13
        publications = results.map { result in
            let row = result.first + UInt32(index) * result.stride
            let count = Int(mesh_publication_prepare(m, row, 1, nil))
            var stores = [prepared_publication](repeating: prepared_publication(), count: count)
            _ = stores.withUnsafeMutableBufferPointer { mesh_publication_prepare(m, row, 1, $0.baseAddress) }
            for i in stores.indices {
                let pointer = UnsafeMutableRawPointer(bitPattern: UInt(stores[i].destination))!
                let span = MeshSpan(data: UnsafeMutableRawBufferPointer(start: pointer, count: 8), memory: memory)
                let buffer = span.metal(device: device)
                backing.append(buffer)
                stores[i].destination = buffer.gpuAddress + UInt64(span.metalOffset)
            }
            let records = stores.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
            }
            backing.append(records)
            return records
        }
        resources = backing + [completion.buffer]
    }

    // design/algorithm-sources.md#resident-metal
    public static let source = #"""
    #ifndef __METAL_MEMORY_SCOPE_SYSTEM__
    #define __METAL_MEMORY_SCOPE_SYSTEM__ 3
    #endif
    #include <metal_atomic>
    namespace mesh {
    // design/algorithm-sources.md#resident-metal
    inline void fence() {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst,
            static_cast<metal::thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__));
    }
    // design/algorithm-sources.md#resident-metal
    inline void cohere(volatile coherent(system) device uint* data, uint count, uint index, uint stride) {
        for (uint i = index; i < count; i += stride) data[i] = data[i];
        fence();
    }
    }
    """#
}

// design/algorithm-sources.md#resident-metal
// design/prepared-machine.md#M37
private func meshResident(_ device: MTLDevice, memory: MeshMemory, inputs: [[MeshMetalOperand]], outputs: [[MeshMetalOperand]],
    results: [mesh_section], body: @escaping (MeshMetalFrame) throws -> Void, count: Int) -> MeshLaunch {
    return MeshLaunch(function: nil, resources: [], dependencies: 0, prepare: { function in
        for index in 0..<count {
            try body(MeshMetalFrame(memory: memory, device: device, function: function, index: index,
                inputs: inputs.map { $0[min(index, $0.count - 1)] }, outputs: outputs.map { $0[index] },
                results: results))
        }
    })
}
