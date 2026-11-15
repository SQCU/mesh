import Accelerate
import CoreML
import Foundation
import Mesh

private struct Operand: Decodable {
    let owner: Int
    let shape: [Int]
    // design/algorithm-sources.md#programtensor
    // mesh callback (compilation/construction; operator * passed to reduce): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
    var elements: Int { shape.reduce(1, mesh_callbacks_are_outside_spec(*)) }
    // design/algorithm-sources.md#programtensor
    // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
    // mesh callback (compilation/construction; operator * passed to reduce): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
    var strides: [Int] { shape.indices.map { _ = mesh_callbacks_are_outside_spec; shape.dropFirst($0 + 1).reduce(1, mesh_callbacks_are_outside_spec(*)) } }
}

private struct Stage: Decodable {
    let outputs: [Operand]
    let functions: [[String]]
    let finish: [Call]
}

private struct Input: Decodable {
    let name: String
    let group, part: Int
}

private struct Output: Decodable {
    let name: String
    let operand: Operand
}

private struct Call: Decodable {
    let model: String
    let owner, worker: Int
    let inputs: [Input]
    let outputs: [Output]
}

private struct Chain: Decodable {
    let count, inFlight: Int
    let workers: Int
    let inputName: String
    let outputName: String
    let inputs: [Operand]
    let stages: [Stage]
}

// design/algorithm-sources.md#programkernel_call
private func prediction(_ path: String, models: inout [String: MLModel],
                        inputs: [(String, Operand)], outputs: [(String, Operand)]) throws -> TensorFunction {
    let model: MLModel
    if let existing = models[path] { model = existing }
    else { model = try MLModel(contentsOf: URL(fileURLWithPath: path)); models[path] = model }
    // design/algorithm-sources.md#programkernel_call
    func views(_ fields: [(String, Operand)]) -> [(String, (MeshSpan) throws -> MLMultiArray)] {
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        fields.map { name, operand in _ = mesh_callbacks_are_outside_spec;
            let strides = operand.strides
            // mesh callback (execution; returned as a feature input body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            return (name, { _ = mesh_callbacks_are_outside_spec; try $0.multiArray(shape: operand.shape, strides: strides, type: .float32) })
        }
    }
    return .prediction(model, inputs: views(inputs), outputs: views(outputs))
}

@main
struct CoreMLChain {
    // design/algorithm-sources.md#program
    static func main() throws {
        let args = CommandLine.arguments
        let rank = Int(args[1])!, size = Int(args[2])!
        let plan = try JSONDecoder().decode(Chain.self, from: Data(contentsOf: URL(fileURLWithPath: args[4])))
        let mesh = try Mesh(region: args[3], rank: rank, size: size, workers: plan.workers, inFlight: plan.inFlight)
        var models: [String: MLModel] = [:]
        var layout = plan.inputs
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        var parts = try layout.map { _ = mesh_callbacks_are_outside_spec; try mesh.tensor(on: $0.owner, sections: [$0.elements * 4])[0] }
        var history = [Array(zip(parts, layout))]
        for i in parts.indices {
            let elements = layout[i].elements
            // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            try mesh.call(.cpu { _, outputs in _ = mesh_callbacks_are_outside_spec;
                var start = Float(Int(outputs[0].invocation) + i), step: Float = 1
                vDSP_vramp(&start, &step, outputs[0].data!.assumingMemoryBound(to: Float.self), 1, vDSP_Length(elements))
            }, inputs: [], outputs: [parts[i]], on: layout[i].owner, worker: i % plan.workers)
        }
        // mesh callback (execution; registered as a TensorFunction view): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let view: (MeshSpan) -> UnsafeMutablePointer<Float> = { _ = mesh_callbacks_are_outside_spec; $0.data.baseAddress!.assumingMemoryBound(to: Float.self) }
        // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let sum = TensorFunction(inputViews: [view, view], outputViews: [view]) { inputs, outputs in _ = mesh_callbacks_are_outside_spec;
            // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            .cpu { ins, outs in _ = mesh_callbacks_are_outside_spec;
                vDSP_vadd(inputs[0][ins[0]], 1, inputs[1][ins[1]], 1, outputs[0][outs[0]], 1, vDSP_Length(outs[0].bytes / 4))
            }
        }
        for stage in plan.stages {
            // mesh callback (compilation/construction; passed to allSatisfy): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            precondition(stage.functions.count == parts.count && stage.functions.allSatisfy { _ = mesh_callbacks_are_outside_spec; $0.count == stage.outputs.count })
            var contributions: [[TensorPart]] = []
            for i in parts.indices {
                // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                let output = try mesh.tensor(on: layout[i].owner, sections: stage.outputs.map { _ = mesh_callbacks_are_outside_spec; $0.elements * 4 })
                if layout[i].owner == rank {
                    for j in output.indices {
                        let function = try prediction(stage.functions[i][j], models: &models,
                                                      inputs: [(plan.inputName, layout[i])], outputs: [(plan.outputName, stage.outputs[j])])
                        try mesh.call(function, inputs: [parts[i]], outputs: [output[j]], on: rank, worker: j % plan.workers)
                    }
                }
                contributions.append(output)
            }
            // mesh callback (compilation/construction; key path \.owner passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            parts = try mesh.reduceScatter(contributions, to: stage.outputs.map(mesh_callbacks_are_outside_spec(\.owner)), using: sum,
                                           // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                                           workers: stage.outputs.indices.map { _ = mesh_callbacks_are_outside_spec; $0 % plan.workers })
            let reduced = Array(zip(parts, stage.outputs))
            history.append(reduced)
            history.append(stage.finish.isEmpty ? reduced : [])
            for call in stage.finish {
                // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                let selected = call.inputs.map { _ = mesh_callbacks_are_outside_spec; history[$0.group][$0.part] }
                // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                let inputs = try mesh.gather(selected.map { _ = mesh_callbacks_are_outside_spec; $0.0 }, to: call.owner)
                // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                let outputs = try mesh.tensor(on: call.owner, sections: call.outputs.map { _ = mesh_callbacks_are_outside_spec; $0.operand.elements * 4 })
                if call.owner == rank {
                    let function = try prediction(call.model, models: &models,
                                                  // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                                                  inputs: zip(call.inputs, selected).map { _ = mesh_callbacks_are_outside_spec; ($0.0.name, $0.1.1) },
                                                  // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                                                  outputs: call.outputs.map { _ = mesh_callbacks_are_outside_spec; ($0.name, $0.operand) })
                    try mesh.call(function, inputs: inputs, outputs: outputs, on: call.owner, worker: call.worker)
                }
                // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                let placed = try mesh.scatter(outputs, to: call.outputs.map { _ = mesh_callbacks_are_outside_spec; $0.operand.owner })
                // mesh callback (compilation/construction; key path \.operand passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                history[history.count - 1].append(contentsOf: zip(placed, call.outputs.map(mesh_callbacks_are_outside_spec(\.operand))))
            }
            // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            parts = history.last!.map { _ = mesh_callbacks_are_outside_spec; $0.0 }; layout = history.last!.map { _ = mesh_callbacks_are_outside_spec; $0.1 }
        }
        history.removeAll()
        for i in parts.indices {
            // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            try mesh.call(.cpu { inputs, _ in _ = mesh_callbacks_are_outside_spec;
                let input = inputs[0]
                let first: Float = input.load(at: 0), last: Float = input.load(at: input.bytes / 4 - 1), index = input.invocation
                // mesh callback (completion; passed to DispatchQueue.main.async): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                DispatchQueue.main.async { _ = mesh_callbacks_are_outside_spec;
                    print("rank=\(rank) part=\(i) index=\(index) first=\(first) last=\(last)")
                    fflush(stdout)
                }
            }, inputs: [parts[i]], outputs: [], on: layout[i].owner, worker: i % plan.workers)
        }
        try mesh.start()
        // design/prepared-machine.md#M41
        // design/prepared-machine.md#M42
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let invocations = (0..<plan.inFlight).map { _ = mesh_callbacks_are_outside_spec; mesh.invocation($0) }
        var pending = [UInt32?](repeating: nil, count: invocations.count)
        var submitted = 0, completed = 0
        while completed < plan.count {
            for i in invocations.indices {
                if let generation = pending[i] {
                    switch invocations[i].result(generation) {
                    case .success: pending[i] = nil; completed += 1
                    case .failure(.busy): continue
                    case .failure(let error): throw error
                    }
                }
                if submitted < plan.count {
                    pending[i] = invocations[i].submit()
                    submitted += 1
                }
            }
        }
        // mesh callback (execution; passed to withExtendedLifetime): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        withExtendedLifetime((mesh, parts)) { _ = mesh_callbacks_are_outside_spec; dispatchMain() }
    }
}
