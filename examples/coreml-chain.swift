import Accelerate
import CoreML
import Foundation
import Mesh

private struct Operand: Decodable {
    let owner: Int
    let shape: [Int]
    // design/algorithm-sources.md#programtensor
    var elements: Int { shape.reduce(1, *) }
    // design/algorithm-sources.md#programtensor
    var strides: [Int] { shape.indices.map { shape.dropFirst($0 + 1).reduce(1, *) } }
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
    let count: Int
    let inFlight: Int
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
        fields.map { name, operand in
            let strides = operand.strides
            return (name, { try $0.multiArray(shape: operand.shape, strides: strides, type: .float32) })
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
        var parts = try layout.map { try mesh.tensor(on: $0.owner, sections: [$0.elements * 4])[0] }
        var history = [Array(zip(parts, layout))]
        for i in parts.indices {
            let elements = layout[i].elements
            try mesh.call(.cpu { _, outputs in
                var start = Float(Int(outputs[0].index) + i), step: Float = 1
                vDSP_vramp(&start, &step, outputs[0].data!.assumingMemoryBound(to: Float.self), 1, vDSP_Length(elements))
            }, inputs: [], outputs: [parts[i]], on: layout[i].owner, worker: i % plan.workers)
        }
        let sum = TensorFunction.cpu { inputs, outputs in
            vDSP_vadd(inputs[0].data!.assumingMemoryBound(to: Float.self), 1,
                      inputs[1].data!.assumingMemoryBound(to: Float.self), 1,
                      outputs[0].data!.assumingMemoryBound(to: Float.self), 1, vDSP_Length(outputs[0].bytes / 4))
        }
        for stage in plan.stages {
            precondition(stage.functions.count == parts.count && stage.functions.allSatisfy { $0.count == stage.outputs.count })
            var contributions: [[TensorPart]] = []
            for i in parts.indices {
                let output = try mesh.tensor(on: layout[i].owner, sections: stage.outputs.map { $0.elements * 4 })
                if layout[i].owner == rank {
                    for j in output.indices {
                        let function = try prediction(stage.functions[i][j], models: &models,
                                                      inputs: [(plan.inputName, layout[i])], outputs: [(plan.outputName, stage.outputs[j])])
                        try mesh.call(function, inputs: [parts[i]], outputs: [output[j]], on: rank, worker: j % plan.workers)
                    }
                }
                contributions.append(output)
            }
            parts = try mesh.reduceScatter(contributions, to: stage.outputs.map(\.owner), using: sum,
                                           workers: stage.outputs.indices.map { $0 % plan.workers })
            let reduced = Array(zip(parts, stage.outputs))
            history.append(reduced)
            history.append(stage.finish.isEmpty ? reduced : [])
            for call in stage.finish {
                let selected = call.inputs.map { history[$0.group][$0.part] }
                let inputs = try mesh.gather(selected.map { $0.0 }, to: call.owner)
                let outputs = try mesh.tensor(on: call.owner, sections: call.outputs.map { $0.operand.elements * 4 })
                if call.owner == rank {
                    let function = try prediction(call.model, models: &models,
                                                  inputs: zip(call.inputs, selected).map { ($0.0.name, $0.1.1) },
                                                  outputs: call.outputs.map { ($0.name, $0.operand) })
                    try mesh.call(function, inputs: inputs, outputs: outputs, on: call.owner, worker: call.worker)
                }
                let placed = try mesh.scatter(outputs, to: call.outputs.map { $0.operand.owner })
                history[history.count - 1].append(contentsOf: zip(placed, call.outputs.map(\.operand)))
            }
            parts = history.last!.map { $0.0 }; layout = history.last!.map { $0.1 }
        }
        history.removeAll()
        for i in parts.indices {
            try mesh.call(.cpu { inputs, _ in
                let input = inputs[0], data = input.data!.assumingMemoryBound(to: Float.self)
                let first = data[0], last = data[Int(input.bytes) / 4 - 1], index = input.index
                DispatchQueue.main.async {
                    print("rank=\(rank) part=\(i) index=\(index) first=\(first) last=\(last)")
                    fflush(stdout)
                }
            }, inputs: [parts[i]], outputs: [], on: layout[i].owner, worker: i % plan.workers)
        }
        try mesh.start()
        for index in 0..<plan.count {
            if case .failure(.busy) = mesh.submit(index) {
                FileHandle.standardError.write(Data("mesh: submit(\(index)) busy\n".utf8))
                break
            }
        }
        withExtendedLifetime((mesh, parts)) { dispatchMain() }
    }
}
