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
    let finish: [String?]
}

private struct Chain: Decodable {
    let count: Int
    let workers: Int
    let inputName: String
    let outputName: String
    let inputs: [Operand]
    let stages: [Stage]
}

// design/algorithm-sources.md#programkernel_call
private func prediction(_ path: String, models: inout [String: MLModel], input: Operand, output: Operand,
                        inputName: String, outputName: String) throws -> TensorFunction {
    let model: MLModel
    if let existing = models[path] { model = existing }
    else { model = try MLModel(contentsOf: URL(fileURLWithPath: path)); models[path] = model }
    let inputStrides = input.strides, outputStrides = output.strides
    return TensorFunction(inputViews: [{ span -> MLFeatureProvider in
        let value = try span.multiArray(shape: input.shape, strides: inputStrides, type: .float32)
        return try MLDictionaryFeatureProvider(dictionary: [inputName: value])
    }], outputViews: [{ span -> MLPredictionOptions in
        let value = try span.multiArray(shape: output.shape, strides: outputStrides, type: .float32)
        let options = MLPredictionOptions()
        options.outputBackings = [outputName: value]
        return options
    }]) { inputs, outputs in
        .prediction(model) { x, y in (inputs[0][x[0]], outputs[0][y[0]]) }
    }
}

@main
struct CoreMLChain {
    // design/algorithm-sources.md#program
    static func main() throws {
        let args = CommandLine.arguments
        let rank = Int(args[1])!, size = Int(args[2])!
        let plan = try JSONDecoder().decode(Chain.self, from: Data(contentsOf: URL(fileURLWithPath: args[4])))
        let mesh = try Mesh(region: args[3], rank: rank, size: size, workers: plan.workers, count: plan.count)
        var models: [String: MLModel] = [:]
        var layout = plan.inputs
        var parts = try layout.map { try mesh.tensor(on: $0.owner, sections: [$0.elements * 4])[0] }
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
            precondition(stage.finish.count == stage.outputs.count)
            var contributions: [[TensorPart]] = []
            for i in parts.indices {
                let output = try mesh.tensor(on: layout[i].owner, sections: stage.outputs.map { $0.elements * 4 })
                if layout[i].owner == rank {
                    for j in output.indices {
                        let function = try prediction(stage.functions[i][j], models: &models, input: layout[i], output: stage.outputs[j],
                                                      inputName: plan.inputName, outputName: plan.outputName)
                        try mesh.call(function, inputs: [parts[i]], outputs: [output[j]], on: rank, worker: j % plan.workers)
                    }
                }
                contributions.append(output)
            }
            parts = try mesh.reduceScatter(contributions, to: stage.outputs.map(\.owner), using: sum,
                                           workers: stage.outputs.indices.map { $0 % plan.workers })
            for j in parts.indices {
                if let path = stage.finish[j] {
                    let output = try mesh.tensor(on: stage.outputs[j].owner, sections: [stage.outputs[j].elements * 4])
                    if stage.outputs[j].owner == rank {
                        let function = try prediction(path, models: &models, input: stage.outputs[j], output: stage.outputs[j],
                                                      inputName: plan.inputName, outputName: plan.outputName)
                        try mesh.call(function, inputs: [parts[j]], outputs: output, on: rank, worker: j % plan.workers)
                    }
                    parts[j] = output[0]
                }
            }
            layout = stage.outputs
        }
        try mesh.start()
        for index in 0..<mesh.count { mesh.submit(index) }
        withExtendedLifetime((mesh, parts)) { dispatchMain() }
    }
}
