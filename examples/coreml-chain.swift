import CoreML
import Foundation
import Mesh

@main
struct CoreMLChain {
    // design/algorithm-sources.md#programkernel_call
    static func main() throws {
        let args = CommandLine.arguments
        let rank = Int(args[1])!, size = Int(args[2])!, width = Int(args[5])!
        let mesh = try Mesh(region: args[3], rank: rank, size: size, workers: 3, count: 4)
        let model = try MLModel(contentsOf: URL(fileURLWithPath: args[4]))
        let features: (MeshSpan) throws -> MLFeatureProvider = { input in
            let x = try input.multiArray(shape: [1, width], strides: [width, 1], type: .float32)
            return try MLDictionaryFeatureProvider(dictionary: [args[6]: x])
        }
        let options: (MeshSpan) throws -> MLPredictionOptions = { output in
            let y = try output.multiArray(shape: [1, width], strides: [width, 1], type: .float32)
            let options = MLPredictionOptions()
            options.outputBackings = [args[7]: y]
            return options
        }
        let transform: (MeshBindings<MLFeatureProvider>, MeshBindings<MLPredictionOptions>) -> TensorFunction = { inputs, outputs in
            .prediction(model) { x, y in (inputs[x[0]], outputs[y[0]]) }
        }
        let input = try mesh.tensor(on: 0, sections: [width * 4, width * 4, width * 4])
        let sectionCount = input.count
        for i in input.indices {
            try mesh.call(.cpu { _, outputs in
                let x = outputs[0].data!.assumingMemoryBound(to: Float.self)
                for j in 0..<width { x[j] = Float((Int(outputs[0].index) * sectionCount + i) * width + j) }
            }, inputs: [], outputs: [input[i]], on: 0, worker: i)
        }
        let placed = try mesh.gather(input, to: size - 1)
        let partials = try mesh.tensor(on: size - 1, sections: input.map(\.bytes))
        for i in partials.indices { mesh.map(transform, input: placed[i], output: partials[i], inputView: features, outputView: options, worker: i) }
        let collected = try mesh.gather(partials, to: 0)
        let result = try mesh.tensor(on: 0, sections: input.map(\.bytes))
        for i in result.indices { mesh.map(transform, input: collected[i], output: result[i], inputView: features, outputView: options, worker: i) }
        try mesh.start()
        for index in 0..<mesh.count { mesh.submit(index) }
        withExtendedLifetime((mesh, result)) { dispatchMain() }
    }
}
