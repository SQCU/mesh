import Foundation
import Mesh

// design/algorithm-sources.md#program
func linearChain(_ mesh: Mesh, coordinates: [[Float]]) throws -> [TensorPart] {
    let transform = TensorFunction.cpu { inputs, outputs in
        let a = inputs[0].data!.assumingMemoryBound(to: Float.self)
        let b = outputs[0].data!.assumingMemoryBound(to: Float.self)
        b[0] = a[0] + a[1]; b[1] = a[1] + a[2]; b[2] = a[2] + a[0]
    }
    let add = TensorFunction.cpu { inputs, outputs in
        let a = inputs[0].data!.assumingMemoryBound(to: Float.self)
        let b = inputs[1].data!.assumingMemoryBound(to: Float.self)
        let c = outputs[0].data!.assumingMemoryBound(to: Float.self)
        for i in 0..<3 { c[i] = a[i] + b[i] }
    }
    let owners = [0, mesh.size - 1, 0]
    var transformed: [TensorPart] = [], consumed: [TensorPart] = []
    for i in 0..<3 {
        let input = try mesh.tensor(on: 0, sections: [12])
        try mesh.call(.cpu { _, outputs in
            let p = outputs[0].data!.assumingMemoryBound(to: Float.self)
            for j in 0..<3 { p[j] = i == j ? coordinates[Int(outputs[0].index)][i] : 0 }
        }, inputs: [], outputs: input, on: 0, worker: i)
        let placed = try mesh.scatter(input, to: [owners[i]])
        let output = try mesh.tensor(on: owners[i], sections: [12])
        try mesh.map(transform, inputs: [placed], outputs: output, workers: [i])
        transformed.append(output[0])
        let consumerInput = try mesh.send(output[0], to: mesh.size - 1)
        let consumerOutput = try mesh.tensor(on: mesh.size - 1, sections: [12])
        try mesh.map(transform, inputs: [[consumerInput]], outputs: consumerOutput, workers: [i])
        consumed.append(consumerOutput[0])
    }
    let reconstructed = try mesh.reduce(transformed, to: 0, using: add, worker: 0)
    let consumerResult = try mesh.reduce(consumed, to: mesh.size - 1, using: add, worker: 0)
    return [reconstructed, consumerResult]
}

@main
struct LinearChain {
    // design/algorithm-sources.md#program
    static func main() throws {
        let rank = Int(CommandLine.arguments[1])!
        let size = Int(CommandLine.arguments[2])!
        let mesh = try Mesh(region: CommandLine.arguments[3], rank: rank, size: size, workers: 3, count: 4)
        let outputs = try linearChain(mesh, coordinates: (0..<mesh.count).map { [Float($0 + 1), 2, 3] })
        try mesh.start()
        for index in [3, 0, 2, 1] { mesh.submit(index) }
        withExtendedLifetime((mesh, outputs)) { dispatchMain() }
    }
}
