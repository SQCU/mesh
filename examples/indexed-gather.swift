import Foundation
import Mesh

private struct Plan: Decodable {
    let valuesOwner, indicesOwner, rows, columns, neighbours: Int
    let consumers: [Int]
    let routes: [[Int]]
}

@main
struct IndexedGather {
    // design/algorithm-sources.md#programmap
    static func main() throws {
        let args = CommandLine.arguments
        let rank = Int(args[1])!, size = Int(args[2])!
        let p = try JSONDecoder().decode(Plan.self, from: Data(contentsOf: URL(fileURLWithPath: args[4])))
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let placement = Placement(routes: Dictionary(uniqueKeysWithValues: p.routes.map { _ = mesh_callbacks_are_outside_spec;
            (Placement.Edge($0[0], $0.last!), Array($0.dropFirst()))
        }))
        let mesh = try Mesh(region: args[3], rank: rank, size: size, workers: 2, placement: placement)
        let rows = p.rows, columns = p.columns, neighbours = p.neighbours
        let values = try mesh.tensor(on: p.valuesOwner, sections: [rows * columns * 4, rows * columns * 4])
        // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let produceValues = TensorFunction.cpu { _, outputs in _ = mesh_callbacks_are_outside_spec;
            for output in outputs {
                let data = output.data!.assumingMemoryBound(to: Float.self)
                for i in 0..<(output.bytes / 4) { data[i] = Float(i + 1) }
            }
        }
        try mesh.call(produceValues, inputs: [], outputs: values, on: p.valuesOwner, worker: 0)
        let indices = try mesh.tensor(on: p.indicesOwner, sections: [rows * 4, rows * neighbours * 4])
        // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let produceIndices = TensorFunction.cpu { _, outputs in _ = mesh_callbacks_are_outside_spec;
            let experts = outputs[0].data!.assumingMemoryBound(to: Int32.self)
            let neighbours = outputs[1].data!.assumingMemoryBound(to: Int32.self)
            let count = outputs[1].bytes / outputs[0].bytes
            for row in 0..<(outputs[0].bytes / 4) {
                experts[row] = Int32(row % 2)
                for k in 0..<count { neighbours[row * count + k] = Int32((row + k + 1) % rows) }
            }
        }
        try mesh.call(produceIndices, inputs: [], outputs: indices, on: p.indicesOwner, worker: 1)
        for owner in p.consumers {
            let inputs = try mesh.gather(values + indices, to: owner)
            // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            let weights = try (0..<2).map { expert in _ = mesh_callbacks_are_outside_spec;
                // mesh callback (preparation; passed to mesh.constant): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                try mesh.constant(on: owner, bytes: columns * 4) { span in _ = mesh_callbacks_are_outside_spec;
                    let values = span.data.baseAddress!.assumingMemoryBound(to: Float.self)
                    for i in 0..<(span.data.count / 4) { values[i] = Float(1 + expert * i) }
                }
            }
            let outputs = try mesh.tensor(on: owner, sections: [rows * 4, rows * columns * 4])
            // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            let routedExpert = TensorFunction.cpu { inputs, outputs in _ = mesh_callbacks_are_outside_spec;
                let out = outputs[0].data!.assumingMemoryBound(to: Float.self)
                let columns = inputs[1].bytes / 4
                for row in 0..<(outputs[0].bytes / 4) {
                    let expert = Int(inputs[3].load(at: row, as: Int32.self))
                    var dot: Float = 0
                    for column in 0..<columns {
                        dot += inputs[0].load(at: row * columns + column, as: Float.self)
                            * inputs[1 + expert].load(at: column, as: Float.self)
                    }
                    out[row] = dot
                }
            }
            try mesh.map(routedExpert, inputs: [[inputs[0]], [weights[0]], [weights[1]], [inputs[2]]],
                         outputs: [outputs[0]], workers: [0])
            // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            let neighbourhoodSum = TensorFunction.cpu { inputs, outputs in _ = mesh_callbacks_are_outside_spec;
                let out = outputs[0].data!.assumingMemoryBound(to: Float.self)
                for row in 0..<(outputs[0].bytes / 4 / columns) {
                    for column in 0..<columns {
                        var sum: Float = 0
                        for k in 0..<neighbours {
                            let index = Int(inputs[1].load(at: row * neighbours + k, as: Int32.self))
                            sum += inputs[0].load(at: index * columns + column, as: Float.self)
                        }
                        out[row * columns + column] = sum
                    }
                }
            }
            try mesh.map(neighbourhoodSum, inputs: [[inputs[1]], [inputs[3]]], outputs: [outputs[1]], workers: [1])
            for i in outputs.indices {
                // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                try mesh.call(.cpu { inputs, _ in _ = mesh_callbacks_are_outside_spec;
                    let first: Float = inputs[0].load(at: 0), last: Float = inputs[0].load(at: inputs[0].bytes / 4 - 1)
                    // mesh callback (completion; passed to DispatchQueue.main.async): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                    DispatchQueue.main.async { _ = mesh_callbacks_are_outside_spec;
                        print("rank=\(rank) part=\(i) first=\(first) last=\(last)")
                        fflush(stdout)
                    }
                }, inputs: [outputs[i]], outputs: [], on: owner, worker: i)
            }
        }
        try mesh.start()
        // design/prepared-machine.md#M41
        mesh.invocation(0).submit()
        // mesh callback (execution; passed to withExtendedLifetime): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        withExtendedLifetime(mesh) { _ = mesh_callbacks_are_outside_spec; dispatchMain() }
    }
}
