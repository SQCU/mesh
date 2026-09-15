import Foundation
import Mesh

@main
struct IndexedGather {
    // design/algorithm-sources.md#programmap
    static func main() throws {
        // idx is an ordinary TensorPart, produced by a supplied function or received via send.
        // map binds idx as one more input operand, so the consumer's pending count (X2)
        // includes it and the map fires only when idx is present.
        // Indexed reads, expert selection and neighbour sums live entirely in supplied closures.
        // Mesh sees operands and never an index.
        let args = CommandLine.arguments
        let rank = Int(args[1])!, size = Int(args[2])!
        let mesh = try Mesh(region: args[3], rank: rank, size: size, workers: 2)
        let owner = size == 2 ? 1 : 0
        let rows = 4, columns = 3, neighbours = 2
        let values = try mesh.tensor(on: owner, sections: [rows * columns * 4, rows * columns * 4])
        let x = values[0], table = values[1]
        // design/algorithm-sources.md#programwrite
        let produceValues = TensorFunction.cpu { _, outputs in
            for output in outputs {
                let data = output.data!.assumingMemoryBound(to: Float.self)
                for i in 0..<(output.bytes / 4) { data[i] = Float(i + 1) }
            }
        }
        try mesh.call(produceValues, inputs: [], outputs: [x, table], on: owner, worker: 0)

        var expertIdx = try mesh.tensor(on: 0, sections: [rows * 4])[0]
        var idx = try mesh.tensor(on: 0, sections: [rows * neighbours * 4])[0]
        // design/algorithm-sources.md#programmap
        let produceExperts = TensorFunction.cpu { _, outputs in
            let indices = outputs[0].data!.assumingMemoryBound(to: Int32.self)
            for row in 0..<(outputs[0].bytes / 4) { indices[row] = Int32(row % 2) }
        }
        // design/algorithm-sources.md#programmap
        let produceNeighbours = TensorFunction.cpu { _, outputs in
            let indices = outputs[0].data!.assumingMemoryBound(to: Int32.self)
            let rows = outputs[0].bytes / 4 / neighbours
            for row in 0..<rows {
                for k in 0..<neighbours { indices[row * neighbours + k] = Int32((row + k + 1) % rows) }
            }
        }
        try mesh.call(produceExperts, inputs: [], outputs: [expertIdx], on: 0, worker: 0)
        try mesh.call(produceNeighbours, inputs: [], outputs: [idx], on: 0, worker: 1)
        if size == 2 {
            expertIdx = try mesh.send(expertIdx, to: 1)
            idx = try mesh.send(idx, to: 1)
        }

        // design/algorithm-sources.md#programtensor
        let w0 = try mesh.constant(on: owner, bytes: columns * 4) { span in
            let weights = span.data.baseAddress!.assumingMemoryBound(to: Float.self)
            for i in 0..<(span.data.count / 4) { weights[i] = 1 }
        }
        // design/algorithm-sources.md#programtensor
        let w1 = try mesh.constant(on: owner, bytes: columns * 4) { span in
            let weights = span.data.baseAddress!.assumingMemoryBound(to: Float.self)
            for i in 0..<(span.data.count / 4) { weights[i] = Float(i + 1) }
        }
        let expertOut = try mesh.tensor(on: owner, sections: [rows * 4])[0]
        // design/algorithm-sources.md#programmap
        let routedExpert = TensorFunction.cpu { inputs, outputs in
            let x = inputs[0].data!.assumingMemoryBound(to: Float.self)
            let w0 = inputs[1].data!.assumingMemoryBound(to: Float.self)
            let w1 = inputs[2].data!.assumingMemoryBound(to: Float.self)
            let idx = inputs[3].data!.assumingMemoryBound(to: Int32.self)
            let out = outputs[0].data!.assumingMemoryBound(to: Float.self)
            let columns = inputs[1].bytes / 4
            for row in 0..<(outputs[0].bytes / 4) {
                let weights = idx[row] == 0 ? w0 : w1
                var dot: Float = 0
                for column in 0..<columns { dot += x[row * columns + column] * weights[column] }
                out[row] = dot
            }
        }
        try mesh.map(routedExpert, inputs: [[x], [w0], [w1], [expertIdx]], outputs: [expertOut], workers: [0])

        let out = try mesh.tensor(on: owner, sections: [rows * columns * 4])[0]
        // design/algorithm-sources.md#programmap
        let neighbourhoodSum = TensorFunction.cpu { inputs, outputs in
            let table = inputs[0].data!.assumingMemoryBound(to: Float.self)
            let idx = inputs[1].data!.assumingMemoryBound(to: Int32.self)
            let out = outputs[0].data!.assumingMemoryBound(to: Float.self)
            for row in 0..<(outputs[0].bytes / 4 / columns) {
                for column in 0..<columns {
                    var sum: Float = 0
                    for k in 0..<neighbours { sum += table[Int(idx[row * neighbours + k]) * columns + column] }
                    out[row * columns + column] = sum
                }
            }
        }
        try mesh.map(neighbourhoodSum, inputs: [[table], [idx]], outputs: [out], workers: [1])

        // design/algorithm-sources.md#programkernel_call
        try mesh.call(.cpu { inputs, _ in
            let data = inputs[0].data!.assumingMemoryBound(to: Float.self)
            let first = data[0], second = data[1], third = data[2]
            // design/algorithm-sources.md#programkernel_call
            DispatchQueue.main.async {
                print("routed experts: \(first), \(second), \(third)")
                fflush(stdout)
            }
        }, inputs: [expertOut], outputs: [], on: owner, worker: 0)
        // design/algorithm-sources.md#programkernel_call
        try mesh.call(.cpu { inputs, _ in
            let data = inputs[0].data!.assumingMemoryBound(to: Float.self)
            let first = data[0], second = data[1], third = data[2]
            // design/algorithm-sources.md#programkernel_call
            DispatchQueue.main.async {
                print("neighbourhood sum: \(first), \(second), \(third)")
                fflush(stdout)
            }
        }, inputs: [out], outputs: [], on: owner, worker: 1)
        try mesh.start()
        mesh.submit(0)
        // design/algorithm-sources.md#program
        withExtendedLifetime((mesh, expertOut, out)) { dispatchMain() }
    }
}
