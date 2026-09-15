import Accelerate
import Foundation
import Mesh

private struct Plan: Decodable {
    let owners, rows, latent, hidden, width: [Int]
    let blocks, count, workers: Int
}

// design/algorithm-sources.md#programkernel_call
private func contract(_ mesh: Mesh, terms: [Int], outputs: [(owner: Int, bytes: Int)],
                      workers: [Int], using combine: TensorFunction,
                      _ kernel: (Int, Int) -> (TensorFunction, [TensorPart])) throws -> [TensorPart] {
    var contributions: [[TensorPart]] = []
    for k in terms.indices {
        let parts = try mesh.tensor(on: terms[k], sections: outputs.map(\.bytes))
        for o in outputs.indices {
            let (function, operands) = kernel(k, o)
            try mesh.call(function, inputs: mesh.gather(operands, to: terms[k]), outputs: [parts[o]],
                          on: terms[k], worker: workers[o])
        }
        contributions.append(parts)
    }
    return try mesh.reduceScatter(contributions, to: outputs.map(\.owner), using: combine, workers: workers)
}

// design/algorithm-sources.md#programkernel_call
private func product(rows: Int, columns: Int, inner: Int, transpose: Bool = false) -> TensorFunction {
    let m = Int32(rows), n = Int32(columns), k = Int32(inner)
    let trans = transpose ? CblasTrans : CblasNoTrans, lda = transpose ? m : k
    return .cpu { inputs, outputs in
        cblas_sgemm(CblasRowMajor, trans, CblasNoTrans, m, n, k, 1,
                    inputs[0].data!.assumingMemoryBound(to: Float.self), lda,
                    inputs[1].data!.assumingMemoryBound(to: Float.self), n, 0,
                    outputs[0].data!.assumingMemoryBound(to: Float.self), n)
    }
}

@main
struct GramChain {
    // design/algorithm-sources.md#program
    static func main() throws {
        let args = CommandLine.arguments
        let rank = Int(args[1])!, size = Int(args[2])!
        let p = try JSONDecoder().decode(Plan.self, from: Data(contentsOf: URL(fileURLWithPath: args[4])))
        let n = p.owners.count
        precondition(n > 0 && [p.rows, p.latent, p.hidden, p.width].allSatisfy { $0.count == n && $0.allSatisfy { $0 > 0 } })
        precondition(p.owners.allSatisfy { (0..<size).contains($0) } && p.blocks > 0 && p.count > 0 && p.workers > 0)
        let mesh = try Mesh(region: args[3], rank: rank, size: size, workers: p.workers, count: p.count)
        let tiles = Array(0..<n * n), workers = tiles.map { $0 % p.workers }
        let rowOwners = tiles.map { p.owners[$0 / n] }, columnOwners = tiles.map { p.owners[$0 % n] }
        let zBytes = tiles.map { p.rows[$0 / n] * p.width[$0 % n] * 4 }
        let hBytes = tiles.map { p.rows[$0 / n] * p.hidden[$0 % n] * 4 }
        let uBytes = tiles.map { p.latent[$0 / n] * p.hidden[$0 % n] * 4 }
        let zLayout = Array(zip(columnOwners, zBytes)), hLayout = Array(zip(columnOwners, hBytes))
        let uLayout = Array(zip(rowOwners, uBytes))
        let add = TensorFunction.cpu { inputs, outputs in
            vDSP_vadd(inputs[0].data!.assumingMemoryBound(to: Float.self), 1,
                      inputs[1].data!.assumingMemoryBound(to: Float.self), 1,
                      outputs[0].data!.assumingMemoryBound(to: Float.self), 1, vDSP_Length(outputs[0].bytes / 4))
        }
        var z = try tiles.map { try mesh.tensor(on: columnOwners[$0], sections: [zBytes[$0]])[0] }
        let produce = TensorFunction.cpu { _, outputs in
            var first = Float(outputs[0].index), step: Float = 0.001
            vDSP_vramp(&first, &step, outputs[0].data!.assumingMemoryBound(to: Float.self), 1,
                       vDSP_Length(outputs[0].bytes / 4))
        }
        for o in tiles { try mesh.call(produce, inputs: [], outputs: [z[o]], on: columnOwners[o], worker: workers[o]) }

        for block in 0..<p.blocks {
            // design/algorithm-sources.md#programtensor
            func weights(_ rows: [Int], _ columns: [Int], seed: Int) throws -> [TensorPart] {
                try tiles.map { o in
                    let scale = 1 / sqrt(Float(rows.reduce(0, +)))
                    return try mesh.constant(on: rowOwners[o], bytes: rows[o / n] * columns[o % n] * 4) { span in
                        let data = span.data.baseAddress!.assumingMemoryBound(to: Float.self)
                        for i in 0..<(span.data.count / 4) { data[i] = sin(Float(i + o * 31 + seed)) * scale }
                    }
                }
            }
            let a = try weights(p.width, p.hidden, seed: 73 + block * 3)
            let r = try weights(p.rows, p.latent, seed: 74 + block * 3)
            let w = try weights(p.hidden, p.width, seed: 75 + block * 3)
            let h = try contract(mesh, terms: p.owners, outputs: hLayout, workers: workers, using: add) { k, o in
                (product(rows: p.rows[o / n], columns: p.hidden[o % n], inner: p.width[k]),
                 [z[o / n * n + k], a[k * n + o % n]])
            }
            let u = try contract(mesh, terms: p.owners, outputs: uLayout, workers: workers, using: add) { k, o in
                (product(rows: p.latent[o / n], columns: p.hidden[o % n], inner: p.rows[k], transpose: true),
                 [r[k * n + o / n], h[k * n + o % n]])
            }
            let y = try contract(mesh, terms: p.owners, outputs: hLayout, workers: workers, using: add) { k, o in
                (product(rows: p.rows[o / n], columns: p.hidden[o % n], inner: p.latent[k]),
                 [r[o / n * n + k], u[k * n + o % n]])
            }
            let down = try contract(mesh, terms: p.owners, outputs: zLayout, workers: workers, using: add) { k, o in
                (product(rows: p.rows[o / n], columns: p.width[o % n], inner: p.hidden[k]),
                 [y[o / n * n + k], w[k * n + o % n]])
            }
            let next = try tiles.map { try mesh.tensor(on: columnOwners[$0], sections: [zBytes[$0]])[0] }
            try mesh.map(add, inputs: [z, down], outputs: next, workers: workers)
            z = next
        }
        try mesh.start()
        for index in 0..<p.count { mesh.submit(index) }
        withExtendedLifetime((mesh, z)) { dispatchMain() }
    }
}
