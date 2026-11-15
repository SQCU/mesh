import Accelerate
import Foundation
import Mesh

private struct Plan: Decodable {
    let owners, rows, latent, hidden, width: [Int]
    let blocks, count, inFlight, workers: Int
    let routes: [[Int]]?
}

// design/algorithm-sources.md#programkernel_call
private func contract(_ mesh: Mesh, terms: [Int], outputs: [(owner: Int, bytes: Int)],
                      workers: [Int], using combine: TensorFunction,
                      _ kernel: (Int, Int) -> (TensorFunction, [TensorPart])) throws -> [TensorPart] {
    var contributions: [[TensorPart]] = []
    for k in terms.indices {
        // mesh callback (compilation/construction; key path \.bytes passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let parts = try mesh.tensor(on: terms[k], sections: outputs.map(mesh_callbacks_are_outside_spec(\.bytes)))
        for o in outputs.indices {
            let (function, operands) = kernel(k, o)
            try mesh.call(function, inputs: mesh.gather(operands, to: terms[k]), outputs: [parts[o]],
                          on: terms[k], worker: workers[o])
        }
        contributions.append(parts)
    }
    // mesh callback (compilation/construction; key path \.owner passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
    return try mesh.reduceScatter(contributions, to: outputs.map(mesh_callbacks_are_outside_spec(\.owner)), using: combine, workers: workers)
}

// design/algorithm-sources.md#programkernel_call
private func product(rows: Int, columns: Int, inner: Int, transpose: Bool = false) -> TensorFunction {
    let m = Int32(rows), n = Int32(columns), k = Int32(inner)
    let trans = transpose ? CblasTrans : CblasNoTrans, lda = transpose ? m : k
    // mesh callback (execution; registered as a TensorFunction view): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
    let view: (MeshSpan) -> UnsafeMutablePointer<Float> = { _ = mesh_callbacks_are_outside_spec; $0.data.baseAddress!.assumingMemoryBound(to: Float.self) }
    // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
    return TensorFunction(inputViews: [view, view], outputViews: [view]) { inputs, outputs in _ = mesh_callbacks_are_outside_spec;
        // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        .cpu { ins, outs in _ = mesh_callbacks_are_outside_spec;
            cblas_sgemm(CblasRowMajor, trans, CblasNoTrans, m, n, k, 1,
                        inputs[0][ins[0]], lda, inputs[1][ins[1]], n, 0, outputs[0][outs[0]], n)
        }
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
        // mesh callback (compilation/construction; passed to allSatisfy): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        precondition(n > 0 && [p.rows, p.latent, p.hidden, p.width].allSatisfy { _ = mesh_callbacks_are_outside_spec; $0.count == n && $0.allSatisfy { _ = mesh_callbacks_are_outside_spec; $0 > 0 } })
        // mesh callback (compilation/construction; passed to allSatisfy): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        precondition(p.owners.allSatisfy { _ = mesh_callbacks_are_outside_spec; (0..<size).contains($0) } && p.blocks > 0 && p.count > 0 && p.workers > 0)
        // mesh callback (compilation/construction; passed to allSatisfy): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        precondition((p.routes ?? []).allSatisfy { _ = mesh_callbacks_are_outside_spec; $0.count >= 2 })
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let placement = Placement(owners: p.owners, routes: Dictionary(uniqueKeysWithValues: (p.routes ?? []).map { _ = mesh_callbacks_are_outside_spec;
            (Placement.Edge($0[0], $0.last!), Array($0.dropFirst()))
        }))
        let mesh = try Mesh(region: args[3], rank: rank, size: size, workers: p.workers, inFlight: p.inFlight, placement: placement)
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let tiles = Array(0..<n * n), workers = tiles.map { _ = mesh_callbacks_are_outside_spec; $0 % p.workers }
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let rowOwners = tiles.map { _ = mesh_callbacks_are_outside_spec; p.owners[$0 / n] }, columnOwners = tiles.map { _ = mesh_callbacks_are_outside_spec; p.owners[$0 % n] }
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let zBytes = tiles.map { _ = mesh_callbacks_are_outside_spec; p.rows[$0 / n] * p.width[$0 % n] * 4 }
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let hBytes = tiles.map { _ = mesh_callbacks_are_outside_spec; p.rows[$0 / n] * p.hidden[$0 % n] * 4 }
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let uBytes = tiles.map { _ = mesh_callbacks_are_outside_spec; p.latent[$0 / n] * p.hidden[$0 % n] * 4 }
        let zLayout = Array(zip(columnOwners, zBytes)), hLayout = Array(zip(columnOwners, hBytes))
        let uLayout = Array(zip(rowOwners, uBytes))
        // mesh callback (execution; registered as a TensorFunction view): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let view: (MeshSpan) -> UnsafeMutablePointer<Float> = { _ = mesh_callbacks_are_outside_spec; $0.data.baseAddress!.assumingMemoryBound(to: Float.self) }
        // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let add = TensorFunction(inputViews: [view, view], outputViews: [view]) { inputs, outputs in _ = mesh_callbacks_are_outside_spec;
            // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            .cpu { ins, outs in _ = mesh_callbacks_are_outside_spec;
                vDSP_vadd(inputs[0][ins[0]], 1, inputs[1][ins[1]], 1, outputs[0][outs[0]], 1, vDSP_Length(outs[0].bytes / 4))
            }
        }
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        var z = try tiles.map { _ = mesh_callbacks_are_outside_spec; try mesh.tensor(on: columnOwners[$0], sections: [zBytes[$0]])[0] }
        // mesh callback (execution; registered as a TensorFunction CPU body): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let produce = TensorFunction.cpu { _, outputs in _ = mesh_callbacks_are_outside_spec;
            var first = Float(outputs[0].invocation), step: Float = 0.001
            vDSP_vramp(&first, &step, outputs[0].data!.assumingMemoryBound(to: Float.self), 1,
                       vDSP_Length(outputs[0].bytes / 4))
        }
        for o in tiles { try mesh.call(produce, inputs: [], outputs: [z[o]], on: columnOwners[o], worker: workers[o]) }

        for block in 0..<p.blocks {
            // design/algorithm-sources.md#programtensor
            func weights(_ rows: [Int], _ columns: [Int], seed: Int) throws -> [TensorPart] {
                // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                try tiles.map { o in _ = mesh_callbacks_are_outside_spec;
                    // mesh callback (compilation/construction; operator + passed to reduce): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                    let scale = 1 / sqrt(Float(rows.reduce(0, mesh_callbacks_are_outside_spec(+))))
                    // mesh callback (preparation; passed to mesh.constant): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                    return try mesh.constant(on: rowOwners[o], bytes: rows[o / n] * columns[o % n] * 4) { span in _ = mesh_callbacks_are_outside_spec;
                        let data = span.data.baseAddress!.assumingMemoryBound(to: Float.self)
                        for i in 0..<(span.data.count / 4) { data[i] = sin(Float(i + o * 31 + seed)) * scale }
                    }
                }
            }
            let a = try weights(p.width, p.hidden, seed: 73 + block * 3)
            let r = try weights(p.rows, p.latent, seed: 74 + block * 3)
            let w = try weights(p.hidden, p.width, seed: 75 + block * 3)
            // mesh callback (compilation/construction; passed to contract): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            let h = try contract(mesh, terms: p.owners, outputs: hLayout, workers: workers, using: add) { k, o in _ = mesh_callbacks_are_outside_spec;
                (product(rows: p.rows[o / n], columns: p.hidden[o % n], inner: p.width[k]),
                 [z[o / n * n + k], a[k * n + o % n]])
            }
            // mesh callback (compilation/construction; passed to contract): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            let u = try contract(mesh, terms: p.owners, outputs: uLayout, workers: workers, using: add) { k, o in _ = mesh_callbacks_are_outside_spec;
                (product(rows: p.latent[o / n], columns: p.hidden[o % n], inner: p.rows[k], transpose: true),
                 [r[k * n + o / n], h[k * n + o % n]])
            }
            // mesh callback (compilation/construction; passed to contract): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            let y = try contract(mesh, terms: p.owners, outputs: hLayout, workers: workers, using: add) { k, o in _ = mesh_callbacks_are_outside_spec;
                (product(rows: p.rows[o / n], columns: p.hidden[o % n], inner: p.latent[k]),
                 [r[o / n * n + k], u[k * n + o % n]])
            }
            // mesh callback (compilation/construction; passed to contract): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            let down = try contract(mesh, terms: p.owners, outputs: zLayout, workers: workers, using: add) { k, o in _ = mesh_callbacks_are_outside_spec;
                (product(rows: p.rows[o / n], columns: p.width[o % n], inner: p.hidden[k]),
                 [y[o / n * n + k], w[k * n + o % n]])
            }
            // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
            let next = try tiles.map { _ = mesh_callbacks_are_outside_spec; try mesh.tensor(on: columnOwners[$0], sections: [zBytes[$0]])[0] }
            try mesh.map(add, inputs: [z, down], outputs: next, workers: workers)
            z = next
        }
        try mesh.start()
        // design/prepared-machine.md#M41
        // design/prepared-machine.md#M42
        // mesh callback (compilation/construction; passed to map): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        let invocations = (0..<mesh.inFlight).map { _ = mesh_callbacks_are_outside_spec; mesh.invocation($0) }
        if p.owners.contains(rank) {
            var pending = [UInt32?](repeating: nil, count: invocations.count)
            var submitted = 0, completed = 0
            while completed < p.count {
                for i in invocations.indices {
                    if let generation = pending[i] {
                        switch invocations[i].result(generation) {
                        case .success: pending[i] = nil; completed += 1
                        case .failure(.busy): continue
                        case .failure(let error): throw error
                        }
                    }
                    if submitted < p.count {
                        pending[i] = invocations[i].submit()
                        submitted += 1
                    }
                }
            }
        }
        // mesh callback (execution; passed to withExtendedLifetime): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
        withExtendedLifetime((mesh, z)) { _ = mesh_callbacks_are_outside_spec; dispatchMain() }
    }
}
