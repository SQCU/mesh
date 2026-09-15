import Foundation
import Mesh

@main
struct SyncOnRemoteFill {
    // design/algorithm-sources.md#collectivesync_on_remote_fill
    static func main() throws {
        let rank = Int(CommandLine.arguments[1])!
        let mesh = try Mesh(region: CommandLine.arguments[2], rank: rank, size: 2, workers: 4)
        let mode = CommandLine.arguments[3]
        let source = try mesh.tensor(on: 0, sections: [4, 4, 4, 4])
        let remote = try mesh.gather(source, to: 1)
        let answer = try mesh.tensor(on: 1, sections: [4, 4, 4, 4])
        let returned = try mesh.gather(answer, to: 0)
        for i in 0..<4 {
            let produce: (MeshOperands, MeshOperands) -> Void = { _, outputs in
                outputs[0].data!.storeBytes(of: Float(i + 1), as: Float.self)
            }
            let function: TensorFunction
            switch mode {
            case "deadlock":
                function = .cpu { inputs, outputs in
                    mesh.syncOnRemoteFill([returned[i]])
                    produce(inputs, outputs)
                }
            case "serial" where i > 0:
                function = .cpu { inputs, outputs in
                    mesh.syncOnRemoteFill([returned[i - 1]])
                    produce(inputs, outputs)
                }
            default:
                function = .cpu(produce)
            }
            try mesh.call(function, inputs: [], outputs: [source[i]], on: 0, worker: i)
            try mesh.call(.cpu { inputs, outputs in
                outputs[0].data!.storeBytes(of: inputs[0].data!.load(as: Float.self) + 1, as: Float.self)
            }, inputs: [remote[i]], outputs: [answer[i]], on: 1, worker: i)
        }
        try mesh.start()
        mesh.submit(0)
        withExtendedLifetime((mesh, returned)) { dispatchMain() }
    }
}
