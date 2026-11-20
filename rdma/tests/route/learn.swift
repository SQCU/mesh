// The fp32 reference's backward (CHECK=1 GRAPH=1 compares model.swift's int8 network against it), and UPGrad's weights. For a batch of B windows, each K refreshes from its initial
// state: the forward again (policy.swift's layers), refresh by refresh with the state carried, the drawn-from contexts'
// w from their refresh's state; each task's surrogate sum(w . dw_t) (dw from the environment's credit, already scaled
// by the advantages); each one's gradient in theta by the graph's autodiff, back through the window's refreshes; and
// their Gram matrix. Then Jacobian descent (Quinton & Rey 2024, UPGrad: each task's gradient projected onto the dual
// cone of the 3 x P Jacobian, the projections averaged) and Adam, in place on theta and its moments.
import Foundation
import Metal
import MetalPerformanceShadersGraph

final class Learn {
  let g = MPSGraph(), theta: MPSGraphTensor, lanes: MPSGraphTensor, mask: MPSGraphTensor, state: MPSGraphTensor, tokens: MPSGraphTensor, index: MPSGraphTensor, dw: MPSGraphTensor
  let grads: [MPSGraphTensor], gram: MPSGraphTensor, w: MPSGraphTensor
  // lanes [K, B, L, IN], mask [K, B, L], state [B, 2, D]; context row r reads refresh-state index[r] = k B + b
  init(_ lay: Layout, windows B: Int, refreshes K: Int, rows R: Int) {
    theta = g.placeholder(shape: dims([lay.count]), dataType: .float32, name: "theta")
    lanes = g.placeholder(shape: dims([K, B, L, IN]), dataType: .float32, name: "lanes")
    mask = g.placeholder(shape: dims([K, B, L]), dataType: .float32, name: "mask")
    state = g.placeholder(shape: dims([B, 2, D]), dataType: .float32, name: "state")
    tokens = g.placeholder(shape: dims([R, IN]), dataType: .float32, name: "tokens")
    index = g.placeholder(shape: dims([R]), dataType: .int32, name: "index")
    dw = g.placeholder(shape: dims([TASKS, R, FS]), dataType: .float32, name: "dw")
    let net = Layers(g, lay, theta: theta)
    var h = layers(g, state, B: B), states: [[MPSGraphTensor]] = [[], []]
    for k in 0..<K {
      let lk = g.reshape(g.sliceTensor(lanes, dimension: 0, start: k, length: 1, name: nil), shape: dims([B, L, IN]), name: nil)
      let mk = g.reshape(g.sliceTensor(mask, dimension: 0, start: k, length: 1, name: nil), shape: dims([B, L]), name: nil)
      h = net.lanes(lk, mk, h, B: B)
      for l in 0..<2 { states[l].append(h[l]) }
    }
    let graph = g
    w = net.contexts(tokens, index, states.map { graph.concatTensors($0, dimension: 0, name: nil) })
    var gs: [MPSGraphTensor] = []
    for t in 0..<TASKS {
      let d = g.reshape(g.sliceTensor(dw, dimension: 0, start: t, length: 1, name: nil), shape: dims([R, FS]), name: nil)
      gs.append(g.gradients(of: g.reductionSum(with: g.multiplication(w, d, name: nil), axes: [0, 1], name: nil), with: [theta], name: nil)[theta]!)
    }
    grads = gs
    var dots: [MPSGraphTensor] = []
    for a in 0..<TASKS { for b in 0..<TASKS { dots.append(g.reductionSum(with: g.multiplication(gs[a], gs[b], name: nil), axes: [0], name: nil)) } }
    gram = g.concatTensors(dots, dimension: 0, name: nil)
  }
}

