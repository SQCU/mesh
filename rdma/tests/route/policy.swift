// The forward (route.h): two recurrent layers on a 64-wide residual stream, each a recurrence and a feed-forward block
// expanding to 192, over every pending decision point's lane tokens at once; then one step more for each context's
// token from its decision point's state; then w = out . rms(output) for each context. The parameters are one flat
// vector theta, sliced here into the blocks the Layout names.
//   GRU:    z = sigmoid(Wz [h; x] + bz), r = sigmoid(Wr [h; x] + br), h~ = tanh(Wh [r h; x] + bh)
//   minGRU: z = sigmoid(Wz x + bz), h~ = tanh(Wh x + bh) (gates from the input: the lane recurrence's gates are one
//           matrix product over every position, the recurrence an elementwise scan)
//   h' = (1 - z) h + z h~; x' = x + h'; out = x' + down relu(up x' + bu) + bd. A padded lane leaves the state as it was.
import Foundation
import MetalPerformanceShadersGraph

let D = Int(ROUTE_D), E = Int(ROUTE_E), IN = Int(ROUTE_IN), FS = Int(ROUTE_FS), L = Int(ROUTE_L), TASKS = Int(ROUTE_TASKS)

// the parameter blocks, in theta's order: (name, shape, offset)
struct Layout {
  let full: Bool
  let blocks: [(name: String, shape: [Int], offset: Int)]
  let count: Int
  init(full: Bool) {
    self.full = full
    var b: [(name: String, shape: [Int], offset: Int)] = [], n = 0
    func add(_ name: String, _ shape: [Int]) { b.append((name, shape, n)); n += shape.reduce(1, *) }
    let gw = full ? 2 * D : D
    add("inp", [D, IN]); add("binp", [D])
    for l in 0..<2 {
      add("z\(l)", [D, gw]); add("bz\(l)", [D])
      if full { add("r\(l)", [D, gw]); add("br\(l)", [D]) }
      add("h\(l)", [D, gw]); add("bh\(l)", [D]); add("up\(l)", [E, D]); add("bu\(l)", [E]); add("down\(l)", [D, E]); add("bd\(l)", [D])
    }
    add("out", [FS, D])
    blocks = b; count = n
  }
  // weights at 1/sqrt(fan-in), biases 0
  func initial(_ g: inout route_rng) -> [Float] {
    var t = [Float](repeating: 0, count: count)
    for (name, shape, offset) in blocks where shape.count == 2 {
      let fan = Double(shape[1])
      for i in 0..<(shape[0] * shape[1]) {
        let u = max(route_uniform(&g), 1e-300), v = route_uniform(&g)
        t[offset + i] = Float(sqrt(-2 * log(u)) * cos(2 * Double.pi * v) / sqrt(fan))
      }
      _ = name
    }
    return t
  }
}

// the forward on graph g: theta [P]; lanes [B, L, IN]; mask [B, L] (1 a real lane); tokens [R, IN]; index [R] (each row's
// decision point). Returns w [R, FS].
func forward(_ g: MPSGraph, _ lay: Layout, theta: MPSGraphTensor, lanes: MPSGraphTensor, mask: MPSGraphTensor, tokens: MPSGraphTensor,
             index: MPSGraphTensor, decisions B: Int, rows R: Int) -> MPSGraphTensor {
  var p: [String: MPSGraphTensor] = [:]
  for (name, shape, offset) in lay.blocks {
    let flat = g.sliceTensor(theta, dimension: 0, start: offset, length: shape.reduce(1, *), name: nil)
    p[name] = g.reshape(flat, shape: shape.map { NSNumber(value: $0) }, name: nil)
  }
  let one = g.constant(1, dataType: .float32)
  func dense(_ x: MPSGraphTensor, _ w: String, _ b: String?) -> MPSGraphTensor {
    let y = g.matrixMultiplication(primary: x, secondary: g.transposeTensor(p[w]!, dimension: 0, withDimension: 1, name: nil), name: nil)
    return b == nil ? y : g.addition(y, p[b!]!, name: nil)
  }
  func mix(_ z: MPSGraphTensor, _ h: MPSGraphTensor, _ c: MPSGraphTensor) -> MPSGraphTensor {   // (1 - z) h + z c
    g.addition(g.multiplication(g.subtraction(one, z, name: nil), h, name: nil), g.multiplication(z, c, name: nil), name: nil)
  }
  func cell(_ x: MPSGraphTensor, _ h: MPSGraphTensor, _ l: Int) -> MPSGraphTensor {   // the GRU's step
    let hx = g.concatTensors([h, x], dimension: 1, name: nil)
    let z = g.sigmoid(with: dense(hx, "z\(l)", "bz\(l)"), name: nil), r = g.sigmoid(with: dense(hx, "r\(l)", "br\(l)"), name: nil)
    let c = g.tanh(with: dense(g.concatTensors([g.multiplication(r, h, name: nil), x], dimension: 1, name: nil), "h\(l)", "bh\(l)"), name: nil)
    return mix(z, h, c)
  }
  func ffn(_ xm: MPSGraphTensor, _ l: Int) -> MPSGraphTensor {
    g.addition(xm, dense(g.reLU(with: dense(xm, "up\(l)", "bu\(l)"), name: nil), "down\(l)", "bd\(l)"), name: nil)
  }
  // the lanes, in turn
  var x = g.reshape(dense(g.reshape(lanes, shape: [NSNumber(value: B * L), NSNumber(value: IN)], name: nil), "inp", "binp"),
                    shape: [NSNumber(value: B), NSNumber(value: L), NSNumber(value: D)], name: nil)
  var final: [MPSGraphTensor] = []
  for l in 0..<2 {
    var gz: MPSGraphTensor? = nil, gc: MPSGraphTensor? = nil
    if !lay.full {   // minGRU: every position's gates at once
      let flat = g.reshape(x, shape: [NSNumber(value: B * L), NSNumber(value: D)], name: nil), shape3 = [NSNumber(value: B), NSNumber(value: L), NSNumber(value: D)]
      gz = g.reshape(g.sigmoid(with: dense(flat, "z\(l)", "bz\(l)"), name: nil), shape: shape3, name: nil)
      gc = g.reshape(g.tanh(with: dense(flat, "h\(l)", "bh\(l)"), name: nil), shape: shape3, name: nil)
    }
    var h = g.constant(0, shape: [NSNumber(value: B), NSNumber(value: D)], dataType: .float32)
    var outs: [MPSGraphTensor] = []
    for t in 0..<L {
      func at(_ y: MPSGraphTensor) -> MPSGraphTensor { g.reshape(g.sliceTensor(y, dimension: 1, start: t, length: 1, name: nil), shape: [NSNumber(value: B), NSNumber(value: D)], name: nil) }
      let hn = lay.full ? cell(at(x), h, l) : mix(at(gz!), h, at(gc!))
      let m = g.reshape(g.sliceTensor(mask, dimension: 1, start: t, length: 1, name: nil), shape: [NSNumber(value: B), 1], name: nil)
      outs.append(hn)
      h = mix(m, h, hn)
    }
    final.append(h)
    let seq = g.concatTensors(outs.map { g.reshape($0, shape: [NSNumber(value: B), 1, NSNumber(value: D)], name: nil) }, dimension: 1, name: nil)
    let xm = g.addition(x, seq, name: nil)
    x = g.reshape(ffn(g.reshape(xm, shape: [NSNumber(value: B * L), NSNumber(value: D)], name: nil), l), shape: [NSNumber(value: B), NSNumber(value: L), NSNumber(value: D)], name: nil)
  }
  // the contexts: one step each from their decision point's state
  var y = dense(tokens, "inp", "binp")
  for l in 0..<2 {
    let hp = g.gather(withUpdatesTensor: final[l], indicesTensor: index, axis: 0, batchDimensions: 0, name: nil)
    let hn: MPSGraphTensor
    if lay.full { hn = cell(y, hp, l) }
    else { hn = mix(g.sigmoid(with: dense(y, "z\(l)", "bz\(l)"), name: nil), hp, g.tanh(with: dense(y, "h\(l)", "bh\(l)"), name: nil)) }
    y = ffn(g.addition(y, hn, name: nil), l)
  }
  let n = g.multiplication(y, g.reverseSquareRoot(with: g.addition(g.mean(of: g.square(with: y, name: nil), axes: [1], name: nil), g.constant(1e-6, dataType: .float32), name: nil), name: nil), name: nil)
  return dense(n, "out", nil)
}

// the forward alone, for one shape of batch (decision points, rows), built once and reused
final class Policy {
  let g = MPSGraph(), theta: MPSGraphTensor, lanes: MPSGraphTensor, mask: MPSGraphTensor, tokens: MPSGraphTensor, index: MPSGraphTensor, w: MPSGraphTensor
  let B: Int, R: Int
  init(_ lay: Layout, decisions B: Int, rows R: Int) {
    self.B = B; self.R = R
    theta = g.placeholder(shape: [NSNumber(value: lay.count)], dataType: .float32, name: "theta")
    lanes = g.placeholder(shape: [NSNumber(value: B), NSNumber(value: L), NSNumber(value: IN)], dataType: .float32, name: "lanes")
    mask = g.placeholder(shape: [NSNumber(value: B), NSNumber(value: L)], dataType: .float32, name: "mask")
    tokens = g.placeholder(shape: [NSNumber(value: R), NSNumber(value: IN)], dataType: .float32, name: "tokens")
    index = g.placeholder(shape: [NSNumber(value: R)], dataType: .int32, name: "index")
    w = forward(g, lay, theta: theta, lanes: lanes, mask: mask, tokens: tokens, index: index, decisions: B, rows: R)
  }
}
