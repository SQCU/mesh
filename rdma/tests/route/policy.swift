// The forward (route.h): two recurrent layers on a 64-wide residual stream, each a recurrence and a feed-forward block
// expanding to 192. At each refresh the layers read the lane tokens in turn from the stream's carried state; then one
// step more for each context's token from the state they left; then w = out . rms(output) for each context. The
// parameters are one flat vector theta, sliced here into the blocks the Layout names.
//   GRU:    z = sigmoid(Wz [h; x] + bz), r = sigmoid(Wr [h; x] + br), h~ = tanh(Wh [r h; x] + bh)
//   minGRU: z = sigmoid(Wz x + bz), h~ = tanh(Wh x + bh) (gates from the input: a refresh's gates are one matrix
//           product over its positions, the recurrence an elementwise scan)
//   h' = (1 - z) h + z h~; x' = x + h'; out = x' + down relu(up x' + bu) + bd. A padded lane leaves the state as it was.
import Foundation
import MetalPerformanceShadersGraph

let D = Int(ROUTE_D), E = Int(ROUTE_E), IN = Int(ROUTE_IN), FS = Int(ROUTE_FS), L = Int(ROUTE_L), TASKS = Int(ROUTE_TASKS)
func dims(_ d: [Int]) -> [NSNumber] { d.map { NSNumber(value: $0) } }

// the parameter blocks, in theta's order: (name, shape, offset); forward.c's route_params_bind reads the same order
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
  func initial(_ g: inout route_rng) -> [Float] {   // weights at 1/sqrt(fan-in), biases 0
    var t = [Float](repeating: 0, count: count)
    for (_, shape, offset) in blocks where shape.count == 2 {
      for i in 0..<(shape[0] * shape[1]) {
        let u = max(route_uniform(&g), 1e-300), v = route_uniform(&g)
        t[offset + i] = Float(sqrt(-2 * log(u)) * cos(2 * Double.pi * v) / sqrt(Double(shape[1])))
      }
    }
    return t
  }
}

// the layers on graph g, their weights sliced from theta
struct Layers {
  let g: MPSGraph, lay: Layout, p: [String: MPSGraphTensor], one: MPSGraphTensor
  init(_ g: MPSGraph, _ lay: Layout, theta: MPSGraphTensor) {
    self.g = g; self.lay = lay; one = g.constant(1, dataType: .float32)
    var p: [String: MPSGraphTensor] = [:]
    for (name, shape, offset) in lay.blocks { p[name] = g.reshape(g.sliceTensor(theta, dimension: 0, start: offset, length: shape.reduce(1, *), name: nil), shape: dims(shape), name: nil) }
    self.p = p
  }
  func dense(_ x: MPSGraphTensor, _ w: String, _ b: String?) -> MPSGraphTensor {
    let y = g.matrixMultiplication(primary: x, secondary: g.transposeTensor(p[w]!, dimension: 0, withDimension: 1, name: nil), name: nil)
    return b == nil ? y : g.addition(y, p[b!]!, name: nil)
  }
  func mix(_ z: MPSGraphTensor, _ h: MPSGraphTensor, _ c: MPSGraphTensor) -> MPSGraphTensor {   // (1 - z) h + z c
    g.addition(g.multiplication(g.subtraction(one, z, name: nil), h, name: nil), g.multiplication(z, c, name: nil), name: nil)
  }
  func gru(_ x: MPSGraphTensor, _ h: MPSGraphTensor, _ l: Int) -> MPSGraphTensor {
    let hx = g.concatTensors([h, x], dimension: 1, name: nil)
    let z = g.sigmoid(with: dense(hx, "z\(l)", "bz\(l)"), name: nil), r = g.sigmoid(with: dense(hx, "r\(l)", "br\(l)"), name: nil)
    let c = g.tanh(with: dense(g.concatTensors([g.multiplication(r, h, name: nil), x], dimension: 1, name: nil), "h\(l)", "bh\(l)"), name: nil)
    return mix(z, h, c)
  }
  func ffn(_ xm: MPSGraphTensor, _ l: Int) -> MPSGraphTensor { g.addition(xm, dense(g.reLU(with: dense(xm, "up\(l)", "bu\(l)"), name: nil), "down\(l)", "bd\(l)"), name: nil) }
  // one refresh's lanes [B, L, IN], mask [B, L], from states h[l] [B, D]: the states they leave
  func lanes(_ lanes: MPSGraphTensor, _ mask: MPSGraphTensor, _ h0: [MPSGraphTensor], B: Int) -> [MPSGraphTensor] {
    var x = g.reshape(dense(g.reshape(lanes, shape: dims([B * L, IN]), name: nil), "inp", "binp"), shape: dims([B, L, D]), name: nil)
    var final: [MPSGraphTensor] = []
    for l in 0..<2 {
      var gz: MPSGraphTensor? = nil, gc: MPSGraphTensor? = nil
      if !lay.full {
        let flat = g.reshape(x, shape: dims([B * L, D]), name: nil)
        gz = g.reshape(g.sigmoid(with: dense(flat, "z\(l)", "bz\(l)"), name: nil), shape: dims([B, L, D]), name: nil)
        gc = g.reshape(g.tanh(with: dense(flat, "h\(l)", "bh\(l)"), name: nil), shape: dims([B, L, D]), name: nil)
      }
      if !lay.full {
        // minGRU's recurrence in its parallel form: with a_t = 1 - m_t z_t and b_t = m_t z_t h~_t (a padded lane: a = 1, b = 0),
        // h_t = exp(C_t) h_0 + sum_(s<=t) exp(C_t - C_s) b_s, C_t = sum_(r<=t) log a_r: a lower-triangular decay matrix a
        // channel times b, every exponent <= 0
        let m = g.reshape(mask, shape: dims([B, L, 1]), name: nil)
        let a = g.subtraction(one, g.multiplication(m, gz!, name: nil), name: nil), b = g.multiplication(g.multiplication(m, gz!, name: nil), gc!, name: nil)
        var ones = [Float](repeating: 0, count: L * L); for t in 0..<L { for q in 0...t { ones[t * L + q] = 1 } }   // the running sum as a product (MPSGraph has no cumulativeSum gradient)
        let lower = g.constant(Data(bytes: ones, count: L * L * 4), shape: dims([1, L, L]), dataType: .float32)
        let C = g.matrixMultiplication(primary: lower, secondary: g.logarithm(with: g.maximum(a, g.constant(1e-30, dataType: .float32), name: nil), name: nil), name: nil)   // [B, L, D]
        let Ct = g.transposeTensor(C, dimension: 1, withDimension: 2, name: nil)   // [B, D, L]
        let diff = g.subtraction(g.reshape(Ct, shape: dims([B, D, L, 1]), name: nil), g.reshape(Ct, shape: dims([B, D, 1, L]), name: nil), name: nil)   // C_t - C_s
        var tri = [Float](repeating: -Float.infinity, count: L * L); for t in 0..<L { for q in 0...t { tri[t * L + q] = 0 } }
        let causal = g.constant(Data(bytes: tri, count: L * L * 4), shape: dims([1, 1, L, L]), dataType: .float32)
        let M = g.exponent(with: g.addition(g.minimum(diff, g.constant(0, dataType: .float32), name: nil), causal, name: nil), name: nil)   // [B, D, L, L]
        let bt = g.reshape(g.transposeTensor(b, dimension: 1, withDimension: 2, name: nil), shape: dims([B, D, L, 1]), name: nil)
        let hs = g.reshape(g.matrixMultiplication(primary: M, secondary: bt, name: nil), shape: dims([B, D, L]), name: nil)
        let all = g.addition(g.transposeTensor(hs, dimension: 1, withDimension: 2, name: nil),
                             g.multiplication(g.exponent(with: C, name: nil), g.reshape(h0[l], shape: dims([B, 1, D]), name: nil), name: nil), name: nil)   // [B, L, D]
        final.append(g.reshape(g.sliceTensor(all, dimension: 1, start: L - 1, length: 1, name: nil), shape: dims([B, D]), name: nil))
        let xm = g.addition(x, all, name: nil)
        x = g.reshape(ffn(g.reshape(xm, shape: dims([B * L, D]), name: nil), l), shape: dims([B, L, D]), name: nil)
        continue
      }
      var h = h0[l], outs: [MPSGraphTensor] = []
      for t in 0..<L {
        func at(_ y: MPSGraphTensor) -> MPSGraphTensor { g.reshape(g.sliceTensor(y, dimension: 1, start: t, length: 1, name: nil), shape: dims([B, D]), name: nil) }
        let hn = lay.full ? gru(at(x), h, l) : mix(at(gz!), h, at(gc!))
        let m = g.reshape(g.sliceTensor(mask, dimension: 1, start: t, length: 1, name: nil), shape: dims([B, 1]), name: nil)
        outs.append(g.reshape(hn, shape: dims([B, 1, D]), name: nil))
        h = mix(m, h, hn)
      }
      final.append(h)
      // the outputs at every position, concatenated (MPSGraph's stack gave layer 0 a wrong gradient here)
      let xm = g.addition(x, g.concatTensors(outs, dimension: 1, name: nil), name: nil)
      x = g.reshape(ffn(g.reshape(xm, shape: dims([B * L, D]), name: nil), l), shape: dims([B, L, D]), name: nil)
    }
    return final
  }
  // the contexts' tokens [R, IN], each one step from row index[r] of states[l] [*, D]: w [R, FS]
  func contexts(_ tokens: MPSGraphTensor, _ index: MPSGraphTensor, _ states: [MPSGraphTensor]) -> MPSGraphTensor {
    var y = dense(tokens, "inp", "binp")
    for l in 0..<2 {
      let hp = g.gather(withUpdatesTensor: states[l], indicesTensor: index, axis: 0, batchDimensions: 0, name: nil)
      let hn = lay.full ? gru(y, hp, l) : mix(g.sigmoid(with: dense(y, "z\(l)", "bz\(l)"), name: nil), hp, g.tanh(with: dense(y, "h\(l)", "bh\(l)"), name: nil))
      y = ffn(g.addition(y, hn, name: nil), l)
    }
    let n = g.multiplication(y, g.reverseSquareRoot(with: g.addition(g.mean(of: g.square(with: y, name: nil), axes: [1], name: nil), g.constant(1e-6, dataType: .float32), name: nil), name: nil), name: nil)
    return dense(n, "out", nil)
  }
}
// a state [B, 2, D] as its two layers [B, D], and back
func layers(_ g: MPSGraph, _ s: MPSGraphTensor, B: Int) -> [MPSGraphTensor] { (0..<2).map { g.reshape(g.sliceTensor(s, dimension: 1, start: $0, length: 1, name: nil), shape: dims([B, D]), name: nil) } }
func joined(_ g: MPSGraph, _ h: [MPSGraphTensor], B: Int) -> MPSGraphTensor { g.concatTensors(h.map { g.reshape($0, shape: dims([B, 1, D]), name: nil) }, dimension: 1, name: nil) }

// one refresh for B streams (their states carried in and out), R context rows: built once a shape
final class Policy {
  let g = MPSGraph(), theta: MPSGraphTensor, lanes: MPSGraphTensor, mask: MPSGraphTensor, state: MPSGraphTensor, tokens: MPSGraphTensor, index: MPSGraphTensor
  let w: MPSGraphTensor, next: MPSGraphTensor
  init(_ lay: Layout, decisions B: Int, rows R: Int) {
    theta = g.placeholder(shape: dims([lay.count]), dataType: .float32, name: "theta")
    lanes = g.placeholder(shape: dims([B, L, IN]), dataType: .float32, name: "lanes")
    mask = g.placeholder(shape: dims([B, L]), dataType: .float32, name: "mask")
    state = g.placeholder(shape: dims([B, 2, D]), dataType: .float32, name: "state")
    tokens = g.placeholder(shape: dims([R, IN]), dataType: .float32, name: "tokens")
    index = g.placeholder(shape: dims([R]), dataType: .int32, name: "index")
    let net = Layers(g, lay, theta: theta)
    let h = net.lanes(lanes, mask, layers(g, state, B: B), B: B)
    w = net.contexts(tokens, index, h); next = joined(g, h, B: B)
  }
}
