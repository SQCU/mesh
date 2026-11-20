(* The fabrics' summary, in OCaml, over memory the caller owns (a numpy array, a cffi array, bytes): `ints` and `dbls`
   are the same bytes as int32s and float64s (external Bigarrays, no copy), `off` each field's index in them (from C's
   offsetof: fabric_stubs.c), `out` the caller's [n][4] float64s. Per fabric: its lanes, its compute nodes, the least
   ingress rate of a compute node (bytes a ns, all its in-lanes), and the mean lane latency (ns). *)
open Bigarray

let fabric_stats (ints : (int32, int32_elt, c_layout) Array1.t) (dbls : (float, float64_elt, c_layout) Array1.t)
    (off : int array) (out : (float, float64_elt, c_layout) Array2.t) =
  let i32 k = Int32.to_int ints.{k} in
  let o_nodes = off.(0) and o_ncompute = off.(1) and o_lanes = off.(2) and o_compute = off.(3) and o_to = off.(4)
  and o_rate = off.(5) and o_latency = off.(6) and stride_i = off.(7) and stride_d = off.(8) in
  for f = 0 to Array2.dim1 out - 1 do
    let bi = f * stride_i and bd = f * stride_d in
    let lanes = i32 (bi + o_lanes) and nc = i32 (bi + o_ncompute) in
    let ingress = Array.make (i32 (bi + o_nodes)) 0.0 and latency = ref 0.0 in
    for l = 0 to lanes - 1 do
      let v = i32 (bi + o_to + l) in
      ingress.(v) <- ingress.(v) +. dbls.{bd + o_rate + l};
      latency := !latency +. dbls.{bd + o_latency + l}
    done;
    let least = ref infinity in
    for k = 0 to nc - 1 do least := Float.min !least ingress.(i32 (bi + o_compute + k)) done;
    out.{f, 0} <- float lanes; out.{f, 1} <- float nc; out.{f, 2} <- !least;
    out.{f, 3} <- (if lanes > 0 then !latency /. float lanes else 0.0)
  done

let () = Callback.register "fabric_stats" fabric_stats
