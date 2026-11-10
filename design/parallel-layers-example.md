# A language model with parallel layers, as a demonstration program

`examples/pl.py`. Inference of a randomly initialized language model whose every block computes attention and the
feed-forward network from one normalized input and adds them. It is run as `examples/tp.py`, `cp.py` and `ep.py` are:
on each node, under that node's bridge, with `--nodes`.

## The turn

> i'm rather disappointed, personally, taht the example dnn used in this repo is one of the boring serially bound one. go find one of the PaLM-like language model implementations that explicitly parallelize attention and ffn like the old pythia implementations (you should be able to find one of these in ~/attn-demo or a repo named suspiciously similar to that) and instate the 'inference' of a random init model of this architecutre as one of the demonstration programs operated through whatever mechanisms we are using for runtime testing of our library. this should more directly expose the nonsuperficiality of the async scheduling properties here for the purposes of our aot compilation and type system, and more directly attach the literature references to something that must be attained for a certain property to hold at the compilation and runtime of certain data processing functions.

(The operator, 2026-11-10 UTC, session 55974018.)

## Sources

- `~/dff_sparsify/legacy/pgpt/pgptlformer.py` (commit e82ef5e; `~/attn_demo/readme.md` says the model moved there),
  class `vit22_tformer`. Its `forward`: "#parallel forward from kingoflolz/mesh-transformer-jax/! check it out!!",
  `inner_states = self.layerwisenorm(h_states)`, `attn_out = self.self_attn(inner_states, attention_mask)`,
  `dense_out = self.feedfor(inner_states)`, `out = skip_out + dense_out + attn_out`. The example takes from it the
  block, the rotary embedding, the gated feed-forward network, the weighted skip, and `PGPT_Lformer.forward`'s branch
  without targets (the last position's logits, `30 * torch.tanh(logits / 30)`).
- B. Wang and A. Komatsuzaki, mesh-transformer-jax, `mesh_transformer/layers.py`, `TransformerLayerShard`, read
  2026-11-09: "We actually combine the FF and dense in one layer (i.e. compute in parallel) to minimize all reduces."
  Its `__call__`: `attn_out = self.self_attn(q, v, k, bias)`, `dense_out = self.ff(x)`,
  `return g_psum(attn_out + dense_out)`.
- A. Chowdhery et al., PaLM, arXiv:2204.02311, section 2, "Parallel Layers", read 2026-11-09 through ar5iv: the
  block is y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x)), citing Wang and Komatsuzaki 2021, with "roughly 15%
  faster training speed at large scales" because the feed-forward and attention input matrix multiplications can be
  fused. (Only that fragment is quoted; the rest of the sentence is paraphrased here.)
- M. Shoeybi et al., Megatron-LM, arXiv:1909.08053, Figure 3, for the placement's splits, as `examples/tp.py` uses.

## Where it differs from the source file

- The source applies the feed-forward network's two input projections as one matrix and splits the result in half
  (`fused_denseproj_in`, `chunk(2, dim=-1)`). The example writes two projections, `denseproj_in` and
  `denseproj_gate`. The function is the same. With one fused matrix split across nodes by its rows, the two halves of
  one hidden unit would lie on different nodes, and this placement operand has no split that says otherwise.
- The causal mask is `is_causal=True`, the form the source uses in its other branch, in place of a mask tensor.
- The rotary tables are made for the compiled sequence length when the module is built, in float32. The source makes
  them on first use, in bfloat16.
- No cross-attention, no second attention, no key and value return, no query and key normalization (the source's
  `identitynorm` setting).

## What this program makes plain

In a serial block the feed-forward network reads the attention's result, so every function of a layer has one
predecessor and the order of a layer is fixed by the program. Here it is not:

1. **Attention and the feed-forward network of a layer do not depend on each other.** Neither reads a value the other
   writes. In the exported graph of `examples/pl.py`, no node of `blocks.0.feed_forward` has a node of
   `blocks.0.attention` among its ancestors, nor the reverse. So the calls' graph has no order between them
   (design/aot-compilation-change-request.md A10: "the order constraints between calls"; A. J. Bernstein, 1966), and
   the one solve of design/implementation-order-2026-11-09.md item 4 is free to run them at once.
2. **The block's sum takes its operands in any order.** `h_states * lambda + dense_out + attn_out` is a sum, whose ⊕
   is commutative and associative. It is a function of plan-to-bridge-change-request.md R4's second kind, and R4's
   "Behaviour at run time", statement 2, is what a run must do with it: "Each task starts when its own page is
   present. It does not wait for another page of the same operand."
3. **One all-reduce a layer, or two that do not wait on each other.** Under Megatron's splits, each branch's output
   projection leaves partial sums on every node. mesh-transformer-jax adds the two branches' partial sums and reduces
   once, "to minimize all reduces". The torch source here does not say that; it says
   `attnoutproj(...) + denseproj_out(...)`, each with a bias. A compiler reaches the single reduction only by using
   that the reduction and the block's sum commute. No quoted source says this compiler does so, and nothing here
   requires it. What is required is the lesser thing: if there are two reductions, neither waits for the other
   (statement 1), and their transfers are ordered by the same program as everything else (R4, statement 3).
4. **The input projections of both branches read one value.** PaLM gives this as the source of its speed. Whether the
   four input matrix products of a layer are fused is the compiler's
   (compilation-composition-change-request.md, "Performance correction: compile the transformations, not their
   textual symptoms"); this program is one where it can be seen.

## Status, 2026-11-09

- Run in process, small sizes (DIM 64, 4 heads, 2 layers, sequence 16, vocabulary 128): the eager model gives finite
  logits of shape (2, 1, 128); a block equals `x * lambda + ffn(norm(x)) + attn(norm(x))`; `torch.export` succeeds;
  `mesh_compile._static` passes for the CPU and the GPU; all 17 operators of the decomposed graph have a lowering;
  the ancestor test of statement 1 holds.
- Not run through the mesh: no bridge is up on this machine, and `mesh.program` stops at attaching to one. Nothing of
  statements 1 to 4 has been observed in a compiled or running program. Which record of a run shows them is the
  outstanding question R4 already records.
- The placement is untested: in particular what the compiler does with the output projections' biases, which are not
  split.
- In the implementation order this program belongs with item 11, as a program a run is made of.
