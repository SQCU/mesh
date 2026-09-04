# One policy, one optimization state

Implementation manifest for `codex/policy-collapse`, September 4, 2026. This refines
the execution of the existing policy; it does not introduce a new learning objective.
The controlling policy and loss requirements remain in [SPECIFICATION.md](SPECIFICATION.md)
and [rl-training-spec.md](rl-training-spec.md).

The subsequent operator-requested [joint-policy learning intervention](JOINT-POLICY-LEARNING.md)
adds a terminal-win comparator and an explicit historical value-learning mixture.
"One policy" below means one complete optimization state per genuinely learned policy,
not a restriction to one policy instance in a shared match. Policy version 12 records
that objective/replay change without changing parameter shapes.

The operator's correction:

> an 'expert' is a metaphor used in pop science papers, there was nothing in this repo's specification which implied anyhting different from routed sparse ffn or equivalent as an implemnetation choice for an embedding or residual-over-embedding for a policy solver implemented as linear algebra...

> to me this idea of freezing some parameters and updating others is suspicious and sounds like a completely invented semantic feature or ontological relationship which would not make sense if it was written up in pseudocode as part of a manifest of all of the policy source, optim source, and tensor op source; we'd see it's nonsensical or doesn't relate to policy or optim pseudocode, requirements, etc.

## Source manifest

| Definition | Single source |
|---|---|
| Parameter tree, including routed FFN banks, router, projections and probe | `cast_header.py:Wally` |
| State-to-policy composition, values, dynamics and actuators | `strategy.py`, composed from `cast_header.py` and `dpp.py` |
| Transition objectives and batch reduction | `online.py:OnlineLearner._item_loss`, `learn` |
| Whole-tree differentiation, clipping and AdamW update | `online.py:OnlineLearner.learn`, using MLX's transformations and optimizer |
| Dense/routed matrix products and their operand derivatives | `matmul.py` |
| Residual-feature Gram contraction | `matmul.py:gram_context`, imported by local and remote execution |
| Whole policy/optimizer/replay/RNG checkpoint | `online.py:OnlineLearner.save`, `_load_full`, with `checkpoint_state.py` |
| Tensor framing and loss recovery | `solver/xonwire.py`, shared by requester and worker |
| Remote call and pullback | `strat_responder.py:RemoteGram`, `matrix_worker.py` |
| Match lifecycle and lineage selection | `curriculum.py` |

These filenames are relative to `xonotic/solver/strat` unless otherwise qualified.
The fixed default, FFN and linear study controls remain separate, explicitly specified
comparators; they are not alternative implementations of the matrix-fusion policy.

The optimization manifest is:

```text
theta, optimizer_state, replay, rng, updates = restore_whole_policy_state()
outputs = policy(theta, actual_state)
transmit_current_action(outputs)
loss = mean(transition_objective(theta, item) for item in actual_replay_batch)
gradient = derivative(loss, theta)
gradient = clip_by_global_norm(gradient)
theta, optimizer_state = AdamW(theta, optimizer_state, gradient)
updates += 1
save_whole_policy_state(theta, optimizer_state, replay, rng, updates)
```

There is no host-dependent trainable subset. Routing selects matrix applications;
it does not select independent policies, optimizer histories or checkpoints.
An unused route can have zero derivative, which is not a parameter-freezing rule.
The loss's detached targets/advantages remain explicit parts of the loss definition,
and discrete route indices remain nondifferentiable. Neither licenses freezing a
parameter because its computation executes elsewhere.

## Pure distributed tensor operation

For residual rows `R` of shape `n × d` and the policy's probe `p` of shape `d`:

```text
G = transpose(R) R / n
T = tanh(G)
c = T p
```

For the arriving context cotangent `q`, the pullback is:

```text
dp = transpose(T) q
dG = (q transpose(p)) * (1 - T * T)
dR = R (dG + transpose(dG)) / n
```

The implementation obtains this pullback by differentiating the same `gram_context`
function, not by keeping another handwritten derivative. `RemoteGram` exposes **both**
`R` and `p` as transformation inputs and returns **both** derivatives. The derivative
of `R` continues through the routed FFN, router and input projection; `dp` reaches the
same policy's probe. Gram statistics are detached observational outputs, not loss terms.

The current placement keeps the parameterized FFN and projections with the policy
and executes this Gram contraction and its pullback on the peer. All parameters and
AdamW state therefore have one owner. This changes placement, not the mathematical
policy. It retains peer work during optimization, rather than using remote inference
and silently substituting local-only backpropagation.

| Application operation | Request tensor rows | Response tensor rows |
|---|---|---|
| `GRAM_REQ` / `GRAM_RESP` (15 / 16) | `p`, followed by every row of `R` | one row containing `c` |
| `GRAM_GRAD_REQ` / `GRAM_GRAD_RESP` (17 / 18) | `p`, `q`, followed by every row of `R` | `dp`, followed by every row of `dR` |
| `GRAM_META` (19) | — | minimum/maximum/finite mass of `G`, processed rows, elapsed seconds |

The frame width supplies `d`; the tensor extent supplies `n`. No worker architecture,
checkpoint, learning rate, seed, bank count or top-k configuration is needed. Session
and request identity stay in the 48-byte structural header. For these operations its
tick field distinguishes an action-deadline measurement (1) from background work (0).
The probe is a literal tensor operand, not model identity encoded as floating-point data.
The game frame format and sealed `mesh.h` ABI are unchanged. The new application kinds
require the matching game relay, requester and worker; old kinds are not reinterpreted.

A worker holds reassembly buffers and its most recent completed response, not model
state. It can replay that response or recompute a pure request after losing its cache.
Neither can apply an optimizer update twice. Frame loss recovery remains in `xonwire.py`.
Orderly cancellation computes the same local operation/pullback from the supplied
operands; it cannot manufacture a zero input or parameter gradient.

## Checkpoints, measurements and remaining evidence

There is one checkpoint path per learned policy. The removed `checkpoint.scale.npz`
was an overlapping whole-model copy with an independent optimizer, not a valid shard.
Existing files are not deleted or silently merged. Policy version 11 records this
execution/optimization change; old versions remain reported source provenance, not
evidence that an old split checkpoint contains a coherent trained whole policy.

The retained `initial_policy` control and trained policy can use the same stateless
worker with their own operands. Their distribution no longer needs to be disabled.
A local counterfactual retains the actual action call's `R`, `p` and `c`. Training
cannot overwrite it or change its parameter interpretation. Its timing concerns the
Gram contraction; a whole-plan substitution is an estimate, not a separately executed
all-local game plan. Processed input rows and returned context rows are distinct
coordinates: a contraction returns one context row, not `n` residual rows.

At default widths the FFN banks occupy 2 GiB. This design does not transmit them on
each call, but widens the forward activation transfer from 128 to 2048 coordinates per
residual row. End-to-end bandwidth, action deadline and behavioral benefit must be
remeasured at actual workload extents. Source reduction is not evidence of a speedup.

New conditionals concern tensor/result shape, operation framing, socket ownership,
response replay and measurement cadence. They either preserve the complete operation,
continue serving/retrying, or choose exact local evaluation during orderly cancellation.
They do not restrict a node class or truncate a requested tensor. No live bridge,
installer, firewall, power policy or game service is changed by this source refactor.
