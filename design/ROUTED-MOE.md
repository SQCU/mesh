# Routed experts

The current gate, sparse SwiGLU block, validity masks, ordinary load-balancing
loss and derivative ownership are specified in
[POLICY-PROGRAM.md](POLICY-PROGRAM.md). All experts are routed; the operator
explicitly excluded shared experts.

The relevant primary precedents are
[DeepSeek-V3 equations 13–15 and 17–20](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)
and its official
[gate and expert implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py).
They support normalized selected sigmoid affinities and the frequency-times-router
probability auxiliary objective. The project uses the operator-requested ordinary
balancing loss and does not inherit shared experts or load-dependent routing bias.

Removed tests and earlier RPC measurements do not establish an alternative MoE
specification. [FEATURE-GRAM-PROVENANCE.md](FEATURE-GRAM-PROVENANCE.md) records why
the former feature-Gram/probe branch was deleted.
