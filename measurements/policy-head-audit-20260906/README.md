# Policy/value head audit — September 6, 2026

The [ranked review](../../design/POLICY-HEAD-AUDIT.md) separates confirmed
common-IR violations, inherited output-loss assumptions, measured capacity
limits and previously authorized auxiliary tasks. These artifacts describe the
audited matrix-policy version 19. They do not claim that the replacement heads
have been implemented.

- `numerics.json`: frozen-IR counterexamples, isolated actor/main-critic parameter
  gradients, the default mean-readout rank and the Gaussian entropy-floor scale.
- `provenance.json`: selected original user and assistant messages, including the
  explicit early query-value imitation request omitted from the previous review.
- `source-sha256.json`: identities of the audited source, not deployment pins.

The operator subsequently removed the reproduction harness with the repository's
test suites. The retained measurements record that implementation; they do not
define the replacement policy specification.

All five recorded boundary counterexamples reproduced. Raw-state changes after
the frozen IR changed policy means while leaving both main values unchanged.
The default 256-by-128 mean-readout Jacobian measured rank 128. The one-nat
Gaussian entropy penalty has a zero-penalty scale of 0.6577446 native rate units,
65.77 times the scale at a zero readout coordinate. These are mathematical and
data-flow findings; they do not establish a particular native fault's cause or
playing strength. The probes operated no engine, service, bridge or remote node.
