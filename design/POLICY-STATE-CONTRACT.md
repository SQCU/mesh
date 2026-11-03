# One policy, one optimization state

The current parameter tree, complete tensor program, common policy/value heads,
losses, placement and continuation are defined in
[POLICY-PROGRAM.md](POLICY-PROGRAM.md). This page preserves the operator
requirements that motivated that ownership boundary:

> an 'expert' is a metaphor used in pop science papers, there was nothing in this repo's specification which implied anyhting different from routed sparse ffn or equivalent as an implemnetation choice for an embedding or residual-over-embedding for a policy solver implemented as linear algebra...

> to me this idea of freezing some parameters and updating others is suspicious and sounds like a completely invented semantic feature or ontological relationship which would not make sense if it was written up in pseudocode as part of a manifest of all of the policy source, optim source, and tensor op source; we'd see it's nonsensical or doesn't relate to policy or optim pseudocode, requirements, etc.

Routing selects sparse FFN computation within the owning policy. Remote execution
returns the full operand derivatives to that policy's optimizer. Placement does
not create a separate policy, checkpoint lineage or trainable subset.

[FEATURE-GRAM-PROVENANCE.md](FEATURE-GRAM-PROVENANCE.md) records the withdrawn
feature-Gram/probe interpretation and its original-message evidence. Historical
measurement records describe their recorded implementations; removed test suites
do not define the current program or completion criteria.
