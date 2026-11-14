# Bounded polar Repeat draft: superseded

Source c5df7c6f correctly represented the JAX condition/count with one existing Repeat body, but backend review found added host synchronization: metal_lower.mm1837-1857 and metal_program.h239-250 submit and wait around GPU-produced loop conditions/passes. This draft was not accepted as a performance improvement and its queued native observation was cancelled. numerical-47.status.md records the corrective source; no benchmark comparison was used to justify removing the added synchronization.
