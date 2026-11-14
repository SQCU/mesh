# Bound invocation simplification

SAME593 (59368744-da56-491a-871a-1be809c0cc93, claude-opus-5-5), source HOLD. Only Bound.__init__/invoked changed; native C/API and peer paging hunks are separate.

MetalProgramIssued now supplies the invocation count directly. The Python self.executions mirror, ended() helper and completion polling are removed. Existing MetalProgramEnd owns waiting, cancellation and notification draining. The original run/CHECK error is captured before End can overwrite the message buffer and remains primary. Arguments remain held through End; native submitted-buffer ownership and pre-entry no-consumption semantics are preserved.

Supervisor read metal_program.m1888–1917/1972–2027, and independently passed AST and pyflakes. Exact staged source is +13/−17 raw lines; nonblank noncomment lines in the two methods fall36→32. This removes one mirrored state field and one runtime polling implementation; it adds no helper, API or backend. No documentation relocation is counted as source reduction.

The ordinary tall batched Least public caller passed compile, native archive reload and invocation on this source, exit0/stderr0. The existing invalid eig→valid same Program observation is next for the error/recovery path. Exact scope/counts are in integration-native-bound-source.json; numerical evidence is retained separately.
