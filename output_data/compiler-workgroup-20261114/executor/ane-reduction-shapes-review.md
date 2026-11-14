# Runtime MIL reduction source checkpoint

This is the exact SAME e48 continuation from `38f67112`, only `metal_neural.mm` (+205/-16). Existing typed REGION
identity bits, ARGMIN/ARGMAX index identity and ordinary combiner identities now supply empty-domain results. The
nonempty ASSIGN result is its last actual reduction contribution. Runtime coupled reductions retain the same
power-of-two half/half COMBINE tree over actual R, padding only its actual identity tail; FINALIZE and Finish remain
the output consumer. Unknown carried dimensions and compact parallel-axis correspondence are explicit MIL operands.

The worker and supervisor strict Objective-C++ syntax checks passed with the actual Makefile TINT_CFLAGS and
`-Wall -Wextra -Werror`; the supervisor log is empty and diff checks passed. No shared build, target, imports,
evaluator, ABI change or placement change. Original malformed sources remain untouched. No runtime tree, ANE
eligibility, numerical accuracy, reduced physical work or performance claim is made from source checks.

The Branch/Store review confirms that the false branch receives the same incoming per-ValueID prior map, not a true
branch update. It also identifies a material existing gap: a different ValueID aliasing an earlier write's storage
can still bind the original function input. This requires the existing physical storage/layout/offset correspondence
to reach MIL Need/Store before the planned spectral ANE client; a new alias solver or latest-writer guess is not
authorized. Nonempty unsigned extrema remain another explicitly recorded signed-MIL semantic limit.

SAME session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, actual claude-opus-5-5, resume76 ended successfully in 9 turns.
The source and worker are held while compiler co-reviews the change and the next exact alias operand boundary.
