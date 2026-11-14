# Distributed outer output ownership: status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`, at mesh HEAD `3f0555c5`.
**SOURCE-READY, HOLD.** `native.py` only. No target run, SSH, build, staging or commit.

## Cause (source)

- **What the driver passes:** `driver.py:95` constructs a group's body with `output_roles=roles` only where the group
  is its function's output processor's group, else `()`.
- **What construction then makes:** `lower_metal.construct` (`:275`) zips the body's output leaves with those roles.
  A non-output group's body has none, so its `ConstructedMachine.outputs`, and `Body.outputs`, are empty.
- **What `resolved` did:** it built the outer returned outputs on every node. For a leaf produced by a call whose
  function has a numerical body here, it read `body.outputs[o]`, which raised `IndexError` on the four non-output
  nodes. A local body does not mean this node owns the outer output.

## Producer → consumer (the existing operand, carried directly)

- **The ownership producer:** `schedule.analyse` gives `CallAnalysis.output_machine = machines[output_processor]`:
  - `schedule.py:502` declares the field, and the `CallAnalysis` construction sets it;
  - it is the outer output node's processor's machine, in the driver's processor→machine domain (`driver.py:59`).
- **This node's machine:** `native.compile_program` already computes `here = planning.nodes[node]` under planning, else
  `node` (`native.py:1378`). That is the physical machine, the same domain as the driver's entry choice
  (`driver.py:79`). It is not `Target.node`, which is the plan's map index under planning.
- **The runtime's own rule:** `runtime.Program.__call__` (`:49–55`) returns None on an entry other than the output
  processor's, and zips the full `exported.outputs` (strict) only on the output entry.
- **The consumer:** `Target.resolved(captured, analysis, joint, here)`.
  - The outer returned-output recipe (`returned`) is the output leaves and kinds only where `here ==
    analysis.output_machine`, and empty otherwise.
  - On the owner, every position is as before: a land copy's canonical `self.landed` record, a local call's mapped
    `body.outputs[o]`, a caller-owned root form, a placeholder pass-through, or a literal.
- **Unchanged on every node:** operands (inputs, results), `joined` roots, wires, constants, generator operands, every
  row's program and its side effects (writes stay in the native program). Also unchanged: the planner, the copy
  producer, the original output identities and the numerical operations.
- **Nothing added:** no new field, guessed processor, cardinality rule, placeholder output, transfer fallback or
  exception skip.

## Diff and checks

- **The diff (3 code lines plus the comment):**
  - `def resolved(self, captured, analysis, joint, here)`;
  - `returned = zip(output leaves, exported.outputs, strict=True) if here == analysis.output_machine else ()`;
  - `compile_program`'s call passes `here`.
- **Checks:**
  - `py_compile` and `pyflakes` pass, no line is over 120 characters, and `git diff --check` is clean;
  - an AST check finds `resolved` takes (self, captured, analysis, joint, here) and its one call passes 4 arguments.
- **Not addressed:** the output node's separate binding issue (under e48's review).
