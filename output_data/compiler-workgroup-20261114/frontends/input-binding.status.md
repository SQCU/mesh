# Original input binding

SAME974 sessions 66–68 are terminal and held. Supervisor AST checks of export.py,
ir.py, runtime.py and archive.py and exact-path diff checks pass. No native caller
has run for this change; executor's matching source/import window remains held.

The lost operand was the outer call interface: runtime flattened caller keyword
insertion order directly into compiled user slots. Exported now retains
`interface = (inspect.Signature, TreeSpec)`, produced once from the actual
BoundArguments before per-call interpretation can reuse local names.

`calls` uses Signature.bind and apply_defaults before FX tracing: FX emits slots
for the actual bound/defaulted parameters. `called` retains its explicit outer
slots; omitted defaults are specialized inside that leaf export. The Signature
retains original Parameter objects, including kinds and defaults for compiled
slots. The TreeSpec contains those bound values in parameter order; a variadic
parameter remains its actual tuple. No prefix-length/name heuristic remains.

Arguments.prepare calls that Signature's bind and apply_defaults, then the
retained TreeSpec.flatten_up_to. This canonical owner selects nested dict keys in
the compiled order and owns structure errors. No new parser, recursive walker,
fresh-spec mismatch branch, compatibility/default mode or source replay remains.
Canonical missing/duplicate/unknown parameter errors propagate unchanged.

Archive saves state and required default objects together in one torch.save
payload, preserving shared tensor storage. JSON retains parameter names/kinds and
treespec_dumps(spec). weights_only load reconstructs the same Signature/spec;
there is no callable pickle, textual default parser or legacy-schema fallback.
Internal outlined Exported records need no public interface and are unchanged.

The ordinary public command is in metal workgroup
frontends/torch/input-binding-native-command.md: a two-input noncommutative known
call compiled positionally, invoked with reversed kwargs, then saved/dropped/
restored and invoked with reversed kwargs again. Source-only checkpoint: no
numerical or archive observation claim yet. The earlier tensor-state/alias/RNG
handoff review found this input binding loss; it did not require a new test grid.
