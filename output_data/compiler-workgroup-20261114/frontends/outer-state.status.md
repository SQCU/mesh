# Outer captured tensor state

SAME974 sessions 63–65 terminated successfully and are held. Source review and
AST checks of export.py, ir.py and runtime.py pass; exact-path diff checks pass.
The public native caller remains reserved after executor's receive build.

`Captured.get_attr` obtains the actual tensor and converts it through the capture's
one FakeTensorMode. `capture` lifts it to the existing state-placeholder owner,
retaining its original parameter/buffer role and real tensor. Existing holders
are reused; distinct views retain shared storage. The first graph node stays
alive through state/token insertion and the existing final cleanup removes dead
attributes. User-input ordering and function-local output roles are unchanged.

The consumer is already connected: CallOperands uses original placeholder
storage/views, native binding keeps the original input ordinal and view delta,
and runtime.Arguments binds parameter/buffer handles on every invocation. Fixed
identity and geometry do not freeze tensor contents or create per-call copies.

The bounded rejection-filter review covered export.py, ir.py, runtime.py and
capture validation. One silent default changed: unknown scheduled input names no
longer become None through state.get; indexing the existing declared binding
domains raises KeyError. Known live/user inputs still bind during prepare.
Requested comments explicitly prohibit rejection filters even in temporary
integration validation. Existing compile exceptions propagate unchanged.

Alias queries returning None, viewable's contiguous-copy choice, intermediate
mutation handling, selected-machine input projection, and non-output participant
returns are legitimate semantics and were preserved. No new gate framework,
source reevaluation, alias solver, dependency patch or runtime regime was added.

The one ordinary caller is recorded in metal workgroup
frontends/torch/outer-state-native-command.md: two known calls share outer
noncontiguous buffer views and a parameter, mutate real state, return an alias,
and reuse one Program after changing parameter contents. No target result is
claimed by this source checkpoint.
