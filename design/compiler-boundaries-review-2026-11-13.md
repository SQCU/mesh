# Memorandum: three compiler-boundary findings from the 2026-11-13 source review

Style guide: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights),
for clear language, descriptive links, and code formatting.

## The request and its evidentiary limit

The operator, 2026-11-13, requesting this memorandum about the preceding review:

> we need this written up as a memorandum somewhere btw, important find

In the same turn:

> if these are claude-written documents that append over time and grow more claims, they might be incorrect citations where they 'say' that the codebase does something, holding stale state or stale properties indefinitely (as there is no mechanism for these static documents to remain representations of what code is or what the code should be!) perversely, the schedule emit code is a 'safe' citation (because it's runnable code, and if someone mangles it, rendering it unrunnable, there is a compact deduction available by running the code and noticing it erroring.) but internal prose-claims within a reposotiry, especially `is`-claims, are never safe to pass along, mention in autocompact messages, cite, etc. a claim that's true about code can only remain true about code by being the result of an end to end run of that code through its sanctioned calling paths and sanctioned input domain, (which tragically makes unit tests just as dangerous and untrustworthy as self-referential inner citations).

Codex read the source on 2026-11-13 at mesh `1afc3686979bf709690d1d95ac344e8e131c6ff6` and schedule-emit
`f41c64a`. The mesh files cited below had no changes relative to that commit. The unrelated working-tree edit to
`compiler/mesh_compiler/placement.py` was excluded. Line numbers below identify that reading, not future revisions.

This memorandum records source excerpts and the review concerns inferred from them. No end-to-end compiler or
runtime execution was performed for these findings. Reachability on sanctioned inputs, numerical behavior, runtime
cost, coverage, and completion were not established. This record supplies locations to examine, not a replacement
specification or a statement of what later code does. A later implementation claim needs evidence from that code's
sanctioned calling paths and input domain, as the quoted instruction requires.

## The comparison that prompted the findings

Reference source: `~/schedule-emit/schedule_emit/driver.py:14-21`, `ir.py:57-76`, and `runtime.py:27-75`, at `f41c64a`.
Source under review: [driver.py](../compiler/mesh_compiler/driver.py), `compile_module`, lines 14-30, at the mesh
revision above. The reading found the calls `export`, `place`, `on_devices`, `schedule`, and `argument_buffers`,
followed by construction of `Scheduled`. The three findings below concern boundaries beneath that composition;
the sequence of names alone was not treated as evidence that the complete implementation satisfied the assignment.

## 1. Access discovery invokes expression emission

Source: [lower_metal.py](../compiler/mesh_compiler/lower_metal.py), `Fused.read` at 183-185,
`Bound.__call__` at 247-251, and `Lowering.reads`, `fit`, and `settle` at 461-488.

The inspected `reads` body was:

```python
def reads(self, form):
    bind, index = Bound(), ["0"] * len(form.shape)
    if isinstance(form, Product):
        bind(form.left), bind(form.right), form.expression(bind, index, "t")
    else:
        form.read(bind, index)
    return bind
```

The inspected `Bound.__call__` appended `(view, write)` to `self.accesses` and returned a buffer-name string.
`Fused.read` invoked `self.expression`. `fit` used `len(self.reads(...))`; `settle` used
`self.reads(...).accesses` to construct operands for `mesh_graph.dependences_views`.

Review finding: these bodies coupled access discovery to invocation of expression-emission callables with dummy
indices. The generated expression was discarded by `reads`; the accumulated mutations to `Bound` were consumed.
This was not a finding of text reparsing. It was a finding that a compiler analysis obtained its operands through
the behavior of the emitter rather than through explicit access data carried into lowering.

The proposed revision target was that producer-to-consumer connection: retain the read/write views as data for
`fit` and `settle`, then emit from the resulting representation. This review did not establish a numerical failure
or the runtime cost of the existing path.

## 2. Native emission retains calls into Python

Sources: [lower_metal.py](../compiler/mesh_compiler/lower_metal.py), `Lowering.function` at 1191-1223;
[mesh_metal4.py](../rdma/mesh_metal4.py), `program` at 103-122 and `Function` at 133-196;
[lower_coreml.py](../compiler/mesh_compiler/lower_coreml.py), `Lowering.function` at 246-310.

The inspected emitter included:

```python
if kind is SENT:
    body += [f"host[0]({given[0]});", "if (f->failed) {", *lines, *ended, "return; }"]
elif kind in (RECEIVED, COPIED):
    k, storage, outstanding = given
    body += [f"a[{storage}] = host[{1 if kind is RECEIVED else 2}]({k});", "if (f->failed) {", *outstanding,
             *ended, "return; }", *lines]
```

The inspected binding used `HOSTED = C.CFUNCTYPE(Q, U)` and:

```python
self.hosted = (HOSTED * 3)(HOSTED(self.sent), HOSTED(self.received), HOSTED(self.copied))
```

`Function.__call__` passed `self.hosted` to the native call. The three Python methods resolved the recorded
send, receive, or copy operands and invoked their operators. `Lowering.function` supplied send and receive
records; the Core ML emitter also supplied copy records. The source of [engines.py](../examples/engines.py),
lines 111-115, connected `compile_module`, those lowerers, `Program`, and a program call; that client was not run
in this review.

Review finding: the native-program construction retained a Python callback boundary for those operations.
The existence of a compiled outer function did not, by itself, establish that communication and copies had been
compiled through to their final consumers. The proposed revision target was that boundary. No callback count,
latency, or result for an executed program was measured here.

## 3. Scheduling acquires configuration inside the transformation

Sources: [driver.py](../compiler/mesh_compiler/driver.py), `compile_module` at 14-30;
[schedule.py](../compiler/mesh_compiler/schedule.py), `image` at 107-121, `solved` at 148-163,
`planned` at 216-256, and `schedule` at 262-276.

The inspected call chain was:

```text
compile_module -> schedule -> planned -> image
                                    -> solved
```

`schedule` selected `planned` when its filtered `carried` list was nonempty. In the inspected bodies, `image`
called `mesh_attach`, copied bridge-region metadata into a private mapping, and called `mesh_detach`.
`planned` read nodes through that context and read `os.environ["MESH_PROGRAM"]`; `solved` read
`os.environ["MESH_GROUP"]` and the monotonic clock before calling `mesh_plan`.

Review finding: the declared scheduling arguments did not account for all the configuration acquired by this
path. The proposed revision target was the boundary between acquiring that configuration and transforming a graph
with it as an operand. Acquiring a machine model was not itself the objection. This source reading did not
establish what a live bridge, a particular environment, or any sanctioned distributed input would produce.
