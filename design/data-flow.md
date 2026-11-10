# The data flow

The program quoted below is `~/schedule-emit` at f41c64a. Everything under "The program" is its source, quoted by file and line.

## The turns

The operator, in order (UTC):

> "My estimate is that the schedule-and-emit part is a few hundred lines. The basis is that plan_schedule is 140 lines and that MLIR's while lowering is, from memory, about a hundred." scope this out and have a subagent implement this approach in isolation; in a different repo from mesh or metal microbench (not a worktree or something; make something on `~/` for this). as the structural requirements are torch.export (which is part of torch, not this repo, no?), arbitrary sample torch programs should be emittable and runnable without any relationship to the mesh repo's idiom or history in something like that linecount estimate; we can review their work once they submit something.
> " and that MLIR's while lowering is, from memory, about a hundred. "
> then we probably need to review the november 9th workgroup's local goals, whether they were fractally inlining features of their calling context like anything else that has string-typed inputs or outputs, what scope of requirements they were interacting with, if any, while the schedule and emit subagent does whatever they're on

(2026-11-09 20:44)

> " match eager, and keep the same fingerprint. "
> okay cool so time for one more code review and refactor to get rid of extraneous features, behaviors, overhead in linecount, overhead in performance, of the demo IR-er and lowerer pair. remmber that our goal here is to articulate what the data flow will be for the mesh modules that have the same role in their own data flow as this counterfactual example code.
> "  - making received tensors contiguous;" this is the kind of superfluosu feature we have to prematurely stop even in the counterfactual case; contiguity is a scheduled operation that must be justified by a solver; in torch making a tensor contiguous is a *call* which necessarily introduces extra overhead in redundant writes, overconstraint on the placement of literal memory in literal addresses, etcetera. this is also true outside of torch and we already cited a ton of literature on why introducing haphazard arbitrary extra constriants to code is prohibited, idiomatic, etc. arbitrary extra constraitns includes things like guards, syncs, and waits, by the way, if it somehow wasn't discernible.

(2026-11-09 22:19)

> "The sizes are not comparable one to one. The prototype has no page-level transport, no tensor sharding and no kernels of its own. " is it the same data flow and the same type -> type flow as the required compiler? try to answer only this question and the precise questions the user wrote in the previous turns instead of volunteering extra distinctions

(2026-11-09 22:51)

> yes, i do want every issue which would make the ir operations and lowerer a bad example for agents addressed. data flow and responsibility inversions are very serious and have to be addressed even in our examples to make the scope of responsibility for each module unambiguous and to better match the literature. we do not want and especially do nto need or benefit from idiomatic compiler behavior!

(2026-11-09 23:27)

> "It made the cross-device copy an explicit node on the sending processor, before the send, so the scheduler sees it as a function like any other." behaviorally appropriate. okay. we now have a reference program which counterexamples whatever wide-spanning and comprehensive pathologies are in the 'existing code' in the mesh lib and mesh repo. review this transcript: what was the immediate review process or work requirement which prompted this counterfactual IR manipulation and lowerer?

(2026-11-10 00:43)

> document this w/in the repository. moving on: what are the requirements from the outstanding topics in the change request documents whose path to implementation is now clarified by the counterfactual program-over-IR and lowerer? this should give us a definite list of requirements and an order in which they must be implemented, alongside a rough estiamte of the numebr of lines of code each ultimately requires if tackled in the right order and by centralizing responsibilities into modules rather than unextensible arboreal inlined sprawl

(2026-11-10 00:50)

> "1. Inventory the roughly 12,600 lines written on Nov 9 as either composition (binding, composing, scheduling) or authored numerics (LU, QR, SVD, least squares, the 1,726-line software binary64 library). The second kind is what the clause above does not ask for.
> 2. Decide the 20,081-line Dawn/Tint patch on its purpose. It exists to get LiteRT's kernels as typed instructions so mesh can fuse inside them. If the delegate's own linking seam or a separate small launch suffices, the patch is not required." actually neither of these are required sincew e can look at the overall functional flow of the old implementation trapped in mesh and metal microbench. if the data flow has the wrong directions at any point, we don't need to read any part of it to totally replace it, since it has no contents which relate to the requirements for any function which has the wrong causal direction. isn't that cool?

(2026-11-10 00:58)

> for example we have a literal second repository iwth a priced and measured linecount which demonstrates the exact data flow mesh must have. what is that repo and are we using the exact same data flow yet? if we aren't, what's the cope? what's the excuse? who made up a limit saying we have to underperform our own examples to ourselves?

(2026-11-10 20:26)

> let us review. once more. why were any of the change request documents written? why was there a lowerer and data flow example directory made in a separate scope to demonstrate the simplicity of a certain list of requirements? for any change requests outstanding before that separate repo was made, what structural chagnes require the total reappraisal and total rewriting of the change request documents with respect to the strictly available xample? What hapened tothe user's statements that a definite data flow be written exactly as documentation and used to recondition and restate all outstanding requirements?

(2026-11-10 21:02)

## The program

### The types that pass between stages

`schedule_emit/ir.py`, lines 29–43:

```python
@torch.library.custom_op("schedule_emit::send", mutates_args=())
def send(value: Tensor, channel: int) -> None:
    LOCAL.channels[channel].put(value)


@send.register_fake
def send_fake(value, channel):
    return None


@torch.library.custom_op("schedule_emit::recv", mutates_args=())
def recv(channel: int) -> Tensor:
    return LOCAL.channels[channel].get()


```

`schedule_emit/ir.py`, lines 57–76:

```python
@dataclass
class Exported:
    graph_module: torch.fx.GraphModule
    state: dict        # placeholder name -> (state key, tensor) for params/buffers/constants
    tokens: list       # placeholder names of effect tokens
    user_inputs: list  # placeholder names, in flattened call order
    outputs: list      # per graph output: (OutputKind, state key or None)
    out_spec: object


@dataclass
class Scheduled:
    """What the compile stages produce and every later stage consumes."""
    exported: Exported     # call signature and initial state
    modules: dict          # processor -> fx.GraphModule, nodes in scheduled order
    devices: dict          # processor -> torch.device
    output_processor: str  # the processor whose GraphModule holds the results
    channels: int          # channels are numbered 0 .. channels-1
    inputs: dict           # processor -> placeholder targets of its GraphModule, in order
    feedforward: list      # processors whose arguments edge is feedforward (no buffer bound)
```

### The order of the stages

`schedule_emit/driver.py`, lines 14–21:

```python
def compile_module(module, args, processors, placement, cost=unit_cost, **export_options):
    """processors: {name: torch device}; placement: fx node -> processor name; cost: fx node -> float."""
    devices = {name: torch.device(d) for name, d in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    modules = schedule(on_devices(placed, devices), cost)
    inputs, feedforward = argument_buffers(modules, output_processor)
    return Scheduled(exported, modules, devices, output_processor, channels, inputs, feedforward)
```

### Each stage's function

- `schedule_emit/export.py`:17: `def export(module, args, **export_options):`
- `schedule_emit/placement.py`:26: `def place(graph_module, placement):`
- `schedule_emit/devices.py`:22: `def on_devices(graph_module, devices):`
- `schedule_emit/schedule.py`:99: `def schedule(module, cost=unit_cost):`
- `schedule_emit/schedule.py`:136: `def argument_buffers(modules, output_processor):`

### What takes the scheduled value

- `schedule_emit/lower_inductor.py`:96: `def lower(scheduled):`
- `schedule_emit/runtime.py`:28: `def __init__(self, scheduled, functions=None):`
- `schedule_emit/runtime.py`:59: `def __call__(self, *args, **kwargs):`
- `schedule_emit/archive.py`:57: `def save(directory, scheduled):`
- `schedule_emit/archive.py`:98: `def load(directory):`

### What each module imports from the others

- `schedule_emit/export.py`: `from .ir import Exported`
- `schedule_emit/placement.py`: `from .ir import RECV, SEND`
- `schedule_emit/devices.py`: `from .ir import RECV, SEND`
- `schedule_emit/schedule.py`: `from .ir import RECV, SEND, channel_of`
- `schedule_emit/driver.py`: `from .devices import on_devices`, `from .export import export`, `from .ir import Scheduled`, `from .placement import place`, `from .schedule import argument_buffers, schedule, unit_cost`
- `schedule_emit/runtime.py`: `from . import ir`
- `schedule_emit/lower_inductor.py`: `from .ir import RECV, SEND`
- `schedule_emit/archive.py`: `from .ir import Exported, Scheduled`
