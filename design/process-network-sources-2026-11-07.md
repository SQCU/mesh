# Dataflow process networks, CUDA streams, reduction operators, and PyTorch's lowering hooks: sources, quoted verbatim

Retrieved 2026-11-07. Each block below was copied by script from the cited lines of the file named; nothing in a block is retyped.

## 1. Lee and Parks (1995)

- **Source:** Edward A. Lee and Thomas M. Parks, "Dataflow Process Networks", *Proceedings of the IEEE* 83(5):773–801, May 1995. DOI 10.1109/5.381846 (Crossref).
- **Where read:** the Ptolemy project's copy, `https://ptolemy.berkeley.edu/publications/papers/95/processNets/proceedings.pdf` (linked from `…/processNets/`), saved as `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit8/dpn/dpn.pdf`; text by `pdftotext -layout`, `lit8/dpn/dpn.txt`. Pages are the copy's own ("13 of 63"), from its running heads.

`dpn.txt` L558–560, L568–573, L584–592, §2.4 "Dataflow, functional languages, and process networks", pp. 12–13:

```text
          A dataflow actor, when it fires, maps input tokens into output tokens. Thus, an actor,
applied to one or more streams, will fire repeatedly. A set of firing rules specify when an actor can
fire. Specifically, these rules dictate precisely what tokens must be available at the inputs for the

actor to fire. A firing consumes input tokens and produces output tokens. A sequence of such fir-
ings is a particular type of Kahn process that we call a dataflow process. A network of such pro-
cesses is called a dataflow process network.

is that each actor firing be functional, and that the set of firing rules be sequential. Here, “func-
tional” means that an actor firing lacks side effects and that the output tokens are purely a function
of the input tokens consumed in that firing. This condition is stronger than the Kahn condition that
a process be functional, meaning that the output sequences are a function of the input
sequences[58]. With Kahn’s condition, actors can have and manipulate state. We later relax this
constraint so that actors can have and manipulate state as well. “Sequential” means that the firing
rules can be tested in a pre-defined order using only blocking reads. A little notation will help
make this rather technical definition precise.

```

`dpn.txt` L378–384, L394–402, §1, the nondeterminate merge, p. 8:

```text
        As a rather different example, suppose that actor D in figure 1 is a nondeterminate merge
(any of the three variants discussed by Panagaden and Shanbhogue [78]). Its behavior is that if a
data value (a token) is available on either input, it can immediately move that token to its output.
Now, the output depends on the order in which B and C produce their outputs, and on the timing
with which D examines its inputs. It has been shown that a nondeterminate merge must be either
unfair or nonmonotonic, and hence not continuous [21]. Although rather involved technically,
unfair intuitively means that it favors one input or the other.
        A network with a nondeterminate merge clearly might be nondeterminate, but it might
also be determinate. For example, suppose that C in figure 1 never actually produces any outputs.
Then the nondeterminate merge in D will not make the network nondeterminate.

        The nondeterminate merge does not satisfy one of Kahn’s conditions for a process net-
work, that reads from channels be blocking. This constraint makes it impossible for a process to
test an input for the presence of data. Thus, if D is a nondeterminate merge, then the graph in fig-
ure 1 is not, strictly speaking, a Kahn process network.

```

`dpn.txt` L1460–1472, a delay on a feedback loop, p. 33:

```text

        Consider the example in figure 7b. A single firing of actor B obviously defines a complete
cycle. The feedback loop is used to implement a recurrence, so the feedback channel will store
tokens from one firing of the hierarchical node for use in the next firing. With this usage, the hier-
archical node has state, and is therefore not functional even if fB is. In this case, the feedback loop
must be initialized with tokens in order to avoid deadlock.

        The shaded diamond is called a delay, which is typically implemented as an initial token
in the channel. It cannot be described by F = map (f ), where f is functional, but its behavior is eas-
ily defined by F(X) = i : X, where X is the input sequence, i is the initial token, and “:” is the con-
catenation operator. The initial token enables the first firing of actor B if it requires a token on the
top input. It is called a “delay” because for any channel with a unit delay, the n-th token read from
the channel is the (n − 1)-th written to it. A feedback loop with delay effectively stores state, mak-
```

`dpn.txt` L1537–1556, actors with state, p. 35:

```text
        The basic observation is that internal state in a primitive or a hierarchical node is syntactic
sugar (a convenient syntactic shorthand) for delays on feedback loops at the top level of the
graph. In other words, there is no reason to actually put all such feedback loops at the top level if
semantics can be maintained with a more convenient syntax. With this observation, we can now
allow actors with state. These become more like objects than functions, since they represent both
data and methods for operating on the data. The (implicit) feedback loop around any actor or hier-
archical node with state also establishes a precedence relationship between successive firings of
the actor. This precedence serializes the actor firings, thus ensuring proper state updates.

        Once we allow actors with state, it is a simple extension to allow actors with other side
effects, such as those handling I/O. The inherently sequential nature of an actor that outputs a
stream to a file, for example, is simply represented by a feedback loop that does not carry any
meaningful data, but establishes precedences between successive firings of the actor.

        If actors have state, the notation F = map (f ) is no longer directly valid. With a little adap-
tation, however, we can still use it. If we wish to model an actor with p inputs and q outputs, plus
                               p+1        q+1                                     p+1        q+1
state, we can define F: S            →S         based on an actor function f: S         →S         , where the
extra argument carries the state from one firing to the next.

```

## 2. NVIDIA, CUDA C++ Programming Guide v13.0: streams

- **Source:** as design/capture-replay-sources-2026-11-07.md §4, the legacy single-page guide (v13.0, "Last updated on Nov 02, 2025"), read as text (`lit3/cr-cuda/legacy-13.0.2.txt`).

`legacy-13.0.2.txt` L1917–1918, §6.2.8.5 "Streams":

```text
#### 6.2.8.5. Streams
Applications manage the concurrent operations described above through streams. A stream is a sequence of commands (possibly issued by different host threads) that execute in order. Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently; this behavior is not guaranteed and should therefore not be relied upon for correctness (for example, inter-kernel communication is undefined). The commands issued on a stream may execute when all the dependencies of the command are met. The dependencies could be previously launched commands on same stream or dependencies from other streams. The successful completion of synchronize call guarantees that all the commands launched are completed.
```

## 3. MPI 4.1: the reduction operator

- **Source:** as design/completion-and-period-sources-2026-11-07.md §1 (MPI 4.1, November 2, 2023), text by `pdftotext -layout` of `lit3/mpi41/mpi41-report.pdf`, `lit3/mpi41/mpi41-report.txt`; the bare numbers are the PDF's margin line numbers.

`mpi41-report.txt` L22222–22234, §6.9.1 "Reduce", p. 226:

```text
         The operation op is always assumed to be associative. All predefined operations are also
2
     assumed to be commutative. Users may define operations that are assumed to be associative,
3
     but not commutative. The “canonical” evaluation order of a reduction is determined by the
4
     ranks of the MPI processes in the group. However, the implementation can take advantage of
5
     associativity, or associativity and commutativity in order to change the order of evaluation.
6
     This may change the result of the reduction for operations that are not strictly associative
7
     and commutative, such as floating point addition.
```

## 4. PyTorch: the opset, the backend contract, the device registration, the Metal codegen, DTensor's placements

- **Source:** PyTorch, commit `0c7eac65dffe892199f608583aaa0b4c025e9c4c`, as design/warmup-events-handles-sources-2026-11-07.md §1 (BSD-3-Clause), fetched from `raw.githubusercontent.com/pytorch/pytorch/<sha>/<path>` into `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit8/torch/`.

`docs/source/user_guide/torch_compiler/torch.compiler_ir.md` L5–10:

```markdown
PyTorch 2.0 offers two set of IRs for backends to interface with: Core Aten IR and Prims IR.

## Core Aten IR

Core aten ops is the core subset of aten operators that can be used to compose other operators.
Core aten IR is fully functional, and there is no `inplace` or `_out` variants in this opset.
```

`docs/source/user_guide/torch_compiler/torch.compiler_custom_backends.md` L9, L14:

```markdown
`(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable`.
The returned callable should have the same contract as the `forward` function of the original `torch.fx.GraphModule`
```

`torch/_inductor/codegen/common.py` L484–492, L721–728:

```python
def register_backend_for_device(
    device: str,
    device_scheduling: SchedulingConstructor,
    device_wrapper_codegen: WrapperConstructor,
    device_cpp_wrapper_codegen: WrapperConstructor | None = None,
    device_fx_wrapper_codegen: WrapperConstructor | None = None,
    device_custom_pass: CustomGraphModulePass | None = None,
    device_custom_config: ConfigModule | None = None,
) -> None:
    if get_scheduling_for_device("mps") is None:
        register_backend_for_device(
            "mps",
            MetalScheduling,
            PythonWrapperCodegen,
            CppWrapperMps,
            WrapperFxCodegen,
        )
```

`torch/_inductor/codegen/mps.py` L1–2, L636, L1341:

```python
# This is not a feature-complete compiler backend
# Just an early prototype that shows that one can compile elementwise ops into a Metal shader
class MetalKernel(SIMDKernel):
class MetalScheduling(SIMDScheduling):
```

`torch/_inductor/codegen/metal_flex_attention_template.py` L2, L514–515:

```python
"""Metal shader template for flex attention on MPS"""
    constant int* kv_num_blocks [[buffer(4)]],
    constant int* kv_indices [[buffer(5)]],
```

`torch/distributed/tensor/placement_types.py` L1762–1775:

```python
class Partial(torch._C._distributed.Partial):
    # reduce_ops that distribute over addition, enabling per-input linearity
    # for bilinear ops like mm: reduce_op(A_i @ B) = reduce_op(A_i) @ B
    LINEAR_REDUCE_OPS: tuple[str, ...] = ("sum", "avg")
    ALL_REDUCE_OPS: tuple[str, ...] = ("sum", "avg", "min", "max", "product")

    """
    The ``Partial(reduce_op)`` placement describes the DTensor that is pending
    reduction on a specified ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension holds the partial value of the global Tensor. User can
    redistribute the ``Partial`` DTensor to a ``Replicate`` or ``Shard(dim)``
    placement on the specified ``DeviceMesh`` dimension using ``redistribute``,
    which would trigger necessary communication operations under the hood (i.e.
    ``allreduce``, ``reduce_scatter``).
```

`torch/export/graph_signature.py` L81–85, L121–128, the kinds of inputs and outputs (file `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit6/torch/torch_export_graph_signature.py`):

```python
class InputKind(Enum):
    USER_INPUT = auto()
    PARAMETER = auto()
    BUFFER = auto()
    CONSTANT_TENSOR = auto()
class OutputKind(Enum):
    USER_OUTPUT = auto()
    LOSS_OUTPUT = auto()
    BUFFER_MUTATION = auto()
    PARAMETER_MUTATION = auto()
    GRADIENT_TO_PARAMETER = auto()
    GRADIENT_TO_USER_INPUT = auto()
    USER_INPUT_MUTATION = auto()
```

`torch/export/graph_signature.py` L255–266, the docstring's functional signature:

```text
            %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
            %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
            return (add_2, add_1)

    Resulting ExportGraphSignature of the functional Graph would be::

        # inputs
        p_my_parameter: PARAMETER target='my_parameter'
        b_my_buffer1: BUFFER target='my_buffer1' persistent=True
        b_my_buffer2: BUFFER target='my_buffer2' persistent=True
        x1: USER_INPUT
        x2: USER_INPUT
```
