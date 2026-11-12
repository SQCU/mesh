# design/the-compiler.md, order 4, "Core ML"; Apple, coremltools 9.0, mlmodel/format/MIL.proto and Model.proto
import ctypes as C
import pathlib
import sys
import tempfile
import weakref

import torch
from torch.export.graph_signature import OutputKind
from torch.fx.node import map_arg

from .ir import RECV, SEND

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_coreml

ATEN, ENGINE, HOST, OPSET, SPECIFICATION = torch.ops.aten, "ane", torch.device("cpu"), "CoreML7", 8
TO_COPY = ATEN._to_copy.default
HALF, WHOLE, TRUTH, STRING = 10, 23, 1, 2
ARRAYS = {torch.float16: (0, 65552, 10), torch.float32: (1, 65568, 11), torch.int32: (2, 131104, 23)}
CASTS = {HALF: "fp16", WHOLE: "int32", TRUTH: "bool", 11: "fp32"}
BINARY = {ATEN.add.Tensor: "add", ATEN.add.Scalar: "add", ATEN.sub.Tensor: "sub", ATEN.sub.Scalar: "sub",
          ATEN.mul.Tensor: "mul", ATEN.mul.Scalar: "mul", ATEN.div.Tensor: "real_div", ATEN.div.Scalar: "real_div",
          ATEN.pow.Tensor_Scalar: "pow"}
UNARY = {ATEN.tanh.default: "tanh", ATEN.sigmoid.default: "sigmoid", ATEN.relu.default: "relu",
         ATEN.silu.default: "silu", ATEN.exp.default: "exp", ATEN.sqrt.default: "sqrt", ATEN.rsqrt.default: "rsqrt",
         ATEN.abs.default: "abs", ATEN.clone.default: "identity"}
RESHAPED = {ATEN.view.default, ATEN._unsafe_view.default, ATEN.reshape.default, ATEN.unsqueeze.default,
            ATEN.squeeze.dims, ATEN.squeeze.dim}
PRODUCTS = {ATEN.mm.default, ATEN.bmm.default, ATEN.matmul.default, ATEN.addmm.default}
REDUCED = {ATEN.mean.dim: "reduce_mean", ATEN.sum.dim_IntList: "reduce_sum"}
MODES = {"none": "EXACT", "tanh": "TANH_APPROXIMATION"}
PASSED = {"memory_format", "approximate", "dtype", "layout", "device", "pin_memory", "non_blocking"}


class NotLowered(NotImplementedError):
    pass


class Name(str):
    pass


def varint(number):
    number &= (1 << 64) - 1
    out = bytearray()
    while number > 127:
        out.append(number & 127 | 128)
        number >>= 7
    return bytes(out + bytes([number]))


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    value = value.encode() if isinstance(value, str) else value
    return varint(number << 3 | 2) + varint(len(value)) + value


def entry(key, value):
    return field(1, key) + field(2, value)


def kind(dtype):
    return HALF if dtype.is_floating_point else TRUTH if dtype == torch.bool else WHOLE


def tensor_type(scalar, shape):
    return field(1, field(1, scalar) + field(2, len(shape)) + b"".join(field(3, field(1, field(1, n))) for n in shape))


def named(name, scalar, shape):
    return field(1, name) + field(2, tensor_type(scalar, shape))


def immediate(given):
    if isinstance(given, str):
        return STRING, (), field(4, field(1, given))
    tensor = torch.as_tensor(given)
    scalar, flat = kind(tensor.dtype), tensor.detach().reshape(-1)
    if scalar == HALF:
        flat = flat.to(torch.float16).contiguous()
        return scalar, tuple(tensor.shape), field(7, field(1, C.string_at(flat.data_ptr(), flat.nbytes)))
    return scalar, tuple(tensor.shape), field(2 if scalar == WHOLE else 3,
                                              field(1, b"".join(varint(int(v)) for v in flat.tolist())))


def feature(name, value):
    array = field(1, b"".join(varint(n) for n in value.shape)) + field(2, ARRAYS[value.dtype][1])
    return field(1, name) + field(3, field(5, array))


def host(node):
    return node.target in (SEND, RECV) or \
        node.target is TO_COPY and torch.device(node.kwargs.get("device", HOST)) != HOST


class Lowering:
    def __init__(self, module, constants):
        self.constants, self.program, self.missing, self.arguments, self.results = constants, [], [], [], ()
        handlers = {**dict.fromkeys(BINARY, self.binary), **dict.fromkeys(UNARY, self.unary),
                    **dict.fromkeys(RESHAPED, self.reshaped), **dict.fromkeys(PRODUCTS, self.product),
                    **dict.fromkeys(REDUCED, self.reduced), ATEN.permute.default: self.permuted,
                    ATEN.cat.default: self.concatenated, ATEN.gelu.default: self.gelu,
                    ATEN._softmax.default: self.softmax, TO_COPY: self.converted}
        self.begin()
        for node in module.graph.nodes:
            if node.op == "placeholder":
                self.arguments.append(node)
            elif node.op == "output":
                self.close()
                self.results = node.args[0]
            elif host(node):
                self.close()
                self.program.append((node.target, node))
            elif all(read in constants for read in node.all_input_nodes):
                with torch.no_grad():
                    constants[node] = node.target(*map_arg(node.args, constants.__getitem__),
                                                  **map_arg(node.kwargs, constants.__getitem__))
            elif node.target not in handlers or set(node.kwargs) - PASSED:
                self.missing.append(str(node.target))
            else:
                self.made[node] = handlers[node.target](node)

    def begin(self):
        self.operations, self.inputs, self.given, self.made = [], [], {}, {}

    def crosses(self, node):
        value = node.meta["val"]
        if value.dtype not in ARRAYS or not value.dim():
            self.missing.append(f"{node.name} crosses as {value.dtype} of rank {value.dim()}")
        return ARRAYS.get(value.dtype, ARRAYS[torch.float16])

    def operation(self, name, scalar, shape, /, **inputs):
        bound = b""
        for key, given in inputs.items():
            several = given if isinstance(given, tuple) else (given,)
            names = [g if isinstance(g, Name) else self.constant(g) for g in several]
            bound += field(2, entry(key, b"".join(field(1, field(1, bound_name)) for bound_name in names)))
        made = Name(f"v{len(self.operations)}")
        self.operations.append(field(1, name) + bound + field(3, named(made, scalar, shape)) + self.titled(made))
        return made

    def titled(self, made):
        scalar, shape, payload = immediate(str(made))
        return field(5, entry("name", field(2, tensor_type(scalar, shape)) + field(3, field(1, payload))))

    def constant(self, given):
        made = Name(f"v{len(self.operations)}")
        scalar, shape, payload = immediate(given)
        value = field(2, tensor_type(scalar, shape)) + field(3, field(1, payload))
        self.operations.append(field(1, "const") + field(3, named(made, scalar, shape)) + self.titled(made)
                               + field(5, entry("val", value)))
        return made

    def emit(self, name, node, /, **inputs):
        value = node.meta["val"]
        return self.operation(name, kind(value.dtype), tuple(value.shape), **inputs)

    def read(self, operand, like=None):
        if not isinstance(operand, torch.fx.Node):
            return torch.tensor(operand, dtype=like) if like else operand
        if operand in self.constants:
            return self.constants[operand].to(like) if like else self.constants[operand]
        if operand in self.made:
            return self.made[operand]
        if operand not in self.given:
            value, (_, _, scalar) = operand.meta["val"], self.crosses(operand)
            self.given[operand] = Name(f"a{len(self.inputs)}")
            self.inputs.append(operand)
            if scalar != kind(value.dtype):
                self.given[operand] = self.operation("cast", kind(value.dtype), tuple(value.shape),
                                                     x=self.given[operand], dtype=CASTS[kind(value.dtype)])
        return self.given[operand]

    def close(self):
        outputs = [node for node in self.made if any(user not in self.made for user in node.users)]
        if not self.made:
            return
        names = []
        for node in outputs:
            value, (_, _, scalar) = node.meta["val"], self.crosses(node)
            names.append(self.made[node] if scalar == kind(value.dtype) else
                         self.operation("cast", scalar, tuple(value.shape), x=self.made[node], dtype=CASTS[scalar]))
        block = b"".join(field(2, name) for name in names) + b"".join(field(3, o) for o in self.operations)
        function = b"".join(field(1, named(f"a{k}", self.crosses(node)[2], tuple(node.meta["val"].shape)))
                            for k, node in enumerate(self.inputs)) + field(2, OPSET) + field(3, entry(OPSET, block))
        described = b"".join(field(1, feature(f"a{k}", node.meta["val"])) for k, node in enumerate(self.inputs))
        described += b"".join(field(10, feature(name, node.meta["val"])) for name, node in zip(names, outputs))
        program = field(1, 1) + field(2, entry("main", function))
        model = field(1, SPECIFICATION) + field(2, described) + field(502, program)
        self.program.append((None, (model, self.inputs, [(node, name) for node, name in zip(outputs, names)])))
        self.begin()

    def binary(self, node):
        like = node.meta["val"].dtype
        return self.emit(BINARY[node.target], node, x=self.read(node.args[0], like), y=self.read(node.args[1], like))

    def unary(self, node):
        return self.emit(UNARY[node.target], node, x=self.read(node.args[0]))

    def reshaped(self, node):
        return self.emit("reshape", node, x=self.read(node.args[0]), shape=list(node.meta["val"].shape))

    def permuted(self, node):
        return self.emit("transpose", node, x=self.read(node.args[0]), perm=list(node.args[1]))

    def product(self, node):
        *added, left, right = node.args
        left, right = self.read(left), self.read(right)
        if isinstance(right, torch.Tensor) and right.dim() == 2:
            made = self.emit("linear", node, x=left, weight=right.mT)
        else:
            made = self.emit("matmul", node, x=left, y=right, transpose_x=False, transpose_y=False)
        return self.emit("add", node, x=self.read(added[0]), y=made) if added else made

    def reduced(self, node):
        source, over, kept = (*node.args, False)[:3]
        return self.emit(REDUCED[node.target], node, x=self.read(source), axes=list(over), keep_dims=bool(kept))

    def concatenated(self, node):
        dimension = node.args[1] if len(node.args) > 1 else 0
        return self.emit("concat", node, values=tuple(self.read(part) for part in node.args[0]), axis=dimension,
                         interleave=False)

    def gelu(self, node):
        return self.emit("gelu", node, x=self.read(node.args[0]), mode=MODES[node.kwargs.get("approximate", "none")])

    def softmax(self, node):
        return self.emit("softmax", node, x=self.read(node.args[0]), axis=node.args[1])

    def converted(self, node):
        source, to = node.args[0], kind(node.meta["val"].dtype)
        if kind(source.meta["val"].dtype) == to:
            return self.emit("identity", node, x=self.read(source))
        return self.emit("cast", node, x=self.read(source), dtype=CASTS[to])

    def function(self, stem):
        directory = tempfile.TemporaryDirectory()
        program, compiled = [], []
        for target, segment in self.program:
            if target is not None:
                program.append((target, segment))
                continue
            model, inputs, outputs = segment
            source = pathlib.Path(directory.name) / f"{stem}.{len(compiled)}.mlmodel"
            source.write_bytes(model)
            compiled.append(source.with_suffix(".mlmodelc"))
            program.append((None, (mesh_coreml.model(source, compiled[-1]), inputs,
                                   [f"a{k}".encode() for k in range(len(inputs))],
                                   [node for node, _ in outputs], [name.encode() for _, name in outputs])))
        constants, arguments, results, transferred = self.constants, self.arguments, self.results, []

        def views(names, tensors):
            array = (mesh_coreml.View * len(tensors))()
            for view, name, tensor in zip(array, names, tensors):
                view.name, view.pages, view.rank, view.type = name, tensor.data_ptr(), tensor.dim(), \
                    ARRAYS[tensor.dtype][0]
                view.shape = (C.c_int64 * tensor.dim())(*tensor.shape)
                view.stride = (C.c_int64 * tensor.dim())(*tensor.stride())
            return array

        def run(*given):
            values = {**constants, **dict(zip(arguments, given))}
            for target, entry_ in program:
                if target is SEND:
                    SEND(values[entry_.args[0]], entry_.args[1])
                elif target is RECV:
                    values[entry_] = RECV(entry_.args[0])
                elif target is TO_COPY:
                    values[entry_] = TO_COPY(values[entry_.args[0]], **entry_.kwargs)
                else:
                    model, inputs, read, outputs, written = entry_
                    made = [torch.empty(tuple(n.meta["val"].shape), dtype=n.meta["val"].dtype) for n in outputs]
                    backed = mesh_coreml.predict(model, views(read, [values[n] for n in inputs]), views(written, made))
                    transferred.extend([model] * (backed < len(made)))
                    values.update(zip(outputs, made))
            return tuple(values[r] if isinstance(r, torch.fx.Node) else r for r in results)

        run.plan = lambda: [mesh_coreml.plan(path) for path in compiled]
        run.transferred = transferred
        weakref.finalize(run, lambda: ([mesh_coreml.LIB.mesh_coreml_release(entry_[0]) for target, entry_ in program
                                        if target is None], directory.cleanup()))
        return run


def lower(scheduled):
    exported, functions, missing = scheduled.exported, {}, {}
    mutated = {key for output, key in exported.outputs if output == OutputKind.BUFFER_MUTATION}
    for processor, module in scheduled.modules.items():
        if scheduled.engines.get(processor) != ENGINE:
            functions[processor] = module
            continue
        constants = {node: exported.state[node.target][1].detach() for node in module.graph.nodes
                     if node.op == "placeholder" and node.target in exported.state
                     and exported.state[node.target][0] not in mutated}
        lowering = Lowering(module, constants)
        if lowering.missing:
            missing[processor] = sorted(set(lowering.missing))
        else:
            functions[processor] = lowering.function(processor)
    if missing:
        raise NotLowered("; ".join(f"{processor}: {', '.join(names)}" for processor, names in missing.items()))
    return functions
