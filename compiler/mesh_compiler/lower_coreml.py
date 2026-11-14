# design/the-compiler.md, order 4, Core ML; Apple, coremltools 9.0, mlmodel/format/MIL.proto and Model.proto
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
import mesh_graph
import mesh_metal4

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
# rdma/mesh-coreml.m struct mesh_coreml_view
VIEW = "struct mesh_coreml_view { const char *name; void *pages; int64_t *shape, *stride; int32_t rank, type; };\n"


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


def feature(name, shape, array):
    return field(1, name) + field(3, field(5, field(1, b"".join(varint(n) for n in shape)) + field(2, array)))


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
        declared = described = b""
        given = [f"a{k}" for k in range(len(self.inputs))]
        for number, nodes, titles in ((1, self.inputs, given), (10, outputs, names)):
            for node, title in zip(nodes, titles):
                (_, array, scalar), shape = self.crosses(node), tuple(node.meta["val"].shape)
                described += field(number, feature(title, shape, array))
                declared += field(1, named(title, scalar, shape)) * (number == 1)
        function = declared + field(2, OPSET) + field(3, entry(OPSET, block))
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

    # design/compilation-composition-change-request.md, items 5 and 7; rdma/mesh-coreml.m mesh_coreml_predict
    def function(self, stem):
        directory, constants = tempfile.TemporaryDirectory(), self.constants
        fixed = [node for node in constants if node not in self.arguments]
        held = [*self.arguments, *fixed]
        for target, segment in self.program:
            held += [segment] * (target in (RECV, TO_COPY)) + (segment[2] and [node for node, _ in segment[2]]
                                                                if target is None else [])
        slots = {node: slot for slot, node in enumerate(dict.fromkeys(held))}
        count, handles, entries, sends, receives, copies, made, compiled = len(slots) + 1, [], [], [], [], [], [], []

        def view(node):
            value = node.meta["val"] if node not in fixed else constants[node]
            return (slots[node], value.dtype, tuple(value.shape), tuple(value.stride()), 0)

        def handle(pointer):
            handles.append(pointer)
            return f"h[{count + len(handles) - 1}]"

        begun = ["auto predict = (int (*)(void *, const mesh_coreml_view *, uint32_t, const mesh_coreml_view *, uint32_t, "
                 f"char *, size_t)){handle(C.cast(mesh_coreml.LIB.mesh_coreml_predict, C.c_void_p).value)};"]
        for target, segment in self.program:
            if target is SEND:
                entries.append((mesh_metal4.SENT, [], len(sends)))
                sends.append((view(segment.args[0]), segment.args[1]))
            elif target is RECV:
                entries.append((mesh_metal4.RECEIVED, [], len(receives), slots[segment], []))
                receives.append((slots[segment], segment.args[0]))
            elif target is TO_COPY:
                entries.append((mesh_metal4.COPIED, [], len(copies), slots[segment], []))
                copies.append((slots[segment], view(segment.args[0]), dict(segment.kwargs)))
            else:
                model, inputs, outputs = segment
                source = pathlib.Path(directory.name) / f"{stem}.{len(compiled)}.mlmodel"
                source.write_bytes(model)
                compiled.append(source.with_suffix(".mlmodelc"))
                loaded, lines = mesh_coreml.model(source, compiled[-1]), []
                groups = (("x", inputs, [f"a{k}" for k in range(len(inputs))]),
                          ("y", [node for node, _ in outputs], [name for _, name in outputs]))
                for group, nodes, names in groups:
                    described = []
                    for k, (node, name) in enumerate(zip(nodes, names)):
                        slot, dtype, shape, stride, _ = view(node)
                        lines.append(f"static int64_t {group}{k}n[] = {{{', '.join(map(str, shape))}}}, "
                                     f"{group}{k}s[] = {{{', '.join(map(str, stride))}}};")
                        described.append(f'{{"{name}", (void *)a[{slot}], {group}{k}n, {group}{k}s, {len(shape)}, '
                                         f"{ARRAYS[dtype][0]}}}")
                    lines.append(f"const mesh_coreml_view {group}[] = {{{', '.join(described)}}};")
                lines += [f"int written = predict({handle(loaded)}, x, {len(inputs)}, y, {len(outputs)}, f->message, "
                          "sizeof f->message);", "if (written < 0) {", *mesh_metal4.failed("", []), "}",
                          f"a[{count - 1}] += written < {len(outputs)};"]
                entries.append((mesh_metal4.COMMIT, lines))
                for node, _ in outputs:
                    slot, dtype, shape, stride, _ = view(node)
                    made.append((slot, dtype, mesh_graph.extent(shape, stride)))
        source = mesh_metal4.PROGRAM + VIEW + mesh_metal4.program(entries, count, begun, [])
        results = tuple(r if not isinstance(r, torch.fx.Node) else constants[r] if r in fixed else view(r)
                        for r in self.results)
        run = mesh_metal4.Function(
            mesh_metal4.native(source).run, count, handles, [], [(slots[node], position, False)
                                                                 for position, node in enumerate(self.arguments)],
            made, [], sends, receives, copies, results, (SEND, RECV, TO_COPY), directory,
            [(slots[node], constants[node]) for node in fixed])
        run.plan, run.transferred, run.source = lambda: [mesh_coreml.plan(path) for path in compiled], Count(run.a, count - 1), source
        weakref.finalize(run, lambda models=handles[1:]: [mesh_coreml.LIB.mesh_coreml_release(model) for model in models])
        return run


class Count:
    def __init__(self, table, at):
        self.table, self.at = table, at

    def __len__(self):
        return self.table[self.at]


lower(scheduled):
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
            functions[processor].module = module
    if missing:
        raise NotLowered("; ".join(f"{processor}: {', '.join(names)}" for processor, names in missing.items()))
    return functions
