"""Compute-share declarations for the common whole-function compiler.

Declare shapes and choose shares before execution. ``Operation.prepare`` binds
weights, outputs and host views through one ``declarations.Function`` graph.
Host shares lower to the native recorder's host producer/consumer records. Only result
consumers depend on host completion. No operation times itself, chooses shares,
loads a model, or joins a future while it runs.
"""
import ctypes
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import torch

_objc = None


def _send(pointer, selector, restype=ctypes.c_void_p):
    global _objc
    if _objc is None:
        _objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        _objc.sel_registerName.restype, _objc.sel_registerName.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    call = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', _objc))
    return call(pointer, _objc.sel_registerName(selector))


def host(t):
    """An MPS tensor's memory as a CPU tensor of the same shape and dtype, in place (its storage's MTLBuffer is
    shared), or None where its storage is not CPU-visible.  The caller orders the GPU's and the host's accesses."""
    if t.device.type == 'cpu':
        return t
    if t.device.type != 'mps':
        return None
    buffer = t.untyped_storage().data_ptr()
    if _send(buffer, b'storageMode', ctypes.c_ulong) != 0:
        return None
    base = _send(buffer, b'contents')
    elements = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride())) if t.numel() else 0
    nbytes = elements * t.element_size()
    raw = (ctypes.c_uint8 * nbytes).from_address(base + t.storage_offset() * t.element_size())
    view = torch.frombuffer(raw, dtype=t.dtype, count=elements).as_strided(t.shape, t.stride()) if nbytes else torch.empty(t.shape, dtype=t.dtype)
    view._mesh_storage = t
    return view


def _host_address(value):
    """Bind the shared allocation and its declared offset once at preparation."""
    from . import _stream
    base, _, layout = _stream.page_backing(value)
    return base + (layout.index(0) if layout.count else layout.offset)


def _host_bindings(source, target, input_layouts, output_layouts, binding_layout):
    """CPU and Core ML consume the same resolved operands and copy records."""
    from . import layout
    from .operand import TensorView
    prepared, before, after = [], [], []
    for values, requirements, writing in ((source, input_layouts, False), (target, output_layouts, True)):
        devices = {name: 'cpu' if host(TensorView.of(value).storage) is None else value.device.type for name, value in values.items()}
        placed, commands = layout.bindings(values, requirements, writing=writing, binding=binding_layout, devices=devices)
        (after if writing else before).extend(commands)
        prepared.append(placed)
    return *prepared, before, after


_CORE_ML = None


def _coreml():
    """Load the native compiler, ANE page binding and CPU task interfaces."""
    global _CORE_ML
    if _CORE_ML is None:
        from mesh import RDMA
        try:
            lib = ctypes.CDLL(os.path.join(os.environ.get('MESH_LIBDIR', RDMA), 'libmesh-coreml.dylib'))
            lib.mesh_coreml_load.restype = ctypes.c_long
            lib.mesh_coreml_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.mesh_coreml_release.argtypes = [ctypes.c_long]
            lib.mesh_coreml_bind.restype = ctypes.c_void_p
            lib.mesh_coreml_bind.argtypes = [ctypes.c_long, ctypes.POINTER(ModelView), ctypes.c_size_t,
                                             ctypes.POINTER(ModelView), ctypes.c_size_t]
            lib.mesh_coreml_prepare.restype, lib.mesh_coreml_prepare.argtypes = ctypes.c_char_p, [ctypes.c_void_p] * 4
            lib.mesh_coreml_retain.argtypes = [ctypes.c_void_p]
            lib.mesh_coreml_unbind.argtypes = [ctypes.c_void_p]
            lib.mesh_coreml_error.restype = ctypes.c_char_p
            lib.mesh_coreml_placement.restype = ctypes.c_char_p
            lib.mesh_coreml_placement.argtypes = [ctypes.c_long]
            lib.mesh_coreml_contract.restype = ctypes.c_char_p
            lib.mesh_coreml_contract.argtypes = [ctypes.c_long]
            lib.mesh_coreml_compile.restype = ctypes.c_int
            lib.mesh_coreml_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.mesh_cpu_program_create.argtypes = [ctypes.POINTER(CpuInstruction), ctypes.c_size_t, ctypes.c_char_p]
            lib.mesh_cpu_program_create.restype = ctypes.c_void_p
            lib.mesh_cpu_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.mesh_cpu_compile.restype = ctypes.c_char_p
            lib.mesh_cpu_program_prepare.argtypes = [ctypes.c_void_p] * 4
            lib.mesh_cpu_program_prepare.restype = ctypes.c_char_p
            lib.mesh_cpu_program_retain.argtypes = [ctypes.c_void_p]
            lib.mesh_cpu_program_release.argtypes = [ctypes.c_void_p]
            lib.mesh_cpu_product_bind.argtypes = [ctypes.POINTER(CpuCommand)]
            lib.mesh_cpu_product_bind.restype = ctypes.c_int64
            lib.mesh_cpu_product_workspace.argtypes = [ctypes.POINTER(CpuCommand), ctypes.c_void_p]
            lib.mesh_cpu_command_release.argtypes = [ctypes.POINTER(CpuCommand)]
            _CORE_ML = lib
        except OSError:
            _CORE_ML = False
    return _CORE_ML or None


def _gelu(x):
    return torch.nn.functional.gelu(x, approximate='tanh')


def ffn(x, gate, up, down, out=None):
    """down(gelu_tanh(x gate^T) * (x up^T)) for x [n, H], whatever the operands are: where their type has a fused
    couple (`ffn`, as a coded matrix's) and no gradient is due, that; else torch's own products, which autograd
    differentiates."""
    fused = getattr(type(gate), 'ffn', None)
    if fused is not None and not (torch.is_grad_enabled() and x.requires_grad):
        y = fused(x, gate, up, down)
        return y if out is None else out.copy_(y)
    h = (_gelu(torch.mm(x, gate.T).float()) * torch.mm(x, up.T).float()).to(x.dtype)
    return torch.mm(h, down.T) if out is None else torch.mm(h, down.T, out=out)


class Engine:
    """An implementation operand lowered through the shared function compiler."""
    name, profile = None, None
    host_side = True

    def __init__(self, **options):
        from mesh_compute import implementations
        self.options = options
        self.implementation = options['implementation'] if 'implementation' in options else implementations()[self.profile]

    def required(self, op, dtype):
        return dict(interfaces=[op.kind], input_types=[str(dtype).removeprefix('torch.')], arithmetic_types=[],
                    constants=['weights'] if op.constant else [])

    def eligible(self, op, dtype):
        from mesh_compute import eligibility
        return eligibility(self.required(op, dtype), self.implementation)

    def scalar_type(self, role, dtype):
        name = self.implementation[role + '_type']
        return dtype if name == 'source' else getattr(torch, name)

    @classmethod
    def missing(cls):
        """Why the node lacks this engine, or None."""
        return None

    def input_dtype(self, dtype):
        return self.scalar_type('input', dtype)

    def out_dtype(self, dtype):
        return self.scalar_type('output', dtype)

    def sum_dtype(self, dtype):
        return self.scalar_type('sum', dtype)

    def prepare(self, op, start, end):
        return getattr(self, 'prepare_' + op.kind)(op, start, end)

    def run(self, op, shard, x, out):
        return getattr(self, 'run_' + op.kind)(op, shard, x, out)


class Mps(Engine):
    """GPU products declared to the whole-function compiler."""
    name, host_side, profile = 'mps', False, 'metal_engine'

    @classmethod
    def missing(cls):
        return None if torch.backends.mps.is_available() else 'no MPS device'

    def prepare_linear(self, op, start, end):
        return op.weight[start:end]

    def run_linear(self, op, shard, x, out):
        torch.mm(x, shard.T, out=out)

    def prepare_ffn(self, op, start, end):
        return op.gate[start:end], op.up[start:end], op.down[:, start:end]

    def run_ffn(self, op, shard, x, out):
        ffn(x, *shard, out=out)


class Cpu(Engine):
    """The CPU's matrix units through prepared Accelerate commands: the input read in place as fp32 (the
    GPU's prologue converts it), the fp32 shard resident in host memory, the output written in place in fp32."""
    name, profile = 'cpu', 'accelerate_engine'

    def prepare_linear(self, op, start, end):
        from . import empty
        weight = op.weight[start:end].detach()
        return empty(weight.shape, dtype=self.input_dtype(weight.dtype), device='cpu').copy_(weight)

    def prepare_ffn(self, op, start, end):
        from . import empty
        def cpu(t):
            return empty(t.shape, dtype=self.input_dtype(t.dtype), device='cpu').copy_(t.detach())
        return cpu(op.gate[start:end]), cpu(op.up[start:end]), cpu(op.down[:, start:end])

    def declaration(self, op, shard, rows):
        from .compilation import api_layouts
        layouts = api_layouts('cpu_matrix')
        weights = (shard,) if op.kind == 'linear' else shard
        weights = {f'weight{i}': w for i, w in enumerate(weights)}
        input_layouts = dict(input=layouts['input'], **{name: layouts['weight'] for name in weights})
        output_layouts = {'output': layouts['output']}
        input_dtype, output_dtype = self.input_dtype(weights['weight0'].dtype), self.out_dtype(weights['weight0'].dtype)
        operands = dict(input=dict(shape=[rows, weights['weight0'].shape[1]], dtype=str(input_dtype).removeprefix('torch.'),
                                   requirements=[input_layouts['input']], role=True),
                        output=dict(shape=[rows, weights['weight0'].shape[0] if op.kind == 'linear' else weights['weight2'].shape[0]],
                                    dtype=str(output_dtype).removeprefix('torch.'), requirements=[output_layouts['output']], role=True))
        operands.update({name: dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix('torch.'),
                                   requirements=[input_layouts[name]], role=True) for name, value in weights.items()})
        recipe = cpu_recipe(op.kind, operands, weights=list(weights), implementation=self.implementation)
        return dict(definition=dict(provider='cpu', operation=op.kind, weights=list(weights),
                                    implementation=self.implementation, recipe=recipe), weights=weights,
                    input_layouts=input_layouts, output_layouts=output_layouts, binding_layout='indexed',
                    input_dtype=input_dtype, output_dtype=output_dtype)

    def bind(self, definition, source, target):
        source, target, before, after = _host_bindings(source, target,
            definition['input_layouts'], definition['output_layouts'], definition['binding_layout'])
        native = _coreml()
        if native is None:
            raise RuntimeError('CPU commands require the native engine library')
        from . import layout, _stream
        from .operand import TensorView, requirements
        recipe = definition['recipe']
        values = {name: TensorView.of(value) for name, value in {**source, **target}.items()}
        for name, binding in recipe['bindings'].items():
            if binding['role']:
                continue
            dense = _stream.TensorLayout(binding['shape'], layout.strides(binding['shape']))
            selected = requirements(dense, binding['requirements'])
            values[name] = layout.allocate(selected, getattr(torch, binding['dtype']), 'cpu')
        records = []
        for command in recipe['commands']:
            operands = {role: values[name] for role, name in command['operands'].items()}
            x, w, y = (operands[role] for role in ('input', 'weight', 'output'))
            if command['kind'] == 'product':
                records.append((_CPU_PRODUCT, CpuCommand(None,
                    x.shape[0], w.shape[0], x.shape[1], x.stride(0), w.stride(0), y.stride(0),
                    x.stride(1), w.stride(1), y.stride(1), _host_address(x), _host_address(w), _host_address(y), None)))
            elif command['kind'] == 'gelu':
                records.append((_CPU_GELU, CpuCommand(None, x.shape[0], x.shape[1], 0, 0, 0, 0, 0, 0, 0,
                    _host_address(x), _host_address(w), _host_address(y), _host_address(operands['scratch']))))
            else:
                raise ValueError(f"unknown declared CPU command {command['kind']}")
        reads = dict.fromkeys(name for command in recipe['commands'] for name in command['reads']
                              if recipe['bindings'][name]['role'])
        writes = dict.fromkeys(name for command in recipe['commands'] for name in command['writes'])
        outputs = tuple(values[name] for name in writes)
        pages = None
        def workspace(size):
            nonlocal pages
            from . import empty
            # These instructions run in order within one host call. Assign
            # one workspace that covers all their declared requirements.
            pages = empty(size, dtype=torch.uint8, device='cpu')
            return _host_address(pages)
        binding, address = _cpu_bind(native, _cpu_records(records), workspace, self.implementation)
        if pages is not None:
            outputs += (pages,)
        return CpuProgram(native, binding, _cpu_declaration(records, address, self.implementation), tuple(values[name] for name in reads), outputs, before, after)


def cpu_recipe(operation, operands, *, weights, implementation):
    """The existing CPU numerical program, declared before any pages are bound."""
    bindings = {name: dict(value) for name, value in operands.items()}
    commands = []

    def product(x, w, y):
        commands.append(dict(kind='product', operands=dict(input=x, weight=w, output=y),
                             reads=[x, w], writes=[y]))

    if operation == 'linear':
        product('input', weights[0], 'output')
    elif operation == 'ffn':
        shape = [bindings['input']['shape'][0], bindings[weights[0]]['shape'][0]]
        for name in ('gate', 'up', 'scratch'):
            bindings[name] = dict(shape=shape, dtype=implementation['output_type'],
                                  requirements=['contiguous'], role=False)
        product('input', weights[0], 'gate')
        product('input', weights[1], 'up')
        commands.append(dict(kind='gelu', operands=dict(input='gate', weight='up', output='gate', scratch='scratch'),
                             reads=['gate', 'up'], writes=['gate', 'scratch']))
        product('gate', weights[2], 'output')
    else:
        raise ValueError(f'no declared CPU numerical program for {operation}')
    # MeshCpuCommand uses float pointers, sgemm/BNNS Float32 and vvtanhf.
    # This is the declared numerical ABI, independent of physical page layout.
    invalid = {name: binding['dtype'] for name, binding in bindings.items() if binding['dtype'] != 'float32'}
    if invalid:
        raise ValueError(f'MeshCpuCommand requires float32 named operands: {invalid}')
    return dict(bindings=bindings, commands=commands)


class CpuCommand(ctypes.Structure):
    _fields_ = [('run', ctypes.c_void_p)] + [(name, ctypes.c_int) for name in ('rows', 'columns', 'inner', 'input_stride', 'weight_stride', 'output_stride',
                                                                          'input_column', 'weight_column', 'output_column')] \
             + [(name, ctypes.c_void_p) for name in ('input', 'weight', 'output', 'scratch')]
    _fields_ += [(name, ctypes.c_int) for name in ('input_transpose', 'weight_transpose')]


class CpuInstruction(ctypes.Structure):
    _fields_ = [('run', ctypes.c_void_p), ('arguments', ctypes.c_void_p), ('argument_bytes', ctypes.c_size_t),
                ('release_arguments', ctypes.c_void_p), ('library', ctypes.c_char_p), ('symbol', ctypes.c_char_p)]


def _native_entry(prepare, binding, retain, release):
    return (ctypes.cast(prepare, ctypes.c_void_p).value, binding,
            ctypes.cast(retain, ctypes.c_void_p).value, ctypes.cast(release, ctypes.c_void_p).value)


_CPU_PRODUCT, _CPU_GELU = 0, 1


def _cpu_records(records):
    declarations = []
    for kind, command in records:
        declared = CpuCommand.from_buffer_copy(command)
        declared.run = None
        declarations.append(dict(kind=kind, arguments=bytes(declared),
            places=tuple((getattr(CpuCommand, name).offset, 0) for name in _CPU_FIELDS)))
    return declarations


def _cpu_bind(native, records, workspace, implementation):
    """Resolve declared entries and argument records once into the native program."""
    commands, arguments, instructions, needs = [], [], [], []
    transferred = False
    try:
        for record in records:
            if 'kind' in record:
                command = CpuCommand.from_buffer_copy(record['arguments'])
                if record['kind'] == _CPU_PRODUCT:
                    size = native.mesh_cpu_product_bind(ctypes.byref(command))
                    if size < 0:
                        raise ValueError('CPU matrix views could not be bound to a bulk contraction')
                    if size:
                        needs.append((command, size))
                elif record['kind'] == _CPU_GELU:
                    command.run = ctypes.cast(native.mesh_cpu_gelu, ctypes.c_void_p)
                else:
                    raise ValueError(f"unknown declared Accelerate command {record['kind']}")
                commands.append(command)
                instructions.append(CpuInstruction(command.run, ctypes.addressof(command), ctypes.sizeof(command),
                    ctypes.cast(native.mesh_cpu_command_release, ctypes.c_void_p), None, None))
            else:
                library = native.mesh_cpu_compile(record['source'].encode(),
                    json.dumps(record['compiler'], separators=(',', ':')).encode())
                if library is None:
                    raise RuntimeError(native.mesh_coreml_error().decode())
                data = ctypes.create_string_buffer(bytes(record['arguments']))
                arguments.append(data)
                instructions.append(CpuInstruction(None, ctypes.addressof(data), len(record['arguments']), None,
                    library, record['symbol'].encode()))
        address = workspace(max(size for _, size in needs)) if needs else 0
        for command, _ in needs:
            native.mesh_cpu_product_workspace(ctypes.byref(command), address)
        bound = (CpuInstruction * len(instructions))(*instructions)
        binding = native.mesh_cpu_program_create(bound, len(bound), json.dumps(implementation['resources'], separators=(',', ':')).encode())
        if not binding:
            raise RuntimeError(native.mesh_coreml_error().decode())
        transferred = True
        return binding, address
    finally:
        if not transferred:
            for command in commands:
                native.mesh_cpu_command_release(ctypes.byref(command))


_CPU_FIELDS = ('input', 'weight', 'output', 'scratch')


def _cpu_declaration(records, workspace, implementation):
    return _cpu_serialized(_cpu_records(records), workspace, implementation)


def _cpu_serialized(records, workspace, implementation):
    """One replay declaration for static numerical entries and compiled symbols.

    Pointer words remain ordinary relocation places in copied argument records;
    executable addresses are resolved from declarations during preparation.
    """
    metadata = dict(implementation=implementation, instructions=[dict(
        {key: value for key, value in record.items() if key not in ('arguments', 'places')},
        argument_bytes=len(record['arguments'])) for record in records])
    header = json.dumps(metadata, separators=(',', ':')).encode()
    data = struct.pack('<QQ', len(header), workspace) + header
    places = [(8, 0)]
    for record in records:
        places.extend((len(data) + offset, relocation) for offset, relocation in record.get('places', ()))
        data += record['arguments']
    return 'cpu', data, places


def cpu_native(source, compiler, symbol, arguments, places, *, implementation, inputs=(), outputs=()):
    """Bind a compiled CPU entry through the existing host program and recorder.

    ``places`` uses MetalDeclaration's (byte offset, relocation kind) records.
    """
    native = _coreml()
    if native is None:
        raise RuntimeError('CPU commands require the native engine library')
    records = [dict(source=source, compiler=compiler, symbol=symbol, arguments=bytes(arguments), places=tuple(places))]
    binding, workspace = _cpu_bind(native, records, None, implementation)
    return CpuProgram(native, binding, _cpu_serialized(records, workspace, implementation),
                      tuple(inputs), tuple(outputs), (), ())


def rebuild(kind, data):
    """design/replay-templates-2026-10-29.md: the native entry (prepare, context, retain, release) a host declaration makes,
    its addresses already the instance's; the context's one reference is the caller's."""
    native = _coreml()
    if native is None:
        raise RuntimeError('host operations require the native engine library')
    if kind == 'cpu':
        length, workspace = struct.unpack_from('<QQ', data)
        header = json.loads(data[16:16 + length])
        at, records = 16 + length, []
        for instruction in header['instructions']:
            end = at + instruction['argument_bytes']
            if end > len(data):
                raise ValueError('CPU replay argument record exceeds its declaration')
            records.append(dict(instruction, arguments=data[at:end]));at = end
        if at != len(data):
            raise ValueError('CPU replay has bytes outside its declared argument records')
        binding, _ = _cpu_bind(native, records, lambda size: workspace, header['implementation'])
        return _native_entry(native.mesh_cpu_program_prepare, binding, native.mesh_cpu_program_retain, native.mesh_cpu_program_release)
    if kind == 'coreml':
        length, = struct.unpack_from('<Q', data)
        header = json.loads(data[8:8 + length])
        if header['implementation']['compiler'] != 'coreml':
            raise ValueError('model replay requires its declared provider implementation')
        addresses = struct.unpack_from(f'<{len(header["inputs"]) + len(header["outputs"])}Q', data, _coreml_addresses(length))
        handle = _load_coreml(native, header['compiled'], header['configuration'])
        try:
            views = _model_views(header['inputs'] + header['outputs'], addresses)
            inputs, outputs = views[:len(header['inputs'])], views[len(header['inputs']):]
            binding = native.mesh_coreml_bind(handle, (ModelView * len(inputs))(*inputs), len(inputs),
                                              (ModelView * len(outputs))(*outputs), len(outputs))
        finally:
            native.mesh_coreml_release(handle)
        if not binding:
            raise RuntimeError(native.mesh_coreml_error().decode())
        return _native_entry(native.mesh_coreml_prepare, binding, native.mesh_coreml_retain, native.mesh_coreml_unbind)
    raise ValueError(f'no host operation of kind {kind}')


class CpuProgram:
    def __init__(self, native, binding, declaration, inputs, outputs, before, after):
        from . import _stream
        self.native, self.binding, self.declaration = native, binding, declaration
        self.inputs, self.outputs = inputs, outputs
        self.before, self.after = _stream.Commands(before), _stream.Commands(after)
        self.entry = _native_entry(self.native.mesh_cpu_program_prepare, self.binding,
                                   self.native.mesh_cpu_program_retain, self.native.mesh_cpu_program_release)

    def append(self, schedule):
        schedule.append(self.before)
        schedule.host(self.entry, self.inputs, self.outputs, self.declaration)
        schedule.append(self.after)

    def __del__(self):
        if getattr(self, 'binding', None):
            self.native.mesh_cpu_program_release(self.binding)


class _Compiled:
    """A model share: its declared operation kind and typed constant arrays."""

    def __init__(self, kind, arrays):
        self.kind, self.arrays = kind, arrays


class Ane(Engine):
    """Core ML graph shares with explicit tile and channel block configuration.

    Compilation and model loading happen during preparation. Direct ANE requests
    bind guaranteed caller-owned input and output pages.
    """
    name, profile = 'ane', 'ane_engine'

    def __init__(self, **options):
        super().__init__(**options)
        self.tile, self.block = int(options['tile']), int(options['block'])
        self.model = dict(options['model'], implementation=self.implementation)

    def input_dtype(self, dtype):
        return getattr(torch, self.model['input']['dtype'])

    def out_dtype(self, dtype):
        return getattr(torch, self.model['output']['dtype'])

    def sum_dtype(self, dtype):
        return getattr(torch, self.model['sum_dtype'])

    @classmethod
    def missing(cls):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            return 'coremltools is not installed (rdma/requirements.txt)'
        return None

    def prepare_linear(self, op, start, end):
        return _Compiled('linear', (op.weight[start:end].detach().to('cpu', getattr(torch, self.model['weight_dtype'])).numpy(),))

    def prepare_ffn(self, op, start, end):
        numpy = lambda t: t.detach().to('cpu', getattr(torch, self.model['weight_dtype'])).numpy()
        return _Compiled('ffn', (numpy(op.gate[start:end]), numpy(op.up[start:end]), numpy(op.down[:, start:end])))

    def declaration(self, op, shard, rows):
        compiled, _, placement = neural_engine_share(shard.kind, shard.arrays, rows, self.tile, self.block, self.model)
        contract = model_pages(placement['contract'])
        source, target = self.model['input'], self.model['output']
        columns = shard.arrays[0].shape[0] if shard.kind == 'linear' else shard.arrays[2].shape[0]
        return dict(definition=dict(provider='coreml', compiled=str(compiled), configuration=self.model, contract=contract,
                                    implementation=provider_implementation(self.implementation, contract, placement)), weights={},
                    input_layouts=model_requirements(contract['inputs'], {source['name']: (rows, shard.arrays[0].shape[1])}),
                    output_layouts=model_requirements(contract['outputs'], {target['name']: (rows, columns)}),
                    binding_layout=self.model['binding_layout'],
                    input_dtype=_MODEL_TYPES[contract['inputs'][source['name']]['dtype']],
                    output_dtype=_MODEL_TYPES[contract['outputs'][target['name']]['dtype']])


class ModelPages(ctypes.Structure):
    _fields_ = [('base', ctypes.c_void_p), ('bytes', ctypes.c_size_t), ('offset', ctypes.c_size_t)]


class ModelView(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p), ('pages', ModelPages), ('shape', ctypes.POINTER(ctypes.c_int64)),
                ('strides', ctypes.POINTER(ctypes.c_int64)), ('rank', ctypes.c_int), ('dtype', ctypes.c_int)]


_MODEL_TYPES = (torch.float16, torch.float32, torch.int32, torch.float64, torch.int8)


def provider_implementation(implementation, contract, placement=None):
    """Retain the compiled provider facts on the selected capability operand."""
    provider = dict(contract=contract)
    if placement is not None:
        provider['placement'] = placement
    return dict(implementation, output_storage=contract.get('output_storage', implementation['output_storage']),
                provider=provider)


def model_pages(contract):
    """Require the provider's storage guarantee before assigning output pages."""
    if contract['output_storage'] != 'caller-pages':
        raise ValueError('prepared page execution requires guaranteed caller-owned output pages; public Core ML outputBackings are advisory')
    return contract


def model_requirements(features, shapes):
    """Provider layouts and storage extents before producer page placement."""
    from . import _stream
    result = {}
    for name, shape in shapes.items():
        feature = features[name]
        layout = 'strided' if feature['strides'] is None else {
            'view': _stream.TensorLayout(feature['shape'], feature['strides']).reshape(tuple(shape)).state}
        result[name] = dict(layout=layout, capacity=feature['bytes'] or 0, alignment=feature['alignment'])
    return result


def model_operand(feature, shape, *, device='cpu'):
    """Allocate a provider's named logical view from its prepared contract."""
    from . import _stream, layout
    from .operand import required_layout
    requirement = model_requirements({'value': feature}, {'value': shape})['value']
    selected = required_layout(_stream.TensorLayout(shape, layout.strides(shape)), requirement['layout'])
    dtype = _MODEL_TYPES[feature['dtype']]
    first = selected.index(0) if selected.count else 0
    return layout.allocate(selected.shift(-first), dtype, device, extent=requirement['capacity'],
                           alignment=requirement['alignment'])


def _model_views(specs, addresses):
    """Resolve each relocated logical address to its actual bridge-page owner."""
    from . import _stream
    return [ModelView(name.encode(), ModelPages(*_stream.host_page_backing(address)),
                      (ctypes.c_int64 * len(shape))(*shape), (ctypes.c_int64 * len(shape))(*strides),
                      len(shape), dtype) for (name, shape, strides, dtype), address in zip(specs, addresses)]


def _coreml_addresses(length):
    """Where a Core ML declaration's addresses start: after its length and JSON header, 8-byte aligned."""
    return 8 + (length + 7) // 8 * 8


class Prediction:
    """One direct ANE invocation's pages, bound before the numerical function."""
    def __init__(self, native, model, source, target, *, compiled, configuration, contract, implementation):
        from . import _stream
        from .operand import TensorView
        self.native = native
        self.placement = json.loads(native.mesh_coreml_placement(model))
        self.implementation = provider_implementation(implementation, contract, self.placement)
        input_layouts = model_requirements(contract['inputs'], {name: value.shape for name, value in source.items()})
        output_layouts = model_requirements(contract['outputs'], {name: value.shape for name, value in target.items()})
        source, target, before, after = _host_bindings(source, target, input_layouts, output_layouts, configuration['binding_layout'])
        self.before, self.after = _stream.Commands(before), _stream.Commands(after)
        self.inputs, self.outputs = tuple(source.values()), tuple(target.values())
        def specs(values, features):
            made = []
            for name, original in values.items():
                operand = TensorView.of(original)
                feature = features[name]
                if operand.dtype != _MODEL_TYPES[feature['dtype']]:
                    raise ValueError(f'{name}: operand type differs from its prepared provider contract')
                shape, strides = ((list(operand.shape), list(operand.layout.strides)) if feature['strides'] is None else
                                  (feature['shape'], feature['strides']))
                made.append(((name, shape, strides, feature['dtype']),
                             _host_address(operand)))
            return made
        given, made = specs(source, contract['inputs']), specs(target, contract['outputs'])
        inputs = (ModelView * len(given))(*_model_views([v for v, _ in given], [a for _, a in given]))
        outputs = (ModelView * len(made))(*_model_views([v for v, _ in made], [a for _, a in made]))
        self.binding = native.mesh_coreml_bind(model, inputs, len(inputs), outputs, len(outputs))
        if not self.binding:
            raise RuntimeError(native.mesh_coreml_error().decode())
        # The compiled model, all load constraints and its views survive replay preparation.
        header = json.dumps({'compiled': str(compiled), 'configuration': configuration, 'implementation': self.implementation, 'inputs': [v for v, _ in given],
                             'outputs': [v for v, _ in made]}).encode()
        at = _coreml_addresses(len(header))
        addresses = [a for _, a in given + made]
        data = struct.pack('<Q', len(header)) + header.ljust(at - 8, b' ') + struct.pack(f'<{len(addresses)}Q', *addresses)
        self.declaration = 'coreml', data, [(at + 8 * k, 0) for k in range(len(addresses))]
        self.entry = _native_entry(native.mesh_coreml_prepare, self.binding,
                                   native.mesh_coreml_retain, native.mesh_coreml_unbind)

    def __del__(self):
        if getattr(self, 'binding', None):
            self.native.mesh_coreml_unbind(self.binding)

    def append(self, schedule):
        schedule.append(self.before)
        schedule.host(self.entry, self.inputs, self.outputs, self.declaration)
        schedule.append(self.after)


def _load_coreml(native, compiled, configuration):
    handle = native.mesh_coreml_load(os.fsencode(compiled), json.dumps(configuration).encode())
    if handle < 0:
        raise RuntimeError(native.mesh_coreml_error().decode())
    return handle


def coreml_contract(compiled, configuration):
    """Read the compiled provider contract before allocating its operands."""
    native = _coreml()
    handle = _load_coreml(native, compiled, configuration)
    try:
        return json.loads(native.mesh_coreml_contract(handle))
    finally:
        native.mesh_coreml_release(handle)


def coreml_graph(compiled, inputs, outputs, *, configuration, contract, implementation):
    """Bind a compiled graph with guaranteed caller-owned output pages.

    ``configuration`` supplies backend, compute units, function and devices.
    The shared loader inspects the provider's preferred placement before binding.
    Graph conversion belongs to preparation; no operation-kind dispatch or
    model loading occurs in the returned host record. The common schedule
    derives dependencies from the declared input and output views. ``contract``
    supplies the provider's types, layouts, footprints and address alignment,
    which the caller has already consumed when assigning producer storage.
    """
    model_pages(contract)
    configuration = dict(configuration, implementation=implementation)
    native = _coreml()
    if native is None:
        raise RuntimeError('Core ML graph requires the native shared-memory binding')
    handle = _load_coreml(native, compiled, configuration)
    try:
        return Prediction(native, handle, inputs, outputs, compiled=compiled, configuration=configuration, contract=contract, implementation=implementation)
    finally:
        native.mesh_coreml_release(handle)


def page_state_program(program, pairs):
    """Expose exported effects and aliases before provider placement.

    Returns the numerical program, named input/result metadata, result aliases
    and captured weight operands. A placeholder keeps its input name
    while several result names denote the same pages. No identity operation or
    copied value is introduced.
    """
    import copy
    from dataclasses import replace
    from torch.export import ExportedProgram
    from torch.export.exported_program import ModuleCallEntry, ModuleCallSignature
    from torch.export.graph_signature import ExportGraphSignature, InputKind, OutputKind, TensorArgument
    from torch.utils._pytree import tree_structure
    from mesh_graph import needed

    signature = copy.deepcopy(program.graph_signature)
    effects = {spec.target: spec for spec in signature.output_specs
               if spec.kind in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)}
    missing = effects.keys() - pairs.keys()
    if missing:
        raise ValueError(f'exported state effects need named page pairs: {sorted(missing)}')
    graph = copy.deepcopy(program.graph)
    nodes = {node.name: node for node in graph.nodes}
    returned = next(node for node in graph.nodes if node.op == 'output')
    if any(not isinstance(spec.arg, TensorArgument) for spec in signature.output_specs):
        raise ValueError('the exported tensor interface requires named tensor results')
    returned_inputs = {value.name for value in returned.args[0] if value.op == 'placeholder'}
    renames, given, weights = {}, [], {}
    for spec in signature.input_specs:
        effect = spec.target if spec.kind == InputKind.BUFFER else spec.arg.name
        if effect in effects:
            renames[spec.arg.name] = pairs[effect]['input']
            if spec.kind == InputKind.BUFFER:
                spec = replace(spec, kind=InputKind.USER_INPUT, target=None, persistent=None)
        elif spec.arg.name in returned_inputs and spec.kind != InputKind.USER_INPUT:
            weights[spec.arg.name] = (program.state_dict[spec.target] if spec.target in program.state_dict
                                      else program.constants[spec.target])
            spec = replace(spec, kind=InputKind.USER_INPUT, target=None, persistent=None)
        given.append(spec)
    given = ([spec for spec in given if spec.kind != InputKind.USER_INPUT] +
             [spec for spec in given if spec.kind == InputKind.USER_INPUT])
    first = next(node for node in graph.nodes if node.op != 'placeholder')
    for spec in given:
        first.prepend(nodes[spec.arg.name])
    inputs = {renames.get(spec.arg.name, spec.arg.name): nodes[spec.arg.name].meta['val']
              for spec in given if spec.kind == InputKind.USER_INPUT}
    outputs, aliases, canonical, made, values = {}, {}, {}, [], []
    for spec, value in zip(signature.output_specs, returned.args[0], strict=True):
        name = (pairs[spec.target]['output'] if spec.kind in
                (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION) else spec.arg.name)
        outputs[name] = value.meta['val']
        if value.op == 'placeholder':
            aliases[name] = renames.get(value.name, value.name)
        elif value in canonical:
            if name != canonical[value]:
                aliases[name] = canonical[value]
        else:
            canonical[value] = name
            renames[value.name] = name
            made.append(replace(spec, kind=OutputKind.USER_OUTPUT, target=None))
            values.append(value)
    # Rename only the provider's actual values; named aliases stay declarations.
    for old, name in renames.items():
        nodes[old].name = name
        if nodes[old].op == 'placeholder':
            nodes[old].target = name
    for spec in given + made:
        spec.arg.name = renames.get(spec.arg.name, spec.arg.name)
    returned.args = (tuple(values),)
    ordered = list(graph.nodes)
    live = {i for span in needed([node.all_input_nodes for node in ordered],
                                [(node,) for node in ordered], (returned,)) for i in span}
    for i in range(len(ordered) - 1, -1, -1):
        if i not in live:
            graph.erase_node(ordered[i])
    retained = {node.name for node in graph.nodes if node.op == 'placeholder'}
    given = [spec for spec in given if spec.arg.name in retained]
    captured = {spec.target for spec in given if spec.kind != InputKind.USER_INPUT}
    signature = ExportGraphSignature(given, made)
    user_inputs = [spec.arg for spec in given if spec.kind == InputKind.USER_INPUT]
    user_outputs = [spec.arg for spec in made]
    flat = ModuleCallSignature(user_inputs, user_outputs,
        tree_structure((tuple(None for _ in user_inputs), {})),
        tree_structure(tuple(None for _ in user_outputs)))
    prepared = ExportedProgram(program.graph_module, graph, signature,
        {name: value for name, value in program.state_dict.items() if name in captured},
        copy.deepcopy(program.range_constraints), [ModuleCallEntry('', flat)],
        constants={name: value for name, value in program.constants.items() if name in captured},
        verifiers=program.verifiers)
    return prepared, inputs, outputs, aliases, weights


def export_coreml_graph(program, *, directory, configuration):
    """Prepare the ordinary host declaration before assigning operand pages.

    The returned definition and layout requirements feed ``host.declare``.
    ``input_meta`` and ``output_meta`` retain the source graph's named tensor
    types, including aliases which require no provider output or host task.
    ``weights`` supplies returned captures through the existing weight operands.
    """
    implementation = configuration['implementation']
    import copy
    import coremltools as ct
    from coremltools.converters.mil.frontend.torch.utils import TORCH_DTYPE_TO_MIL_DTYPE
    from torch.export.graph_signature import InputKind
    program, inputs, outputs, aliases, weights = page_state_program(program, configuration['state_pairs'])
    nodes = {node.name: node for node in program.graph.nodes}
    sources = [spec for spec in program.graph_signature.input_specs if spec.kind == InputKind.USER_INPUT]
    results = program.graph_signature.output_specs
    provider = copy.deepcopy(configuration)
    provider['state_pairs'] = {name: pair for name, pair in configuration['state_pairs'].items()
                               if pair['output'] not in aliases}
    source_names, result_names = [spec.arg.name for spec in sources], [spec.arg.name for spec in results]
    if provider['backend'] == 'ane':
        symbols = provider['ane']['symbols']
        symbols['inputs'] = {name: symbols['inputs'][name] for name in source_names}
        symbols['outputs'] = {name: symbols['outputs'][name] for name in result_names}
    contract, compiled = {'inputs': {}, 'outputs': {}}, None
    if results:
        native = _coreml()
        if native is None:
            raise RuntimeError('Core ML graph requires the native shared-memory binding')
        def feature(spec):
            return ct.TensorType(name=spec.arg.name,
                                 dtype=TORCH_DTYPE_TO_MIL_DTYPE[nodes[spec.arg.name].meta['val'].dtype])
        model = ct.convert(program, inputs=[feature(spec) for spec in sources], outputs=[feature(spec) for spec in results],
                           convert_to='mlprogram', compute_units=coreml_units(provider['units']),
                           minimum_deployment_target=getattr(ct.target, provider['deployment']),
                           compute_precision=getattr(ct.precision, provider['precision']), skip_model_load=True)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        package, compiled = directory / 'graph.mlpackage', directory / 'graph.mlmodelc'
        model.save(str(package))
        try:
            if native.mesh_coreml_compile(os.fsencode(package), os.fsencode(compiled)):
                raise RuntimeError(native.mesh_coreml_error().decode())
        finally:
            shutil.rmtree(package)
        contract = model_pages(coreml_contract(compiled, provider))
    input_layouts = dict.fromkeys(inputs, 'indexed')
    output_layouts = dict.fromkeys(outputs, 'indexed')
    input_layouts.update(model_requirements(contract['inputs'], {name: inputs[name].shape for name in contract['inputs']}))
    output_layouts.update(model_requirements(contract['outputs'], {name: outputs[name].shape for name in contract['outputs']}))
    return dict(definition=dict(provider='coreml', compiled=None if compiled is None else str(compiled), configuration=provider,
                               contract=contract, implementation=provider_implementation(implementation, contract),
                               output_aliases=aliases, state_pairs=configuration['state_pairs']),
                input_layouts=input_layouts, output_layouts=output_layouts, binding_layout=configuration['binding_layout'],
                input_meta=inputs, output_meta=outputs, weights=weights)


def coreml_units(units):
    import coremltools as ct
    return {0: ct.ComputeUnit.CPU_AND_NE, 1: ct.ComputeUnit.ALL, 2: ct.ComputeUnit.CPU_ONLY, 3: ct.ComputeUnit.CPU_AND_GPU}[units]


def neural_engine_model(kind, arrays, rows, tile, block, configuration):
    """Prepare a declared linear/FFN share using configured Core ML operands.

    Tile and channel blocks choose convolution domains, including their tails.
    Input/output names and types, weight type, precision, deployment and compute
    units come from the same configuration consumed by the binding frontend.
    """
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types, Program, Function
    width = arrays[0].shape[1]
    units = arrays[0].shape[0]
    block = int(block)
    spans = [(b, min(b + block, units)) for b in range(0, units, block)]
    conv = lambda array: mb.const(val=np.ascontiguousarray(array, dtype=configuration['weight_dtype']).reshape(array.shape[0], array.shape[1], 1, 1))

    source, target = configuration['input'], configuration['output']
    kinds = {'float16': types.fp16, 'float32': types.fp32, 'int32': types.int32, 'float64': types.fp64, 'int8': types.int8}
    casts = {'float16': 'fp16', 'float32': 'fp32', 'int32': 'int32', 'float64': 'fp64', 'int8': 'int8'}
    program = Program()
    with Function({source['name']: mb.TensorSpec(shape=(rows, width), dtype=kinds[source['dtype']])},
                  opset_version=getattr(ct.target, configuration['deployment'])) as function:
        x = function.inputs[source['name']]
        if source['dtype'] != configuration['weight_dtype']:
            x = mb.cast(x=x, dtype=casts[configuration['weight_dtype']])
        if kind == 'linear':
            weights = [(conv(arrays[0][a:b]),) for a, b in spans]
        else:
            weights = [(conv(arrays[0][a:b]), conv(arrays[1][a:b]), conv(arrays[2][:, a:b])) for a, b in spans]
        channels = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, width, 1, rows])
        if rows <= tile:
            pieces = [channels]
        elif rows % tile:
            pieces = mb.split(x=channels, split_sizes=[tile] * (rows // tile) + [rows % tile], axis=3)
        else:
            pieces = mb.split(x=channels, num_splits=rows // tile, axis=3)
        outputs = []
        for p in pieces:
            if kind == 'linear':
                parts = [mb.conv(x=p, weight=w[0]) for w in weights]
                outputs.append(mb.concat(values=parts, axis=1) if len(parts) > 1 else parts[0])
            else:
                total = None
                for g, u, d in weights:
                    h = mb.mul(x=mb.gelu(x=mb.conv(x=p, weight=g), mode='TANH_APPROXIMATION'), y=mb.conv(x=p, weight=u))
                    part = mb.conv(x=h, weight=d)
                    total = part if total is None else mb.add(x=total, y=part)
                outputs.append(total)
        y = mb.concat(values=outputs, axis=3) if len(outputs) > 1 else outputs[0]
        n = arrays[0].shape[0] if kind == 'linear' else arrays[2].shape[0]
        y = mb.transpose(x=mb.reshape(x=y, shape=[n, rows]), perm=[1, 0])
        y = mb.cast(x=y, dtype=casts[target['dtype']], name=target['name']) if configuration['weight_dtype'] != target['dtype'] else mb.identity(x=y, name=target['name'])
        function.set_outputs([y])
    program.add_function(configuration['function'], function)
    program.default_function_name = configuration['function']
    program.export_as_multifunction = True
    return ct.convert(program, convert_to='mlprogram', compute_units=coreml_units(configuration['units']),
                      compute_precision=getattr(ct.precision, configuration['precision']),
                      minimum_deployment_target=getattr(ct.target, configuration['deployment']), skip_model_load=True)


def neural_engine_share(kind, arrays, rows, tile, block, configuration):
    """Compile a configured model share and retain its provider placement.

    Names, types, row/channel tiles, precision, target, compute units and allowed
    devices are compiler operands. Artifacts are reused by those operands and
    constants. Provider placement is checked during preparation, before binding.
    """
    import numpy as np
    cache = Path(os.path.expanduser('~/.cache/mesh-engines'))
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(b''.join(np.ascontiguousarray(a).tobytes() for a in arrays) + json.dumps([kind, rows, tile, block, configuration], sort_keys=True).encode()).hexdigest()
    native = _coreml()
    if native is None:
        raise RuntimeError('Neural Engine share: rdma/libmesh-coreml.dylib does not load')
    compiled = cache / f'{digest}-b{block}.mlmodelc'
    if not compiled.exists():
        package = cache / f'{digest}-b{block}.mlpackage'
        neural_engine_model(kind, arrays, rows, tile, block, configuration).save(str(package))
        failed = native.mesh_coreml_compile(str(package).encode(), str(compiled).encode())
        shutil.rmtree(package, ignore_errors=True)
        if failed:
            raise RuntimeError(f'Neural Engine share: {native.mesh_coreml_error().decode()}')
    handle = _load_coreml(native, compiled, configuration)
    try:
        return compiled, block, dict(json.loads(native.mesh_coreml_placement(handle)),
                                     contract=json.loads(native.mesh_coreml_contract(handle)))
    finally:
        native.mesh_coreml_release(handle)



ENGINES = {e.name: e for e in (Mps, Cpu, Ane)}


class Engines:
    """The engines one node's operations may share their units among, in order (the order of their shares along an
    operation's units), each with its options; the evidence of the operations using them; each decision's search."""

    def __init__(self, names, **options):
        from mesh_compute import hardware
        self.hardware = hardware()
        names = [names] if isinstance(names, str) else list(names)
        unknown = [n for n in names if n not in ENGINES]
        if unknown:
            raise ValueError(f'engines: {unknown} are none of {sorted(ENGINES)}')
        lacking = {n: ENGINES[n].missing() for n in names if ENGINES[n].missing()}
        if lacking:
            raise RuntimeError(f'engines: this node lacks {lacking}')
        self.names = names
        self.engines = [ENGINES[n](**options.get(n, {})) for n in names]
        self.operations, self.decisions = [], {}
        self.parameters = options.get('compilation', {})
        self.source_scope = options.get('source_scope')


class Operation:
    """An operation declaration. Shares and shapes are configuration operands.

    ``prepare`` binds an input, its final output and every engine's scratch and
    model. The returned call uses those bindings; it never chooses or loads a
    shard. Placement changes require preparing the next whole function.
    """
    kind = combine = None

    def __init__(self, units, width, engines, name, grain, constant):
        self.units, self.width, self.engines, self.grain, self.constant = units, width, engines, grain, constant
        self.name = name
        if engines is not None:
            engines.operations.append(self)

    def declare(self, rows, dtype, *, capacities):
        """Work rates for this operation and shape, in the configured engine order.

        Rates are supplied operands, not CPU/GPU/ANE core counts. Whole-call
        observations refine them through the same assignment solver as routes.
        A zero explicitly excludes an engine from this declaration.
        """
        from mesh_compute import work_domain
        if self.name is None or self.engines.source_scope is None:
            raise ValueError('engine work requires a declared operation name and source_scope')
        key = f'{self.name}@{rows}'
        required = dict(interfaces=[self.kind], input_types=[str(dtype).removeprefix('torch.')], arithmetic_types=[],
                        constants=['weights'] if self.constant else [])
        self.engines.decisions[key] = work_domain(required,
            [engine.implementation for engine in self.engines.engines], capacities,
            units=self.units, grain=self.grain,
            identity=dict(scope=self.engines.source_scope, operation=self.name, interface=self.kind))
        return key

    def prepare(self, x, *, parts, output=None):
        if self.name is None or self.engines.source_scope is None:
            raise ValueError('engine preparation requires a declared operation name and source_scope')
        return Prepared(self, x, parts, output)

    def finish(self, y, first=0):
        return y


class Prepared:
    """One declared graph for all engine shares, casts and their consumers."""
    def __init__(self, op, x, parts, output=None):
        self.op, self.x = op, x
        rows, dtype = x.shape[0], x.dtype
        if len(parts) != len(op.engines.engines) or sum(parts) != op.units or min(parts) < 0:
            raise ValueError('engine shares must partition the declared units')
        from . import empty
        self.output = empty((rows, op.width), dtype=dtype) if output is None else output
        if self.output.shape != (rows, op.width) or self.output.dtype != dtype:
            raise ValueError('output must match the declared shape and dtype')
        output_device = self.output.device
        configuration = {k: p['value'] for k, p in op.engines.parameters.items()}
        shares, start = [], 0
        sum_types = [engine.sum_dtype(dtype) for engine, count in zip(op.engines.engines, parts) if count]
        from functools import reduce
        accumulator = reduce(torch.promote_types, sum_types, dtype)
        for engine, count in zip(op.engines.engines, parts):
            end = start + count
            if count:
                candidate = engine.eligible(op, dtype)
                if candidate['missing']:
                    raise ValueError(f'{engine.name} cannot execute the declared operation: {candidate}')
                shard = engine.prepare(op, start, end)
                declaration = engine.declaration(op, shard, rows) if engine.host_side else None
                if declaration is not None:
                    candidate = dict(candidate, implementation=declaration['definition']['implementation'])
                decision = op.engines.decisions.get(f'{op.name}@{rows}')
                if decision is not None:
                    decision['implementations'][op.engines.engines.index(engine)] = candidate
                shares.append((engine, shard, start, end, declaration))
            start = end

        from .host import declare
        def function(value):
            sources, pieces = {dtype: value}, []
            for engine, shard, first, end, declaration in shares:
                needed = engine.input_dtype(dtype) if declaration is None else declaration['input_dtype']
                if needed not in sources:
                    sources[needed] = value.to(dtype=needed, device=output_device)
                source = sources[needed]
                shape = (rows, end - first if op.combine == 'concat' else op.width)
                out = torch.empty(shape, dtype=engine.out_dtype(dtype) if declaration is None else declaration['output_dtype'], device=output_device)
                if declaration is None:
                    engine.run(op, shard, source, out)
                else:
                    source_name, = [name for name in declaration['input_layouts'] if name not in declaration['weights']]
                    target_name, = declaration['output_layouts']
                    out = declare({source_name: source, **declaration['weights']}, {target_name: out}, declaration['definition'],
                                  input_layouts=declaration['input_layouts'], output_layouts=declaration['output_layouts'],
                                  binding_layout=declaration['binding_layout'])[target_name]
                pieces.append(out)
            if op.combine == 'concat':
                return tuple(op.finish(piece.to(dtype), first) for piece, (_, _, first, _, _) in zip(pieces, shares))
            y = pieces[0].to(accumulator) if pieces else torch.zeros((rows, op.width), dtype=accumulator, device=output_device)
            for piece in pieces[1:]:
                y = y + piece.to(accumulator)
            return op.finish(y.to(dtype))

        from .declarations import Function
        from .operand import TensorView
        destinations = ([TensorView.of(self.output).slice(1, first, end - first) for _, _, first, end, _ in shares]
                        if op.combine == 'concat' else [self.output])
        self.function = Function(function, x, configuration=configuration,
                                 source_scope=dict(scope=op.engines.source_scope, operation=op.name))
        self.function.declare(None, destinations=destinations)

    def append(self, schedule):
        self.function.append(schedule)


class Linear(Operation):
    """y = x W^T (+ b), W [N, K]: its N outputs shared (they concatenate)."""
    kind, combine = 'linear', 'concat'

    def __init__(self, weight, bias=None, engines=None, name=None, grain=64, constant=True):
        self.weight, self.bias = weight, bias
        super().__init__(weight.shape[0], weight.shape[0], engines, name, grain, constant)

    def torch(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def finish(self, y, first=0):
        return y if self.bias is None else y + self.bias.narrow(0, first, y.shape[-1])


class FFN(Operation):
    """y = down(gelu_tanh(x Wg^T) * (x Wu^T)), Wg and Wu [I, H], Wd [H, I]: its I intermediate neurons shared, each
    engine's share a partial of the whole output (they sum): Megatron's column-then-row split [Shoeybi et al. 2019].
    The operands are any tensors: coded matrices (torch_mesh.streamed) share their neurons as their codes' rows and
    columns (grain a multiple of 32), the GPU's share their fused couple, another engine's share decoded."""
    kind, combine = 'ffn', 'sum'

    def __init__(self, gate, up, down, engines=None, name=None, grain=128, constant=True):
        self.gate, self.up, self.down = gate, up, down
        super().__init__(gate.shape[0], down.shape[0], engines, name, grain, constant)

    def torch(self, x):
        return ffn(x.reshape(-1, x.shape[-1]), self.gate, self.up, self.down).reshape(*x.shape[:-1], -1)


def placement(engines, observations, *, step, family):
    # design/whole-function.md#numerical-operands
    from assignment import assign, domains
    decisions = engines.decisions
    assigned = assign(domains(decisions, {}, engines.parameters), observations, step=step, family=family)
    for key, parts in assigned['parts'].items():
        if key in engines.decisions:
            engines.decisions[key]['parts'] = parts
    for key, value in assigned['operands'].items():
        engines.parameters[key]['value'] = value
    return assigned
