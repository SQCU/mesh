import ctypes as c
import math
import itertools
from pathlib import Path
import time

import numpy as np

from .tensor import Dimension, Tensor
from .tensor_metal import source


class View(c.Structure):
    _fields_ = [('offset', c.c_uint64), ('size', c.c_uint64), ('shape', c.c_uint64 * 8),
        ('stride', c.c_uint64 * 8), ('dtype', c.c_uint32), ('rank', c.c_uint32),
        ('pool', c.c_uint32), ('reserved', c.c_uint32), ('page_start', c.c_uint64),
        ('origin', c.c_uint64), ('page_stride', c.c_uint32), ('payload', c.c_uint32)]


class Command(c.Structure):
    _fields_ = [('kernel', c.c_uint32), ('argument_offset', c.c_uint32),
               ('grid', c.c_uint32 * 3), ('group', c.c_uint32 * 3)]


_LIBRARY = None


def library():
    global _LIBRARY
    if _LIBRARY is None:
        path = Path(__file__).resolve().parents[3] / 'rdma' / 'libmesh-tensor.dylib'
        _LIBRARY = c.CDLL(str(path), use_errno=True)
        declarations = {
            'create': (c.c_void_p, [c.c_char_p]), 'error': (c.c_char_p, [c.c_void_p]),
            'kernel': (c.c_int, [c.c_void_p, c.c_char_p]),
            'reserve': (c.c_int, [c.c_void_p, c.c_size_t, c.POINTER(View), c.c_size_t,
                                c.POINTER(c.c_uint64), c.c_size_t, c.POINTER(c.c_uint32), c.c_size_t]),
            'data': (c.c_void_p, [c.c_void_p]),
            'memory': (c.c_void_p, [c.c_void_p]), 'memory_free': (None, [c.c_void_p]),
            'phase': (c.c_int, [c.c_void_p, c.c_uint32, c.POINTER(Command), c.c_size_t]),
            'submit': (c.c_int, [c.c_void_p, c.c_uint32]), 'status': (c.c_int, [c.c_void_p]),
            'free': (None, [c.c_void_p]),
        }
        for name, (result, arguments) in declarations.items():
            function = getattr(_LIBRARY, 'mesh_tensor_' + name)
            function.restype, function.argtypes = result, arguments
    return _LIBRARY


def leaves(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for part in value.values(): yield from leaves(part)
    elif isinstance(value, (tuple, list)):
        for part in value: yield from leaves(part)


class Memory:
    def __init__(self, lib, program):
        self.lib, self.handle = lib, lib.mesh_tensor_memory(program)

    def __del__(self):
        self.lib.mesh_tensor_memory_free(self.handle)


def map_leaves(function, value):
    if isinstance(value, Tensor): return function(value)
    if isinstance(value, dict): return {name: map_leaves(function, part) for name, part in value.items()}
    if isinstance(value, tuple) and hasattr(value, '_fields'): return type(value)(*(map_leaves(function, part) for part in value))
    if isinstance(value, (tuple, list)): return type(value)(map_leaves(function, part) for part in value)
    return value


class Executable:
    def __del__(self):
        handle = getattr(self, 'handle', None)
        if handle: self.lib.mesh_tensor_free(handle)

    def __init__(self, graph, phases, schedule=None):
        self.graph, self.phases = graph, phases
        self.lib = library()
        text, self.kernels = source(graph)
        self.handle = self.lib.mesh_tensor_create(text.encode())
        self.check(0 if self.handle else -1)
        self.kernel_ids = {}
        for name in dict.fromkeys(['mesh_tensor_zero'] + [item['name'] for item in self.kernels]):
            index = self.lib.mesh_tensor_kernel(self.handle, name.encode())
            self.check(index if index < 0 else 0)
            self.kernel_ids[name] = index
        self.by_node = {item['node']: item for item in self.kernels}
        arguments = []
        for item in self.kernels:
            item['argument_offset'] = len(arguments)
            arguments.extend(item['arguments'])
            item['zero_offset'] = len(arguments)
            arguments.append(item['node'])
        self.arguments = (c.c_uint32 * len(arguments))(*arguments)
        self.schedule = dict(schedule) if schedule is not None else {name: self.dependencies(value) for name, value in phases.items()}
        self.segments = {}
        for name, indices in tuple(self.schedule.items()):
            segments = []
            if graph.regions:
                for part, (owner, group) in enumerate(itertools.groupby(indices, key=lambda index: graph.nodes[index][4])):
                    key = f'{name}.{part}'
                    members = tuple(group)
                    self.schedule[key] = members
                    segments.append((owner, key, members))
            self.segments[name] = segments
        self.phase_ids = {name: index for index, name in enumerate(self.schedule)}
        self.capacity = None
        self.arrays = {}
        self.allocations = self.bytes = self.submissions = 0
        self.progress = None
        self.cancel = None
        self.remote = None
        aliases = {}
        self.alias_groups = {}
        for value, operation, inputs, _, _ in graph.nodes:
            root = aliases[inputs[0].index] if operation in ('reshape', 'stop_gradient') else value.index
            aliases[value.index] = root
            self.alias_groups.setdefault(root, []).append(value.index)
        if graph.regions:
            from .tensor_mesh import RemoteProgram
            self.remote = RemoteProgram(self)

    def check(self, status):
        if status:
            message = self.lib.mesh_tensor_error(self.handle)
            raise RuntimeError(message.decode() if message else f'Metal tensor operation failed: {status}')

    def dependencies(self, outputs):
        needed = set()
        def visit(value):
            if value.index in needed: return
            needed.add(value.index)
            for dependency in self.graph.nodes[value.index][2]: visit(dependency)
        for value in leaves(outputs): visit(value)
        return tuple(index for index in sorted(needed) if index in self.by_node)

    def reserve(self, capacity):
        capacity = tuple(capacity)
        if capacity == self.capacity:
            return
        self.finish()
        retained = {name: self.arrays[value.index].copy() for name, value in self.graph.inputs.items()
                    if value.index in self.arrays and (name.startswith('parameter.') or name.startswith('optimizer.') or name.startswith('accumulator.'))}
        shapes = {value.index: tuple(size.resolve(capacity) if isinstance(size, Dimension) else size for size in value.shape)
                  for value, _, _, _, _ in self.graph.nodes}
        roots = {}
        for value, operation, inputs, _, _ in self.graph.nodes:
            roots[value.index] = roots[inputs[1].index] if operation == 'assign' else roots[inputs[0].index] if operation in ('reshape', 'stop_gradient') else value.index
        active = {index for indices in self.schedule.values() for index in indices}
        used = set(active)
        for index in active:
            used.update(value.index for value in self.graph.nodes[index][2])
        used.update(value.index for outputs in self.phases.values() for value in leaves(outputs))
        used.update(roots[index] for index in tuple(used))
        last = {roots[index]: roots[index] for index in used}
        for index in active:
            for value in self.graph.nodes[index][2]:
                last[roots[value.index]] = max(last[roots[value.index]], index)
        pinned = {roots[index] for index in used if self.graph.nodes[roots[index]][1] in ('input', 'constant')}
        pinned.update(roots[value.index] for outputs in self.phases.values() for value in leaves(outputs))
        offsets, available, occupied = {}, [], []
        offset = 0
        order = sorted(last, key=lambda index: (-1 if index in pinned else index, index))
        for index in order:
            start = -1 if index in pinned else index
            retained_blocks = []
            for end, address, length in occupied:
                if end < start: available.append((address, length))
                else: retained_blocks.append((end, address, length))
            occupied = retained_blocks
            length = max(256, (math.prod(shapes[index]) * np.dtype(self.graph.nodes[index][0].dtype).itemsize + 255) // 256 * 256)
            available.sort()
            joined = []
            for address, extent in available:
                if joined and joined[-1][0] + joined[-1][1] == address:
                    joined[-1] = joined[-1][0], joined[-1][1] + extent
                else: joined.append((address, extent))
            available = joined
            fit = next((i for i, (_, extent) in enumerate(available) if extent >= length), None)
            if fit is None:
                address, offset = offset, offset + length
            else:
                address, extent = available.pop(fit)
                if extent > length: available.append((address + length, extent - length))
            offsets[index] = address
            occupied.append((float('inf') if index in pinned else last[index], address, length))
        views = []
        for value, operation, inputs, _, _ in self.graph.nodes:
            shape = shapes[value.index]
            stride, strides = 1, []
            for length in reversed(shape):
                strides.append(stride)
                stride *= length
            view = View(offset=offsets.get(roots[value.index], 0), size=math.prod(shape), rank=len(shape))
            for axis, length in enumerate(shape):
                view.shape[axis], view.stride[axis] = length, strides[len(shape)-axis-1]
            views.append(view)
        self.staging_offset = offset
        if self.remote is not None:
            offset += max((sum(math.prod(shapes[index]) * np.dtype(self.graph.nodes[index][0].dtype).itemsize
                for index in self.remote.exports[key][1]) for segments in self.segments.values()
                for owner, key, _ in segments if owner), default=0)
        self.views = (View * len(views))(*views)
        dims = (c.c_uint64 * len(capacity))(*capacity)
        self.check(self.lib.mesh_tensor_reserve(self.handle, offset, self.views, len(views),
            dims, len(capacity), self.arguments, len(self.arguments)))
        base = self.lib.mesh_tensor_data(self.handle)
        self.memory = (c.c_ubyte * offset).from_address(base)
        self.memory.owner = Memory(self.lib, self.handle)
        self.arrays = {value.index: np.ndarray(shapes[value.index], dtype=value.dtype,
                       buffer=self.memory, offset=views[value.index].offset) for value, _, _, _, _ in self.graph.nodes if value.index in used}
        for value, data in self.graph.constants.values():
            if value.index in self.arrays: np.copyto(self.arrays[value.index], data)
        for name, (value, initial) in self.graph.parameters.items():
            np.copyto(self.arrays[value.index], retained['parameter.' + name] if 'parameter.' + name in retained else np.asarray(initial))
            self.graph.parameters[name] = value, self.arrays[value.index]
        for name, value in self.graph.inputs.items():
            if value.index not in self.arrays: continue
            if name.startswith('optimizer.') or name.startswith('accumulator.'):
                self.arrays[value.index].fill(0)
                if name in retained: np.copyto(self.arrays[value.index], retained[name])
            elif name in retained:
                np.copyto(self.arrays[value.index], retained[name])
        for name, indices in self.schedule.items():
            commands = []
            for index in indices:
                item = self.by_node[index]
                if item['clear']:
                    commands.append(Command(self.kernel_ids['mesh_tensor_zero'], item['zero_offset'],
                        (c.c_uint32 * 3)(max(1, (views[index].size + 255)//256), 1, 1), (c.c_uint32 * 3)(256, 1, 1)))
                mode = item['mode']
                threads = 256
                if mode == 'matmul':
                    shape = shapes[index]
                    grid = ((shape[-1]+31)//32, (shape[-2]+63)//64, math.prod(shape[:-2]))
                elif isinstance(mode, tuple) and mode[0] == 'expert':
                    shape = shapes[index]
                    grid = ((shape[-1]+31)//32, (shape[-2]+63)//64, shapes[mode[1]][0])
                elif mode == 'reduce':
                    grid = (views[index].size, 1, 1)
                elif isinstance(mode, tuple) and mode[0] in ('neighborhood', 'edges'):
                    threads = 32
                    grid = ((shapes[mode[1]][0] if mode[0] == 'neighborhood' else views[mode[1]].size), 1, 1)
                else:
                    size = views[mode[1]].size if isinstance(mode, tuple) else views[index].size
                    grid = ((size+255)//256, 1, 1)
                commands.append(Command(self.kernel_ids[item['name']], item['argument_offset'],
                    (c.c_uint32 * 3)(*(max(1, value) for value in grid)), (c.c_uint32 * 3)(threads, 1, 1)))
            self.check(self.lib.mesh_tensor_phase(self.handle, self.phase_ids[name], (Command * len(commands))(*commands), len(commands)))
        self.allocations += 1
        self.bytes = offset
        self.capacity = capacity
        if self.graph.regions:
            self.remote.reserve()

    def load(self, values):
        self.finish()
        for name, value in values.items():
            index = self.graph.inputs[name].index
            if index in self.arrays:
                np.copyto(self.arrays[index], np.asarray(value))

    def submit(self, phase):
        self.finish()
        if self.remote is None:
            self.check(self.lib.mesh_tensor_submit(self.handle, self.phase_ids[phase]))
        else:
            self.remote.run(phase)
        self.submissions += 1

    def finish(self, timeout=30):
        started = time.monotonic()
        status = self.lib.mesh_tensor_status(self.handle)
        while status == 0:
            if self.progress is not None: self.progress()
            if (self.cancel is not None and self.cancel()) or time.monotonic() - started >= timeout:
                raise TimeoutError('submitted tensor program remains in flight; its buffers remain borrowed')
            time.sleep(.0001)
            status = self.lib.mesh_tensor_status(self.handle)
        self.check(status if status < 0 else 0)

    def run(self, phase, values=()):
        self.load(dict(values))
        self.submit(phase)
        self.finish()
        return map_leaves(lambda value: self.arrays[value.index], self.phases[phase])

    def report(self):
        return {'backend': 'persistent_metal', 'graph_nodes': len(self.graph.nodes),
                'kernel_variants': len(self.kernel_ids), 'arena_bytes': self.bytes,
                'arena_realizations': self.allocations, 'submissions': self.submissions,
                'mesh': {} if self.remote is None else self.remote.report(),
                'phases': {name: len(indices) for name, indices in self.schedule.items()}}

    def work(self, phase):
        flops = moved = rows = 0
        for index in self.schedule[phase]:
            output, operation, inputs, attributes, _ = self.graph.nodes[index]
            shape = self.arrays[index].shape
            size = self.arrays[index].size
            moved += self.arrays[index].nbytes + sum(self.arrays[value.index].nbytes for value in inputs)
            if operation == 'matmul':
                inner = self.arrays[inputs[0].index].shape[-2 if attributes.get('transpose_left') else -1]
                flops += 2 * size * inner
                rows += math.prod(shape[:-1])
            elif operation == 'expert_matmul':
                flops += 2 * size * self.arrays[inputs[0].index].shape[-1]
                rows += shape[0]
            elif operation in ('expert_input_vjp', 'expert_weight_vjp'):
                source, weights = (self.arrays[value.index].shape for value in inputs[:2])
                flops += 2 * source[0] * weights[1] * weights[2]
                rows += math.prod(shape[:-1])
        return {'flops': flops, 'bytes': moved, 'rows': rows}
