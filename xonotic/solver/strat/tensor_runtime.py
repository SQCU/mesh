import ctypes as c
import math
from pathlib import Path

import numpy as np

from mesh import ABSENT, WRITING, Metadata, RowMap, RowFunction, RowBinding, Rows, MemorySpan, _lib
from .tensor import Dimension, Tensor
from .tensor_metal import source, native_encoder


class View(c.Structure):
    _fields_ = [('offset', c.c_uint64), ('size', c.c_uint64), ('shape', c.c_uint64 * 8),
        ('stride', c.c_uint64 * 8)] + [(name, c.c_uint32) for name in ('dtype', 'rank', 'first', 'page_bytes')]


# ../../../design/algorithm-sources.md#literal-row-functions
def leaves(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for part in value.values(): yield from leaves(part)
    elif isinstance(value, (tuple, list)):
        for part in value: yield from leaves(part)


# ../../../design/algorithm-sources.md#literal-row-functions
def map_leaves(function, value):
    if isinstance(value, Tensor): return function(value)
    if isinstance(value, dict): return {name: map_leaves(function, part) for name, part in value.items()}
    if isinstance(value, tuple) and hasattr(value, '_fields'): return type(value)(*(map_leaves(function, part) for part in value))
    if isinstance(value, (tuple, list)): return type(value)(map_leaves(function, part) for part in value)
    return value


class PageLease:
    # ../../../design/algorithm-sources.md#contiguous-backing-page-views
    def __init__(self, pages, mapping):
        self.pages = pages
        self.mapping = mapping
        self.physical = tuple(pages.contents.table[mapping.first + index].page for index in range(mapping.count))
        spans = (MemorySpan * mapping.count)(*(MemorySpan(_lib.mesh_row_data(pages, mapping.first + index), pages.contents.bytes) for index in range(mapping.count)))
        address, length = c.c_void_p(), c.c_size_t()
        status = _lib.mesh_memory_view(spans, len(spans), c.byref(address), c.byref(length))
        if status: raise OSError(status, 'returned page view mapping')
        self.address, self.length = address.value, length.value

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def __del__(self):
        if hasattr(self, 'address'):
            _lib.mesh_memory_release(self.address, self.length)
            for index in range(self.mapping.count): _lib.mesh_row_release(self.pages, self.mapping.first + index)

    # ../../../design/algorithm-sources.md#contiguous-backing-page-views
    def array(self, shape, dtype):
        storage = (c.c_uint8 * self.length).from_address(self.address)
        storage.owner = self
        return np.ndarray(shape, dtype=dtype, buffer=storage)


class Realization:
    # ../../../design/algorithm-sources.md#complete-page-ownership
    def __init__(self, executable, name, ordinal, owner_nodes):
        self.executable, self.name, self.ordinal = executable, name, ordinal
        self.context = executable.context
        memory = self.context.contents.M.contents
        self.node, self.page_bytes = memory.node, memory.pgsz
        self.owner_nodes = owner_nodes
        needed = executable.dependencies(executable.exports[name])
        roots = executable.roots
        self.used = {roots[index] for index in needed}
        self.inputs = {root for root in self.used if executable.graph.nodes[root][1] in ('input', 'constant')}
        local = {root for root in self.used if executable.owner(root) == self.node}
        consumers = {root: set() for root in self.used}
        for index in needed:
            if index in executable.by_node:
                owner = executable.owner(index)
                for value in executable.graph.nodes[index][2]: consumers[roots[value.index]].add(owner)
        for value in leaves(executable.exports[name]): consumers[roots[value.index]].add(owner_nodes[0])
        self.transfers = [(root, executable.owner(root), peer)
            for root in sorted(self.used) for peer in sorted(consumers[root])
            if peer != executable.owner(root)]
        self.used = local | {root for root in self.used if self.node in consumers[root]}
        counts = {root: max(1, (math.prod(executable.shapes[root]) * np.dtype(executable.graph.nodes[root][0].dtype).itemsize + self.page_bytes - 1) // self.page_bytes) for root in self.used}
        local_functions = [root for root in sorted(local) if root in executable.by_node or root in self.inputs]
        numerical = [index for index in needed if index in executable.by_node]
        metadata_transfers = [(len(executable.graph.nodes) + index, executable.owner(index), owner_nodes[0])
            for index in numerical if executable.owner(index) != owner_nodes[0]]
        self.transfers.extend(metadata_transfers)
        remote_metadata = [index for index in numerical if executable.owner(index) != self.node] if self.node == owner_nodes[0] else []
        self.pages = executable.pages
        self.maps, self.arrays = {}, {}
        first = executable.next_row
        for root in sorted(self.used):
            shared = executable.storage.get(root) if root in executable.shared else None
            physical = shared[0][0] if shared is not None else self.allocate(counts[root]) if root in local else 0
            self.maps[root] = RowMap(first, counts[root], 0, physical, 0, 0, None)
            if root in local:
                address = c.addressof(memory) + memory.data_off + physical * self.page_bytes
                storage = (c.c_uint8 * (counts[root] * self.page_bytes)).from_address(address)
                storage.owner = self.pages
                self.arrays[root] = np.ndarray(executable.shapes[root], dtype=executable.graph.nodes[root][0].dtype, buffer=storage)
                if root in executable.shared: executable.storage[root] = tuple(range(physical, physical + counts[root])), self.arrays[root]
            first += counts[root]
        views = []
        for value, _, _, _, _ in executable.graph.nodes:
            shape = executable.shapes[value.index]
            view = View(size=math.prod(shape), dtype=np.dtype(value.dtype).itemsize, rank=len(shape), page_bytes=self.page_bytes)
            root = roots[value.index]
            if root in self.maps:
                mapping = self.maps[root]
                view.first = mapping.first
            stride = 1
            for axis in reversed(range(len(shape))):
                view.shape[axis], view.stride[axis] = shape[axis], stride
                stride *= shape[axis]
            views.append(view)
            if root in self.arrays: self.arrays[value.index] = self.arrays[root].reshape(shape)
        self.views = (View * len(views))(*views)
        self.functions = (RowFunction * len(local_functions))()
        self.function_maps, self.metadata, self.calls = [], [], []
        for index in remote_metadata:
            key = len(executable.graph.nodes) + index
            mapping = RowMap(first, 1, 0, 0, 0, 0, None)
            first += 1
            self.maps[key] = mapping
            self.metadata.append(mapping)
        self.returns = [self.maps[roots[value.index]] for value in leaves(executable.exports[name]) if roots[value.index] in self.maps and self.node == owner_nodes[0]]
        self.returns = list({mapping.first: mapping for mapping in self.returns}.values())
        self.returns.extend(self.metadata)
        self.retained_inputs = []
        for index, root in enumerate(local_functions):
            node = executable.graph.nodes[root]
            inputs = (RowMap * len(node[2]))(*(self.maps[roots[value.index]] for value in node[2]))
            metadata = None if root in self.inputs else RowMap(first, 1, 0, self.allocate(1), 0, 0, None)
            first += int(metadata is not None)
            mapping = self.maps[root]
            if root in self.inputs:
                outputs = (RowMap * mapping.count)(*(RowMap(mapping.first + page, 1, 0, mapping.physical + page, 0, 0, None) for page in range(mapping.count)))
            else:
                outputs = (RowMap * 2)(mapping, metadata)
            self.function_maps.append((inputs, outputs))
            self.functions[index] = RowFunction(inputs, outputs, len(inputs), len(outputs), 1)
            if root in self.inputs:
                self.retained_inputs.append(mapping)
                self.returns.append(mapping)
                self.calls.append((index, None))
            else:
                if self.node == owner_nodes[0]:
                    self.metadata.append(metadata); self.returns.append(metadata)
                else:
                    key = len(executable.graph.nodes) + root
                    self.maps[key] = metadata
                self.calls.append((index, root))
        self.sources = [(index, root) for index, root in enumerate(local_functions) if root in self.inputs]
        self.return_maps = (RowMap * len(self.returns))(*self.returns)
        executable.next_row = first
        self.bindings = None
        self.stamp = 0
        self.indices = [(c.c_uint32 * 1)() for _ in self.calls]

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def __del__(self):
        if getattr(self, 'native', None): self.native.graph_free(self.handle)

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def allocate(self, count):
        physical = _lib.mesh_rows_allocate(self.pages, count, self.page_bytes)
        if physical == ABSENT: raise OSError(c.get_errno(), 'tensor operand allocation')
        return physical

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def bind(self, tables, offsets):
        bindings = []
        for root, source_node, destination in self.transfers:
            if self.node not in (source_node, destination): continue
            mapping = self.maps[root]
            receive = int(self.node == destination)
            peer = source_node if receive else destination
            remote = offsets[peer][self.name][self.ordinal] + [item for item in self.transfers if peer in item[1:]].index((root, source_node, destination))
            bindings.append(RowBinding(mapping.first, mapping.count, remote, 0, peer, receive, tables[peer][self.name][self.ordinal]))
        self.bindings = (RowBinding * len(bindings))(*bindings)

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def configure(self):
        self.native, self.handle, self.encoders = native_encoder(self)

    # ../../../design/algorithm-sources.md#literal-row-functions
    def scan(self):
        for index, native in self.calls:
            if native is None: continue
            function = self.functions[index]
            stamp = self.stamp if self.node == self.owner_nodes[0] else self.pages.contents.table[function.input[0].first].stamp
            selected = _lib.mesh_rows_issue(self.pages, c.byref(function), stamp, self.indices[index], 1)
            if selected:
                self.encoders[index](self.handle, stamp)

    # ../../../design/algorithm-sources.md#literal-row-functions
    def present(self):
        return all(_lib.mesh_rows_present(self.pages, mapping, 0, self.stamp) for mapping in self.return_maps)

    # ../../../design/algorithm-sources.md#contiguous-backing-page-views
    def values(self):
        values = {}
        for value in leaves(self.executable.exports[self.name]):
            root = self.executable.roots[value.index]
            if root not in values:
                lease = PageLease(self.pages, self.maps[root])
                if root in self.inputs and root in self.executable.storage: lease.backing = self.executable.storage[root][1]
                values[root] = lease.array(self.executable.shapes[root], self.executable.graph.nodes[root][0].dtype)
        metadata = []
        for mapping in self.metadata:
            lease = PageLease(self.pages, mapping)
            value = Metadata.from_address(lease.address)
            value.owner = lease
            metadata.append(value)
        return map_leaves(lambda value: values[self.executable.roots[value.index]].reshape(self.executable.shapes[value.index]), self.executable.exports[self.name]), tuple(metadata)


class Executable:
    # ../../../design/algorithm-sources.md#complete-page-ownership
    def __init__(self, graph, exports, owner_nodes=None):
        self.graph, self.exports = graph, exports
        self.context = _lib.mesh_context()
        status = _lib.mesh_attach(self.context, None)
        if status: raise OSError(status, 'tensor mesh attachment')
        self.text, self.kernels = source(graph)
        self.by_node = {item['node']: item for item in self.kernels}
        arguments = []
        for item in self.kernels:
            item['argument_offset'] = len(arguments); arguments.extend(item['arguments'])
            item['zero_offset'] = len(arguments); arguments.append(item['node'])
        self.arguments = (c.c_uint32 * len(arguments))(*arguments)
        self.roots = {}
        for value, operation, inputs, _, _ in graph.nodes:
            self.roots[value.index] = self.roots[inputs[0].index] if operation in ('reshape', 'stop_gradient') else value.index
        self.owner_nodes = owner_nodes or {0: self.context.contents.M.contents.node, **{region['owner']: region['peer'] for region in graph.regions.values()}}
        self.shared = {self.roots[value.index] for name, value in graph.inputs.items() if name.startswith(('parameter.', 'optimizer.', 'accumulator.'))}
        self.shared.update(self.roots[value.index] for value, _ in graph.constants.values())
        self.input_names = {value.index: name for name, value in graph.inputs.items()}
        self.storage = {}
        self.capacity = None
        self.realizations, self.arrays = {}, {}
        self.generations = {name: 0 for name in exports}
        self.submissions = 0
        self.progress = None
        self.cancel = None
        self.remote = None

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def owner(self, index):
        operation, owner = self.graph.nodes[index][1], self.graph.nodes[index][4]
        return self.owner_nodes[0 if operation in ('input', 'constant', 'dimension') else owner]

    # ../../../design/algorithm-sources.md#literal-row-functions
    def dependencies(self, outputs):
        needed = set()
        # ../../../design/algorithm-sources.md#literal-row-functions
        def visit(value):
            if value.index in needed: return
            needed.add(value.index)
            for dependency in self.graph.nodes[value.index][2]: visit(dependency)
        for value in leaves(outputs): visit(value)
        return tuple(sorted(needed))

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def reserve(self, capacity, configure=True):
        capacity = tuple(capacity)
        if capacity == self.capacity: return
        self.capacity = capacity
        self.shapes = {value.index: tuple(size.resolve(capacity) if isinstance(size, Dimension) else size for size in value.shape) for value, _, _, _, _ in self.graph.nodes}
        count = sum(max(1, (math.prod(self.shapes[value.index]) * np.dtype(value.dtype).itemsize + self.context.contents.M.contents.pgsz - 1) // self.context.contents.M.contents.pgsz) + 3 for value, _, _, _, _ in self.graph.nodes)
        self.pages = _lib.mesh_rows_create(self.context, max(1, count * 2 * len(self.exports)), self.context.contents.table_count)
        if not self.pages: raise OSError(c.get_errno(), 'complete tensor graph page table')
        self.next_row = 0
        self.realizations = {}
        self.native_memory = None
        for name, outputs in self.exports.items():
            self.realizations[name] = [Realization(self, name, index, self.owner_nodes) for index in range(2)]
        self.arrays = {}
        for plans in self.realizations.values():
            for plan in plans: self.arrays.update(plan.arrays)
        for value, data in self.graph.constants.values():
            if value.index in self.arrays: np.copyto(self.arrays[value.index], data)
        for name, (value, initial) in self.graph.parameters.items():
            if value.index in self.arrays:
                np.copyto(self.arrays[value.index], np.asarray(initial))
                self.graph.parameters[name] = value, self.arrays[value.index]
        if configure:
            from .tensor_mesh import RemoteProgram
            self.remote = RemoteProgram(self)
            self.remote.configure()

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def table_ids(self):
        return {name: [plan.pages.contents.identity for plan in plans] for name, plans in self.realizations.items()}

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def configure(self, tables):
        offsets = {peer: {} for peer in tables}
        for peer in tables:
            offset = 0
            for name, plans in self.realizations.items():
                offsets[peer][name] = []
                for plan in plans:
                    offsets[peer][name].append(offset)
                    offset += sum(peer in transfer[1:] for transfer in plan.transfers)
        plans = [plan for values in self.realizations.values() for plan in values]
        for plan in plans: plan.bind(tables, offsets)
        self.functions = (RowFunction * sum(len(plan.functions) for plan in plans))(*(function for plan in plans for function in plan.functions))
        self.bindings = (RowBinding * sum(len(plan.bindings) for plan in plans))(*(binding for plan in plans for binding in plan.bindings))
        self.returns = (RowMap * sum(len(plan.return_maps) for plan in plans))(*(mapping for plan in plans for mapping in plan.return_maps))
        status = _lib.mesh_rows_realize(self.pages, self.functions, len(self.functions), self.bindings, len(self.bindings), self.returns, len(self.returns))
        if status: raise OSError(status, 'complete tensor graph realization')
        offset = 0
        for plan in plans:
            plan.return_maps = tuple(self.returns[offset:offset + len(plan.return_maps)])
            offset += len(plan.return_maps)
            for index, root in plan.sources:
                if self.graph.nodes[root][1] == 'constant':
                    mapping = plan.maps[root]
                    _lib.mesh_rows_map(plan.pages, mapping.first, mapping.physical, mapping.count, mapping.uses, WRITING)
                    _lib.mesh_rows_constant(plan.pages, mapping.first, mapping.count)
            plan.configuration = self.functions, self.bindings, self.returns
            plan.configure()

    # ../../../design/algorithm-sources.md#literal-row-functions
    def commands(self, plan, index):
        item = self.by_node[index]
        view = plan.views[index]
        commands = []
        if item['clear']:
            commands.append(('mesh_tensor_zero', item['zero_offset'], (max(1, (view.size * view.dtype + 255) // 256), 1, 1), (256, 1, 1)))
        mode, threads = item['mode'], 256
        shape = self.shapes[index]
        if mode == 'matmul': grid = ((shape[-1] + 31) // 32, (shape[-2] + 63) // 64, math.prod(shape[:-2]))
        elif isinstance(mode, tuple) and mode[0] == 'expert': grid = ((shape[-1] + 31) // 32, (shape[-2] + 63) // 64, self.shapes[mode[1]][0])
        elif mode == 'reduce': grid = (view.size, 1, 1)
        elif isinstance(mode, tuple) and mode[0] in ('neighborhood', 'edges'):
            threads = 32
            grid = (self.shapes[mode[1]][0] if mode[0] == 'neighborhood' else plan.views[mode[1]].size, 1, 1)
        else: grid = (((plan.views[mode[1]].size if isinstance(mode, tuple) else view.size) + 255) // 256, 1, 1)
        commands.append((item['name'], item['argument_offset'], tuple(max(1, value) for value in grid), (threads, 1, 1)))
        return commands

    # ../../../design/algorithm-sources.md#literal-row-functions
    def scan(self):
        for plans in self.realizations.values():
            for plan in plans: plan.scan()

    # ../../../design/algorithm-sources.md#literal-row-functions
    def submit(self, name):
        plan = self.realizations[name][self.generations[name] % 2]
        plan.stamp += 1
        self.generations[name] += 1
        self.submissions += 1
        return plan, plan.metadata

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def write_inputs(self, plan, values):
        for index, root in plan.sources:
            if self.graph.nodes[root][1] == 'constant': continue
            function = plan.functions[index]
            selected = _lib.mesh_rows_issue(plan.pages, c.byref(function), plan.stamp, plan.indices[index], 1)
            if not selected: continue
            for page, mapping in enumerate(plan.function_maps[index][1]):
                physical = self.storage[root][0][page] if root in self.storage else mapping.physical
                _lib.mesh_rows_map(plan.pages, mapping.first, physical, 1, mapping.uses, WRITING | plan.stamp)
            name = self.input_names.get(root)
            if name in values:
                array = self.storage[root][1] if root in self.storage else plan.arrays[root]
                np.copyto(array, np.asarray(values[name]))
            _lib.mesh_rows_complete(plan.pages, c.byref(function), plan.stamp, plan.indices[index], selected)

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def adopt(self, target, value, plan, array):
        root = self.roots[target]
        mapping = plan.maps[self.roots[value]]
        physical = tuple(plan.pages.contents.table[mapping.first + index].page for index in range(mapping.count))
        self.storage[root] = physical, array
        for name, (parameter, _) in self.graph.parameters.items():
            if self.roots[parameter.index] == root: self.graph.parameters[name] = parameter, array
        for index, alias in self.roots.items():
            if alias == root: self.arrays[index] = array.reshape(self.shapes[index])

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def close(self):
        if not self.pages: return
        _lib.mesh_rows_retire(self.pages)
        self.pages = None
        self.realizations.clear()
        self.arrays.clear()
        self.storage.clear()

    # ../../../design/algorithm-sources.md#asynchronous-metadata-publication
    def report(self):
        return {'backend': 'mesh_page_metal', 'graph_nodes': len(self.graph.nodes), 'kernel_variants': len(self.kernels), 'submissions': self.submissions,
            'functions': {name: len(plans[0].functions) if plans else 0 for name, plans in self.realizations.items()}}

    # ../../../design/algorithm-sources.md#literal-row-functions
    def work(self, name):
        flops = moved = rows = 0
        for index in self.dependencies(self.exports[name]):
            output, operation, inputs, attributes, _ = self.graph.nodes[index]
            size = math.prod(self.shapes[index])
            moved += size * np.dtype(output.dtype).itemsize + sum(math.prod(self.shapes[value.index]) * np.dtype(value.dtype).itemsize for value in inputs)
            if operation in ('matmul', 'expert_matmul'):
                inner = self.shapes[inputs[0].index][-2 if attributes.get('transpose_left') else -1]
                flops += 2 * size * inner
                rows += math.prod(self.shapes[index][:-1])
            elif operation in ('expert_input_vjp', 'expert_weight_vjp'):
                source_shape, weights = (self.shapes[value.index] for value in inputs[:2])
                flops += 2 * source_shape[0] * weights[1] * weights[2]
                rows += math.prod(self.shapes[index][:-1])
        return {'flops': flops, 'bytes': moved, 'rows': rows}
