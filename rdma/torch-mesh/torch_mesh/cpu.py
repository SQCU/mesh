"""AOT C++ scalar commands over the numerical compiler's retained bodies.

Inductor supplies scalar/SIMD arithmetic, loop fusion and reduction code.  The
wrapper is a preparation program: final page relations become address source,
and one native entry receives its already-bound pointer/scalar record.
"""
from contextlib import contextmanager
import ctypes
import os
import shlex

import torch


@contextmanager
def configuration():
    from torch._inductor import config
    # Parallel work belongs to the enclosing declared command graph. Keep the
    # ordinary SIMD/tiling code generator, without an inner worker scheduler.
    with config.patch({'cpp.threads': 1, 'cpp.dynamic_threads': False,
                       'cpp.enable_kernel_profile': False,
                       'cpp.enable_kernel_context_guard': False}):
        yield


def compilation(implementation):
    """The upstream compiler's concrete command, retained as preparation data."""
    from torch._inductor.cpp_builder import CppBuilder, CppTorchOptions
    from torch._inductor.cpu_vec_isa import pick_vec_isa
    from mesh import RDMA
    build = implementation['build']
    libraries = [os.path.join(os.environ.get('MESH_LIBDIR', RDMA), name) for name in build['libraries']]
    with configuration():
        options = CppTorchOptions(vec_isa=pick_vec_isa(), include_pytorch=False,
                                  aot_mode=True, openmp=False,
                                  extra_flags=tuple(map(shlex.quote, (*libraries, *build['link_flags']))))
        builder = CppBuilder('mesh_cpu', ['MESH_CPU_SOURCE'], options,
                             output_dir='MESH_CPU_OUTPUT_DIRECTORY')
        target = builder.get_target_file_path()
        argv = shlex.split(builder.get_command_line())
    return dict(argv=[('source' if value == 'MESH_CPU_SOURCE' else
                       'output' if value == target else value) for value in argv],
                source_argument='source', output_argument='output')


def declaration(source, function, specs, access, accesses, implementation, compiler):
    """Return the ordinary preparation callable used by the source wrapper."""
    def bind(*values):
        from .native import _records
        from .operand import byte_operand
        operands, constants = [], {}
        for slot, (spec, value) in enumerate(zip(specs, values, strict=True)):
            operands.append(byte_operand(value) if spec['kind'] == 'pointer' else None)
            if spec['kind'] != 'pointer':
                constants[slot] = value
        _records.get().append(dict(kind='cpu', source=source, function=function,
            argument_specs=specs, operands=operands, constants=constants,
            names=tuple(spec['name'] for spec in specs if spec['kind'] == 'pointer'),
            types=tuple(spec['dtype'] for spec in specs if spec['kind'] == 'pointer'),
            binding_origins=tuple(0 if operand is not None else None for operand in operands),
            index_dtype=torch.uint8, access=access, accesses=accesses,
            implementation=implementation, compiler=compiler))
    return bind


def index_source(arguments, bindings):
    """One composed original-byte to physical-byte expression per argument."""
    result = []
    for argument in arguments:
        slot = argument['slot']
        original = argument['source'].components(argument['source_dtype'].itemsize)
        physical = bindings[slot][2].reshape(tuple(original.shape))
        result.append(physical.storage_byte_declaration(slot, original, argument['dtype'].itemsize))
    return ''.join(result)


PREAMBLE = '''#include <torch/csrc/inductor/cpp_prefix.h>
using std::min; using std::max;
template<class T> inline T select(T a, T b, bool condition) { return condition ? b : a; }
'''


def argument_record(command, pointer_values):
    """The native entry's fixed ABI and its explicit pointer relocations."""
    fields, values, places = [], {}, []
    for slot, spec in enumerate(command['argument_specs']):
        name = f'a{slot}'
        if spec['kind'] == 'pointer':
            fields.append((name, ctypes.c_void_p))
            values[name] = pointer_values[slot]
        else:
            kind = ctypes.c_float if spec['type'] == 'const float' else ctypes.c_int64
            fields.append((name, kind))
            values[name] = command['constants'][slot]
    record = type('CpuArguments', (ctypes.Structure,), {'_fields_': fields})
    for slot, spec in enumerate(command['argument_specs']):
        if spec['kind'] == 'pointer':
            places.append((getattr(record, f'a{slot}').offset, slot))
    return bytes(record(**values)), tuple(places)


def prepare(command):
    """Bind the final native instruction through the existing CPU program."""
    from .engines import cpu_native
    from . import _stream
    bindings = {slot: _stream.page_backing(operand)
                for slot, operand in enumerate(command['operands']) if operand is not None}
    arguments, pointers = argument_record(command,
        {slot: binding[0] for slot, binding in bindings.items()})
    source = PREAMBLE + index_source(command['arguments'], bindings) + command['source']
    reads, writes = [], []
    for slot, access, operand in command['reaches']:
        if access & 1:
            reads.append(operand)
        if access & 2:
            writes.append(operand)
    return cpu_native(source, command['compiler'], command['function'], arguments, [(offset, 0) for offset, slot in pointers],
                      implementation=command['implementation'], inputs=tuple(reads), outputs=tuple(writes))


def scheduling(storage_layouts, command_accesses, computed_domains, implementation):
    """The existing C++ scheduler, bound to one selected implementation."""
    import sympy
    from torch._inductor.codegen.cpp import CppScheduling, CppKernel, CppVecKernel, CppTile2DKernel, CppKernelProxy, KernelGroup
    from torch._inductor.codegen.common import BracesBuffer
    from torch._inductor.scheduler import SchedulerNode
    from torch._inductor.virtualized import V

    class PageIndices:
        def page_index(self, name, index):
            name = V.graph.scheduler.mutation_real_name.get(name, name)
            if name not in storage_layouts:
                raise ValueError(f'numerical access {name} has no assigned page operand')
            # This symbol is a source function, resolved to one final layout
            # expression before compilation. It is not a runtime table lookup.
            slots = self.args.__dict__.setdefault('mesh_page_slots', {})
            slot = slots.setdefault(name, len(slots))
            return sympy.Function(f'mesh_page_{slot}', integer=True)(index)

        def load(self, name, index):
            return super().load(name, self.page_index(name, index))

        def store(self, name, index, value, mode=None):
            return super().store(name, self.page_index(name, index), value, mode)

        def store_reduction(self, name, index, value):
            return super().store_reduction(name, self.page_index(name, index), value)

    class Scalar(PageIndices, CppKernel):
        pass

    class Vector(PageIndices, CppVecKernel):
        pass

    class Tile(PageIndices, CppTile2DKernel):
        pass

    class Proxy(CppKernelProxy):
        kernel_cls, vec_kernel_cls, tile2d_kernel_cls = Scalar, Vector, Tile

    class Group(KernelGroup):
        def codegen_group(self, name=None):
            self.stack.close()
            if not self.scheduled_nodes:
                return ''
            code = BracesBuffer()
            arg_defs, _, _ = self.args.cpp_argdefs()
            code.writeline(f'extern "C" void mesh_cpu_kernel({", ".join(arg_defs)})')
            with code.indent():
                for old, new in self.args.aliases():
                    code.writeline(f'auto {old} = {new};')
                code.splice(self.loops_code)
            return code.getvalue()

    class Source(CppScheduling):
        kernel_proxy_cls = Proxy

        def reset_kernel_group(self):
            with configuration():
                self.kernel_group = Group()

        def codegen_node(self, node):
            with configuration():
                return super().codegen_node(node)

        def flush(self):
            with configuration():
                return super().flush()

        def define_kernel(self, source, nodes, kernel_args=None):
            from mesh_relation import Algebra
            from . import _stream
            args = self.kernel_group.args if kernel_args is None else kernel_args
            _, calls, types = args.cpp_argdefs()
            pointers = [value for value, dtype in zip(calls, types, strict=True) if dtype.endswith('*')]
            specs = tuple(dict(slot=slot, kind='pointer' if dtype.endswith('*') else 'scalar',
                               name=value, type=dtype,
                               **({'dtype': V.graph.get_dtype(value)} if dtype.endswith('*') else {}))
                          for slot, (value, dtype) in enumerate(zip(calls, types, strict=True)))
            for name, slot in args.__dict__.get('mesh_page_slots', {}).items():
                source = source.replace(f'mesh_page_{slot}(', f'mesh_storage_{calls.index(name)}(')
            # The numerical body remains upstream source; the only outer ABI
            # is the same immutable native instruction argument record.
            fields = ''.join(f'  {spec["type"]} a{slot};\n' for slot, spec in enumerate(specs))
            source += ('struct MeshCpuArguments {\n' + fields + '};\n'
                       'extern "C" void mesh_cpu_entry(const void *arguments) {\n'
                       '  const auto &a = *static_cast<const MeshCpuArguments *>(arguments);\n'
                       '  mesh_cpu_kernel(' + ', '.join(f'a.a{slot}' for slot in range(len(specs))) + ');\n}\n')
            algebra, accesses = Algebra(), []
            for node in nodes:
                if not isinstance(node, SchedulerNode):
                    continue
                for domain in computed_domains(node.node, node._body, algebra):
                    work = algebra.map(domain['work'], (), lambda a, b: []).domain()
                    for side in ('reads', 'writes'):
                        for item in domain[side].values():
                            name = V.graph.scheduler.mutation_real_name.get(item['name'], item['name'])
                            if name in pointers:
                                accesses.append(dict(side=side, slot=calls.index(name), access=item,
                                    work=work, origins=tuple(node.node.get_origins())))
            index = len(command_accesses)
            command_accesses.append(tuple(accesses))
            reads = {item['slot'] for item in accesses if item['side'] == 'reads'}
            writes = {item['slot'] for item in accesses if item['side'] == 'writes'}
            access = sum(1 << slot for slot in reads) | sum(1 << (slot + _stream.compilation['ACCESS_WRITE_SHIFT']) for slot in writes)
            wrapper = V.graph.wrapper_code
            name = f'mesh_cpu_{wrapper.next_kernel_suffix()}'
            selected = implementation(nodes) if callable(implementation) else implementation
            wrapper.header.writeline('from torch_mesh import cpu as mesh_cpu')
            wrapper.header.writeline(f'{name} = mesh_cpu.declaration({source!r}, "mesh_cpu_entry", '
                                     f'{specs!r}, {access}, mesh_accesses[{index}], {selected!r}, {compilation(selected)!r})')
            return name

    return Source
