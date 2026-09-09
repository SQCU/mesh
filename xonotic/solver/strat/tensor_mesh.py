import ctypes as c
import errno
import json
import queue
import struct
import threading
import time
import uuid
import zlib

import numpy as np

from mesh import _lib as mesh_library
from solver.xonwire import MAGIC, WIRE, FrameStream, Reassembler, REQUEST_TIMEOUT_S
from .tensor import Graph, Tensor, Dimension
from .tensor_runtime import Executable, library

CONTROL, STATUS = WIRE['TENSOR_CONTROL'], WIRE['TENSOR_STATUS']


class Span(c.Structure):
    _fields_ = [('stream', c.c_uint64), ('offset', c.c_uint64), ('bytes', c.c_uint64), ('target', c.c_uint64)]


class Binding(c.Structure):
    _fields_ = [('stream', c.c_uint64), ('offset', c.c_uint64), ('tensor', c.c_uint32), ('padding', c.c_uint32)]


def encode(value):
    data = zlib.compress(json.dumps(value, separators=(',', ':')).encode())
    return np.frombuffer(data, np.uint8).astype(np.float32).reshape(-1, 1)


def decode(value):
    return json.loads(zlib.decompress(np.asarray(value, dtype=np.uint8).tobytes()))


def symbolic(value):
    if isinstance(value, Dimension): return {'dimension': value.op, 'args': symbolic(value.args)}
    if isinstance(value, dict): return {key: symbolic(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)): return [symbolic(part) for part in value]
    return value


def restore(value):
    if isinstance(value, dict):
        if 'dimension' in value: return Dimension(value['dimension'], restore(value['args']))
        return {key: restore(part) for key, part in value.items()}
    if isinstance(value, list): return tuple(restore(part) for part in value)
    return value


def manifest(executable, owners):
    graph = executable.graph
    return {'nodes': [[value.shape, value.dtype, op, [part.index for part in inputs], attributes, owner]
                     for value, op, inputs, attributes, owner in graph.nodes],
            'inputs': {name: value.index for name, value in graph.inputs.items()},
            'constants': [[value.index, data.tolist()] for value, data in graph.constants.values()],
            'outputs': {name: executable.remote.exports[name][1] for segments in executable.segments.values()
                        for owner, name, indices in segments if owner in owners},
            'schedule': {name: indices for segments in executable.segments.values()
                         for owner, name, indices in segments if owner in owners}}


def compile_program(description, capacity):
    graph = Graph()
    for shape, dtype, op, inputs, attrs, owner in restore(description)['nodes']:
        value = Tensor(graph, len(graph.nodes), shape, dtype)
        graph.nodes.append((value, op, tuple(graph.nodes[index][0] for index in inputs), attrs, owner))
    graph.inputs = {name: graph.nodes[index][0] for name, index in description['inputs'].items()}
    graph.constants = {index: (graph.nodes[index][0], np.asarray(data, dtype=graph.nodes[index][0].dtype))
                       for index, data in description['constants']}
    schedule = description['schedule']
    executable = Executable(graph, {name: tuple(graph.nodes[index][0] for index in description['outputs'][name]) for name in schedule}, schedule)
    executable.reserve(capacity)
    return executable


class NativeTransport:
    def __init__(self, mesh):
        self.mesh, self.lib = mesh, library()
        declarations = {
            'transport': (c.c_void_p, [c.c_void_p]),
            'transport_bytes': (c.c_size_t, [c.c_void_p]),
            'transport_progress': (c.c_int, [c.c_void_p]),
            'transport_free': (c.c_int, [c.c_void_p]),
            'channel': (c.c_void_p, [c.c_void_p, c.c_void_p, c.c_int, c.c_int, c.c_uint32, c.c_uint64, c.c_char_p, c.c_size_t]),
            'transfer': (c.c_int, [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64]),
            'transfer_plan': (c.c_int, [c.c_void_p, c.POINTER(Span), c.c_size_t, c.POINTER(Binding), c.c_size_t]),
            'transfer_release': (c.c_int, [c.c_void_p]),
            'transfer_commit': (c.c_int, [c.c_void_p]),
            'transfer_status': (c.c_int, [c.c_void_p]),
            'transfer_cancel': (None, [c.c_void_p, c.c_int]),
            'channel_free': (c.c_int, [c.c_void_p]),
        }
        for name, (result, arguments) in declarations.items():
            function = getattr(self.lib, 'mesh_tensor_' + name)
            function.restype, function.argtypes = result, arguments
        self.handle = self.lib.mesh_tensor_transport(mesh_library.mesh_context())
        if not self.handle: raise OSError(c.get_errno(), 'native tensor transport attachment failed')
        self.bytes = self.lib.mesh_tensor_transport_bytes(self.handle)
        self.channels = []
        mesh.receive_with(self, struct.pack('<I', MAGIC))

    def progress(self):
        activity = self.lib.mesh_tensor_transport_progress(self.handle)
        for channel in tuple(self.channels):
            if channel.retired and not self.lib.mesh_tensor_channel_free(channel.handle):
                self.channels.remove(channel)
                channel.handle = None
        return activity

    def channel(self, executable, peer, receiving, identity, extent, server=False):
        channel = Channel(self, executable, peer, receiving, identity, extent, server)
        self.channels.append(channel)
        return channel

    def close(self):
        self.progress()
        for channel in tuple(self.channels):
            channel.cancel()
            if not self.lib.mesh_tensor_channel_free(channel.handle):
                self.channels.remove(channel)
                channel.handle = None
        return errno.EBUSY if self.channels else self.lib.mesh_tensor_transport_free(self.handle)


class Channel:
    def __init__(self, transport, executable, peer, receiving, identity, extent, server):
        self.transport, self.executable = transport, executable
        token = uuid.UUID(identity).bytes
        self.handle = transport.lib.mesh_tensor_channel(transport.handle, executable.handle, peer,
            receiving, 2 if bool(receiving) != server else 1, int.from_bytes(token[:8], 'little'), token + token, extent)
        if not self.handle: raise OSError(c.get_errno(), 'native tensor page binding failed')
        self.sequence = 0
        self.transferred = 0
        self.extent = extent
        self.retired = False
        self.plans = {}
        self.total = 0

    def plan(self, indices, direct=False, staging=False):
        if self.handle is None: raise OSError(errno.ESTALE, 'native page binding was retired during transport recovery')
        key = tuple(indices), direct, staging
        if key not in self.plans:
            spans, bindings, total = [], [], 0
            for index in indices:
                size = self.executable.arrays[index].nbytes
                if size:
                    target = self.executable.views[index].offset
                    spans.append(Span(total, self.executable.staging_offset + total if staging else target, size, target))
                    if direct:
                        bindings.extend(Binding(total, self.executable.views[alias].offset, alias, 0)
                            for alias in self.executable.alias_groups[index])
                    total += size
            if total > self.extent: bindings = []
            self.plans[key] = (Span * len(spans))(*spans), (Binding * len(bindings))(*bindings), total
        spans, bindings, self.total = self.plans[key]
        self.executable.check(self.transport.lib.mesh_tensor_transfer_plan(self.handle, spans, len(spans), bindings, len(bindings)))

    def begin(self, offset, count):
        if self.handle is None: raise OSError(errno.ESTALE, 'native page binding was retired during transport recovery')
        self.sequence += 1
        self.executable.check(self.transport.lib.mesh_tensor_transfer(self.handle, self.sequence, offset, count))
        self.transferred += count

    def status(self):
        return -errno.ESTALE if self.handle is None else self.transport.lib.mesh_tensor_transfer_status(self.handle)

    def release(self):
        if self.handle:
            self.executable.check(self.transport.lib.mesh_tensor_transfer_release(self.handle))

    def commit(self):
        if self.total:
            self.executable.check(self.transport.lib.mesh_tensor_transfer_commit(self.handle))
            self.executable.finish()

    def cancel(self):
        if self.handle:
            self.transport.lib.mesh_tensor_transfer_cancel(self.handle, errno.ECANCELED)
        self.retired = True


def chunks(channel):
    return ((offset, min(channel.extent, channel.total - offset)) for offset in range(0, channel.total, channel.extent))


class RemoteProgram:
    def __init__(self, executable):
        self.executable = executable
        self.identity = str(uuid.uuid4())
        self.generation = 0
        self.sequence = 0
        self.peers = {}
        self.completed = self.fallbacks = self.uploads = 0
        self.direct_regions = 0
        for region in executable.graph.regions.values():
            endpoint = region['executor']
            peer = self.peers.setdefault(region['peer'], {'endpoint': endpoint, 'owners': set(), 'instance': None})
            peer['owners'].add(region['owner'])
        self.exports = {}
        consumers = {}
        aliases = {}
        for value, operation, inputs, _, _ in executable.graph.nodes:
            aliases[value.index] = aliases[inputs[0].index] if operation in ('reshape', 'stop_gradient') else value.index
        for value, _, inputs, _, _ in executable.graph.nodes:
            for part in inputs: consumers.setdefault(aliases[part.index], set()).add(aliases[value.index])
        from .tensor_runtime import leaves
        for phase, segments in executable.segments.items():
            phase_nodes = set(executable.schedule[phase])
            output_ids = {aliases[value.index] for value in leaves(executable.phases[phase])}
            for _, key, indices in segments:
                members = set(indices)
                incoming = sorted({aliases[value.index] for index in indices for value in executable.graph.nodes[index][2]
                                   if aliases[value.index] not in members and executable.graph.nodes[aliases[value.index]][1] != 'constant'})
                outgoing = sorted(index for index in indices if index in output_ids or consumers.get(index, set()).intersection(phase_nodes) - members)
                self.exports[key] = incoming, outgoing
        self.parameters = {value.index for value, _ in executable.graph.parameters.values()}

    def rpc(self, peer, message, timeout=REQUEST_TIMEOUT_S):
        endpoint = peer['endpoint']
        self.sequence += 1
        response, metrics = peer['stream'].exchange(CONTROL, self.sequence, 0, encode(dict(message, program=self.identity)),
            endpoint.node, {STATUS: peer['receiver']}, cancel=lambda: endpoint.stopping['signal'] is not None,
            backlog=endpoint.backlog, retry_s=.5, timeout_s=timeout)
        if response is None: raise TimeoutError(f'tensor control request {message["operation"]}: {metrics}')
        result = decode(response[STATUS][1])
        if result.get('error'): raise RuntimeError(result['error'])
        return result

    def reserve(self, node=None):
        selected = self.peers.items() if node is None else ((node, self.peers[node]),)
        for node, peer in selected:
            endpoint = peer['endpoint']
            try:
                native = endpoint.mesh.native or NativeTransport(endpoint.mesh)
                peer.setdefault('stream', FrameStream(endpoint.mesh))
                peer.setdefault('receiver', Reassembler(STATUS, 1, endpoint.mesh.usable))
                binding = str(uuid.uuid4())
                result = self.rpc(peer, {'operation': 'install', 'description': symbolic(manifest(self.executable, peer['owners'])),
                                        'capacity': self.executable.capacity, 'binding': binding, 'extent': native.bytes})
                for name in ('send', 'receive'):
                    if name in peer: peer.pop(name).cancel()
                peer['send'] = native.channel(self.executable, node, False, binding, result['extent'])
                peer['receive'] = native.channel(self.executable, node, True, binding, result['extent'])
                peer['instance'] = result['instance']
                peer['ready'] = True
                peer.pop('error', None)
            except Exception as error:
                peer['ready'] = False
                self.failure(peer, error)

    def failure(self, peer, error):
        detail = f'{type(error).__name__}: {error}'
        changed = peer.get('error') != detail
        peer['error'] = detail
        peer['retry_at'] = time.monotonic() + 1
        peer['endpoint'].last = {'completed': False, 'local_fallback': True, 'backend': 'persistent_metal', 'error': peer['error']}
        if changed: print(json.dumps({'event': 'tensor_remote_recovery', 'peer': peer['endpoint'].node, 'error': peer['error']}), flush=True)

    def transfer(self, peer, channel, indices, staging=False):
        endpoint = peer['endpoint']
        channel.plan(indices, staging=staging)
        for offset, count in chunks(channel):
            channel.begin(offset, count)
            started = time.monotonic()
            while channel.status() == 0:
                endpoint.mesh.pump()
                for frame, source in endpoint.mesh.read(np.uint8, max_batches=1): endpoint.backlog.append((frame.copy(), source))
                if endpoint.stopping['signal'] is not None or time.monotonic() - started > REQUEST_TIMEOUT_S:
                    channel.cancel()
                    raise TimeoutError(f'native tensor page transfer: offset={offset}, bytes={count}')
                time.sleep(.0001)
            if channel.status() < 0: raise OSError(-channel.status(), 'native tensor page transfer failed')

    def run(self, phase):
        for owner, key, indices in self.executable.segments[phase]:
            peer = next((value for value in self.peers.values() if owner in value['owners']), None)
            if peer is not None and (not peer.get('ready') or peer['send'].handle is None or peer['send'].transport is not peer['endpoint'].mesh.native):
                peer['ready'] = False
                if time.monotonic() >= peer.get('retry_at', 0): self.reserve(peer['endpoint'].node)
                if not peer.get('ready'):
                    self.fallbacks += 1
                    peer = None
            if peer is not None:
                try:
                    incoming, outgoing = self.exports[key]
                    reply = self.rpc(peer, {'operation': 'begin', 'phase': key, 'generation': self.generation,
                        'inputs': incoming, 'outputs': outgoing, 'parameters': sorted(self.parameters.intersection(incoming)),
                        'commit_generation': self.generation + 1 if phase == 'update' else self.generation})
                    self.transfer(peer, peer['send'], reply['inputs'])
                    self.transfer(peer, peer['receive'], outgoing, staging=True)
                    result = self.rpc(peer, {'operation': 'complete', 'invocation': reply['invocation']})
                    if not result.get('completed'): raise RuntimeError('remote tensor invocation did not commit')
                except Exception as error:
                    self.fallbacks += 1
                    self.failure(peer, error)
                    for name in ('send', 'receive'):
                        if name in peer: peer[name].cancel()
                    peer['ready'] = False
                else:
                    peer['receive'].commit()
                    self.completed += 1
                    self.direct_regions += int(result['direct_pages'])
                    self.uploads += len(self.parameters.intersection(reply['inputs']))
                    peer['endpoint'].last = {'completed': True, 'local_fallback': False, 'backend': 'persistent_mesh_metal',
                        'operation': key, 'roundtrip_s': result['elapsed_s'], 'worker_compute_s': result['compute_s'],
                        'parameter_generation': self.generation, 'parameter_uploads': self.uploads,
                        'direct_pages': result['direct_pages'],
                        'native_bytes_sent': peer['send'].transferred, 'native_bytes_received': peer['receive'].transferred}
                    continue
            self.executable.finish()
            self.executable.check(self.executable.lib.mesh_tensor_submit(self.executable.handle, self.executable.phase_ids[key]))
            self.executable.finish()
        if phase == 'update': self.generation += 1

    def report(self):
        return {'completed_regions': self.completed, 'local_recoveries': self.fallbacks,
                'direct_page_regions': self.direct_regions,
                'parameter_uploads': self.uploads, 'generation': self.generation,
                'peers': {str(node): {'ready': peer.get('ready', False), 'error': peer.get('error'),
                         'instance': peer['instance']} for node, peer in self.peers.items()}}


class Worker:
    def __init__(self, transport, meter):
        self.transport = transport
        self.meter = meter
        self.instance = str(uuid.uuid4())
        self.programs, self.responses, self.receivers, self.pending = {}, {}, {}, set()
        self.activity = {}
        self.installing = {}
        self.compiling = set()
        self.jobs, self.results = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=self.compile, name='tensor-compiler', daemon=True)
        self.thread.start()

    def compile(self):
        while (job := self.jobs.get()) is not None:
            key, identity, header, message, executable = job
            try:
                executable = executable or compile_program(message['description'], message['capacity'])
                executable.reserve(message['capacity'])
                self.results.put((key, identity, header, message, executable, None))
            except Exception as error:
                self.results.put((key, identity, header, message, None, f'{type(error).__name__}: {error}'))

    def reply(self, source, identity, header, result):
        parts = ((STATUS, encode(result)),)
        previous = self.responses.get(identity[:2])
        if previous is None or previous[0][2] <= identity[2]:
            self.responses[identity[:2]] = identity, parts
        self.transport.rows(source, header, parts)

    def receive(self, source, header, frame):
        identity = source, header['session'], header['req_id']
        self.activity[identity[:2]] = time.monotonic()
        previous = self.responses.get(identity[:2])
        if previous is not None and identity[2] <= previous[0][2]:
            if identity == previous[0] and header['offset'] == 0: self.transport.rows(source, header, previous[1])
            return
        if identity in self.pending: return
        channel = source, header['session'], len(frame)
        receiver = self.receivers.setdefault(channel, Reassembler(CONTROL, 1, len(frame)))
        record = receiver.feed(frame)
        if record is None: return
        try:
            message = decode(receiver.stage[:record['rows']])
            key = source, message['program']
            program = self.programs.get(key)
            if program is not None: program['last_used'] = time.monotonic()
            operation = message['operation']
            if operation == 'install':
                if program is not None:
                    for name in ('send', 'receive'):
                        if name in program: program[name].cancel()
                    program['active'] = False
                    program['parameters'].clear()
                    if 'completion' in program:
                        previous, previous_header = program.pop('completion')
                        self.pending.discard(previous)
                        self.reply(source, previous, previous_header, {'error': 'tensor invocation replaced during recovery'})
                superseded = self.installing.get(key)
                if superseded is not None:
                    _, previous, previous_header, _ = superseded
                    self.pending.discard(previous)
                    self.reply(source, previous, previous_header, {'error': 'tensor installation replaced by newer binding'})
                self.pending.add(identity)
                self.installing[key] = key, identity, header, message
            elif program is None:
                self.reply(source, identity, header, {'error': 'remote program was replaced; install current program again'})
            elif operation == 'begin':
                if key in self.installing or key in self.compiling: raise RuntimeError('tensor installation is still progressing')
                if program.get('active'): raise RuntimeError('native tensor invocation already active')
                generation = message['generation']
                parameters = set(message['parameters'])
                inputs = [index for index in message['inputs'] if index not in parameters or program['parameters'].get(index) != generation]
                program['receive'].plan(inputs, direct=True)
                program.update(active=True, identity=identity, generation=generation, inputs=inputs,
                    commit_generation=message['commit_generation'],
                    outputs=message['outputs'], parameter_inputs=parameters, phase=message['phase'],
                    state='input', chunks=chunks(program['receive']),
                    waiting=False, started=time.monotonic(), compute_s=0, direct_pages=False)
                self.reply(source, identity, header, {'inputs': inputs, 'invocation': list(identity)})
            elif operation == 'complete':
                if tuple(message['invocation']) != program.get('identity'): raise RuntimeError('tensor invocation identity differs')
                if program.get('active'):
                    self.pending.add(identity)
                    program['completion'] = identity, header
                else:
                    self.reply(source, identity, header, program['result'])
            else:
                raise ValueError(f'unknown tensor control operation: {operation}')
        except Exception as error:
            self.reply(source, identity, header, {'error': f'{type(error).__name__}: {error}'})

    def progress(self):
        mesh = self.transport.mesh
        if mesh is None: return
        if mesh.native is None: NativeTransport(mesh)
        for key, job in tuple(self.installing.items()):
            program = self.programs.get(key)
            if key not in self.compiling and (program is None or all(program[name].handle is None for name in ('send', 'receive') if name in program)):
                self.compiling.add(key)
                self.jobs.put((*job, None if program is None else program['executable']))
                del self.installing[key]
        while not self.results.empty():
            key, identity, header, message, executable, error = self.results.get_nowait()
            self.compiling.discard(key)
            self.pending.discard(identity)
            if error:
                self.reply(key[0], identity, header, {'error': error})
            else:
                program = self.programs.get(key)
                if program is None:
                    program = {'executable': executable, 'parameters': {}, 'active': False, 'last_used': time.monotonic()}
                    self.programs[key] = program
                for name in ('send', 'receive'):
                    if name in program: program.pop(name).cancel()
                if key in self.installing:
                    self.reply(key[0], identity, header, {'error': 'tensor installation replaced by newer binding'})
                    continue
                try:
                    extent = min(mesh.native.bytes, message['extent'])
                    program['receive'] = mesh.native.channel(executable, key[0], True, message['binding'], extent, server=True)
                    program['send'] = mesh.native.channel(executable, key[0], False, message['binding'], extent, server=True)
                    self.reply(key[0], identity, header, {'instance': self.instance, 'capacity': executable.capacity, 'extent': extent})
                except Exception as error:
                    for name in ('send', 'receive'):
                        if name in program: program[name].cancel()
                    self.programs.pop(key, None)
                    self.reply(key[0], identity, header, {'error': f'{type(error).__name__}: {error}'})
        for key, program in tuple(self.programs.items()):
            if key not in self.installing and key not in self.compiling and not program.get('active') and time.monotonic() - program['last_used'] > 240:
                for name in ('send', 'receive'):
                    if name in program: program[name].cancel()
                del self.programs[key]
                continue
            if not program.get('active'): continue
            try:
                self.advance(program)
            except Exception as error:
                for name in ('send', 'receive'): program[name].cancel()
                program['result'] = {'error': f'{type(error).__name__}: {error}'}
                program['active'] = False
                program['parameters'].clear()
            if not program['active'] and 'completion' in program:
                identity, header = program.pop('completion')
                self.pending.discard(identity)
                self.reply(key[0], identity, header, program['result'])
        active = {identity[:2] for identity in self.pending}
        for channel, observed in tuple(self.activity.items()):
            if channel not in active and time.monotonic() - observed > 240:
                self.activity.pop(channel)
                self.responses.pop(channel, None)
                self.receivers = {key: value for key, value in self.receivers.items() if key[:2] != channel}

    def advance(self, program):
        executable = program['executable']
        if time.monotonic() - program['started'] > 120: raise TimeoutError('native tensor invocation exceeded its execution budget')
        state = program['state']
        if state == 'compute':
            status = executable.lib.mesh_tensor_status(executable.handle)
            if status < 0: executable.check(status)
            if status == 0: return
            program['compute_s'] = time.monotonic() - program['compute_started']
            program['receive'].release()
            program['send'].plan(program['outputs'])
            program.update(state='output', chunks=chunks(program['send']))
            return
        channel = program['receive' if state == 'input' else 'send']
        if program['waiting']:
            status = channel.status()
            if status < 0: raise OSError(-status, 'native tensor transfer failed')
            if status == 0: return
            program['direct_pages'] |= status == 2
            program['waiting'] = False
        part = next(program['chunks'], None)
        if part is not None:
            channel.begin(*part)
            program['waiting'] = True
        elif state == 'input':
            for index in program['parameter_inputs']: program['parameters'][index] = program['generation']
            executable.submit(program['phase'])
            program.update(state='compute', compute_started=time.monotonic())
        else:
            if program['receive'].sequence and program['receive'].status() == 0: return
            program['active'] = False
            for index in program['parameter_inputs']: program['parameters'][index] = program['commit_generation']
            program['result'] = {'completed': True, 'elapsed_s': time.monotonic() - program['started'],
                'compute_s': program['compute_s'], 'direct_pages': program['direct_pages']}
            work = executable.work(program['phase'])
            self.meter.record(program['compute_s'], work['flops'], None, work['bytes'], work['bytes'],
                operations={'host_role': 'matrix', 'operation': program['phase'], 'backend': 'persistent_mesh_metal',
                    'flop_model': 'matrix_products_only', 'byte_model': 'logical_tensor_operands',
                    'row_unit': 'matrix_output_rows', 'direct_pages': program['direct_pages']}, rows=work['rows'])
            print(json.dumps({'event': 'tensor_region_complete', 'phase': program['phase'], **program['result']}), flush=True)

    def close(self):
        self.jobs.put(None)
        for program in self.programs.values():
            for name in ('send', 'receive'):
                if name in program: program[name].cancel()

    def report(self):
        return {'programs': len(self.programs), 'active': sum(program.get('active', False) for program in self.programs.values()),
                'compiling': len(self.compiling), 'installing': len(self.installing), 'instance': self.instance}
