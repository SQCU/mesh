import json
import uuid
import zlib

import numpy as np

from solver.xonwire import WIRE, FrameStream, Reassembler, REQUEST_TIMEOUT_S
from .tensor import Graph, Tensor, Dimension
from .tensor_runtime import Executable, leaves

CONTROL, STATUS = WIRE['TENSOR_CONTROL'], WIRE['TENSOR_STATUS']


# ../../../design/algorithm-sources.md#complete-page-ownership
def encode(value):
    data = zlib.compress(json.dumps(value, separators=(',', ':')).encode())
    return np.frombuffer(data, np.uint8).astype(np.float32).reshape(-1, 1)


# ../../../design/algorithm-sources.md#complete-page-ownership
def decode(value):
    return json.loads(zlib.decompress(np.asarray(value, dtype=np.uint8).tobytes()))


# ../../../design/algorithm-sources.md#complete-page-ownership
def symbolic(value):
    if isinstance(value, Dimension): return {'dimension': value.op, 'args': symbolic(value.args)}
    if isinstance(value, dict): return {key: symbolic(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)): return [symbolic(part) for part in value]
    return value


# ../../../design/algorithm-sources.md#complete-page-ownership
def restore(value):
    if isinstance(value, dict):
        if 'dimension' in value: return Dimension(value['dimension'], restore(value['args']))
        return {key: restore(part) for key, part in value.items()}
    if isinstance(value, list): return tuple(restore(part) for part in value)
    return value


# ../../../design/algorithm-sources.md#complete-page-ownership
def manifest(executable):
    graph = executable.graph
    return symbolic({'nodes': [[value.shape, value.dtype, operation, [part.index for part in inputs], attributes, owner]
        for value, operation, inputs, attributes, owner in graph.nodes],
        'inputs': {name: value.index for name, value in graph.inputs.items()},
        'constants': [[value.index, data.tolist()] for value, data in graph.constants.values()],
        'parameters': [[name, value.index, np.asarray(data).tolist()] for name, (value, data) in graph.parameters.items()],
        'exports': {name: [value.index for value in leaves(values)] for name, values in executable.exports.items()},
        'owners': executable.owner_nodes, 'capacity': executable.capacity})


# ../../../design/algorithm-sources.md#complete-page-ownership
def compile_program(description):
    description = restore(description)
    graph = Graph()
    for shape, dtype, operation, inputs, attributes, owner in description['nodes']:
        value = Tensor(graph, len(graph.nodes), shape, dtype)
        graph.nodes.append((value, operation, tuple(graph.nodes[index][0] for index in inputs), attributes, owner))
    graph.inputs = {name: graph.nodes[index][0] for name, index in description['inputs'].items()}
    graph.constants = {index: (graph.nodes[index][0], np.asarray(data, dtype=graph.nodes[index][0].dtype)) for index, data in description['constants']}
    graph.parameters = {name: (graph.nodes[index][0], np.asarray(data, dtype=graph.nodes[index][0].dtype)) for name, index, data in description['parameters']}
    exports = {name: tuple(graph.nodes[index][0] for index in indices) for name, indices in description['exports'].items()}
    executable = Executable(graph, exports, {int(owner): node for owner, node in description['owners'].items()})
    executable.reserve(description['capacity'], configure=False)
    return executable


class RemoteProgram:
    # ../../../design/algorithm-sources.md#complete-page-ownership
    def __init__(self, executable):
        self.executable, self.identity = executable, str(uuid.uuid4())
        self.peers = {region['peer']: region['executor'] for region in executable.graph.regions.values()}
        self.sequence = 0
        self.streams = {node: FrameStream(endpoint.mesh) for node, endpoint in self.peers.items()}
        self.receivers = {node: Reassembler(STATUS, 1, endpoint.mesh.usable) for node, endpoint in self.peers.items()}

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def request(self, node, operation, **values):
        endpoint = self.peers[node]
        self.sequence += 1
        response, details = self.streams[node].exchange(CONTROL, self.sequence, 0,
            encode(dict(values, operation=operation, program=self.identity)), node, {STATUS: self.receivers[node]},
            cancel=lambda: endpoint.stopping['signal'] is not None, backlog=endpoint.backlog,
            retry_s=float('inf'), timeout_s=REQUEST_TIMEOUT_S)
        if response is None: raise TimeoutError(f'tensor configuration: {details}')
        result = decode(response[STATUS][1])
        if 'error' in result: raise RuntimeError(result['error'])
        return result

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def configure(self):
        local = self.executable.context.contents.M.contents.node
        tables = {local: self.executable.table_ids()}
        description = manifest(self.executable)
        for node in self.peers:
            tables[node] = self.request(node, 'install', description=description)['tables']
        for node in self.peers: self.request(node, 'bind', tables=tables)
        self.executable.configure(tables)

    # ../../../design/algorithm-sources.md#asynchronous-metadata-publication
    def report(self):
        return {'program': self.identity, 'participants': tuple(self.peers)}


class Worker:
    # ../../../design/algorithm-sources.md#complete-page-ownership
    def __init__(self, transport, meter):
        self.transport, self.meter = transport, meter
        self.programs, self.configurations, self.receivers = {}, {}, {}

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def receive(self, source, header, frame):
        identity = source, header['session']
        receiver = self.receivers.setdefault(identity, Reassembler(CONTROL, 1, len(frame)))
        record = receiver.feed(frame)
        if record is None: return
        message = decode(receiver.stage[:record['rows']])
        key = source, message['program']
        try:
            if message['operation'] == 'install':
                executable = compile_program(message['description'])
                self.configurations[key] = executable
                result = {'tables': executable.table_ids()}
            elif message['operation'] == 'bind':
                executable = self.configurations.pop(key)
                executable.configure({int(node): tables for node, tables in message['tables'].items()})
                self.programs[key] = executable
                result = {'configured': True}
            else:
                raise ValueError(f'unknown configuration operation: {message["operation"]}')
        except Exception as error:
            result = {'error': f'{type(error).__name__}: {error}'}
        self.transport.rows(source, header, ((STATUS, encode(result)),))

    # ../../../design/algorithm-sources.md#literal-row-functions
    def progress(self):
        for executable in self.programs.values(): executable.scan()

    # ../../../design/algorithm-sources.md#complete-page-ownership
    def close(self):
        for executable in self.programs.values(): executable.close()
        self.programs.clear()
        self.configurations.clear()

    # ../../../design/algorithm-sources.md#asynchronous-metadata-publication
    def report(self):
        return {'programs': len(self.programs), 'tables': sum(len(plans) for executable in self.programs.values() for plans in executable.realizations.values())}
