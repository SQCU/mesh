import json
from collections import OrderedDict
from pathlib import Path
import shutil
import sys

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb, Function, Program
from coremltools.converters.mil.mil import types


# design/algorithm-sources.md#coreml-partial-execution
def compile_part(request, destination):
    specification = json.loads(Path(request).read_text())
    arguments = OrderedDict()
    for i, (rows, columns, depth, half) in enumerate(specification['rectangles']):
        dtype = types.fp16 if half else types.fp32
        arguments[f'x{i}'] = mb.placeholder(shape=(rows, depth), dtype=dtype)
        arguments[f'w{i}'] = mb.placeholder(shape=(depth, columns), dtype=dtype)
    program = Program()
    with Function(arguments, opset_version=ct.target.macOS15) as function:
        parts = []
        for i, (_, _, depth, half) in enumerate(specification['rectangles']):
            x, w = function.inputs[f'x{i}'], function.inputs[f'w{i}']
            pieces = []
            for first in range(0, depth, 32 if half else depth):
                last = min(first + (32 if half else depth), depth)
                left = mb.slice_by_index(x=x, begin=(0, first), end=(x.shape[0], last))
                right = mb.slice_by_index(x=w, begin=(first, 0), end=(last, w.shape[1]))
                product = mb.matmul(x=left, y=right)
                pieces.append(mb.cast(x=product, dtype='fp32'))
            while len(pieces) > 1:
                pieces = [mb.add(x=pieces[j], y=pieces[j+1]) if j+1 < len(pieces) else pieces[j]
                          for j in range(0, len(pieces), 2)]
            scaled = mb.mul(x=pieces[0], y=np.float32(specification['alpha']))
            parts.append(mb.reshape(x=scaled, shape=(-1,)))
        joined = parts[0] if len(parts) == 1 else mb.concat(values=parts, axis=0)
        output = mb.cast(x=joined, dtype='fp16' if specification['output_half'] else 'fp32', name='z')
        function.set_outputs([output])
    program.add_function('main', function)
    model = ct.convert(program, minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT32)
    destination = Path(destination).with_suffix('.mlmodelc')
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(model.get_compiled_model_path(), destination)


if __name__ == '__main__':
    compile_part(*sys.argv[1:])
