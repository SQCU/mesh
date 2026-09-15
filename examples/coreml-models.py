import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb


# design/algorithm-sources.md#programkernel_call
def export(program, path):
    model = ct.convert(program, minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT32, skip_model_load=True)
    package = path.with_suffix('.mlpackage')
    model.save(str(package))
    return ct.models.utils.compile_model(str(package), str(path.with_suffix('.mlmodelc')))


# design/algorithm-sources.md#programkernel_call
def projection(rows, weights, path):
    # design/algorithm-sources.md#programkernel_call
    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, weights.shape[0]))])
    def program(x):
        return mb.matmul(x=x, y=weights, name='y')
    return export(program, path)


# design/algorithm-sources.md#programkernel_call
def activation(rows, width, path):
    # design/algorithm-sources.md#programkernel_call
    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, width))])
    def program(value):
        return mb.gelu(x=value, mode='TANH_APPROXIMATION', name='y')
    return export(program, path)


# design/algorithm-sources.md#programkernel_call
def residual(rows, width, path):
    # design/algorithm-sources.md#programkernel_call
    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, width)), mb.TensorSpec(shape=(rows, width))])
    def program(value, skip):
        return mb.add(x=value, y=skip, name='y')
    return export(program, path)


# design/algorithm-sources.md#programkernel_call
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--owners', type=int, nargs='+', required=True)
    parser.add_argument('--widths', type=int, nargs='+', required=True)
    parser.add_argument('--hidden', type=int, nargs='+', required=True)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--blocks', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=73)
    args = parser.parse_args()
    assert len(args.owners) == len(args.widths) == len(args.hidden)
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    layouts = [[dict(owner=owner, shape=[args.rows, width]) for owner, width in zip(args.owners, widths)]
               for widths in (args.widths, args.hidden)]
    activations = {width: activation(args.rows, width, root / f'activation-{width}') for width in set(args.hidden)}
    residuals = {width: residual(args.rows, width, root / f'residual-{width}') for width in set(args.widths)}
    stages = []
    for block in range(args.blocks):
        for direction, (inputs, outputs) in enumerate(((args.widths, args.hidden), (args.hidden, args.widths))):
            functions = [[projection(args.rows,
                                     rng.standard_normal((k, n)).astype(np.float32) / np.float32(np.sqrt(sum(inputs))),
                                     root / f'block-{block}-{direction}-{i}-{j}')
                          for j, n in enumerate(outputs)] for i, k in enumerate(inputs)]
            finish = []
            for j, (owner, width) in enumerate(zip(args.owners, outputs)):
                selected = [dict(name='value', group=2 * len(stages) + 1, part=j)]
                if direction:
                    selected.append(dict(name='skip', group=4 * block, part=j))
                finish.append(dict(model=(residuals if direction else activations)[width], owner=owner,
                                   worker=j % args.workers, inputs=selected,
                                   outputs=[dict(name='y', operand=layouts[1 - direction][j])]))
            stages.append(dict(outputs=layouts[1 - direction], functions=functions, finish=finish))
    plan = dict(count=args.count, workers=args.workers, inputName='x', outputName='y',
                inputs=layouts[0], stages=stages)
    (root / 'chain.json').write_text(json.dumps(plan, indent=2) + '\n')


if __name__ == '__main__':
    main()
