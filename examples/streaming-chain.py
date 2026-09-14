import argparse
import signal
import sys

import numpy as np

from mesh import BlockSpec, Program, ShapeDtypeStruct, kernels
from mesh.nn import linear


# design/algorithm-sources.md#pallas-panel-composition
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('up_weight')
    parser.add_argument('down_weight')
    parser.add_argument('--producer', type=int, required=True)
    parser.add_argument('--consumer', type=int, required=True)
    parser.add_argument('--region')
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    parser.add_argument('--tile-rows', type=int, default=16)
    parser.add_argument('--tile-k', type=int, default=16)
    parser.add_argument('--tile-columns', type=int, default=16)
    args = parser.parse_args()
    values = np.load(args.input)
    weights = (np.load(args.up_weight), np.load(args.down_weight))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with Program(backend=args.backend, region=args.region) as program:
        x = program.tensor(values.shape, (args.tile_rows, args.tile_k), dtype=values.dtype)
        up_weight, down_weight = tuple(program.tensor(w.shape, dtype=w.dtype) for w in weights)
        for tensor, value in zip((up_weight, down_weight), weights):
            program.constant(tensor[0, 0], value)
        up = linear(program, x, up_weight, tile_rows=args.tile_rows,
                    tile_k=args.tile_k, tile_columns=args.tile_columns, peer=args.producer)
        value, = kernels.arguments(1)
        spec = BlockSpec(up.block_shape, lambda i, j: (i, j))
        hidden = program.kernel_call(kernels.expression(value / (1 + (0 - value).exp())),
            grid=up.grid, in_specs=(spec,), out_specs=spec,
            out_shape=ShapeDtypeStruct(up.shape, up.dtype), peer=args.producer)(up)
        received = program.tensor(hidden.shape, hidden.block_shape, dtype=hidden.dtype)
        program.copy(hidden.on(args.producer), received.on(args.consumer))
        down = linear(program, received, down_weight, tile_rows=args.tile_rows,
                      tile_k=args.tile_k, tile_columns=args.tile_columns, peer=args.consumer)
        returned = program.tensor(down.shape, down.block_shape, dtype=down.dtype)
        program.copy(down.on(args.consumer), returned.on(args.producer))
        outputs = {index: program.export(ref) for index, ref in returned.blocks.items()} if program.node == args.producer else {}
        program.realize()
        if program.node != args.producer:
            signal.pause()
            return
        for (i, j), ref in x.blocks.items():
            row, column = i * x.block_shape[0], j * x.block_shape[1]
            with program.write(ref) as destination:
                destination[...] = values[row:row + ref.shape[0], column:column + ref.shape[1]]
        while outputs:
            for index, output in tuple(outputs.items()):
                if output.ready:
                    print(index, output.array.tolist(), flush=True)
                    output.consume()
                    del outputs[index]
            if program.report.code:
                raise RuntimeError(f'Mesh execution error: {program.report.code}')


if __name__ == '__main__':
    main()
