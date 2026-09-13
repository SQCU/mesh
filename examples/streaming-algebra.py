import argparse
import json
import signal

import numpy as np
from mesh import Program
from mesh.nn import ffn


# design/algorithm-sources.md#streaming-ffn
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int)
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    args = parser.parse_args()
    running = True

    # design/algorithm-sources.md#streaming-ffn
    def stop(*unused):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    with Program(backend=args.backend) as program:
        rng = np.random.default_rng(701)
        rows, k, hidden, output, tile = 256, 128, 128, 128, 128
        data = tuple(rng.standard_normal((rows, k), dtype=np.float32) / 8 for _ in range(2))
        up = tuple(tuple(rng.standard_normal((k, hidden), dtype=np.float32) / 16 for _ in range(2)) for _ in range(2))
        down = tuple(rng.standard_normal((hidden, output), dtype=np.float32) / 16 for _ in range(2))
        inputs = tuple(program.tensor(value.shape) for value in data)

        # design/algorithm-sources.md#streaming-ffn
        def weight(value):
            tensor = program.tensor(value.shape)
            program.constant(tensor[0, 0], value)
            return tensor

        up_tensors = tuple(tuple(weight(value) for value in group) for group in up)
        down_tensors = tuple(weight(value) for value in down)
        exchanges = []

        # design/algorithm-sources.md#streaming-ffn
        def exchange(value):
            received = program.tensor(value.shape, value.block_shape)
            program.copy(value.on(0), received.on(1), queue=len(exchanges))
            exchanges.append(received)
            return received

        y = ffn(program, inputs, up_tensors, down_tensors, tile_rows=tile, exchange=exchange)
        returned = program.tensor(y.shape, y.block_shape)
        program.copy(y.on(1), returned.on(0), queue=0)
        result = program.export(returned[0, 0]) if args.rank == 0 else None
        program.realize()
        if args.rank == 1:
            while running:
                program.scan()
            return
        for tensor, value in zip(inputs, data):
            with program.write(tensor[0, 0]) as target:
                target[...] = value
        while not result.ready:
            program.scan()
        expected = np.zeros((rows, output), np.float64)
        for group, projection in zip(up, down):
            value = sum(x.astype(np.float64) @ w for x, w in zip(data, group))
            expected += (value / (1 + np.exp(-value))) @ projection
        error = float(np.max(np.abs(result.array - expected)))
        if not np.allclose(result.array, expected, atol=2e-4, rtol=2e-4):
            raise ArithmeticError('Streamed FFN differs from the numerical reference')
        print(json.dumps(dict(backend=args.backend, max_absolute_error=error,
            input_partitions=2, hidden_sections=2, row_sections=2)), flush=True)
        result.consume()


if __name__ == '__main__':
    main()
