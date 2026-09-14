import numpy as np

from mesh import Program


class Batch(list):
    # ../../design/algorithm-sources.md#programwrite
    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, column = index
            return super().__getitem__(row)[column]
        return super().__getitem__(index)

    @property
    # ../../design/algorithm-sources.md#programwrite
    def nbytes(self):
        return sum(value.nbytes for value in self)


class Frames:
    # ../../design/algorithm-sources.md#programwrite
    def __init__(self, peer=None, slots=64, usable=16384):
        self.program = Program()
        self.node = self.program.node
        self.peer = 1 - self.node if peer is None else peer
        self.slots, self.usable, self.stride = slots, usable, usable
        self.pending = {}
        self.outputs, self.inputs = [], []
        for sender, receiver in ((min(self.node, self.peer), max(self.node, self.peer)),
                                 (max(self.node, self.peer), min(self.node, self.peer))):
            for _ in range(slots):
                source, destination = (self.program.tensor((1, usable // 4), dtype=np.uint32) for _ in range(2))
                self.program.copy(source.on(sender), destination.on(receiver), queue=0)
                if sender == self.node:
                    self.outputs.append(source[0, 0])
                else:
                    self.inputs.append(self.program.export(destination[0, 0]))
        self.program.realize()

    # ../../design/algorithm-sources.md#programwrite
    def reserve(self, node, count):
        if node != self.peer or count < 0 or count > self.slots:
            raise ValueError('Frame reservation must fit the configured peer ring')
        refs = tuple(ref for ref in self.outputs if ref.writable)[:count]
        if len(refs) != count:
            return None
        result = Batch()
        for ref in refs:
            writer = self.program.write(ref)
            array = writer.__enter__().view(np.uint8).reshape(-1)
            self.pending[array.ctypes.data] = writer
            result.append(array)
        return result

    # ../../design/algorithm-sources.md#programwrite
    def send(self, frames, node):
        if node != self.peer:
            raise ValueError('Frames target a different configured peer')
        for frame in frames:
            writer = self.pending.pop(frame.ctypes.data)
            writer.__exit__(None, None, None)
        return len(frames)

    # ../../design/algorithm-sources.md#programwrite
    def read(self, dtype=np.uint8, max_batches=1):
        for result in self.inputs[:max_batches * self.slots]:
            if not result.ready:
                continue
            try:
                yield result.array.view(dtype).reshape(-1), self.peer
            finally:
                result.consume()

    # ../../design/algorithm-sources.md#programwrite
    def close(self):
        self.program.close()
        return 0
