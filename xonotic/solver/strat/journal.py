import json
import os
import hashlib
import io
import struct

import numpy as np


class RecordedSteps:
    """Reader for previously recorded inputs, used only during input preparation."""
    def __init__(self, path):
        self.path = path
        self.position = 0

    def read(self, position=0):
        from .checkpoint_state import unpack_state
        if not os.path.isfile(self.path):
            return
        with open(self.path, "rb") as source:
            source.seek(position)
            while True:
                start = source.tell()
                header = source.read(40)
                if not header:
                    break
                if len(header) != 40:
                    print(json.dumps({"event": "training_journal_incomplete_tail", "path": self.path, "offset": start}), flush=True)
                    break
                size = struct.unpack("<Q", header[:8])[0]
                remaining = os.fstat(source.fileno()).st_size - source.tell()
                data = source.read(min(size, remaining))
                if len(data) != size or hashlib.sha256(data).digest() != header[8:]:
                    print(json.dumps({"event": "training_journal_incomplete_tail", "path": self.path, "offset": start}), flush=True)
                    break
                with np.load(io.BytesIO(data), allow_pickle=False) as payload:
                    state = unpack_state(payload)
                self.position = source.tell()
                yield self.position, state
