import json
import os
import hashlib
import io
import struct

import numpy as np



class TrainingJournal:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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

    def append(self, state):
        from .checkpoint_state import pack_state, write_payload
        data = io.BytesIO()
        write_payload(data, pack_state(state))
        value = data.getvalue()
        with open(self.path, "ab") as target:
            target.write(struct.pack("<Q", len(value)) + hashlib.sha256(value).digest() + value)
            target.flush()
            os.fsync(target.fileno())
            self.position = target.tell()
        return self.position

    def repair_tail(self, valid_position):
        if os.path.isfile(self.path) and os.path.getsize(self.path) != valid_position:
            with open(self.path, "r+b") as target:
                target.seek(valid_position)
                with open(self.path + '.incomplete', 'wb') as evidence:
                    evidence.write(target.read())
                target.truncate(valid_position)
                target.flush()
                os.fsync(target.fileno())


class Journal:
    def __init__(self, store=None):
        self.positions = {}
        self.store = store
        self.errors = []

    def seek(self, path, position):
        with path.open('rb') as handle:
            head = handle.read(min(4096, position))
            handle.seek(max(0, position - 128))
            tail = handle.read(min(128, position))
        self.positions[str(path)] = position, head, tail
        if self.store:
            self.store.move(path, position, head, tail)

    def read(self, path, prefix=""):
        key = str(path)
        saved = self.positions.get(key) or (self.store.cursor(path) if self.store else None)
        with path.open('rb') as handle:
            position, head, tail = saved or (0, b'', b'')
            size = os.fstat(handle.fileno()).st_size
            actual_head = handle.read(len(head))
            handle.seek(max(0, position - len(tail)))
            actual_tail = handle.read(len(tail))
            if size < position or actual_head != head or actual_tail != tail:
                position = 0
            handle.seek(position)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line.endswith(b'\n'):
                    position = start
                    break
                position = handle.tell()
                if not prefix or prefix.encode() in line:
                    try:
                        yield json.loads(line.partition(prefix.encode())[2] if prefix else line)
                    except (ValueError, UnicodeError) as error:
                        self.errors.append({'path': key, 'offset': start, 'error': str(error)})
                        self.errors = self.errors[-64:]
            handle.seek(0)
            head = handle.read(min(4096, position))
            handle.seek(max(0, position - 128))
            tail = handle.read(min(128, position))
            self.positions[key] = position, head, tail
            if self.store:
                self.store.move(path, position, head, tail)
