from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Frame", "Replay"]

@dataclass(eq=False)
class Frame:
    id: int
    chorus: Any
    nbytes: int

class Replay:
    def __init__(self, capacity: int = 0, max_bytes: int = 1 << 30):
        self.capacity = max(0, int(capacity))
        self.max_bytes = int(max_bytes)
        self._items: list[dict] = []
        self._bytes = 0
        self._serial = 0
        self._frame_serial = 0

    def intern(self, chorus) -> Frame:
        n = sum(int(np.asarray(a).nbytes) for a in chorus)
        frame = Frame(self._frame_serial, chorus, n)
        self._frame_serial += 1
        return frame

    def frame(self, frame: Frame):
        return frame.chorus

    def push(self, item: dict):
        stored = dict(item, _serial=self._serial)
        self._serial += 1
        self._items.append(stored)
        self._evict()
        return stored

    @staticmethod
    def _item_bytes(item):
        return sum(
            int(np.asarray(value).nbytes)
            for value in item.values()
            if isinstance(value, (np.ndarray, np.generic))
        )

    def _frames(self):
        frames = {}
        for item in self._items:
            for name in ("frame_in", "frame_out"):
                frame = item[name]
                frames[frame.id] = frame
        return frames

    def _recount(self):
        self._bytes = sum(frame.nbytes for frame in self._frames().values())
        self._bytes += sum(self._item_bytes(item) for item in self._items)

    def _evict(self) -> None:
        self._recount()
        while self._items and (
            (self.capacity and len(self._items) > self.capacity)
            or self._bytes > self.max_bytes
        ):
            self._items.pop(0)
            self._recount()

    def materialize(self, item: dict) -> dict:
        out = dict(item)
        out["chorus_in"] = out["frame_in"].chorus
        out["chorus_out"] = out["frame_out"].chorus
        return out

    def sample(self, batch: int, rng: np.random.Generator) -> list[dict]:
        if not self._items:
            return []
        picks = rng.integers(0, len(self._items), size=min(batch, len(self._items)))
        out = []
        for p in picks:
            out.append(dict(self._items[int(p)]))
        return out

    def mean_age(self, items) -> float:
        return float(np.mean([self._serial - int(item["_serial"]) for item in items]))

    def release_unreferenced(self):
        self._recount()

    def export_payload(self, prefix="__replay__"):
        frames = sorted(self._frames().values(), key=lambda frame: frame.id)
        frame_index = {frame.id: index for index, frame in enumerate(frames)}
        payload = {}
        metadata = {
            "serial": self._serial,
            "frame_serial": self._frame_serial,
            "frames": [],
            "items": [],
        }
        for index, frame in enumerate(frames):
            keys = []
            for column, value in enumerate(frame.chorus):
                key = f"{prefix}frame_{index}_{column}"
                payload[key] = np.asarray(value)
                keys.append(key)
            metadata["frames"].append({"id": frame.id, "keys": keys})
        for index, item in enumerate(self._items):
            encoded = {}
            for name, value in item.items():
                if isinstance(value, Frame):
                    encoded[name] = {"frame": frame_index[value.id]}
                elif isinstance(value, (np.ndarray, np.generic)):
                    key = f"{prefix}item_{index}_{name}"
                    payload[key] = np.asarray(value)
                    encoded[name] = {"array": key}
                else:
                    encoded[name] = {"value": value}
            metadata["items"].append(encoded)
        payload[prefix + "meta"] = np.asarray(json.dumps(metadata, separators=(",", ":")))
        return payload

    def restore_payload(self, data, prefix="__replay__"):
        key = prefix + "meta"
        if key not in data.files:
            return False
        from .inputs import ChorusArrays

        metadata = json.loads(str(data[key]))
        frames = []
        for record in metadata["frames"]:
            chorus = ChorusArrays(*(np.asarray(data[name]) for name in record["keys"]))
            frames.append(Frame(int(record["id"]), chorus,
                                sum(int(value.nbytes) for value in chorus)))
        self._items = []
        for record in metadata["items"]:
            item = {}
            for name, encoded in record.items():
                if "frame" in encoded:
                    item[name] = frames[int(encoded["frame"])]
                elif "array" in encoded:
                    item[name] = np.asarray(data[encoded["array"]])
                else:
                    item[name] = encoded["value"]
            self._items.append(item)
        self._serial = int(metadata["serial"])
        self._frame_serial = int(metadata["frame_serial"])
        self._evict()
        return True

    def report(self):
        return {
            "bytes_per_transition": round(self._bytes / max(1, len(self._items)), 3),
            "frames": len(self._frames()),
        }

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def bytes_per_state(self) -> float:
        return self._bytes / max(1, len(self._items))
