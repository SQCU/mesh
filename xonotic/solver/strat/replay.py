"""REPLAY — the one ring. Stores raw chorus arrays; derives nothing.

A replay entry holds exactly what cannot be recomputed:

    chorus_in / chorus_out   the seven ChorusArrays the composer consumes
    actions, behavior_logp   what was sampled, and under which policy
    reward, discount         the resolved return terms
    winner_mask, next_winner_mask
    dyn_y, dyn_u, target_delta   DINA's reduced state / action / observed delta

It does NOT store `logits`, `ir`, `query`, the values, the coupling, or any
per-(player, instrument) block. Those are outputs of ``strategy()`` and are
recomputed on sample — which is the whole reason a transition costs kilobytes
instead of hundreds of kilobytes, and the reason there is no ``(l, m, ·)``
tensor in the program at all.

Frames are INTERNED: consecutive transitions share endpoints, so a chorus is
stored once and referenced by index. `chorus_out` of one transition is
`chorus_in` of the next.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Replay"]


class Replay:
    """Ring of transitions, bounded by count and by bytes, oldest evicted first."""

    def __init__(self, capacity: int = 4096, max_bytes: int = 1 << 30):
        self.capacity = int(capacity)
        self.max_bytes = int(max_bytes)
        self._frames: list[Any] = []      # interned ChorusArrays
        self._frame_bytes: list[int] = []
        self._items: list[dict] = []
        self._bytes = 0

    # -- interning ---------------------------------------------------------

    def intern(self, chorus) -> int:
        """Store a chorus once; return its frame index."""
        self._frames.append(chorus)
        n = sum(int(np.asarray(a).nbytes) for a in chorus)
        self._frame_bytes.append(n)
        self._bytes += n
        return len(self._frames) - 1

    def frame(self, index: int):
        return self._frames[index]

    # -- transitions -------------------------------------------------------

    def push(self, item: dict) -> None:
        """`item` carries frame INDICES (`frame_in`, `frame_out`), never arrays."""
        self._items.append(item)
        self._bytes += sum(
            int(np.asarray(v).nbytes)
            for v in item.values()
            if isinstance(v, (np.ndarray, np.generic))
        )
        self._evict()

    def _evict(self) -> None:
        while self._items and (len(self._items) > self.capacity or self._bytes > self.max_bytes):
            dropped = self._items.pop(0)
            self._bytes -= sum(
                int(np.asarray(v).nbytes)
                for v in dropped.values()
                if isinstance(v, (np.ndarray, np.generic))
            )
            live = min((i["frame_in"] for i in self._items), default=len(self._frames))
            while self._frames and live > 0:
                self._bytes -= self._frame_bytes.pop(0)
                self._frames.pop(0)
                for i in self._items:
                    i["frame_in"] -= 1
                    i["frame_out"] -= 1
                live -= 1

    def sample(self, rng: np.random.Generator, batch: int) -> list[dict]:
        """Draw a minibatch, resolving frame indices back into chorus arrays."""
        if not self._items:
            return []
        picks = rng.integers(0, len(self._items), size=min(batch, len(self._items)))
        out = []
        for p in picks:
            item = dict(self._items[int(p)])
            item["chorus_in"] = self._frames[item["frame_in"]]
            item["chorus_out"] = self._frames[item["frame_out"]]
            out.append(item)
        return out

    # -- reporting ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def bytes_per_state(self) -> float:
        return self._bytes / max(1, len(self._items))
