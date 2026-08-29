import os, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rdma"))
from mesh import Mesh

MAGIC, VERSION, REQ, RESP = 0x584D5348, 1, 1, 2
HDR = struct.Struct("<IHHIIHHHHII")
HDRSZ = HDR.size
REQ_WIDTH, RESP_WIDTH, MAXCHUNK, TXSLOTS = 16, 8, 64, 4096
TEAMS = 5


def rows_per_slot(usable, width):
    return (usable - HDRSZ) // (width * 4)


def pack_hdr(kind, req_id, tick, width, rows, chunk, chunks, rows_total):
    return np.frombuffer(HDR.pack(MAGIC, VERSION, kind, req_id, tick, width,
                                  rows, chunk, chunks, rows_total, 0), np.uint8)


def parse_hdr(buf):
    if buf.size < HDRSZ:
        return None
    magic, ver, kind, req_id, tick, width, rows, chunk, chunks, rows_total, flags = \
        HDR.unpack(buf[:HDRSZ].tobytes())
    if magic != MAGIC or ver != VERSION or chunks == 0 or chunks > MAXCHUNK or chunk >= chunks:
        return None
    if width == 0 or rows == 0 or buf.size < HDRSZ + rows * width * 4:
        return None
    return dict(kind=kind, req_id=req_id, tick=tick, width=width, rows=rows,
                chunk=chunk, chunks=chunks, rows_total=rows_total, flags=flags)


def payload(buf, h):
    n = h["rows"] * h["width"] * 4
    return np.frombuffer(buf[HDRSZ:HDRSZ + n].tobytes(), np.float32).reshape(h["rows"], h["width"])


REGRESS_K = 3


class Reassembler:
    """Reassembles chunked blocks, freshest req_id wins.

    A bare monotonic guard cannot tell a replay from a server restart, and a
    worker outlives server sessions by design. REGRESS_K consecutive chunks
    below the high-water mark are adopted as a new session; fewer are dropped
    as stragglers.
    """

    def __init__(self, kind, width, maxrows, usable):
        self.kind, self.width = kind, width
        self.rps = rows_per_slot(usable, width)
        self.stage = np.zeros((maxrows, width), np.float32)
        self.id, self.mask, self.want, self.rows, self.tick = 0, 0, 0, 0, 0
        self.dropped, self.regress, self.resync = 0, 0, 0

    def feed(self, buf):
        h = parse_hdr(buf)
        if h is None or h["kind"] != self.kind or h["width"] != self.width:
            return None
        if h["req_id"] < self.id:
            self.regress += 1
            if self.regress < REGRESS_K:
                self.dropped += 1
                return None
            self.resync += 1
        self.regress = 0
        if h["req_id"] != self.id:
            if self.id and self.mask != self.want:
                self.dropped += 1
            self.id, self.mask, self.rows, self.tick = h["req_id"], 0, h["rows_total"], h["tick"]
            self.want = (1 << h["chunks"]) - 1
        base = h["chunk"] * self.rps
        if base + h["rows"] > self.stage.shape[0]:
            self.dropped += 1
            return None
        self.stage[base:base + h["rows"]] = payload(buf, h)
        self.mask |= 1 << h["chunk"]
        if self.mask == self.want:
            return dict(req_id=self.id, tick=self.tick, rows=min(self.rows, self.stage.shape[0]))
        return None


class TxWindow:
    def __init__(self, mesh, slots=TXSLOTS):
        self.m, self.slots, self.cur = mesh, min(slots, mesh.slots), 0

    def send(self, kind, req_id, tick, X, node):
        rows_total, width = X.shape
        rps = rows_per_slot(self.m.usable, width)
        chunks = max(1, (rows_total + rps - 1) // rps)
        if self.cur + chunks > self.slots:
            self.cur = 0
        for c in range(chunks):
            rows = X[c * rps:(c + 1) * rps]
            s = self.m.slot(self.cur + c, np.uint8)
            s[:HDRSZ] = pack_hdr(kind, req_id, tick, width, rows.shape[0], c, chunks, rows_total)
            n = rows.size * 4
            s[HDRSZ:HDRSZ + n] = np.frombuffer(np.ascontiguousarray(rows, np.float32).tobytes(), np.uint8)
        took = self.m.write(self.cur, chunks, node)
        self.cur += chunks
        return took, chunks
