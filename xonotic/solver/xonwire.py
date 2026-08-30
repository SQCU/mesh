import os, struct, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rdma"))
from mesh import Mesh

MAGIC, VERSION, REQ, RESP = 0x584D5348, 1, 1, 2
HDR = struct.Struct("<IHHIIHHHHII")
HDRSZ = HDR.size
MAXCHUNK, TXSLOTS = 64, 4096


def rows_per_slot(usable, width):
    return (usable - HDRSZ) // (width * 4)


def pack_hdr(kind, req_id, tick, width, rows, chunk, chunks, rows_total):
    return np.frombuffer(
        HDR.pack(MAGIC, VERSION, kind, req_id, tick, width, rows, chunk, chunks, rows_total, 0),
        np.uint8,
    )


def parse_hdr(buf):
    if buf.size < HDRSZ:
        return None
    fields = HDR.unpack(buf[:HDRSZ].tobytes())
    magic, version, kind, req_id, tick, width, rows, chunk, chunks, rows_total, flags = fields
    valid = (
        magic == MAGIC
        and version == VERSION
        and 0 < chunks <= MAXCHUNK
        and chunk < chunks
        and width > 0
        and rows > 0
        and buf.size >= HDRSZ + rows * width * 4
    )
    if not valid:
        return None
    return dict(
        kind=kind, req_id=req_id, tick=tick, width=width, rows=rows,
        chunk=chunk, chunks=chunks, rows_total=rows_total, flags=flags,
    )


def payload(buf, header):
    size = header["rows"] * header["width"] * 4
    return np.frombuffer(buf[HDRSZ:HDRSZ + size].tobytes(), np.float32).reshape(
        header["rows"], header["width"]
    )


class Reassembler:
    def __init__(self, kind, width, maxrows, usable):
        self.kind, self.width = kind, width
        self.rps = rows_per_slot(usable, width)
        self.stage = np.zeros((maxrows, width), np.float32)
        self.id, self.mask, self.want, self.rows, self.tick = 0, 0, 0, 0, 0
        self.dropped, self.regress, self.resync = 0, 0, 0

    def feed(self, buf):
        header = parse_hdr(buf)
        if header is None or header["kind"] != self.kind or header["width"] != self.width:
            return None
        if header["req_id"] < self.id:
            self.regress += 1
            if self.regress < 3:
                self.dropped += 1
                return None
            self.resync += 1
        self.regress = 0
        if header["req_id"] != self.id:
            if self.id and self.mask != self.want:
                self.dropped += 1
            self.id = header["req_id"]
            self.mask = 0
            self.rows = header["rows_total"]
            self.tick = header["tick"]
            self.want = (1 << header["chunks"]) - 1
        base = header["chunk"] * self.rps
        if base + header["rows"] > self.stage.shape[0]:
            self.dropped += 1
            return None
        self.stage[base:base + header["rows"]] = payload(buf, header)
        self.mask |= 1 << header["chunk"]
        if self.mask == self.want:
            return dict(req_id=self.id, tick=self.tick, rows=min(self.rows, self.stage.shape[0]))
        return None


class TxWindow:
    def __init__(self, mesh, slots=TXSLOTS):
        self.mesh, self.slots, self.cursor = mesh, min(slots, mesh.slots), 0

    def send(self, kind, req_id, tick, rows, node):
        rows_total, width = rows.shape
        rps = rows_per_slot(self.mesh.usable, width)
        chunks = max(1, (rows_total + rps - 1) // rps)
        if self.cursor + chunks > self.slots:
            self.cursor = 0
        for chunk in range(chunks):
            part = rows[chunk * rps:(chunk + 1) * rps]
            slot = self.mesh.slot(self.cursor + chunk, np.uint8)
            slot[:HDRSZ] = pack_hdr(
                kind, req_id, tick, width, part.shape[0], chunk, chunks, rows_total
            )
            size = part.size * 4
            slot[HDRSZ:HDRSZ + size] = np.frombuffer(
                np.ascontiguousarray(part, np.float32).tobytes(), np.uint8
            )
        took = self.mesh.write(self.cursor, chunks, node)
        self.cursor += chunks
        return took, chunks
