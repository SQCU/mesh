import errno, os, re, socket, struct, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rdma"))
from mesh import Mesh

WIRE_DEFINITION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rdma", "xonwire.def")
with open(WIRE_DEFINITION) as stream:
    WIRE = {
        name: int(value, 0)
        for name, value in re.findall(r"MESH_WIRE\((\w+),\s*(0x[0-9a-fA-F]+|\d+)\)", stream.read())
    }
MAGIC, VERSION = WIRE["MAGIC"], WIRE["VERSION"]
OBSERVATION_KIND, CART_KIND = WIRE["OBSERVATION"], WIRE["CART"]
EVENT_KIND, STRATEGY_KIND = WIRE["EVENT"], WIRE["STRATEGY"]
EXPERT_REQ, EXPERT_RESP, EXPERT_META_KIND = WIRE["EXPERT_REQ"], WIRE["EXPERT_RESP"], WIRE["EXPERT_META"]
EXPERT_TRAIN_REQ, EXPERT_GRAD_REQ = WIRE["EXPERT_TRAIN_REQ"], WIRE["EXPERT_GRAD_REQ"]
EXPERT_GRAD_RESP, EXPERT_GRAD_META_KIND = WIRE["EXPERT_GRAD_RESP"], WIRE["EXPERT_GRAD_META"]
EXPERT_BATCH_BEGIN = WIRE["EXPERT_BATCH_BEGIN"]
EXPERT_BATCH_COMMIT, EXPERT_BATCH_RESP = WIRE["EXPERT_BATCH_COMMIT"], WIRE["EXPERT_BATCH_RESP"]
EXPERT_META = dict(
    MATRIX_MIN=0, MATRIX_MAX=1, MATRIX_FINITE_MASS=2, ROWS=3, ELAPSED=4,
)
EXPERT_META_VALUE_WIDTH = max(EXPERT_META.values()) + 1
EXPERT_GRAD_META = dict(ROWS=0, ELAPSED=1, GRADIENT_NORM=2, UPDATES=3)
EXPERT_GRAD_META_WIDTH = max(EXPERT_GRAD_META.values()) + 1
HDR = struct.Struct("<IHHQQIIIIII")
HDRSZ = HDR.size
assert HDRSZ == WIRE["HDRBYTES"]
LOCAL_HDR = struct.Struct("<iII")

def expert_meta_width(experts):
    return EXPERT_META_VALUE_WIDTH + int(experts)

def values_per_slot(usable):
    return (usable - HDRSZ) // 4

def pack_hdr(kind, req_id, tick, width, values, offset, values_total):
    return np.frombuffer(
        HDR.pack(MAGIC, VERSION, kind, offset, values_total, req_id, tick,
                 width, values, 0, 0),
        np.uint8,
    )

def parse_hdr(buf):
    if buf.size < HDRSZ:
        return None
    fields = HDR.unpack(buf[:HDRSZ].tobytes())
    magic, version, kind, offset, values_total, req_id, tick, width, values, _, _ = fields
    valid = (
        magic == MAGIC
        and version == VERSION
        and width > 0
        and ((values_total == 0 and values == 0 and offset == 0)
             or (values > 0 and offset < values_total and offset + values <= values_total))
        and buf.size >= HDRSZ + values * 4
    )
    if not valid:
        return None
    return dict(
        kind=kind, req_id=req_id, tick=tick, width=width, values=values,
        offset=offset, values_total=values_total,
    )

def payload(buf, header):
    return np.frombuffer(buf, np.float32, header["values"], HDRSZ)

def frame_count(rows, usable):
    values = np.asarray(rows).size
    vps = values_per_slot(usable)
    return (values + vps - 1) // vps

def frame_waves(kind, req_id, tick, rows, usable, wave_slots):
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if rows.ndim != 2 or not rows.size:
        return
    width = rows.shape[1]
    flat = rows.reshape(-1)
    vps = values_per_slot(usable)
    frame_mass = (flat.size + vps - 1) // vps
    first = 0
    while first < frame_mass:
        wave = min(frame_mass - first, wave_slots)
        frames = np.zeros((wave, usable), np.uint8)
        base = first * vps
        count = min(wave * vps, flat.size - base)
        values = np.zeros(wave * vps, np.float32)
        values[:count] = flat[base:base + count]
        frames[:, HDRSZ:HDRSZ + vps * 4] = values.reshape(wave, vps).view(np.uint8)
        for local in range(wave):
            offset = (first + local) * vps
            nvalues = min(vps, flat.size - offset)
            frames[local, :HDRSZ] = pack_hdr(
                kind, req_id, tick, width, nvalues, offset, flat.size,
            )
        yield frames
        first += wave

def send_datagram_rows(service, address, node, usable, kind, req_id, tick, rows):
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    frame_mass = frame_count(rows, usable)
    socket_slots = max(1, service.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) // usable)
    for frames in frame_waves(kind, req_id, tick, rows, usable, socket_slots):
        first = 0
        measured_slots = len(frames)
        while first < len(frames):
            count = min(len(frames) - first, measured_slots)
            header = LOCAL_HDR.pack(int(node), usable, count)
            try:
                service.sendmsg((header, frames[first:first + count]), (), socket.MSG_DONTWAIT, address)
                first += count
            except OSError as exc:
                if exc.errno == errno.EMSGSIZE and count > 1:
                    measured_slots = (count + 1) // 2
                    continue
                if exc.errno not in (errno.ENOBUFS, errno.EAGAIN):
                    raise
                time.sleep(0.0005)
    return frame_mass

def recv_datagram_frames(service, flags=0):
    header, address = service.recvfrom(LOCAL_HDR.size, socket.MSG_PEEK | flags)
    if len(header) != LOCAL_HDR.size:
        service.recvfrom(LOCAL_HDR.size, flags)
        return None
    node, framebytes, count = LOCAL_HDR.unpack(header)
    bytes_total = framebytes * count
    envelope = bytearray(LOCAL_HDR.size)
    slot = bytearray(bytes_total)
    received, _, _, address = service.recvmsg_into((envelope, slot), 0, flags)
    if received != LOCAL_HDR.size + bytes_total or LOCAL_HDR.unpack(envelope) != (node, framebytes, count):
        return None
    return node, address, np.frombuffer(slot, np.uint8).reshape(count, framebytes)

class Reassembler:
    def __init__(self, kind, width, usable):
        self.kind, self.width = kind, width
        self.vps = values_per_slot(usable)
        self.stage = np.empty((0, width), np.float32)
        self.id, self.tick = 0, 0
        self.want, self.have, self.seen = 0, 0, bytearray()
        self.dropped, self.resync = 0, 0

    def feed(self, buf):
        header = parse_hdr(buf)
        if header is None or header["kind"] != self.kind or header["width"] != self.width:
            return None
        offset = header["offset"]
        total = header["values_total"]
        if total == 0:
            self.id = header["req_id"]
            self.tick = header["tick"]
            self.want = self.have = 1
            self.seen = bytearray((1,))
            self.stage = np.empty((0, self.width), np.float32)
            return dict(req_id=self.id, tick=self.tick, rows=0, frame_mass=1)
        if total % self.width or offset % self.vps:
            self.dropped += 1
            return None
        index = offset // self.vps
        want = (total + self.vps - 1) // self.vps
        expected = min(self.vps, total - offset)
        if header["values"] != expected or index >= want:
            self.dropped += 1
            return None
        if header["req_id"] != self.id:
            if self.id and self.have != self.want and offset:
                self.dropped += 1
                return None
            if self.id and self.have != self.want:
                self.dropped += 1
            if self.id:
                self.resync += 1
            self.id = header["req_id"]
            self.tick = header["tick"]
            self.want = want
            self.have = 0
            self.seen = bytearray((want + 7) // 8)
            self.stage = np.empty((total // self.width, self.width), np.float32)
        if want != self.want or total != self.stage.size:
            self.dropped += 1
            return None
        bit = 1 << (index & 7)
        if not self.seen[index >> 3] & bit:
            self.seen[index >> 3] |= bit
            self.have += 1
        self.stage.reshape(-1)[offset:offset + expected] = payload(buf, header)
        if self.have == self.want:
            return dict(req_id=self.id, tick=self.tick, rows=len(self.stage),
                        frame_mass=self.want)
        return None

class FrameStream:
    def __init__(self, mesh):
        self.mesh = mesh

    def send(self, kind, req_id, tick, rows, node, cancel=None):
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        if rows.ndim != 2 or not rows.size:
            return 0, 0
        frame_mass = frame_count(rows, self.mesh.usable)
        sent = 0
        for frames in frame_waves(
            kind, req_id, tick, rows, self.mesh.usable, self.mesh.slots,
        ):
            took = 0
            while took < len(frames):
                written = self.mesh.send(frames[took:], node)
                if written:
                    took += written
                    sent += written
                    continue
                if cancel is not None and cancel():
                    return sent, frame_mass
                time.sleep(0.0005)
        return sent, frame_mass
