"""The game's wire at HEAD (rdma/xonwire.def, darkplaces-work/mesh_ipc.c), read and written by the solver and the
policy responder.

Frames: a 48-byte header (magic, version, kind, value offset, value total, request id, tick, width, values, session
low, session high) and float32 values; a message is its kind's rows, cut into frames of one width.  Reassembler joins
one kind's frames (57628982^:xonotic/solver/xonwire.py), RuntimeFrames joins OBSERVATION, CART, TEAM and STATE by
engine session and tick with EVENT optional and OUTCOME kept apart (design/TRAINING-CONTINUATION.md).

Engine is the local end the game's engine talks to (mesh_ipc.c meshx_relay_*; design/POLICY-TRANSPORT.md "local
control envelope"): a datagram socket at MESH_EXPERT_SOCKET whose datagrams carry `<iII` (node, frame bytes, count)
and count frames.  The engine's address is the source of its datagrams.  Between nodes the rows ride the library
(solver/worker.py, solver/strat/strat_responder.py), never this socket."""
import errno
import json
import os
import re
import socket
import struct
from pathlib import Path

import numpy as np

WIRE_DEFINITION = Path(__file__).resolve().parents[2] / 'rdma' / 'xonwire.def'
WIRE = {name: int(value, 0) for name, value in
        re.findall(r"MESH_WIRE\((\w+),\s*(0x[0-9a-fA-F]+|\d+)\)", WIRE_DEFINITION.read_text())}
MAGIC, VERSION = WIRE["MAGIC"], WIRE["VERSION"]
OBSERVATION_KIND, CART_KIND = WIRE["OBSERVATION"], WIRE["CART"]
EVENT_KIND, STRATEGY_KIND = WIRE["EVENT"], WIRE["STRATEGY"]
TEAM_KIND = WIRE["TEAM"]
STATE_KIND = WIRE["STATE"]
OUTCOME_KIND = WIRE["OUTCOME"]
HDR = struct.Struct("<IHHQQIIIIII")
HDRSZ = HDR.size
assert HDRSZ == WIRE["HDRBYTES"]
LOCAL_HDR = struct.Struct("<iII")
LOCAL_SLOTS = 64


def report(event, **values):
    print(json.dumps({'event': event, **values}), flush=True)


def kind_of(name):
    """A response kind given by name (an entry of xonwire.def) or number; None where the wire defines no such name."""
    return int(name) if isinstance(name, int) or str(name).isdigit() else WIRE.get(str(name))


def values_per_slot(usable):
    return (usable - HDRSZ) // 4


def pack_hdr(kind, req_id, tick, width, values, offset, values_total, session=0):
    return np.frombuffer(
        HDR.pack(MAGIC, VERSION, kind, offset, values_total, req_id, tick,
                 width, values, session & 0xffffffff, session >> 32),
        np.uint8,
    )


def parse_hdr(buf):
    if buf.size < HDRSZ:
        return None
    fields = HDR.unpack(buf[:HDRSZ].tobytes())
    magic, version, kind, offset, values_total, req_id, tick, width, values, session_lo, session_hi = fields
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
        offset=offset, values_total=values_total, session=session_lo | (session_hi << 32),
    )


def payload(buf, header):
    return np.frombuffer(buf, np.float32, header["values"], HDRSZ)


def frame_count(rows, usable):
    values = np.asarray(rows).size
    vps = values_per_slot(usable)
    return max(1, (values + vps - 1) // vps)


# mesh_ipc.c meshx_publish: a message's frames, each `usable` bytes, values cut at multiples of the frame's capacity
# (meshx_slot checks offset % values_per_page); an empty message keeps its explicit zero-row header.
def frames(kind, req_id, tick, rows, usable, session=0):
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError('framed rows must be a matrix')
    flat = rows.reshape(-1)
    vps = values_per_slot(usable)
    out = np.zeros((frame_count(rows, usable), usable), np.uint8)
    for index, frame in enumerate(out):
        offset = index * vps
        count = min(vps, flat.size - offset) if flat.size else 0
        frame[:HDRSZ] = pack_hdr(kind, req_id, tick, rows.shape[1], count, offset if count else 0, flat.size, session)
        frame[HDRSZ:HDRSZ + count * 4].view(np.float32)[:] = flat[offset:offset + count]
    return out


class Reassembler:
    def __init__(self, kind, width, usable):
        self.kind, self.width = kind, width
        self.vps = values_per_slot(usable)
        self.storage = np.empty((0, width), np.float32)
        self.stage = self.storage
        self.id, self.tick = 0, 0
        self.session = 0
        self.want, self.have, self.seen = 0, 0, bytearray()
        self.dropped, self.resync = 0, 0

    def reserve(self, rows):
        if rows > len(self.storage):
            self.storage = np.empty((rows, self.width), np.float32)
        self.stage = self.storage[:rows]

    def feed(self, buf):
        header = parse_hdr(buf)
        if header is None or header["kind"] != self.kind or header["width"] != self.width:
            return None
        vps = values_per_slot(buf.size)
        if vps != self.vps:
            self.vps = vps
            self.want = self.have = 0
            self.resync += 1
        offset = header["offset"]
        total = header["values_total"]
        if total == 0:
            self.id = header["req_id"]
            self.tick = header["tick"]
            self.session = header["session"]
            self.want = self.have = 1
            self.seen = bytearray((1,))
            self.reserve(0)
            return dict(req_id=self.id, tick=self.tick, session=self.session, rows=0, frame_mass=1)
        if total % self.width or offset % self.vps:
            self.dropped += 1
            return None
        index = offset // self.vps
        want = (total + self.vps - 1) // self.vps
        expected = min(self.vps, total - offset)
        if header["values"] != expected or index >= want:
            self.dropped += 1
            return None
        if not self.want or (header["session"], header["req_id"], header["tick"]) != (self.session, self.id, self.tick):
            if self.id and self.have != self.want and offset:
                self.dropped += 1
                return None
            if self.id and self.have != self.want:
                self.dropped += 1
            if self.id:
                self.resync += 1
            self.id = header["req_id"]
            self.tick = header["tick"]
            self.session = header["session"]
            self.want = want
            self.have = 0
            self.seen = bytearray((want + 7) // 8)
            self.reserve(total // self.width)
        if want != self.want or total != self.stage.size:
            self.dropped += 1
            return None
        bit = 1 << (index & 7)
        if not self.seen[index >> 3] & bit:
            self.seen[index >> 3] |= bit
            self.have += 1
        self.stage.reshape(-1)[offset:offset + expected] = payload(buf, header)
        if self.have == self.want:
            return dict(req_id=self.id, tick=self.tick, session=self.session, rows=len(self.stage),
                        frame_mass=self.want)
        return None


class RuntimeFrames:
    """One engine's messages joined into snapshots: OBSERVATION, CART, TEAM and STATE of one (session, tick) make a
    snapshot, and the latest complete snapshot is the one slot take() empties; EVENT rows and OUTCOME records
    accumulate until taken (design/TRAINING-CONTINUATION.md)."""

    def __init__(self, usable):
        self.usable = usable
        self.receivers, self.events, self.outcomes = {}, {}, {}
        self.assembling, self.latest = None, None

    def feed(self, buffer):
        header = parse_hdr(buffer)
        if header is None or header["kind"] not in (OBSERVATION_KIND, CART_KIND, TEAM_KIND, STATE_KIND, EVENT_KIND, OUTCOME_KIND):
            return
        kind = header["kind"]
        assembly = (kind, header["width"])
        if assembly not in self.receivers:
            self.receivers[assembly] = Reassembler(kind, header["width"], self.usable)
        receiver = self.receivers[assembly]
        complete = receiver.feed(buffer)
        if complete is None:
            return
        if kind == OUTCOME_KIND:
            for row in receiver.stage:
                identity = ":".join(str(int(value)) for value in row[:3])
                self.outcomes[identity] = {"kind": "score_win" if row[3] > 0 else "tie", "actor_team": int(row[3]),
                    "time": float(row[4]), "episode_id": identity, "source": "engine_outcome"}
            return
        key = (complete["session"], complete["tick"])
        if kind == EVENT_KIND:
            self.events[(*key, complete["req_id"])] = receiver.stage.copy()
            return
        if self.assembling is None or self.assembling[0] != key:
            self.assembling = (key, {})
        self.assembling[1][kind] = (complete, receiver.stage.copy())
        if len(self.assembling[1]) == 4:
            self.latest, self.assembling = self.assembling, None

    def take(self):
        if self.latest is None:
            return None
        (session, tick), records = self.latest
        self.latest = None
        events = {entry: self.events.pop(entry) for entry in tuple(self.events)
                  if entry[0] != session or entry[1] <= tick}
        return (session, tick), records, events

    def replay_snapshot(self, event_frames, outcomes):
        self.outcomes.update(outcomes)
        for key in event_frames:
            self.events.pop(key, None)

    def export_state(self):
        return {'outcomes': self.outcomes, 'events': self.events}

    def restore_state(self, saved):
        transport = saved.get('transport', saved)
        self.outcomes = dict(transport.get('outcomes', {}))
        self.events = dict(transport.get('events', transport.get('pending_events', {})))

    def report(self):
        return {"snapshot_ready": self.latest is not None, "pending_event_frames": len(self.events),
                "pending_event_rows": sum(len(rows) for rows in self.events.values()),
                "outcomes": len(self.outcomes),
                "assemblies": [{"kind": receiver.kind, "request": receiver.id, "session": receiver.session, "tick": receiver.tick,
                    "frames_received": receiver.have, "frames_expected": receiver.want,
                    "dropped_frames": receiver.dropped, "resynchronizations": receiver.resync}
                    for receiver in self.receivers.values()]}


class Engine:
    """The worker end of the engine's local socket (57628982^:xonotic/solver/strat/runtime_transport.py's local half,
    matrix_worker.py's bind_service).  The caller waits for the socket to be readable (fileno(), select) and
    receive() takes every datagram queued, returning the data frames as (node, address, frame); send() sends a
    message's frames to an engine at once, cut into datagrams of at most `slots` frames, halving where the socket
    refuses a datagram's size (EMSGSIZE, as mesh_ipc.c does for its own).  The engine reads the latest response it
    holds, so a send the socket refuses is reported and the next response is the next send."""

    def __init__(self, path, usable, slots=LOCAL_SLOTS, buffer_bytes=16 << 20):
        self.path, self.usable, self.slots = str(path), int(usable), int(slots)
        if values_per_slot(self.usable) < 1:
            raise ValueError(f'a frame of {self.usable} bytes holds no value past its {HDRSZ}-byte header')
        self.service = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.service.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
        self.service.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        self.service.bind(self.path)
        self.service.setblocking(False)
        self.receive_buffer = bytearray(self.service.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        self.errors = {}
        self.received = self.sent = self.invalid = 0

    def fileno(self):
        return self.service.fileno()

    def receive(self):
        taken = []
        view = memoryview(self.receive_buffer)
        while True:
            try:
                size, address = self.service.recvfrom_into(view)
            except BlockingIOError:
                return taken
            except OSError as error:
                self.error('local_receive', error)
                return taken
            if size < LOCAL_HDR.size:
                self.invalid += 1
                continue
            node, framebytes, count = LOCAL_HDR.unpack_from(self.receive_buffer)
            if framebytes < HDRSZ or framebytes * count != size - LOCAL_HDR.size:
                self.invalid += 1
                continue
            data = np.frombuffer(self.receive_buffer, np.uint8, framebytes * count, LOCAL_HDR.size).reshape(count, framebytes)
            for frame in data:
                taken.append((node, address, frame.copy()))
            self.received += count

    def send(self, address, node, framed):
        first = 0
        while first < len(framed):
            part = framed[first:first + self.slots]
            try:
                self.service.sendmsg((LOCAL_HDR.pack(node, part.shape[1], len(part)), part), (), socket.MSG_DONTWAIT, address)
            except OSError as error:
                if error.errno == errno.EMSGSIZE and len(part) > 1:
                    self.slots = (len(part) + 1) // 2
                    continue
                self.error(str(address), error)
                return
            first += len(part)
            self.sent += len(part)

    def error(self, operation, error):
        message = f'{type(error).__name__}: {error}'
        if self.errors.get(operation) != message:
            report('runtime_transport_error', operation=operation, error=message)
            self.errors[operation] = message

    def report(self):
        return {'received_frames': self.received, 'sent_frames': self.sent, 'invalid_datagrams': self.invalid}

    def close(self):
        self.service.close()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
