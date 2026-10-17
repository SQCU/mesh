"""The game's wire at HEAD (rdma/xonwire.def, darkplaces-work/mesh_ipc.c), read and written by the solver and the
policy responder.

Frames: a 48-byte header (magic, version, kind, value offset, value total, request id, tick, width, values, session
low, session high) and float32 values; a message is its kind's rows, cut into frames of one width.  Reassembler joins
one kind's frames (57628982^:xonotic/solver/xonwire.py), RuntimeFrames joins OBSERVATION, CART, TEAM and STATE by
engine session and tick with EVENT optional and OUTCOME kept apart (design/TRAINING-CONTINUATION.md).

Engine is the local end the game's engine talks to (mesh_ipc.c meshx_relay_*; design/POLICY-TRANSPORT.md "local
control envelope"): a datagram socket at MESH_EXPERT_SOCKET whose datagrams carry `<iII` (node, frame bytes, count)
and count frames; zero frame bytes is a hello, answered with five `<Q` (slots, stride, usable, incarnation, queued).
Between nodes the rows ride the library (solver/worker.py, solver/strat/strat_responder.py), never this socket."""
import errno
import json
import os
import re
import socket
import struct
import uuid
from collections import deque
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
LOCAL_STATUS = struct.Struct("<QQQQQ")
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
    def __init__(self, usable, capacity=8):
        self.usable, self.capacity = usable, capacity
        self.receivers, self.pending, self.events, self.outcomes = {}, {}, {}, {}
        self.event_seen = set()
        self.watermarks = {}
        self.active_session = None
        self.retired_sessions = set()
        self.dropped = self.duplicates = 0

    def feed(self, buffer):
        header = parse_hdr(buffer)
        if header is None or header["kind"] not in (OBSERVATION_KIND, CART_KIND, TEAM_KIND, STATE_KIND, EVENT_KIND, OUTCOME_KIND):
            return
        kind, session, tick = (header[key] for key in ("kind", "session", "tick"))
        if kind not in (OUTCOME_KIND, EVENT_KIND) and session in self.retired_sessions:
            self.duplicates += 1
            return
        if kind not in (OUTCOME_KIND, EVENT_KIND) and tick <= self.watermarks.get(session, -1):
            self.duplicates += 1
            return
        event_key = (session, tick, header['req_id'])
        if kind == EVENT_KIND and event_key in self.event_seen:
            self.duplicates += 1
            return
        assembly = (kind, header["width"], *event_key) if kind == EVENT_KIND else (kind, header["width"])
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
                    "time": float(row[4]), "episode_id": identity, "source": "engine_durable_journal"}
            return
        key = (session, tick)
        if kind == EVENT_KIND:
            self.events[event_key] = receiver.stage.copy()
            self.event_seen.add(event_key)
            self.receivers.pop(assembly)
        else:
            self.pending.setdefault(key, {})[kind] = (complete, receiver.stage.copy())
        if len(self.pending) > self.capacity:
            terminals = {}
            for key, records in self.pending.items():
                terminals.setdefault(self.terminal_identity(key, records), key)
            retained = {key for identity, key in terminals.items() if identity is not None}
            disposable = [key for key in self.pending if key not in retained]
            for key in disposable[:len(self.pending) - self.capacity]:
                self.pending.pop(key)
                self.dropped += 1

    @staticmethod
    def terminal_identity(key, records):
        from payload.tools.strategy_io_schema import TS
        if len(records) == 4 and records[TEAM_KIND][1][0, TS['FINISHED']]:
            session, _ = key
            return f"{session >> 24}:{session & 0xffffff}:{int(records[TEAM_KIND][1][0, TS['EPISODE']])}"

    def take(self):
        ready = [key for key, records in self.pending.items() if len(records) == 4]
        if not ready:
            return None
        key = next((key for key in ready if self.terminal_identity(key, self.pending[key]) is not None), ready[-1])
        records = self.pending[key]
        terminal = self.terminal_identity(key, records)
        if terminal is not None and terminal not in self.outcomes:
            return None
        self.accept_snapshot(key)
        session, tick = key
        events = {entry: self.events.pop(entry) for entry in tuple(self.events)
                  if (entry[0] == session and entry[1] <= tick) or entry[0] in self.retired_sessions}
        return key, records, events

    def accept_snapshot(self, key):
        session, tick = key
        if self.active_session is not None and self.active_session != session:
            self.retired_sessions.add(self.active_session)
        self.active_session = session
        for entry in tuple(self.pending):
            owner, stamp = entry
            if (owner == session and stamp <= tick) or owner in self.retired_sessions:
                self.pending.pop(entry)
        self.watermarks[session] = tick

    def replay_snapshot(self, key, event_frames, outcomes):
        self.accept_snapshot(key)
        self.outcomes.update(outcomes)
        self.event_seen.update(event_frames)
        for key in event_frames:
            self.events.pop(key, None)
        for assembly in tuple(self.receivers):
            if assembly[0] == EVENT_KIND and assembly[2:] in event_frames:
                self.receivers.pop(assembly)

    def export_state(self):
        return {**vars(self), 'receivers': {key: {**vars(receiver),
            'seen': np.frombuffer(receiver.seen, dtype=np.uint8)}
            for key, receiver in self.receivers.items()}}

    def restore_state(self, saved):
        if 'transport' in saved:
            state = dict(saved['transport'])
            receivers = state.pop('receivers')
            self.__dict__.update(state)
            self.receivers = {}
            for key, fields in receivers.items():
                receiver = Reassembler(fields['kind'], fields['width'], self.usable)
                receiver.__dict__.update(fields, seen=bytearray(fields['seen']))
                self.receivers[key] = receiver
        else:
            self.watermarks, self.outcomes = saved['watermarks'], saved['outcomes']
            self.events, self.event_seen = saved.get('pending_events', {}), saved.get('event_seen', set())
            self.active_session = saved.get('active_session')
            self.retired_sessions = saved.get('retired_sessions', set())

    def report(self):
        return {"pending_snapshots": len(self.pending), "dropped_frames": self.dropped,
                "pending_event_frames": len(self.events), "pending_event_rows": sum(len(rows) for rows in self.events.values()),
                "retained_event_identities": len(self.event_seen),
                "duplicate_fragments": self.duplicates, "durable_outcomes": len(self.outcomes),
                "pending_by_kind": {str(kind): sum(kind in records for records in self.pending.values())
                                    for kind in (OBSERVATION_KIND, CART_KIND, TEAM_KIND, STATE_KIND)},
                "assemblies": [{"kind": receiver.kind, "request": receiver.id, "session": receiver.session, "tick": receiver.tick,
                    "frames_received": receiver.have, "frames_expected": receiver.want,
                    "dropped_frames": receiver.dropped, "resynchronizations": receiver.resync}
                    for receiver in self.receivers.values()]}


class Engine:
    """The worker end of the engine's local socket (57628982^:xonotic/solver/strat/runtime_transport.py's local half,
    matrix_worker.py's bind_service): every receive and send nonblocking, a bounded number a call, so a slow or departed
    engine never holds the caller.  receive() answers hellos itself and returns the data frames it took as (node,
    address, frame); send() queues a message's frames for an engine, cut into datagrams of at most LOCAL_SLOTS frames,
    halving where the socket refuses a datagram's size (EMSGSIZE, as mesh_ipc.c does for its own)."""

    def __init__(self, path, usable, slots=LOCAL_SLOTS, buffer_bytes=16 << 20):
        self.path, self.usable, self.slots = str(path), int(usable), int(slots)
        if values_per_slot(self.usable) < 1:
            raise ValueError(f'a frame of {self.usable} bytes holds no value past its {HDRSZ}-byte header')
        self.instance = uuid.uuid4().int & ((1 << 64) - 1)
        self.service = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.service.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
        self.service.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        self.service.bind(self.path)
        self.service.setblocking(False)
        self.receive_buffer = bytearray(LOCAL_HDR.size + LOCAL_SLOTS * self.usable)
        self.local = {}
        self.clients = set()
        self.errors = {}
        self.received = self.sent = self.hellos = self.invalid = 0

    def queued(self):
        return sum(len(pending) for pending in self.local.values())

    def status(self, address):
        self.local.setdefault(address, deque()).appendleft((LOCAL_HDR.pack(0, 0, WIRE['LOCAL_VERSION']), np.frombuffer(
            LOCAL_STATUS.pack(self.slots, self.usable, self.usable, self.instance, self.queued()), np.uint8)))

    def receive(self, budget=256):
        taken = []
        view = memoryview(self.receive_buffer)
        for _ in range(budget):
            try:
                size, address = self.service.recvfrom_into(view)
            except BlockingIOError:
                break
            except OSError as error:
                self.error('local_receive', error)
                break
            if size < LOCAL_HDR.size:
                self.invalid += 1
                continue
            node, framebytes, count = LOCAL_HDR.unpack_from(self.receive_buffer)
            if address:
                self.clients.add(address)
            if not framebytes:
                if count != WIRE['LOCAL_VERSION']:
                    self.error('local_version', ValueError(f"local mesh protocol {count}, expected {WIRE['LOCAL_VERSION']}"))
                    continue
                self.hellos += 1
                if address:
                    self.status(address)
                continue
            if framebytes < HDRSZ or framebytes * count != size - LOCAL_HDR.size:
                self.invalid += 1
                continue
            data = np.frombuffer(self.receive_buffer, np.uint8, framebytes * count, LOCAL_HDR.size).reshape(count, framebytes)
            for frame in data:
                taken.append((node, address, frame.copy()))
            self.received += count
        return taken

    def send(self, address, node, framed):
        pending = self.local.setdefault(address, deque())
        for first in range(0, len(framed), self.slots):
            part = framed[first:first + self.slots]
            pending.append((LOCAL_HDR.pack(node, part.shape[1], len(part)), part))

    def progress(self, budget=256):
        moved = 0
        for address, pending in list(self.local.items()):
            for _ in range(budget):
                if not pending:
                    break
                envelope, part = pending[0]
                try:
                    self.service.sendmsg((envelope, part), (), socket.MSG_DONTWAIT, address)
                except OSError as error:
                    if error.errno == errno.EMSGSIZE and part.ndim == 2 and len(part) > 1:
                        node = LOCAL_HDR.unpack(envelope)[0]
                        half = (len(part) + 1) // 2
                        pending.popleft()
                        pending.appendleft((LOCAL_HDR.pack(node, part.shape[1], len(part) - half), part[half:]))
                        pending.appendleft((LOCAL_HDR.pack(node, part.shape[1], half), part[:half]))
                        self.slots = max(1, half)
                        continue
                    if error.errno in (errno.ENOENT, errno.ECONNREFUSED):
                        report('runtime_game_departed', address=str(address), undeliverable_datagrams=len(pending))
                        self.clients.discard(address)
                        pending.clear()
                        break
                    if error.errno not in (errno.EAGAIN, errno.ENOBUFS):
                        self.error(str(address), error)
                    break
                pending.popleft()
                self.sent += len(part) if part.ndim == 2 else 0
                moved += 1
            if not pending:
                del self.local[address]
        return moved

    def error(self, operation, error):
        message = f'{type(error).__name__}: {error}'
        if self.errors.get(operation) != message:
            report('runtime_transport_error', operation=operation, error=message)
            self.errors[operation] = message

    def report(self):
        return {'received_frames': self.received, 'sent_frames': self.sent, 'hellos': self.hellos,
                'invalid_datagrams': self.invalid, 'game_clients': len(self.clients), 'queued_datagrams': self.queued()}

    def close(self):
        self.local.clear()
        self.service.close()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
