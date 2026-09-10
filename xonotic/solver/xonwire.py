import json, os, re, socket, struct, time

import numpy as np

WIRE_DEFINITION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rdma", "xonwire.def")
with open(WIRE_DEFINITION) as stream:
    WIRE = {
        name: int(value, 0)
        for name, value in re.findall(r"MESH_WIRE\((\w+),\s*(0x[0-9a-fA-F]+|\d+)\)", stream.read())
    }
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
REQUEST_TIMEOUT_S = 5.0

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
    return (values + vps - 1) // vps

# ../../design/algorithm-sources.md#complete-page-ownership
def frame_waves(kind, req_id, tick, rows, usable, wave_slots, session, mesh, node):
    rows = np.asarray(rows, dtype=np.float32)
    width = rows.shape[1]
    flat = rows.reshape(-1)
    vps = values_per_slot(usable)
    frame_mass = (flat.size + vps - 1) // vps
    for first in range(0, frame_mass, wave_slots):
        count = min(frame_mass - first, wave_slots)
        frames = mesh.reserve(node, count)
        while frames is None:
            yield ()
            frames = mesh.reserve(node, count)
        for local in range(count):
            offset = (first + local) * vps
            nvalues = min(vps, flat.size - offset)
            frames[local, :HDRSZ] = pack_hdr(kind, req_id, tick, width, nvalues, offset, flat.size, session)
            frames[local, HDRSZ:HDRSZ + nvalues * 4].view(np.float32)[:] = flat[offset:offset + nvalues]
        yield frames


# ../../design/algorithm-sources.md#complete-page-ownership
def recv_datagram_frames(service, mesh, flags=0):
    header, address = service.recvfrom(LOCAL_HDR.size, socket.MSG_PEEK | flags)
    if len(header) != LOCAL_HDR.size:
        service.recvfrom(LOCAL_HDR.size, flags)
        return None
    node, framebytes, count = LOCAL_HDR.unpack(header)
    if not framebytes:
        service.recvfrom(LOCAL_HDR.size, flags)
        if count != WIRE['LOCAL_VERSION']:
            raise ValueError(f"local mesh protocol {count}, expected {WIRE['LOCAL_VERSION']}")
        return node, address, None
    if framebytes != mesh.usable:
        raise ValueError('game ingress must use the configured mesh page size')
    frames = mesh.reserve(node, count)
    if frames is None:
        raise BlockingIOError('literal receive pages are occupied')
    envelope = bytearray(LOCAL_HDR.size)
    received, _, _, address = service.recvmsg_into((envelope, frames), 0, flags)
    if received != LOCAL_HDR.size + frames.nbytes or LOCAL_HDR.unpack(envelope) != (node, framebytes, count):
        raise ValueError('incomplete game ingress page values')
    return node, address, frames

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

class FrameStream:
    def __init__(self, mesh):
        self.mesh = mesh
        self.session = int.from_bytes(os.urandom(8), "little")

    # ../../design/algorithm-sources.md#complete-page-ownership
    def exchange(self, kind, req_id, tick, rows, node, receivers, *, cancel, backlog, retry_s, timeout_s):
        received = {}
        pending = iter(frame_waves(kind, req_id, tick, rows, self.mesh.usable, self.mesh.slots, self.session, self.mesh, node))
        exhausted = False
        offers = 0
        started = time.monotonic()
        while not cancel() and time.monotonic() - started < timeout_s:
            self.mesh.pump()
            if not exhausted:
                frames = next(pending, None)
                exhausted = frames is None
                if frames is not None and len(frames):
                    offers += self.mesh.send(frames, node)
            for buf, source in self.mesh.read(np.uint8, max_batches=1):
                header = parse_hdr(buf)
                if header is not None and source == node and header["kind"] in receivers:
                    if header["session"] not in (0, self.session) or header["req_id"] != req_id or header["tick"] != tick:
                        continue
                    receiver = receivers[header["kind"]]
                    record = receiver.feed(buf)
                    if record is not None:
                        received[header["kind"]] = (record, receiver.stage[:record["rows"]].copy())
                else:
                    backlog.append((buf.copy(), source))
            if len(received) == len(receivers):
                return received, {"request_frame_offers": offers, "request_replays": 0, "timed_out": False}
        return None, {"request_frame_offers": offers, "request_replays": 0, "timed_out": not cancel(), "transaction_budget_s": timeout_s,
                      "response_parts": {str(kind): {"request": value.id, "tick": value.tick, "session": value.session,
                                                       "have": value.have, "want": value.want} for kind, value in receivers.items()}}

    def send(self, kind, req_id, tick, rows, node, cancel=None):
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        if rows.ndim != 2: raise ValueError('framed rows must be a matrix')
        started = time.monotonic()
        usable, pending, frames = None, None, None
        error = None
        sent = 0
        while time.monotonic() - started < REQUEST_TIMEOUT_S and not (cancel is not None and cancel()):
            try:
                self.mesh.pump()
                if usable != self.mesh.usable:
                    usable = self.mesh.usable
                    pending = iter(frame_waves(kind, req_id, tick, rows, usable, self.mesh.slots, self.session, self.mesh, node))
                    frames = None
                if frames is None:
                    frames = next(pending, None)
                    took = 0
                if frames is None: return sent, frame_count(rows, usable)
                if not len(frames):
                    frames = None
                    continue
                written = self.mesh.send(frames[took:], node)
                took += written
                sent += written
                if took == len(frames): frames = None
            except OSError as exception:
                error = f'{type(exception).__name__}: {exception}'
            time.sleep(0.0005)
        print(json.dumps({'event': 'frame_publication_incomplete', 'kind': kind, 'request': req_id,
            'session': self.session, 'offers': sent, 'elapsed_s': time.monotonic() - started, 'error': error}), flush=True)
        return sent, frame_count(rows, self.mesh.usable)
