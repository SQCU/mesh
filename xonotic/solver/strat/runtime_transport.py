import errno
import json
import socket
import time
import uuid
from collections import defaultdict, deque

import numpy as np

from mesh import Mesh
from solver.xonwire import (
    LOCAL_HDR, LOCAL_STATUS, WIRE, HDRSZ,
    Reassembler, frame_count, frame_waves, parse_hdr, recv_datagram_frames,
)

NUMERICAL_REQUESTS = (WIRE["TENSOR_CONTROL"],)


def report(event, **values):
    print(json.dumps({'event': event, **values}), flush=True)


class RuntimeTransport:
    def __init__(self, service, numerical):
        self.service, self.numerical = service, numerical
        self.mesh = None
        self.instance = uuid.uuid4().int & ((1 << 64) - 1)
        self.attach_at = 0
        self.network = deque()
        self.messages = deque()
        self.local = defaultdict(deque)
        self.clients = set()
        self.reframing = {}
        self.errors = {}
        self.unrouted = deque()
        self.received = self.sent = 0
        self.service.setblocking(False)

    def status(self, address):
        if self.mesh is None:
            return
        data = LOCAL_STATUS.pack(self.mesh.slots, self.mesh.stride, self.mesh.usable,
            self.instance, self.queued() + self.mesh.inflight())
        self.local[address].append((LOCAL_HDR.pack(0, 0, WIRE['LOCAL_VERSION']), data))

    def transmit(self, node, frames):
        self.network.append((node, frames, 0))

    def queued(self):
        return sum(len(frames) - first for _, frames, first in self.network) + sum(count for _, _, count in self.messages)

    def rows(self, node, header, parts):
        counts = []
        for kind, rows in parts:
            count = frame_count(rows, self.mesh.usable)
            frames = frame_waves(kind, header['req_id'], header['tick'], rows,
                                self.mesh.usable, 64, header['session'])
            self.messages.append((node, frames, count))
            counts.append(count)
        return counts

    def outgoing(self, node, address, frames):
        if self.mesh is None or frames.shape[1] == self.mesh.usable:
            self.transmit(node, frames)
            return
        for frame in frames:
            header = parse_hdr(frame)
            if header is None:
                raise ValueError('cannot reframe an unrecognized game packet')
            if not header['values_total']:
                empty = np.zeros((1, self.mesh.usable), np.uint8)
                empty[0, :HDRSZ] = frame[:HDRSZ]
                self.transmit(node, empty)
                continue
            key = (address, node, header['kind'], header['session'], header['width'], len(frame))
            if key not in self.reframing:
                self.reframing[key] = Reassembler(header['kind'], header['width'], len(frame))
            receiver = self.reframing[key]
            record = receiver.feed(frame)
            if record is not None:
                self.rows(node, header, ((header['kind'], receiver.stage[:record['rows']].copy()),))

    def progress(self, accepting=True):
        activity = 0
        now = time.monotonic()
        if self.mesh is None and now >= self.attach_at:
            self.attach_at = now + 1
            try:
                self.mesh = Mesh(wait=False)
                report('runtime_transport_attached', region=self.mesh.region, slots=self.mesh.slots,
                       usable=self.mesh.usable, instance=self.instance)
                for address in self.clients:
                    self.status(address)
            except OSError as error:
                self.error('attach', error)
        for _ in range(256 if accepting else 0):
            try:
                packet = recv_datagram_frames(self.service, socket.MSG_DONTWAIT)
            except BlockingIOError:
                break
            except (OSError, ValueError) as error:
                self.error('local_receive', error)
                break
            activity += 1
            if packet is None:
                report('runtime_local_frame_invalid')
                continue
            node, address, frames = packet
            self.clients.add(address)
            if frames is None:
                self.status(address)
            else:
                try:
                    self.outgoing(node, address, frames)
                except ValueError as error:
                    self.error('local_payload', error)
        if self.mesh is not None:
            try:
                self.mesh.pump()
            except OSError as error:
                self.error('reattach', error)
                if not self.mesh.close():
                    self.mesh = None
        if self.mesh is not None:
            try:
                for frame, node in self.mesh.read(np.uint8, max_batches=1):
                    activity += 1
                    self.received += 1
                    header = parse_hdr(frame)
                    if header is not None and header['kind'] in NUMERICAL_REQUESTS:
                        self.numerical(node, header, frame)
                    else:
                        self.unrouted.append((node, frame.copy().reshape(1, -1)))
            except (OSError, ValueError) as error:
                self.error('receive', error)
            while self.unrouted and self.clients:
                node, frames = self.unrouted.popleft()
                envelope = LOCAL_HDR.pack(node, frames.shape[1], len(frames))
                for address in self.clients:
                    self.local[address].append((envelope, frames))
            for _ in range(8):
                if not self.messages or len(self.network) >= 256:
                    break
                node, waves, count = self.messages[0]
                frames = next(waves, None)
                if frames is None:
                    self.messages.popleft()
                else:
                    self.transmit(node, frames)
                    self.messages[0] = node, waves, count - len(frames)
            for _ in range(256):
                if not self.network:
                    break
                node, frames, first = self.network[0]
                if frames.shape[1] != self.mesh.usable:
                    self.network.popleft()
                    self.outgoing(node, '', frames[first:])
                    continue
                sent = self.mesh.send(frames[first:first + 64], node)
                if not sent:
                    break
                activity += sent
                self.sent += sent
                first += sent
                if first == len(frames):
                    self.network.popleft()
                else:
                    self.network[0] = node, frames, first
        for address, pending in list(self.local.items()):
            for _ in range(256):
                if not pending:
                    break
                envelope, frames = pending[0]
                try:
                    self.service.sendmsg((envelope, frames), (), socket.MSG_DONTWAIT, address)
                except OSError as error:
                    self.error(address, error)
                    if error.errno in (errno.ENOENT, errno.ECONNREFUSED):
                        report('runtime_game_departed', address=address, undeliverable_datagrams=len(pending))
                        self.clients.discard(address)
                        pending.clear()
                        self.reframing = {key: value for key, value in self.reframing.items() if key[0] != address}
                    break
                pending.popleft()
                activity += 1
            if not pending:
                del self.local[address]
        return activity

    def error(self, operation, error):
        message = f'{type(error).__name__}: {error}'
        if self.errors.get(operation) != message:
            report('runtime_transport_error', operation=operation, error=message)
            self.errors[operation] = message

    def close(self, deadline):
        while self.mesh is not None and time.monotonic() < deadline:
            if self.network or self.messages:
                self.progress(accepting=False)
            if not self.network and not self.messages:
                error = self.mesh.close()
                if not error:
                    self.mesh = None
                    return True
                if error != errno.EBUSY:
                    report('runtime_detach_error', errno=error)
            time.sleep(0.001)
        report('runtime_detach_pending', queued_frames=self.queued(),
               inflight=0 if self.mesh is None else self.mesh.inflight())
        return self.mesh is None
