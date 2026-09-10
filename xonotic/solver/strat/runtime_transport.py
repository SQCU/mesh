import errno
import json
import socket
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
        self.mesh = Mesh()
        self.instance = uuid.uuid4().int & ((1 << 64) - 1)
        self.messages = deque()
        self.local = defaultdict(deque)
        self.clients = set()
        self.reframing = {}
        self.errors = {}
        self.unrouted = deque()
        self.received = self.sent = 0
        self.service.setblocking(False)

    def status(self, address):
        data = LOCAL_STATUS.pack(self.mesh.slots, self.mesh.stride, self.mesh.usable,
            self.instance, self.queued())
        self.local[address].append((LOCAL_HDR.pack(0, 0, WIRE['LOCAL_VERSION']), data))

    def transmit(self, node, frames):
        sent = self.mesh.send(frames, node)
        self.sent += sent
        return sent

    def queued(self):
        return sum(count for _, _, count in self.messages)

    def rows(self, node, header, parts):
        counts = []
        for kind, rows in parts:
            count = frame_count(rows, self.mesh.usable)
            frames = frame_waves(kind, header['req_id'], header['tick'], rows,
                                self.mesh.usable, 64, header['session'], self.mesh, node)
            self.messages.append((node, frames, count))
            counts.append(count)
        return counts

    def progress(self, accepting=True):
        activity = 0
        for _ in range(256 if accepting else 0):
            try:
                packet = recv_datagram_frames(self.service, self.mesh, socket.MSG_DONTWAIT)
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
                self.transmit(node, frames)
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
        for _ in range(len(self.messages)):
            node, waves, count = self.messages.popleft()
            frames = next(waves, None)
            if frames is not None:
                activity += self.transmit(node, frames) if len(frames) else 0
                self.messages.append((node, waves, count - len(frames)))
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

    # ../../../design/algorithm-sources.md#context-lifetime
    def close(self):
        self.messages.clear()
        self.local.clear()
        self.unrouted.clear()
        status = self.mesh.close()
        self.mesh = None
        return status == 0
