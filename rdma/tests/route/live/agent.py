"""A node's agent in the live routing loop (README.md): attaches its bridge's region, publishes the learner's w into the
bridge's split tables (the prior, w = 0, until the learner sends one), turns the bridge's records on and ships them to
the learner as they come; on SIGTERM or the learner's end, the records off and the tables withdrawn (the bridge's own
rule routes again). Memory and the socket only: nothing is written anywhere.

usage: python agent.py LEARNER_HOST PORT LIB [SAMPLE]     (LIB: live.c built as a shared library)"""
import ctypes, signal, socket, struct, sys, time
import numpy as np

host, port, libpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SAMPLE = int(sys.argv[4]) if len(sys.argv) > 4 else 16   # one piece in SAMPLE recorded, the same pieces at every node
lib = ctypes.CDLL(libpath)
lib.live_attach.restype = ctypes.c_int
lib.live_record_bytes.restype = ctypes.c_uint32
lib.live_self.restype = ctypes.c_uint32
lib.live_view.restype = ctypes.c_uint32
lib.live_contexts.restype = ctypes.c_uint32
lib.live_drain.restype = ctypes.c_uint32
P = ctypes.c_void_p
if lib.live_attach():
    sys.exit('agent: no bridge region of this build to attach')
RB = lib.live_record_bytes()
MOST = 4096
buf = np.zeros(MOST * RB, np.uint8)
dropped = ctypes.c_uint64(0)


def view():
    nodes = (ctypes.c_uint32 * 8)(); serial = ctypes.c_uint64(0)
    n = lib.live_view(nodes, ctypes.byref(serial))
    keys = np.zeros((256, 4), np.uint32)
    c = lib.live_contexts(keys.ctypes.data_as(P))
    return list(nodes[:n]), serial.value, keys[:c].copy()


def send(sock, kind, payload):
    sock.sendall(struct.pack('<cI', kind, len(payload)) + payload)


def recv_exact(sock, n):
    b = bytearray()
    while len(b) < n:
        x = sock.recv(n - len(b))
        if not x: raise ConnectionError('learner gone')
        b += x
    return bytes(b)


nodes, serial, keys = view()
self = lib.live_self()
w = np.zeros((len(keys), 40), np.float32)
lib.live_publish(keys.ctypes.data_as(P), ctypes.c_uint32(len(keys)), w.ctypes.data_as(P), ctypes.c_uint64(0))
sock = socket.create_connection((host, port))
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
hello = struct.pack('<III', self, RB, len(nodes)) + np.array(nodes, np.uint32).tobytes() + keys.tobytes()
send(sock, b'H', hello)
stop = False
def end(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM, end); signal.signal(signal.SIGINT, end)
lib.live_logging(ctypes.c_uint32(SAMPLE))
sock.setblocking(False)
inbox = bytearray()
try:
    while not stop:
        n = lib.live_drain(buf.ctypes.data_as(P), ctypes.c_uint32(MOST), ctypes.byref(dropped))
        if n:
            sock.setblocking(True)
            send(sock, b'R', struct.pack('<Q', dropped.value) + buf[:n * RB].tobytes())
            sock.setblocking(False)
        try:
            x = sock.recv(1 << 20)
            if not x: break
            inbox += x
        except BlockingIOError:
            pass
        while len(inbox) >= 5:
            kind, length = struct.unpack('<cI', inbox[:5])
            if len(inbox) < 5 + length: break
            body = bytes(inbox[5:5 + length]); del inbox[:5 + length]
            if kind == b'W':
                point = struct.unpack('<Q', body[:8])[0]
                w = np.frombuffer(body[8:], np.float32).reshape(len(keys), 40).copy()
                lib.live_publish(keys.ctypes.data_as(P), ctypes.c_uint32(len(keys)), w.ctypes.data_as(P), ctypes.c_uint64(point))
            elif kind == b'D':      # the learner's withdrawal: the bridge's own rule
                lib.live_withdraw()
            elif kind == b'P':      # the prior again
                w = np.zeros((len(keys), 40), np.float32)
                lib.live_publish(keys.ctypes.data_as(P), ctypes.c_uint32(len(keys)), w.ctypes.data_as(P), ctypes.c_uint64(0))
        nodes2, serial2, keys2 = view()
        if serial2 != serial and len(keys2) and not np.array_equal(keys2, keys):
            break          # the view's contexts changed: the learner's tables are for the old ones; end (it restarts us)
        if not n: time.sleep(0.0005)
finally:
    lib.live_logging(0)
    lib.live_withdraw()
