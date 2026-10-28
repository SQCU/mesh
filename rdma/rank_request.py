"""Configuration requests to a resident worker through the shared SSH launcher.

Only the existing worker's request/reply stream passes here. Tensor operations
remain native mesh calls; this module implements no collective or routing policy.
"""
import contextlib
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading


class Connection:
    def __init__(self, node, path):
        import mesh_nodes
        command = (f'PYTHONPATH={mesh_nodes.home_path(node.source + "/rdma")} {node.cluster.python} '
                   f'-m rank_request {shlex.quote(str(path))}')
        self.process = subprocess.Popen(node.argv(command), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.closed = False

    def sendall(self, data):
        self.process.stdin.write(data)
        self.process.stdin.flush()

    def recv(self, size):
        data = self.process.stdout.read1(size)
        if not data and self.process.wait() != 0:
            raise OSError(self.process.stderr.read().decode(errors='replace')[-2000:])
        return data

    def shutdown(self, _how=socket.SHUT_RDWR):
        if not self.process.stdin.closed:
            self.process.stdin.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.shutdown()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()
        self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def serve(path, incoming=None, outgoing=None):
    """Forward one existing JSON request to this node's resident Unix socket."""
    incoming = sys.stdin.buffer if incoming is None else incoming
    outgoing = sys.stdout.buffer if outgoing is None else outgoing
    line = incoming.readline()
    if not line.endswith(b'\n'):
        raise ValueError('an entire rank configuration request is required')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(os.path.expanduser(str(path)))
        connection.sendall(line)

        def abandoned():
            os.read(incoming.fileno(), 1)
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)

        threading.Thread(target=abandoned, daemon=True).start()
        while data := connection.recv(1 << 16):
            outgoing.write(data)
            outgoing.flush()


if __name__ == '__main__':
    serve(Path(sys.argv[1]))
