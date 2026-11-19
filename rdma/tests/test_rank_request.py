import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh_nodes
from rank_request import Connection


class LocalLauncher:
    """Exercise the real imported worker helper without requiring an SSH host."""
    source = 'mesh with spaces'

    def __init__(self, path):
        self.path, self.commands = path, []

    def argv(self, command):
        self.commands.append(command)
        return [sys.executable, str(Path(__file__).resolve().parents[1] / 'rank_request.py'), str(self.path)]


class RankRequestTests(unittest.TestCase):
    def test_zero_compute_worker_receives_complete_configuration_and_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'worker.sock'
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path)); listener.listen()
            observed = []
            request = {'rank': 2, 'program': {'idle': [2], 'nodes': [101, 102, 103],
                       'flows': [{'nodes': [101, 102, 103]}]}, 'steps': 3}
            response = b'{"steps":3,"function_seconds":0.01}\n{"result":0}\n'

            def worker():
                with listener.accept()[0] as peer:
                    stream = peer.makefile('rb')
                    observed.append(json.loads(stream.readline()))
                    peer.sendall(response)
                    stream.close()

            thread = threading.Thread(target=worker); thread.start()
            launch = LocalLauncher(path)
            with Connection(launch, path) as connection:
                connection.sendall(json.dumps(request).encode() + b'\n')
                received = bytearray()
                while chunk := connection.recv(7): received.extend(chunk)
            thread.join(3); listener.close()
            self.assertFalse(thread.is_alive())
            self.assertEqual(observed, [request])
            self.assertEqual(received, response)
            self.assertIn('-m rank_request', launch.commands[0])
            self.assertIn('mesh with spaces/rdma', launch.commands[0])

    def test_requester_abandonment_reaches_worker_without_an_execution_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'worker.sock'
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path)); listener.listen()
            began, abandoned = threading.Event(), threading.Event()

            def worker():
                with listener.accept()[0] as peer:
                    stream = peer.makefile('rb'); stream.readline(); began.set()
                    if stream.read() == b'': abandoned.set()
                    stream.close()

            thread = threading.Thread(target=worker); thread.start()
            with Connection(LocalLauncher(path), path) as connection:
                connection.sendall(b'{"rank":2}\n')
                self.assertTrue(began.wait(3))
                connection.shutdown()
                self.assertTrue(abandoned.wait(3))
            thread.join(3); listener.close()

    def test_shared_ssh_launcher_preserves_proxy_and_has_no_payload_code(self):
        node = mesh_nodes.Node('103', 'fd00::3', via='fd00::1')
        args = node.argv('configured command')
        self.assertEqual(args[0], 'ssh')
        self.assertIn('mdot@fd00::3', args)
        self.assertTrue(any(arg.startswith('ProxyCommand=ssh ') for arg in args))
        self.assertEqual(args[-1], 'configured command')


if __name__ == '__main__':
    unittest.main()
