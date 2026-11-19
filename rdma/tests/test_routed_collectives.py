"""Arithmetic and queue progress through production whole-graph physical lowering."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RoutedCollectives(unittest.TestCase):
    def test_arithmetic_sparse_multigraph_and_relay_only_rank(self):
        compiler = shutil.which('clang') or shutil.which('cc')
        if compiler is None:
            self.skipTest('C compiler unavailable')
        with tempfile.TemporaryDirectory(prefix='mesh-routed-') as directory:
            executable = Path(directory) / 'routed'
            subprocess.run([
                compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', '-g',
                '-fsanitize=address,undefined', '-I', str(ROOT),
                str(ROOT / 'tests/routed.c'), str(ROOT / 'mesh-routed.c'),
                str(ROOT / 'mesh-fabric.c'), str(ROOT / 'mesh-collective.c'),
                '-lm', '-o', str(executable),
            ], check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
