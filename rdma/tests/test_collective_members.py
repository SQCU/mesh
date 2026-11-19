"""Compile and evaluate production logical schedules for distinct rank cohorts."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CollectiveMembers(unittest.TestCase):
    def test_sparse_multigraph_arithmetic_and_uninitialized_relays(self):
        compiler = shutil.which('clang') or shutil.which('cc')
        if compiler is None:
            self.skipTest('C compiler unavailable')
        with tempfile.TemporaryDirectory(prefix='mesh-members-') as directory:
            executable = Path(directory)/'members'
            compiled = subprocess.run([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', '-g',
                '-fsanitize=address,undefined', '-I', str(ROOT),
                str(ROOT/'tests/collective-members.c'), str(ROOT/'mesh-collective.c'), '-lm', '-o', str(executable)],
                capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout+compiled.stderr)
            ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(ran.returncode, 0, ran.stdout+ran.stderr)
            self.assertIn('relay poison, global zero gaps PASS', ran.stdout)


if __name__ == '__main__':
    unittest.main()
