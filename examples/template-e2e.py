"""design/replay-templates-2026-10-29.md: the end-to-end test of templates, on a real program (the bandwidth workload).

  python examples/template-e2e.py CONFIGURATION|-     (each rank, as any example; RANK, WORLD_SIZE, MASTER_ADDR/PORT)

Each rank runs the program twice, each time a new process with its own communicator. The first prepares the program as
it always does (lowering, planning, recording), measures it and saves it as one template. The second makes the program
from that template on newly allocated operands, with no lowering or planning, and measures it. Each run's rows are the
program's own (us, GB/s per collective and size, implementation template-save or template-load, its preparation
seconds), in the program's order, the saving run's then the loading run's; each run checks its outputs against what
each collective defines, and a run whose outputs differ fails. The template lives in a temporary directory, removed
after both runs."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BANDWIDTH = Path(__file__).with_name('bandwidth.py')


def main():
    configuration = json.load(sys.stdin) if sys.argv[1] == '-' else json.loads(Path(sys.argv[1]).read_text())
    port = int(os.environ.get('MASTER_PORT', '29500'))
    with tempfile.TemporaryDirectory(prefix='template-') as directory:
        for k, phase in enumerate(('save', 'load')):
            run = {**configuration, 'template': {'directory': directory, 'phase': phase}}
            subprocess.run([sys.executable, str(BANDWIDTH), '-'], input=json.dumps(run).encode(),
                           env={**os.environ, 'MASTER_PORT': str(port + k)}, check=True)


if __name__ == '__main__':
    main()
