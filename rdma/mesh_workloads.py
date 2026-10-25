"""Shared example input preparation. Program implementations remain library examples."""
import importlib
import json
from pathlib import Path
import sys


def build(examples, module, configuration, steps, rows):
    sys.path.insert(0, str(Path(examples).resolve()))
    program = importlib.import_module(module)
    config = json.loads(Path(configuration).read_text())
    if not Path(config['weights']).exists():
        program.inputs(config, int(steps), int(rows))


def build_main(argv=None):
    build(*(sys.argv[1:] if argv is None else argv))


if __name__ == '__main__':
    if sys.argv[1] != 'build':
        raise SystemExit('mesh_workloads build EXAMPLES MODULE CONFIG STEPS ROWS')
    build_main(sys.argv[2:])
