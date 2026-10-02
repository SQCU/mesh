"""torch.distributed's "mesh" backend (torch_mesh/backend.py) over ../libnccl-mesh.dylib.

  make -C .. libmesh.dylib libnccl-mesh.dylib
  uv pip install --python <venv>/bin/python -e .

The torch.backends entry point registers the backend when torch is imported, so a script only names
it: torch.distributed.init_process_group(backend="mesh")."""
from setuptools import setup

setup(name='torch-mesh', version='0.2', packages=['torch_mesh'],
      entry_points={'torch.backends': ['mesh = torch_mesh:_autoload']})
