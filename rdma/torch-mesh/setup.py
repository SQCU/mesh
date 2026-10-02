"""torch.distributed's "mesh" backend (torch_mesh/backend.py) over ../libnccl-mesh.dylib; torch_mesh/_stream.mm
hands libnccl-mesh's Metal path torch's MPS command buffer.

  make -C .. libmesh.dylib libnccl-mesh.dylib
  uv pip install --python <venv>/bin/python --no-build-isolation -e .

The torch.backends entry point registers the backend when torch is imported, so a script only names
it: torch.distributed.init_process_group(backend="mesh")."""
import os

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
from setuptools import setup  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CppExtension  # noqa: E402

setup(name='torch-mesh', version='0.3', packages=['torch_mesh'],
      ext_modules=[CppExtension('torch_mesh._stream', ['torch_mesh/_stream.mm'],
                                extra_link_args=['-framework', 'Metal', '-framework', 'Foundation'])],
      cmdclass={'build_ext': BuildExtension},
      entry_points={'torch.backends': ['mesh = torch_mesh:_autoload']})
