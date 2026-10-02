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

RDMA = os.environ.get('MESH_RDMA') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
setup(name='torch-mesh', version='0.4', packages=['torch_mesh'],
      ext_modules=[CppExtension('torch_mesh._stream', ['torch_mesh/_stream.mm'], include_dirs=[RDMA], library_dirs=[RDMA],
                                libraries=['nccl-mesh'],
                                extra_link_args=[f'-Wl,-rpath,{RDMA}', '-framework', 'Metal', '-framework', 'Foundation'])],
      cmdclass={'build_ext': BuildExtension},
      entry_points={'torch.backends': ['mesh = torch_mesh:_autoload']})
