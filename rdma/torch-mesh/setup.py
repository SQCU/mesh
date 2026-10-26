"""torch.distributed's "mesh" backend (torch_mesh/backend.py) over ../libnccl-mesh.dylib; torch_mesh/_stream.mm
binds FX numerical and transport commands into the common native recorder.

  make -C .. all install
  uv pip install --python <venv>/bin/python --no-build-isolation -e .

The torch.backends entry point registers the backend when torch is imported, so a script only names
it: torch.distributed.init_process_group(backend="mesh"). Numerical functions use
torch_mesh.declarations.Function to declare and prepare their complete graph."""
import os
import subprocess
from pathlib import Path

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
RDMA = os.environ.get('MESH_RDMA') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
profile = os.environ.get('METAL_CONFIGURATION') or str(Path(RDMA).parent / 'etc/compute/metal.mk')
os.environ['MACOSX_DEPLOYMENT_TARGET'] = subprocess.check_output(
    ['make', '-s', '-f', profile, 'deployment-target'], text=True).strip()
target = '-mmacosx-version-min=' + os.environ['MACOSX_DEPLOYMENT_TARGET']
from setuptools import setup  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CppExtension  # noqa: E402

setup(name='torch-mesh', version='0.4', packages=['torch_mesh'], package_data={'torch_mesh': ['*.metal']},
      ext_modules=[CppExtension('torch_mesh._stream', ['torch_mesh/_stream.mm'], include_dirs=[RDMA, str(Path(RDMA).parent / 'etc/compute')], library_dirs=[RDMA],
                                libraries=['nccl-mesh'],
                                extra_compile_args=[target, '-Werror=unguarded-availability-new'],
                                extra_link_args=[target, f'-Wl,-rpath,{RDMA}', '-framework', 'Metal', '-framework', 'Foundation'])],
      cmdclass={'build_ext': BuildExtension},
      entry_points={'torch.backends': ['mesh = torch_mesh:_autoload']})
