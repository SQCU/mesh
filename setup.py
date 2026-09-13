from pathlib import Path
import shutil
import subprocess

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py


class NativeDistribution(Distribution):
    # design/algorithm-sources.md#indexed-library-functions
    def has_ext_modules(self):
        return True


class Build(build_py):
    # design/algorithm-sources.md#indexed-library-functions
    def run(self):
        subprocess.run(['make', '-C', 'rdma', 'libmesh.dylib', 'libmesh-algebra.dylib'], check=True)
        super().run()
        target = Path(self.build_lib) / 'mesh' / 'lib'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('libmesh.dylib', 'libmesh-algebra.dylib'):
            shutil.copy2(Path('rdma') / name, target / name)
        headers = Path(self.build_lib) / 'mesh' / 'include'
        headers.mkdir(parents=True, exist_ok=True)
        for name in ('mesh.h', 'mesh-dataflow.h', 'mesh-algebra.h', 'module.modulemap'):
            shutil.copy2(Path('rdma') / name, headers / name)


setup(cmdclass={'build_py': Build}, distclass=NativeDistribution)
