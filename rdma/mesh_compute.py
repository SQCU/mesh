"""Hardware inputs read once during preparation, never during numerical calls."""
from functools import cache
import json
import plistlib
import subprocess


def _read(*command):
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'compute capability {" ".join(command)}: {result.stderr.decode().strip()}')
    return result.stdout


def _sysctl(name):
    value = _read('sysctl', '-n', name).decode().strip()
    if not value:
        raise RuntimeError(f'compute capability {name}: no value')
    return value


@cache
def hardware():
    """Keep CPU classes and cache clusters separate from the GPU core count.

    Counts describe hardware, not operation rates or CPU affinity promises.
    Missing facts are preparation errors; there is no substitute platform.
    """
    levels = []
    for index in range(int(_sysctl('hw.nperflevels'))):
        prefix = f'hw.perflevel{index}'
        levels.append({'name': _sysctl(prefix + '.name'), **{
            field: int(_sysctl(prefix + '.' + key)) for field, key in (
                ('physical_cores', 'physicalcpu'), ('logical_cores', 'logicalcpu'),
                ('cores_per_l2', 'cpusperl2'), ('l2_bytes', 'l2cachesize'))}})
    physical, logical = (int(_sysctl('hw.' + kind + 'cpu')) for kind in ('physical', 'logical'))
    if not levels or any(v <= 0 for level in levels for k, v in level.items() if k != 'name'):
        raise RuntimeError('compute capability: missing CPU performance levels')
    if sum(level['physical_cores'] for level in levels) != physical or sum(level['logical_cores'] for level in levels) != logical:
        raise RuntimeError('compute capability: CPU performance levels do not cover the reported cores')
    devices = plistlib.loads(_read('ioreg', '-a', '-r', '-c', 'AGXAccelerator', '-d', '1'))
    counts = {device['gpu-core-count'] for device in devices if 'gpu-core-count' in device}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise RuntimeError(f'compute capability: expected one positive gpu-core-count, got {sorted(counts)}')
    return {'model': _sysctl('hw.model'), 'chip': _sysctl('machdep.cpu.brand_string'),
            'cpu': {'physical_cores': physical, 'logical_cores': logical, 'levels': levels},
            'gpu': {'cores': counts.pop()}}


if __name__ == '__main__':
    print(json.dumps(hardware(), separators=(',', ':')))
