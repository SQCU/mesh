"""Hardware inputs read once during preparation, never during numerical calls."""
from functools import cache
import ctypes
import json
import plistlib
import subprocess
from pathlib import Path
import re


@cache
def implementations():
    """Declared compiler capabilities; loading this catalog is preparation."""
    path = Path(__file__).resolve().parents[1] / 'etc/compute/implementations.json'
    return json.loads(path.read_text())


@cache
def implementation_assignment():
    """Configured initial choices; later preparations consume supplied choices."""
    path = Path(__file__).resolve().parents[1] / 'etc/compute/assignment.json'
    return json.loads(path.read_text())


def eligibility(required, implementation):
    """Compare explicit semantic sets, retaining the reasons with the candidate.

    Frontends supply requirements from their existing typed declarations.
    Implementations state admitted values and required properties in the same
    vocabulary. Layout relations remain operands of the shared placement solve;
    they are not approximated by another eligibility predicate here.
    """
    missing = []
    for field, admitted in implementation['accepts'].items():
        values = [value for value in required[field] if value not in admitted]
        if values:
            missing.append(dict(field=field, relation='accepts', values=values))
    for field, necessary in implementation['requires'].items():
        values = [value for value in necessary if value not in required[field]]
        if values:
            missing.append(dict(field=field, relation='requires', values=values))
    return dict(implementation=implementation, required=required, missing=missing)


def select_implementation(required, catalog, selected=None):
    """Resolve a declared choice, or the sole legal implementation.

    Several legal implementations require an assignment operand. Neither the
    catalog's ordering nor a source tensor's device chooses between them.
    """
    declarations = [eligibility(required, candidate) for candidate in catalog]
    choices = [item for item in declarations if not item['missing'] and
               (selected is None or item['implementation']['id'] == selected)]
    if len(choices) != 1:
        raise ValueError(f'work requires one legal implementation assignment; selected={selected}, declarations={declarations}')
    return choices[0]['implementation']


def work_domain(required, candidates, capacities, *, units, grain, identity):
    """One ordinary assignment domain over the declared legal implementations.

    Work rates are supplied or measured. Hardware core counts are not rates.
    A candidate's inability to implement these requirements is a zero upper
    bound in this solve, not a runtime substitution or an error in another
    candidate's otherwise legal program.
    """
    import math
    from assignment import project
    if len(candidates) != len(capacities) or not candidates:
        raise ValueError('work assignment needs a rate for each declared implementation')
    if any(not math.isfinite(rate) or rate < 0 for rate in capacities):
        raise ValueError('implementation work rates must be finite and nonnegative')
    declarations = [eligibility(required, candidate) for candidate in candidates]
    rates = [rate if not declaration['missing'] else 0.0
             for rate, declaration in zip(capacities, declarations, strict=True)]
    total = sum(rates)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f'no declared implementation has a positive work rate for {required}: {declarations}')
    high = [units if rate else 0 for rate in rates]
    parts = project([units * (rate / total) for rate in rates], units, grain, [0] * len(rates), high)
    return dict(parts=parts, grain=grain, low=[0] * len(rates), high=high,
                units=units, capacity=rates, implementations=declarations,
                identity=dict(work=identity, implementations=[candidate['id'] for candidate in candidates]))


@cache
def compilation():
    """The native build manifest, for offline graph importers without Metal."""
    path = Path(__file__).resolve().parents[1] / 'etc/compute/metal.def'
    fields = {}
    for line in path.read_text().splitlines():
        line = line.split('//', 1)[0].strip()
        if not line:
            continue
        field = re.fullmatch(r'MESH_METAL_VALUE\(([A-Z_]+),\s*([0-9]+)\)', line)
        if field is None:
            raise ValueError(f'{path}: invalid compilation operand {line!r}')
        fields[field[1]] = int(field[2])
    return fields


def _read(*command):
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'compute capability {" ".join(command)}: {result.stderr.decode().strip()}')
    return result.stdout


_LIBC = ctypes.CDLL(None, use_errno=True)


def _sysctl(name, text=False):
    """`name`'s value as sysctl(8) prints it, read in this process (sysctlbyname): a forked sysctl would copy this
    process's page tables, a rank's region among them, to print one number."""
    size = ctypes.c_size_t(0)
    if _LIBC.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) or not size.value:
        raise RuntimeError(f'compute capability {name}: no value')
    value = ctypes.create_string_buffer(size.value)
    if _LIBC.sysctlbyname(name.encode(), value, ctypes.byref(size), None, 0):
        raise RuntimeError(f'compute capability {name}: {ctypes.get_errno()}')
    raw = value.raw[:size.value]
    result = raw.rstrip(b'\0').decode().strip() if text else str(int.from_bytes(raw, 'little', signed=True))
    if not result:
        raise RuntimeError(f'compute capability {name}: no value')
    return result


@cache
def hardware():
    """Keep CPU classes and cache clusters separate from the GPU core count.

    Counts describe hardware, not operation rates or CPU affinity promises.
    Missing facts are preparation errors; there is no substitute platform.
    """
    levels = []
    for index in range(int(_sysctl('hw.nperflevels'))):
        prefix = f'hw.perflevel{index}'
        levels.append({'name': _sysctl(prefix + '.name', True), **{
            field: int(_sysctl(prefix + '.' + key)) for field, key in (
                ('physical_cores', 'physicalcpu'), ('logical_cores', 'logicalcpu'),
                ('cores_per_l2', 'cpusperl2'), ('l2_bytes', 'l2cachesize'))}})
    physical, logical = (int(_sysctl('hw.' + kind + 'cpu')) for kind in ('physical', 'logical'))
    if not levels or any(v <= 0 for level in levels for k, v in level.items() if k != 'name'):
        raise RuntimeError('compute capability: missing CPU performance levels')
    if sum(level['physical_cores'] for level in levels) != physical or sum(level['logical_cores'] for level in levels) != logical:
        raise RuntimeError('compute capability: CPU performance levels do not cover the reported cores')
    import mesh
    cores = mesh.lib.mesh_gpu_cores()   # libmesh's one reading of the GPU's cores (mesh-fabric.c)
    if not cores:
        raise RuntimeError('compute capability: the GPU states no gpu-core-count (mesh_gpu_cores)')
    system = plistlib.loads(Path('/System/Library/CoreServices/SystemVersion.plist').read_bytes())   # what sw_vers prints
    return {'model': _sysctl('hw.model', True), 'chip': _sysctl('machdep.cpu.brand_string', True),
            'os': {'version': system['ProductVersion'], 'build': system['ProductBuildVersion']},
            'cpu': {'physical_cores': physical, 'logical_cores': logical, 'levels': levels},
            'gpu': {'cores': cores}}


if __name__ == '__main__':
    print(json.dumps(hardware(), separators=(',', ':')))
