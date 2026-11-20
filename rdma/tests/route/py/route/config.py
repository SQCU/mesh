"""route.toml, with --set overrides; paths resolved against the file's directory, {rank} filled"""
import copy
import os
import tomllib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {('run', k) for k in ('fabric', 'theta_in', 'theta', 'theta_int8', 'library', 'metallib')}


def load(path=None, sets=()):
    path = path or os.path.join(HERE, 'route.toml')
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    for s in sets:
        key, value = s.split('=', 1); section, name = key.split('.', 1)
        old = cfg[section][name]
        cfg[section][name] = type(old)(value) if not isinstance(old, (bool, list)) else (value.lower() in ('1', 'true', 'yes') if isinstance(old, bool) else value.split(','))
    base = os.path.dirname(os.path.abspath(path))
    for section, name in PATHS:
        v = cfg[section][name]
        if v:
            cfg[section][name] = os.path.normpath(os.path.join(base, os.path.expanduser(v)))
    cfg['run']['telemetry'] = cfg['run']['telemetry'].format(rank=cfg['run']['rank'])
    return cfg


def flat(cfg):
    """the distribution's view: its section, with the capacity it fills to"""
    d = copy.deepcopy(cfg['distribution']); d['refreshes'] = cfg['capacity']['refreshes']
    return d
