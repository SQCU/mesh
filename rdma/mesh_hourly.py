"""The hourly (launchd's io.mesh.hourly, on each node at :28): main's normal programs, run on this node's collective as
it is found, whether or not they work. Each result and each fault is a JSON line in REPORT, written by each node that
survives the run.

1. The collective, read without configuration by mesh_preflight.py's discovery (its probe, beacons and cables) and the
   graph module: a node is in one when any of its RDMA ports is PORT_ACTIVE (mesh_preflight.joined), and its
   collective is every such node it reaches over cables (mesh_graph.reach). A node in no collective writes one line
   saying so and does nothing else: no checkout is touched, nothing is built or run.
2. The ranks, by one rule each member computes alike from what it found: rank 0 is the member first by host name, the
   others follow in the order a breadth-first walk over the cables from rank 0 reaches them (mesh_graph.walk), each
   node's neighbours taken by host name.
3. ~/mesh and ~/metal-microbench fetched and reset to origin/main, except on a node whose ~/mesh is a link to a
   checkout elsewhere: that is the development layout (~/mesh -> ~/dox/mesh), where a working checkout is not moved
   under its author. The commits each line ran are in it.
4. What main's programs need, built: make -C ~/mesh/rdma, then make -C ~/metal-microbench with the targets the E2B
   pair's programs use (BUILD).
5. main's programs, as written: for each placement metal-microbench holds (tools/mesh/placements/*.json),
   tools/mesh/grid.py --placement (the decode prepared on the pair) and tools/mesh/run_pair.py --splits (the E2B decode
   on the pair). Each is a driver: rank 0 runs it, with rank 1 as its peer (--nodes, MESH_PEER), and reaches rank 1
   by ssh. It uses two of the live members; it runs as written, and its line records the outcome. Every other member
   writes its own line for each, naming the rank whose line holds the outcome, then ends: an hourly running on a node
   is a collision to the programs' preflight (mesh_preflight.py), except the one the launch itself runs in.

Deadlines are clock instants every member computes alike: the minute the run began (launchd starts every member at
:28), PREPARE_S after it for the checkout and the build, then each program's slot. Rank 0 starts its first program at
the end of the preparation, when every member's build has ended. Every external command runs in a process group of its
own and is ended at its deadline: SIGTERM to the group, GRACE_S for it to exit, then SIGKILL to what is left of it
except a bridge (mesh-flow, mesh-heartbeat). A bridge is never SIGKILLed: so ended, it cannot withdraw its RDMA
registrations and leaves the provider wedged (metal-microbench tools/mesh/run_pair.py bridges_down). A bridge still
running is named in the line ("bridge pid N did not exit") and left running.

REPORT holds one line a run (`stage`: run: this node, its collective in rank order, its rank, the members absent and
why, the checkout and the build) and one line per workload per run (both commits, the workload, its program, its ranks
and this node's rank, its exit status, how it ended, its seconds, the JSON lines it printed and the result record it
wrote, and on a failure the last 4,000 characters of its error output). A program's whole output is in
~/.cache/mesh/hourly/<workload>.out and .err, its last run's. A stage that fails is recorded and the run goes on."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_disk  # noqa: E402  the one capacity guard of every write (mesh-disk.h)
import mesh_graph  # noqa: E402  the graph module (C9)
import mesh_preflight  # noqa: E402  the discovery: probe, beacons, cables

HOME = Path.home()
REPOS = ('mesh', 'metal-microbench')
REPORT, OUTPUT = HOME / '.cache/mesh/hourly.jsonl', HOME / '.cache/mesh/hourly'
PREPARE_S, RUN_S, GIT_S, GRACE_S, TAIL = 120, 3300, 30, 10, 4000
BUILD = (('mesh', ['make', '-C', str(HOME / 'mesh/rdma')]),
         ('metal-microbench', ['make', '-C', str(HOME / 'metal-microbench'), '.build/e2b_replay',
                               '.build/litert-schema/tflite/Model.py',
                               '.build/litert-reference/gemma-4-E2B-it.litertlm']))
began = time.time()
epoch = began - began % 60
run = {'schema': 2, 'run': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))}
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # a stop of this job stops the running command's group too


def record(**row):
    line = json.dumps({**run, **row}, separators=(',', ':'), default=str)
    print(line, flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    mesh_disk.write(REPORT, line + '\n', 'the hourly report', mode='a')


def failed():
    return traceback.format_exc()[-TAIL:]


def heads():
    out = {}
    for repo in REPOS:
        try:
            out[repo] = subprocess.run(['git', '-C', str(HOME / repo), 'rev-parse', 'HEAD'], capture_output=True,
                                       text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            out[repo] = None
    return out


def group(pgid):
    """{pid: executable} of the processes in process group `pgid`."""
    found = {}
    for line in mesh_preflight.run('ps', '-axo', 'pid=,pgid=,comm=').splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1] == str(pgid):
            found[int(parts[0])] = parts[2].strip()
    return found


def stop(program):
    """What is left of `program`'s process group ended: SIGTERM, GRACE_S for it to exit, then SIGKILL to each process
    left except a bridge. The bridges left running, by pid."""
    pgid = program.pid
    if program.poll() is None or group(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + GRACE_S
    while time.time() < end and (program.poll() is None or group(pgid)):
        time.sleep(0.2)
    left = group(pgid)
    bridges = {p: c for p, c in left.items() if os.path.basename(c) in mesh_preflight.BRIDGES}
    for p in left:
        if p not in bridges:
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
    try:
        program.wait(timeout=GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    return ['bridge pid %d did not exit (%s)' % (p, c) for p, c in sorted(bridges.items())]


def launch(name, argv, deadline, cwd=None, env=None):
    """`argv` in a process group of its own, its output to OUTPUT/<name>.out and .err, ended at `deadline` (a clock
    instant): its exit, how it ended, its seconds, the bridges it left, its output and, on a failure, its error's
    tail."""
    started, program, ended, left = time.time(), None, 'exited', []
    out, err = OUTPUT / (name + '.out'), OUTPUT / (name + '.err')
    try:
        with mesh_disk.sink(out) as o, mesh_disk.sink(err) as e:
            program = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=o, stderr=e,
                                       preexec_fn=os.setpgrp)
            try:
                program.wait(timeout=max(deadline - time.time(), 0.1))
            except subprocess.TimeoutExpired:
                ended = 'at its deadline'
            finally:
                left = stop(program)
    except Exception:
        return {'exit': None, 'ended': 'not started' if program is None else 'runner fault',
                'started': round(started, 3), 'seconds': round(time.time() - started, 2), 'stdout': '',
                'error': failed()}
    texts = []
    for path in (out, err):
        try:
            texts.append(path.read_text(errors='replace'))
        except OSError:
            texts.append('')
    result = {'exit': program.returncode, 'ended': ended, 'started': round(started, 3),
              'seconds': round(time.time() - started, 2), 'stdout': texts[0]}
    if left:
        result['bridges'] = left
    if program.returncode or ended != 'exited':
        result['error'] = (texts[1].strip() or texts[0].strip())[-TAIL:]
    return result


def checkout(deadline):
    """~/mesh and ~/metal-microbench fetched and reset to origin/main, each step's outcome; on a node whose ~/mesh is a
    link (the development layout), neither moved."""
    link = HOME / 'mesh'
    if link.is_symlink():
        return {'moved': False,
                'why': '~/mesh is a link to %s: a development checkout, not moved' % os.path.realpath(link)}
    done = {}
    for repo in REPOS:
        steps = []
        for verb in (['fetch', 'origin', 'main'], ['reset', '--hard', 'origin/main']):
            got = launch('git-%s-%s' % (repo, verb[0]), ['git', '-C', str(HOME / repo), *verb],
                         min(time.time() + GIT_S, deadline))
            got.pop('stdout')
            steps.append({'git': ' '.join(verb), **got})
        done[repo] = steps
    return done


def build(deadline):
    done = []
    for repo, argv in BUILD:
        got = launch('make-' + repo, argv, deadline)
        got.pop('stdout')
        done.append({'make': ' '.join(argv[1:]), **got})
    return done


def workloads(pair, via):
    """main's programs, as written, for each placement metal-microbench holds: the decode prepared on the pair, then
    the E2B decode on the pair. Each is driven from rank 0, its peer rank 1."""
    peers = [via[n] for n in pair[1:]]
    env = {**os.environ, **({'MESH_PEER': peers[0]} if peers else {})}
    for placement in sorted((HOME / 'metal-microbench/tools/mesh/placements').glob('*.json')):
        yield {'workload': 'grid-' + placement.stem, 'slot': 300, 'env': env,
               'argv': [sys.executable, 'tools/mesh/grid.py', '--placement', 'tools/mesh/placements/' + placement.name,
                        '--nodes', ','.join(['local'] + peers)]}
        yield {'workload': 'run_pair-' + placement.stem, 'slot': 600, 'env': env,
               'argv': [sys.executable, 'tools/mesh/run_pair.py', '--splits', placement.stem],
               'result': 'output_data/mesh-pair-run/result.json'}


def measured(text):
    lines = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            lines.append(value)
    return lines


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    here = mesh_preflight.probe()
    run.update(node=here['host'], uuid=here['uuid'], commits=heads())
    if not mesh_preflight.joined(here):
        record(stage='run', collective=[], ports=here['ports'],
               why='no RDMA port of this node is PORT_ACTIVE (ibv_devinfo): it is in no collective, and runs nothing')
        return 0
    stage, order, via = {'stage': 'run'}, [here['host']], {here['host']: 'local'}
    try:
        facts, _, via, silent, cables = mesh_preflight.discover(here, (), os.environ.get('MESH_SSH_IDENTITY'))

        def live(n):
            return sorted({o for _, o in cables(n) if o and mesh_preflight.joined(facts[o])})

        members = mesh_graph.reach([here['host']], live)
        order = [n for n in mesh_graph.walk([min(members)], live) if n in members]
        order += sorted(members.difference(order))
        absent = {name: 'its beacon answered; its probe at %s over ssh did not, within %s s'
                  % (v, mesh_preflight.READ_S) for v, name in silent.items()}
        for n in order:
            for d, o in cables(n):
                if o is None:
                    absent['domain ' + d] = 'a cable from %s to a node not probed' % n
        for n in sorted(set(facts) - members):
            absent[n] = ('probed; no RDMA port PORT_ACTIVE' if not mesh_preflight.joined(facts[n])
                         else 'probed; not reached over cables from this node')
        stage['absent'] = absent
    except Exception:
        stage['error'] = failed()
    rank = order.index(here['host'])
    stage.update(collective=order, rank=rank)
    prepared = epoch + PREPARE_S
    try:
        stage['checkout'] = checkout(prepared)
    except Exception:
        stage['checkout'] = {'error': failed()}
    run['commits'] = heads()
    try:
        stage['build'] = build(prepared)
    except Exception:
        stage['build'] = {'error': failed()}
    pair = order[:2]
    record(**stage, ranks=pair, seconds=round(time.time() - began, 2))
    try:
        items = list(workloads(pair, via))
    except Exception:
        items = []
        record(stage='workloads', error=failed())
    if rank != 0:
        part = ('rank %d, driven over ssh by rank 0 (%s), whose line holds the outcome' % (rank, order[0])
                if here['host'] in pair else
                'not one of its ranks: it runs on a pair, %s; this node is rank %d of %d' % (', '.join(pair), rank,
                                                                                            len(order)))
        for item in items:
            record(workload=item['workload'], program=item['argv'][1:], ranks=pair, rank=rank, exit=None, ended=part,
                   seconds=0, measured=[])
        return 0
    while time.time() < prepared:
        time.sleep(min(1.0, prepared - time.time()))
    deadline, end = prepared, epoch + RUN_S
    for item in items:
        deadline = min(deadline + item['slot'], end)
        row = {'workload': item['workload'], 'program': item['argv'][1:], 'ranks': pair, 'rank': rank}
        if time.time() >= deadline:
            record(**row, exit=None, ended="not started: the run's end (%ds after its start) had passed" % RUN_S,
                   seconds=0, measured=[])
            continue
        try:
            got = launch(item['workload'], item['argv'], deadline, cwd=str(HOME / 'metal-microbench'), env=item['env'])
            found = measured(got.pop('stdout'))
            if item.get('result'):
                path = HOME / 'metal-microbench' / item['result']
                try:
                    if path.stat().st_mtime >= got['started']:
                        found.append(json.loads(path.read_text()))
                except (OSError, ValueError):
                    pass
            record(**row, **got, measured=found)
        except Exception:
            record(**row, exit=None, ended='runner fault', seconds=0, measured=[], error=failed())
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        record(stage='runner', error=failed())
        sys.exit(1)
