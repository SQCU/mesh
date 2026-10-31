"""Planner source with canonical collective preparation.

Its imperative NumPy producers and consumers still need whole-function
lowering; the removed blocking host collective executor is not restored.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rdma'))
import assignment
from assignment import declare
from mesh_mpi import Comm, empty
import mesh
from mesh_source import revision

TEAMS = 5
EXPERTS = 8
FF = 2048
SEED = 20260828
SPLIT = Path(__file__).with_name('split.json')


# Megatron-LM tensor parallelism (Shoeybi et al. 2019) over the planner's routed experts: rank r
# applies its block of every expert's up-projection columns and the same rows of its down
# projection, so its expert output is a partial sum that one all-reduce completes.  Router and
# objective basis are small and replicated.  Every rank holds every column, so the blocks are a
# run-time operand: each tick slices them by the shares the balancer gave on the tick before.
def model(width):
    rng = np.random.default_rng(SEED)
    return tuple(rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[-2])
                 for shape in ((width, EXPERTS), (EXPERTS, width, FF), (EXPERTS, FF, width), (width, TEAMS)))


# Route each bot to an expert and apply this rank's share of it, grouping rows per expert rather
# than gathering a weight matrix per row.
def solve(positions, route, up, down, out):
    selected = np.argmax(positions @ route, axis=1)
    out[:] = 0
    for expert in range(EXPERTS):
        rows = np.nonzero(selected == expert)[0]
        if rows.size:
            out[rows] = np.maximum(positions[rows] @ up[expert], 0) @ down[expert]


def main():
    # The source declaration below uses the canonical plan. Numerical NumPy
    # producers/consumers still require whole-function lowering; mesh_mpi does
    # not restore an imperative host executor for this legacy application.
    split = json.loads(SPLIT.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument('ticks', type=int)
    parser.add_argument('bots', type=int)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--columns', default=','.join(map(str, split['columns'])),
                        help="config_t0: each rank's expert columns, in rank order (default: split.json)")
    args = parser.parse_args()
    columns = [int(value) for value in args.columns.split(',')]
    if sum(columns) != FF:
        parser.error(f'--columns must sum to {FF}')
    if args.width < len(columns):
        parser.error('--width must hold one segment time per rank')
    # The existing numerical reduction carries the prior completed tick's
    # elapsed time, identical evidence for every replicated controller. The
    # current tick retains its exact proposal until all its computations end.
    comm = Comm()
    rank = comm.Get_rank()
    members = mesh.configuration_members(comm.comm)
    local = empty((args.bots + 1, args.width), np.float32)
    local[...] = 0
    total = empty(local.shape, local.dtype)
    source = {'function': 'xonotic-planner', 'statement': 'expert-output-reduction'}
    calls = [{'what': mesh.lib.MESH_ALLREDUCE, 'count': local.size, 'type': mesh.lib.ncclFloat32,
              'op': mesh.lib.ncclSum, 'bindings': (local, total), 'route_scope': source,
              'sources': [{'source': source, 'occurrences': {'shape': [1], 'steps': []},
                           'work': {'shape': [local.nbytes], 'steps': []}}]}]
    work = {'columns': {'parts': columns, 'grain': split['grain'], 'low': split['low']}}
    graphs = {}
    workload = {'bots': args.bots, 'width': args.width, 'experts': EXPERTS, 'columns': FF, 'teams': TEAMS}
    family = {'source': 'xonotic-planner', 'members': members, 'workload': workload}
    implementation = revision(ROOT)
    observations = []
    previous = None
    shares = columns
    route, up, down, objective = model(args.width)
    positions = np.random.default_rng(7).standard_normal((args.bots, args.width)).astype(np.float32) * .1
    positions[:, 0] = np.arange(args.bots)
    objectives = np.full(args.bots, -1)
    switches = collective = solving = 0
    print(json.dumps({'event': 'planner_started', 'rank': rank, 'bots': args.bots, 'width': args.width,
                      'experts': EXPERTS, 'columns': columns[rank]}), flush=True)
    start = ready = time.monotonic()
    for tick in range(args.ticks):
        decisions = assignment.domains(work, graphs)
        proposal = assignment.assign(decisions, observations, step=tick, family=family) if rank == 0 else None
        proposal = mesh.broadcast_configuration(comm.comm, proposal)
        shares = proposal['parts']['columns']
        routing = assignment.route_operands(proposal['parts'], 'world', graphs['world']) if graphs else None
        plan = declare(comm.comm, calls, routing=routing)
        graphs = {'world': assignment.graph(comm.comm, plan)}
        work['columns'] = {**work['columns'], 'parts': list(shares)}
        realized = assignment.domains(work, graphs)
        proposal.update(parts={key: value['parts'] for key, value in realized.items()},
                        domains=assignment.choice_domains(realized), graphs=graphs, workload=workload,
                        implementation=implementation, episode=f'planner:{tick}')
        function_started = began = time.monotonic()
        low = sum(shares[:rank])
        share = slice(low, low + shares[rank])
        solve(positions, route, up[:, :, share], down[:, share], local[:args.bots])
        solving += time.monotonic() - began
        # Previous completed tick latency rides the existing numerical reduction.
        # It is consumed only under its immutable proposal; this tick's shares stay fixed.
        local[args.bots] = 0
        local[args.bots, rank] = previous['seconds'] if previous is not None else 0.0
        began = time.monotonic()
        comm.Allreduce(local, total)
        ready = time.monotonic()
        collective += ready - began
        feedback = (previous['proposal'], float(max(total[args.bots, :len(columns)]))) if previous is not None else None
        experts = total[:args.bots]
        picks = np.argmax(experts @ objective, axis=1)
        switches += int(np.count_nonzero(objectives != picks))
        objectives = picks
        positions[:, 1:] += .35 * objective[1:, picks].T
        positions[:, 1:] *= .98
        previous = {'proposal': proposal, 'seconds': time.monotonic() - function_started}
        if feedback is not None:
            observations.append({**{key: feedback[0][key] for key in
                ('parts', 'domains', 'graphs', 'family', 'workload', 'implementation', 'proposal', 'episode')},
                'scope': 'function', 'complete': True, 'seconds': feedback[1],
                'timing_scope': 'maximum rank whole-tick latency carried by the existing reduction'})
            observations[:] = assignment.window(observations)
    elapsed = time.monotonic() - start
    comm.Free()
    print(json.dumps({'event': 'planner_finished', 'rank': rank, 'ticks': args.ticks,
                      'bot_plans': args.ticks * args.bots, 'seconds': elapsed,
                      'bot_plans_per_second': args.ticks * args.bots / elapsed, 'tick_ms': 1e3 * elapsed / args.ticks,
                      'solve_ms': 1e3 * solving / args.ticks, 'collective_ms': 1e3 * collective / args.ticks,
                      'switches': switches,
                      'objective_split': np.bincount(objectives, minlength=TEAMS).tolist(),
                      'shares': shares}), flush=True)


if __name__ == '__main__':
    main()
