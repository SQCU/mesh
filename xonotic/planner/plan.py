"""A whole-function planner: indexed products, collective writes and state updates."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rdma'))
import assignment
import torch
import torch.distributed._functional_collectives as collective
from torch_mesh.declarations import Function
from torch_mesh import _stream
from mesh_statistics import now, measure
sys.path.insert(0, str(ROOT / 'xonotic'))
from prepared import World, pages
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
# preparation operand: each declared function binds the shares supplied by its configuration.
def model(width):
    rng = np.random.default_rng(SEED)
    return tuple(rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[-2])
                 for shape in ((width, EXPERTS), (EXPERTS, width, FF), (EXPERTS, FF, width), (width, TEAMS)))


def tick_function(group, rank, shares):
    low, width = sum(shares[:rank]), shares[rank]

    def tick(positions, route, up, down, objective, previous):
        selected = torch.argmax(positions @ route, dim=1)
        # Indexed operands stay page views of the original weights. No expert
        # loop, host-selected row list, or per-row weight copy is a program step.
        u = up[:, :, low:low + width].index_select(0, selected)
        d = down[:, low:low + width].index_select(0, selected)
        partial = torch.bmm(torch.relu(torch.bmm(positions[:, None, :], u)), d)[:, 0, :]
        reduced = collective.wait_tensor(collective.all_reduce(
            torch.cat((partial.reshape(-1), previous)), 'sum', group.group_name))
        picks = torch.argmax(reduced[:positions.numel()].reshape(positions.shape) @ objective, dim=1)
        update = positions[:, 1:] + .35 * objective[1:].T.index_select(0, picks)
        positions[:, 1:].copy_(update * .98)
        return picks, reduced[positions.numel():]
    return tick


def main():
    split = json.loads(SPLIT.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument('ticks', type=int)
    parser.add_argument('bots', type=int)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--compiler', type=Path, help='explicit compiler configuration JSON')
    parser.add_argument('--columns', default=','.join(map(str, split['columns'])),
                        help="config_t0: each rank's expert columns, in rank order (default: split.json)")
    args = parser.parse_args()
    columns = [int(value) for value in args.columns.split(',')]
    if sum(columns) != FF:
        parser.error(f'--columns must sum to {FF}')
    # The existing numerical reduction carries the prior completed tick's
    # elapsed time, identical evidence for every replicated controller. The
    # current tick retains its exact proposal until all its computations end.
    world = World(args.device)
    rank, members = world.rank, world.members
    configuration = json.loads(args.compiler.read_text()) if args.compiler else None
    work = {'columns': {'parts': columns, 'grain': split['grain'], 'low': split['low']}}
    graphs = {}
    workload = {'bots': args.bots, 'width': args.width, 'experts': EXPERTS, 'columns': FF, 'teams': TEAMS}
    family = {'source': 'xonotic-planner', 'members': members, 'workload': workload}
    implementation = revision(ROOT)
    observations = []
    previous = None
    shares = columns
    constants = tuple(pages(value, device=args.device) for value in model(args.width))
    positions = np.random.default_rng(7).standard_normal((args.bots, args.width)).astype(np.float32) * .1
    positions[:, 0] = np.arange(args.bots)
    positions = pages(positions, device=args.device)
    prior = pages(np.zeros(len(columns), np.float32), device=args.device)
    from torch_mesh.engines import host
    prior_host = host(prior).numpy()
    objectives = np.full(args.bots, -1)
    prepared = {}
    switches = 0
    print(json.dumps({'event': 'planner_started', 'rank': rank, 'bots': args.bots, 'width': args.width,
                      'experts': EXPERTS, 'columns': columns[rank]}), flush=True)
    measured_seconds = 0.0
    for tick in range(args.ticks):
        decisions = assignment.domains(work, graphs)
        proposal = assignment.assign(decisions, observations, step=tick, family=family) if rank == 0 else None
        proposal = world.configuration(proposal)
        shares = proposal['parts']['columns']
        routing = assignment.route_operands(proposal['parts'], 'world', graphs['world']) if graphs else None
        key = tuple(shares)
        if key not in prepared:
            function = Function(tick_function(world.group, rank, shares), positions, *constants, prior,
                                configuration=configuration, source_scope={'function': 'xonotic-planner'})
            destinations = (pages(np.zeros(args.bots, np.int64), device=args.device),
                            pages(np.zeros(len(columns), np.float32), device=args.device))
            calls = function.declare(world.group, destinations=destinations)
            prepared.clear()
            prepared[key] = function, calls, tuple(host(value).numpy() for value in destinations)
        function, calls, (picks, times) = prepared[key]
        world.group.plan(calls, routing=routing)
        graphs = {'world': assignment.graph(world.native, world.group._plan)}
        work['columns'] = {**work['columns'], 'parts': list(shares)}
        realized = assignment.domains(work, graphs)
        proposal.update(parts={key: value['parts'] for key, value in realized.items()},
                        domains=assignment.choice_domains(realized), graphs=graphs, workload=workload,
                        implementation=implementation, episode=f'planner:{tick}')
        function.prepare()
        prior_host[...] = 0
        prior_host[rank] = previous['seconds'] if previous is not None else 0.0
        function_started = now()
        function.run()
        completion = function.completion()
        # The next tick consumes these results. This application boundary
        # observes one whole function; it inserts no numerical dispatch/drain.
        ended = _stream.completion_end(completion)
        feedback = (previous['proposal'], float(max(times[:len(columns)]))) if previous is not None else None
        switches += int(np.count_nonzero(objectives != picks))
        objectives[...] = picks
        previous = {'proposal': proposal, **measure(function_started, ended,
                    args.bots * (args.width + 1) * np.dtype(np.float32).itemsize, args.bots)}
        measured_seconds += previous['seconds']
        if feedback is not None:
            observations.append({**{key: feedback[0][key] for key in
                ('parts', 'domains', 'graphs', 'family', 'workload', 'implementation', 'proposal', 'episode')},
                'scope': 'function', 'complete': True, 'seconds': feedback[1],
                'timing_scope': 'maximum rank whole-tick latency carried by the existing reduction'})
            observations[:] = assignment.window(observations)
    elapsed = measured_seconds
    world.free()
    print(json.dumps({'event': 'planner_finished', 'rank': rank, 'ticks': args.ticks,
                      'bot_plans': args.ticks * args.bots, 'seconds': elapsed,
                      'bot_plans_per_second': args.ticks * args.bots / elapsed, 'tick_ms': 1e3 * elapsed / args.ticks,
                      'useful_gigabytes_per_second': args.ticks * args.bots * (args.width + 1) * 4 / elapsed / 1e9,
                      'switches': switches,
                      'objective_split': np.bincount(objectives, minlength=TEAMS).tolist(),
                      'shares': shares}), flush=True)


if __name__ == '__main__':
    main()
