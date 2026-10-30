"""The desk's link use from its members' bytes landed (design/measurement-regressions-2026-10-30.md#link-use).

python3 mesh_landed.py <window_ms> <watch>... : each watch one node's `mesh-stat` line and then its `mesh-stat landed`
lines (mesh-stat.c landed_watch), taken beside a run and read afterwards, so nothing it measures runs differently. A
node's inbound capacity is its ports' stated bandwidth; its use in a window is the bytes its ports landed in it (each
pair the bridge's own clock of its bytes) over that capacity. The members' clocks are aligned at each one's first
landing, the run's first collective. Computer-pounds: each member's use times its weight (WEIGHT, by hardware model),
summed over the members.
"""
import json
import sys

# each model's weight in pounds, as Apple states it: the Mac Studio with an Ultra chip, the Mac mini with an M4 Pro
WEIGHT = {'Mac17,15': 8.0, 'Mac16,11': 1.6}


def watch(path):
    """(node, model, inbound capacity in bytes a second, [(ns, bytes landed on all its ports)])"""
    lines = open(path).read().splitlines()
    state = json.loads(lines[0])
    model = next(n['model'] for n in state['nodes'] if n['node'] == state['node'])
    capacity = sum(p['bandwidth'] for p in state['peers'] if p['phase'] == 'PAIRED') / 8
    points = []
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        points.append((row['ns'], sum(port[3] for port in row['ports'])))
    return state['node'], model, capacity, points


def use(points, width):
    """each window's bytes landed, by window from the first landing after the watch began (the first pair is the ports'
    state as the watch began, stamped with whatever landed last before it)"""
    first, windows = points[1][0], {}
    for (_, before), (at, after) in zip(points, points[1:]):
        windows[(at - first) // width] = windows.get((at - first) // width, 0) + after - before
    return windows


def main(argv):
    width = int(float(argv[0]) * 1e6)
    nodes = [watch(path) for path in argv[1:]]
    seconds = width / 1e9
    report, members = {'window_ms': width / 1e6, 'nodes': []}, []
    for node, model, capacity, points in nodes:
        if len(points) < 2:
            report['nodes'].append({'node': node, 'model': model, 'landed_bytes': 0})
            continue
        windows = use(points, width)
        peak = max(windows.values()) / seconds
        members.append((node, WEIGHT[model], capacity, windows))
        report['nodes'].append({'node': node, 'model': model, 'inbound_capacity_gbs': capacity / 1e9,
                                'landed_bytes': points[-1][1] - points[0][1], 'peak_gbs': peak / 1e9,
                                'peak_use': peak / capacity})
    end = max(max(w) for _, _, _, w in members) + 1
    desk = [sum(pounds * w.get(k, 0) / seconds / capacity for _, pounds, capacity, w in members) for k in range(end)]
    report['members_pounds'] = sum(pounds for _, pounds, _, _ in members)
    report['peak_computer_pounds'] = max(desk)
    report['windows_over'] = {f'{level}': {'windows': len(over), 'mean_computer_pounds': sum(over) / len(over)}
                              for level in (4, 8, 12, 16) for over in [[v for v in desk if v > level]] if over}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
