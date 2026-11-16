# design/unequal-shards-change-request.md items 6, 7, 11, 13 and 14
import collections
import pathlib
import sys

from .ir import RECV, SEND, channel_of

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import allocate
import mesh_graph


# compiler/mesh_compiler/schedule.py schedule, dependence and assemble: function `key`'s one source relation (its
# SourceRelation, every machine module's meta["relation"]) as rdma/mesh_graph.py Graph.from_conflicts, the parent of
# its projection: with observed `units` (each one measured unit's identities), rdma/mesh-graph.h
# mesh_graph_relation_contract (Graph.contract) onto the vertices of their source tasks and the two ends of each actual
# external transfer (source.supplied's channel pair of a channel any machine module's meta["transfers"] retains; a pair
# assemble erased is no boundary), every other vertex (a view, an erased pair, a value the typed construction
# discharged) no event of it; without (`units` None: supplied costs) the relation itself. Its vertices then go through
# Graph.quotient onto classes (rdma/mesh_graph.py components) joining each unit's members and the projected shards of
# one partitioned node (its original's `of`); S. Zhang et al., HAP, EuroSys 2024, sections 3.2 and 5.2;
# rdma/mesh-graph.h mesh_graph_longest over the prerequisite (incoming) rows of the quotient's forward edges in its
# topological order, an edge entering a class with an external receive weighing 1, so a receive and its numerical
# followers share a level. `level` by projected source task k and `node` by every source task (meta["tasks"] of the copy
# meta["vertices"] holds; the node its meta["decomposed"] original), `received` by level each external receive with its
# sent value's original, `members` each unit's identities in that source task domain (a generated operation's,
# lower_metal's generator advance, has none), `units` each unit's level (None without members), `inside` each cycle of
# classes (then no level: no topological order) and each external receive edge within one class, with the units of its
# classes, none of them costed, and `graphs` the owners (relation, projection, quotient, its order, its lengths View;
# None with a cycle)
Stages = collections.namedtuple("Stages", "level node received members units inside graphs")


def crossing(node, external):
    return getattr(node, "target", None) is RECV and channel_of(node) in external


def stages(scheduled, key, units=None):
    modules = tuple(scheduled.modules[key].values())
    source, external = modules[0].meta["relation"], {c for module in modules for c in module.meta["transfers"]}
    vertex, original = {}, {}
    for module in modules:
        tasks, decomposed = module.meta["tasks"], module.meta["decomposed"]
        for v, copy in enumerate(module.meta["vertices"]):
            if copy in tasks:
                vertex[tasks[copy][0]] = v
                original[v] = decomposed.get(copy, copy)
    count, members = len(source.functions), [tuple(k for k in tasks if k in vertex) for tasks in units or ()]
    at_vertex = {node: v for v, node in enumerate(source.functions)}
    ends = [(sent, node) for kind, sent, node, channel in source.supplied if kind == "channel" and channel in external]
    kept = [units is None] * count
    for k in (k for tasks in members for k in tasks):
        kept[vertex[k]] = True
    for end in (end for pair in ends for end in pair):
        kept[at_vertex[end]] = True
    graph = mesh_graph.Graph.from_conflicts(source.functions, source.relation)
    projected = graph if units is None else graph.contract(kept)
    group = [original.get(v, node) for v, node in enumerate(source.functions)]
    labels = {c: count + j for j, c in enumerate(dict.fromkeys(
        g.meta["of"] for v, g in enumerate(group) if kept[v] and "of" in g.meta))}
    joined = [(v, labels[g.meta["of"]]) for v, g in enumerate(group) if kept[v] and "of" in g.meta] + [
        (vertex[k], count + len(labels) + u) for u, tasks in enumerate(members) for k in tasks]
    component = mesh_graph.components(count + len(labels) + len(members), joined + [(b, a) for a, b in joined])
    index = {c: j for j, c in enumerate(dict.fromkeys(component[:count]))}
    class_of = [index[c] for c in component[:count]]
    boundary = [False] * len(index)
    for v, node in enumerate(source.functions):
        boundary[class_of[v]] |= crossing(node, external)
    quotient = projected.quotient(list(index), class_of)
    edges = [(quotient.at[g], quotient.at[h]) for g, h in quotient.edges()]
    held, within = [class_of[vertex[tasks[0]]] if tasks else None for tasks in members], {}
    for j, c in enumerate(mesh_graph.components(len(index), edges)):
        within.setdefault(c, []).append(j)
    cycles = [classes for classes in within.values() if len(classes) > 1]
    inside = [{"cycle": [node.name for v, node in enumerate(source.functions) if class_of[v] in classes],
               "units": [u for u, j in enumerate(held) if j in classes], "costed": False} for classes in cycles]
    inside += [{"edge": (a.name, b.name), "units": [u for u, c in enumerate(held) if c == class_of[graph.at[b]]],
                "costed": False} for a, b in projected.edges()
               if crossing(b, external) and class_of[graph.at[a]] == class_of[graph.at[b]]]
    order = quotient.order()
    lengths = None if cycles else quotient.longest(order, [float(boundary[h]) for _, h in edges], True)
    at = [None if lengths is None else lengths[c] for c in class_of]
    received = {}
    for sent, node in ends:
        value = source.producers[sent.args[0]]
        received.setdefault(at[at_vertex[node]], []).append(
            (group[at_vertex[node]], group[at_vertex[value]] if value in at_vertex else value))
    return Stages({k: at[v] for k, v in vertex.items() if kept[v]}, {k: group[v] for k, v in vertex.items()}, received,
                  members, [None if c is None else at[vertex[tasks[0]]] for c, tasks in zip(held, members)], inside,
                  (graph, projected, quotient, order, lengths))


# design/unequal-shards-change-request.md item 7; A. Lastovetsky et al., arXiv:1109.3074
def kind(node):
    return node.meta.get("of", node.name)


def units(node):
    return node.meta.get("units", (1, 1))


def speed(times, x):
    return x / times[x]


# S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3): one cost of function `key`'s stage (its Stages level) on
# execution resource `resource`, which executes the tasks of logical processors `ranks` (indices in `processors`),
# fixed + sum coefficient B[part, rank] over the full share variables, `provenance` its measure
Cost = collections.namedtuple("Cost", "key stage resource ranks fixed coefficients provenance")


# A. Lastovetsky et al., arXiv:1109.3074, steps 1 and 2: each original node of a function's tasks once, of supplied speeds
# `times` {processor: {kind: {x: seconds}}} at its `units` (x, n) (compiler/mesh_compiler/placement.py: a shard's x
# its current local units, n the full units), speed(x) on its processor (its resource): a shard (an `of` node) of a
# part (`segment`) is its share variable's (part, rank its processor's index in `processors`) coefficient n / speed(x);
# every other node, a shard in no part among them, its fixed current work x / speed(x), exactly times[x]. A transfer's
# time is `solved`'s collective time, not a node's. Rejection filters are not part of even temporary integration
# validation: work without a share variable stays a fixed cost, never a node dropped, and a missing supplied time
# raises at its lookup
def supplied(staged, segment, processors, times):
    rank, made = {p: j for j, p in enumerate(processors)}, []
    for key, stage in staged.items():
        first = {}
        for k, node in stage.node.items():
            first.setdefault(node, k)
        for node, k in first.items():
            if node.target in (SEND, RECV):
                continue
            processor, (x, n), part = node.meta["processor"], units(node), segment(node)
            held = times[processor][kind(node)]
            if "of" in node.meta and part is not None:
                fixed, coefficients = 0.0, {(part, rank[processor]): n / speed(held, x)}
            else:
                fixed, coefficients = held[x], {}
            made.append(Cost(key, stage.level[k], processor, (rank[processor],), fixed, coefficients, None))
    return made


# the distinction of a metal_ir_work_association: its Unit loop, a represented map no loop indexes alone, or no source
# operand
def distinction(association):
    return ("no source operand" if association.operand == mesh_graph.NONE else
            "a represented map with no Unit loop" if association.loop == mesh_graph.NONE else "a Unit loop")


# One Work term `t` of `work` (compiler/mesh_compiler/native.py Work), `held` the associations of its operation
# (metal_ir_work_associations `operation`): its count at the current extents (each factor at its loops' last index,
# extent - 1) and as (constant, {variable: coefficient}) in the full share variables, or None and the exact (factor,
# atom, reason; None where it is the operation's) it has none: an association of the operation without a Unit loop (no
# source operand, or a represented map) leaves its share dependence unknown. A loop's variable: the one (part `segment`,
# rank) of its associations' tasks' original nodes at original full extent F; over its admitted extents x F = h F /
# `units` (h whole units), an atom c ((x F - 1) floordiv d) is c F / d x - c exactly where it has no modulus and d is 1
# or divides F / `units`
def counted(work, t, held, stage, segment, rank, units):
    for association in held:
        if association.loop == mesh_graph.NONE:
            return None, (None, None, distinction(association))
    current, constant, bound = 1, 1, None
    for f in range(t.first, t.first + t.factors):
        e = work.factors[f]
        value, b, a = e.constant, e.constant, {}
        for j in range(e.first, e.first + e.terms):
            atom = work.atoms[j]
            if atom.dim >= t.extents:
                return None, (f, j, "a symbol")
            extent = work.extents[t.domain + atom.dim]
            associations = [association for association in held if association.loop == atom.dim]
            nodes = [stage.node[axis.task] for axis in associations]
            if any(segment(node) is None for node in nodes):
                return None, (f, j, "an association without a part")
            variables = tuple(dict.fromkeys(((segment(node), rank[node.meta["processor"]]), axis.full)
                                            for axis, node in zip(associations, nodes)))
            divisor = max(atom.divisor, 1)
            point = (extent.extent - 1) // divisor
            point = point % atom.modulus if atom.modulus else point
            value += atom.coefficient * point
            if not variables:
                b += atom.coefficient * point
                continue
            if len(variables) > 1:
                return None, (f, j, "distinct share variables of one loop")
            ((variable, full),) = variables
            if atom.modulus or divisor > 1 and (full % units or (full // units) % divisor):
                return None, (f, j, "not affine over whole units")
            b -= atom.coefficient
            a[variable] = a.get(variable, 0) + atom.coefficient * full // divisor
        current *= value
        if not a:
            constant *= b
        elif bound is not None:
            return None, (f, None, "a product of share factors")
        else:
            bound = b, a
    if bound is None:
        return current, (constant, {})
    return current, (constant * bound[0], {v: constant * c for v, c in bound[1].items()})


# Work term i of `work` as its retained records give it: its key, kind and coefficient (operations by resource and
# width, unknown), its source operations (origins), each factor's expression and atoms, each domain extent with its
# source associations (task, full, value, dimension), and every association of its operation (`held`: original
# operation and operand, typed value and dimension, loop, `distinction`), each with its task's original source node in
# `stage` (processor, of, units)
def evidence(work, i, held, stage):
    t = work.terms[i]

    def node(task):
        found = stage.node.get(task)
        return None if found is None else {"name": found.name, "processor": found.meta.get("processor"),
                                           "of": found.meta.get("of"), "units": found.meta.get("units")}

    def associated(axis):
        return {"task": axis.task, "full": axis.full, "value": axis.value, "dimension": axis.dimension,
                "node": node(axis.task)}

    def none(x):
        return None if x == mesh_graph.NONE else x
    return {"term": i, "kernel": t.kernel, "member": t.member, "phase": t.phase, "operation": t.operation,
            "associations": [{"task": a.task, "full": a.full, "operation": a.operation, "source": a.source,
                              "operand": none(a.operand), "value": none(a.value), "dimension": a.dimension,
                              "loop": none(a.loop), "distinction": distinction(a), "node": node(a.task)} for a in held],
            "kind": t.kind, "coefficient": {"operations": [list(row) for row in t.coefficient.operations],
                                            "unknown": t.coefficient.unknown},
            "origins": list(work.origins[t.origin:t.origin + t.origins]),
            "factors": [{"factor": f, "constant": e.constant, "atoms": [
                {"atom": j, "dim": a.dim, "coefficient": a.coefficient, "divisor": a.divisor, "modulus": a.modulus}
                for j in range(e.first, e.first + e.terms) for a in [work.atoms[j]]]}
                for f in range(t.first, t.first + t.factors) for e in [work.factors[f]]],
            "extents": [{"extent": x.extent, "loop": x.loop, "axes": [associated(axis) for axis in
                                                                     work.axes[x.axis:x.axis + x.axes]]}
                        for x in work.extents[t.domain:t.domain + t.extents]]}


# compiler/mesh_compiler/native.py Bound.observe and Bound.work (metal_ir.h metal_ir_work_terms,
# metal_ir_work_associations by each term's operation, and metal_ir_dispatch_numerics); S. Zhang et al., HAP, EuroSys
# 2024, section 3.2; A. Lastovetsky et al.,
# arXiv:1109.3074, steps 1 and 2: the executed occurrence slices of one observed invocation joined where their member
# dispatches share a numeric scope (call, kernel, [first, last); a later twin's its first phase's) through
# rdma/mesh_graph.py components, each joined set one measured unit: its `samples` (each slice's residue, last sample
# seconds and executions, its site's issued occurrences) and `estimated_seconds` the sum of each slice's last sample
# times its executions (a sample-based estimate at this observation's issued site multiplicities, not a measured total;
# conditional ICB range selection, numerical internal branches, unknown primitive Work and future data-dependent counts
# are not in it), its identities projected onto the source task domain (`stages` members; the others `generated`, their
# work kept in the unit's), its level its members' class, its resource (Target.node, its native unit kind: a GPU span or
# a host call), its ranks its source tasks' logical processors, its scopes' `occurrences` each slice's issued executions
# per member dispatch of that scope that is a primary (metal_ir_numeric `later` 0: a later twin's readiness split is
# part of its primary's occurrence, its duration in the estimate, its Work not charged again), the retained Work terms
# of each scope charged once per issued occurrence, its size W the sum of their
# primitive operation counts (every resource and width; an unknown term identified, never counted), its speed
# W(current) / estimated_seconds and its time W(shares) / speed over every variable of its terms. `reports` each unit
# with unknown work or none (`costed` False and
# its `reason`: mixed kinds, no source task, no level, no counted work, or a term `counted` gives none), each such term
# with its `evidence`, and each cycle
# or receive edge `stages` finds within one class (not costed), with the sites and dispatches of its units; the stages
# of every function
def measured(bound, scheduled, segment, processors, observations, units):
    rank, made, reports, by, operated = {p: j for j, p in enumerate(processors)}, [], [], {}, {}
    scoped = [tuple(dict.fromkeys((o.call, n.kernel, n.first, n.last) for n in (bound.work[o.call].numerics[d]
                                                                                for d in o.dispatches)
                                  if n.kernel != mesh_graph.NONE)) for o in observations]
    scopes = {s: len(observations) + k for k, s in enumerate(dict.fromkeys(s for held in scoped for s in held))}
    component = mesh_graph.components(len(observations) + len(scopes), [
        edge for u, held in enumerate(scoped) for s in held for edge in ((u, scopes[s]), (scopes[s], u))])
    joined = {}
    for u, held in enumerate(scoped):
        if held:
            joined.setdefault(component[u], []).append(u)
    keyed = {}
    for members in joined.values():
        call = observations[members[0]].call
        keyed.setdefault(bound.target.bodies[bound.target.rows[call].function].key, []).append(members)
    tasked = {key: [tuple(dict.fromkeys(k for u in members for k in observations[u].identities)) for members in held]
              for key, held in keyed.items()}
    staged = {key: stages(scheduled, key, tasked.get(key, [])) for key in scheduled.modules}

    def described(members):
        return {"call": observations[members[0]].call,
                "sites": tuple(dict.fromkeys(observations[u].site for u in members)),
                "dispatches": tuple(d for u in members for d in observations[u].dispatches),
                "observations": tuple(members)}

    for key, held in keyed.items():
        stage = staged[key]
        reports += [{**found, "key": key, "units": [described(held[u]) for u in found["units"]]}
                    for found in stage.inside]
        for members, identities, tasks, level in zip(held, tasked[key], stage.members, stage.units):
            call = observations[members[0]].call
            work, kinds = bound.work[call], tuple(dict.fromkeys(observations[u].kind for u in members))
            estimated = sum(observations[u].sample_seconds * observations[u].executions for u in members)
            scopes = tuple(dict.fromkeys(s for u in members for s in scoped[u]))
            occurrences = dict.fromkeys(scopes, 0)
            for u in members:
                for n in (work.numerics[d] for d in observations[u].dispatches):
                    if n.kernel != mesh_graph.NONE and not n.later:
                        occurrences[call, n.kernel, n.first, n.last] += observations[u].executions
            about = {**described(members), "key": key, "kinds": kinds, "tasks": tasks,
                     "samples": tuple((u, observations[u].residue, observations[u].sample_seconds,
                                       observations[u].executions) for u in members),
                     "estimated_seconds": estimated,
                     "generated": tuple(k for k in identities if k not in stage.node), "scopes": scopes,
                     "occurrences": tuple(occurrences.items())}
            if call not in by:
                by[call], operated[call] = {}, {}
                for i in range(work.count):
                    by[call].setdefault(work.terms[i].kernel, []).append(i)
                for association in (work.associations[:work.associated] if work.associated else ()):
                    operated[call].setdefault(association.operation, []).append(association)
            charged = {}
            for scope in scopes:
                _, kernel, first, last = scope
                for i in by[call].get(kernel, ()):
                    if first <= work.terms[i].member < last:
                        charged[i] = charged.get(i, 0) + occurrences[scope]
            counts, unknown, failed = [[0.0] * 3 for _ in range(3)], [], []
            size, constant, coefficients = 0.0, 0.0, {}
            for i, times in charged.items():
                t = work.terms[i]
                owned = operated[call].get(t.operation, ())
                if t.coefficient.unknown:
                    unknown.append(evidence(work, i, owned, stage))
                current, form = counted(work, t, owned, stage, segment, rank, units)
                if current is None:
                    failed.append({**evidence(work, i, owned, stage), "factor": form[0], "atom": form[1],
                                   "reason": form[2]})
                    continue
                each = 0.0
                for r in range(3):
                    for w in range(3):
                        counts[r][w] += t.coefficient.operations[r][w] * current * times
                        each += t.coefficient.operations[r][w] * times
                size += each * current
                constant += each * form[0]
                for variable, c in form[1].items():
                    coefficients[variable] = coefficients.get(variable, 0.0) + each * c
            about.update(work=counts, unknown=unknown, size=size)
            reason = ("mixed unit kinds" if len(kinds) != 1 else "no source task" if not tasks else
                      "no level: a cycle of classes" if level is None else "no counted work" if not size else
                      "a term without a count expression" if failed else None)
            if reason is not None:
                reports.append({**about, "terms": failed, "reason": reason, "costed": False})
                continue
            if unknown:
                reports.append({**about, "costed": True})
            made.append(Cost(key, level, (bound.target.node, kinds[0]),
                             tuple(dict.fromkeys(rank[stage.node[k].meta["processor"]] for k in tasks)),
                             estimated * constant / size,
                             {variable: estimated * c / size for variable, c in coefficients.items()}, about))
    return made, reports, staged


# S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3): the costs of one stage (function key, level) on one resource
# summed into one row, {(key, level, resource): [fixed, {(part, rank): coefficient}, ranks]}
def summed(costs):
    rows = {}
    for cost in costs:
        row = rows.setdefault((cost.key, cost.stage, cost.resource), [0.0, {}, set()])
        row[0] += cost.fixed
        for variable, c in cost.coefficients.items():
            row[1][variable] = row[1].get(variable, 0.0) + c
        row[2].update(cost.ranks)
    return rows


# S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3): each (function key, level) one stage of `coupled` with its
# summed rows; its parts' collective times `comm(receive)` of its external receives, a unit of the largest share of
# part `segment` of the receive's sent value's original, each part's largest
def solved(staged, segment, costs, comm, upper):
    index, rows, times = {}, [], {}
    for (key, level, _), (fixed, terms, _) in summed(costs).items():
        rows.append((index.setdefault((key, level), len(index)), fixed, terms))
    for key, stage in staged.items():
        for level, received in stage.received.items():
            held = times.setdefault(index.setdefault((key, level), len(index)), {})
            for node, sent in received:
                time, part = comm(node), segment(sent)
                if part is None and time:
                    raise ValueError(f"the collective time of {node.name} has no part")
                if part is not None:
                    held[part] = max(held.get(part, 0.0), time)
    parts = sorted(dict.fromkeys([part for _, _, terms in rows for part, _ in terms] +
                                 [part for held in times.values() for part in held]))
    return coupled(parts, [times.get(i, {}) for i in range(len(index))], rows, upper)


# S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3); rdma/mesh-graph.h mesh_graph_shares_coupled: each part's shares
# of the ranks in one solve, `comm[i][part]` stage i's collective time a unit of that part's largest share and each row
# (stage, fixed, {(part, rank): coefficient}) one stage's coupled row
def coupled(parts, comm, rows, upper):
    index = {part: p for p, part in enumerate(parts)}
    solved = mesh_graph.shares_coupled([[stage.get(part, 0.0) for part in parts] for stage in comm],
                                       [(i, fixed, {(index[part], j): c for (part, j), c in terms.items()})
                                        for i, fixed, terms in rows], upper, len(parts))
    return dict(zip(parts, solved))


# the current share of every full share variable: a part of `given` ({part: units by rank}) its units over their sum,
# another's each of its shards' `units` (x, n) x / n (part `segment`, rank its processor's index in `processors`)
def state(staged, segment, processors, given):
    rank, found = {p: j for j, p in enumerate(processors)}, {}
    for stage in staged.values():
        for node in dict.fromkeys(stage.node.values()):
            if "of" in node.meta and segment(node) is not None and segment(node) not in given:
                x, n = units(node)
                found.setdefault((segment(node), rank[node.meta["processor"]]), set()).add(x / n)
    for part, held in given.items():
        for j, unit in enumerate(held):
            found[part, j] = {unit / sum(held)}
    return found


# X. Jia et al., Whale, USENIX ATC 2022, Algorithm 1: rank j's load at part `part`'s units `held`, the value of each
# summed row (`summed`) whose resource executes rank j's tasks, its every other part's variable at its `state` share
def load(rows, current, part):
    def share(p, k, held):
        if p == part:
            return held[k] / sum(held)
        if len(current[p, k]) != 1:
            raise ValueError(f"part {p} rank {k} holds the shares {sorted(current[p, k])}")
        return next(iter(current[p, k]))

    def loaded(j, held):
        return sum(fixed + sum(c * share(p, k, held) for (p, k), c in terms.items())
                   for fixed, terms, ranks in rows.values() if j in ranks)
    return loaded


# X. Jia et al., Whale, USENIX ATC 2022, Algorithm 1
def shifted(units, upper, load):
    units = list(units)
    while True:
        use = [unit / sum(units) / most for unit, most in zip(units, upper)]
        peak = max(range(len(units)), key=use.__getitem__)
        if use[peak] <= 1:
            return tuple(units)
        free = [j for j, most in enumerate(upper) if (units[j] + 1) / sum(units) <= most]
        units[min(free, key=lambda j: load(j, units))] += 1
        units[peak] -= 1


# design/unequal-shards-change-request.md item 11
Ragged = collections.namedtuple("Ragged", "values offsets")


# design/unequal-shards-change-request.md item 13
def rows(ragged, speeds):
    cuts, held = allocate.separators(ragged.offsets.tolist(), speeds), []
    while isinstance(ragged, Ragged):
        held.append((ragged.offsets.diff(), tuple(cuts)))
        cuts, ragged = ragged.offsets[cuts].tolist(), ragged.values
    return *held, (ragged, tuple(cuts))
