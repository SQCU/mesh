"""The graph module (rubric C9): the graph algorithms mesh's Python code runs, with their consumers as operands.

A caller supplies its vertices and a neighbours function. Nothing here knows what a vertex or an edge stands for.
"""
import collections


def walk(starts, neighbours):
    """The vertices reachable from ``starts``, themselves included, in the order a breadth-first walk first reaches
    them: ``starts`` in their order, then the vertices ``neighbours(v)`` gives, in its order, for each v so reached.
    Each vertex is visited once, so a graph with cycles (a fabric's cables) ends."""
    order, seen, queue = [], set(), collections.deque(starts)
    while queue:
        v = queue.popleft()
        if v not in seen:
            seen.add(v)
            order.append(v)
            queue.extend(neighbours(v))
    return order


def reach(starts, neighbours):
    """The set of vertices reachable from ``starts``, themselves included: walk's vertices, unordered."""
    return set(walk(starts, neighbours))
