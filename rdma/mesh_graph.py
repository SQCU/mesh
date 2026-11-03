"""The graph module (rubric C9): the graph algorithms mesh's Python code runs, with their consumers as operands.

A caller supplies its vertices and a neighbours function. Nothing here knows what a vertex or an edge stands for.
"""


def reach(starts, neighbours):
    """The set of vertices reachable from ``starts``, themselves included. ``neighbours(v)`` gives the vertices
    adjacent to v. Each vertex is visited once, so a graph with cycles (a fabric's cables) ends."""
    seen, stack = set(), list(starts)
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(neighbours(v))
    return seen
