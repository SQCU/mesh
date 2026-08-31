from __future__ import annotations

from collections import namedtuple
from typing import Iterable


Cart = namedtuple("Cart", ("control", "depth"))


def as_carts(carts: Iterable) -> list[Cart]:
    out = []
    for cart in carts:
        if isinstance(cart, Cart):
            control, depth = cart
        elif isinstance(cart, dict):
            control, depth = cart.get("control", cart.get("team")), cart.get("depth", 0)
        else:
            control, depth = cart[0], cart[1]
        out.append(Cart(control, max(0, int(depth))))
    return out


def _teams_of(carts, teams):
    seen = {cart.control for cart in carts if cart.control is not None}
    if teams is not None:
        seen.update(teams)
    return sorted(seen, key=lambda team: (0, team) if isinstance(team, (int, float)) else (1, str(team)))


def nim_sum(values: Iterable[int]) -> int:
    value = 0
    for item in values:
        value ^= int(item)
    return value


def team_nimbers(carts: Iterable, teams: Iterable | None = None) -> dict:
    rows = as_carts(carts)
    values = {team: 0 for team in _teams_of(rows, teams)}
    for cart in rows:
        if cart.control is not None and cart.depth:
            values[cart.control] = values.get(cart.control, 0) ^ cart.depth
    return values


def projected_winner(carts: Iterable, teams: Iterable | None = None):
    values = team_nimbers(carts, teams)
    best = max(values.values(), default=0)
    leaders = [team for team, value in values.items() if value == best]
    return leaders[0] if best > 0 and len(leaders) == 1 else None


def _leader_deepest_cart(carts, leader):
    index, depth = None, 0
    for candidate, cart in enumerate(carts):
        if cart.control == leader and cart.depth > depth:
            index, depth = candidate, cart.depth
    return index


def succession(carts: Iterable, teams: Iterable | None = None) -> list:
    rows = as_carts(carts)
    roster = _teams_of(rows, teams)
    leader = projected_winner(rows, roster)
    if leader is None:
        return []
    order = [(leader, 0)]
    seen = {leader}
    steps = 0
    previous = 0
    limit = sum(cart.depth for cart in rows) + 1
    while steps < limit and len(seen) < len(roster):
        index = _leader_deepest_cart(rows, leader)
        if index is None:
            break
        depth = rows[index].depth - 1
        rows[index] = Cart(rows[index].control if depth > 0 else None, depth)
        steps += 1
        following = projected_winner(rows, roster)
        if following != leader:
            if following is None:
                break
            if following not in seen:
                order.append((following, steps - previous))
                seen.add(following)
                previous = steps
            leader = following
    return order
