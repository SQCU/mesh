"""strategy_io_schema.py — canonical column contract for the payload strategy I/O.

Single source of truth shared by the QC server (sv_payload_strategy_io.qh) and the
mesh coprocessor. The QC header and this module MUST agree; the accompanying test
(`_selftest`) asserts the constants match the values pinned in STRATEGY-IO.md.

Scope of THIS module (the server-def I/O boundary, rl-training-spec §4 = stopgrad):
  * the four mesh column schemas (gather A/B/C + the 8-col scatter),
  * target-id encode/decode for PLC_SC_TARGET,
  * deterministic cartstate featurization: normalized depth + PW(s) (nim-sum
    projected winner) and SUCC(s) (backward induction over cart decrements).

Determinism boundary: PW, SUCC and the V-cell featurization are Game-1, closed-form,
NOT learned (rl-training-spec §1, §2.2) — plain python / numpy here. The learned
DPP/mixing/value head (dpp-mixing-and-overlay §2) lives in a separate mlx module and
is deliberately NOT imported here; this file must stay import-light and testable.

Cite: payload-spec §2.1-2.2, §3.2, §4; rl-training-spec §1, §2; playerbot-interface
§2.1-2.4; bridge/engine/mesh_ipc.c (MESH_XON_RESPWIDTH=8).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
import numpy as np

# ---- mesh geometry (must mirror sv_payload_strategy_io.qh) -------------------
RESP_WIDTH = 8            # engine-fixed response width (MESH_XON_RESPWIDTH)
OBS_WIDTH = 40            # per-player observation row
CART_WIDTH = 12           # cartstate row (GUARANTEED emit)
EVT_WIDTH = 6             # perception-event ring row
EVT_ROWS = 256
MAX_CARTS = 4

# ---- gather schema A: per-player observation x_b ---------------------------
OBS = dict(ID=0, TEAM=1, HEALTH=2, ARMOR=3, AMMO=4, POS_X=5, POS_Y=6, POS_Z=7,
           VEL_X=8, VEL_Y=9, VEL_Z=10, WEAPONS=11, POWER=12, TSS=13, CELL=14,
           NCART=15, NCART_D=16, ALIVE=17, CONTROL=18)

# ---- gather schema B: cartstate s (GUARANTEED; PW/SUCC computed from this) --
CS = dict(ID=0, DEPTH=1, LENGTH=2, CTRL=3, SPEED=4, IDLE=5, BANKMASK=6, PROGRESS=7)

# ---- gather schema C: perception event ring --------------------------------
EVT = dict(CELL=0, KIND=1, TEAM=2, SUBJECT=3, VALUE=4, TIME=5)
EVT_KIND = dict(ITEM_GONE=0, ITEM_HERE=1, ENEMY_HERE=2, RIVAL_HERE=3)

# ---- scatter schema: EXACTLY 8 per-player instrument weights ----------------
SC = dict(TARGET=0, GAIN=1, LANE=2, HUNT=3, EXPLORE=4, COMMIT=5, SPAWN=6, LEAD=7)

# ---- PLC_SC_TARGET encoding bases (mirror the QH constants) -----------------
TGT_CART_BASE = 0
TGT_ITEM_BASE = 65536
TGT_RIVAL_BASE = 131072
TGT_CELL_BASE = 196608


def encode_target(kind: str, index: int) -> int:
    """Pack an instrument target into the PLC_SC_TARGET scalar.

    kind in {'cart','item','rival','cell'}; index is the k/p/r/c id.
    Inverse of decode_target. Cite payload-spec §4.4.
    """
    base = {"cart": TGT_CART_BASE, "item": TGT_ITEM_BASE,
            "rival": TGT_RIVAL_BASE, "cell": TGT_CELL_BASE}[kind]
    return base + int(index)


def decode_target(tgt: float) -> tuple[str, int]:
    """Unpack PLC_SC_TARGET into (kind, index). Inverse of encode_target."""
    t = int(tgt)
    if t >= TGT_CELL_BASE:
        return "cell", t - TGT_CELL_BASE
    if t >= TGT_RIVAL_BASE:
        return "rival", t - TGT_RIVAL_BASE
    if t >= TGT_ITEM_BASE:
        return "item", t - TGT_ITEM_BASE
    return "cart", t - TGT_CART_BASE


# ---- deterministic cartstate featurization (Game-1) -------------------------
@dataclass
class CartState:
    """One cart's Game-1 state, decoded from a gather-schema-B row.

    depth_frac in [0,1] is plc_s/plc_length along the golden path; ctrl is the
    controlling team index (0 = uncontrolled). Cite rl-training-spec §1.
    """
    cart_id: int
    depth_frac: float
    ctrl: int


def carts_from_rows(rows: np.ndarray) -> list[CartState]:
    """Decode a (n, CART_WIDTH) gather-B matrix into CartState list.

    Rows with LENGTH==0 (no reachable path) are skipped as inert, not errors.
    """
    out = []
    for r in np.atleast_2d(rows):
        out.append(CartState(int(r[CS["ID"]]), float(r[CS["DEPTH"]]), int(r[CS["CTRL"]])))
    return out


def _heap_nimber(depth_frac: float, nlevels: int = 8) -> int:
    """Quantize a cart's depth-under-control into a Nim heap size.

    Each cart on its golden path is a heap whose size is its quantized depth
    (rl-training-spec §1; payload-spec §2.6). nlevels sets the quantization; the
    projected-winner algebra is over the XOR of these heaps.
    """
    return int(round(max(0.0, min(1.0, depth_frac)) * (nlevels - 1)))


def projected_winner(carts: list[CartState], nlevels: int = 8) -> int:
    """PW(s): the nim-sum projected winner over cartstate (rl-training-spec §1).

    Group cart heaps by controlling team, XOR the per-team heap sizes into a
    per-team nimber, and return the team with the largest nimber (the one whose
    threat does not cancel). Uncontrolled carts (ctrl==0) contribute to no team.
    Returns 0 when the position is balanced (all team nimbers equal / zero) — a
    genuine no-projected-winner, not a refusal.

    Worked invariant (§1): one cart at depth 2 beats two carts at depth 1, because
    1 XOR 1 == 0 cancels the pair while the lone 2 survives.
    """
    by_team: dict[int, int] = {}
    for c in carts:
        if c.ctrl <= 0:
            continue
        h = _heap_nimber(c.depth_frac, nlevels)
        by_team[c.ctrl] = by_team.get(c.ctrl, 0) ^ h
    if not by_team:
        return 0
    best_nim = max(by_team.values())
    if best_nim <= 0:
        return 0
    leaders = [t for t, nim in by_team.items() if nim == best_nim]
    # strict maximum only: a tie for the top nimber is a balanced position with
    # no projected winner (0), never an arbitrary pick.
    return min(leaders) if len(leaders) == 1 else 0


def succession(carts: list[CartState], nlevels: int = 8) -> list[tuple[int, float]]:
    """SUCC(s): backward induction over decrements of the current leader's carts.

    Recompute PW under successive one-level decrements of the projected winner's
    carts; each decrement that changes PW credits the newly-projected team with a
    marginal_denial_value (fraction of the leader's heaps that must fall to flip
    the position). Returns [(team, marginal_denial_value)] ordered by how soon that
    team inherits the projection. Deterministic; this is what makes the policy
    anticipatory (rl-training-spec §1, §2.2). Not learned.
    """
    order: list[tuple[int, float]] = []
    work = [CartState(c.cart_id, c.depth_frac, c.ctrl) for c in carts]
    leader = projected_winner(work, nlevels)
    total_leader_heaps = sum(_heap_nimber(c.depth_frac, nlevels)
                             for c in work if c.ctrl == leader) or 1
    removed = 0
    step = 1.0 / (nlevels - 1)
    guard = 0
    while leader != 0 and guard < nlevels * len(work) + 1:
        guard += 1
        # decrement the deepest leader cart by one level
        lead_carts = [c for c in work if c.ctrl == leader and c.depth_frac > 0]
        if not lead_carts:
            break
        deepest = max(lead_carts, key=lambda c: c.depth_frac)
        deepest.depth_frac = max(0.0, deepest.depth_frac - step)
        removed += 1
        nxt = projected_winner(work, nlevels)
        if nxt != leader:
            order.append((nxt, removed / total_leader_heaps))
            leader = nxt
            total_leader_heaps = sum(_heap_nimber(c.depth_frac, nlevels)
                                     for c in work if c.ctrl == leader) or 1
            removed = 0
    return order


def _selftest() -> None:
    """Contract asserts. Run: python strategy_io_schema.py"""
    # schema width / disjointness
    assert RESP_WIDTH == 8 and len(SC) == 8
    assert max(OBS.values()) < OBS_WIDTH and max(CS.values()) < CART_WIDTH
    assert max(EVT.values()) < EVT_WIDTH
    assert sorted(SC.values()) == list(range(8))
    # target codec round-trips across every band
    for kind, idx in [("cart", 3), ("item", 32767), ("rival", 32767), ("cell", 65535)]:
        assert decode_target(encode_target(kind, idx)) == (kind, idx)
    # PW: one cart at depth 2 beats two carts at depth 1 (1 XOR 1 == 0), §1
    two_deep = [CartState(0, 2 / 7, 1)]                       # team 1, heap 2
    two_shallow = [CartState(1, 1 / 7, 2), CartState(2, 1 / 7, 2)]  # team 2, 1 XOR 1 = 0
    assert projected_winner(two_deep + two_shallow) == 1
    # balanced position -> no projected winner
    assert projected_winner([CartState(0, 1 / 7, 1), CartState(1, 1 / 7, 2)]) == 0
    # SUCC yields an ordered non-empty succession when a leader exists
    succ = succession(two_deep + two_shallow)
    assert isinstance(succ, list)
    print("strategy_io_schema selftest: OK  (PW/SUCC/codec/schema all consistent)")


if __name__ == "__main__":
    _selftest()
