"""INPUTS — assemble the computed chorus for the composer. Invents nothing.

``strategy.strategy()`` takes seven arrays. This file produces them from what the
engine and the licensed computed modules already give us, and does no more than
that: it calls, it does not derive.

    XAN   xan        raw per-player engine rows, VERBATIM      (l, d_x)
    ZED   zed        per-instrument descriptors                (m, d_z)
    cells cell_slots RHO-contracted per-cell slots             (c, d_c)
    GIGI  gigi       bounded-support spatial mask              (l, c)
    DEE   dee        DPP marginal inclusion                    (m,)
    PIA/SUE/NIM      closed-form cartstate semantics           (l, d_sem)
    MASK  mask       which actions exist                       (l, m)

What this file must never do (SPEC §7):
  * put a derived quantity into XAN. XAN is the engine's own row — health, armor,
    ammo, weapons bitmask, position, velocity, team. The deleted
    ``estimator.state_from_runtime`` filled ``x[:, :8]`` with own-nimber,
    rival-nimber max/mean, team-size fraction and an is-current-winner flag, i.e.
    it handed the solver the conclusions the learned operator exists to produce.
    Those belong in ``semantics``, as PIA/SUE/NIM, where they are read as
    closed-form Game-1 features and are never mistaken for raw state.
  * compute a pairwise feature. The per-(player, instrument) quantity is
    ``quinn · kay``, computed inside the composer.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["ChorusArrays", "assemble"]


class ChorusArrays(NamedTuple):
    """Exactly the seven arguments ``strategy.strategy()`` consumes."""

    xan: np.ndarray         # (l, d_x)   raw engine rows, verbatim
    zed: np.ndarray         # (m, d_z)   instrument descriptors
    cell_slots: np.ndarray  # (c, d_c)   RHO-contracted cell slots
    gigi: np.ndarray        # (l, c)     bounded-support spatial mask
    dee: np.ndarray         # (m,)       DPP marginal inclusion
    semantics: np.ndarray   # (l, d_sem) PIA / SUE / NIM, closed-form
    mask: np.ndarray        # (l, m)     which actions exist


def assemble(
    obs_rows: np.ndarray,      # (l, d_x)  the engine's per-player OBS columns
    batch,                     # InstrumentBatch: .descriptors (m, d_z), .eligible (l, m)
    cell_slots: np.ndarray,    # (c, d_c)  from featurize (RHO already applied)
    gigi: np.ndarray,          # (l, c)    from featurize (GIGI's bounded mask)
    dee: np.ndarray,           # (m,)      from dpp (DEE)
    semantics: np.ndarray,     # (l, d_sem) from game / game_value (PIA/SUE/NIM)
) -> ChorusArrays:
    """Bundle the seven arrays. Every one arrives already computed elsewhere.

    There is deliberately no arithmetic in this function beyond ``asarray``: the
    moment this file starts computing a feature it has become the thing §7
    forbids, and the thing the deleted ``state_from_runtime`` was.
    """
    return ChorusArrays(
        xan=np.asarray(obs_rows, dtype=np.float32),
        zed=np.asarray(batch.descriptors, dtype=np.float32),
        cell_slots=np.asarray(cell_slots, dtype=np.float32),
        gigi=np.asarray(gigi, dtype=np.float32),
        dee=np.asarray(dee, dtype=np.float32),
        semantics=np.asarray(semantics, dtype=np.float32),
        mask=np.asarray(batch.eligible, dtype=bool),
    )
