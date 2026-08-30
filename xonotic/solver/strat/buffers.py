"""Two deterministic strategy-side buffers: the OBSERVATION buffer and the REPLAY buffer.

This module is pure plumbing -- the deterministic, non-differentiable substrate the
learned heads (``dpp.py`` / ``head.py`` / ``value.py``) read from and write to. Nothing
here differentiates: both buffers are containers, so they are plain python / numpy by
design (`rl-training-spec.md` §4 "computed (deterministic): ``s, b, PW, SUCC``"; the
observation buffer feeds the belief ``b``, and the replay buffer stores the on-policy
transitions the policy gradient replays -- neither is a learned surface). Learned/mlx
objects (activations, log-probs, the query) are stored *as-is*: this file never
introspects, stacks, or differentiates them, so it carries no mlx dependency.

There are two independent buffers, answering to two different spec sections.

(a) ``ObservationBuffer`` -- the per-team buffer of contextual OBSERVATION events
    (`payload-spec.md` §2.2 "Observation buffer -> belief", step 1; `playerbot-interface.md`
    §6 "Perception -> observation buffer"). Each entry is a timestamped contextual event
    deposited at a V-cell -- "RL gone at cell c" (item spawn/despawn) or "enemy of team t
    at cell c" (enemy-seen). It is written from **perception-gated** bot observations: an
    observation only becomes an event if it passes ALL THREE necessary gates
    (`payload-spec.md` §2.2 step 1) --
        (a) the feature is in the bot's **view frustum**,
        (b) it is **unoccluded** by a line-of-sight raycast,
        (c) it is **within 2 V-cells** of range (a hard cap so one pathological long
            sightline cannot rewrite the whole buffer).
    Nothing enters that no bot actually saw -- that is what makes stealth emergent
    (§2.2) and is the ONLY path by which enemy positions reach the strategy operator.
    This buffer is the input to ``featurize.py`` (which builds the per-cell slot vectors
    ``f_c`` and integrates them into the egocentric belief ``beta_b`` of §2.2 stages
    2-5). This module deposits the raw, gated, timestamped events and exposes a per-cell
    view of them; the temporal contraction ``rho(dt)`` and spatial kernel ``g`` of
    §2.2 (stages 3-5) are ``featurize.py``'s job, not this container's.

(b) ``ReplayBuffer`` -- the on-policy transition buffer of the REINFORCE policy gradient
    (`rl-training-spec.md` §2.1 POLICY-GRADIENT: "per-step ``(state, activations, action,
    logpi)`` buffered to the crediting horizon and replayed"; §2.2 normalized register:
    "buffer(state,activation,a,logpi)->horizon->replay"). Per strategy step it stores one
    ``Transition`` -- the strategy ``state``, the per-player activations ``A_team`` /
    ``A_player`` (the single shared ``W_all`` is invariant; the per-team / per-player
    distinctions live entirely in these activations, §0/§4), the sampled ``action`` and
    its ``logpi`` (from ``head.sample_strategy``). It retains transitions to the crediting
    horizon and yields rollouts for the policy gradient. The crediting horizon and rollout
    segmentation are marked ``[OPEN]`` in the spec (§6), so both are parameters here with
    faithful defaults (keep the whole buffered window; one rollout = the window).

Neither buffer computes reward / value / advantage -- those are ``value.py`` (advantage)
and the raw reward (``[OPEN]``, §2.1). The replay buffer only stores what the policy
gradient needs to *recompute* ``log pi`` and weight it by the (separately computed)
per-player advantage; it exposes a ``rollouts`` iterator the training loop drives.

Spec: `payload-spec.md` §2.2 (observation buffer -> belief; the three perception gates),
      `rl-training-spec.md` §2 (definitions; POLICY-GRADIENT buffering), §6 (open: horizon,
      segmentation). Companion consumers: `featurize.py` (belief), `value.py` (advantage),
      `head.py` (sampling / log-probs).

Public surface
--------------
Observation side:
- ``EventKind``           : the contextual-event kinds (ITEM_SPAWN/ITEM_DESPAWN/ENEMY_SEEN).
- ``Observation``         : a raw candidate observation from one bot (carries the gate inputs).
- ``ContextualEvent``     : a stored, gated, timestamped per-cell event.
- ``PerceptionGate``      : the three-gate (frustum + LOS + 2-V-cell) predicate.
- ``ObservationBuffer``   : per-team event store; ``observe`` (gated push), ``events``,
                            ``cell_view`` (per-cell latest, the featurize.py input),
                            ``prune_before``, ``clear``.
Replay side:
- ``Transition``          : one buffered ``(step, state, a_team, a_player, action, logpi, info)``.
- ``ReplayBuffer``        : horizon-bounded transition store; ``push``, ``rollouts``,
                            ``latest``, ``collate``, ``__len__``, ``clear``.
"""

from __future__ import annotations

from collections import deque, namedtuple
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np


# =========================================================================== #
# (a) OBSERVATION BUFFER  (payload-spec.md §2.2 step 1; playerbot-interface.md §6)
# =========================================================================== #

class EventKind(Enum):
    """The contextual-event kinds a perception pass can deposit (`payload-spec.md` §2.2).

    ``ITEM_SPAWN`` / ``ITEM_DESPAWN`` are the *status* transitions of a map feature
    ("RL back at cell c" / "RL gone at cell c") -- edge-triggered, so a repeated
    identical status for the same subject is suppressed (`playerbot-interface.md` §6:
    "edge-triggered spawn/despawn"). ``ENEMY_SEEN`` is a sighting *event* ("enemy of
    team t at cell c") -- every sighting is recorded, since re-sightings refresh the
    last-observed time / cell that ``featurize.py``'s ``rho(dt)`` relaxes.
    """
    ITEM_SPAWN = "item_spawn"
    ITEM_DESPAWN = "item_despawn"
    ENEMY_SEEN = "enemy_seen"

    @property
    def is_status(self) -> bool:
        """True for edge-triggered status kinds (item spawn/despawn), False for events."""
        return self in (EventKind.ITEM_SPAWN, EventKind.ITEM_DESPAWN)


# A raw candidate observation emitted by one bot's perception pass. It carries BOTH the
# semantic content (team/kind/cell/subject/t) AND the three gate inputs the buffer checks
# (`payload-spec.md` §2.2 step 1). Keeping the gate inputs on the observation -- rather
# than doing raycasts here -- keeps this container deterministic and unit-testable: the
# engine (or a test) supplies whether the feature was in frustum, whether the LOS raycast
# was clear, and the observer->subject V-cell graph distance; the buffer enforces that all
# three pass. ``subject`` identifies what was seen (an item id, or an ("enemy", team, id)
# tuple) so status edges can be deduplicated per subject. ``payload`` is an optional dict
# of extra featurize.py fields (respawn-phase estimate, standability, ...).
Observation = namedtuple(
    "Observation",
    ("team", "observer", "t", "cell", "kind", "subject",
     "in_frustum", "los_clear", "vcell_dist", "payload"),
)
Observation.__new__.__defaults__ = (True, True, 0.0, None)  # in_frustum, los_clear, vcell_dist, payload


# A stored, gated, timestamped contextual event at a V-cell -- what actually lives in the
# per-team buffer and what ``featurize.py`` reads to build the per-cell slot vector ``f_c``.
ContextualEvent = namedtuple(
    "ContextualEvent",
    ("team", "observer", "t", "cell", "kind", "subject", "payload"),
)


class PerceptionGate:
    """The three necessary perception gates of `payload-spec.md` §2.2 (step 1).

    An observation deposits an event ONLY if it passes all three -- ``in_frustum`` AND
    ``los_clear`` AND ``vcell_dist <= vcell_cap`` (default cap = 2 V-cells, the spec's
    hard range cap). All three are necessary (§2.2: "All three gates necessary"); the cap
    exists so a pathological long-sightline custom map cannot let one glance rewrite the
    whole buffer.

    The gate is deterministic and side-effect free. The three predicates are configurable
    (so a test or a richer engine pass can override any one), but each defaults to reading
    the correspondingly named field off the :class:`Observation`.
    """

    def __init__(
        self,
        vcell_cap: float = 2.0,
        frustum_pred: Optional[Callable[[Observation], bool]] = None,
        los_pred: Optional[Callable[[Observation], bool]] = None,
        range_pred: Optional[Callable[[Observation], bool]] = None,
    ):
        self.vcell_cap = float(vcell_cap)
        self._frustum = frustum_pred or (lambda o: bool(o.in_frustum))
        self._los = los_pred or (lambda o: bool(o.los_clear))
        self._range = range_pred or (lambda o: o.vcell_dist is not None
                                     and float(o.vcell_dist) <= self.vcell_cap)

    def passes(self, obs: Observation) -> bool:
        """True iff the observation clears frustum AND LOS AND the 2-V-cell range cap."""
        return bool(self._frustum(obs) and self._los(obs) and self._range(obs))

    # Per-gate readout, for diagnostics / tests (which gate rejected an observation).
    def gate_flags(self, obs: Observation) -> dict:
        """``{'frustum':bool,'los':bool,'range':bool,'pass':bool}`` for one observation."""
        f, l, r = bool(self._frustum(obs)), bool(self._los(obs)), bool(self._range(obs))
        return {"frustum": f, "los": l, "range": r, "pass": f and l and r}


class ObservationBuffer:
    """Per-team buffer of perception-gated contextual events (`payload-spec.md` §2.2).

    Each team owns an ordered, timestamped log of :class:`ContextualEvent`. Events are
    written ONLY through :meth:`observe`, which runs every raw observation through the
    :class:`PerceptionGate` (frustum + LOS + 2-V-cell) and, for status kinds
    (item spawn/despawn), edge-triggers -- a repeated identical status for the same
    subject deposits nothing (`playerbot-interface.md` §6). This is the object
    ``featurize.py`` reads: :meth:`cell_view` returns the per-cell latest-event material
    it needs (last-observed time, observed-enemy presence, item status per cell), and the
    temporal / spatial mixing of §2.2 stages 3-5 (``rho(dt)`` decay, graph-distance kernel
    ``g``) is applied there, over this buffer's contents, not here.

    Determinism: events keep strict insertion order; ties in time never reorder. There is
    **no team belief object** (§2.2) -- only this per-team event buffer; beliefs are
    per-bot and are computed downstream by ``featurize.py`` from these events.

    Parameters
    ----------
    gate : PerceptionGate, optional
        The three-gate predicate. Defaults to ``PerceptionGate()`` (2-V-cell cap).
    edge_triggered : bool
        If True (default), status kinds (item spawn/despawn) are edge-triggered per
        subject: an identical consecutive status is suppressed. Event kinds (enemy-seen)
        are always recorded regardless.
    capacity : int, optional
        Per-team max retained events (a ring). ``None`` = unbounded. Oldest drop first.
    """

    def __init__(
        self,
        gate: Optional[PerceptionGate] = None,
        edge_triggered: bool = True,
        capacity: Optional[int] = None,
    ):
        self.gate = gate if gate is not None else PerceptionGate()
        self.edge_triggered = bool(edge_triggered)
        self.capacity = capacity
        # team -> deque[ContextualEvent]
        self._events: dict[Any, deque] = {}
        # (team, subject) -> last recorded status EventKind (for edge-triggering)
        self._last_status: dict[tuple, EventKind] = {}

    # -- writing -------------------------------------------------------------- #

    def observe(self, obs: Observation) -> Optional[ContextualEvent]:
        """Gate one raw observation; deposit and return the event, or ``None`` if rejected.

        Returns ``None`` when the observation fails any perception gate (`payload-spec.md`
        §2.2 step 1) OR is an edge-suppressed duplicate status. On a pass, appends a
        :class:`ContextualEvent` to the observer's team buffer and returns it.
        """
        if not self.gate.passes(obs):
            return None

        kind = obs.kind if isinstance(obs.kind, EventKind) else EventKind(obs.kind)

        if self.edge_triggered and kind.is_status:
            key = (obs.team, obs.subject)
            if self._last_status.get(key) == kind:
                return None  # identical consecutive status -> not an edge, suppress
            self._last_status[key] = kind

        event = ContextualEvent(
            team=obs.team, observer=obs.observer, t=obs.t, cell=obs.cell,
            kind=kind, subject=obs.subject, payload=obs.payload,
        )
        buf = self._events.get(obs.team)
        if buf is None:
            buf = deque(maxlen=self.capacity)
            self._events[obs.team] = buf
        buf.append(event)
        return event

    def observe_many(self, observations: Iterable[Observation]) -> list:
        """Gate a batch of observations in order; return the list of deposited events."""
        return [e for e in (self.observe(o) for o in observations) if e is not None]

    # -- reading -------------------------------------------------------------- #

    def teams(self) -> list:
        """The teams that currently own any buffered event (insertion order)."""
        return list(self._events.keys())

    def events(self, team: Any, since: Optional[float] = None,
               kinds: Optional[Sequence] = None) -> list:
        """Ordered events for ``team``, optionally filtered by ``since`` (t >=) and ``kinds``.

        The raw, gated event stream featurize.py folds through ``rho(dt)`` (§2.2 stage 3).
        ``kinds`` may hold :class:`EventKind` values (or their string values).
        """
        buf = self._events.get(team)
        if not buf:
            return []
        kset = None
        if kinds is not None:
            kset = {k if isinstance(k, EventKind) else EventKind(k) for k in kinds}
        out = []
        for e in buf:
            if since is not None and e.t < since:
                continue
            if kset is not None and e.kind not in kset:
                continue
            out.append(e)
        return out

    def cell_view(self, team: Any, since: Optional[float] = None) -> dict:
        """Per-cell latest-event summary for ``team`` -- the direct ``featurize.py`` input.

        Returns ``{cell: {...}}`` where each cell record carries the material §2.2 stage 2
        packs into the slot vector ``f_c``: the most recent status kind seen at the cell
        (``item_status``), the last-observed time (``last_t``), whether an enemy was seen
        there (``enemy_seen`` + ``last_enemy_t``), and the winning event's ``payload``.
        Only the LATEST event of each type per cell is kept -- featurize.py applies the
        temporal decay; this is the raw newest observation, deterministic under insertion
        order (later events overwrite earlier ones at equal or greater ``t``).

        This does NOT decay, mask, or integrate -- ``rho(dt)`` (stage 3), the spatial
        kernel ``g`` (stage 4), and the egocentric integration (stage 5) are all
        ``featurize.py``'s, over this dict.
        """
        view: dict = {}
        for e in self.events(team, since=since):
            rec = view.get(e.cell)
            if rec is None:
                rec = {"cell": e.cell, "last_t": e.t, "item_status": None,
                       "item_t": None, "enemy_seen": False, "last_enemy_t": None,
                       "payload": e.payload}
                view[e.cell] = rec
            # newest-wins on the shared last_t / payload
            if e.t >= rec["last_t"]:
                rec["last_t"] = e.t
                rec["payload"] = e.payload
            if e.kind.is_status:
                if rec["item_t"] is None or e.t >= rec["item_t"]:
                    rec["item_status"] = e.kind
                    rec["item_t"] = e.t
            elif e.kind == EventKind.ENEMY_SEEN:
                rec["enemy_seen"] = True
                if rec["last_enemy_t"] is None or e.t >= rec["last_enemy_t"]:
                    rec["last_enemy_t"] = e.t
        return view

    def prune_before(self, t: float) -> int:
        """Drop all events with ``e.t < t`` from every team; return how many were dropped.

        Coarse retention hook (a hard cutoff); the soft forgetting of §2.2 stage 3 is
        ``featurize.py``'s ``rho(dt)`` relaxation to prior, not a deletion. Edge-trigger
        state is left intact so a post-prune identical status still edge-suppresses.
        """
        dropped = 0
        for team, buf in self._events.items():
            keep = deque((e for e in buf if e.t >= t), maxlen=self.capacity)
            dropped += len(buf) - len(keep)
            self._events[team] = keep
        return dropped

    def clear(self, team: Optional[Any] = None) -> None:
        """Clear one team's events (and its edge state), or all teams if ``team`` is None."""
        if team is None:
            self._events.clear()
            self._last_status.clear()
        else:
            self._events.pop(team, None)
            for key in [k for k in self._last_status if k[0] == team]:
                self._last_status.pop(key, None)

    def __len__(self) -> int:
        """Total buffered events across all teams."""
        return sum(len(b) for b in self._events.values())


# =========================================================================== #
# (b) REPLAY BUFFER  (rl-training-spec.md §2.1 / §2.2 POLICY-GRADIENT buffering)
# =========================================================================== #

# One buffered strategy-step transition. Stores exactly what the REINFORCE policy gradient
# needs to replay (`rl-training-spec.md` §2.1: "per-step (state, activations, action,
# logpi) buffered to the crediting horizon and replayed"):
#   step      : the strategy-step index (monotone; for horizon windows / segmentation).
#   state     : the strategy state s the action was sampled under (cartstate + belief
#               features; stopgrad from the policy gradient's view, §2.1). Stored as-is.
#   a_team    : per-team activation A_team (the single W_all is shared; the team-level
#               distinctions live here, §0/§4). Stored as-is (may be an mlx array).
#   a_player  : per-player activation A_player (per-player distinctions). Stored as-is.
#   action    : the sampled instrument index/indices (from head.sample_strategy).
#   logpi     : log pi(action | s, b, SUCC) at sampling time (behavior log-prob).
#   info      : optional dict for anything the training loop later attaches (raw reward,
#               computed return, per-player advantage from value.advantage, ...). The
#               buffer never reads it -- reward/value/advantage are value.py / §2.1's, not
#               this container's.
Transition = namedtuple(
    "Transition",
    ("step", "state", "a_team", "a_player", "action", "logpi", "info"),
)
Transition.__new__.__defaults__ = (None,)  # info


class ReplayBuffer:
    """On-policy transition buffer for the REINFORCE policy gradient (`rl-training-spec.md` §2.1).

    Stores one :class:`Transition` per strategy step and retains transitions **to the
    crediting horizon**, then yields **rollouts** the policy-gradient training loop replays
    (§2.1 / §2.2: "buffer(state,activation,a,logpi)->horizon->replay"). The buffer is a
    pure container: it neither samples actions nor computes log-probs / advantage; the
    training loop recomputes ``log pi`` fresh from ``W_all`` on each replayed transition
    (``head.strategy_log_prob``) and weights it by the per-player advantage
    (``value.advantage``). Only ``W_all`` learns; everything this buffer holds is either
    ``stopgrad`` (state) or a behavior-time record (activations / action / behavior logpi).

    On-policy discipline: because REINFORCE is on-policy, a rollout is only valid against
    the ``W_all`` that generated it. :meth:`clear` (or ``drain=True`` on :meth:`rollouts`)
    empties the buffer after a gradient step so stale off-policy transitions are not
    replayed against updated weights.

    Horizon & segmentation are ``[OPEN]`` in the spec (§6), so both are parameters:

    Parameters
    ----------
    horizon : int, optional
        The crediting horizon = max retained transitions (a sliding window / ring). When
        the buffer is full the oldest transition drops. ``None`` = unbounded (retain the
        whole episode). This is the ``[OPEN]`` crediting horizon of §6.
    """

    def __init__(self, horizon: Optional[int] = None):
        self.horizon = horizon
        self._buf: deque = deque(maxlen=horizon)
        self._next_step = 0

    # -- writing -------------------------------------------------------------- #

    def push(
        self,
        state: Any,
        a_team: Any,
        a_player: Any,
        action: Any,
        logpi: Any,
        step: Optional[int] = None,
        info: Optional[dict] = None,
    ) -> Transition:
        """Buffer one strategy-step transition; return the stored :class:`Transition`.

        Fields are stored verbatim (no copy, no stacking, no differentiation) so mlx
        activations / log-probs pass straight through. ``step`` defaults to a monotone
        internal counter (so horizon windows and segmentation are well-defined even if the
        caller does not track step indices). ``info`` is an optional dict the training loop
        may later populate with reward / return / advantage; the buffer never reads it.
        """
        if step is None:
            step = self._next_step
        self._next_step = max(self._next_step, step + 1)
        tr = Transition(step=step, state=state, a_team=a_team, a_player=a_player,
                        action=action, logpi=logpi, info=info)
        self._buf.append(tr)
        return tr

    def push_transition(self, tr: Transition) -> Transition:
        """Buffer a pre-built :class:`Transition` (advances the internal step counter)."""
        self._next_step = max(self._next_step, tr.step + 1)
        self._buf.append(tr)
        return tr

    # -- reading / replay ----------------------------------------------------- #

    def latest(self, n: Optional[int] = None) -> list:
        """The most recent ``n`` transitions in order (all of them if ``n`` is None)."""
        items = list(self._buf)
        return items if n is None else items[-n:]

    def rollouts(
        self,
        segment_len: Optional[int] = None,
        drop_last: bool = False,
        drain: bool = False,
    ) -> list:
        """Yield rollouts (contiguous transition segments) for the policy gradient.

        A rollout is a contiguous, in-order run of buffered transitions -- the unit the
        REINFORCE gradient sums over (`rl-training-spec.md` §2.1: ``grad J =
        E_rollout[ sum_u A_u (.) grad log pi ]``). ``segment_len`` is the ``[OPEN]``
        rollout segmentation of §6:

        - ``None`` (default): ONE rollout = the whole buffered window (the faithful
          default -- the entire crediting-horizon window is one credit-assignment span).
        - an int: fixed-length contiguous segments, in order; a trailing short segment is
          kept unless ``drop_last`` is set.

        Returns a ``list`` of rollouts, each a ``list[Transition]`` in step order. If
        ``drain`` is True the buffer is emptied afterward (on-policy: consume then discard
        so the next gradient step does not replay stale transitions). Deterministic: no
        shuffling; order is strictly the buffered step order.
        """
        items = list(self._buf)
        if segment_len is None:
            out = [items] if items else []
        else:
            if segment_len <= 0:
                raise ValueError("segment_len must be a positive int or None")
            out = [items[i:i + segment_len] for i in range(0, len(items), segment_len)]
            if drop_last and out and len(out[-1]) < segment_len:
                out.pop()
        if drain:
            self.clear()
        return out

    @staticmethod
    def collate(transitions: Sequence[Transition], field: str) -> list:
        """Pull one field out of a rollout, in order -- e.g. ``collate(roll, 'logpi')``.

        A convenience for the training loop: gather the ``action`` / ``logpi`` /
        ``a_player`` / ``state`` column of a rollout so it can be stacked by mlx / numpy
        by the caller. This container deliberately does NOT stack (it does not know the
        backend of the stored objects); it just extracts the column in step order.
        """
        return [getattr(tr, field) for tr in transitions]

    def clear(self) -> None:
        """Empty the buffer (drop all transitions). The step counter is left monotone."""
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self) -> Iterator:
        return iter(list(self._buf))
