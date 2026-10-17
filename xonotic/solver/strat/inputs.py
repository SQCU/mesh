from typing import NamedTuple

import numpy as np

from payload.tools.strategy_io_schema import EVENT_PUBLIC_TEAM, EVT, OBS

class ChorusArrays(NamedTuple):
    state: np.ndarray
    context: np.ndarray
    observations: np.ndarray
    team_ids: np.ndarray
    residual: np.ndarray
    delta: np.ndarray
    relaxation_time: np.ndarray
    layout: np.ndarray
    context_present: np.ndarray
    participant_present: np.ndarray
    carts: np.ndarray
    teams: np.ndarray
    observation_present: np.ndarray
    cart_present: np.ndarray
    team_present: np.ndarray
    page_headers: np.ndarray
    participant_rows: np.ndarray
    navigation_nodes: np.ndarray
    navigation_edges: np.ndarray
    navigation_cells: np.ndarray
    neighborhood_indices: np.ndarray
    neighborhood_distances: np.ndarray
    neighborhood_present: np.ndarray
    neighborhood_radius: np.ndarray
    navigation_node_present: np.ndarray
    navigation_edge_present: np.ndarray
    navigation_cell_present: np.ndarray
    word_read_times: np.ndarray
    held_velocity: np.ndarray
    command_residual: np.ndarray


def participant_successors(frame, requested):
    from .state_steering import STATE_HEADER
    fields = [STATE_HEADER[name] for name in ('ID', 'OWNER_GENERATION', 'SESSION_HI', 'SESSION_LO')]
    keys = [list(zip(map(tuple, source.page_headers[:, 0, fields]), source.team_ids)) for source in (frame, requested)]
    positions = {key: index for index, key in enumerate(keys[0])}
    return np.asarray([positions.get(key, -1) for key in keys[1]], dtype=np.int64)


def native_inputs(state, observations, carts, teams, context, delta, tau, navigation):
    from .state_steering import unpack_pages
    observations = np.asarray(observations, dtype=np.float32)
    occupied = observations[:, OBS['PRESENT']] > 0
    players = {int(row[OBS['ID']]): index for index, row in enumerate(observations)
               if occupied[index] and row[OBS['TEAM']] >= 1}
    pages = unpack_pages(state, list(players))
    owners = np.asarray([players[int(value)] for value in pages.headers[:, 0]], dtype=np.int32)
    headers = np.zeros((*pages.layout.shape[:2], pages.page_headers.shape[-1]), dtype=np.float32)
    headers[tuple(pages.locations.T)] = pages.page_headers
    event_teams = context[None, :, EVT['TEAM']]
    visibility = occupied[:, None] & ((event_teams == EVENT_PUBLIC_TEAM)
        | (observations[:, OBS['TEAM'], None] == event_teams))
    neighborhood = navigation.prepare(observations[:, OBS['POS_X']:OBS['POS_Z'] + 1],
        context[:, EVT['POS_X']:EVT['POS_Z'] + 1], visibility) if navigation else None
    neighborhood = dict(navigation_nodes=np.zeros((0, 4)), navigation_edges=np.zeros((0, 3)),
        navigation_cells=np.zeros((0, 2)), neighborhood_indices=np.broadcast_to(np.arange(len(context)), visibility.shape),
        neighborhood_distances=np.zeros(visibility.shape), neighborhood_present=visibility,
        neighborhood_radius=np.asarray(1.)) if neighborhood is None else neighborhood
    neighborhood = dict(neighborhood,
        neighborhood_radius=np.broadcast_to(neighborhood['neighborhood_radius'], (len(observations),)).copy())
    return pages, ChorusArrays(
        pages.state, np.asarray(context, dtype=np.float32),
        observations, observations[owners, OBS['TEAM']].astype(np.int64),
        pages.residual, np.asarray(delta, dtype=np.float32), np.asarray(tau, dtype=np.float32), pages.layout,
        np.ones(len(context), dtype=bool), np.ones(len(pages.state), dtype=bool),
        np.asarray(carts, dtype=np.float32), np.asarray(teams, dtype=np.float32), occupied,
        np.ones(len(carts), dtype=bool), np.ones(len(teams), dtype=bool), headers, owners,
        *(np.asarray(neighborhood[name], dtype=dtype) for name, dtype in (
            ('navigation_nodes', np.float32), ('navigation_edges', np.float32), ('navigation_cells', np.float32),
            ('neighborhood_indices', np.int32), ('neighborhood_distances', np.float32),
            ('neighborhood_present', bool), ('neighborhood_radius', np.float32))),
        *(np.ones(len(neighborhood[name]), dtype=bool) for name in ('navigation_nodes', 'navigation_edges', 'navigation_cells')),
        pages.word_read_times, pages.held_velocity, pages.command_residual,
    )


def capacity_for(*frames):
    return tuple(1 << (max(1, max(values)) - 1).bit_length() for values in zip(*(
        (len(frame.state), frame.layout.shape[1], len(frame.context), len(frame.observations), len(frame.carts), len(frame.teams),
         len(frame.navigation_nodes), len(frame.navigation_edges), len(frame.navigation_cells), frame.neighborhood_indices.shape[1])
        for frame in frames)))


def frame_shapes(frame, capacity):
    players, pages, events, observations, carts, teams, nodes, edges, cells, neighbors = capacity
    width = frame.layout.shape[-1] - 3
    return ChorusArrays(
        (players, pages * width), (events, frame.context.shape[-1]),
        (observations, frame.observations.shape[-1]), (players,), (players, pages * width),
        frame.delta.shape, frame.relaxation_time.shape, (players, pages, width + 3),
        (events,), (players,), (carts, frame.carts.shape[-1]), (teams, frame.teams.shape[-1]),
        (observations,), (carts,), (teams,), (players, pages, frame.page_headers.shape[-1]), (players,),
        (nodes, 4), (edges, 3), (cells, 2), (observations, neighbors), (observations, neighbors),
        (observations, neighbors), (observations,), (nodes,), (edges,), (cells,),
        (players, pages * width), (players, pages * width), (players, pages * width),
    )


def fill_frame(target, frame, scratch):
    for name, source, destination in zip(frame._fields, frame, target):
        destination.fill(0)
        if name in ('state', 'residual', 'word_read_times', 'held_velocity', 'command_residual'):
            width = frame.layout.shape[-1] - 3
            source = source.reshape(*frame.layout.shape[:2], width)
            destination = destination.reshape(*target.layout.shape[:2], width)
        destination[tuple(slice(0, size) for size in source.shape)] = source
    shape = frame.neighborhood_indices.shape
    indices = target.neighborhood_indices[:shape[0], :shape[1]]
    mask = scratch[:shape[0], :shape[1]]
    for start, offset in ((len(frame.context), len(target.context) - len(frame.context)),
            (len(frame.context) + len(frame.navigation_nodes), len(target.navigation_nodes) - len(frame.navigation_nodes)),
            (len(frame.context) + len(frame.navigation_nodes) + len(frame.navigation_edges), len(target.navigation_edges) - len(frame.navigation_edges))):
        np.greater_equal(frame.neighborhood_indices, start, out=mask)
        np.add(indices, offset, out=indices, where=mask)
