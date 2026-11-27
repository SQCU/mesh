from __future__ import annotations

import json
import math
import random
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


Q3_MAGIC = b"IBSP"
Q3_VERSION = 46
Q3_NUM_LUMPS = 17

LUMP_ENTITIES = 0
LUMP_TEXTURES = 1
LUMP_PLANES = 2
LUMP_NODES = 3
LUMP_LEAFS = 4
LUMP_LEAFFACES = 5
LUMP_LEAFBRUSHES = 6
LUMP_MODELS = 7
LUMP_BRUSHES = 8
LUMP_BRUSHSIDES = 9
LUMP_VERTICES = 10
LUMP_TRIANGLES = 11
LUMP_EFFECTS = 12
LUMP_FACES = 13
LUMP_LIGHTMAPS = 14
LUMP_LIGHTGRID = 15
LUMP_PVS = 16

CONTENTS_SOLID = 0x1
SURF_NODRAW = 0x80

DEFAULT_STOCK_MAP_POOL = (
    "boil",
    "runningman",
    "warfare",
    "stormkeep",
    "fuse",
    "solarium",
    "opium",
    "trident",
    "leave_em_behind",
    "silentsiege",
    "dance",
    "atelier",
)

STRIP_ENTITY_CLASSES = {
    "info_player_deathmatch",
    "info_player_start",
    "info_player_team",
    "info_player_team1",
    "info_player_team2",
    "info_player_team3",
    "info_player_team4",
    "info_player_team5",
    "info_player_team6",
    "info_player_team7",
    "info_player_team8",
    "info_player_race",
    "trigger_race_checkpoint",
    "dom_team",
    "dom_controlpoint",
    "runematch_spawn_point",
    "info_autoscreenshot",
    "item_flag_team1",
    "item_flag_team2",
    "item_flag_team3",
    "item_flag_team4",
    "ctf_team",
    "plc_team",
    "plc_path",
    "func_plc_cart",
    "onslaught_generator",
    "onslaught_controlpoint",
    "onslaught_link",
    "target_speaker",
    "misc_gamemodel",
    "misc_model",
    "trigger_teleport",
    "info_teleport_destination",
    "misc_teleporter_dest",
}

TARGET_REF_KEYS = ("targetname", "target", "target2", "target3", "target4", "killtarget", "pathtarget")

GAS_THEMES = (
    ("textures/exx/floor-tread01", "textures/exx/wall-bigrib02", "textures/exx/trim-01", "textures/exx/base-metal03"),
    ("textures/exx/base-crete01", "textures/exx/panel-metalbig04", "textures/exx/trim-01", "textures/exx/base-metal03"),
    ("textures/exx/floor-clang01", "textures/exx/panel-metalbig04", "textures/exx/trim-01", "textures/exx/wall-bigrib02"),
)

GAS_ITEMS = (
    "weapon_machinegun",
    "weapon_electro",
    "weapon_devastator",
    "weapon_vortex",
    "weapon_hagar",
    "item_health_medium",
    "item_armor_medium",
)

# Exact NumPy dtypes matching Quake 3 IBSP v46 (model_brush.h)
DT_TEX = np.dtype([("name", "S64"), ("surf", "<i4"), ("cont", "<i4")])
DT_PLANE = np.dtype([("n", "<f4", (3,)), ("d", "<f4")])
DT_NODE = np.dtype([("plane", "<i4"), ("children", "<i4", (2,)), ("mins", "<i4", (3,)), ("maxs", "<i4", (3,))])
DT_LEAF = np.dtype([
    ("cluster", "<i4"), ("area", "<i4"), ("mins", "<i4", (3,)), ("maxs", "<i4", (3,)),
    ("firstleafface", "<i4"), ("numleaffaces", "<i4"), ("firstleafbrush", "<i4"), ("numleafbrushes", "<i4"),
])
DT_MODEL = np.dtype([("mins", "<f4", (3,)), ("maxs", "<f4", (3,)), ("firstface", "<i4"), ("numfaces", "<i4"), ("firstbrush", "<i4"), ("numbrushes", "<i4")])
DT_BRUSH = np.dtype([("firstside", "<i4"), ("numsides", "<i4"), ("tex", "<i4")])
DT_BRUSHSIDE = np.dtype([("plane", "<i4"), ("tex", "<i4")])
DT_VERT = np.dtype([("pos", "<f4", (3,)), ("uv", "<f4", (2,)), ("lm_uv", "<f4", (2,)), ("norm", "<f4", (3,)), ("color", "u1", (4,))])
DT_EFFECT = np.dtype([("name", "S64"), ("brush", "<i4"), ("unknown", "<i4")])
DT_FACE = np.dtype([
    ("tex", "<i4"), ("effect", "<i4"), ("type", "<i4"), ("firstvert", "<i4"), ("numverts", "<i4"),
    ("firsttri", "<i4"), ("numtris", "<i4"), ("lm_idx", "<i4"), ("lm_start", "<i4", (2,)), ("lm_size", "<i4", (2,)),
    ("lm_origin", "<f4", (3,)), ("lm_vecs", "<f4", (2, 3)), ("normal", "<f4", (3,)), ("patch_size", "<i4", (2,)),
])

assert DT_TEX.itemsize == 72
assert DT_PLANE.itemsize == 16
assert DT_NODE.itemsize == 36
assert DT_LEAF.itemsize == 48
assert DT_MODEL.itemsize == 40
assert DT_BRUSH.itemsize == 12
assert DT_BRUSHSIDE.itemsize == 8
assert DT_VERT.itemsize == 44
assert DT_EFFECT.itemsize == 72
assert DT_FACE.itemsize == 104


def _rot_xy_arr(xy: np.ndarray, rot: int) -> np.ndarray:
    r = rot & 3
    if r == 0:
        return xy.copy()
    x, y = xy[..., 0], xy[..., 1]
    if r == 1:
        return np.stack((-y, x), axis=-1)
    if r == 2:
        return np.stack((-x, -y), axis=-1)
    return np.stack((y, -x), axis=-1)


def _rot_vec3_arr(v: np.ndarray, rot: int) -> np.ndarray:
    if (rot & 3) == 0:
        return v.copy()
    out = v.copy()
    out[..., :2] = _rot_xy_arr(v[..., :2], rot)
    return out


def _rot_vec3(v: tuple[float, float, float], rot: int) -> tuple[float, float, float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    r = rot & 3
    if r == 1:
        return (-y, x, z)
    if r == 2:
        return (-x, -y, z)
    if r == 3:
        return (y, -x, z)
    return (x, y, z)


def _rot_aabb_arr(mins: np.ndarray, maxs: np.ndarray, rot: int) -> tuple[np.ndarray, np.ndarray]:
    r = rot & 3
    if r == 0:
        return mins.copy(), maxs.copy()
    x0, y0, z0 = mins[..., 0], mins[..., 1], mins[..., 2]
    x1, y1, z1 = maxs[..., 0], maxs[..., 1], maxs[..., 2]
    if r == 1:
        return np.stack((-y1, x0, z0), axis=-1), np.stack((-y0, x1, z1), axis=-1)
    if r == 2:
        return np.stack((-x1, -y1, z0), axis=-1), np.stack((-x0, -y0, z1), axis=-1)
    return np.stack((y0, -x1, z0), axis=-1), np.stack((y1, -x0, z1), axis=-1)


def _parse_entities_lump(text: str) -> list[dict[str, str]]:
    ents: list[dict[str, str]] = []
    for blk in re.findall(r"\{[^{}]*\}", text):
        kv: dict[str, str] = {}
        for k, v in re.findall(r'"([^"]+)"\s+"([^"]*)"', blk):
            kv[k] = v
        if kv:
            ents.append(kv)
    return ents


def _format_entities_lump(ents: list[dict[str, str]]) -> bytes:
    lines: list[str] = []
    for kv in ents:
        lines.append("{")
        for k, v in kv.items():
            lines.append(f'"{k}" "{v}"')
        lines.append("}")
    return ("\n".join(lines) + "\n\x00").encode("utf-8")


def _parse_vec3_str(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split()]
    if len(parts) < 3:
        parts = (parts + [0.0, 0.0, 0.0])[:3]
    return (parts[0], parts[1], parts[2])


def _fmt_vec3(p: tuple[float, float, float]) -> str:
    return f"{p[0]:.1f} {p[1]:.1f} {p[2]:.1f}"


def _parse_waypoints_text(text: str) -> list[tuple[tuple[float, float, float], tuple[float, float, float], int]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    i = 0
    while i + 2 < len(lines):
        try:
            a = _parse_vec3_str(lines[i])
            b = _parse_vec3_str(lines[i + 1])
            flags = int(float(lines[i + 2]))
            out.append((a, b, flags))
            i += 3
        except ValueError:
            i += 1
    return out


def _format_waypoints_bytes(wps: list[tuple[tuple[float, float, float], tuple[float, float, float], int]]) -> bytes:
    lines: list[str] = []
    for a, b, fl in wps:
        lines.append(f"{a[0]:.1f} {a[1]:.1f} {a[2]:.1f}")
        lines.append(f"{b[0]:.1f} {b[1]:.1f} {b[2]:.1f}")
        lines.append(str(int(fl)))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _format_waypoints_cache_bytes(
    wps: list[tuple[tuple[float, float, float], tuple[float, float, float], int]],
    edges: list[tuple[int, int]],
) -> bytes:
    adj: dict[int, set[int]] = {i: set() for i in range(len(wps))}
    for a, b in edges:
        if 0 <= a < len(wps) and 0 <= b < len(wps) and a != b:
            adj[a].add(b)
            adj[b].add(a)
    lines = ["//version 1.01", "0"]
    for i in range(len(wps)):
        nbs = sorted(adj[i])
        lines.append(" ".join(str(j) for j in nbs))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_expander_edges(M: int, rng: random.Random, min_deg: int = 3) -> list[tuple[int, int]]:
    if M <= 1:
        return []
    if M == 2:
        return [(0, 1)]
    if M == 3:
        return [(0, 1), (1, 2), (0, 2)]
    target_d = min(M - 1, max(3, min_deg))
    edges: set[tuple[int, int]] = set()
    for i in range(M):
        a, b = i, (i + 1) % M
        edges.add((min(a, b), max(a, b)))
    deg = [2] * M
    cands = []
    for i in range(M):
        for j in range(i + 2, M):
            if i == 0 and j == M - 1:
                continue
            dist_ring = min(j - i, M - (j - i))
            cands.append((dist_ring, rng.random(), i, j))
    cands.sort(key=lambda x: (-x[0], x[1]))
    for _, _, i, j in cands:
        if (deg[i] < target_d or deg[j] < target_d) and deg[i] < target_d + 1 and deg[j] < target_d + 1:
            edges.add((i, j))
            deg[i] += 1
            deg[j] += 1
        if all(d >= target_d for d in deg):
            break
    for i in range(M):
        if deg[i] < target_d:
            for j in range(M):
                if j != i and (min(i, j), max(i, j)) not in edges:
                    edges.add((min(i, j), max(i, j)))
                    deg[i] += 1
                    deg[j] += 1
                    if deg[i] >= target_d:
                        break
    return sorted(edges)


def _pack_ibsp(lump_bytes: list[bytes]) -> bytes:
    assert len(lump_bytes) == Q3_NUM_LUMPS
    header_size = 8 + Q3_NUM_LUMPS * 8
    cur = (header_size + 3) & ~3
    dir_entries = []
    padded = []
    for b in lump_bytes:
        dir_entries.append((cur, len(b)))
        pad = (-len(b)) & 3
        padded.append(b + (b"\x00" * pad))
        cur += len(b) + pad
    hdr = bytearray(header_size)
    struct.pack_into("<4si", hdr, 0, Q3_MAGIC, Q3_VERSION)
    for i, (ofs, ln) in enumerate(dir_entries):
        struct.pack_into("<ii", hdr, 8 + i * 8, ofs, ln)
    pad_hdr = b"\x00" * (((header_size + 3) & ~3) - header_size)
    return b"".join([bytes(hdr), pad_hdr, *padded])


@dataclass
class ParsedStockBSP:
    name: str
    entities: list[dict[str, str]]
    textures: np.ndarray
    planes: np.ndarray
    nodes: np.ndarray
    leafs: np.ndarray
    leaffaces: np.ndarray
    leafbrushes: np.ndarray
    models: np.ndarray
    brushes: np.ndarray
    brushsides: np.ndarray
    verts: np.ndarray
    tris: np.ndarray
    effects: np.ndarray
    faces: np.ndarray
    lightmaps: bytes
    waypoints: list[tuple[tuple[float, float, float], tuple[float, float, float], int]]
    walk_pts: list[tuple[float, float, float]]
    spawns: list[tuple[tuple[float, float, float], float]]
    aabb_min: np.ndarray
    aabb_max: np.ndarray


def _load_stock_bsp_from_pk3(pk3_path: Path, map_name: str) -> ParsedStockBSP:
    with zipfile.ZipFile(pk3_path, "r") as z:
        raw = z.read(f"maps/{map_name}.bsp")
        wp_text = ""
        wp_key = f"maps/{map_name}.waypoints"
        if wp_key in z.namelist():
            wp_text = z.read(wp_key).decode("utf-8", errors="ignore")

    magic, ver = struct.unpack_from("<4si", raw, 0)
    if magic != Q3_MAGIC or ver != Q3_VERSION:
        raise ValueError(f"Unsupported BSP {map_name}: {magic!r} v{ver}")
    lumps = [struct.unpack_from("<ii", raw, 8 + i * 8) for i in range(Q3_NUM_LUMPS)]

    def _slice(idx: int) -> bytes:
        ofs, ln = lumps[idx]
        return raw[ofs : ofs + ln]

    ent_text = _slice(LUMP_ENTITIES).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    ents = _parse_entities_lump(ent_text)
    textures = np.frombuffer(_slice(LUMP_TEXTURES), dtype=DT_TEX).copy()
    planes = np.frombuffer(_slice(LUMP_PLANES), dtype=DT_PLANE).copy()
    nodes = np.frombuffer(_slice(LUMP_NODES), dtype=DT_NODE).copy()
    leafs = np.frombuffer(_slice(LUMP_LEAFS), dtype=DT_LEAF).copy()
    leaffaces = np.frombuffer(_slice(LUMP_LEAFFACES), dtype="<i4").copy()
    leafbrushes = np.frombuffer(_slice(LUMP_LEAFBRUSHES), dtype="<i4").copy()
    models = np.frombuffer(_slice(LUMP_MODELS), dtype=DT_MODEL).copy()
    brushes = np.frombuffer(_slice(LUMP_BRUSHES), dtype=DT_BRUSH).copy()
    brushsides = np.frombuffer(_slice(LUMP_BRUSHSIDES), dtype=DT_BRUSHSIDE).copy()
    verts = np.frombuffer(_slice(LUMP_VERTICES), dtype=DT_VERT).copy()
    tris = np.frombuffer(_slice(LUMP_TRIANGLES), dtype="<i4").copy()
    effects = np.frombuffer(_slice(LUMP_EFFECTS), dtype=DT_EFFECT).copy()
    faces = np.frombuffer(_slice(LUMP_FACES), dtype=DT_FACE).copy()
    lightmaps = _slice(LUMP_LIGHTMAPS)

    wps = _parse_waypoints_text(wp_text)
    walk_pts = [
        ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)
        for a, b, fl in wps
        if (fl & 0x3F) == 0
    ]
    spawns: list[tuple[tuple[float, float, float], float]] = []
    for e in ents:
        cn = e.get("classname", "")
        if cn.startswith("info_player_") and "origin" in e:
            spawns.append((_parse_vec3_str(e["origin"]), float(e.get("angle", "0"))))
    if not walk_pts:
        walk_pts = [p for p, _ in spawns]
    if not spawns and walk_pts:
        spawns = [(p, 0.0) for p in walk_pts[:8]]

    aabb_min = models[0]["mins"].astype(np.float64)
    aabb_max = models[0]["maxs"].astype(np.float64)
    return ParsedStockBSP(
        name=map_name,
        entities=ents,
        textures=textures,
        planes=planes,
        nodes=nodes,
        leafs=leafs,
        leaffaces=leaffaces,
        leafbrushes=leafbrushes,
        models=models,
        brushes=brushes,
        brushsides=brushsides,
        verts=verts,
        tris=tris,
        effects=effects,
        faces=faces,
        lightmaps=lightmaps,
        waypoints=wps,
        walk_pts=walk_pts,
        spawns=spawns,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )


def _grid_dims(M: int) -> tuple[int, int]:
    best = (M, 1)
    best_score = (M, abs(M - 1))
    for r in range(1, int(math.sqrt(M)) + 2):
        c = int(math.ceil(M / float(r)))
        score = (c * r - M, abs(c - r))
        if score < best_score:
            best_score = score
            best = (c, r)
    return best


class B3313BinaryStitcher:
    """Zero-q3map2 in-memory binary IBSP v46 supermap stitcher."""

    def __init__(self, maps_pk3: str | Path):
        p = Path(maps_pk3).expanduser()
        if not p.exists():
            for alt in ("xonotic-maps-all29-ded.pk3", "xonotic-20230620-maps.pk3"):
                cand = p.parent / alt
                if cand.exists():
                    p = cand
                    break
        self.maps_pk3 = p
        self._cache: dict[str, ParsedStockBSP] = {}

    def get_stock(self, map_name: str) -> ParsedStockBSP:
        if map_name not in self._cache:
            self._cache[map_name] = _load_stock_bsp_from_pk3(self.maps_pk3, map_name)
        return self._cache[map_name]

    def stitch(
        self,
        *,
        map_name: str,
        seed: int,
        teams: int,
        carts: int,
        players: int,
        tiles: int,
        stock_pool: tuple[str, ...] = DEFAULT_STOCK_MAP_POOL,
        expander_min_degree: int = 3,
        allow_rotations: bool = True,
        checkpoints_per_cart: int = 4,
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        rng = random.Random(seed)
        K = max(2, min(24, int(teams)))
        J = max(K + 1, min(256, int(carts)))
        M = max(1, min(24, int(tiles)))
        N = max(max(2 * K, 12), min(240, int(players)))

        pool = list(stock_pool) or list(DEFAULT_STOCK_MAP_POOL)
        chosen: list[tuple[ParsedStockBSP, int]] = []
        for ti in range(M):
            src = self.get_stock(pool[(seed + ti * 7 + rng.randrange(len(pool))) % len(pool)])
            rot = rng.randrange(4) if allow_rotations else 0
            chosen.append((src, rot))

        # Compute rotated local AABBs and pack cleanly onto a disjoint 2D grid centered near origin
        rot_aabbs = []
        max_span_x = 0.0
        max_span_y = 0.0
        for src, rot in chosen:
            mn, mx = _rot_aabb_arr(src.aabb_min, src.aabb_max, rot)
            rot_aabbs.append((mn, mx))
            max_span_x = max(max_span_x, float(mx[0] - mn[0]))
            max_span_y = max(max_span_y, float(mx[1] - mn[1]))

        gap = 512.0
        pitch_x = math.ceil((max_span_x + gap) / 64.0) * 64.0
        pitch_y = math.ceil((max_span_y + gap) / 64.0) * 64.0
        cols, rows = _grid_dims(M)
        grid_w = cols * pitch_x
        grid_h = rows * pitch_y

        translations: list[np.ndarray] = []
        world_aabbs: list[tuple[np.ndarray, np.ndarray]] = []
        for ti, (mn, mx) in enumerate(rot_aabbs):
            cx_i = ti % cols
            cy_i = ti // cols
            cell_cx = (cx_i + 0.5) * pitch_x - 0.5 * grid_w
            cell_cy = (cy_i + 0.5) * pitch_y - 0.5 * grid_h
            loc_c = 0.5 * (mn + mx)
            tx = float(round(cell_cx - loc_c[0]))
            ty = float(round(cell_cy - loc_c[1]))
            tz = float(round(-loc_c[2]))
            T = np.array([tx, ty, tz], dtype=np.float64)
            translations.append(T)
            world_aabbs.append((mn + T, mx + T))

        # 1. Merge textures, planes, brushsides, vertices, tris, lightmaps; separate world vs submodel faces & brushes
        tex_list, plane_list, bs_list, vert_list, tri_list, eff_list, lm_chunks = [], [], [], [], [], [], []
        world_faces_list, sub_faces_list = [], []
        world_brushes_list, sub_brushes_list = [], []
        leaf_list, leafface_list, leafbrush_list = [], [], []
        submap_nodes_list = []
        kept_submodels_meta: list[tuple[np.ndarray, np.ndarray, int, int, int, int]] = []
        submodel_remap: list[dict[int, int]] = []

        tex_off = plane_off = bs_off = vert_off = tri_off = eff_off = lm_off = 0
        # First pass: compute global world_face / world_brush totals so submodel faces/brushes land right after
        total_world_faces = sum(int(src.models[0]["numfaces"]) for src, _ in chosen)
        total_world_brushes = sum(int(src.models[0]["numbrushes"]) for src, _ in chosen)

        w_face_cursor = 0
        s_face_cursor = total_world_faces
        w_brush_cursor = 0
        s_brush_cursor = total_world_brushes
        next_submodel_idx = 1

        # Reserve M-1 top-level BSP split nodes at indices [0, M-1)
        top_nodes_count = max(0, M - 1)
        node_cursor = top_nodes_count
        leaf_cursor = 0
        lf_cursor = 0
        lb_cursor = 0
        submap_root_nodes: list[int] = []

        for ti, ((src, rot), T) in enumerate(zip(chosen, translations)):
            T_f32 = T.astype(np.float32)
            T_i32 = np.round(T).astype(np.int32)

            # Textures
            tex_list.append(src.textures)

            # Planes: rotate normal, shift dist += dot(normal_rot, T)
            pl = src.planes.copy()
            pl["n"] = _rot_vec3_arr(pl["n"], rot)
            pl["d"] = pl["d"] + (pl["n"].astype(np.float64) @ T).astype(np.float32)
            plane_list.append(pl)

            # Brushsides
            bs = src.brushsides.copy()
            bs["plane"] += plane_off
            bs["tex"] += tex_off
            bs_list.append(bs)

            # Vertices
            vt = src.verts.copy()
            vt["pos"] = _rot_vec3_arr(vt["pos"], rot) + T_f32
            vt["norm"] = _rot_vec3_arr(vt["norm"], rot)
            vert_list.append(vt)

            # Triangles
            tri_list.append(src.tris)

            # Faces: build old->new remap putting world faces in [0, total_world_faces) and submodel faces after
            m0 = src.models[0]
            wf0, wfn = int(m0["firstface"]), int(m0["numfaces"])
            wb0, wbn = int(m0["firstbrush"]), int(m0["numbrushes"])

            fc = src.faces.copy()
            fc["tex"] += tex_off
            fc["effect"] = np.where(fc["effect"] >= 0, fc["effect"] + eff_off, -1)
            fc["firstvert"] += vert_off
            fc["firsttri"] += tri_off
            if len(src.lightmaps) > 0:
                fc["lm_idx"] = np.where(fc["lm_idx"] >= 0, fc["lm_idx"] + lm_off, -1)
            fc["lm_origin"] = _rot_vec3_arr(fc["lm_origin"], rot) + T_f32
            fc["lm_vecs"] = _rot_vec3_arr(fc["lm_vecs"], rot)
            fc["normal"] = _rot_vec3_arr(fc["normal"], rot)

            face_remap = np.full(len(fc), -1, dtype=np.int32)
            world_faces_list.append(fc[wf0 : wf0 + wfn])
            face_remap[wf0 : wf0 + wfn] = np.arange(w_face_cursor, w_face_cursor + wfn, dtype=np.int32)
            w_face_cursor += wfn

            br = src.brushes.copy()
            br["firstside"] += bs_off
            br["tex"] += tex_off
            brush_remap = np.full(len(br), -1, dtype=np.int32)
            world_brushes_list.append(br[wb0 : wb0 + wbn])
            brush_remap[wb0 : wb0 + wbn] = np.arange(w_brush_cursor, w_brush_cursor + wbn, dtype=np.int32)
            w_brush_cursor += wbn

            # Submodels *1.. of this stock map (e.g. jumppad triggers, func_wall)
            sm_map: dict[int, int] = {}
            for sm_i in range(1, len(src.models)):
                sm = src.models[sm_i]
                sf0, sfn = int(sm["firstface"]), int(sm["numfaces"])
                sb0, sbn = int(sm["firstbrush"]), int(sm["numbrushes"])
                new_sf0 = s_face_cursor
                if sfn > 0:
                    sub_faces_list.append(fc[sf0 : sf0 + sfn])
                    face_remap[sf0 : sf0 + sfn] = np.arange(s_face_cursor, s_face_cursor + sfn, dtype=np.int32)
                    s_face_cursor += sfn
                new_sb0 = s_brush_cursor
                if sbn > 0:
                    sub_brushes_list.append(br[sb0 : sb0 + sbn])
                    brush_remap[sb0 : sb0 + sbn] = np.arange(s_brush_cursor, s_brush_cursor + sbn, dtype=np.int32)
                    s_brush_cursor += sbn
                sm_mn, sm_mx = _rot_aabb_arr(sm["mins"], sm["maxs"], rot)
                kept_submodels_meta.append((sm_mn + T_f32, sm_mx + T_f32, new_sf0, sfn, new_sb0, sbn))
                sm_map[sm_i] = next_submodel_idx
                next_submodel_idx += 1
            submodel_remap.append(sm_map)

            # Effects
            if len(src.effects) > 0:
                ef = src.effects.copy()
                valid_b = (ef["brush"] >= 0) & (ef["brush"] < len(brush_remap))
                ef["brush"] = np.where(valid_b, brush_remap[np.clip(ef["brush"], 0, max(0, len(brush_remap) - 1))], -1)
                eff_list.append(ef)

            # Leaffaces & Leafbrushes remapped
            lf = face_remap[src.leaffaces]
            lb = brush_remap[src.leafbrushes]
            leafface_list.append(lf)
            leafbrush_list.append(lb)

            # Leafs (cluster = -1 so empty PVS renders/collides cleanly everywhere)
            lfa = src.leafs.copy()
            lfa["cluster"] = -1
            lfa["area"] = 0
            l_mn, l_mx = _rot_aabb_arr(lfa["mins"], lfa["maxs"], rot)
            lfa["mins"] = l_mn + T_i32
            lfa["maxs"] = l_mx + T_i32
            lfa["firstleafface"] += lf_cursor
            lfa["firstleafbrush"] += lb_cursor
            leaf_list.append(lfa)

            # Nodes
            nd = src.nodes.copy()
            nd["plane"] += plane_off
            ch = nd["children"]
            nd["children"] = np.where(ch >= 0, ch + node_cursor, ch - leaf_cursor)
            n_mn, n_mx = _rot_aabb_arr(nd["mins"], nd["maxs"], rot)
            nd["mins"] = n_mn + T_i32
            nd["maxs"] = n_mx + T_i32
            submap_nodes_list.append(nd)
            submap_root_nodes.append(node_cursor)

            if len(src.lightmaps) > 0:
                lm_chunks.append(src.lightmaps)
                lm_off += len(src.lightmaps) // (128 * 128 * 3)

            tex_off += len(src.textures)
            plane_off += len(src.planes)
            bs_off += len(src.brushsides)
            vert_off += len(src.verts)
            tri_off += len(src.tris)
            eff_off += len(src.effects)
            node_cursor += len(src.nodes)
            leaf_cursor += len(src.leafs)
            lf_cursor += len(lf)
            lb_cursor += len(lb)

        # Append 1 common/trigger texture for synthesized portal boxes
        trig_tex = np.zeros(1, dtype=DT_TEX)
        trig_tex[0]["name"] = b"textures/common/trigger"
        trig_tex[0]["surf"] = SURF_NODRAW
        trig_tex[0]["cont"] = CONTENTS_SOLID
        trig_tex_idx = tex_off
        tex_list.append(trig_tex)
        tex_off += 1

        # 2. Build top-level axis-aligned BSP split tree over the M disjoint submap AABBs
        top_planes: list[np.ndarray] = []
        top_nodes = np.zeros(top_nodes_count, dtype=DT_NODE)
        top_node_alloc = [0]

        def _build_top_bsp(indices: list[int]) -> int:
            if len(indices) == 1:
                return submap_root_nodes[indices[0]]
            nid = top_node_alloc[0]
            top_node_alloc[0] += 1
            centers = np.array([0.5 * (world_aabbs[i][0] + world_aabbs[i][1]) for i in indices])
            spread = centers.max(axis=0) - centers.min(axis=0)
            axis = 0 if spread[0] >= spread[1] else 1
            ordered = sorted(indices, key=lambda i: float(centers[indices.index(i), axis]))
            mid = len(ordered) // 2
            left_ids = ordered[:mid]
            right_ids = ordered[mid:]
            left_max = max(float(world_aabbs[i][1][axis]) for i in left_ids)
            right_min = min(float(world_aabbs[i][0][axis]) for i in right_ids)
            split_d = 0.5 * (left_max + right_min)

            p_idx = plane_off + len(top_planes)
            pl = np.zeros(1, dtype=DT_PLANE)
            pl[0]["n"][axis] = 1.0
            pl[0]["d"] = float(split_d)
            top_planes.append(pl)

            # Q3 BSP convention: children[0] = front (coord >= split_d -> right_ids), children[1] = back (left_ids)
            c_right = _build_top_bsp(right_ids)
            c_left = _build_top_bsp(left_ids)
            u_min = np.min([world_aabbs[i][0] for i in indices], axis=0)
            u_max = np.max([world_aabbs[i][1] for i in indices], axis=0)
            top_nodes[nid]["plane"] = p_idx
            top_nodes[nid]["children"] = (c_right, c_left)
            top_nodes[nid]["mins"] = np.floor(u_min).astype(np.int32)
            top_nodes[nid]["maxs"] = np.ceil(u_max).astype(np.int32)
            return nid

        if M > 1:
            _build_top_bsp(list(range(M)))
            plane_list.extend(top_planes)
            plane_off += len(top_planes)

        # 3. Synthesize entities, portals (with brush submodels), spawns, carts, and waypoints
        all_ents: list[dict[str, str]] = [{"classname": "worldspawn", "message": map_name}]
        all_wps: list[tuple[tuple[float, float, float], tuple[float, float, float], int]] = []
        wp_edges: list[tuple[int, int]] = []
        tile_walk_pts: list[list[tuple[float, float, float]]] = []
        tile_spawns: list[list[tuple[tuple[float, float, float], float]]] = []

        for ti, ((src, rot), T) in enumerate(zip(chosen, translations)):
            prefix = f"t{ti}_"
            sm_map = submodel_remap[ti]
            for e in src.entities:
                cn = e.get("classname", "")
                if not cn or cn == "worldspawn" or cn in STRIP_ENTITY_CLASSES:
                    continue
                ne = dict(e)
                if "model" in ne and ne["model"].startswith("*"):
                    old_m = int(ne["model"][1:])
                    if old_m not in sm_map:
                        continue
                    ne["model"] = f"*{sm_map[old_m]}"
                if "origin" in ne:
                    p = _rot_vec3(_parse_vec3_str(ne["origin"]), rot)
                    ne["origin"] = _fmt_vec3((p[0] + T[0], p[1] + T[1], p[2] + T[2]))
                if "angle" in ne:
                    try:
                        ne["angle"] = f"{(float(ne['angle']) + 90.0 * rot) % 360.0:.1f}"
                    except ValueError:
                        pass
                for rk in TARGET_REF_KEYS:
                    if rk in ne and ne[rk]:
                        ne[rk] = prefix + ne[rk]
                all_ents.append(ne)

            # Transform waypoints & spawns for this tile
            t_walks: list[tuple[float, float, float]] = []
            base_wp = len(all_wps)
            for a, b, fl in src.waypoints:
                ra = _rot_vec3(a, rot)
                rb = _rot_vec3(b, rot)
                wa = (ra[0] + T[0], ra[1] + T[1], ra[2] + T[2])
                wb = (rb[0] + T[0], rb[1] + T[1], rb[2] + T[2])
                all_wps.append((wa, wb, fl))
                if (fl & 0x3F) == 0:
                    t_walks.append(((wa[0] + wb[0]) * 0.5, (wa[1] + wb[1]) * 0.5, (wa[2] + wb[2]) * 0.5))
            # Link local same-level k-NN waypoints cleanly
            n_loc = len(all_wps) - base_wp
            if n_loc > 1:
                pts_loc = np.array([0.5 * (np.array(all_wps[base_wp + i][0]) + np.array(all_wps[base_wp + i][1])) for i in range(n_loc)])
                for i in range(n_loc):
                    dxy = np.hypot(pts_loc[:, 0] - pts_loc[i, 0], pts_loc[:, 1] - pts_loc[i, 1])
                    dz = np.abs(pts_loc[:, 2] - pts_loc[i, 2])
                    ok = np.where((dxy > 16.0) & (dxy <= 384.0) & (dz <= 48.0))[0]
                    if len(ok):
                        order = ok[np.argsort(dxy[ok])[:4]]
                        for j in order:
                            wp_edges.append((base_wp + i, base_wp + int(j)))

            t_sps = []
            for p, ang in src.spawns:
                rp = _rot_vec3(p, rot)
                t_sps.append(((rp[0] + T[0], rp[1] + T[1], rp[2] + T[2]), (ang + 90.0 * rot) % 360.0))
            if not t_walks:
                t_walks = [p for p, _ in t_sps]
            tile_walk_pts.append(t_walks)
            tile_spawns.append(t_sps)

        # Synthesize non-Euclidean portal expander edges (trigger_teleport brush boxes + info_teleport_destination)
        portal_planes: list[np.ndarray] = []
        portal_bs: list[np.ndarray] = []
        portal_brushes: list[np.ndarray] = []
        joints_meta: list[dict[str, Any]] = []
        edges_exp = _build_expander_edges(M, rng, min_deg=expander_min_degree)
        used_portal_pts: list[list[tuple[float, float, float]]] = [[] for _ in range(M)]

        def _pick_portal_pair(ti: int) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
            pts = tile_walk_pts[ti]
            used = used_portal_pts[ti]
            cands = list(pts)
            rng.shuffle(cands)
            for p in cands:
                if any(math.hypot(p[0] - u[0], p[1] - u[1]) < 280.0 for u in used):
                    continue
                exits = [
                    q for q in pts
                    if 110.0 <= math.hypot(q[0] - p[0], q[1] - p[1]) <= 280.0 and abs(q[2] - p[2]) <= 32.0
                ]
                if exits:
                    q = exits[0]
                    used.append(p)
                    ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 360.0
                    return p, q, ang
            p = pts[len(used) % len(pts)]
            used.append(p)
            q = (p[0] + 128.0, p[1], p[2])
            return p, q, 0.0

        def _append_trigger_box(center: tuple[float, float, float]) -> int:
            nonlocal plane_off, bs_off, s_brush_cursor, next_submodel_idx
            cx, cy, cz = center
            mn = np.array([cx - 36.0, cy - 36.0, cz - 16.0], dtype=np.float32)
            mx = np.array([cx + 36.0, cy + 36.0, cz + 96.0], dtype=np.float32)
            pls = np.zeros(6, dtype=DT_PLANE)
            pls[0] = ((1.0, 0.0, 0.0), float(mx[0]))
            pls[1] = ((-1.0, 0.0, 0.0), float(-mn[0]))
            pls[2] = ((0.0, 1.0, 0.0), float(mx[1]))
            pls[3] = ((0.0, -1.0, 0.0), float(-mn[1]))
            pls[4] = ((0.0, 0.0, 1.0), float(mx[2]))
            pls[5] = ((0.0, 0.0, -1.0), float(-mn[2]))
            p0 = plane_off
            portal_planes.append(pls)
            plane_off += 6

            bss = np.zeros(6, dtype=DT_BRUSHSIDE)
            for k in range(6):
                bss[k]["plane"] = p0 + k
                bss[k]["tex"] = trig_tex_idx
            bs0 = bs_off
            portal_bs.append(bss)
            bs_off += 6

            br = np.zeros(1, dtype=DT_BRUSH)
            br[0]["firstside"] = bs0
            br[0]["numsides"] = 6
            br[0]["tex"] = trig_tex_idx
            b0 = s_brush_cursor
            portal_brushes.append(br)
            s_brush_cursor += 1

            sm_idx = next_submodel_idx
            kept_submodels_meta.append((mn, mx, s_face_cursor, 0, b0, 1))
            next_submodel_idx += 1
            return sm_idx

        for eid, (u, v) in enumerate(edges_exp):
            pu, qu, angu = _pick_portal_pair(u)
            pv, qv, angv = _pick_portal_pair(v)
            sm_u = _append_trigger_box(pu)
            sm_v = _append_trigger_box(pv)
            du_name = f"b3313_dest_{eid}_u"
            dv_name = f"b3313_dest_{eid}_v"
            all_ents.append({"classname": "info_teleport_destination", "targetname": dv_name, "origin": _fmt_vec3((qv[0], qv[1], qv[2] + 8.0)), "angle": f"{angv:.1f}"})
            all_ents.append({"classname": "info_teleport_destination", "targetname": du_name, "origin": _fmt_vec3((qu[0], qu[1], qu[2] + 8.0)), "angle": f"{angu:.1f}"})
            all_ents.append({"classname": "trigger_teleport", "model": f"*{sm_u}", "target": dv_name, "spawnflags": "4"})
            all_ents.append({"classname": "trigger_teleport", "model": f"*{sm_v}", "target": du_name, "spawnflags": "4"})
            joints_meta.append({"id": eid, "tile_a": u, "tile_b": v, "portal_a": list(pu), "portal_b": list(pv)})

        # Emit K plc_team + 4*K info_player_team spawns distributed across tiles
        for t in range(K):
            all_ents.append({"classname": "plc_team", "cnt": str(t)})
        for t in range(K):
            ti = t % M
            sps = tile_spawns[ti]
            for s_i in range(4):
                p, ang = sps[(t // M * 4 + s_i) % len(sps)]
                all_ents.append({"classname": "info_player_team", "origin": _fmt_vec3(p), "angle": f"{ang:.1f}", "team": str(t + 1)})

        # Emit J plc_path + func_plc_cart tracks along real walkable nodes in each tile
        ncp = max(1, int(checkpoints_per_cart))
        for ci in range(J):
            ti = ci % M
            pts = tile_walk_pts[ti]
            start = pts[(ci * 5) % len(pts)]
            # Pick a far same-floor walkable point inside this tile for a clean track
            far = max(pts, key=lambda q: (math.hypot(q[0] - start[0], q[1] - start[1]) if abs(q[2] - start[2]) <= 64.0 else -1.0))
            if math.hypot(far[0] - start[0], far[1] - start[1]) < 128.0:
                far = (start[0] + 480.0, start[1], start[2])
            nodes = [
                (
                    start[0] + (far[0] - start[0]) * k / float(ncp),
                    start[1] + (far[1] - start[1]) * k / float(ncp),
                    start[2] + (far[2] - start[2]) * k / float(ncp),
                )
                for k in range(ncp + 1)
            ]
            names = [f"plc{ci}n{ni}" for ni in range(len(nodes))]
            for ni, p in enumerate(nodes):
                kv = {"classname": "plc_path", "targetname": names[ni], "origin": _fmt_vec3(p)}
                if ni + 1 < len(nodes):
                    kv["target"] = names[ni + 1]
                if ni > 0:
                    kv["spawnflags"] = "1"
                all_ents.append(kv)
            all_ents.append({
                "classname": "func_plc_cart",
                "origin": _fmt_vec3(nodes[0]),
                "target": names[0],
                "radius": "160",
                "height": "96",
                "plc_speed_scale": "1",
            })

        # 4. Assemble final lumps
        plane_list.extend(portal_planes)
        bs_list.extend(portal_bs)
        sub_brushes_list.extend(portal_brushes)

        g_min = np.min([a[0] for a in world_aabbs], axis=0).astype(np.float32)
        g_max = np.max([a[1] for a in world_aabbs], axis=0).astype(np.float32)
        models_arr = np.zeros(1 + len(kept_submodels_meta), dtype=DT_MODEL)
        models_arr[0] = (g_min, g_max, 0, total_world_faces, 0, total_world_brushes)
        for idx_sm, (sm_mn, sm_mx, sf0, sfn, sb0, sbn) in enumerate(kept_submodels_meta, start=1):
            models_arr[idx_sm] = (sm_mn, sm_mx, sf0, sfn, sb0, sbn)

        all_nodes = np.concatenate(([top_nodes] if top_nodes_count > 0 else []) + submap_nodes_list)
        all_faces = np.concatenate(world_faces_list + sub_faces_list) if (world_faces_list or sub_faces_list) else np.zeros(0, dtype=DT_FACE)
        all_brushes = np.concatenate(world_brushes_list + sub_brushes_list) if (world_brushes_list or sub_brushes_list) else np.zeros(0, dtype=DT_BRUSH)
        all_effects = np.concatenate(eff_list) if eff_list else np.zeros(0, dtype=DT_EFFECT)
        ent_bytes = _format_entities_lump(all_ents)

        bsp_bytes = _pack_ibsp([
            ent_bytes,
            np.concatenate(tex_list).tobytes(),
            np.concatenate(plane_list).tobytes(),
            all_nodes.tobytes(),
            np.concatenate(leaf_list).tobytes(),
            np.concatenate(leafface_list).tobytes(),
            np.concatenate(leafbrush_list).tobytes(),
            models_arr.tobytes(),
            all_brushes.tobytes(),
            np.concatenate(bs_list).tobytes(),
            np.concatenate(vert_list).tobytes(),
            np.concatenate(tri_list).tobytes(),
            all_effects.tobytes(),
            all_faces.tobytes(),
            b"".join(lm_chunks),
            b"",
            b"",
        ])

        mapinfo_bytes = (
            f"title {map_name}\n"
            f"description Parametric B3313 binary-stitched supermap (K={K}, J={J}, N={N}, M={M})\n"
            f"author mesh-bsp-stream\n"
            f"gametype dm\n"
            f"gametype tdm\n"
            f"gametype plc\n"
        ).encode("utf-8")

        spec = {
            "name": map_name,
            "family": "b3313_binary_stitch",
            "seed": int(seed),
            "teams": K,
            "carts": J,
            "players": N,
            "tiles": M,
            "maxclients": min(255, N + 15),
            "submaps": [src.name for src, _ in chosen],
            "bsp_bytes": len(bsp_bytes),
        }
        vfs_files = {
            f"maps/{map_name}.bsp": bsp_bytes,
            f"maps/{map_name}.ent": ent_bytes[:-1],
            f"maps/{map_name}.waypoints": _format_waypoints_bytes(all_wps),
            f"maps/{map_name}.waypoints.cache": _format_waypoints_cache_bytes(all_wps, wp_edges),
            f"maps/{map_name}.mapinfo": mapinfo_bytes,
            f"maps/{map_name}.joints.json": json.dumps({"spec": spec, "joints": joints_meta}).encode("utf-8"),
        }
        return spec, vfs_files


def _compile_boxes_to_ibsp(
    boxes: list[tuple[float, float, float, float, float, float, str, str]],
    entities: list[dict[str, str]],
) -> bytes:
    """Direct zero-q3map2 IBSP v46 compiler for axis-aligned convex box worlds (`gas_world`)."""
    tex_map: dict[str, int] = {}
    tex_rows: list[tuple[bytes, int, int]] = []

    def _tex_id(name: str) -> int:
        if name not in tex_map:
            tex_map[name] = len(tex_rows)
            tex_rows.append((name.encode("ascii"), 0, CONTENTS_SOLID))
        return tex_map[name]

    planes_list: list[tuple[tuple[float, float, float], float]] = []
    bs_list: list[tuple[int, int]] = []
    brushes_list: list[tuple[int, int, int]] = []
    verts_list: list[tuple] = []
    tris_list: list[int] = []
    faces_list: list[tuple] = []
    box_aabbs: list[tuple[np.ndarray, np.ndarray]] = []

    # 6 outward faces of an axis-aligned box: (normal, quad_corners_ccw_from_outside, is_top)
    for x0, y0, z0, x1, y1, z1, tex_top, tex_side in boxes:
        t_top = _tex_id(tex_top)
        t_side = _tex_id(tex_side)
        box_aabbs.append((np.array([x0, y0, z0], np.float32), np.array([x1, y1, z1], np.float32)))

        p0 = len(planes_list)
        planes_list.extend([
            ((1.0, 0.0, 0.0), float(x1)),
            ((-1.0, 0.0, 0.0), float(-x0)),
            ((0.0, 1.0, 0.0), float(y1)),
            ((0.0, -1.0, 0.0), float(-y0)),
            ((0.0, 0.0, 1.0), float(z1)),
            ((0.0, 0.0, -1.0), float(-z0)),
        ])
        bs0 = len(bs_list)
        side_texs = (t_side, t_side, t_side, t_side, t_top, t_side)
        for k in range(6):
            bs_list.append((p0 + k, side_texs[k]))
        brushes_list.append((bs0, 6, t_side))

        quads = [
            ((1.0, 0.0, 0.0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], t_side),
            ((-1.0, 0.0, 0.0), [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], t_side),
            ((0.0, 1.0, 0.0), [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], t_side),
            ((0.0, -1.0, 0.0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], t_side),
            ((0.0, 0.0, 1.0), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], t_top),
            ((0.0, 0.0, -1.0), [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], t_side),
        ]
        for norm, pts, tid in quads:
            v0 = len(verts_list)
            t0 = len(tris_list)
            for px, py, pz in pts:
                u = (px + py) / 128.0
                v = pz / 128.0 if abs(norm[2]) < 0.5 else py / 128.0
                verts_list.append(((px, py, pz), (u, v), (0.0, 0.0), norm, (220, 220, 220, 255)))
            tris_list.extend([0, 2, 1, 0, 3, 2])
            faces_list.append((
                tid, -1, 1, v0, 4, t0, 6, -1, (0, 0), (0, 0),
                pts[0], ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), norm, (0, 0),
            ))

    # Build a clean axis-aligned BSP tree over boxes (each leaf holds <= 4 overlapping brushes)
    nodes_out: list[tuple] = []
    leafs_out: list[tuple] = []
    lf_out: list[int] = []
    lb_out: list[int] = []

    def _build_bsp(b_ids: list[int], depth: int = 0) -> int:
        u_min = np.min([box_aabbs[i][0] for i in b_ids], axis=0)
        u_max = np.max([box_aabbs[i][1] for i in b_ids], axis=0)
        if len(b_ids) <= 4 or depth >= 14:
            leaf_idx = len(leafs_out)
            lf0 = len(lf_out)
            lb0 = len(lb_out)
            for bi in b_ids:
                lb_out.append(bi)
                for fk in range(6):
                    lf_out.append(bi * 6 + fk)
            leafs_out.append((
                -1, 0,
                tuple(int(math.floor(x)) for x in u_min),
                tuple(int(math.ceil(x)) for x in u_max),
                lf0, len(lf_out) - lf0,
                lb0, len(lb_out) - lb0,
            ))
            return -(leaf_idx + 1)

        spread = u_max - u_min
        axis = int(np.argmax(spread))
        mids = [0.5 * float(box_aabbs[i][0][axis] + box_aabbs[i][1][axis]) for i in b_ids]
        split_d = float(np.median(mids))
        right = [i for i in b_ids if float(box_aabbs[i][1][axis]) > split_d]
        left = [i for i in b_ids if float(box_aabbs[i][0][axis]) < split_d]
        if not right or not left or (len(right) == len(b_ids) and len(left) == len(b_ids)):
            leaf_idx = len(leafs_out)
            lf0 = len(lf_out)
            lb0 = len(lb_out)
            for bi in b_ids:
                lb_out.append(bi)
                for fk in range(6):
                    lf_out.append(bi * 6 + fk)
            leafs_out.append((
                -1, 0,
                tuple(int(math.floor(x)) for x in u_min),
                tuple(int(math.ceil(x)) for x in u_max),
                lf0, len(lf_out) - lf0,
                lb0, len(lb_out) - lb0,
            ))
            return -(leaf_idx + 1)

        p_idx = len(planes_list)
        nvec = [0.0, 0.0, 0.0]
        nvec[axis] = 1.0
        planes_list.append((tuple(nvec), split_d))
        nid = len(nodes_out)
        nodes_out.append((0, (0, 0), (0, 0, 0), (0, 0, 0)))
        c_right = _build_bsp(right, depth + 1)
        c_left = _build_bsp(left, depth + 1)
        nodes_out[nid] = (
            p_idx,
            (c_right, c_left),
            tuple(int(math.floor(x)) for x in u_min),
            tuple(int(math.ceil(x)) for x in u_max),
        )
        return nid

    root_ref = _build_bsp(list(range(len(boxes))), 0)
    if root_ref < 0:
        # Ensure node 0 always exists (Mod_Q3BSP_Load expects >= 1 node)
        u_min = np.min([a[0] for a in box_aabbs], axis=0)
        u_max = np.max([a[1] for a in box_aabbs], axis=0)
        p_idx = len(planes_list)
        planes_list.append(((0.0, 0.0, 1.0), float(u_max[2] + 1024.0)))
        nodes_out.append((
            p_idx, (root_ref, root_ref),
            tuple(int(math.floor(x)) for x in u_min),
            tuple(int(math.ceil(x)) for x in u_max),
        ))

    g_min = np.min([a[0] for a in box_aabbs], axis=0)
    g_max = np.max([a[1] for a in box_aabbs], axis=0)
    models_arr = np.array([(g_min, g_max, 0, len(faces_list), 0, len(brushes_list))], dtype=DT_MODEL)

    return _pack_ibsp([
        _format_entities_lump(entities),
        np.array(tex_rows, dtype=DT_TEX).tobytes(),
        np.array(planes_list, dtype=DT_PLANE).tobytes(),
        np.array(nodes_out, dtype=DT_NODE).tobytes(),
        np.array(leafs_out, dtype=DT_LEAF).tobytes(),
        np.array(lf_out, dtype="<i4").tobytes(),
        np.array(lb_out, dtype="<i4").tobytes(),
        models_arr.tobytes(),
        np.array(brushes_list, dtype=DT_BRUSH).tobytes(),
        np.array(bs_list, dtype=DT_BRUSHSIDE).tobytes(),
        np.array(verts_list, dtype=DT_VERT).tobytes(),
        np.array(tris_list, dtype="<i4").tobytes(),
        b"",
        np.array(faces_list, dtype=DT_FACE).tobytes(),
        b"",
        b"",
        b"",
    ])


def build_gas_world_bsp_in_memory(
    *,
    map_name: str,
    seed: int,
    teams: int,
    carts: int,
    players: int,
    tiles: int,
    tile_size: float = 1792.0,
    joint_width: float = 320.0,
    joint_length: float = 96.0,
    checkpoints_per_cart: int = 4,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rng = random.Random(seed)
    tex_floor, tex_wall, tex_trim, tex_ceil = GAS_THEMES[seed % len(GAS_THEMES)]
    S = float(tile_size)
    wall_t = float(joint_length)
    jw = float(joint_width)
    ceil_h = 512.0
    door_h = 256.0
    K = max(2, min(24, int(teams)))
    J = max(K + 1, min(256, int(carts)))
    M_req = max(1, min(24, int(tiles)))
    N = max(max(2 * K, 12), min(240, int(players)))
    cols, rows = _grid_dims(M_req)
    M = cols * rows
    pitch = S + wall_t
    total_x = cols * pitch - wall_t
    total_y = rows * pitch - wall_t
    T = 64.0

    boxes: list[tuple[float, float, float, float, float, float, str, str]] = [
        (-T, -T, -T, total_x + T, total_y + T, 0.0, tex_floor, tex_wall),
        (-T, -T, ceil_h, total_x + T, total_y + T, ceil_h + T, tex_ceil, tex_ceil),
        (-T, -T, -T, 0.0, total_y + T, ceil_h + T, tex_wall, tex_wall),
        (total_x, -T, -T, total_x + T, total_y + T, ceil_h + T, tex_wall, tex_wall),
        (-T, -T, -T, total_x + T, 0.0, ceil_h + T, tex_wall, tex_wall),
        (-T, total_y, -T, total_x + T, total_y + T, ceil_h + T, tex_wall, tex_wall),
    ]
    tiles_meta = []
    joints_meta = []
    for cy in range(rows):
        for cx in range(cols):
            x0, y0 = cx * pitch, cy * pitch
            tiles_meta.append({"id": cy * cols + cx, "x0": x0, "y0": y0, "x1": x0 + S, "y1": y0 + S, "center": (x0 + 0.5 * S, y0 + 0.5 * S, 0.0)})

    half_w = 0.5 * jw
    jid = 0
    for cy in range(rows):
        for cx in range(cols - 1):
            a_id, b_id = cy * cols + cx, cy * cols + cx + 1
            y0, xw0, y1 = cy * pitch, cx * pitch + S, cy * pitch + S
            xw1 = xw0 + wall_t
            dual = (rows == 1 and cols >= 2) or (cy in (0, rows - 1) and cx in (0, cols - 2) and M >= 4)
            centers = [y0 + S * 0.30, y0 + S * 0.70] if dual else [0.5 * (y0 + y1)]
            cur = y0
            for myc in centers:
                d0, d1 = myc - half_w, myc + half_w
                if d0 > cur + 1.0:
                    boxes.append((xw0, cur, 0.0, xw1, d0, ceil_h, tex_wall, tex_wall))
                boxes.append((xw0, d0, door_h, xw1, d1, ceil_h, tex_wall, tex_wall))
                cur = d1
                joints_meta.append({"id": jid, "tile_a": a_id, "tile_b": b_id, "origin": (0.5 * (xw0 + xw1), myc, 0.0), "normal": (1.0, 0.0, 0.0)})
                jid += 1
            if y1 > cur + 1.0:
                boxes.append((xw0, cur, 0.0, xw1, y1, ceil_h, tex_wall, tex_wall))

    for cy in range(rows - 1):
        for cx in range(cols):
            a_id, b_id = cy * cols + cx, (cy + 1) * cols + cx
            x0, x1, yw0 = cx * pitch, cx * pitch + S, cy * pitch + S
            yw1 = yw0 + wall_t
            dual = cols == 1 and rows >= 2
            centers = [x0 + S * 0.30, x0 + S * 0.70] if dual else [0.5 * (x0 + x1)]
            cur = x0
            for mxc in centers:
                d0, d1 = mxc - half_w, mxc + half_w
                if d0 > cur + 1.0:
                    boxes.append((cur, yw0, 0.0, d0, yw1, ceil_h, tex_wall, tex_wall))
                boxes.append((d0, yw0, door_h, d1, yw1, ceil_h, tex_wall, tex_wall))
                cur = d1
                joints_meta.append({"id": jid, "tile_a": a_id, "tile_b": b_id, "origin": (mxc, 0.5 * (yw0 + yw1), 0.0), "normal": (0.0, 1.0, 0.0)})
                jid += 1
            if x1 > cur + 1.0:
                boxes.append((cur, yw0, 0.0, x1, yw1, ceil_h, tex_wall, tex_wall))

    for cy in range(rows - 1):
        for cx in range(cols - 1):
            xw0, yw0 = cx * pitch + S, cy * pitch + S
            boxes.append((xw0, yw0, 0.0, xw0 + wall_t, yw0 + wall_t, ceil_h, tex_trim, tex_trim))

    ents: list[dict[str, str]] = [{"classname": "worldspawn", "message": map_name}]
    wps: list[tuple[tuple[float, float, float], tuple[float, float, float], int]] = []
    wp_pts: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []
    tile_wps: dict[int, list[int]] = {}

    def _add_wp(x: float, y: float, z: float = 28.0) -> int:
        p = (round(x, 1), round(y, 1), round(z, 1))
        wps.append((p, p, 0))
        wp_pts.append(p)
        return len(wps) - 1

    for t in tiles_meta:
        mx, my = t["center"][0], t["center"][1]
        boxes.append((mx - 44.0, my - 44.0, 0.0, mx + 44.0, my + 44.0, 192.0, tex_trim, tex_wall))
        ents.append({"classname": "light", "origin": _fmt_vec3((mx, my, ceil_h - 64.0)), "light": "850"})
        ents.append({"classname": rng.choice(GAS_ITEMS), "origin": _fmt_vec3((mx + 140.0, my, 28.0))})
        mat = []
        margin = 160.0
        for iy in range(5):
            row = []
            py = t["y0"] + margin + (S - 2.0 * margin) * iy / 4.0
            for ix in range(5):
                px = t["x0"] + margin + (S - 2.0 * margin) * ix / 4.0
                if ix == 2 and iy == 2:
                    px += 96.0
                row.append(_add_wp(px, py, 28.0))
            mat.append(row)
        for iy in range(5):
            for ix in range(5):
                if ix + 1 < 5:
                    edges.append((mat[iy][ix], mat[iy][ix + 1]))
                if iy + 1 < 5:
                    edges.append((mat[iy][ix], mat[iy + 1][ix]))
                if ix + 1 < 5 and iy + 1 < 5:
                    edges.append((mat[iy][ix], mat[iy + 1][ix + 1]))
        tile_wps[t["id"]] = [w for r in mat for w in r]

    for jt in joints_meta:
        ox, oy, _ = jt["origin"]
        nx, ny, _ = jt["normal"]
        off = 0.5 * wall_t + 120.0
        wa = _add_wp(ox - nx * off, oy - ny * off, 28.0)
        wc = _add_wp(ox, oy, 28.0)
        wb = _add_wp(ox + nx * off, oy + ny * off, 28.0)
        edges.append((wa, wc))
        edges.append((wc, wb))
        for wp_end, tid in ((wa, jt["tile_a"]), (wb, jt["tile_b"])):
            cands = sorted(tile_wps[tid], key=lambda w: math.hypot(wp_pts[w][0] - wp_pts[wp_end][0], wp_pts[w][1] - wp_pts[wp_end][1]))
            for nb in cands[:2]:
                edges.append((wp_end, nb))

    # Spawns & Teams
    for t in range(K):
        ents.append({"classname": "plc_team", "cnt": str(t)})
    corners = [(220.0, 220.0, 45.0), (S - 220.0, S - 220.0, 225.0), (220.0, S - 220.0, 315.0), (S - 220.0, 220.0, 135.0)]
    for t_idx in range(K):
        tm = tiles_meta[t_idx % M]
        ax, ay, yaw = corners[(t_idx // M) % len(corners)]
        bx, by = tm["x0"] + ax, tm["y0"] + ay
        for dx, dy in ((-48.0, -48.0), (48.0, -48.0), (-48.0, 48.0), (48.0, 48.0)):
            ents.append({"classname": "info_player_team", "origin": _fmt_vec3((bx + dx, by + dy, 32.0)), "angle": f"{yaw:.1f}", "team": str(t_idx + 1)})

    # Carts (cross-joint + intra-tile)
    tracks: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    n_cross = min(len(joints_meta), int(J * 0.35)) if M > 1 else 0
    j_ord = list(joints_meta)
    rng.shuffle(j_ord)
    for jt in j_ord[:n_cross]:
        ox, oy, _ = jt["origin"]
        nx, ny, _ = jt["normal"]
        sgn = 1.0 if (jt["id"] % 2 == 0) else -1.0
        tracks.append(((ox - sgn * nx * 420.0, oy - sgn * ny * 420.0, 24.0), (ox + sgn * nx * 420.0, oy + sgn * ny * 420.0, 24.0)))
    lane_tpls = [
        ((240, 320), (1040, 320)), ((1552, 1472), (752, 1472)), ((320, 1552), (320, 752)), ((1472, 240), (1472, 1040)),
        ((540, 620), (1252, 620)), ((1252, 1172), (540, 1172)), ((620, 1252), (620, 540)), ((1172, 540), (1172, 1252)),
    ]
    for k in range(J - len(tracks)):
        tm = tiles_meta[k % M]
        slot = k // M
        (ax, ay), (bx, by) = lane_tpls[slot % len(lane_tpls)]
        sh = (slot // len(lane_tpls)) * 48.0
        tracks.append(((tm["x0"] + ax + sh, tm["y0"] + ay, 24.0), (tm["x0"] + bx + sh, tm["y0"] + by, 24.0)))

    ncp = max(1, int(checkpoints_per_cart))
    for ci, (p0, p1) in enumerate(tracks):
        nodes = [
            (p0[0] + (p1[0] - p0[0]) * k / float(ncp), p0[1] + (p1[1] - p0[1]) * k / float(ncp), 24.0)
            for k in range(ncp + 1)
        ]
        names = [f"plc{ci}n{ni}" for ni in range(len(nodes))]
        for ni, p in enumerate(nodes):
            kv = {"classname": "plc_path", "targetname": names[ni], "origin": _fmt_vec3(p)}
            if ni + 1 < len(nodes):
                kv["target"] = names[ni + 1]
            if ni > 0:
                kv["spawnflags"] = "1"
            ents.append(kv)
        ents.append({"classname": "func_plc_cart", "origin": _fmt_vec3(nodes[0]), "target": names[0], "radius": "160", "height": "96", "plc_speed_scale": "1"})

    bsp_bytes = _compile_boxes_to_ibsp(boxes, ents)
    ent_bytes = _format_entities_lump(ents)[:-1]
    mapinfo_bytes = (
        f"title {map_name}\n"
        f"description Parametric zero-copy gas_world arena (K={K}, J={J}, N={N}, M={M})\n"
        f"author mesh-bsp-stream\n"
        f"gametype dm\n"
        f"gametype tdm\n"
        f"gametype plc\n"
    ).encode("utf-8")

    spec = {
        "name": map_name,
        "family": "gas_world_bsp",
        "seed": int(seed),
        "teams": K,
        "carts": J,
        "players": N,
        "tiles": M,
        "joint_width": jw,
        "maxclients": min(255, N + 15),
        "bsp_bytes": len(bsp_bytes),
    }
    vfs_files = {
        f"maps/{map_name}.bsp": bsp_bytes,
        f"maps/{map_name}.ent": ent_bytes,
        f"maps/{map_name}.waypoints": _format_waypoints_bytes(wps),
        f"maps/{map_name}.waypoints.cache": _format_waypoints_cache_bytes(wps, edges),
        f"maps/{map_name}.mapinfo": mapinfo_bytes,
        f"maps/{map_name}.joints.json": json.dumps({"spec": spec, "joints": joints_meta}).encode("utf-8"),
    }
    return spec, vfs_files


def sample_world_spec(seed: int, world_cfg) -> dict[str, Any]:
    rng = random.Random(seed)
    k_lo, k_hi = int(world_cfg.teams_min), int(world_cfg.teams_max)
    if world_cfg.teams_dist == "uniform":
        K = rng.randint(k_lo, k_hi)
    else:
        K = max(k_lo, min(k_hi, int(round(math.exp(rng.gauss(float(world_cfg.teams_log_mean), float(world_cfg.teams_log_std)))))))

    if world_cfg.carts_dist == "uniform":
        mult = rng.uniform(float(world_cfg.carts_per_team_min), float(world_cfg.carts_per_team_max))
    else:
        mult = math.exp(rng.gauss(float(world_cfg.carts_mult_log_mean), float(world_cfg.carts_mult_log_std)))
    J = max(K + int(world_cfg.carts_min_extra), min(int(world_cfg.carts_max), int(round(K * mult))))

    per_team = max(
        int(world_cfg.players_per_team_min),
        min(int(world_cfg.players_per_team_max), int(round(rng.uniform(float(world_cfg.players_per_team_low), float(world_cfg.players_per_team_high))))),
    )
    N = max(max(int(world_cfg.players_per_team_min) * K, int(world_cfg.players_min)), min(int(world_cfg.players_max), K * per_team))

    m_gas = max(1, int(round(N / max(float(world_cfg.players_per_tile), 1.0))))
    m_cart = max(1, int(math.ceil(J / max(float(world_cfg.max_carts_per_tile), 1.0))))
    M = max(int(world_cfg.tiles_min), min(int(world_cfg.tiles_max), max(m_gas, m_cart)))

    fams = list(world_cfg.family_weights.keys())
    wts = [max(0.0, float(world_cfg.family_weights[f])) for f in fams]
    tot = sum(wts) or 1.0
    r = rng.random() * tot
    acc = 0.0
    family = fams[-1]
    for f, w in zip(fams, wts):
        acc += w
        if r <= acc:
            family = f
            break

    jw = float(rng.choice(world_cfg.joint_widths))
    return {
        "seed": int(seed),
        "family": family,
        "teams": K,
        "carts": J,
        "players": N,
        "tiles": M,
        "joint_width": jw,
    }


def synthesize_world_bundle(
    params: dict[str, Any],
    world_cfg,
    *,
    stitcher: B3313BinaryStitcher | None = None,
    name_prefix: str = "stream",
) -> tuple[dict[str, Any], dict[str, bytes]]:
    seed = int(params["seed"])
    K, J, N, M = int(params["teams"]), int(params["carts"]), int(params["players"]), int(params["tiles"])
    family = str(params["family"])
    tag = "b3313" if family == "b3313_binary_stitch" else "gas"
    map_name = f"{name_prefix}_{tag}_s{seed}_k{K}_j{J}_n{N}"
    if family == "b3313_binary_stitch":
        if stitcher is None:
            raise ValueError("B3313BinaryStitcher required for b3313_binary_stitch family")
        return stitcher.stitch(
            map_name=map_name,
            seed=seed,
            teams=K,
            carts=J,
            players=N,
            tiles=M,
            stock_pool=tuple(world_cfg.stock_map_pool),
            expander_min_degree=int(world_cfg.expander_min_degree),
            allow_rotations=bool(world_cfg.allow_rotations),
            checkpoints_per_cart=int(world_cfg.checkpoints_per_cart),
        )
    return build_gas_world_bsp_in_memory(
        map_name=map_name,
        seed=seed,
        teams=K,
        carts=J,
        players=N,
        tiles=M,
        tile_size=float(world_cfg.tile_size),
        joint_width=float(params.get("joint_width", world_cfg.joint_widths[0])),
        joint_length=float(world_cfg.joint_length),
        checkpoints_per_cart=int(world_cfg.checkpoints_per_cart),
    )
