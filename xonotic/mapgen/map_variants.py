#!/usr/bin/env python3
"""Stock Xonotic Map AST Parser, Affine Variant Engine, and Parametric Geometry Motifs.

Provides B3313-style parametric variants and architectural motif injection directly on
real decompiled Xonotic stock maps (`.bsp` -> `q3map2 -convert -format map` -> AST):
  - Exact rigid/symmetry transforms: rotation (0/90/180/270), mirror (x/y with brush
    plane winding reversal & patchDef2 row reversal), xy/z scale, and atlas translation.
  - Material/atmosphere palettes: stock, industrial_exx, obsidian_ruin, crimson_forge, cryo_lab.
  - Parametric architectural geometry motifs injected onto verified walkable nodes:
      * monolith_baffles   (sightline-breaking cover pillars / L-baffles)
      * skybridge_overpass (solid elevated catwalk + ground-anchored ramps, 0 crawlspace)
      * jumppad_shaft      (trigger_push + target_position vertical kinetic launchers)
      * bunker_archway     (twin buttresses + overhead lintel framing thoroughfares)
      * pickup_remix       (heavy_ordnance / high_mobility / fortress_armor economies)
"""
from __future__ import annotations

import copy
import math
import random
import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from spiralgen import fmt, quad_prism, tri_prism, vlen, vsub

# Curated verified catalog of stock Xonotic maps that round-trip cleanly with 0 leaks.
STOCK_MAP_CATALOG = [
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
]

# Non-structural / gamemode-specific entities stripped before unified payload tiling.
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
}

TARGET_REF_KEYS = ("targetname", "target", "target2", "target3", "target4", "killtarget", "pathtarget")

PALETTES = {
    "stock": {
        "floor": None, "wall": None, "trim": None, "ambient": None, "light_scale": 1.0,
    },
    "industrial_exx": {
        "floor": "exx/floor-tread01",
        "wall": "exx/wall-bigrib02",
        "trim": "exx/trim-01",
        "ambient": "28",
        "light_scale": 1.05,
    },
    "obsidian_ruin": {
        "floor": "exx/base-metal03",
        "wall": "exx/base-crete01",
        "trim": "exx/trim-02",
        "ambient": "22",
        "light_scale": 0.90,
    },
    "crimson_forge": {
        "floor": "exx/floor-clang01",
        "wall": "exx/panel-metalbig04",
        "trim": "exx/trim-01",
        "ambient": "32",
        "light_scale": 1.15,
    },
    "cryo_lab": {
        "floor": "exx/base-crete01",
        "wall": "exx/panel-metalbig04",
        "trim": "exx/base-metal03",
        "ambient": "36",
        "light_scale": 1.10,
    },
}

PRESERVE_SHADER_PREFIXES = (
    "common/", "skies/", "sky", "effects", "liquids/", "water", "lava", "slime",
    "clip", "caulk", "nodraw", "trigger", "weapclip", "botclip", "playerclip", "warpzone",
)

PICKUP_ECONOMIES = {
    "stock": [],
    "heavy_ordnance": ["weapon_devastator", "weapon_vortex", "weapon_mortar", "item_armor_big"],
    "high_mobility": ["weapon_electro", "weapon_hagar", "weapon_crylink", "item_health_medium"],
    "fortress_armor": ["item_armor_big", "item_health_mega", "weapon_machinegun", "weapon_vortex"],
}

ALL_MOTIFS = (
    "monolith_baffles",
    "skybridge_overpass",
    "jumppad_shaft",
    "bunker_archway",
    "pickup_remix",
)


@dataclass
class SubmapVariantSpec:
    base_map: str
    rot_deg: int = 0                    # 0, 90, 180, 270
    mirror_axis: str = "none"           # "none", "x", "y"
    xy_scale: float = 1.0               # [0.85, 1.25]
    z_scale: float = 1.0                # [0.90, 1.25]
    palette: str = "stock"
    motifs: list[str] = field(default_factory=list)
    pickup_economy: str = "stock"
    seed: int = 0

    def label(self) -> str:
        tags = []
        if self.rot_deg:
            tags.append(f"r{self.rot_deg}")
        if self.mirror_axis != "none":
            tags.append(f"m{self.mirror_axis}")
        if abs(self.xy_scale - 1.0) > 0.02:
            tags.append(f"s{self.xy_scale:.2f}")
        if self.palette != "stock":
            tags.append(self.palette)
        tags.extend(self.motifs)
        return f"{self.base_map}[{','.join(tags)}]" if tags else self.base_map


@dataclass
class ParsedEntity:
    kv: dict[str, str]
    brushes: list[str]   # Raw `{ ... }` brush or patch blocks (without outer braces)


@dataclass
class StockMapAsset:
    name: str
    world_kv: dict[str, str]
    world_brushes: list[str]
    entities: list[ParsedEntity]
    waypoints: list[tuple[tuple[float, float, float], tuple[float, float, float], int]]
    wp_edges: list[tuple[int, int]]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]


def _preserve_shader(shader: str) -> bool:
    s = shader.lower()
    return any(s.startswith(p) or f"/{p}" in s for p in PRESERVE_SHADER_PREFIXES)


def _remap_shader(shader: str, palette: str) -> str:
    if palette == "stock" or _preserve_shader(shader):
        return shader
    pal = PALETTES.get(palette, PALETTES["stock"])
    sl = shader.lower()
    if any(k in sl for k in ("floor", "ground", "grate", "tread", "step", "tile", "crete")):
        return pal["floor"] or shader
    if any(k in sl for k in ("trim", "pipe", "light", "beam", "frame", "support", "border")):
        return pal["trim"] or shader
    return pal["wall"] or shader


def _parse_map_text(text: str) -> tuple[dict[str, str], list[str], list[ParsedEntity]]:
    """Fast brace-depth parser for Quake3 / q3map2-converted `.map` files."""
    world_kv: dict[str, str] = {}
    world_brushes: list[str] = []
    entities: list[ParsedEntity] = []

    kv_re = re.compile(r'^\s*"([^"]+)"\s+"([^"]*)"\s*$')
    depth = 0
    cur_kv: dict[str, str] = {}
    cur_brushes: list[str] = []
    cur_brush_lines: list[str] = []
    in_brush = False
    is_world = True

    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        if depth == 0:
            if s == "{":
                depth = 1
                cur_kv = {}
                cur_brushes = []
            continue
        if depth == 1:
            if s == "{":
                depth = 2
                in_brush = True
                cur_brush_lines = []
            elif s == "}":
                depth = 0
                if is_world and cur_kv.get("classname") == "worldspawn":
                    world_kv = cur_kv
                    world_brushes = cur_brushes
                    is_world = False
                else:
                    entities.append(ParsedEntity(kv=cur_kv, brushes=cur_brushes))
            else:
                m = kv_re.match(raw)
                if m:
                    cur_kv[m.group(1)] = m.group(2)
            continue
        # depth >= 2 (inside brush or patchDef2)
        if in_brush:
            if s == "{":
                depth += 1
                cur_brush_lines.append(raw)
            elif s == "}":
                depth -= 1
                if depth == 1:
                    in_brush = False
                    cur_brushes.append("\n".join(cur_brush_lines))
                    cur_brush_lines = []
                else:
                    cur_brush_lines.append(raw)
            else:
                cur_brush_lines.append(raw)
    return world_kv, world_brushes, entities


def _parse_waypoints_text(text: str) -> list[tuple[tuple[float, float, float], tuple[float, float, float], int]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("//")]
    wps = []
    i = 0
    while i + 2 < len(lines):
        a = lines[i].strip("'").split()
        b = lines[i + 1].strip("'").split()
        fl = int(float(lines[i + 2]))
        if len(a) == 3 and len(b) == 3:
            wps.append(((float(a[0]), float(a[1]), float(a[2])), (float(b[0]), float(b[1]), float(b[2])), fl))
        i += 3
    return wps


def _parse_wp_cache_edges(
    cache_text: str,
    wps: list[tuple[tuple[float, float, float], tuple[float, float, float], int]],
) -> list[tuple[int, int]]:
    lookup: dict[tuple[int, int, int], int] = {}
    for idx, (mn, mx, _) in enumerate(wps):
        c = (round(0.5 * (mn[0] + mx[0])), round(0.5 * (mn[1] + mx[1])), round(0.5 * (mn[2] + mx[2])))
        lookup[c] = idx
        lookup[(round(mn[0]), round(mn[1]), round(mn[2]))] = idx
    edges: set[tuple[int, int]] = set()
    for ln in cache_text.splitlines():
        s = ln.strip()
        if "*" not in s or s.startswith("//"):
            continue
        parts = s.split("*")
        if len(parts) != 2:
            continue
        pa = parts[0].strip().strip("'").split()
        pb = parts[1].strip().strip("'").split()
        if len(pa) != 3 or len(pb) != 3:
            continue
        ka = (round(float(pa[0])), round(float(pa[1])), round(float(pa[2])))
        kb = (round(float(pb[0])), round(float(pb[1])), round(float(pb[2])))
        ia, ib = lookup.get(ka), lookup.get(kb)
        if ia is not None and ib is not None and ia != ib:
            edges.add((min(ia, ib), max(ia, ib)))
    return sorted(edges)


def _compute_map_bbox(world_brushes: list[str], entities: list[ParsedEntity]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    coord_re = re.compile(r"\(\s*([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)")
    xs, ys, zs = [], [], []
    for br in world_brushes:
        for m in coord_re.finditer(br):
            xs.append(float(m.group(1)))
            ys.append(float(m.group(2)))
            zs.append(float(m.group(3)))
    for ent in entities:
        for br in ent.brushes:
            for m in coord_re.finditer(br):
                xs.append(float(m.group(1)))
                ys.append(float(m.group(2)))
                zs.append(float(m.group(3)))
    if not xs:
        return (0.0, 0.0, 0.0), (1024.0, 1024.0, 512.0)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def load_stock_map(
    name: str,
    xonotic_dir: Path,
    cache_dir: Path = Path("/tmp/xonotic_stock_cache"),
) -> StockMapAsset:
    """Extract, decompile (cached), and parse a real stock Xonotic `.bsp` + `.waypoints`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = cache_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    conv_map = maps_dir / f"{name}_converted.map"
    bsp_path = maps_dir / f"{name}.bsp"
    wp_path = maps_dir / f"{name}.waypoints"
    wpc_path = maps_dir / f"{name}.waypoints.cache"

    if not conv_map.exists():
        # Locate maps pk3
        pk3s = sorted((xonotic_dir / "data").glob("xonotic-*maps*.pk3"))
        if not pk3s:
            raise FileNotFoundError(f"No xonotic-*-maps*.pk3 found in {xonotic_dir / 'data'}")
        with zipfile.ZipFile(pk3s[-1], "r") as zf:
            names = set(zf.namelist())
            for ext in (".bsp", ".waypoints", ".waypoints.cache"):
                member = f"maps/{name}{ext}"
                if member in names:
                    (maps_dir / f"{name}{ext}").write_bytes(zf.read(member))
        if not bsp_path.exists():
            raise FileNotFoundError(f"maps/{name}.bsp not found in {pk3s[-1]}")
        q3 = Path(__file__).resolve().parents[2] / "bin/mesh-q3map2"
        r = subprocess.run(
            [str(q3), "-game", "xonotic", "-fs_basepath", str(xonotic_dir), "-convert", "-format", "map", str(bsp_path)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not conv_map.exists():
            raise RuntimeError(f"q3map2 -convert failed on {name}:\n{(r.stdout + r.stderr)[-500:]}")

    world_kv, world_brushes, entities = _parse_map_text(conv_map.read_text(errors="ignore"))
    wps = _parse_waypoints_text(wp_path.read_text(errors="ignore")) if wp_path.exists() else []
    wp_edges = _parse_wp_cache_edges(wpc_path.read_text(errors="ignore"), wps) if (wpc_path.exists() and wps) else []
    bmin, bmax = _compute_map_bbox(world_brushes, entities)
    return StockMapAsset(
        name=name,
        world_kv=world_kv,
        world_brushes=world_brushes,
        entities=entities,
        waypoints=wps,
        wp_edges=wp_edges,
        bbox_min=bmin,
        bbox_max=bmax,
    )


class AffineTransform:
    """Exact 3D affine transform: center -> scale -> mirror -> rotate_z -> translate."""

    def __init__(
        self,
        center: tuple[float, float, float],
        rot_deg: int,
        mirror_axis: str,
        xy_scale: float,
        z_scale: float,
        dst_offset: tuple[float, float, float],
    ):
        self.cx, self.cy, self.cz = center
        self.rot_deg = int(rot_deg) % 360
        self.mirror_axis = mirror_axis
        self.xy_scale = float(xy_scale)
        self.z_scale = float(z_scale)
        self.dx, self.dy, self.dz = dst_offset
        rad = math.radians(self.rot_deg)
        self.cos_r = round(math.cos(rad)) if self.rot_deg in (0, 90, 180, 270) else math.cos(rad)
        self.sin_r = round(math.sin(rad)) if self.rot_deg in (0, 90, 180, 270) else math.sin(rad)
        self.flips_winding = mirror_axis in ("x", "y")

    def point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        lx = (x - self.cx) * self.xy_scale
        ly = (y - self.cy) * self.xy_scale
        lz = (z - self.cz) * self.z_scale
        if self.mirror_axis == "x":
            lx = -lx
        elif self.mirror_axis == "y":
            ly = -ly
        rx = lx * self.cos_r - ly * self.sin_r
        ry = lx * self.sin_r + ly * self.cos_r
        return (rx + self.dx, ry + self.dy, lz + self.dz)

    def yaw(self, deg: float) -> float:
        a = float(deg)
        if self.mirror_axis == "x":
            a = 180.0 - a
        elif self.mirror_axis == "y":
            a = -a
        a = (a + self.rot_deg) % 360.0
        return a


PLANE_RE = re.compile(
    r"^\s*\(\s*([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*\)\s*"
    r"\(\s*([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*\)\s*"
    r"\(\s*([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*\)\s*"
    r"(\S+)(.*)$"
)
PATCH_PT_RE = re.compile(
    r"\(\s*([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*\)"
)


def transform_brush_block(block: str, xf: AffineTransform, palette: str) -> str:
    """Transform a brush or patchDef2 block cleanly, reversing winding on mirror."""
    if "patchDef" in block:
        out_lines = []
        row_lines: list[str] = []
        in_rows = False
        shader_done = False
        for raw in block.splitlines():
            s = raw.strip()
            if not shader_done and s and not s.startswith("patchDef") and s not in ("{", "}") and not s.startswith("("):
                out_lines.append("\t\t\t" + _remap_shader(s, palette))
                shader_done = True
                continue
            if s.startswith("( ("):
                in_rows = True
                def _sub_pt(m):
                    nx, ny, nz = xf.point(float(m.group(1)), float(m.group(2)), float(m.group(3)))
                    return f"( {nx:.3f} {ny:.3f} {nz:.3f} {m.group(4)} {m.group(5)} )"
                row_lines.append("\t\t\t\t" + PATCH_PT_RE.sub(_sub_pt, s))
                continue
            if in_rows and s == ")":
                if xf.flips_winding:
                    row_lines.reverse()
                out_lines.extend(row_lines)
                row_lines = []
                in_rows = False
            out_lines.append(raw)
        if row_lines:
            if xf.flips_winding:
                row_lines.reverse()
            out_lines.extend(row_lines)
        return "\n".join(out_lines)

    out_lines = []
    for raw in block.splitlines():
        m = PLANE_RE.match(raw)
        if not m:
            out_lines.append(raw)
            continue
        p1 = xf.point(float(m.group(1)), float(m.group(2)), float(m.group(3)))
        p2 = xf.point(float(m.group(4)), float(m.group(5)), float(m.group(6)))
        p3 = xf.point(float(m.group(7)), float(m.group(8)), float(m.group(9)))
        if xf.flips_winding:
            p2, p3 = p3, p2
        shader = _remap_shader(m.group(10), palette)
        rest = m.group(11)
        out_lines.append(
            f"\t\t( {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f} ) "
            f"( {p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f} ) "
            f"( {p3[0]:.3f} {p3[1]:.3f} {p3[2]:.3f} ) {shader}{rest}"
        )
    return "\n".join(out_lines)


def _box_block(x0, y0, z0, x1, y1, z1, tex_top, tex_side=None) -> str:
    ts = tex_side or tex_top
    br = quad_prism(
        (float(x0), float(y0), float(z0)),
        (float(x1), float(y0), float(z0)),
        (float(x1), float(y1), float(z0)),
        (float(x0), float(y1), float(z0)),
        (0.0, 0.0, float(z1 - z0)),
        tex_top, ts,
    )
    txt = br.to_map(0.5).strip()
    # Strip outer { } so it matches ParsedEntity/world_brushes inner format
    lines = txt.splitlines()
    return "\n".join(lines[1:-1])


def _ramp_blocks(a: tuple[float, float, float], b: tuple[float, float, float], width: float, z_base: float, tex_top: str, tex_side: str) -> list[str]:
    """Solid ground-anchored ramp prism from point a to point b down to z_base."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 16.0:
        return []
    nx, ny = (-dy / L) * (width * 0.5), (dx / L) * (width * 0.5)
    p00 = (a[0] - nx, a[1] - ny, a[2])
    p01 = (b[0] - nx, b[1] - ny, b[2])
    p11 = (b[0] + nx, b[1] + ny, b[2])
    p10 = (a[0] + nx, a[1] + ny, a[2])
    drop = -(max(a[2], b[2]) - z_base + 24.0)
    out = []
    for tri in ((p00, p01, p11), (p00, p11, p10)):
        br = tri_prism(tri[0], tri[1], tri[2], (0.0, 0.0, drop), tex_top, tex_side)
        if br is not None:
            lines = br.to_map(0.5).strip().splitlines()
            out.append("\n".join(lines[1:-1]))
    return out


def instantiate_stock_variant(
    asset: StockMapAsset,
    spec: SubmapVariantSpec,
    tile_id: int,
    dst_offset: tuple[float, float, float],
) -> dict:
    """Apply affine transform + material palette + entity namespacing + parametric motifs
    to a real stock Xonotic map asset, returning transformed brushes, entities, waypoints,
    edges, and exact AABB."""
    cx = 0.5 * (asset.bbox_min[0] + asset.bbox_max[0])
    cy = 0.5 * (asset.bbox_min[1] + asset.bbox_max[1])
    cz = 0.5 * (asset.bbox_min[2] + asset.bbox_max[2])
    xf = AffineTransform((cx, cy, cz), spec.rot_deg, spec.mirror_axis, spec.xy_scale, spec.z_scale, dst_offset)
    pal = PALETTES.get(spec.palette, PALETTES["stock"])
    prefix = f"m{tile_id}_"
    rng = random.Random(spec.seed + 1009 * tile_id)

    t_brushes = [transform_brush_block(b, xf, spec.palette) for b in asset.world_brushes]

    # Transform & namespace kept entities (and capture verified stock spawn pads before stripping)
    t_entities: list[ParsedEntity] = []
    stock_spawns: list[tuple[tuple[float, float, float], float]] = []
    for ent in asset.entities:
        cls = ent.kv.get("classname", "")
        if cls in ("info_player_deathmatch", "info_player_team1", "info_player_team2", "info_player_team3", "info_player_team4", "info_player_start"):
            if "origin" in ent.kv:
                parts = ent.kv["origin"].split()
                if len(parts) == 3:
                    sp = xf.point(float(parts[0]), float(parts[1]), float(parts[2]))
                    ang = xf.yaw(float(ent.kv.get("angle", "0")))
                    stock_spawns.append((sp, ang))
        if cls in STRIP_ENTITY_CLASSES:
            continue
        # Mirror flips handedness on internal warpzones; skip internal warpzones only when mirrored
        if xf.flips_winding and "warpzone" in cls:
            continue
        if not ent.brushes and "origin" not in ent.kv:
            continue
        nkv = dict(ent.kv)
        for rk in TARGET_REF_KEYS:
            if rk in nkv and nkv[rk]:
                nkv[rk] = prefix + nkv[rk]
        if "origin" in nkv:
            parts = nkv["origin"].split()
            if len(parts) == 3:
                nx, ny, nz = xf.point(float(parts[0]), float(parts[1]), float(parts[2]))
                nkv["origin"] = f"{nx:.1f} {ny:.1f} {nz:.1f}"
        if "angle" in nkv:
            try:
                ang = float(nkv["angle"])
                if ang >= 0.0:
                    nkv["angle"] = f"{xf.yaw(ang):.1f}"
            except ValueError:
                pass
        if "angles" in nkv:
            parts = nkv["angles"].split()
            if len(parts) == 3:
                try:
                    nkv["angles"] = f"{float(parts[0]):.1f} {xf.yaw(float(parts[1])):.1f} {float(parts[2]):.1f}"
                except ValueError:
                    pass
        if cls == "light" and "light" in nkv and pal["light_scale"] != 1.0:
            try:
                nkv["light"] = f"{float(nkv['light']) * pal['light_scale']:.1f}"
            except ValueError:
                pass
        ebr = [transform_brush_block(b, xf, spec.palette) for b in ent.brushes]
        t_entities.append(ParsedEntity(kv=nkv, brushes=ebr))

    # Transform waypoints
    t_wps: list[tuple[tuple[float, float, float], tuple[float, float, float], int]] = []
    wp_centers: list[tuple[float, float, float]] = []
    for mn, mx, fl in asset.waypoints:
        # Transform bounding box corners cleanly
        pts = [
            xf.point(x, y, z)
            for x in (mn[0], mx[0])
            for y in (mn[1], mx[1])
            for z in (mn[2], mx[2])
        ]
        nmn = (min(p[0] for p in pts), min(p[1] for p in pts), min(p[2] for p in pts))
        nmx = (max(p[0] for p in pts), max(p[1] for p in pts), max(p[2] for p in pts))
        t_wps.append((nmn, nmx, fl))
        wp_centers.append((0.5 * (nmn[0] + nmx[0]), 0.5 * (nmn[1] + nmx[1]), 0.5 * (nmn[2] + nmx[2])))

    t_edges = list(asset.wp_edges)
    # Build walkable floor node index (exclude teleports/jump-only/generated)
    walk_ids = [i for i, (_, _, fl) in enumerate(t_wps) if (fl & (1 | 2 | 4 | 8 | 16 | 32)) == 0]
    if not walk_ids:
        walk_ids = list(range(len(t_wps)))

    # Inject parametric architectural geometry motifs directly onto real walkable stock geometry!
    tex_floor = pal["floor"] or "exx/floor-tread01"
    tex_wall = pal["wall"] or "exx/wall-bigrib02"
    tex_trim = pal["trim"] or "exx/trim-01"
    applied_motifs: list[str] = []
    used_pads: list[tuple[float, float, float]] = []

    def _clear_of_used(p: tuple[float, float, float], min_d: float = 240.0) -> bool:
        return all(vlen(vsub(p, u)) >= min_d for u in used_pads)

    # Compute local openness (distance to 2nd nearest neighbor) to identify wide arena spaces
    open_nodes = []
    for wi in walk_ids:
        p = wp_centers[wi]
        dists = sorted(vlen(vsub(p, wp_centers[wj])) for wj in walk_ids if wj != wi)
        if len(dists) >= 3:
            open_nodes.append((dists[1], wi, p))
    open_nodes.sort(key=lambda x: -x[0])

    # Motif 1: monolith_baffles (cover pillars / L-baffles in wide open rooms)
    if "monolith_baffles" in spec.motifs and open_nodes:
        placed = 0
        for d_open, wi, p in open_nodes:
            if placed >= 3 or d_open < 180.0:
                break
            if not _clear_of_used(p, 360.0):
                continue
            # Offset pillar slightly from the waypoint node so bots walk smoothly around it!
            px, py, pz = p[0] + 84.0, p[1] + 56.0, p[2] - 24.0
            t_brushes.append(_box_block(px - 28.0, py - 28.0, pz - 8.0, px + 28.0, py + 28.0, pz + 168.0, tex_trim, tex_wall))
            used_pads.append(p)
            placed += 1
        if placed:
            applied_motifs.append(f"monolith_baffles({placed})")

    # Motif 2: bunker_archway (fortified flanking buttresses + overhead lintel framing a lane)
    if "bunker_archway" in spec.motifs and open_nodes:
        placed = 0
        for d_open, wi, p in open_nodes:
            if placed >= 2 or d_open < 200.0:
                break
            if not _clear_of_used(p, 420.0):
                continue
            # Align archway perpendicular to nearest neighbor vector
            nbs = sorted((vlen(vsub(p, wp_centers[wj])), wj) for wj in walk_ids if wj != wi)
            if not nbs:
                continue
            nb_p = wp_centers[nbs[0][1]]
            dx, dy = nb_p[0] - p[0], nb_p[1] - p[1]
            L = math.hypot(dx, dy) or 1.0
            tx, ty = -dy / L, dx / L
            # Twin buttresses spaced 320u apart (safe clearance >= 256u!) + lintel at z+176..216
            pz = p[2] - 24.0
            for sgn in (-1.0, 1.0):
                bx, by = p[0] + tx * 160.0 * sgn, p[1] + ty * 160.0 * sgn
                t_brushes.append(_box_block(bx - 24.0, by - 24.0, pz - 8.0, bx + 24.0, by + 24.0, pz + 176.0, tex_trim, tex_wall))
            lx0, lx1 = min(p[0] - tx * 184.0, p[0] + tx * 184.0) - 24.0, max(p[0] - tx * 184.0, p[0] + tx * 184.0) + 24.0
            ly0, ly1 = min(p[1] - ty * 184.0, p[1] + ty * 184.0) - 24.0, max(p[1] - ty * 184.0, p[1] + ty * 184.0) + 24.0
            t_brushes.append(_box_block(lx0, ly0, pz + 176.0, lx1, ly1, pz + 216.0, tex_trim, tex_trim))
            used_pads.append(p)
            placed += 1
        if placed:
            applied_motifs.append(f"bunker_archway({placed})")

    # Motif 3: skybridge_overpass (solid elevated deck + ground-anchored ramps between two flat pads)
    if "skybridge_overpass" in spec.motifs and len(walk_ids) >= 6:
        best_pair = None
        for ia in walk_ids[:40]:
            pa = wp_centers[ia]
            if not _clear_of_used(pa, 260.0):
                continue
            for ib in walk_ids[:40]:
                if ib <= ia:
                    continue
                pb = wp_centers[ib]
                dh = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                dz = abs(pb[2] - pa[2])
                if 480.0 <= dh <= 760.0 and dz <= 24.0 and _clear_of_used(pb, 260.0):
                    best_pair = (ia, ib, pa, pb)
                    break
            if best_pair:
                break
        if best_pair:
            ia, ib, pa, pb = best_pair
            z_floor = min(pa[2], pb[2]) - 24.0
            z_deck = z_floor + 84.0
            ma = (pa[0] + (pb[0] - pa[0]) * 0.28, pa[1] + (pb[1] - pa[1]) * 0.28, z_deck)
            mb = (pa[0] + (pb[0] - pa[0]) * 0.72, pa[1] + (pb[1] - pa[1]) * 0.72, z_deck)
            t_brushes.extend(_ramp_blocks((pa[0], pa[1], z_floor), ma, 112.0, z_floor, tex_floor, tex_wall))
            t_brushes.extend(_ramp_blocks(ma, mb, 112.0, z_floor, tex_floor, tex_wall))
            t_brushes.extend(_ramp_blocks(mb, (pb[0], pb[1], z_floor), 112.0, z_floor, tex_floor, tex_wall))
            # Add elevated midpoint waypoint linked to ia & ib
            mid_wp = (0.5 * (ma[0] + mb[0]), 0.5 * (ma[1] + mb[1]), z_deck + 28.0)
            mid_idx = len(t_wps)
            t_wps.append((mid_wp, mid_wp, 0))
            wp_centers.append(mid_wp)
            t_edges.append((min(ia, mid_idx), max(ia, mid_idx)))
            t_edges.append((min(ib, mid_idx), max(ib, mid_idx)))
            used_pads.extend([pa, pb, mid_wp])
            applied_motifs.append("skybridge_overpass(1)")

    # Motif 4: jumppad_shaft (real trigger_push + target_position vertical booster!)
    if "jumppad_shaft" in spec.motifs and len(walk_ids) >= 6:
        j_placed = 0
        for ia in walk_ids:
            if j_placed >= 2:
                break
            pa = wp_centers[ia]
            if not _clear_of_used(pa, 320.0):
                continue
            for ib in walk_ids:
                pb = wp_centers[ib]
                dz = pb[2] - pa[2]
                dh = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                if 140.0 <= dz <= 380.0 and 180.0 <= dh <= 560.0 and _clear_of_used(pb, 220.0):
                    tname = f"{prefix}jpad_t{j_placed}"
                    pz = pa[2] - 24.0
                    # Low physical jump pad pedestal + trigger_push brush
                    t_brushes.append(_box_block(pa[0] - 36.0, pa[1] - 36.0, pz - 4.0, pa[0] + 36.0, pa[1] + 36.0, pz + 6.0, "exx/trim-01", "exx/base-metal03"))
                    trig_br = _box_block(pa[0] - 32.0, pa[1] - 32.0, pz + 4.0, pa[0] + 32.0, pa[1] + 32.0, pz + 36.0, "common/trigger")
                    t_entities.append(ParsedEntity(kv={"classname": "trigger_push", "target": tname}, brushes=[trig_br]))
                    t_entities.append(ParsedEntity(kv={
                        "classname": "target_position",
                        "targetname": tname,
                        "origin": f"{pb[0]:.1f} {pb[1]:.1f} {pb[2] + 32.0:.1f}",
                    }, brushes=[]))
                    used_pads.extend([pa, pb])
                    j_placed += 1
                    break
        if j_placed:
            applied_motifs.append(f"jumppad_shaft({j_placed})")

    # Motif 5: pickup_remix (parametric resource economy overlay)
    econ_items = PICKUP_ECONOMIES.get(spec.pickup_economy, [])
    if ("pickup_remix" in spec.motifs or econ_items) and walk_ids:
        pool = econ_items or PICKUP_ECONOMIES["heavy_ordnance"]
        picks = rng.sample(walk_ids, min(4, len(walk_ids)))
        for idx_p, wi in enumerate(picks):
            p = wp_centers[wi]
            t_entities.append(ParsedEntity(kv={
                "classname": pool[idx_p % len(pool)],
                "origin": f"{p[0]:.1f} {p[1]:.1f} {p[2] + 12.0:.1f}",
            }, brushes=[]))
        applied_motifs.append(f"pickup_remix({spec.pickup_economy or 'heavy_ordnance'})")

    # Compute exact transformed AABB
    corners = [
        xf.point(x, y, z)
        for x in (asset.bbox_min[0], asset.bbox_max[0])
        for y in (asset.bbox_min[1], asset.bbox_max[1])
        for z in (asset.bbox_min[2], asset.bbox_max[2])
    ]
    aabb_min = (min(c[0] for c in corners), min(c[1] for c in corners), min(c[2] for c in corners))
    aabb_max = (max(c[0] for c in corners), max(c[1] for c in corners), max(c[2] for c in corners))

    return {
        "tile_id": tile_id,
        "base_map": asset.name,
        "variant_label": spec.label(),
        "spec": {
            "base_map": spec.base_map, "rot_deg": spec.rot_deg, "mirror_axis": spec.mirror_axis,
            "xy_scale": spec.xy_scale, "z_scale": spec.z_scale, "palette": spec.palette,
            "motifs": applied_motifs, "pickup_economy": spec.pickup_economy,
        },
        "brushes": t_brushes,
        "entities": t_entities,
        "waypoints": t_wps,
        "wp_centers": wp_centers,
        "wp_edges": sorted(set(t_edges)),
        "walk_ids": walk_ids,
        "stock_spawns": stock_spawns,
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
    }
