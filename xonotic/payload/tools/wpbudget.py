#!/usr/bin/env python3
"""
wpbudget.py -- measure the real relationship between a fused megamap's saved
waypoint count, its entity count, the bot count, and DarkPlaces' compiled-in
10,000,000-jump PRVM runaway limit.

Everything here is measurement plumbing.  It builds waypoint-set variants of a
real fused map, packages them as an override pk3 into a throwaway per-run
userdir, boots the real dedicated server headless on a free port, and classifies
the outcome from the real engine log.

Owned file: this one.  It does not touch anything else under payload/tools/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# fixed locations
# ---------------------------------------------------------------------------

ENGINE = Path("/Users/mdot/dox/xonotic/build-engine/darkplaces-dedicated")
BASEDIR = Path("/Users/mdot/dox/xonotic/Xonotic")
SRC_MAPS = Path("/Users/mdot/dox/xonotic/fusebuild/prev/data/maps")
OUTDIR = Path("/Users/mdot/dox/xonotic/fusebuild/budget")
MAPNAME = "fused"
# the .ent must come from the SAME fuse run as the installed .bsp: the prev/
# artifact .ent is from a different fuse run and its warpzone targets do not
# resolve against the shipped bsp (Program error in WarpZone_InitStep_UpdateTransform).
ENT_SRC = Path("/Users/mdot/dox/xonotic/fusebuild/budget/entsrc/maps/fused.ent")

# Ports that are LIVE and must never be touched.
FORBIDDEN_PORTS = {26012, 26042}
PORT_RANGE = range(26101, 26141)

# The engine's compiled-in limit (prvm_execprogram.h:414).
RUNAWAY_LIMIT = 10_000_000

# Override pk3 must sort after zzzz-fused.pk3 so its maps/ entries win.
OVERRIDE_PK3 = "zzzzz-wpbudget.pk3"


# ---------------------------------------------------------------------------
# waypoint file model
# ---------------------------------------------------------------------------

VEC_RE = re.compile(r"^'\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*'\s*$")


def parse_vec(s: str):
    m = VEC_RE.match(s.strip())
    if not m:
        # some writers omit the quotes
        parts = s.strip().strip("'").split()
        if len(parts) != 3:
            raise ValueError(f"bad vector line: {s!r}")
        return tuple(float(p) for p in parts)
    return tuple(float(g) for g in m.groups())


@dataclass
class Waypoint:
    m1_s: str          # exact original text of the mins line
    m2_s: str          # exact original text of the maxs line
    flags_s: str       # exact original text of the flags line
    m1: tuple
    m2: tuple

    @property
    def stationary(self) -> bool:
        return self.m1 == self.m2

    @property
    def flags(self) -> int:
        return int(float(self.flags_s.strip()))

    @property
    def origin(self):
        return tuple((a + b) * 0.5 for a, b in zip(self.m1, self.m2))

    @property
    def key(self) -> str:
        """The text the .cache file uses to name this waypoint."""
        return self.m1_s.strip()


@dataclass
class WaypointSet:
    header: list          # the leading // lines, verbatim (WAYPOINT_TIME must
                          # match the cache header or the QC relinks everything)
    wps: list             # list[Waypoint]
    cache_header: list
    links: list           # list[(from_key, to_key)]

    def counts(self):
        stat = sum(1 for w in self.wps if w.stationary)
        return dict(waypoints=len(self.wps), stationary=stat,
                    boxes=len(self.wps) - stat, links=len(self.links))


def load_set(mapdir: Path, mapname: str = MAPNAME) -> WaypointSet:
    wp_lines = (mapdir / f"{mapname}.waypoints").read_text().splitlines()
    i = 0
    while i < len(wp_lines) and wp_lines[i].startswith("//"):
        i += 1
    header, body = wp_lines[:i], wp_lines[i:]
    wps = []
    for k in range(0, len(body) - 2, 3):
        a, b, f = body[k], body[k + 1], body[k + 2]
        if not a.strip():
            continue
        wps.append(Waypoint(a, b, f, parse_vec(a), parse_vec(b)))

    c_lines = (mapdir / f"{mapname}.waypoints.cache").read_text().splitlines()
    j = 0
    while j < len(c_lines) and c_lines[j].startswith("//"):
        j += 1
    c_header, c_body = c_lines[:j], c_lines[j:]
    links = []
    for ln in c_body:
        if not ln.strip():
            continue
        parts = ln.split("*")
        if len(parts) != 2:
            continue
        links.append((parts[0].strip(), parts[1].strip()))
    return WaypointSet(header, wps, c_header, links)


def write_set(ws: WaypointSet, outdir: Path, mapname: str = MAPNAME):
    outdir.mkdir(parents=True, exist_ok=True)
    out = list(ws.header)
    for w in ws.wps:
        out += [w.m1_s, w.m2_s, w.flags_s]
    (outdir / f"{mapname}.waypoints").write_text("\n".join(out) + "\n")
    cout = list(ws.cache_header)
    for a, b in ws.links:
        cout.append(f"{a}*{b}")
    (outdir / f"{mapname}.waypoints.cache").write_text("\n".join(cout) + "\n")


# ---------------------------------------------------------------------------
# resampling: decimate down (farthest-point) and densify up (link midpoints)
# ---------------------------------------------------------------------------

def d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def fmt_vec(v) -> str:
    def f(x):
        s = f"{x:.1f}"
        return s[:-2] if s.endswith(".0") else s
    return "'%s %s %s'" % (f(v[0]), f(v[1]), f(v[2]))


def decimate(ws: WaypointSet, target: int) -> WaypointSet:
    """Keep every non-stationary / flagged waypoint; farthest-point-spread the
    plain stationary ones down to `target` total; contract links onto survivors."""
    forced = [w for w in ws.wps if (not w.stationary) or w.flags != 0]
    pool = [w for w in ws.wps if w.stationary and w.flags == 0]
    n_free = max(0, target - len(forced))
    if n_free >= len(pool):
        return ws  # nothing to drop

    # farthest-point sampling seeded from the forced set (or pool[0])
    seeds = [w.origin for w in forced] or [pool[0].origin]
    best = [min(d2(p.origin, s) for s in seeds) for p in pool]
    chosen_idx = []
    if not forced:
        chosen_idx.append(0)
        best = [d2(p.origin, pool[0].origin) for p in pool]
    while len(chosen_idx) < n_free:
        k = max(range(len(pool)), key=lambda i: best[i])
        if best[k] <= 0 and len(chosen_idx) > 0:
            # degenerate duplicates left; just take them in order
            remaining = [i for i in range(len(pool)) if i not in set(chosen_idx)]
            chosen_idx += remaining[: n_free - len(chosen_idx)]
            break
        chosen_idx.append(k)
        ok = pool[k].origin
        for i, p in enumerate(pool):
            dd = d2(p.origin, ok)
            if dd < best[i]:
                best[i] = dd
    chosen = set(chosen_idx)
    survivors = forced + [pool[i] for i in sorted(chosen)]

    # rebuild in the original file order for stability
    keep = {id(w) for w in survivors}
    new_wps = [w for w in ws.wps if id(w) in keep]
    return _contract(ws, new_wps)


def _contract(ws: WaypointSet, new_wps: list) -> WaypointSet:
    """Redirect every dropped node's links to its nearest survivor."""
    surv_keys = {w.key for w in new_wps}
    surv_list = new_wps
    by_key = {w.key: w for w in ws.wps}

    remap = {}
    for w in ws.wps:
        if w.key in surv_keys:
            remap[w.key] = w.key
        else:
            nearest = min(surv_list, key=lambda s: d2(s.origin, w.origin))
            remap[w.key] = nearest.key

    seen = set()
    new_links = []
    for a, b in ws.links:
        ra = remap.get(a)
        rb = remap.get(b)
        if ra is None or rb is None:
            continue
        if ra == rb:
            continue          # self-link after contraction: drop
        if (ra, rb) in seen:
            continue
        seen.add((ra, rb))
        new_links.append((ra, rb))
    return WaypointSet(ws.header, new_wps, ws.cache_header, new_links)


def densify(ws: WaypointSet, target: int) -> WaypointSet:
    """Grow the set past the shipped count by subdividing the longest links.
    Each inserted waypoint sits on the midpoint of a link that the mapper's own
    linker already declared walkable, so the graph stays real: A-B becomes
    A-M and M-B (in both directions where both were present)."""
    wps = list(ws.wps)
    links = list(ws.links)
    by_key = {w.key: w for w in wps}
    if len(wps) >= target:
        return ws

    # work on a mutable adjacency; repeatedly split the geometrically longest
    # undirected link that has not been split yet
    def link_len(l):
        a, b = by_key.get(l[0]), by_key.get(l[1])
        if a is None or b is None:
            return -1.0
        return d2(a.origin, b.origin)

    import heapq
    heap = []
    for idx, l in enumerate(links):
        if l[0] < l[1]:
            heapq.heappush(heap, (-link_len(l), l[0], l[1]))
    linkset = set(links)

    added = 0
    guard = 0
    while len(wps) < target and heap and guard < 200000:
        guard += 1
        neg, a, b = heapq.heappop(heap)
        wa, wb = by_key.get(a), by_key.get(b)
        if wa is None or wb is None:
            continue
        mid = tuple((x + y) * 0.5 for x, y in zip(wa.origin, wb.origin))
        key = fmt_vec(mid)
        if key in by_key:
            continue
        mw = Waypoint(key, key, "0", parse_vec(key), parse_vec(key))
        wps.append(mw)
        by_key[key] = mw
        added += 1
        # rewire: for each direction that existed, route it via the midpoint
        for (u, v) in ((a, b), (b, a)):
            if (u, v) in linkset:
                linkset.discard((u, v))
                linkset.add((u, key))
                linkset.add((key, v))
        for (u, v) in ((a, key), (key, b)):
            if u < v:
                heapq.heappush(heap, (-d2(by_key[u].origin, by_key[v].origin), u, v))
            else:
                heapq.heappush(heap, (-d2(by_key[u].origin, by_key[v].origin), v, u))

    # keep link list ordered deterministically
    new_links = [l for l in links if l in linkset]
    extra = [l for l in sorted(linkset) if l not in set(links)]
    return WaypointSet(ws.header, wps, ws.cache_header, new_links + extra)


def resample(ws: WaypointSet, target: int) -> WaypointSet:
    if target >= len(ws.wps):
        return densify(ws, target)
    return decimate(ws, target)


# ---------------------------------------------------------------------------
# entity file
# ---------------------------------------------------------------------------

ENT_BLOCK_RE = re.compile(r"\{.*?\}", re.S)


def load_ents(path: Path) -> list:
    txt = path.read_text(errors="replace")
    return ENT_BLOCK_RE.findall(txt)


def ent_classname(block: str) -> str:
    m = re.search(r'"classname"\s+"([^"]*)"', block)
    return m.group(1) if m else ""


def resize_ents(blocks: list, target: int) -> list:
    """Trim toward `target`, or pad by cloning pickups when asked to grow.

    Trimming must never break a target/targetname reference: dropping one half of
    a warpzone pair aborts the server in WarpZone_InitStep_UpdateTransform long
    before any budget question can be asked.  So every block whose targetname is
    referenced by some other block is protected, as is every referrer."""
    ESSENTIAL = {"worldspawn", "info_player_deathmatch", "info_player_start",
                 "info_player_team1", "info_player_team2", "info_player_team3",
                 "info_player_team4", "trigger_teleport", "misc_teleporter_dest",
                 "target_position", "func_door", "trigger_multiple",
                 "trigger_hurt", "target_objective",
                 "trigger_warpzone", "func_camera", "trigger_warpzone_position",
                 "trigger_push", "target_push"}
    if len(blocks) <= target:
        # grow: clone item/weapon pickups.  They carry no targetname, so cloning
        # cannot duplicate a reference, and each one is a real edict with a real
        # spawnfunc -- exactly the cost the budget question is about.
        clones = [b for b in blocks
                  if ent_classname(b).startswith(("item_", "weapon_"))
                  and '"targetname"' not in b and '"target"' not in b]
        if not clones:
            return list(blocks)
        out = list(blocks)
        i = 0
        while len(out) < target:
            src = clones[i % len(clones)]
            k = len(out)

            def bump(m, k=k):
                x, y, z = (float(v) for v in m.group(1).split())
                return '"origin" "%g %g %g"' % (x + (k % 7) * 4, y + (k % 5) * 4, z)
            out.append(re.sub(r'"origin"\s+"([^"]*)"', bump, src, count=1))
            i += 1
        return out

    # shrink
    named = set()
    for b in blocks:
        for k in ("target", "target2", "target3", "target4", "killtarget"):
            for mo in re.finditer(r'"%s"\s+"([^"]+)"' % k, b):
                named.add(mo.group(1))
    keep_idx, pool = [], []
    for i, b in enumerate(blocks):
        cn = ent_classname(b)
        tn = re.search(r'"targetname"\s+"([^"]+)"', b)
        referenced = bool(tn and tn.group(1) in named)
        refers = any('"%s"' % k in b for k in
                     ("target", "target2", "target3", "target4", "killtarget"))
        if cn in ESSENTIAL or cn.startswith("info_player") or referenced or refers:
            keep_idx.append(i)
        else:
            pool.append(i)
    need_drop = len(blocks) - target
    if need_drop >= len(pool):
        return [blocks[i] for i in keep_idx]
    step = len(pool) / need_drop
    dropped = {pool[int(i * step)] for i in range(need_drop)}
    return [b for i, b in enumerate(blocks) if i not in dropped]


def write_ents(blocks: list, path: Path):
    path.write_text("\n".join(blocks) + "\n")


# ---------------------------------------------------------------------------
# run harness
# ---------------------------------------------------------------------------

def port_free(p: int) -> bool:
    if p in FORBIDDEN_PORTS:
        return False
    r = subprocess.run(["lsof", "-nP", f"-iTCP:{p}"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        return False
    r2 = subprocess.run(["lsof", "-nP", f"-iUDP:{p}"],
                        capture_output=True, text=True)
    if r2.stdout.strip():
        return False
    return True


def pick_port() -> int:
    for p in PORT_RANGE:
        if port_free(p):
            return p
    raise RuntimeError("no free port in 26101-26140")


@dataclass
class RunResult:
    label: str = ""
    n_waypoints: int = 0
    n_links: int = 0
    n_ents: int = 0
    bots: int = 0
    port: int = 0
    outcome: str = ""           # boot_ok | runaway | other_error | timeout | launch_fail
    seconds: float = 0.0
    match_started: bool = False
    profile: list = field(default_factory=list)   # top PRVM profile lines
    errors: list = field(default_factory=list)
    log: str = ""
    norunaway: bool = False
    soak_seconds: float = 0.0
    soak_survived: bool = False
    exit_code: int = -999
    match_start_s: float = 0.0
    bots_connected: int = 0
    ai_seconds: float = 0.0
    notes: list = field(default_factory=list)


RUNAWAY_RE = re.compile(r"runaway loop counter hit limit of (\d+) jumps")
HOSTERR_RE = re.compile(r"^Host_Error:?\s*(.*)$", re.M)
OBJERR_RE = re.compile(r"^.*(OBJECT ERROR|assertion failed|VM_remove|Bad boxes|SVQC error).*$", re.M)
# PRVM profile rows: "<statements> <builtins> <calls> ... functionname"
PROF_ROW_RE = re.compile(r"^\s*(\d{4,})\s+\d+\s+\d+.*?([A-Za-z_][\w:.]*)\s*$", re.M)


def build_variant(rundir: Path, ws: WaypointSet, ent_blocks=None) -> dict:
    """Write a per-run userdir whose override pk3 shadows maps/fused.waypoints
    (+ optionally maps/fused.ent) from the installed zzzz-fused.pk3."""
    data = rundir / "data"
    stage = rundir / "stage" / "maps"
    if stage.exists():
        shutil.rmtree(rundir / "stage")
    stage.mkdir(parents=True)
    data.mkdir(parents=True, exist_ok=True)
    write_set(ws, stage)
    files = [f"{MAPNAME}.waypoints", f"{MAPNAME}.waypoints.cache"]
    if ent_blocks is not None:
        write_ents(ent_blocks, stage / f"{MAPNAME}.ent")
        files.append(f"{MAPNAME}.ent")
    pk3 = data / OVERRIDE_PK3
    with zipfile.ZipFile(pk3, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(stage / f, f"maps/{f}")
    return dict(pk3=str(pk3), files=files)


def run_server(rundir: Path, bots: int, *, timeout=300.0, soak=0.0,
               norunaway=False, timelimit=5, label="") -> RunResult:
    port = pick_port()
    logf = rundir / "server.log"
    cmd = [str(ENGINE), "-xonotic",
           "-basedir", str(BASEDIR),
           "-userdir", str(rundir)]
    if norunaway:
        cmd.append("-norunaway")
    cmd += ["+developer", "0", "+sv_public", "0",
            "+port", str(port), "+sv_autopause", "0",
            "+maxplayers", "32", "+bot_join_empty", "1",
            "+bot_number", str(bots), "+skill", "5",
            "+g_warmup", "0", "+timelimit", str(timelimit),
            "+map", MAPNAME]

    res = RunResult(label=label, bots=bots, port=port, norunaway=norunaway)
    t0 = time.time()
    fh = open(logf, "wb")
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True)
        hold = soak if soak > 0 else timeout
        deadline = t0 + hold + 30.0
        while True:
            rc = proc.poll()
            now = time.time()
            if rc is not None:
                res.seconds = now - t0
                res.exit_code = rc
                break
            txt = _tail(logf)
            nc = txt.count(" connected")
            if not res.match_started and nc >= max(1, bots):
                res.match_started = True
                res.match_start_s = now - t0
            res.ai_seconds = (now - t0 - res.match_start_s) if res.match_started else 0.0
            if now - t0 >= hold:
                res.seconds = now - t0
                res.soak_seconds = now - t0
                res.soak_survived = True
                _kill(proc)
                break
            if now >= deadline:
                res.outcome = "timeout"
                res.seconds = now - t0
                _kill(proc)
                break
            time.sleep(1.0)
    finally:
        if proc is not None:
            _kill(proc)
        fh.close()

    log = logf.read_text(errors="replace")
    res.log = str(logf)
    res.n_ents = 0
    _classify(res, log)
    return res


def _tail(p: Path, nbytes=200000) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            sz = f.tell()
            f.seek(max(0, sz - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


MATCH_MARKERS = ("Match has already begun", ":gamestart", "player is playing now",
                 ":join:bot", "connected", "Sv_")


def _match_running(txt: str) -> bool:
    # the reliable marker for "the map loaded and the sim is ticking"
    return (":gamestart:" in txt) or ("Client \"" in txt and ":join:" in txt) \
        or ("Bot" in txt and ":join:bot" in txt)


def _kill(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:
            pass
    for _ in range(40):
        if proc.poll() is not None:
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def _classify(res: RunResult, log: str):
    m = RUNAWAY_RE.search(log)
    if m:
        res.outcome = "runaway"
        res.errors.append(m.group(0))
        # the PRVM profile block the engine dumps immediately above it
        head = log[: m.start()]
        rows = PROF_ROW_RE.findall(head[-20000:])
        rows = [(int(a), b) for a, b in rows]
        rows.sort(reverse=True)
        res.profile = [f"{a} {b}" for a, b in rows[:15]]
        return
    he = HOSTERR_RE.findall(log)
    if he:
        res.errors += [f"Host_Error: {x}" for x in he[:5]]
    others = OBJERR_RE.findall(log)
    if others:
        res.notes += sorted(set(x.strip()[:120] for x in others))[:8]
    # the process surviving the whole hold window is the only success criterion;
    # the spawnpoint-in-solid nags are non-fatal and appear in every single run
    res.bots_connected = log.count(" connected")
    if not res.outcome:
        if res.soak_survived and res.match_started:
            res.outcome = "boot_ok"
        elif res.soak_survived:
            res.outcome = "alive_no_bots"
        elif he:
            res.outcome = "other_error"
        else:
            res.outcome = "exited_%s" % res.exit_code



# ---------------------------------------------------------------------------
# statement-cost profiling
#
# PRVM_Profile() zeroes every counter after it prints (prvm_exec.c:418-425), so
# two `prvm_profile server` dumps bracket an interval and the second dump is a
# pure delta over it.  The row for StartFrame then reads:
#     callcount    = number of StartFrame entry points in the window
#     profile_total = statements executed inside those entry points (incl. callees)
# and profile_total / callcount is exactly the per-entry-point statement cost that
# the engine's 10,000,000-jump budget is spent against.
# ---------------------------------------------------------------------------

# [CallCount] [Statement] [BuiltinCt] [StmtTotal] [BltnTotal] [self] name
PROFROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[-\d.]+%\s+(\S+)\s*$", re.M)


def parse_profile(txt: str) -> dict:
    """Parse the LAST 'server Profile:' block in txt into {name: dict}."""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    idx = txt.rfind("server Profile:")
    if idx < 0:
        return {}
    block = txt[idx:]
    out = {}
    for cc, st, bc, stt, btt, name in PROFROW_RE.findall(block):
        out[name] = dict(callcount=int(cc), statements=int(st),
                         builtins=int(bc), stmt_total=int(stt),
                         bltn_total=int(btt))
    return out


def run_profile(rundir: Path, bots: int, *, settle=90.0, window=30.0,
                norunaway=True, label="") -> dict:
    """Boot, let it settle, reset the profile, run `window` seconds, dump."""
    port = pick_port()
    logf = rundir / "server.log"
    cmd = [str(ENGINE), "-xonotic", "-basedir", str(BASEDIR),
           "-userdir", str(rundir)]
    if norunaway:
        cmd.append("-norunaway")
    cmd += ["+developer", "0", "+sv_public", "0", "+port", str(port),
            "+sv_autopause", "0", "+maxplayers", "32", "+bot_join_empty", "1",
            "+bot_number", str(bots), "+skill", "5", "+g_warmup", "0",
            "+timelimit", "20", "+map", MAPNAME]
    fh = open(logf, "wb")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE, start_new_session=True)
    rec = dict(label=label, bots=bots, port=port, settle=settle, window=window)
    try:
        t0 = time.time()
        # wait for the bots to be in and playing
        while time.time() - t0 < settle:
            if proc.poll() is not None:
                rec["died_before_profile"] = True
                break
            time.sleep(1.0)
        if proc.poll() is None:
            proc.stdin.write(b"prvm_profile server 1\n")   # reset counters
            proc.stdin.flush()
            time.sleep(2.0)
            mark = os.path.getsize(logf)
            t1 = time.time()
            while time.time() - t1 < window and proc.poll() is None:
                time.sleep(1.0)
            rec["window_actual"] = time.time() - t1
            if proc.poll() is None:
                proc.stdin.write(b"prvm_profile server 3000\n")
                proc.stdin.flush()
                time.sleep(3.0)
            rec["mark"] = mark
    finally:
        try:
            if proc.stdin:
                proc.stdin.write(b"quit\n")
                proc.stdin.flush()
        except Exception:
            pass
        time.sleep(2.0)
        _kill(proc)
        fh.close()
    txt = logf.read_text(errors="replace")
    mark = rec.get("mark", 0)
    rec["profile"] = parse_profile(txt[mark:] if mark else txt)
    rec["log"] = str(logf)
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def prep_rundir(root: Path, tag: str) -> Path:
    d = root / f"run-{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "data").mkdir(parents=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["info", "variant", "run", "ladder-n",
                                     "ladder-b", "ents", "soak", "profile"])
    ap.add_argument("-n", type=int, default=None, help="target waypoint count")
    ap.add_argument("-b", "--bots", type=int, default=12)
    ap.add_argument("-e", "--ents", type=int, default=None, help="target entity count")
    ap.add_argument("--ns", type=str, default="300,450,600,750,900,1200,1600")
    ap.add_argument("--bs", type=str, default="2,8,12,16,24")
    ap.add_argument("--soak", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--norunaway", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--mappk3", default=None,
                    help="profile a DIFFERENT fused map: this pk3 is dropped into "
                         "the run userdir under a name that sorts after the "
                         "installed zzzz-fused.pk3, so its maps/fused.* shadow it. "
                         "The map's own shipped waypoints are used verbatim (no "
                         "decimation, no densification).")
    ap.add_argument("--settle", type=float, default=90.0)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--out", default=str(OUTDIR))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ws = load_set(SRC_MAPS)

    if a.mode == "info":
        print(json.dumps(ws.counts(), indent=2))
        return

    def one(n, bots, ents=None, soak=0.0, norunaway=False, tag=None):
        tag = tag or f"n{n}b{bots}" + (f"e{ents}" if ents else "") + \
            ("-nr" if norunaway else "")
        v = resample(ws, n) if n is not None else ws
        rd = prep_rundir(out, tag)
        eb = None
        if ents is not None:
            eb = resize_ents(load_ents(ENT_SRC), ents)
        build_variant(rd, v, eb)
        r = run_server(rd, bots, timeout=a.timeout, soak=soak,
                       norunaway=norunaway, label=tag)
        r.n_waypoints = len(v.wps)
        r.n_links = len(v.links)
        r.n_ents = len(eb) if eb is not None else 0
        rec = asdict(r)
        (out / f"result-{tag}.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps({k: rec[k] for k in
                          ("label", "n_waypoints", "n_links", "n_ents", "bots",
                           "outcome", "seconds", "match_started",
                           "soak_survived")}), flush=True)
        if r.profile:
            print("  profile:", "; ".join(r.profile[:6]), flush=True)
        if r.errors:
            print("  errors:", " | ".join(r.errors[:3])[:400], flush=True)
        return rec

    results = []
    if a.mode == "variant":
        v = resample(ws, a.n)
        d = out / "variant"
        write_set(v, d)
        print(json.dumps(v.counts(), indent=2))
        return
    if a.mode == "profile":
        recs = []
        for b in [int(x) for x in a.bs.split(",")]:
            if a.mappk3:
                src = Path(a.mappk3)
                tag = (a.tag or "prof") + f"-b{b}"
                rd = prep_rundir(out, tag)
                shutil.copy2(src, rd / "data" / OVERRIDE_PK3)
                mw = load_set(src.parent)          # the map's own shipped set
                v = mw
            else:
                v = resample(ws, a.n) if a.n is not None else ws
                tag = (a.tag or "prof") + f"-n{len(v.wps)}b{b}"
                rd = prep_rundir(out, tag)
                eb = None
                if a.ents is not None:
                    eb = resize_ents(load_ents(ENT_SRC), a.ents)
                build_variant(rd, v, eb)
            r = run_profile(rd, b, settle=a.settle, window=a.window, label=tag)
            r["n_waypoints"] = len(v.wps)
            r["n_links"] = len(v.links)
            r["n_ents"] = len(eb) if eb is not None else 0
            recs.append(r)
            sf = r["profile"].get("StartFrame")
            top = sorted(r["profile"].items(),
                         key=lambda kv: -kv[1]["statements"])[:8]
            print(json.dumps(dict(label=tag, n=len(v.wps), bots=b,
                                  StartFrame=sf,
                                  window=r.get("window_actual"))), flush=True)
            for k, vv in top:
                print(f"    {vv['statements']:>12} stmt  {vv['callcount']:>8} calls  {k}",
                      flush=True)
            (out / f"profile-{tag}.json").write_text(json.dumps(r, indent=2))
        (out / f"profiles-{a.tag or 'prof'}.json").write_text(json.dumps(recs, indent=2))
        return
    if a.mode in ("run", "soak"):
        results.append(one(a.n, a.bots, a.ents,
                           soak=(a.soak or (180.0 if a.mode == "soak" else 0.0)),
                           norunaway=a.norunaway, tag=a.tag))
    elif a.mode == "ladder-n":
        fails = 0
        for n in [int(x) for x in a.ns.split(",")]:
            rec = one(n, a.bots, a.ents, norunaway=a.norunaway)
            results.append(rec)
            fails = fails + 1 if rec["outcome"] != "boot_ok" else 0
            if fails >= 2:
                print("two consecutive failures; stopping the n ladder", flush=True)
                break
    elif a.mode == "ladder-b":
        for b in [int(x) for x in a.bs.split(",")]:
            results.append(one(a.n, b, a.ents, norunaway=a.norunaway))
    elif a.mode == "ents":
        for e in [int(x) for x in a.ns.split(",")]:
            results.append(one(a.n, a.bots, e, norunaway=a.norunaway))

    name = a.tag or a.mode
    (out / f"results-{name}.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}/results-{name}.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
