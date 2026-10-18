"""A stand-in for the engine: frame request blocks, print the plans that come back (ef6adb10^:xonotic/solver/sendreq.py,
onto HEAD's wire).

    PYTHONPATH=xonotic:rdma bin/mesh-python -B -m solver.sendreq CONFIG.json [--bots 64] [--ticks 4] [--json]

It speaks mesh_ipc.c's local protocol to the worker's socket (CONFIG's "socket", "framebytes", "response"): each
tick the OBSERVATION, CART, TEAM and (empty) STATE messages of one session and tick, as payload_strategy_gather
publishes them, then, where the response kind is defined, a blocking read of the rows the worker frames back.  Rows
come from a fixed seed so two runs against differently configured workers are comparable."""
import argparse, json, os, socket, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xonotic"))

from solver.xonwire import (LOCAL_HDR, Reassembler, frames, kind_of, parse_hdr,
                            OBSERVATION_KIND, CART_KIND, TEAM_KIND, STATE_KIND)
from payload.tools.strategy_io_schema import CART_WIDTH, CS, OBS, OBS_WIDTH, TEAM_WIDTH, TS

TEAMS = 5


def features(bots, tick, rng):
    X = np.zeros((bots, OBS_WIDTH), np.float32)
    X[:, OBS["ID"]] = np.arange(1, bots + 1)
    X[:, OBS["TEAM"]] = 1 + np.arange(bots) % TEAMS
    X[:, OBS["PRESENT"]] = 1
    X[:, OBS["ALIVE"]] = 1
    X[:, OBS["HEALTH"]] = 100.0 - (np.arange(bots) % 7) * 5.0
    X[:, OBS["AMMO_SHELLS"]] = 20
    X[:, OBS["POS_X"]:OBS["POS_Z"] + 1] = rng.standard_normal((bots, 3)) * 512
    X[:, OBS["VEL_X"]:OBS["VEL_Z"] + 1] = rng.standard_normal((bots, 3)) * 64
    X[:, OBS["WEAPONS_X"]] = 3
    X[:, OBS["ENGINE_TIME"]] = tick * 0.1
    return X


def carts(tick):
    C = np.zeros((2, CART_WIDTH), np.float32)
    C[:, CS["ID"]] = [0, 1]
    C[:, CS["PATH_LENGTH"]] = 1000
    C[:, CS["PATH_POSITION"]] = [(tick * 7) % 1000, (tick * 3) % 1000]
    C[:, CS["CONTROL_TEAM"]] = [1, 2]
    C[:, CS["POS_X"]] = [-256, 256]
    return C


def teams():
    T = np.zeros((TEAMS, TEAM_WIDTH), np.float32)
    T[:, TS["ID"]] = np.arange(1, TEAMS + 1)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--bots", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    config = json.loads(Path(a.config).read_text())
    response = kind_of(config["response"])

    path = f"/tmp/mesh-engine-{os.getpid()}.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    sock.bind(path)
    buffer = bytearray(sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
    usable = config["framebytes"]
    try:
        session = (os.getpid() & 0xffffff) << 24 | 1
        rng = np.random.default_rng(11)
        rx = Reassembler(response, 8, usable) if response is not None else None
        picks, replies = {}, 0
        for t in range(a.ticks):
            parts = ((OBSERVATION_KIND, features(a.bots, t, rng)), (CART_KIND, carts(t)), (TEAM_KIND, teams()),
                     (STATE_KIND, np.zeros((0, 1551), np.float32)))
            for kind, rows in parts:
                framed = frames(kind, t + 1, t + 1, rows, usable, session)
                for first in range(0, len(framed), 16):
                    part = framed[first:first + 16]
                    sock.sendmsg((LOCAL_HDR.pack(a.node, usable, len(part)), part), (), 0, config["socket"])
            while rx is not None:
                sock.recvfrom_into(buffer)
                node, framebytes, count = LOCAL_HDR.unpack_from(buffer)
                done = None
                for frame in np.frombuffer(buffer, np.uint8, framebytes * count, LOCAL_HDR.size).reshape(count, framebytes):
                    header = parse_hdr(frame)
                    if header is not None and header["kind"] == response:
                        done = rx.feed(frame) or done
                if done:
                    R = rx.stage[:done["rows"]]
                    replies += 1
                    picks[done["req_id"]] = R[:, 1].astype(int).tolist()
                    if not a.json:
                        hist = np.bincount(R[:, 1].astype(int) % TEAMS, minlength=TEAMS).tolist()
                        print(f"recv: req {done['req_id']} tick {done['tick']} rows {done['rows']} node {node} "
                              f"picks {hist} first {R[:6, 1].astype(int).tolist()}", flush=True)
                    break
        out = dict(sent=a.ticks, replies=replies, usable=usable, picks={str(k): v for k, v in picks.items()})
        print(json.dumps(out) if a.json else f"send: {replies}/{a.ticks} answered, frame {usable} bytes",
              flush=True)
    finally:
        sock.close()
        os.unlink(path)


main()
