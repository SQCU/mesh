import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def source_capacity(path, name, fallback):
    values = []
    try:
        with open(path) as handle:
            for line in handle:
                fields = line.replace("=", " ").replace(";", " ").split()
                if name in fields:
                    values.extend(int(field) for field in fields[fields.index(name) + 1:] if field.isdigit())
    except OSError as error:
        print(json.dumps({"event":"source_capacity_read_error","path":path,"name":name,"fallback":fallback,"error":f"{type(error).__name__}: {error}"}), file=sys.stderr)
    return max(values) if values else int(fallback)

def engine_player_capacity(fallback):
    return source_capacity(os.path.join(ROOT, "darkplaces-work", "quakedef.h"), "MAX_SCOREBOARD", fallback)

def team_capacity(fallback):
    return source_capacity(os.path.join(ROOT, "qcsrc", "common", "teams.qh"), "NUM_TEAMS", fallback)

def cart_capacity(fallback):
    return source_capacity(
        os.path.join(ROOT, "qcsrc", "common", "gamemodes", "gamemode", "payload", "payload.qh"),
        "PLC_MAX_CARTS", fallback,
    )

__all__ = ["cart_capacity", "engine_player_capacity", "source_capacity", "team_capacity"]
