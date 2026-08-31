from __future__ import annotations

RESP_WIDTH = 4
OBS_WIDTH = 40
CART_WIDTH = 12
EVT_WIDTH = 6
EVT_ROWS = 256
MAX_CARTS = 4

OBS = dict(ID=0, TEAM=1, HEALTH=2, ARMOR=3, AMMO=4, POS_X=5, POS_Y=6, POS_Z=7,
           VEL_X=8, VEL_Y=9, VEL_Z=10, WEAPONS=11, POWER=12, TSS=13, CELL=14,
           NCART=15, NCART_D=16, ALIVE=17, CONTROL=18, APPLIED_TARGET=19,
           TARGET_RESOLVED=20, GOAL_TARGET=21, GOAL_DISTANCE=22, GOAL_MATCH=23,
           TARGET_TOUCH=24)
CS = dict(ID=0, DEPTH=1, LENGTH=2, CTRL=3, SPEED=4, IDLE=5, BANKMASK=6, PROGRESS=7,
          POS_X=8, POS_Y=9, POS_Z=10)
EVT = dict(CELL=0, KIND=1, TEAM=2, SUBJECT=3, VALUE=4, TIME=5)
EVT_KIND = dict(ITEM_GONE=0, ITEM_HERE=1, ENEMY_HERE=2, RIVAL_HERE=3,
                # cell -> cell navigable adjacency from the stock waypoint graph;
                # CELL is the source cell, SUBJECT the destination, VALUE the
                # link length / 1024. Map geometry, not perception.
                CELL_LINK=4)
SC = dict(TARGET=0, GAIN=1, COMMIT=2, SPAWN=3)

TGT_CART_BASE = 0
TGT_ITEM_BASE = 65536
TGT_RIVAL_BASE = 131072
TGT_CELL_BASE = 196608


def encode_target(kind, index):
    return {"cart": TGT_CART_BASE, "item": TGT_ITEM_BASE,
            "rival": TGT_RIVAL_BASE, "cell": TGT_CELL_BASE}[kind] + int(index)


def decode_target(target):
    target = int(target)
    if target >= TGT_CELL_BASE:
        return "cell", target - TGT_CELL_BASE
    if target >= TGT_RIVAL_BASE:
        return "rival", target - TGT_RIVAL_BASE
    if target >= TGT_ITEM_BASE:
        return "item", target - TGT_ITEM_BASE
    return "cart", target


def _selftest():
    assert RESP_WIDTH == len(SC)
    assert max(OBS.values()) < OBS_WIDTH and max(CS.values()) < CART_WIDTH and max(EVT.values()) < EVT_WIDTH
    for kind, index in (("cart", 3), ("item", 32767), ("rival", 32767), ("cell", 65535)):
        assert decode_target(encode_target(kind, index)) == (kind, index)


if __name__ == "__main__":
    _selftest()
