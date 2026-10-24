#!/usr/bin/python3
# Each fabric port, its RDMA device and whether the IORegistry holds its RDMA peer: "PORT device=D rid=R rdma_peer=..."
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshlib  # noqa: E402

meshlib.system_path()
for port, device, rid, peer in meshlib.rdma():
    print("%s device=%s rid=%s rdma_peer=%s" % (port, device, rid, peer))
