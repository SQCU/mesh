import sys, time
import numpy as np
import mlx.core as mx
from mesh import Mesh

ROLE  = sys.argv[1]
PEER  = int(sys.argv[2])
SECS  = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
BATCH = 512

def weights(d):

    return mx.array(2.0 * np.eye(d, dtype=np.float32))

m = Mesh()
D = m.usable // 4
print(f"{ROLE}: {m.slots} slots, {m.usable} B usable, row width D={D}", flush=True)

if ROLE == "worker":
    W = weights(D); mx.eval(W)
    stage = np.empty((BATCH, D), dtype=np.float32)
    ring = (m.slots // BATCH) * BATCH
    cur = 0
    out, batches, t0 = 0, 0, time.time()
    while time.time() - t0 < SECS:
        n, src = 0, None
        for buf, s in m.read():
            stage[n] = buf[:D]; src = s; n += 1
            if n >= BATCH: break
        if n == 0: continue
        X = mx.array(stage[:n])
        Y = X @ W
        mx.eval(Y)
        Yn = np.asarray(memoryview(Y))
        if cur + n > ring: cur = 0
        m.block(cur, n)[:, :D*4] = Yn.view(np.uint8).reshape(n, D*4)
        sent = 0
        while sent < n:
            sent += m.write(cur + sent, n - sent, src if src is not None else PEER)
        cur += n
        out += n; batches += 1
    dt = time.time() - t0
    print(f"worker: {out} rows, {batches} batches, "
          f"{2*out*D*D/dt/1e9:.1f} GFLOP/s, {out*D*4*8/dt/1e9:.2f} Gbit/s in", flush=True)

else:

    src = np.empty((BATCH, D), dtype=np.float32)
    for i in range(BATCH):
        src[i, 0] = i
        src[i, 1:] = np.arange(D - 1, dtype=np.float32) + i
    m.block(0, BATCH)[:, :D*4] = src.view(np.uint8).reshape(BATCH, D*4)
    sent = got = bad = 0
    t0 = time.time()
    while time.time() - t0 < SECS:
        n = 0
        while n < BATCH:
            k = m.write(n, BATCH - n, PEER)
            if k == 0: break
            n += k
        sent += n
        for buf, _ in m.read():
            j = int(round(buf[0] / 2.0))
            if not (0 <= j < BATCH and np.array_equal(
                    buf[1:D], 2.0 * (np.arange(D - 1, dtype=np.float32) + j))):
                bad += 1
            got += 1
    dt = time.time() - t0
    print(f"driver: sent {sent} rows, verified {got} back, wrong {bad}, "
          f"{2*got*D*D/dt/1e9:.1f} GFLOP/s offloaded, {got*D*4*8/dt/1e9:.2f} Gbit/s back",
          flush=True)
