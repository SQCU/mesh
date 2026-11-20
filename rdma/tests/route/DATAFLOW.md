# The trainer's data flow: as built (8168f512), and consolidated

Marks: **[S]** a synchronization (a lock taken, a wait, a wake, a GPU completion waited on the CPU); **[G]** a guard
(a check that branches or stops); **[A]** an allocation or copy made per item (per run, per refresh, per case), not
held from the start. Rates are one rank's in the pyport2 run (about 6,600 transactions/s, about 150 learning steps/s,
about 19 optimizer steps/s).

## As built

Five kinds of thread hand data to each other one item at a time, each hand-off under a lock.

```
PYTHON SUPPLY (one thread)
  loop:
    n = room(timeout)            [S] caseLock (+ shareLock nested to read `finished`), timed wait
    got = ended(states)          [S] caseLock   [A] endedStates.removeFirst: shifts the array
    pool.add(states)             per state, a Python loop
    cases = draw(n)              numpy, per batch (fine: one allocation for the batch)
    push(cases)                  [S] caseLock   [A] each case's 12 KB struct appended
                                 [S] R0.wake: R0.lock + broadcast

SERVICE (one thread): the forwards and the group bookkeeping
  loop:
    read finished, published     [S] shareLock
    if theta changed:
      snapshot -> staging        [S] shareLock, memcpy of P floats
      fnet.refresh(from:)        [S] commit + waitUntilCompleted, on THIS thread, queued BEHIND the forwards in flight:
                                     the forwards stop for every optimizer step   <-- the trade-off seen
    while R0.runnable < workers: [S] R0.lock each test
      take a case                [S] caseLock  [A] cases.removeFirst: shifts up to 64 x 12 KB
      place(case)                [A] Arrays for h and keys; slot state Array copies
      submit(4 slots)            [S] R0.lock + broadcast (wakes every worker)
    telemetry                    [S] caseLock
    R0.service():
      wait for any of 4 conditions         [S] R0.lock wait on a compound predicate
      [A] filter/map arrays of the sets and runs
      for each completed forward:
        per run: state copy, answer = Array(w rows)    [A] nc x FS floats a run (22 KB)
        c.busy = false                                 [S] R0.lock
        submit(runs)                                   [S] R0.lock + broadcast
      pack the waiting runs into a free set:
        per run: Kept(lanes, tokens)                   [A] L x IN + nc x IN floats a run a refresh (33 KB)
                                                       [G] a forward past its rows: the rest wait for the next
      forwardAsync                                     [A] a command buffer and an encoder a call
        completion handler                             [S] R0.lock + broadcast
    for each ended run:
      states += state                                  [A]
      group done: DoneRun (copies of kept, draws)      [A]
        freeGroups.append                              [S] caseLock + broadcast
        ready.append                                   [S] shareLock + signal
      endedStates += states                            [S] caseLock + broadcast

WORKERS (every hardware thread): one run each pass
  loop:
    wait for a run               [S] R0.lock wait
    answer, decide (sim.c)       the hot loop
    if ended: exportDraws        [A] per refresh: draws, ctx, first, w Arrays
    finished/awaiting.append     [S] R0.lock + broadcast: wakes every worker and the service thread
                                     (two lock hand-offs and a thundering wake per run per refresh)

LEARNER (one thread)
  loop:
    wait for ready groups        [S] shareLock wait
    batch                        [A] removeFirst, flatMap, map
    gather                       memsets; per row loops; [A] grow() when a batch needs more
    net.learn                    [A] room() growth   [S] commit + waitUntilCompleted
    importance ratios            a CPU loop over the GPU's output (needs the wait above)
    exchange.add(grads)          [S] cond + broadcast
    for each mean ready:         [S] cond
      apply: gram -> UPGrad      on the CPU (needs the gram on the CPU)
             adam, pin, codes    [S] commit + waitUntilCompleted
    publish theta                [S] shareLock, memcpy of P floats
    R0.wake                      [S] R0.lock + broadcast

EXCHANGE (one thread)
  loop:
    wait for a step              [S] cond wait
    payload <- accumulator       [S] cond
    meshSum: lease copy, encode  [A] a command buffer   [S] commit + waitUntilCompleted (the mesh's round trip)
    mean, gram                   [A] Arrays; the Gram matrix on the CPU (9 dot products over P, in doubles)
    done.append                  [S] cond + broadcast
```

Per run per refresh: four lock hand-offs (worker, service, twice), two broadcasts, and 55 KB copied into new arrays.
Per optimizer step: three CPU waits on GPU completion (learn, apply, the producers' refresh); the refresh's wait stalls
the forwards. Per exchange round: one CPU wait on the mesh, and the gradient copied three times (accumulator, payload,
lease). The rates the threads reach are set by which wait they meet, not by the work: raising the optimizer's rate
lowers the producers' (40 optimizer steps/s with 22,400 transactions/s; 19 with 26,400).

## Consolidated

One schedule, written once (Python), over arrays held from the start; native calls are whole-array kernels; the GPU's
order is its queues' order and a shared event, never a CPU wait, except where the CPU reads what the GPU made.

```
HELD FROM THE START (sized by the configuration)
  sims[S]                         S slots (the runs in flight)
  arena: lanes[S][K][L][IN], tokens[S][K][C][IN], w[S][K][C][FS], draws[S][K][...]   shared memory: the simulator writes
                                  its refresh's tokens straight into the arena row the forward reads, and its draws into
                                  the arena the credit reads; nothing is packed or copied
  codes[2]                        the learner's int8 weights, two sets alternating (the forwards read one, the learner
                                  writes the other), an MTLSharedEvent the step's value

SCHEDULE (Python, one loop, two halves of the slots alternating so the CPU simulates one while the GPU forwards the other)
  loop, half h:
    wait(forward[h])              [S] the one data dependency: this half's w
    sims_step(h's slots)          one fork-join over the half (dispatch_apply in C): answer, decide, write tokens/draws
                                  into the arena; returns status[] and nc[]
    ended = status == 0           numpy
    refill ended slots from cases[cursor:]   numpy row assignment; cases drawn a batch at a time (draw(n), n the ended)
    rows = arena indices of the pending (numpy cumsum of nc)
    forward[h] = commit(encodeWait(codes event), forward over rows)   no CPU wait
    if ended runs make a learning batch:
      learner queue: index arrays into the arena (no copies)

LEARNER (GPU queue order only)
  one command buffer a step:
    learn over the batch's arena rows (credit, backward, gram)
    exchange (world > 1): route_exchange_encode into the same command buffer (the mesh's step is Metal-encodable)
    UPGrad from the gram, Adam, pin, codes into codes[step % 2]    on the GPU
    signal event = step
  the CPU commits and goes on; it waits only when it reads (telemetry's ratios, a saved theta)
```

Per run per refresh: no lock, no copy (the arena), one share of one fork-join. Per optimizer step: no CPU wait (queue
order and the event carry theta to the forwards). Per exchange: encoded with the step, no thread. The remaining syncs
are the data's own: a half's forward before its simulation, a learning batch's runs ended before it.

## Open, to settle while building it

- The lease (route_exchange_buffer) is the node region's memory; the learner's command buffer can copy the gradients
  into it only through an MTLBuffer over that memory (makeBuffer(bytesNoCopy:), page-aligned). If the region cannot be
  wrapped, the gradients are produced into the lease directly (it is shared memory) rather than copied.
- An exchange inside the step's command buffer makes each step wait, on the GPU, for every rank's same step: the
  accumulating exchange (as built) let a slow rank's steps fold into one round instead. Which costs less is measured,
  both forms, on the four Studios (D9), before the other is dropped.
