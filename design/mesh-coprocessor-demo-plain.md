# The whole-mesh instrument, in plain words

This is the plain-language companion to
[`mesh-coprocessor-demo.md`](mesh-coprocessor-demo.md).

Every connected computer publishes the same small stream: how its shared pages are
owned, how many bytes cross the fabric, how busy its CPU and GPU are, how much memory
traffic it is producing, and what operation counts a running workload can state. The
viewer combines only currently reachable leased nodes. A disconnected node remains in
inventory but stops contributing live capacity.

The phase-space picture uses all four page states. A point near `FREE`, `RECV`, `SEND`,
or `APP` means page ownership is concentrated there. Each computer gets its own anchor
and nearby trajectory, so two machines do not collapse into one dot. Lines show reachable
topology. Moving particles require measured traffic.

FLOP/s and byte/s need not be falsely exact. Known operations supply a lower count;
algorithm and activity envelopes supply an upper count. The viewer reports both,
including variance and missing observations. That is enough to see whether a pipeline is
compute-bound, bandwidth-bound, waiting on the link, or missing its real-time deadline.

The workload chooses its own size. For Xonotic, players may join and leave while the
server runs. The profiler changes player count, observes the result, and searches around
the hardware response. It then varies team and cart counts around that measured point.
It never chooses a desired client count first and bends the algorithm until the numbers
coincide.

The strategy policy contains three different inner-product matrices, each with an exact
feature construction: one mixes participants, one mixes residual feature coordinates,
and one supplies DPP instrument diversity. They are all genuine Gram matrices. Other
score tables, checkpoints, and adjusted solve matrices are not called Gram matrices.

Part of the policy runs on the responder node and the residual expert part runs on
another node. Training follows the same placement as inference: backward rows and scale
parameter updates execute on the expert host instead of returning to a hidden local
implementation. The transport copies the literal tensor rows and does not normalize,
sum, interpret, or cap them.

Xonotic makes the result visible. Teams push and suppress many carts through a fused
stock-map world. Carts have continuous collision-free and standable paths. The server
records exact actions and exact successor state, so the J-lens and J-oracle can measure
how the learned representation relates to later behavior. Sparse win/loss transitions
train the policy; damage, kills, pressure, survival, cart movement, and robustness under
perturbations remain separate observations.

The same mesh observer would work for OCR, simulation, compilation, or another solver.
It reports the fabric and the work placed on it, indifferent to what those rows mean.
