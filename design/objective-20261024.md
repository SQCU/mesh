# Whole-function performance objective

For a fixed function and shape, minimize elapsed time and maximize useful output per second. These are the same
objective when the output extent is fixed. Count the logical output once. Extra relay traffic is not extra output.

Congestion, link capacity, and estimates of fixed and work-dependent cost inform the next page assignment. They do
not introduce a second weighted objective. The compiler can choose among all physical paths, including parallel
cables and relays. Estimates must be corrected by whole-function observations.

Compare each measured shape with its previous frontier using both elapsed time and useful throughput. Coarse host
observations must state their uncertainty. A capacity bound or a small model residual does not establish that the
observed execution is optimal. The numerical and observation interfaces are in [whole-function.md](whole-function.md).
