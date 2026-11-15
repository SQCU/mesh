# Change request: allocate value copies and execute received work

## A3. The allocator

The allocator consumes the program's value lifetimes and happens-before relation. Dilworth/Fulkerson chain
decomposition assigns storage to ordered lifetimes; Lam's modulo variable expansion supplies repeated copies
(Lam 1988 §2.3; Rau 1994). The result is the page table of Kilburn et al. 1962 and Denning 1970, consumed by
numerical bindings and the bridge. Capture item 6 supplies contiguous offsets; collective-values item 14 supplies
cyclic-interval coloring. These shared algorithms also place function parameters, operands and retained compiler values
(operator's 14 November retained-result instruction; AOT A5/A8).

## C3. Value allocation, availability and completion

### C3.1 Received work uses predication and compaction

The bridge records received-page phases and arrival counts. The target compares expected phases, applies
predication by selection and reduction identities, and compacts remaining work through scan/scatter
(Allen, Kennedy, Porterfield and Warren, POPL 1983; Blelloch 1990; Harris, Sengupta and Owens 2007).
The worklist and its length are explicit operands of Metal's indirect dispatch. Cancelled arrivals advance
error/completion counts; successful arrivals publish the received-page phase (libfabric `fi_cntr_readerr`; C3.7). Capture item 3 specifies the
index analysis and the consumers of those values.

### C3.2 Allocation places every value

`mesh_plan` consumes the values and A3's lifetime/order inputs and returns bindings for weights, state, numerical
intermediates, communicated operands, publication records, phases and counts. Loop-invariant values receive their
shared placement and per-execution values receive expanded copies (Lam 1988). Target preparation binds the returned
views, and the declared producers initialize their contents.

### C3.3 Modulo variable expansion produces copies

For each value the compiler derives the lag `L` at which reads of execution `t` precede the next write at `t+L`.
The common unroll factor `u` is the largest of those minimum lags; the graph contains `u` occurrences and execution `t` selects `t mod u`
(Lam 1988; Rau 1994). The allocator places those occurrences using their actual lifetimes. Capture item 2 supplies
the self-timed buffer bound for feedforward operands from execution-time estimates.

### C3.4 Published schedules retain their storage

The compiler allocates each published schedule's state and value sections from its solved lifetimes. The published
artifact keeps the references consumed by its executing functions and transport operations. Deferred reclamation
releases retired storage after its outstanding references finish (McKenney and Slingwine, RCU, 1998; Fraser 2004).

### C3.5 The producer supplies dependencies

The frontend supplies each numerical stage's read/write accesses and source identities. The graph module derives
the dependence relation, and `mesh_plan` consumes that result for A3 and scheduling (Bernstein 1966; AOT A10;
composition item 6). Numerical lowering, transport and the signature share the retained relation and its mappings.

### C3.6 Completion reports the invocation outcome

The emitted program carries received phases, completion counts and the execution status through the producer/consumer
relations in C3.1/C3.7. The client observes the final event or commit feedback with the specified bounded completion
wait (capture item 4; Reed and Kanodia 1979; Apple `MTLEvent.h` and `MTL4CommitFeedback`).

### C3.7 Phase and publication records advance with execution

Preparation initializes one two-bit received-page phase, one count per received range, and one 64-bit publication
word per sent range. Receive completion atomically toggles the phase bit in the Gray sequence `00,01,11,10`;
the reader compares with `Gray((floor(t/u)+1) mod 4)`. The producer publishes `t+1` after its writes and the bridge
consumes that threshold. Counts and queue events advance monotonically (PTX `mbarrier` phase mechanism;
Boehm and Adve 2008; capture item 5's 64-bit release operation). These records occupy their allocated state sections.

## D.3. Prepared values and target submission

### D.3.2 Delegate values use the common allocation

The typed delegate interface returns buffers, accesses and numerical bodies to the allocator. Preparation binds
weights, constants and expanded intermediates to its returned pages, following Lam's invariants and variable
expansion and the composition request's LiteRT/Tint interface.

### D.3.3 CPU-produced inputs use compiled copies

Positions, masks, cache parameters and other CPU-produced values carry their declared producer/consumer relation.
Lam's expansion and the BSL buffer-bound calculation return their copies and invocation bindings. The compiled
CPU producer writes the selected copy and the consuming target reads it (capture item 2).

### D.3.4 Native emission produces Metal 4 submissions

The target compiler consumes scheduled commands and required synchronization and emits `MTL4CommandBuffer`
submissions using `MTL4CommandAllocator`, `MTL4CommandQueue`, event operations and per-commit feedback
(Apple SDK headers; source-lowering item 3; integration item 7). The emitted operations realize the retained effects
and availability relationships of C3.1 and capture item 4.

### D.3.5 Receive posting follows storage lifetime

The bridge reads the compiled receive lists and reposts a backing page after its preceding receive and declared
uses complete. The allocator's dependency order relates the next producer to the prior readers
(MPI 4.1 §3.7/§3.9; Lam 1988; A3).

### D.3.6 Topology changes produce a new realization

The bridge disseminates the link-state map and its version (RFC 2328 §§13/16). Persistent requests carry the
membership-view identity, and view changes complete their outstanding communication with the recorded error
outcome. The next compilation consumes the updated topology (collective-values items 11–13; RCU removal and
libfabric completion counters).
