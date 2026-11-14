# Placement branch audit: source HOLD
- Removed declaration-missing fallbacks: driver.compile_captured declares every outlined key; native.compile_program passes that complete owner to declare_program.
- Removed the SEND/RECEIVE landing guard: placement.insert_send_recv assigns a channel per receiver; decompose.finish renumbers the pair; schedule.assemble removes same-machine transfers; lower_metal.construct records landings only for interface RECV. Executor independently reviewed this producer chain.
- Processor metadata is required from placement.insert_send_recv; missing metadata now raises at the actual consumer.
- Removed shared endpoint !staged: staged = landing and not direct, and the remaining storage/root-source predicate is direct.
- Grouping runs once under the existing network condition; port demand/release and solve_program borrow its required results without repeated None alternatives.
- Local/network selection, caller/joint/unbound ownership, optional public operands, and actual ABSENT release remain.
- schedule.py +16/-21; placement.py unchanged. Supervisor AST, pyflakes, and git diff --check PASS. No import, build, or target run.
