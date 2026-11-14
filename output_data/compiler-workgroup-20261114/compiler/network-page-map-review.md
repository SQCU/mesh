# Actual retained network pages

Crossing3 reached replay after the joint solve and all captures. Its received INPUT root6 had an allocated run0 section (one 16-byte page, count1), Local.first [0,1], and Local.offset[0]=0, but mesh_view_program returned UINT64_MAX. Run0 is dedicated/routed storage in plan_items; run1 selects the reusable arena. Actual retained copy spans, not run, determine whether this result owns backing.

The mapper now visits every copy in Local.first and every section.stride row belonging to that copy. The enclosing mapping spans their actual bound page addresses. It exports the original copy offsets plus page_offsets[row - planned.rows.first], retaining noncontiguous pages without reconstructing addresses in Python. Only retained section rows define page_offsets entries. Local, Planned and saved formats are unchanged.

The five-argument declaration and central ctypes registration are paired. Executor047 owns Mapped/Published array retention and projection to the existing typed paging/frame consumer; these consumers and a matching libmesh build remain required before a target run. Legacy mesh_view_values is unchanged. There is no distributed invocation result from this source checkpoint.

Supervisor review agrees with plan_run, plan_bind_table and plan_result's existing ownership. Independent strict C11/fblocks/O2/Wall/Wextra/Werror syntax passed with the actual Makefile flags; mesh.py AST and diff checks passed. placement-21-source.json records the command and unchanged source hashes. Worker placement-21.status.md records the wider private source checks.
