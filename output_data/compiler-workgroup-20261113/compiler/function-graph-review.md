# Retained function analysis — supervisor source review

Scope: actual `Program`/`Kernel` Tint function owners → canonical mesh graph search → Work, Arithmetic and primitive Counts. Same Claude session `979477c2-7dfd-4d9d-a56e-5f086b3c7471`, explicit/reported `claude-opus-5-5`, resumes 6–9. No extra worker tree.

Reviewed source keeps per-function instruction facts and direct callee multiplicity in the actual module-owned domain, with retained function/index/CSR correspondence. The canonical `mesh_graph_search` owns graph traversal; upstream Tint traversal owns local structured instructions. `Changed` invalidates rows and summaries of the actual rewritten operation functions, retaining unrelated facts. Shared callees are cloned/spliced and remain immutable, as traced by the root integration review. Lowered specialization shares the immutable analysis through the existing source owner.

The source Program's existing `workCache` remains its summary owner. Foreign Kernel modules retain their own completed Work totals; actual Kernel.module ownership selects that analysis. Binary64 library facts/totals are acquired only for reached helper calls. Arithmetic retains completed per-function totals after canonical search. Repeated requests consume those completed summaries. Counts discharges its search into the existing immutable counted result.

Corrections supervised: rejected per-request whole module/CSR rebuild; rejected unconditional binary64 acquisition; required actual activation module operand; removed remaining independent Arithmetic recursion; required retained Arithmetic/foreign/library totals rather than repeated graph evaluation.

Evidence: worker private canonical Makefile syntax checks pass for all eight IR translation units with `-Wall -Wextra -Werror`; supervisor inspected actual source owner/caller/Changed/specialize connections and `git diff --check`. This is source/build evidence, not machine-checked preservation proof or throughput evidence. Matching coordinated build and existing numerical client remain the next execution observation. Other source-relation/fusion and archive work is separately open.
