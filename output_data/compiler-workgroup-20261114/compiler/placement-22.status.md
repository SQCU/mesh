# placement-22 status: joint-root record-to-arena projection (SAME979; source reviewed; HOLD)

**Edit:** one line, `rdma/mesh-collective.c:801` in `plan_rows`. No header, mapper, schema, allocator or Python edits;
no build, target, import or staging.

```c
-      page[v.first+(size_t)c*v.stride+k]=plan_page(x,block,program->root_item[r]+c,k);
+      page[v.first+(size_t)c*v.stride+k]=plan_page(x,block,x->run,x->at[program->root_item[r]+c]+k);
```

## Producers and consumers confirmed in source

| Site | What it does |
|---|---|
| `:530` `plan_items` | Puts the whole run arena in one PLAN_ALONE item `x->run` of `x->blocks` blocks. |
| `:695` `plan_block` | `program->root_item[k] = r`: root k's first record, an index into `x->at`. `x->at[q]` is record q's greedy byte offset divided by `quantum` (`plan_quantum(m)`, one block). |
| `:782` `plan_frame` | Function run copies: `plan_page(x,block,x->run,x->at[p->item[j]+c]+k)`. |
| `:801` (fixed) | Treated `root_item[r]+c` as a plan_item index into `x->item`, the wrong domain. Root copy c's row k now uses the same run item at block `x->at[root_item[r]+c]+k`. |
| `:2411` `mesh_plan_local` | The local planner, already correct: `x.at[root_item[r]+t]*quantum`. |

**Unchanged:** root copies, count and stride; sections; transport; Local; the archive; interference; reserved
capacities.

## Checks

- **Strict:** `cc -std=c11 -fblocks -O2 -Wall -Wextra -Werror -fsyntax-only` with the Makefile flags
  (`/tmp/function-graph-check/rdma-flags.mk`) passes on `rdma/mesh-collective.c`, the one affected unit (no header
  changed).
- **sha256:** `rdma/mesh-collective.c` `d16d38fdffe5bb11ea48645063d26e65c39e9b6eb8a2acaaec2210ff0a1ec655`.

**Review:** root and compiler independently confirmed the record/item distinction. Supervisor repeated the actual-flags strict check on the final source: exit 0, empty placement-22-supervisor-syntax.log. The one-line source diff passes whitespace checking.

**Hold:** source complete; canonical executor build and unchanged distributed caller remain pending. No runtime allocation result is claimed.
