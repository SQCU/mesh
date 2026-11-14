# placement-28: reassembled citation (SAME979; placement.py comment only; uncommitted; terminal HOLD)
- **Citation:** `reassembled` (`placement.py:199-204`) now cites torch 2.14.0 `torch/_inductor/fx_passes/post_grad.py:1131-1157` `is_valid_splitwithsizes_cat` and `:1728-1732` `splitwithsizes_cat_replace`. That pattern replaces a cat of every `split_with_sizes` part of one input, in order, on the split's dim, with `input_`.
- **Our equivalent proof**, over reshard's own `cut` parts: the `aten.slice` pieces share the identical source node (`args[0] is`), have the same dim, are unit-step, and are consecutive from 0 to the source's extent (ordered exact full cover). They are that source. A user-authored source `torch.cat` still allocates; only reshard's reconstruction is replaced.
- **Processor (from 27):** the source keeps its own processor (`:704`, placeholder, or reshard product). Existing `insert_send_recv`/replication moves it; no label is fabricated.
- **Erasure (from 27):** `replaced` producers `meshed :648` and `partitioned :805` hold nodes whose values are then shards, never `[node]`, so a returned source is never erased.
- **Unchanged:** no semantic, matcher, guard, helper or schema change.
- **Limit:** a general non-redundant materialized assembly of an alias view remains open and is not claimed closed.
- **Checks:** `ast.parse`, pyflakes and `git diff --check` pass; placement.py +18/−1 against HEAD, two hunks only, no new line over 120. sha256 `60b431d2117bda87cd701a2f35663ea35a1b4a958808dfd491856d8cfdac584d`. No import, build or run.
