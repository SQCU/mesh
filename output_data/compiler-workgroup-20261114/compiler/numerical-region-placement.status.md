# Repeat lifetimes in the existing joint allocation

SAME979 placement17 is source-ready and held. The source is `metal_ir.mm` `Storages`; executor e48 owns the paired repeat descriptor, dispatch order/cuts and native consumer. No new allocator, nested plan or mesh ABI is introduced.

`metal_ir_storage.begin/end` and all three `lastUse` domains are positions in the original `p.order`, not Actor part IDs. `native.storages` passes these positions unchanged. In the joint planner, `plan_before` compares them within one call and uses the retained call closure across calls. A compound Actor therefore need not erase the child storage intervals.

The existing binding pass now also retains each domain's first and last binding window. This covers DATA storage roots/STEP values, PAGES and stored PAGES, including access-0 bindings. After the ordinary storage fix-ups, the repeat projection consumes e48's inclusive `order_first/order_last` spans directly, inner spans first. Each region visits only its own dispatch span and touched binding domains; stamped arrays avoid a full 3V reset for each independent batch member.

A domain stays live across the back edge when it is used outside the region, its earliest in-region access reads or does not store, or its storage is a named condition/carried root. Its last use reaches the region end; a DATA root's allocation interval covers the region. Body-local scratch written before its reads keeps its existing interval and reuse. Nested regions use the same retained spans. No original binding lifetime is shortened.

This relies on the paired order producer keeping each region contiguous and cutting movable windows at its boundaries. A window extending outside a region only widens the lifetime. Existing alignment, loaded/stored classification and archive persistence are unchanged. e48's later shared RepeatSource accessor substitution preserves the same carried-root input.

Placement17's private strict Objective-C++ syntax check passed with the Makefile's expanded TINT flags and `-O2 -Wall -Wextra -Werror -fsyntax-only`; its complete transcript is `placement-17.jsonl`. Supervisor source review confirmed direct spans, the three binding domains and unchanged `plan_before` consumer. No v37 build, numerical invocation or repeat archive roundtrip has yet been observed.
