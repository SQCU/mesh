# SAME e48: bounded sort scalar size producer

Compiler accepted exact live evidence executor/ane-op9-review.md (c6c14157): QDWH Step calls1/2/3,
_Sort interval[9,13), runtime extentI32 root34, axis length2, actual op9 is bounded prep. Current constructor
metal_ir.mm3855/3910ff computes padded extent by U32 e−1 then SHR/OR. No generic MIL support is requested.

Implement only metal_ir.mm metal_ir_sort bounded scalar size expression. Keep exact existing source API,
static size/network, original extent operand, stage counts, init/compare-exchange/copy, NaN/stability and
placement. New expression computes least power2 >= e with e0→0 on existing valid contract 0<=e<=length.
Use typed integer compares/selects against existing finite static size thresholds; no float log/pow,
no clamp/target branch/variant/bit emulation/backend. A direct expression is P=select(e>0,1,0), then for each
known q=1,2,...<size set P=select(e>q,2q,P). Retain existing source IDs and result ownership. Prove final
P=0 for e0 and P=2^ceil(log2(e)) for positive valid e in documentation, not runtime evaluator. Actual public
constructor U32 arithmetic/domain remains its existing semantics; don't broaden unrelated unsigned MIL work.
Replace displaced bit-smearing prose comment with exact doc citation or formal equivalence; no broad cleanup.

Only source unit + bounded status, actual Makefile TINT_CFLAGS strict Objective-C++ check, HOLD. No shared
build/import/target/numerical run/agents/commit. Original malformed paths unchanged. Supervisor will review,
checkpoint then compiler co-review/build/retry the same requestedANE client.
