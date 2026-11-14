# Bounded sort scalar work — source HOLD

Metal checkpoint9fa86439. SAME e48 session/model retained, supervisor completed citation/strict/checkpoint after
source edit. No shared build, target invocation or ANE claim. ANE-only rounds are held for active priorities2/3/5.

The existing `metal_ir_sort` bounded preparation reads the same extent once. It now initializes
P=select(e>0,1,0), then for each statically retained threshold q=1,2,...<size, emits
P=select(e>q,2q,P). These are construction-time iterations that emit typed comparison/select expressions;
there is no new runtime loop, float log/pow, bound clamp, target branch or integer-bit emulation.

Equivalence on the existing nonnegative active-prefix domain: e=0 makes every condition false, so P=0.
For e=1 only the initial condition is true, so P=1. For 2^(j-1)<e<=2^j, the largest satisfied threshold
is q=2^(j-1); it selects2^j, and later conditions are false. Thus P is the least power of two at least e,
with the same U32 result representation as the original expression. The static network's size already
bounds these thresholds. Actual observed instance: axis length2, runtime I32 extent; runtime value not read.

Downstream work is unchanged: comparison groups P/(2 step) when stage<=P, else0; init runs P, each existing
Batcher compare-exchange uses those groups, final copy uses the original extent. Pair geometry, stable index
ties, NaN ordering, values/indices/source identities and target placement are unchanged.

Only displaced bound-expression prose moved to this document; unrelated source comments untouched.
Supervisor strict Objective-C++ syntax using actual Makefile TINT_CFLAGS and -Wall/-Wextra/-Werror PASS,
empty ane-sort-bound79-syntax.log. Diff check PASS. Original malformed paths unchanged.

This closes the source consumer identified by live op9 evidencec6c14157. It does not establish requested-ANE
prediction, device eligibility, performance, general unsigned arithmetic, Philox or integer bitwise support.
