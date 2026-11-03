# Whole-program correction of the policy/value split

The earlier review found actor-private raw-state projections, an actor-only final
FFN, primary critics reading another row subset, inherited output penalties and
misleading completion claims. The implementation now follows the single complete
[policy program](POLICY-PROGRAM.md), including producer state, episode ownership,
common output projection, optimization, continuation and reporting.

The original [audit evidence](../measurements/policy-head-audit-20260906/README.md)
is historical. Its verification script was deleted with the test suites. In
particular, the early VERA imitation task had explicit user provenance; retiring
it follows the latest common-IR requirement and is not a discovery of forgery.

The [whole-program inventory](../measurements/policy-whole-program-20260906/README.md)
records removed code and input/output omissions addressed by this change. Source
review and compiler results do not supersede user requirements.
