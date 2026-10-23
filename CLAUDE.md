See [AGENTS.md](AGENTS.md) — the accessibility contract that binds all changes to this repo, and [THREAT-MODEL.md](THREAT-MODEL.md) for why.

Short version, in the operator's words (THREAT-MODEL.md): "accessibility is required; inaccessibility is *contractually forbidden*", and "rejection filtering and defensive programming are broadly forbidden by our security model". So code acts. It does not decline to act, hold back a capability by node class, or fail toward a less reachable state.
