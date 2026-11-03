See [AGENTS.md](AGENTS.md) — the accessibility contract that binds all changes to this repo, and [THREAT-MODEL.md](THREAT-MODEL.md) for why.

Short version: accessibility is required, inaccessibility is contractually forbidden. Rejection filtering and defensive programming are banned. Never write code that declines to act, withholds a capability by node class, or fails toward a less reachable state.
