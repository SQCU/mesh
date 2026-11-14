# Exact owned constructor continuation

Same numerical session 64ddaf6e-3a70-4b6a-9d0b-50691a84c6cd, explicit claude-opus-5-5. Implement promptly; no broad reading, no new agent, staging, commits, shared builds.

Only rdma/mesh_graph.py existing signature constructor hunk. Existing signature(processors,tasks,dependences,operations) constructs C Signature, derives it once, returns forms+record then frees. Executor now needs that actual owned result for mesh_graph_compose; reconstructing from record derives again. Extract exactly the existing construction part as build_signature(processors,tasks,dependences,operations)->owned Signature. The convenience signature function calls build_signature, then the existing try forms/record finally free. Caller of build_signature frees via existing free; preserve error behavior. No other wrapper/algorithm or schema. Existing actual C producer stays the sole owner. CR compilation-composition 3 and6/6.4.

Executor will consume build_signature directly and compose with from_archive(plan.signature); do not edit its files. Verify syntax, existing signature result equivalent for existing record operands, and owned constructor+compose/free using actual built libmesh. Compact status signature-owned-result.status.md. Hold READY.
