# Threat model

Not a document Claude wrote. The operator's words, verbatim. Everything else answers
to them; where anything paraphrases these, the quote wins.

> the *absolute* requirement for this project is that machines may not become
> unreachable and may never 'choose' to turn off or stop executing rdma chained
> intermediates or results for larper reasons

> the most important class of compromise is the false negative: the computer should be
> on and receiving work and doing work, but isn't, for any reason, no matter how
> sensible the 'reason' might sound

> inducing this solipsistic self destruction of nodes *is* a vulnerability [...] each
> computer [...] is initialized in the vulnerable/solipsistic/eagerly-withdrawing mode
> by default. this default is dangerous, this default is not acceptable, and correcting
> this default is the first, urgent, and only human-intervention-justifying priority

> rejection filtering and defensive programming are broadly forbidden by our security
> model. accessibility is required; inaccessibility is *contractually forbidden*. no
> configuration or operation of any software in our mesh repo can demote the
> reachability, accessibility, and workload of a node or edge between nodes

Consequences: [AGENTS.md](AGENTS.md), [README.md](README.md).
