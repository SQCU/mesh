# Group-local result retention status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`.
Assignment: `executor/group-locals-assignment.md`, with the review corrections "avoid eager new solves" and
"Resolve real ownership invalidation, preserve exact prior semantics".
File: `/Users/mdot/metal-microbench/metal_transform.mm` only, working tree; 87 insertions and 42 deletions on
metal-microbench ee02e6c4, which did not touch this file. Nothing is staged or committed. The other modified
metal-microbench files belong to other workers.

**READY for review.** The ownership case found earlier is resolved inside this producer→consumer connection by
separating the proof from admission (section 2).

## 1. Producer, representation, consumer (final code)

- **Proof:** `Proven(p, group, mapping)` is the old `Locals` body with only its admission removed. The removed
  parts are:
  - the claim test `V.internal != UINT32_MAX`;
  - the duplicate-root test against `made`;
  - the capacity test with `memory`.

  What remains is the full exact proof: output, external visibility, scatter, the conflict/inside/read scan,
  `Partitioned` digits and range, and gathered coverage. It returns the ordered proven candidates
  `Candidate { uint32_t value; Local local; }`, one per output that passes, in member and operand order. `value` is
  that output, whose claim admission reads. Repeated candidates for the same root are kept.
- **Admission:** `Admitted(p, candidates, mapping)` is the old selection, unchanged in order and arithmetic.
  `memory` starts at Σ member `mapping.memory`. Then, for each candidate in order:
  1. skip if `p.values[value].internal != UINT32_MAX`;
  2. skip if an admitted local already has this root;
  3. skip if `memory + LocalBytes > p.memory`;
  4. otherwise admit it.

  So a same-root candidate is still admitted when an earlier one failed capacity, as the old `made` loop did.
- **Representation:** `struct Group { std::vector<uint32_t> members; std::optional<std::vector<Candidate>>
  candidates; }` is the grouping owner's record in `Transform`. `candidates` is either not yet proven, or the
  proven candidates of exactly `members` under their `mapping[member]`.
  - `Proved(p, group, mapping)` proves only if not yet proven, then reuses the result.
  - Singleton groups hold the empty vector, as in section 3.
- **Cost consumer:** `Profitable(p, group, mapping, op, own, m, joined)`:
  - **fast path:** unchanged, with no proof; the grown group's candidates are then not yet proven.
  - **costed path:** proves `joined = Proven(members + op, together)` once. It computes
    `combined = Time(…, Admitted(*joined, together))` and
    `apart = Time(group, prior, Admitted(Proved(group), prior)) + Time({op}, {own}, {})`.

  Admission there uses the current claims (none are written during grouping), exactly as the old `Time → Locals`.
- **Retention:** on acceptance, `group->candidates = std::move(solved)`: proven, or not yet proven after a fast
  acceptance. A rejected candidate is discarded by `joined.reset()`. A prior group's proof, once made, is reused by
  the `followed` candidate and later candidates.
- **Kernel consumer:** `Internal(p, kernel, Admitted(p, Proved(p, grouped, mapping), member mappings))`.
  - Admission at installation uses the preceding kernels' real claims, as the old `Internal → Locals` did.
  - `Internal` then sets the one owner and appends `Kernel.locals`. Its call condition is unchanged.
  - The conflict and geometry proof is never re-run here.
- **Unchanged:** CPU/GPU eligibility, `Joins` limits, profitability arithmetic, the fast path's work, operation and
  group order, `Mapping` ownership, `Time`'s arithmetic (it takes the admitted vector), and `ComposeCPU`'s
  `a.locals.insert(...)` merge.

## 2. Equivalence with the old code

- **Claim:** under the same claims C, `Admitted(Proven(members, mappings), mappings)` admits exactly the `Local`s
  that old `Locals(members, mappings)` produced.
- **Why:** the old loop tested claim, then duplicate, then the proof, then capacity, per output in the same order.
  The proof reads neither claims, `made` nor `memory`. So moving the claim and duplicate tests after the proof
  changes no admitted element and no element order.
- **The earlier ownership case:** two groups' locals for one storage through disjoint views now behaves exactly as
  before. Installation admission sees the earlier kernel's claim on the output value and skips that candidate.
  Freed capacity can admit a later candidate. The cost path sees no claims, as the old cost did.
- **Proof work:** old work proved only outputs not already claimed and not duplicates. The new proof also proves
  outputs whose root is later suppressed as a duplicate at admission. During grouping no output is claimed.

**Proof and solve counts:**

| | Old (`Locals` calls) | New |
|---|---|---|
| Per costed `Profitable` | 3 | 1 proof (`joined`), plus the prior group's proof only the first time it is consulted for its current membership, plus 2 admissions |
| Fast path | 0 | 0 |
| Per final group of more than one member | 1 | at most 1 proof (only if never proven while costing), plus 1 admission |

## 3. Singleton candidates

Singleton candidates are empty with no scan. This holds for the `{op}` cost term and for every new or
non-Generic/LAPACK singleton group. The source proof:
- For member `i`, an output becomes a candidate only if `read` is true at the end of the scan over operations.
- `read = true` is reached only for a conflicting operation `o != group[i]`. That sets
  `inside = find(group.begin() + i + 1, group.end(), o) != group.end() && ...`.
- A one-member group has no later member, so `inside` is false whenever `read` becomes true.
- So no candidate exists, and admission of none is empty.

## 4. Evidence

- **Baseline:** private strict compile of the unmodified file, with the Makefile rule's flags (`TINT_CFLAGS`,
  `-x objective-c++ -fobjc-arc -O2 -Wall -Wextra -Werror -I . -iquote .build -I ../dox/mesh/rdma`, `-c` to
  `/tmp/archive-worker/locals/baseline.o`): exit 0.
- **Final code:** the same command to `/tmp/archive-worker/locals/retained.o`: exit 0, no warnings
  (`/tmp/archive-worker/locals/compile.log` is empty).
- **Not run:** no shared build and no client. Root coordinates the existing clients after the paired checkpoint.
  No kernel-output comparison has been made.

## Supervisor checkpoint

Codex reviewed the exact proof/admission split and downstream ownership operand. Metal source committed `d6084ff4`; strict private compile passed. The retained candidate proof is reused only while group members/mappings/operation effects are unchanged. No shared build/client evidence yet; that follows the coordinated graph/extent ABI checkpoint.
