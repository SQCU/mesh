# M4 storage cleanup, 2026-09-10

The M4 Data volume had 136 MiB available, with 426 GiB used. After stopping
the evaluation and both mesh bridges normally, cleanup restored 207 GiB
available, with 219 GiB used. No further workload was run during cleanup.

The accumulation included a 41 GiB native compilation cache, 138 GiB of
historical release artifacts, and 63 GiB under the cartlane workload directory.
Repeated game archives and checkpoints were copied between run directories;
observation archives and abandoned temporary writes retained several GiB each.
This was disk exhaustion. Wired memory fell from approximately 2.39 GiB to
1.63 GiB after stopping the processes; the observed zero-RSS zombie did not
account for those allocations.

Cleanup removed the rebuildable native cache and this evaluation's temporary
deployments. Full-content SHA256 comparison identified 113 duplicate large
files totaling 86,417,793,084 bytes. Superseded release checkpoints and inactive
observation artifacts accounted for another 90,105,655,369 bytes. The active
viewer and its run, canonical models and source checkouts, canonical game
assets, a latest release checkpoint, and small measurement records were retained.
The final release and cartlane directories measured 17 GiB and 19 GiB.

The corrected ownership is documented in
the page table (pages-and-functions.md):
canonical Git checkouts and shared installed dependencies replace deployment
snapshots; the game uses its existing assets; transient generated files and
superseded generated continuation artifacts are removed by their application
owner. Observation producers replace one archive, with views into its stored
arrays, and failed temporary archive writes are removed. Policy checkpoint
references address the authoritative continuation bundle rather than copying
optimizer and replay state again.

The numerical repository's `docs/native_artifact_storage.md` specifies native
artifact identity and exact retention of the complete configured batch's artifact
set. Compilation caches no longer accumulate a full copy per run or unrelated
repository revision. Remote numerical outputs and fixtures are removed after
collection. These policies operate outside numerical and transport execution.

The storage changes require subsequent application validation; disk cleanup is
not evidence of numerical acceptance or transport performance.
