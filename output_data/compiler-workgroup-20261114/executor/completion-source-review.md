# Writer completion ownership

Each released Value now retains one actual completion scope: its counter, writer dispatches, actual completion
dispatches and multiplicities. Artifact produces it once; TransportMask reads the retained total. CPU/ANE writers
increment once, GPU writers contribute their actual groups, and the final completion resets the counter and
publishes the Value's release span. Host publication uses the same prepared status words as GPU publication.

SENT and completion bindings retain storage lifetime without fabricated whole-root memory writes. Real bridge
sent-read positions and the subject relation remain. Review found Storages still skipped access0 despite a prior
status claim; the actual skip is now removed, preserving alignment and last use while read/write flags retain
their original meaning. Zeroed counters are execution metadata; the existing planner and preparation own copies.

The archive moves from the already-built v35 to v36 for retained completion vectors. The joint checkpoint also
includes SAME41's unchanged six-vector Work persistence, which had already participated in the matched v35 build.
It does not claim completion of the Work cost model.

SAMEe48 private strict compilation passed the actual native units and callers; the final Storages correction passed
the affected translation unit. Original malformed source remains unchanged and fails parsing. No shared rebuild
or runtime observation is claimed for this snapshot.

The signature join is explicitly unfinished. A potential winning site is not an unconditional publisher.
Canonical graph integration must consume the actual completion scope, including first/later GPU alternatives,
through emission, composition and archive. Current ordinary atomic counts are one increment per host completion,
independent of the publication span; they do not supply that missing enforcement relation.
