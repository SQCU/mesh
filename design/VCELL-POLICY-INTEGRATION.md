# Spatial integration in the policy

[POLICY-PROGRAM.md](POLICY-PROGRAM.md) defines the current complete flow of raw
event, navigation-node, edge and cell rows through their first projections,
per-observer sparse neighborhoods, learned distance/age integration and later
policy mixing. It is the single model contract for these operators.

The supplied navigation metric is graph length (`measure_dimension = 1`). Its
support fraction measures that supplied length, not walkable floor area. Support
radii and achieved fractions remain geometric measurements: disconnected
components and coarse distance boundaries can make the requested band unattainable.
[CARTPATHS.md](../xonotic/payload/CARTPATHS.md), [NAV-SPEC.md](NAV-SPEC.md) and
[FUSION-SPEC.md](FUSION-SPEC.md) describe the source geometry and its ownership.

Historical derivative and native-run measurements retain their original source
scope. The removed verification suite does not define the current representation
or establish playing strength, persistent buffer allocation or complete runtime
coverage.
