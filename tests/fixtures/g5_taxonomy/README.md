# G5 analyzer correctness fixtures

`cases.json` contains explicitly synthetic parser cases, with flat geometry and
analytically specified XY paths/costs. Tests materialize copies in a unique build
directory and run the unchanged independent `Oracle` and `validate_trial` before
calling the production `g5_analysis.read_g5_trial` reader. These are not planner
performance observations.

`retained_approximate` contains byte-identical raw files from the real RRT* CTest
timeout artifact used by the G5-v1 failure probe. `retained_provenance.json`
records each copied hash. Only a runtime copy's request map path is relocated;
the source fixture and all G5-v1 artifacts remain untouched. This case directly
exercises the production branch that previously demanded an exact singleton
endpoint and an exact incumbent snapshot for a valid approximate path.

`test_g5_taxonomy.py --preflight-output <new-directory>` materializes four small
cases outside the primary dataset and writes a read-only analyzer preflight
summary. It runs no planner and performs no statistical inference.
