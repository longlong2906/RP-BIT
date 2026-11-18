"""Independent validation of the actual C++ RRT pruning incumbent fixture."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from independent_verifier import Oracle
from experiment_common import dump, file_hash

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, type=Path)
args = p.parse_args()
oracle = Oracle(args.input / "map.t25d")
with (args.input / "witness_xy.csv").open() as stream:
    path = [[float(row["x_m"]), float(row["y_m"])] for row in csv.DictReader(stream)]
witness = oracle.validate_path(path, [1, 5], [9, 8], 15.)
direct = oracle.validate_path([[1, 5], [9, 5], [9, 8]], [1, 5], [9, 8])
assert witness["valid"] and direct["valid"], (witness, direct)
assert witness["cost_terrain_m"] < direct["cost_terrain_m"]
probe = json.loads((args.input / "rrtstar_pruning_regression.json").read_text())
assert probe["candidate_retained"] and probe["actual_removed"] > 0
assert probe["actual_patched_expression"] < witness["cost_terrain_m"] < probe["old_expression"]
dump(args.input / "independent_validation.json", {"fixture": True, "witness": witness, "direct": direct,
     "map_sha256": file_hash(args.input / "map.t25d"), "path_sha256": file_hash(args.input / "witness_xy.csv")})
print("PASS independent oracle: valid pruning incumbent=15 < direct hill route; actual patch/prune evidence linked")
