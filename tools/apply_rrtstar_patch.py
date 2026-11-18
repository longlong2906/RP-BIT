"""Apply the saved one-line fix to the pinned workspace source, never system OMPL."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / ".deps/ompl-1.7.0/src/ompl/geometric/planners/rrt/src/RRTstar.cpp"
patch = root / "patches/ompl-1.7.0-rrtstar-admissible-terrain-heuristic.patch"
before = "                costToCome, opt_->motionCost(startMotion->state,"
after = "                costToCome, opt_->motionCostHeuristic(startMotion->state,"
text = source.read_text(encoding="utf-8")
patch_text = patch.read_text(encoding="utf-8")
assert "-" + before in patch_text and "+" + after in patch_text
assert text.count(before) + text.count(after) == 1, "unexpected OMPL patch base"
if before in text:
    source.write_text(text.replace(before, after), encoding="utf-8", newline="\n")
print("base_commit=eabb4d0f825666e4553a2ec0ecb9c7a5a33637dd")
for path in (patch, source):
    print(path.relative_to(root), hashlib.sha256(path.read_bytes()).hexdigest())
