"""Apply audited AIT* formula fixes to the pinned workspace OMPL source."""
import hashlib
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
PATCH=ROOT/"patches/ompl-1.7.0-aitstar-paper-formulas.patch"
IMPLICIT=ROOT/".deps/ompl-1.7.0/src/ompl/geometric/planners/informedtrees/aitstar/src/ImplicitGraph.cpp"
AITSTAR=ROOT/".deps/ompl-1.7.0/src/ompl/geometric/planners/informedtrees/src/AITstar.cpp"


def replace_once(path,before,after):
    text=path.read_text(encoding="utf-8")
    count=text.count(before)+text.count(after)
    if count!=1:
        raise RuntimeError(f"unexpected OMPL 1.7.0 patch base in {path}: matches={count}")
    if before in text:
        path.write_text(text.replace(before,after),encoding="utf-8",newline="\n")


def main():
    patch_text=PATCH.read_text(encoding="utf-8")
    replacements=[
        (IMPLICIT,
         """                        costToGo = objective_->betterCost(\n                            costToCome, objective_->motionCostHeuristic(vertex->getState(), goal->getState()));""",
         """                        costToGo = objective_->betterCost(\n                            costToGo, objective_->motionCostHeuristic(vertex->getState(), goal->getState()));"""),
        (IMPLICIT,
         """                        sampler_->sampleUniform(newSamples_.back()->getState(), solutionCost_);\n\n                        // Count how many states we've checked.\n                        ++numSampledStates_;\n\n                        // Check if the sample is valid.\n                        foundValidSample = spaceInformation_->getStateValidityChecker()->isValid(newSamples_.back()->getState());""",
         """                        const bool foundInformedSample =\n                            sampler_->sampleUniform(newSamples_.back()->getState(), solutionCost_);\n\n                        // A false return means the bounded informed sampler did not produce a state.\n                        // Never validate or insert the unchanged state buffer in that case.\n                        if (foundInformedSample)\n                        {\n                            ++numSampledStates_;\n                            foundValidSample = spaceInformation_->getStateValidityChecker()->isValid(newSamples_.back()->getState());\n                        }"""),
        (AITSTAR,
         """                objective_->combineCosts(edgeCostHeuristic, child->getCostToGoToGoal()),\n                parent->getCostToComeFromStart()};""",
         """                objective_->combineCosts(parent->getCostToComeFromStart(), edgeCostHeuristic),\n                parent->getCostToComeFromStart()};"""),
    ]
    for path,before,after in replacements:
        if after.splitlines()[0].strip() not in patch_text:
            raise RuntimeError("saved patch and apply script disagree")
        replace_once(path,before,after)
    print("base_commit=eabb4d0f825666e4553a2ec0ecb9c7a5a33637dd")
    for path in (PATCH,IMPLICIT,AITSTAR):
        print(path.relative_to(ROOT),hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=="__main__":main()
