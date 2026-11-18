#include <ompl/base/MotionValidator.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/samplers/InformedStateSampler.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ob = ompl::base;
namespace og = ompl::geometric;

namespace {

void check(const bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::string readFile(const std::string &path)
{
    std::ifstream input(path);
    check(static_cast<bool>(input), "cannot open audited OMPL source: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void auditPinnedSource(const std::string &root)
{
    const std::string implicit = readFile(root + "/src/ompl/geometric/planners/informedtrees/aitstar/src/ImplicitGraph.cpp");
    const std::string ait = readFile(root + "/src/ompl/geometric/planners/informedtrees/src/AITstar.cpp");
    check(implicit.find("costToGo, objective_->motionCostHeuristic(vertex->getState(), goal->getState())") !=
              std::string::npos,
          "AIT* cost-to-go accumulator is not the running minimum");
    check(implicit.find("const bool foundInformedSample =") != std::string::npos &&
              implicit.find("if (foundInformedSample)") != std::string::npos,
          "AIT* does not honor the informed sampler return value");
    check(ait.find("objective_->combineCosts(parent->getCostToComeFromStart(), edgeCostHeuristic),\n"
                   "                parent->getCostToComeFromStart()};") != std::string::npos,
          "AIT* forward key is not (g+c+h, g+c, g)");
    check(ait.find("objective_->combineCosts(edgeCostHeuristic, child->getCostToGoToGoal()),\n"
                   "                parent->getCostToComeFromStart()};") == std::string::npos,
          "old AIT* (g+c+h, c+h, g) key is still present");
}

void setXY(ob::State *state, const double x, const double y)
{
    auto *value = state->as<ob::RealVectorStateSpace::StateType>();
    value->values[0] = x;
    value->values[1] = y;
}

class FailingSampler final : public ob::InformedSampler
{
public:
    FailingSampler(const ob::ProblemDefinitionPtr &problem, unsigned int attempts, unsigned int &calls)
      : ob::InformedSampler(problem, attempts), calls_(calls)
    {
    }

    bool sampleUniform(ob::State *state, const ob::Cost &) override
    {
        ++calls_;
        setXY(state, 0.25, 0.75);
        return false;
    }

    bool sampleUniform(ob::State *state, const ob::Cost &, const ob::Cost &) override
    {
        ++calls_;
        setXY(state, 0.25, 0.75);
        return false;
    }

    bool hasInformedMeasure() const override
    {
        return false;
    }

    double getInformedMeasure(const ob::Cost &) const override
    {
        return space_->getMeasure();
    }

    double getInformedMeasure(const ob::Cost &, const ob::Cost &) const override
    {
        return space_->getMeasure();
    }

private:
    unsigned int &calls_;
};

class FailingSamplerObjective final : public ob::PathLengthOptimizationObjective
{
public:
    FailingSamplerObjective(const ob::SpaceInformationPtr &si, unsigned int &calls)
      : ob::PathLengthOptimizationObjective(si), calls_(calls)
    {
    }

    ob::InformedSamplerPtr allocInformedStateSampler(const ob::ProblemDefinitionPtr &problem,
                                                      unsigned int attempts) const override
    {
        return std::make_shared<FailingSampler>(problem, attempts, calls_);
    }

private:
    unsigned int &calls_;
};

class AlwaysInvalidMotion final : public ob::MotionValidator
{
public:
    explicit AlwaysInvalidMotion(const ob::SpaceInformationPtr &si) : ob::MotionValidator(si)
    {
    }

    bool checkMotion(const ob::State *, const ob::State *) const override
    {
        ++invalid_;
        return false;
    }

    bool checkMotion(const ob::State *first, const ob::State *, std::pair<ob::State *, double> &last) const override
    {
        ++invalid_;
        if (last.first != nullptr)
            si_->copyState(last.first, first);
        last.second = 0.0;
        return false;
    }
};

void testSamplerFailureDoesNotValidateOrInsertStaleState()
{
    auto space = std::make_shared<ob::RealVectorStateSpace>(2u);
    ob::RealVectorBounds bounds(2u);
    bounds.setLow(0.0);
    bounds.setHigh(1.0);
    space->setBounds(bounds);
    auto si = std::make_shared<ob::SpaceInformation>(space);
    unsigned int staleStateValidityCalls = 0u;
    si->setStateValidityChecker([&](const ob::State *state) {
        const auto *value = state->as<ob::RealVectorStateSpace::StateType>();
        if (value->values[0] == 0.25 && value->values[1] == 0.75)
            ++staleStateValidityCalls;
        return true;
    });
    si->setMotionValidator(std::make_shared<AlwaysInvalidMotion>(si));
    si->setup();

    unsigned int samplerCalls = 0u;
    auto problem = std::make_shared<ob::ProblemDefinition>(si);
    ob::ScopedState<> start(space), goal(space);
    setXY(start.get(), 0.0, 0.0);
    setXY(goal.get(), 1.0, 1.0);
    problem->setStartAndGoalStates(start, goal, 0.0);
    problem->setOptimizationObjective(std::make_shared<FailingSamplerObjective>(si, samplerCalls));

    og::AITstar planner(si);
    planner.setBatchSize(1u);
    planner.trackApproximateSolutions(false);
    planner.setProblemDefinition(problem);
    planner.setup();
    const auto status = planner.solve(ob::PlannerTerminationCondition([&] { return samplerCalls >= 3u; }));
    check(status == ob::PlannerStatus::TIMEOUT, "failing sampler fixture unexpectedly solved");
    check(samplerCalls >= 3u, "AIT* never exercised the failing informed sampler");
    check(staleStateValidityCalls == 0u,
          "AIT* validated a state buffer after the informed sampler returned false");
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        check(argc == 2, "usage: aitstar_patch_tests OMPL_SOURCE_ROOT");
        auditPinnedSource(argv[1]);
        testSamplerFailureDoesNotValidateOrInsertStaleState();
        std::cout << "PASS AIT* sampler contract, informed-set accumulator, and (g+c+h,g+c,g) key\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
