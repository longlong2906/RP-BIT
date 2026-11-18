#include "terrain25d/HauserLazyPRMstar.hpp"

#include <ompl/base/MotionValidator.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/StateSampler.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ob = ompl::base;
using terrain25d::HauserLazyPRMstar;

namespace {

using XY = std::array<double, 2>;

void check(const bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void near(const double actual, const double expected, const std::string &message)
{
    const double tolerance = 1e-10 + 1e-10 * std::max(std::abs(actual), std::abs(expected));
    check(std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tolerance, message);
}

XY xy(const ob::State *state)
{
    const auto *value = state->as<ob::RealVectorStateSpace::StateType>();
    return {value->values[0], value->values[1]};
}

void setXY(ob::State *state, const XY point)
{
    auto *value = state->as<ob::RealVectorStateSpace::StateType>();
    value->values[0] = point[0];
    value->values[1] = point[1];
}

bool samePoint(const XY lhs, const XY rhs)
{
    return std::abs(lhs[0] - rhs[0]) < 1e-12 && std::abs(lhs[1] - rhs[1]) < 1e-12;
}

class SequenceSampler final : public ob::StateSampler {
public:
    SequenceSampler(const ob::StateSpace *space, std::vector<XY> samples)
      : ob::StateSampler(space), samples_(std::move(samples))
    {
        if (samples_.empty())
            throw std::invalid_argument("SequenceSampler needs at least one sample");
    }

    void sampleUniform(ob::State *state) override
    {
        setXY(state, samples_[std::min(next_, samples_.size() - 1u)]);
        ++next_;
    }

    void sampleUniformNear(ob::State *state, const ob::State *, double) override
    {
        sampleUniform(state);
    }

    void sampleGaussian(ob::State *state, const ob::State *, double) override
    {
        sampleUniform(state);
    }

private:
    std::vector<XY> samples_;
    std::size_t next_{0};
};

class PredicateMotionValidator final : public ob::MotionValidator {
public:
    using Predicate = std::function<bool(XY, XY)>;

    PredicateMotionValidator(const ob::SpaceInformationPtr &si, Predicate predicate)
      : ob::MotionValidator(si), predicate_(std::move(predicate))
    {
    }

    bool checkMotion(const ob::State *first, const ob::State *second) const override
    {
        const bool valid = predicate_(xy(first), xy(second));
        valid ? ++valid_ : ++invalid_;
        return valid;
    }

    bool checkMotion(const ob::State *first, const ob::State *second,
                     std::pair<ob::State *, double> &lastValid) const override
    {
        if (checkMotion(first, second))
            return true;
        if (lastValid.first != nullptr)
            si_->copyState(lastValid.first, first);
        lastValid.second = 0.0;
        return false;
    }

private:
    Predicate predicate_;
};

class ScaledLengthObjective final : public ob::OptimizationObjective {
public:
    using UndefinedPredicate = std::function<bool(XY, XY)>;

    ScaledLengthObjective(const ob::SpaceInformationPtr &si, const double scale,
                          UndefinedPredicate undefined = {})
      : ob::OptimizationObjective(si), scale_(scale), undefined_(std::move(undefined))
    {
    }

    ob::Cost stateCost(const ob::State *) const override
    {
        return identityCost();
    }

    ob::Cost motionCost(const ob::State *first, const ob::State *second) const override
    {
        if (undefined_ && undefined_(xy(first), xy(second)))
            return infiniteCost();
        return ob::Cost(scale_ * si_->distance(first, second));
    }

private:
    double scale_;
    UndefinedPredicate undefined_;
};

struct Fixture {
    ob::StateSpacePtr space;
    ob::SpaceInformationPtr si;
    ob::ProblemDefinitionPtr problem;
};

Fixture makeFixture(const std::vector<XY> &samples,
                    const ob::StateValidityCheckerFn &stateValidity,
                    const PredicateMotionValidator::Predicate &motionValidity,
                    const ob::OptimizationObjectivePtr &objective = nullptr)
{
    auto space = std::make_shared<ob::RealVectorStateSpace>(2u);
    ob::RealVectorBounds bounds(2u);
    bounds.setLow(-1.0);
    bounds.setHigh(5.0);
    space->setBounds(bounds);
    space->setStateSamplerAllocator([samples](const ob::StateSpace *stateSpace) {
        return std::make_shared<SequenceSampler>(stateSpace, samples);
    });

    auto si = std::make_shared<ob::SpaceInformation>(space);
    si->setStateValidityChecker(stateValidity);
    si->setMotionValidator(std::make_shared<PredicateMotionValidator>(si, motionValidity));
    si->setup();

    auto problem = std::make_shared<ob::ProblemDefinition>(si);
    ob::ScopedState<> start(space);
    ob::ScopedState<> goal(space);
    setXY(start.get(), {0.0, 0.0});
    setXY(goal.get(), {4.0, 0.0});
    problem->setStartAndGoalStates(start, goal, 0.0);
    problem->setOptimizationObjective(
        objective != nullptr ? objective : std::make_shared<ob::PathLengthOptimizationObjective>(si));
    return Fixture{space, si, problem};
}

void testConnectionFormula()
{
    check(HauserLazyPRMstar::connectionK(0u) == 0u, "k(0)");
    check(HauserLazyPRMstar::connectionK(1u) == 0u, "k(1)");
    check(HauserLazyPRMstar::connectionK(2u) == 1u, "k(2) clamp/self exclusion");
    check(HauserLazyPRMstar::connectionK(3u) == 2u, "k(3) clamp/self exclusion");
    check(HauserLazyPRMstar::connectionK(10u) == 9u, "k(10) clamp");
    check(HauserLazyPRMstar::connectionK(100u) == 20u, "k(100) floor");

    for (std::size_t n = 2u; n < 10000u; ++n)
    {
        const auto raw = static_cast<std::size_t>(
            std::floor(1.1 * std::exp(1.0) * 1.5 * std::log(static_cast<double>(n))));
        const std::size_t expected = std::min(n - 1u, raw);
        check(HauserLazyPRMstar::connectionK(n) == expected, "general k(n) formula");
    }
}

void testOneProposalPerIterationAndStateRejection()
{
    const auto isEndpoint = [](const ob::State *state) {
        const XY point = xy(state);
        return samePoint(point, {0.0, 0.0}) || samePoint(point, {4.0, 0.0});
    };
    Fixture fixture = makeFixture({{2.0, 2.0}}, isEndpoint, [](XY, XY) { return false; });
    HauserLazyPRMstar planner(fixture.si);
    planner.setProblemDefinition(fixture.problem);
    planner.setup();

    const ob::PlannerStatus status = planner.solve(
        ob::PlannerTerminationCondition([&planner] { return planner.numIterations() >= 7u; }));
    check(status == ob::PlannerStatus::TIMEOUT, "rejection-only run must time out, not claim infeasible");
    check(planner.numIterations() == 7u, "iteration count");
    check(planner.numSampleProposals() == 7u, "one proposal per iteration");
    check(planner.numStateRejections() == 7u, "each invalid proposal must be rejected once");
    check(planner.milestoneCount() == 2u, "rejected samples entered roadmap");
    check(planner.numMotionChecks() == 1u, "direct edge should be checked once");
    check(planner.numInvalidEdges() == 1u, "direct invalid edge should be removed once");
    check(planner.edgeCount() == 0u, "invalid edge remained active");

    const auto &progress = planner.getPlannerProgressProperties();
    check(progress.at("sample proposals INTEGER")() == "7", "proposal progress property");
    check(progress.at("state rejections INTEGER")() == "7", "rejection progress property");
}

void testOnlyFirstInvalidEdgePerCandidate()
{
    const XY start{0.0, 0.0};
    const XY goal{4.0, 0.0};
    const XY sample{1.0, 0.0};
    const auto blocked = [=](const XY first, const XY second) {
        const auto isPair = [&](const XY a, const XY b) {
            return (samePoint(first, a) && samePoint(second, b)) ||
                   (samePoint(first, b) && samePoint(second, a));
        };
        return isPair(start, goal) || isPair(start, sample) || isPair(sample, goal);
    };
    Fixture fixture = makeFixture({sample}, [](const ob::State *) { return true; },
                                  [=](const XY first, const XY second) { return !blocked(first, second); });
    HauserLazyPRMstar planner(fixture.si);
    planner.setProblemDefinition(fixture.problem);
    planner.setup();

    const ob::PlannerStatus status = planner.solve(ob::PlannerTerminationCondition(
        [&] { return planner.numInvalidEdges() >= 2u || planner.numIterations() >= 3u; }));
    check(status == ob::PlannerStatus::TIMEOUT, "all-blocked first-invalid fixture must time out");
    check(planner.numIterations() == 1u && planner.numSampleProposals() == 1u,
          "first-invalid fixture proposal count");
    check(planner.numInvalidEdges() == 2u, "must remove direct then first invalid candidate edge");
    check(planner.numMotionChecks() == 2u,
          "second invalid edge of one candidate was checked before recomputation");
    check(planner.edgeCount() == 1u,
          "second invalid edge should remain unchecked/active after first-invalid termination");
}

double dijkstraOracle(const std::vector<std::vector<std::pair<std::size_t, double>>> &adjacency,
                      const std::size_t start, const std::size_t goal)
{
    using Item = std::pair<double, std::size_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    std::vector<double> distance(adjacency.size(), std::numeric_limits<double>::infinity());
    distance[start] = 0.0;
    queue.push({0.0, start});
    while (!queue.empty())
    {
        const auto [cost, vertex] = queue.top();
        queue.pop();
        if (cost > distance[vertex])
            continue;
        for (const auto &[neighbor, weight] : adjacency[vertex])
            if (cost + weight < distance[neighbor])
            {
                distance[neighbor] = cost + weight;
                queue.push({distance[neighbor], neighbor});
            }
    }
    return distance[goal];
}

void testForwardLazyCheckingAndObjectiveCost()
{
    const XY start{0.0, 0.0};
    const XY goal{4.0, 0.0};
    const XY firstSample{1.0, 1.0};
    const XY secondSample{2.0, -1.0};
    const auto blocked = [=](const XY first, const XY second) {
        const auto isPair = [&](const XY a, const XY b) {
            return (samePoint(first, a) && samePoint(second, b)) ||
                   (samePoint(first, b) && samePoint(second, a));
        };
        return isPair(start, goal) || isPair(firstSample, goal);
    };

    Fixture fixture = makeFixture({firstSample, secondSample}, [](const ob::State *) { return true; },
                                  [=](const XY first, const XY second) { return !blocked(first, second); });
    const auto undefined = [=](const XY first, const XY second) {
        return (samePoint(first, firstSample) && samePoint(second, secondSample)) ||
               (samePoint(first, secondSample) && samePoint(second, firstSample));
    };
    auto objective = std::make_shared<ScaledLengthObjective>(fixture.si, 2.0, undefined);
    fixture.problem->setOptimizationObjective(objective);

    HauserLazyPRMstar planner(fixture.si);
    planner.setProblemDefinition(fixture.problem);
    planner.setup();

    unsigned int callbackCount = 0u;
    std::vector<XY> callbackPath;
    double callbackCost = std::numeric_limits<double>::infinity();
    fixture.problem->setIntermediateSolutionCallback(
        [&](const ob::Planner *source, const std::vector<const ob::State *> &states, const ob::Cost cost) {
            check(source == &planner, "callback source planner");
            check(!states.empty(), "callback must contain complete path");
            callbackPath.clear();
            for (const ob::State *state : states)
                callbackPath.push_back(xy(state));
            check(samePoint(callbackPath.front(), start), "callback not oriented start-to-goal");
            check(samePoint(callbackPath.back(), goal), "callback missing actual goal");
            callbackCost = cost.value();
            ++callbackCount;
        });

    const ob::PlannerStatus status = planner.solve(ob::PlannerTerminationCondition(
        [&] { return callbackCount > 0u || planner.numIterations() >= 5u; }));
    check(status == ob::PlannerStatus::EXACT_SOLUTION, "valid lazy roadmap solution not reported");
    check(callbackCount == 1u, "incumbent callback count");
    check(planner.numIterations() == 2u && planner.numSampleProposals() == 2u,
          "deterministic sample sequence/iteration semantics");
    check(planner.numInvalidEdges() == 2u, "first-invalid removal count");
    check(planner.numShortestPathSearches() == 5u, "shortest path not recomputed after each invalid edge");
    check(planner.numMotionChecks() == 5u, "valid-edge cache or forward checking semantics changed");
    check(planner.milestoneCount() == 4u, "accepted milestones");
    check(planner.edgeCount() == 3u, "active edge count after invalid/undefined removals");
    check(planner.numUndefinedCostEdges() == 1u, "nonfinite objective edge was not rejected/countable");

    check(callbackPath.size() == 3u && samePoint(callbackPath[1], secondSample),
          "unexpected final roadmap path");
    std::vector<std::vector<std::pair<std::size_t, double>>> oracleGraph(4u);
    const auto addOracleEdge = [&](const std::size_t first, const std::size_t second, const double weight) {
        oracleGraph[first].push_back({second, weight});
        oracleGraph[second].push_back({first, weight});
    };
    addOracleEdge(0u, 2u, 2.0 * std::sqrt(2.0));  // start--firstSample
    addOracleEdge(0u, 3u, 2.0 * std::sqrt(5.0));  // start--secondSample
    addOracleEdge(3u, 1u, 2.0 * std::sqrt(5.0));  // secondSample--goal
    const double expected = dijkstraOracle(oracleGraph, 0u, 1u);
    near(callbackCost, expected, "callback did not use OptimizationObjective motionCost");
    near(planner.bestCost().value(), expected, "stored incumbent cost");

    planner.clear();
    check(planner.milestoneCount() == 0u && planner.edgeCount() == 0u, "clear roadmap");
    check(planner.numIterations() == 0u && planner.numSampleProposals() == 0u, "clear counters");
    check(std::isinf(planner.bestCost().value()), "clear incumbent");
}

void testIncreasePropagationReconnectsDetachedSubtree()
{
    const XY start{0.0, 0.0};
    const XY goal{4.0, 0.0};
    const XY branch{1.0, 0.0};
    const XY detachedRoot{3.0, 0.0};
    const auto blocked = [=](const XY first, const XY second) {
        const auto isPair = [&](const XY a, const XY b) {
            return (samePoint(first, a) && samePoint(second, b)) ||
                   (samePoint(first, b) && samePoint(second, a));
        };
        // Direct and branch--goal are removed before detachedRoot arrives.
        // branch--detachedRoot then becomes a parent edge whose subtree also
        // contains goal.  Algorithm 5 must invalidate both vertices and seed
        // the start--detachedRoot boundary edge before decrease propagation.
        return isPair(start, goal) || isPair(branch, goal) || isPair(branch, detachedRoot);
    };
    Fixture fixture = makeFixture({branch, detachedRoot}, [](const ob::State *) { return true; },
                                  [=](const XY first, const XY second) { return !blocked(first, second); });
    HauserLazyPRMstar planner(fixture.si);
    planner.setProblemDefinition(fixture.problem);
    planner.setup();

    unsigned int callbackCount = 0u;
    std::vector<XY> callbackPath;
    double callbackCost = std::numeric_limits<double>::infinity();
    fixture.problem->setIntermediateSolutionCallback(
        [&](const ob::Planner *, const std::vector<const ob::State *> &states, const ob::Cost cost) {
            ++callbackCount;
            callbackCost = cost.value();
            callbackPath.clear();
            for (const ob::State *state : states)
                callbackPath.push_back(xy(state));
        });

    const ob::PlannerStatus status = planner.solve(ob::PlannerTerminationCondition(
        [&] { return callbackCount > 0u || planner.numIterations() >= 4u; }));
    check(status == ob::PlannerStatus::EXACT_SOLUTION, "detached subtree was not reconnected");
    check(planner.numIterations() == 2u, "unexpected detached-subtree sample count");
    check(planner.numInvalidEdges() == 3u, "successive invalid parent edges were not removed");
    check(planner.numShortestPathSearches() == 6u,
          "dynamic shortest path was not recomputed after each invalidation");
    check(planner.numMotionChecks() == 6u, "unexpected lazy edge-check sequence");
    check(callbackCount == 1u && callbackPath.size() == 3u,
          "detached-subtree fixture callback/path count");
    check(samePoint(callbackPath.front(), start) && samePoint(callbackPath[1], detachedRoot) &&
              samePoint(callbackPath.back(), goal),
          "increase propagation did not select the boundary reconnection");

    std::vector<std::vector<std::pair<std::size_t, double>>> oracleGraph(4u);
    const auto addOracleEdge = [&](const std::size_t first, const std::size_t second, const double weight) {
        oracleGraph[first].push_back({second, weight});
        oracleGraph[second].push_back({first, weight});
    };
    addOracleEdge(0u, 2u, 1.0);  // start--branch
    addOracleEdge(0u, 3u, 3.0);  // start--detachedRoot boundary reconnection
    addOracleEdge(3u, 1u, 1.0);  // detachedRoot--goal
    const double expected = dijkstraOracle(oracleGraph, 0u, 1u);
    near(callbackCost, expected, "dynamic c/pi result differs from independent Dijkstra");
    near(planner.bestCost().value(), expected, "stored detached-subtree incumbent cost");
}

}  // namespace

int main()
{
    try
    {
        testConnectionFormula();
        testOneProposalPerIterationAndStateRejection();
        testOnlyFirstInvalidEdgePerCandidate();
        testForwardLazyCheckingAndObjectiveCost();
        testIncreasePropagationReconnectsDetachedSubtree();
        std::cout << "PASS HauserLazyPRMstar k(n), proposal/rejection, first-invalid recompute, dynamic subtree reconnect, callback and objective cost\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
