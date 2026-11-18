#include "terrain25d/OmplIntegration.hpp"

#include <ompl/base/PlannerData.h>
#include <ompl/geometric/PathGeometric.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace terrain25d;

namespace {

void check(const bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void near(const double lhs, const double rhs)
{
    check(std::isfinite(lhs) && std::isfinite(rhs) &&
              std::abs(lhs - rhs) <= 1e-8 + 1e-8 * std::max(std::abs(lhs), std::abs(rhs)),
          "numeric mismatch");
}

Terrain flatMap()
{
    return Terrain(21, 21, 0., 0., 1., 1., std::vector<double>(21u * 21u, 0.),
                   std::vector<int>(20u * 20u, 0), {}, .1, 0., 60.);
}

Terrain slopedMap()
{
    std::vector<double> heights(21u * 21u);
    for (std::size_t y = 0; y < 21u; ++y)
        for (std::size_t x = 0; x < 21u; ++x)
            heights[y * 21u + x] = .5 * static_cast<double>(x);
    return Terrain(21, 21, 0., 0., 1., 1., std::move(heights),
                   std::vector<int>(20u * 20u, 0), {}, .1, 0., 60.);
}

struct RunRecord {
    std::vector<Point> solution;
    std::vector<SemRRTstarCommon::NodeAudit> nodes;
    std::uint64_t iterations{};
    std::uint64_t insertions{};
    std::uint64_t rewires{};
    std::uint64_t descendantUpdates{};
    PolicyCounters policy;
};

RunRecord runDeterministic(const unsigned int seed)
{
    auto map = flatMap();
    PolicyCounters counters;
    const Point start{2., 2.}, goal{18., 18.};
    ompl::RNG::setSeed(seed);
    auto problem = makePlannerProblem(map, "sem_rrtstar_common_v1", start, goal, counters, 200, 1.1, 100);
    check(static_cast<bool>(problem.sem), "factory did not create SEM common planner");
    check(!problem.bit && !problem.ait && !problem.lazy && !problem.rrt,
          "SEM common planner routed through another planner family");
    check(problem.planner.get() == problem.sem.get(), "planner pointer mismatch");
    check(problem.sem->getName() == "SEM_RRTstar_common_v1", "planner name mismatch");
    near(SemRRTstarCommon::STEER_RANGE_M, 1.);
    near(SemRRTstarCommon::NEAR_RADIUS_M, 3.);
    near(SemRRTstarCommon::START_CONNECTION_RADIUS_M, 3.);
    near(SemRRTstarCommon::GOAL_BIAS, 0.);
    check(SemRRTstarCommon::INFORMED_ATTEMPTS == 100u, "informed attempt count changed");

    std::vector<double> callbackCosts;
    problem.pdef->setIntermediateSolutionCallback(
        [&](const ob::Planner *, const std::vector<const ob::State *> &states, const ob::Cost cost) {
            check(states.size() >= 2u, "callback path is incomplete");
            const Point first = point(states.front()), last = point(states.back());
            near(first.x, start.x);
            near(first.y, start.y);
            near(last.x, goal.x);
            near(last.y, goal.y);
            if (!callbackCosts.empty())
                check(cost.value() <= callbackCosts.back() + 1e-8, "incumbent increased");
            callbackCosts.push_back(cost.value());
        });

    const auto status = problem.planner->solve(ob::PlannerTerminationCondition(
        [&] { return problem.sem->numIterations() >= 1500u; }));
    check(status == ob::PlannerStatus::EXACT_SOLUTION, "SEM common planner did not find exact solution");
    check(!callbackCosts.empty(), "SEM common planner did not report an incumbent");
    check(counters.sampler_interval_calls == problem.sem->numSampleCalls(),
          "one informed-sampler call is required per iteration");
    check(counters.sampler_finite_bound_calls > 0u, "post-incumbent E_xy sampling was not exercised");
    check(counters.sampler_finite_bound_calls < counters.sampler_interval_calls,
          "pre-incumbent uniform-D sampling was not exercised");
    check(problem.sem->numTreeInsertions() > 0u, "no states inserted");
    check(problem.sem->numStartConnectionChecks() > 0u && problem.sem->numStartConnectionsValid() > 0u,
          "exact start connection was not exercised");
    check(problem.sem->numMotionChecks() > 0u, "motion validator was not used");

    auto geometric = std::dynamic_pointer_cast<ompl::geometric::PathGeometric>(problem.pdef->getSolutionPath());
    check(static_cast<bool>(geometric), "solution is not geometric");
    RunRecord record;
    for (const auto *state : geometric->getStates())
        record.solution.push_back(point(state));
    check(record.solution.size() >= 2u, "solution path is incomplete");
    near(record.solution.front().x, start.x);
    near(record.solution.front().y, start.y);
    near(record.solution.back().x, goal.x);
    near(record.solution.back().y, goal.y);
    double recomputed = 0.;
    for (std::size_t index = 1; index < record.solution.size(); ++index)
    {
        check(map.validMotion(record.solution[index - 1], record.solution[index]),
              "solution contains invalid motion");
        recomputed += map.surfaceEdge(record.solution[index - 1], record.solution[index]).cost;
    }
    near(recomputed, problem.sem->bestCost().value());

    record.nodes = problem.sem->auditNodes();
    check(!record.nodes.empty() && record.nodes.front().root, "goal root missing");
    near(record.nodes.front().costToGoal, 0.);
    for (const auto &node : record.nodes)
    {
        check(node.insertionId < record.nodes.size(), "invalid insertion id");
        if (node.root)
            continue;
        check(node.parentId < record.nodes.size() && node.parentId != node.insertionId,
              "invalid parent id");
        near(node.costToGoal, record.nodes[node.parentId].costToGoal + node.edgeCostToParent);
        std::size_t cursor = node.parentId, hops = 0;
        while (!record.nodes[cursor].root)
        {
            cursor = record.nodes[cursor].parentId;
            check(++hops <= record.nodes.size(), "cycle in backward tree");
        }
    }

    ob::PlannerData data(problem.si);
    problem.planner->getPlannerData(data);
    check(data.numGoalVertices() == 1u && data.numStartVertices() == 1u,
          "PlannerData endpoints missing");
    std::vector<unsigned int> rootEdges;
    data.getEdges(data.getGoalIndex(0), rootEdges);
    check(!rootEdges.empty(), "backward goal root has no outgoing tree edges");

    record.iterations = problem.sem->numIterations();
    record.insertions = problem.sem->numTreeInsertions();
    record.rewires = problem.sem->numRewires();
    record.descendantUpdates = problem.sem->numDescendantCostUpdates();
    record.policy = counters;
    return record;
}

void exactStartConnectionTest()
{
    auto map = slopedMap();
    PolicyCounters counters;
    const Point start{2., 2.}, goal{4., 2.};
    bool attemptsRejected = false;
    try
    {
        auto invalid = makePlannerProblem(map, "sem_rrtstar_common_v1", start, goal, counters,
                                          200, 1.1, 99);
        (void)invalid;
    }
    catch (const std::runtime_error &)
    {
        attemptsRejected = true;
    }
    check(attemptsRejected, "SEM comparator accepted a non-100 informed-attempt policy");

    ompl::RNG::setSeed(991u);
    auto problem = makePlannerProblem(map, "sem_rrtstar_common_v1", start, goal, counters, 200, 1.1, 100);
    std::vector<Point> callback;
    problem.pdef->setIntermediateSolutionCallback(
        [&](const ob::Planner *, const std::vector<const ob::State *> &states, const ob::Cost) {
            if (callback.empty())
                for (const auto *state : states)
                    callback.push_back(point(state));
        });
    const auto status = problem.planner->solve(ob::PlannerTerminationCondition(
        [&] { return problem.sem->numIterations() >= 1u; }));
    check(status == ob::PlannerStatus::EXACT_SOLUTION, "direct exact start connection failed");
    check(callback.size() == 2u, "direct solution should contain exact start and goal");
    near(callback.front().x, start.x);
    near(callback.back().x, goal.x);
    const double expected = map.surfaceEdge(start, goal).cost;
    near(problem.sem->bestCost().value(), expected);
    check(expected > distanceXY(start, goal), "fixture does not distinguish c_H from d_xy");
    check(problem.sem->numStartConnectionChecks() >= 1u && problem.sem->numStartConnectionsValid() >= 1u,
          "direct start connection counters missing");
}

void blockedStartConnectionTest()
{
    Terrain map(11, 11, 0., 0., 1., 1., std::vector<double>(121u, 0.),
                std::vector<int>(100u, 0), {{3., 0., 3.5, 10.}}, .1, 0., 60.);
    PolicyCounters counters;
    const Point start{2., 5.}, goal{4., 5.};
    ompl::RNG::setSeed(992u);
    auto problem = makePlannerProblem(map, "sem_rrtstar_common_v1", start, goal, counters, 200, 1.1, 100);
    const auto status = problem.planner->solve(ob::PlannerTerminationCondition(
        [&] { return problem.sem->numIterations() >= 2u; }));
    check(status == ob::PlannerStatus::TIMEOUT, "blocked exact start connection was accepted");
    check(problem.sem->numStartConnectionChecks() >= 1u,
          "blocked exact start connection was not checked");
    check(problem.sem->numStartConnectionsValid() == 0u,
          "common Terrain motion validator did not reject the obstacle crossing");
}

}  // namespace

int main()
{
    try
    {
        exactStartConnectionTest();
        blockedStartConnectionTest();
        const RunRecord first = runDeterministic(88231u);
        const RunRecord second = runDeterministic(88231u);
        check(first.rewires > 0u, "rewiring was not exercised");
        check(first.descendantUpdates > 0u, "descendant cost propagation was not exercised");
        check(first.solution.size() == second.solution.size(), "deterministic solution size changed");
        for (std::size_t index = 0; index < first.solution.size(); ++index)
        {
            near(first.solution[index].x, second.solution[index].x);
            near(first.solution[index].y, second.solution[index].y);
        }
        check(first.iterations == second.iterations && first.insertions == second.insertions &&
                  first.rewires == second.rewires && first.descendantUpdates == second.descendantUpdates,
              "deterministic planner counters changed");
        check(first.nodes.size() == second.nodes.size(), "deterministic tree size changed");
        for (std::size_t index = 0; index < first.nodes.size(); ++index)
        {
            check(first.nodes[index].parentId == second.nodes[index].parentId,
                  "insertion-id tie-breaking changed the parent tree");
            near(first.nodes[index].costToGoal, second.nodes[index].costToGoal);
        }
        std::cout << "PASS sem_rrtstar_common_v1 backward tree, exact start connection, common c_H/validator, "
                     "D-to-E_xy sampling, rewiring propagation and deterministic insertion IDs\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
