#include "terrain25d/SemRRTstarCommon.hpp"

#include <ompl/base/PlannerData.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/samplers/InformedStateSampler.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/util/Console.h>
#include <ompl/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace terrain25d {

namespace {

constexpr std::size_t NO_NODE = std::numeric_limits<std::size_t>::max();

bool finiteCost(const ob::OptimizationObjectivePtr &objective, const ob::Cost cost)
{
    return objective->isFinite(cost) && std::isfinite(cost.value());
}

}  // namespace

struct SemRRTstarCommon::Impl {
    struct Node {
        ob::State *state{nullptr};
        std::size_t parent{NO_NODE};
        std::vector<std::size_t> children;
        ob::Cost costToGoal{std::numeric_limits<double>::infinity()};
        ob::Cost edgeCostToParent{std::numeric_limits<double>::infinity()};
        bool startConnectionValid{false};
        ob::Cost startEdgeCost{std::numeric_limits<double>::infinity()};
    };

    explicit Impl(SemRRTstarCommon &planner) : planner(planner)
    {
    }

    void freeStates()
    {
        for (auto &node : nodes)
            if (node.state != nullptr)
                planner.getSpaceInformation()->freeState(node.state);
        nodes.clear();
        if (startState != nullptr)
            planner.getSpaceInformation()->freeState(startState);
        startState = nullptr;
    }

    void reset()
    {
        freeStates();
        bestPath.reset();
        bestCost = ob::Cost(std::numeric_limits<double>::infinity());
        bestConnection = NO_NODE;
        initialized = false;
        iterations = 0;
        sampleCalls = 0;
        treeInsertions = 0;
        stateRejections = 0;
        motionChecks = 0;
        rewires = 0;
        descendantCostUpdates = 0;
        startConnectionChecks = 0;
        startConnectionsValid = 0;
        samplerFailures = 0;
    }

    bool costBetter(const ob::Cost lhs, const ob::Cost rhs) const
    {
        return objective->isCostBetterThan(lhs, rhs);
    }

    bool costsEquivalent(const ob::Cost lhs, const ob::Cost rhs) const
    {
        return !costBetter(lhs, rhs) && !costBetter(rhs, lhs);
    }

    std::size_t nearest(const ob::State *state) const
    {
        std::size_t selected = NO_NODE;
        double selectedDistance = std::numeric_limits<double>::infinity();
        for (std::size_t id = 0; id < nodes.size(); ++id)
        {
            const double distance = planner.getSpaceInformation()->distance(nodes[id].state, state);
            if (distance < selectedDistance || (distance == selectedDistance && id < selected))
            {
                selected = id;
                selectedDistance = distance;
            }
        }
        return selected;
    }

    std::vector<std::size_t> near(const ob::State *state) const
    {
        std::vector<std::size_t> result;
        for (std::size_t id = 0; id < nodes.size(); ++id)
            if (planner.getSpaceInformation()->distance(nodes[id].state, state) <=
                SemRRTstarCommon::NEAR_RADIUS_M)
                result.push_back(id);
        return result;  // In insertion-id order by construction.
    }

    bool isAncestor(const std::size_t possibleAncestor, std::size_t node) const
    {
        while (node != NO_NODE)
        {
            if (node == possibleAncestor)
                return true;
            node = nodes[node].parent;
        }
        return false;
    }

    void removeChild(const std::size_t parent, const std::size_t child)
    {
        auto &children = nodes[parent].children;
        const auto found = std::find(children.begin(), children.end(), child);
        if (found == children.end())
            throw ompl::Exception(planner.getName(), "tree child relation is inconsistent");
        children.erase(found);
    }

    void propagateDescendantCosts(const std::size_t root)
    {
        std::vector<std::size_t> queue(nodes[root].children.begin(), nodes[root].children.end());
        for (std::size_t index = 0; index < queue.size(); ++index)
        {
            const std::size_t child = queue[index];
            Node &node = nodes[child];
            node.costToGoal = objective->combineCosts(nodes[node.parent].costToGoal, node.edgeCostToParent);
            ++descendantCostUpdates;
            for (const std::size_t grandchild : node.children)
                queue.push_back(grandchild);
        }
    }

    void evaluateStartConnection(const std::size_t id)
    {
        Node &node = nodes[id];
        if (planner.getSpaceInformation()->distance(startState, node.state) >
            SemRRTstarCommon::START_CONNECTION_RADIUS_M)
            return;
        ++startConnectionChecks;
        ++motionChecks;
        if (!planner.getSpaceInformation()->checkMotion(startState, node.state))
            return;
        const ob::Cost edge = objective->motionCost(startState, node.state);
        if (!finiteCost(objective, edge))
            return;
        node.startConnectionValid = true;
        node.startEdgeCost = edge;
        ++startConnectionsValid;
    }

    std::shared_ptr<ompl::geometric::PathGeometric> makeSolutionPath(const std::size_t connection) const
    {
        auto path = std::make_shared<ompl::geometric::PathGeometric>(planner.getSpaceInformation());
        path->append(startState);
        std::size_t cursor = connection;
        if (planner.getSpaceInformation()->distance(startState, nodes[cursor].state) != 0.0)
            path->append(nodes[cursor].state);
        while (nodes[cursor].parent != NO_NODE)
        {
            cursor = nodes[cursor].parent;
            path->append(nodes[cursor].state);
        }
        return path;
    }

    std::vector<const ob::State *> callbackStates(const std::size_t connection) const
    {
        std::vector<const ob::State *> result{startState};
        std::size_t cursor = connection;
        if (planner.getSpaceInformation()->distance(startState, nodes[cursor].state) != 0.0)
            result.push_back(nodes[cursor].state);
        while (nodes[cursor].parent != NO_NODE)
        {
            cursor = nodes[cursor].parent;
            result.push_back(nodes[cursor].state);
        }
        return result;
    }

    bool updateIncumbent()
    {
        std::size_t selected = NO_NODE;
        ob::Cost selectedCost = objective->infiniteCost();
        for (std::size_t id = 0; id < nodes.size(); ++id)
        {
            const Node &node = nodes[id];
            if (!node.startConnectionValid)
                continue;
            const ob::Cost candidate = objective->combineCosts(node.startEdgeCost, node.costToGoal);
            if (!finiteCost(objective, candidate))
                continue;
            if (selected == NO_NODE || costBetter(candidate, selectedCost) ||
                (costsEquivalent(candidate, selectedCost) && id < selected))
            {
                selected = id;
                selectedCost = candidate;
            }
        }
        if (selected == NO_NODE)
            return false;

        const bool strictlyBetter = bestConnection == NO_NODE || costBetter(selectedCost, bestCost);
        const bool lowerIdTie = bestConnection != NO_NODE && costsEquivalent(selectedCost, bestCost) &&
                                selected < bestConnection;
        if (!strictlyBetter && !lowerIdTie)
            return false;

        bestConnection = selected;
        bestCost = selectedCost;
        bestPath = makeSolutionPath(selected);
        // Replacing an equal-cost incumbent only enforces the deterministic
        // insertion-id tie break; it is not a new objective improvement.
        if (strictlyBetter)
        {
            const auto &callback = planner.getProblemDefinition()->getIntermediateSolutionCallback();
            if (callback)
                callback(&planner, callbackStates(selected), bestCost);
        }
        return true;
    }

    bool initialize(const ob::PlannerTerminationCondition &ptc)
    {
        if (initialized)
            return true;
        if (planner.getProblemDefinition()->getStartStateCount() != 1u)
            throw ompl::Exception(planner.getName(), "requires exactly one start state");

        const ob::State *start = planner.getProblemDefinition()->getStartState(0);
        if (!planner.getSpaceInformation()->isValid(start))
            return false;
        auto *goal = dynamic_cast<ob::GoalSampleableRegion *>(planner.getProblemDefinition()->getGoal().get());
        if (goal == nullptr || !goal->couldSample())
            return false;
        const ob::State *goalState = planner.pis_.nextGoal(ptc);
        if (goalState == nullptr || !planner.getSpaceInformation()->isValid(goalState))
            return false;

        // Do not retain a cloned start across a failed goal initialization.  A
        // later solve() call may retry initialization after a termination
        // condition prevented the goal from being obtained.
        startState = planner.getSpaceInformation()->cloneState(start);
        Node root;
        root.state = planner.getSpaceInformation()->cloneState(goalState);
        root.costToGoal = objective->identityCost();
        root.edgeCostToParent = objective->identityCost();
        nodes.push_back(std::move(root));
        evaluateStartConnection(0u);
        updateIncumbent();
        initialized = true;
        return true;
    }

    SemRRTstarCommon &planner;
    ob::OptimizationObjectivePtr objective;
    ob::InformedSamplerPtr sampler;
    std::vector<Node> nodes;
    ob::State *startState{nullptr};
    std::shared_ptr<ompl::geometric::PathGeometric> bestPath;
    ob::Cost bestCost{std::numeric_limits<double>::infinity()};
    std::size_t bestConnection{NO_NODE};
    bool initialized{false};
    std::uint64_t iterations{0};
    std::uint64_t sampleCalls{0};
    std::uint64_t treeInsertions{0};
    std::uint64_t stateRejections{0};
    std::uint64_t motionChecks{0};
    std::uint64_t rewires{0};
    std::uint64_t descendantCostUpdates{0};
    std::uint64_t startConnectionChecks{0};
    std::uint64_t startConnectionsValid{0};
    std::uint64_t samplerFailures{0};
};

SemRRTstarCommon::SemRRTstarCommon(const ob::SpaceInformationPtr &si)
  : ob::Planner(si, "SEM_RRTstar_common_v1"), impl_(std::make_unique<Impl>(*this))
{
    specs_.recognizedGoal = ob::GOAL_SAMPLEABLE_REGION;
    specs_.approximateSolutions = false;
    specs_.optimizingPaths = true;
    specs_.directed = false;
    specs_.canReportIntermediateSolutions = true;

    addPlannerProgressProperty("iterations INTEGER", [this] { return std::to_string(numIterations()); });
    addPlannerProgressProperty("sample calls INTEGER", [this] { return std::to_string(numSampleCalls()); });
    addPlannerProgressProperty("tree insertions INTEGER", [this] { return std::to_string(numTreeInsertions()); });
    addPlannerProgressProperty("rewires INTEGER", [this] { return std::to_string(numRewires()); });
    addPlannerProgressProperty("best cost REAL", [this] { return std::to_string(bestCost().value()); });
}

SemRRTstarCommon::~SemRRTstarCommon()
{
    impl_->freeStates();
}

void SemRRTstarCommon::setup()
{
    ob::Planner::setup();
    if (si_->getStateDimension() != 2u)
        throw ompl::Exception(getName(), "requires a two-dimensional XY state space");
    if (!pdef_)
    {
        OMPL_INFORM("%s: problem definition is not set, deferring setup completion...", getName().c_str());
        setup_ = false;
        return;
    }
    impl_->objective = pdef_->hasOptimizationObjective()
                           ? pdef_->getOptimizationObjective()
                           : std::make_shared<ob::PathLengthOptimizationObjective>(si_);
    if (!impl_->objective->isSymmetric())
        throw ompl::Exception(getName(), "requires a symmetric optimization objective");
    impl_->sampler = impl_->objective->allocInformedStateSampler(pdef_, INFORMED_ATTEMPTS);
    if (!impl_->sampler)
        throw ompl::Exception(getName(), "objective did not allocate an informed sampler");
}

void SemRRTstarCommon::clear()
{
    setup_ = false;
    ob::Planner::clear();
    impl_->reset();
    impl_->sampler.reset();
    impl_->objective.reset();
}

ob::PlannerStatus SemRRTstarCommon::solve(const ob::PlannerTerminationCondition &ptc)
{
    checkValidity();
    if (!impl_->initialize(ptc))
        return pdef_->getStartStateCount() == 0u || !si_->isValid(pdef_->getStartState(0))
                   ? ob::PlannerStatus::INVALID_START
                   : ob::PlannerStatus::INVALID_GOAL;

    ob::ScopedState<> sample(si_);
    ob::ScopedState<> candidateState(si_);
    while (!ptc)
    {
        ++impl_->iterations;
        ++impl_->sampleCalls;
        const ob::Cost bound = impl_->bestConnection == NO_NODE ? impl_->objective->infiniteCost() : impl_->bestCost;
        if (!impl_->sampler->sampleUniform(sample.get(), bound))
        {
            ++impl_->samplerFailures;
            continue;
        }

        const std::size_t nearestId = impl_->nearest(sample.get());
        const double distance = si_->distance(impl_->nodes[nearestId].state, sample.get());
        if (distance == 0.0)
        {
            ++impl_->stateRejections;
            continue;
        }
        const double fraction = std::min(1.0, STEER_RANGE_M / distance);
        si_->getStateSpace()->interpolate(impl_->nodes[nearestId].state, sample.get(), fraction,
                                          candidateState.get());
        if (!si_->isValid(candidateState.get()))
        {
            ++impl_->stateRejections;
            continue;
        }

        const auto nearIds = impl_->near(candidateState.get());
        struct ParentCandidate {
            std::size_t id;
            ob::Cost edge;
            ob::Cost total;
        };
        std::vector<ParentCandidate> parents;
        parents.reserve(nearIds.size());
        for (const std::size_t id : nearIds)
        {
            const ob::Cost edge = impl_->objective->motionCost(candidateState.get(), impl_->nodes[id].state);
            if (!finiteCost(impl_->objective, edge))
                continue;
            const ob::Cost total = impl_->objective->combineCosts(impl_->nodes[id].costToGoal, edge);
            if (finiteCost(impl_->objective, total))
                parents.push_back({id, edge, total});
        }
        std::sort(parents.begin(), parents.end(), [&](const ParentCandidate &lhs, const ParentCandidate &rhs) {
            if (impl_->costBetter(lhs.total, rhs.total))
                return true;
            if (impl_->costBetter(rhs.total, lhs.total))
                return false;
            return lhs.id < rhs.id;
        });

        const ParentCandidate *selectedParent = nullptr;
        for (const auto &parent : parents)
        {
            ++impl_->motionChecks;
            if (si_->checkMotion(candidateState.get(), impl_->nodes[parent.id].state))
            {
                selectedParent = &parent;
                break;
            }
        }
        if (selectedParent == nullptr)
        {
            ++impl_->stateRejections;
            continue;
        }

        const std::size_t newId = impl_->nodes.size();
        Impl::Node node;
        node.state = si_->cloneState(candidateState.get());
        node.parent = selectedParent->id;
        node.costToGoal = selectedParent->total;
        node.edgeCostToParent = selectedParent->edge;
        impl_->nodes.push_back(std::move(node));
        impl_->nodes[selectedParent->id].children.push_back(newId);
        ++impl_->treeInsertions;
        impl_->evaluateStartConnection(newId);

        for (const std::size_t neighbor : nearIds)
        {
            if (neighbor == selectedParent->id || impl_->nodes[neighbor].parent == NO_NODE ||
                impl_->isAncestor(neighbor, newId))
                continue;
            const ob::Cost edge = impl_->objective->motionCost(impl_->nodes[neighbor].state,
                                                               impl_->nodes[newId].state);
            if (!finiteCost(impl_->objective, edge))
                continue;
            const ob::Cost proposed = impl_->objective->combineCosts(impl_->nodes[newId].costToGoal, edge);
            if (!finiteCost(impl_->objective, proposed) || !impl_->costBetter(proposed, impl_->nodes[neighbor].costToGoal))
                continue;
            ++impl_->motionChecks;
            if (!si_->checkMotion(impl_->nodes[neighbor].state, impl_->nodes[newId].state))
                continue;

            const std::size_t oldParent = impl_->nodes[neighbor].parent;
            impl_->removeChild(oldParent, neighbor);
            impl_->nodes[neighbor].parent = newId;
            impl_->nodes[neighbor].edgeCostToParent = edge;
            impl_->nodes[neighbor].costToGoal = proposed;
            impl_->nodes[newId].children.push_back(neighbor);
            ++impl_->rewires;
            impl_->propagateDescendantCosts(neighbor);
        }
        impl_->updateIncumbent();
    }
    if (impl_->bestPath)
    {
        ob::PlannerSolution solution(impl_->bestPath);
        solution.setPlannerName(getName());
        solution.setOptimized(impl_->objective, impl_->bestCost, impl_->objective->isSatisfied(impl_->bestCost));
        pdef_->addSolutionPath(solution);
        return ob::PlannerStatus::EXACT_SOLUTION;
    }
    return ob::PlannerStatus::TIMEOUT;
}

void SemRRTstarCommon::getPlannerData(ob::PlannerData &data) const
{
    ob::Planner::getPlannerData(data);
    for (std::size_t id = 0; id < impl_->nodes.size(); ++id)
    {
        const ob::PlannerDataVertex vertex(impl_->nodes[id].state, 1);
        if (impl_->nodes[id].parent == NO_NODE)
            data.addGoalVertex(vertex);
        else
            data.addVertex(vertex);
    }
    for (std::size_t child = 0; child < impl_->nodes.size(); ++child)
    {
        const std::size_t parent = impl_->nodes[child].parent;
        if (parent == NO_NODE)
            continue;
        data.addEdge(ob::PlannerDataVertex(impl_->nodes[parent].state, 1),
                     ob::PlannerDataVertex(impl_->nodes[child].state, 1), ob::PlannerDataEdge(),
                     impl_->nodes[child].edgeCostToParent);
    }
    if (impl_->bestConnection != NO_NODE && impl_->startState != nullptr)
    {
        const ob::PlannerDataVertex start(impl_->startState, 1);
        data.addStartVertex(start);
        data.addEdge(ob::PlannerDataVertex(impl_->nodes[impl_->bestConnection].state, 1), start,
                     ob::PlannerDataEdge(), impl_->nodes[impl_->bestConnection].startEdgeCost);
    }
}

std::uint64_t SemRRTstarCommon::numIterations() const noexcept { return impl_->iterations; }
std::uint64_t SemRRTstarCommon::numSampleCalls() const noexcept { return impl_->sampleCalls; }
std::uint64_t SemRRTstarCommon::numTreeInsertions() const noexcept { return impl_->treeInsertions; }
std::uint64_t SemRRTstarCommon::numStateRejections() const noexcept { return impl_->stateRejections; }
std::uint64_t SemRRTstarCommon::numMotionChecks() const noexcept { return impl_->motionChecks; }
std::uint64_t SemRRTstarCommon::numRewires() const noexcept { return impl_->rewires; }
std::uint64_t SemRRTstarCommon::numDescendantCostUpdates() const noexcept { return impl_->descendantCostUpdates; }
std::uint64_t SemRRTstarCommon::numStartConnectionChecks() const noexcept { return impl_->startConnectionChecks; }
std::uint64_t SemRRTstarCommon::numStartConnectionsValid() const noexcept { return impl_->startConnectionsValid; }
std::uint64_t SemRRTstarCommon::numSamplerFailures() const noexcept { return impl_->samplerFailures; }
std::size_t SemRRTstarCommon::vertexCount() const noexcept { return impl_->nodes.size(); }
ob::Cost SemRRTstarCommon::bestCost() const noexcept { return impl_->bestCost; }

std::vector<SemRRTstarCommon::NodeAudit> SemRRTstarCommon::auditNodes() const
{
    std::vector<NodeAudit> result;
    result.reserve(impl_->nodes.size());
    for (std::size_t id = 0; id < impl_->nodes.size(); ++id)
    {
        const auto &node = impl_->nodes[id];
        result.push_back({id, node.parent, node.parent == NO_NODE, node.costToGoal.value(),
                          node.edgeCostToParent.value()});
    }
    return result;
}

}  // namespace terrain25d
