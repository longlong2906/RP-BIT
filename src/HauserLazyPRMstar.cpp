#include "terrain25d/HauserLazyPRMstar.hpp"

#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/datastructures/NearestNeighbors.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/tools/config/SelfConfig.h>
#include <ompl/util/Console.h>
#include <ompl/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace terrain25d {

namespace {

constexpr std::size_t NO_INDEX = std::numeric_limits<std::size_t>::max();

enum class EdgeStatus {
    UNKNOWN,
    VALID,
    INVALID,
};

enum class SearchStatus {
    PATH,
    NO_PATH,
    TERMINATED,
};

enum class ResolveStatus {
    SETTLED,
    OBJECTIVE_SATISFIED,
    TERMINATED,
};

struct QueueItem {
    std::size_t vertex;
    ob::Cost cost;
};

struct QueueCompare {
    const ob::OptimizationObjective *objective;

    bool operator()(const QueueItem &lhs, const QueueItem &rhs) const
    {
        // std::priority_queue puts the element considered "largest" first.
        // Put lhs behind rhs precisely when rhs has the better cost.
        if (objective->isCostBetterThan(rhs.cost, lhs.cost))
            return true;
        if (objective->isCostBetterThan(lhs.cost, rhs.cost))
            return false;
        return lhs.vertex > rhs.vertex;
    }
};

}  // namespace

struct HauserLazyPRMstar::Impl {
    using VertexId = std::size_t;
    using EdgeId = std::size_t;

    struct Vertex {
        ob::State *state{nullptr};
        std::vector<EdgeId> incidentEdges;
        bool isStart{false};
        bool isGoal{false};
        ob::Cost cost{std::numeric_limits<double>::infinity()};
        VertexId parent{NO_INDEX};
        EdgeId parentEdge{NO_INDEX};
    };

    struct Edge {
        VertexId first{NO_INDEX};
        VertexId second{NO_INDEX};
        ob::Cost weight{std::numeric_limits<double>::infinity()};
        EdgeStatus status{EdgeStatus::UNKNOWN};
    };

    struct CandidatePath {
        SearchStatus status{SearchStatus::NO_PATH};
        std::vector<VertexId> vertices;
        std::vector<EdgeId> edges;
        ob::Cost cost{std::numeric_limits<double>::infinity()};
    };

    explicit Impl(HauserLazyPRMstar &planner) : planner(planner)
    {
    }

    void initializeNearestNeighbors()
    {
        if (nearestNeighbors)
            return;

        nearestNeighbors.reset(ompl::tools::SelfConfig::getDefaultNearestNeighbors<VertexId>(&planner));
        nearestNeighbors->setDistanceFunction([this](const VertexId lhs, const VertexId rhs) {
            return planner.getSpaceInformation()->distance(vertices.at(lhs).state, vertices.at(rhs).state);
        });
    }

    void freeStates()
    {
        for (auto &vertex : vertices)
            if (vertex.state != nullptr)
                planner.getSpaceInformation()->freeState(vertex.state);
        vertices.clear();
    }

    void reset()
    {
        freeStates();
        edges.clear();
        startVertices.clear();
        goalVertices.clear();
        if (nearestNeighbors)
            nearestNeighbors->clear();

        bestPath.reset();
        bestCost = ob::Cost(std::numeric_limits<double>::infinity());
        activeEdgeCount = 0;
        iterations = 0;
        sampleProposals = 0;
        stateRejections = 0;
        motionChecks = 0;
        shortestPathSearches = 0;
        invalidEdges = 0;
        undefinedCostEdges = 0;
        pendingDecrease.clear();
        activeDecreaseVertex = NO_INDEX;
        activeDecreaseEdge = 0;
    }

    VertexId addMilestone(ob::State *state, const bool isStart, const bool isGoal)
    {
        const VertexId id = vertices.size();
        vertices.push_back(Vertex{state, {}, isStart, isGoal, objective->infiniteCost(), NO_INDEX, NO_INDEX});
        if (isStart)
            startVertices.push_back(id);
        if (isGoal)
            goalVertices.push_back(id);

        nearestNeighbors->add(id);
        if (isStart)
            improveRoot(id);

        const std::size_t k = HauserLazyPRMstar::connectionK(vertices.size());
        if (k == 0)
            return id;

        // Probe one non-self neighbor beyond the k-nearest cutoff.  The usual
        // case stays O(k log k); only an actual distance tie at the cutoff
        // needs a radius query so the deterministic (distance, id) rule can
        // choose among every tied milestone.
        std::vector<VertexId> neighbors;
        nearestNeighbors->nearestK(id, std::min(vertices.size(), k + 2u), neighbors);
        neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), id), neighbors.end());
        const auto distanceToNew = [this, id](const VertexId other) {
            return planner.getSpaceInformation()->distance(vertices.at(id).state, vertices.at(other).state);
        };
        const auto distanceThenId = [this, id](const VertexId lhs, const VertexId rhs) {
            const double lhsDistance = planner.getSpaceInformation()->distance(vertices.at(id).state,
                                                                                vertices.at(lhs).state);
            const double rhsDistance = planner.getSpaceInformation()->distance(vertices.at(id).state,
                                                                                vertices.at(rhs).state);
            if (lhsDistance < rhsDistance)
                return true;
            if (rhsDistance < lhsDistance)
                return false;
            return lhs < rhs;
        };
        std::sort(neighbors.begin(), neighbors.end(), distanceThenId);
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        if (neighbors.size() > k && distanceToNew(neighbors[k - 1u]) == distanceToNew(neighbors[k]))
        {
            const double cutoff = distanceToNew(neighbors[k - 1u]);
            nearestNeighbors->nearestR(id, std::nextafter(cutoff, std::numeric_limits<double>::infinity()), neighbors);
            neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), id), neighbors.end());
            std::sort(neighbors.begin(), neighbors.end(), distanceThenId);
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        if (neighbors.size() > k)
            neighbors.resize(k);

        std::vector<EdgeId> addedEdges;
        for (const VertexId neighbor : neighbors)
        {
            const ob::Cost weight = objective->motionCost(state, vertices.at(neighbor).state);
            if (!objective->isFinite(weight) || !std::isfinite(weight.value()))
            {
                ++undefinedCostEdges;
                continue;
            }
            const EdgeId edgeId = edges.size();
            edges.push_back(Edge{id, neighbor, weight, EdgeStatus::UNKNOWN});
            vertices.at(id).incidentEdges.push_back(edgeId);
            vertices.at(neighbor).incidentEdges.push_back(edgeId);
            addedEdges.push_back(edgeId);
            ++activeEdgeCount;
        }

        for (const EdgeId edgeId : addedEdges)
        {
            const Edge &edge = edges.at(edgeId);
            relax(edge.first, edge.second, edgeId);
            relax(edge.second, edge.first, edgeId);
        }
        return id;
    }

    bool finite(const ob::Cost cost) const
    {
        return objective->isFinite(cost) && std::isfinite(cost.value());
    }

    void enqueueDecrease(const VertexId vertex)
    {
        pendingDecrease.push_back(QueueItem{vertex, vertices.at(vertex).cost});
        std::push_heap(pendingDecrease.begin(), pendingDecrease.end(), QueueCompare{objective.get()});
    }

    void improveRoot(const VertexId vertex)
    {
        const ob::Cost initial = objective->initialCost(vertices.at(vertex).state);
        if (!finite(initial))
            return;
        Vertex &root = vertices.at(vertex);
        if (root.parent == NO_INDEX || objective->isCostBetterThan(initial, root.cost))
        {
            root.cost = initial;
            root.parent = vertex;
            root.parentEdge = NO_INDEX;
            enqueueDecrease(vertex);
        }
    }

    void relax(const VertexId from, const VertexId to, const EdgeId edgeId)
    {
        const Vertex &source = vertices.at(from);
        if (!finite(source.cost))
            return;
        const Edge &edge = edges.at(edgeId);
        if (edge.status == EdgeStatus::INVALID)
            return;

        const ob::Cost candidate = objective->combineCosts(source.cost, edge.weight);
        if (!finite(candidate))
            return;
        Vertex &target = vertices.at(to);
        if (target.parent == NO_INDEX || objective->isCostBetterThan(candidate, target.cost))
        {
            target.cost = candidate;
            target.parent = from;
            target.parentEdge = edgeId;
            enqueueDecrease(to);
        }
    }

    bool propagateDecreases(const ob::PlannerTerminationCondition &ptc)
    {
        const QueueCompare compare{objective.get()};
        for (;;)
        {
            if (activeDecreaseVertex == NO_INDEX)
            {
                if (pendingDecrease.empty())
                    return true;
                if (ptc)
                    return false;

                std::pop_heap(pendingDecrease.begin(), pendingDecrease.end(), compare);
                const QueueItem current = pendingDecrease.back();
                pendingDecrease.pop_back();
                if (objective->isCostBetterThan(vertices.at(current.vertex).cost, current.cost))
                    continue;  // Stale queue entry.
                activeDecreaseVertex = current.vertex;
                activeDecreaseEdge = 0;
            }

            const auto &incident = vertices.at(activeDecreaseVertex).incidentEdges;
            while (activeDecreaseEdge < incident.size())
            {
                if (ptc)
                    return false;
                const EdgeId edgeId = incident[activeDecreaseEdge++];
                const Edge &edge = edges.at(edgeId);
                if (edge.status == EdgeStatus::INVALID)
                    continue;
                const VertexId neighbor = edge.first == activeDecreaseVertex ? edge.second : edge.first;
                relax(activeDecreaseVertex, neighbor, edgeId);
            }
            activeDecreaseVertex = NO_INDEX;
        }
    }

    void removeInvalidEdge(const EdgeId edgeId)
    {
        Edge &edge = edges.at(edgeId);
        edge.status = EdgeStatus::INVALID;
        --activeEdgeCount;
        ++invalidEdges;

        VertexId affectedRoot = NO_INDEX;
        if (vertices.at(edge.first).parentEdge == edgeId)
            affectedRoot = edge.first;
        else if (vertices.at(edge.second).parentEdge == edgeId)
            affectedRoot = edge.second;
        if (affectedRoot == NO_INDEX)
            return;

        // Alg. 5 increase propagation: invalidate the detached shortest-path
        // subtree, seed every boundary reconnection, then run the same
        // decrease propagation used for edge insertions.
        std::vector<std::vector<VertexId>> children(vertices.size());
        for (VertexId vertex = 0; vertex < vertices.size(); ++vertex)
            if (vertices[vertex].parent != NO_INDEX && vertices[vertex].parent != vertex)
                children[vertices[vertex].parent].push_back(vertex);

        std::vector<bool> affected(vertices.size(), false);
        std::vector<VertexId> stack{affectedRoot};
        affected[affectedRoot] = true;
        for (std::size_t index = 0; index < stack.size(); ++index)
            for (const VertexId child : children[stack[index]])
                if (!affected[child])
                {
                    affected[child] = true;
                    stack.push_back(child);
                }

        for (const VertexId vertex : stack)
        {
            vertices[vertex].cost = objective->infiniteCost();
            vertices[vertex].parent = NO_INDEX;
            vertices[vertex].parentEdge = NO_INDEX;
        }
        for (const VertexId vertex : stack)
            if (vertices[vertex].isStart)
                improveRoot(vertex);
        for (const VertexId vertex : stack)
            for (const EdgeId boundaryEdgeId : vertices[vertex].incidentEdges)
            {
                const Edge &boundaryEdge = edges.at(boundaryEdgeId);
                if (boundaryEdge.status == EdgeStatus::INVALID)
                    continue;
                const VertexId neighbor = boundaryEdge.first == vertex ? boundaryEdge.second : boundaryEdge.first;
                if (!affected[neighbor])
                    relax(neighbor, vertex, boundaryEdgeId);
            }
    }

    CandidatePath shortestPath(const ob::PlannerTerminationCondition &ptc)
    {
        ++shortestPathSearches;
        CandidatePath result;
        if (!propagateDecreases(ptc))
        {
            result.status = SearchStatus::TERMINATED;
            return result;
        }
        if (startVertices.empty() || goalVertices.empty())
            return result;

        VertexId selectedGoal = NO_INDEX;
        ob::Cost selectedCost = objective->infiniteCost();
        for (const VertexId goal : goalVertices)
        {
            if (ptc)
            {
                result.status = SearchStatus::TERMINATED;
                return result;
            }
            if (vertices.at(goal).parent == NO_INDEX)
                continue;
            const ob::Cost total = objective->combineCosts(
                vertices.at(goal).cost, objective->terminalCost(vertices.at(goal).state));
            if (selectedGoal == NO_INDEX || objective->isCostBetterThan(total, selectedCost))
            {
                selectedGoal = goal;
                selectedCost = total;
            }
        }
        if (selectedGoal == NO_INDEX)
            return result;

        result.status = SearchStatus::PATH;
        result.cost = selectedCost;
        VertexId cursor = selectedGoal;
        result.vertices.push_back(cursor);
        while (vertices.at(cursor).parent != cursor)
        {
            if (ptc)
            {
                result.status = SearchStatus::TERMINATED;
                result.vertices.clear();
                result.edges.clear();
                return result;
            }
            if (vertices.at(cursor).parent == NO_INDEX || vertices.at(cursor).parentEdge == NO_INDEX ||
                result.vertices.size() > vertices.size())
            {
                // This should be impossible, but treating a corrupt predecessor
                // chain as no path is safer than looping past the deadline.
                result.status = SearchStatus::NO_PATH;
                result.vertices.clear();
                result.edges.clear();
                return result;
            }
            result.edges.push_back(vertices.at(cursor).parentEdge);
            cursor = vertices.at(cursor).parent;
            result.vertices.push_back(cursor);
        }
        std::reverse(result.vertices.begin(), result.vertices.end());
        std::reverse(result.edges.begin(), result.edges.end());
        return result;
    }

    ResolveStatus resolveLazyCandidate(const ob::PlannerTerminationCondition &ptc)
    {
        for (;;)
        {
            CandidatePath candidate = shortestPath(ptc);
            if (candidate.status == SearchStatus::TERMINATED)
                return ResolveStatus::TERMINATED;
            if (candidate.status == SearchStatus::NO_PATH)
                return ResolveStatus::SETTLED;

            // A shortest optimistic path that cannot improve the incumbent
            // proves that no unchecked path in the current roadmap can do so.
            if (bestPath && !objective->isCostBetterThan(candidate.cost, bestCost))
                return ResolveStatus::SETTLED;

            bool removedInvalidEdge = false;
            for (std::size_t i = 0; i < candidate.edges.size(); ++i)
            {
                if (ptc)
                    return ResolveStatus::TERMINATED;

                Edge &edge = edges.at(candidate.edges[i]);
                if (edge.status == EdgeStatus::VALID)
                    continue;

                ++motionChecks;
                const ob::State *from = vertices.at(candidate.vertices[i]).state;
                const ob::State *to = vertices.at(candidate.vertices[i + 1]).state;
                if (planner.getSpaceInformation()->checkMotion(from, to))
                {
                    edge.status = EdgeStatus::VALID;
                    continue;
                }

                // Hauser's forward selector: remove only the first invalid
                // edge, update c/pi by increase propagation, then extract the
                // next shortest candidate.
                removeInvalidEdge(candidate.edges[i]);
                removedInvalidEdge = true;
                break;
            }
            if (removedInvalidEdge)
                continue;

            auto path = std::make_shared<ompl::geometric::PathGeometric>(planner.getSpaceInformation());
            std::vector<const ob::State *> callbackStates;
            callbackStates.reserve(candidate.vertices.size());
            for (const VertexId vertex : candidate.vertices)
            {
                const ob::State *state = vertices.at(vertex).state;
                path->append(state);
                callbackStates.push_back(state);
            }

            bestPath = std::move(path);
            bestCost = candidate.cost;
            const auto &callback = planner.getProblemDefinition()->getIntermediateSolutionCallback();
            if (callback)
                callback(&planner, callbackStates, bestCost);

            return objective->isSatisfied(bestCost) ? ResolveStatus::OBJECTIVE_SATISFIED
                                                    : ResolveStatus::SETTLED;
        }
    }

    HauserLazyPRMstar &planner;
    ob::OptimizationObjectivePtr objective;
    ob::StateSamplerPtr sampler;
    std::shared_ptr<ompl::NearestNeighbors<VertexId>> nearestNeighbors;
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<VertexId> startVertices;
    std::vector<VertexId> goalVertices;
    std::shared_ptr<ompl::geometric::PathGeometric> bestPath;
    ob::Cost bestCost{std::numeric_limits<double>::infinity()};
    std::size_t activeEdgeCount{0};
    std::uint64_t iterations{0};
    std::uint64_t sampleProposals{0};
    std::uint64_t stateRejections{0};
    std::uint64_t motionChecks{0};
    std::uint64_t shortestPathSearches{0};
    std::uint64_t invalidEdges{0};
    std::uint64_t undefinedCostEdges{0};
    std::vector<QueueItem> pendingDecrease;
    VertexId activeDecreaseVertex{NO_INDEX};
    std::size_t activeDecreaseEdge{0};
};

HauserLazyPRMstar::HauserLazyPRMstar(const ob::SpaceInformationPtr &si)
  : ob::Planner(si, "HauserLazyPRMstar"), impl_(std::make_unique<Impl>(*this))
{
    specs_.recognizedGoal = ob::GOAL_SAMPLEABLE_REGION;
    specs_.approximateSolutions = false;
    specs_.optimizingPaths = true;
    specs_.directed = false;
    specs_.canReportIntermediateSolutions = true;

    addPlannerProgressProperty("iterations INTEGER", [this] { return std::to_string(numIterations()); });
    addPlannerProgressProperty("best cost REAL", [this] { return std::to_string(bestCost().value()); });
    addPlannerProgressProperty("milestone count INTEGER", [this] { return std::to_string(milestoneCount()); });
    addPlannerProgressProperty("edge count INTEGER", [this] { return std::to_string(edgeCount()); });
    addPlannerProgressProperty("sample proposals INTEGER",
                               [this] { return std::to_string(numSampleProposals()); });
    addPlannerProgressProperty("state rejections INTEGER",
                               [this] { return std::to_string(numStateRejections()); });
    addPlannerProgressProperty("motion checks INTEGER", [this] { return std::to_string(numMotionChecks()); });
    addPlannerProgressProperty("shortest path searches INTEGER",
                               [this] { return std::to_string(numShortestPathSearches()); });
    addPlannerProgressProperty("invalid edges INTEGER", [this] { return std::to_string(numInvalidEdges()); });
    addPlannerProgressProperty("undefined cost edges INTEGER",
                               [this] { return std::to_string(numUndefinedCostEdges()); });
}

HauserLazyPRMstar::~HauserLazyPRMstar()
{
    impl_->freeStates();
}

void HauserLazyPRMstar::setup()
{
    ob::Planner::setup();
    if (si_->getStateDimension() != 2u)
        throw ompl::Exception(getName(), "requires a two-dimensional XY state space");

    impl_->initializeNearestNeighbors();
    impl_->sampler = si_->allocStateSampler();

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
}

void HauserLazyPRMstar::clear()
{
    ob::Planner::clear();
    impl_->reset();
}

std::size_t HauserLazyPRMstar::connectionK(const std::size_t n) noexcept
{
    if (n <= 1u)
        return 0u;

    constexpr double REWIRE_FACTOR = 1.1;
    constexpr double DIMENSION_FACTOR = 1.5;  // 1 + 1/d for d=2.
    const double raw = REWIRE_FACTOR * std::exp(1.0) * DIMENSION_FACTOR * std::log(static_cast<double>(n));
    const std::size_t maximum = n - 1u;  // Exclude the milestone itself.
    if (!std::isfinite(raw) || raw >= static_cast<double>(maximum))
        return maximum;
    return std::min(maximum, static_cast<std::size_t>(std::floor(raw)));
}

ob::PlannerStatus HauserLazyPRMstar::solve(const ob::PlannerTerminationCondition &ptc)
{
    checkValidity();
    auto *goal = dynamic_cast<ob::GoalSampleableRegion *>(pdef_->getGoal().get());
    if (goal == nullptr)
    {
        OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
        return ob::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
    }

    while (const ob::State *state = pis_.nextStart())
        impl_->addMilestone(si_->cloneState(state), true, false);
    if (impl_->startVertices.empty())
    {
        OMPL_ERROR("%s: There are no valid initial states", getName().c_str());
        return ob::PlannerStatus::INVALID_START;
    }

    if (!goal->couldSample())
    {
        OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
        return ob::PlannerStatus::INVALID_GOAL;
    }
    if (impl_->goalVertices.empty() || goal->maxSampleCount() > impl_->goalVertices.size())
    {
        const ob::State *state = impl_->goalVertices.empty() ? pis_.nextGoal(ptc) : pis_.nextGoal();
        if (state != nullptr)
            impl_->addMilestone(si_->cloneState(state), false, true);
    }
    if (impl_->goalVertices.empty())
    {
        OMPL_ERROR("%s: Unable to find a valid goal state", getName().c_str());
        return ob::PlannerStatus::INVALID_GOAL;
    }

    ResolveStatus resolution = impl_->resolveLazyCandidate(ptc);
    bool stop = resolution != ResolveStatus::SETTLED;
    ob::State *workState = si_->allocState();
    while (!stop && !ptc)
    {
        ++impl_->iterations;
        ++impl_->sampleProposals;
        impl_->sampler->sampleUniform(workState);  // Exactly one proposal per iteration.
        if (!si_->isValid(workState))
        {
            ++impl_->stateRejections;
            continue;
        }

        impl_->addMilestone(si_->cloneState(workState), false, false);
        resolution = impl_->resolveLazyCandidate(ptc);
        stop = resolution != ResolveStatus::SETTLED;
    }
    si_->freeState(workState);

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

void HauserLazyPRMstar::getPlannerData(ob::PlannerData &data) const
{
    ob::Planner::getPlannerData(data);

    for (const auto &vertex : impl_->vertices)
    {
        const ob::PlannerDataVertex dataVertex(vertex.state, 1);
        if (vertex.isStart)
            data.addStartVertex(dataVertex);
        else if (vertex.isGoal)
            data.addGoalVertex(dataVertex);
        else
            data.addVertex(dataVertex);

        if (vertex.isStart && vertex.isGoal)
            data.markGoalState(vertex.state);
    }

    for (const auto &edge : impl_->edges)
    {
        if (edge.status == EdgeStatus::INVALID)
            continue;
        const ob::PlannerDataVertex first(impl_->vertices.at(edge.first).state, 1);
        const ob::PlannerDataVertex second(impl_->vertices.at(edge.second).state, 1);
        data.addEdge(first, second, ob::PlannerDataEdge(), edge.weight);
        data.addEdge(second, first, ob::PlannerDataEdge(), edge.weight);
    }
}

std::uint64_t HauserLazyPRMstar::numIterations() const noexcept
{
    return impl_->iterations;
}

std::uint64_t HauserLazyPRMstar::numSampleProposals() const noexcept
{
    return impl_->sampleProposals;
}

std::uint64_t HauserLazyPRMstar::numStateRejections() const noexcept
{
    return impl_->stateRejections;
}

std::uint64_t HauserLazyPRMstar::numMotionChecks() const noexcept
{
    return impl_->motionChecks;
}

std::uint64_t HauserLazyPRMstar::numShortestPathSearches() const noexcept
{
    return impl_->shortestPathSearches;
}

std::uint64_t HauserLazyPRMstar::numInvalidEdges() const noexcept
{
    return impl_->invalidEdges;
}

std::uint64_t HauserLazyPRMstar::numUndefinedCostEdges() const noexcept
{
    return impl_->undefinedCostEdges;
}

std::size_t HauserLazyPRMstar::milestoneCount() const noexcept
{
    return impl_->vertices.size();
}

std::size_t HauserLazyPRMstar::edgeCount() const noexcept
{
    return impl_->activeEdgeCount;
}

ob::Cost HauserLazyPRMstar::bestCost() const noexcept
{
    return impl_->bestCost;
}

}  // namespace terrain25d
