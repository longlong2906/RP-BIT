#pragma once

#include <ompl/base/Planner.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace terrain25d {

namespace ob = ompl::base;

/**
 * Incremental Lazy-PRM* baseline following Hauser (2015).
 *
 * States are proposed uniformly by the state-space sampler and rejected when
 * invalid.  Every accepted milestone is connected to
 * floor(1.1 * e * (1 + 1 / 2) * log(n)) Euclidean-XY nearest neighbours.
 * Candidate shortest paths use the problem's OptimizationObjective for edge
 * weights.  Motions are checked lazily, removing only the first invalid edge
 * before recomputing the shortest path.
 */
class HauserLazyPRMstar final : public ob::Planner {
public:
    explicit HauserLazyPRMstar(const ob::SpaceInformationPtr &si);
    ~HauserLazyPRMstar() override;

    void setup() override;
    void clear() override;
    ob::PlannerStatus solve(const ob::PlannerTerminationCondition &ptc) override;
    void getPlannerData(ob::PlannerData &data) const override;

    /** Number of non-self neighbours used when the roadmap has n vertices. */
    static std::size_t connectionK(std::size_t n) noexcept;

    std::uint64_t numIterations() const noexcept;
    std::uint64_t numSampleProposals() const noexcept;
    std::uint64_t numStateRejections() const noexcept;
    std::uint64_t numMotionChecks() const noexcept;
    std::uint64_t numShortestPathSearches() const noexcept;
    std::uint64_t numInvalidEdges() const noexcept;
    std::uint64_t numUndefinedCostEdges() const noexcept;
    std::size_t milestoneCount() const noexcept;
    std::size_t edgeCount() const noexcept;
    ob::Cost bestCost() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace terrain25d
