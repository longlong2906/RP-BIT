#pragma once

#include <ompl/base/Planner.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace terrain25d {

namespace ob = ompl::base;

/**
 * Controlled SEM-RRT* comparator used by this project.
 *
 * This is deliberately not labelled as the official SEM-RRT* implementation.
 * It keeps only the common planning core required by the registered comparison:
 * a goal-rooted RRT* tree in XY, the problem's surface-length objective and
 * validity checker, and Euclidean informed sampling after the first incumbent.
 * Risk, turn-angle and rolling constraints are intentionally absent.
 */
class SemRRTstarCommon final : public ob::Planner {
public:
    struct NodeAudit {
        std::size_t insertionId;
        std::size_t parentId;
        bool root;
        double costToGoal;
        double edgeCostToParent;
    };

    static constexpr double STEER_RANGE_M = 1.0;
    static constexpr double NEAR_RADIUS_M = 3.0;
    static constexpr double START_CONNECTION_RADIUS_M = 3.0;
    static constexpr double GOAL_BIAS = 0.0;
    static constexpr unsigned int INFORMED_ATTEMPTS = 100u;

    explicit SemRRTstarCommon(const ob::SpaceInformationPtr &si);
    ~SemRRTstarCommon() override;

    void setup() override;
    void clear() override;
    ob::PlannerStatus solve(const ob::PlannerTerminationCondition &ptc) override;
    void getPlannerData(ob::PlannerData &data) const override;

    std::uint64_t numIterations() const noexcept;
    std::uint64_t numSampleCalls() const noexcept;
    std::uint64_t numTreeInsertions() const noexcept;
    std::uint64_t numStateRejections() const noexcept;
    std::uint64_t numMotionChecks() const noexcept;
    std::uint64_t numRewires() const noexcept;
    std::uint64_t numDescendantCostUpdates() const noexcept;
    std::uint64_t numStartConnectionChecks() const noexcept;
    std::uint64_t numStartConnectionsValid() const noexcept;
    std::uint64_t numSamplerFailures() const noexcept;
    std::size_t vertexCount() const noexcept;
    ob::Cost bestCost() const noexcept;
    std::vector<NodeAudit> auditNodes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace terrain25d
