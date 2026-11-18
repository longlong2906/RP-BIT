#pragma once
#include "terrain25d/OmplIntegration.hpp"
#include <functional>
#include <optional>

namespace terrain25d {
struct PriPathStats {
    bool valid=false;
    std::string reason;
    double length=0.,length_xy=0.,pri=0.,risk_sum=0.;
    std::vector<std::size_t> cells;
};
PriPathStats priPathStats(const Terrain&,const SafetyPolicy&,const std::vector<Point>&);
struct PriMask {
    unsigned episode_id=0;
    double threshold=0.;
    std::string recipe,status,sha256;
    std::vector<int> allowed;
    std::vector<std::size_t> excluded;
};
std::string priMaskSha256(const std::vector<int>&);
std::vector<PriMask> priPortfolioMasks(const SafetyField&,const PriPathStats&,Point,Point,
                                      const std::function<bool()> &cancel={});
struct PriPortfolioOptions {
    double length_cap_ratio=1.,budget_s=20.;
    // Explicit test-only work cap. Wall deadline remains enforced.
    unsigned work_iterations_per_episode=0;
};
struct PriCandidate {
    std::size_t id=0;
    unsigned episode_id=0,callback_id=0;
    double callback_s=0.,score_completed_s=0.;
    bool eligible=false,published=false,anchor_phase=false;
    std::string reason;
    PriPathStats stats;
    std::vector<Point> path;
};
struct PriEpisode {
    unsigned episode_id=0,planner_instance_id=0,iterations=0,batch_transitions=0;
    bool fresh_planner=false,solve_called=false;
    double began_s=0.,setup_completed_s=0.,finished_s=0.;
    std::string mask_sha256,status;
};
struct PriPortfolioResult {
    std::string status="NO_SOLUTION",stage="NO_ANCHOR";
    std::optional<std::size_t> anchor,selected;
    std::optional<double> length_cap;
    double actual_wall_s=0.,mask_build_s=0.;
    PolicyCounters counters;
    std::vector<PriCandidate> candidates;
    std::vector<PriMask> masks;
    std::vector<PriEpisode> episodes;
};
bool priCandidateWithinBudget(double length,double cap,double finished,double deadline);
bool priCandidateBetter(const PriCandidate&,const PriCandidate&);
// Pure selection transition, also exercised with synthetic clock values.
void considerPriCandidate(PriPortfolioResult&,PriCandidate&,std::optional<std::size_t>&anchorBest,
                          double anchorDeadline,double globalDeadline,bool fixedWork=false);
PriPortfolioResult runPriPortfolio(const Terrain&,std::shared_ptr<const SafetyField>,Point,Point,
                                  const PriPortfolioOptions&);
} // namespace terrain25d
