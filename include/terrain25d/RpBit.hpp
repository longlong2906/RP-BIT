#pragma once
#include "terrain25d/OmplIntegration.hpp"
#include <functional>

// A4 / RP-BIT*: real OMPL BIT* minimising J = L_H + lambda * E / 40 with a
// risk-potential lower bound, a risk-aware informed sampler, a fast edge
// integral and a declared J-aware path refinement. Existing variants are untouched.
namespace terrain25d {

struct RpEdgeCost {
    bool defined=false;    // terrain data exists along the whole connector, inside the field
    bool supported=false;  // every positive-length piece lies in SEM-supported cells
    double length=0.,exposure=0.;  // L_H [m], integral of F_R ds_H [risk*m]
};
// Double-precision single-pass counterpart of surfaceCost + semExposureCanonical.
// Cells come from exact comparisons with the canonical raster lines and index
// stepping, never from re-classified midpoints, so split edges stay additive.
class RpEdgeIntegrator {
public:
    RpEdgeIntegrator(const Terrain &terrain,const SafetyField &field);
    RpEdgeCost evaluate(Point a,Point b) const;
    const SafetyField &field() const { return field_; }
private:
    const Terrain &terrain_;
    const SafetyField &field_;
    std::vector<double> xs_,ys_;
};

// Lower bound of J between a fixed source and any point: kappa-scaled 8-connected
// Dijkstra on SEM cell corners, extended inside each open cell from its four sides
// by a w_c-Lipschitz function (continuous across cells): the largest such extension
// everywhere except in the source's own cells, where the smallest one is used.
// Any continuous, cell-wise w_c-Lipschitz function P satisfies |P(x)-P(y)| <= J(x,y).
class RiskPotential {
public:
    RiskPotential(const SafetyField &field,double riskCap,double lambda,Point source);
    // max(0, P(q)-P(source)) with numerical slack; +infinity in blocked or
    // relaxed-unreachable cells; never NaN.
    double at(Point q) const;
    // Not larger than the infimum of at() over the closed cell; +infinity if unusable.
    double cellLowerBound(int i,int j) const;
    int nx() const { return nx_; }
    int ny() const { return ny_; }
private:
    double raw(Point q,int i,int j) const;
    bool sourceCell(int i,int j) const { return i>=si0_&&i<=si1_&&j>=sj0_&&j<=sj1_; }
    int nx_,ny_,si0_=0,si1_=0,sj0_=0,sj1_=0;
    double x0_,y0_,res_,offset_=0.;
    std::vector<double> weight_,u_;
};

// O(1) minimum open-cell risk over a square superset of an edge's bounding box.
class RiskBoxMin {
public:
    RiskBoxMin(const SafetyField &field,double riskCap);
    double minRisk(Point a,Point b) const;
private:
    int nx_,ny_;
    double x0_,y0_,res_,global_;
    std::vector<std::vector<double>> levels_;
};

struct RpBitOptions {
    double lambda=30.;
    unsigned samples_per_batch=100,batch_after_first_solution=200,sample_attempts=100;
    double rewire_factor=1.1;
    bool potential=true;   // false: d_Phi bounds and the legacy atlas sampler (ablation control)
};
struct RpCounters {
    std::uint64_t fast_integral_calls=0,potential_evaluations=0,box_bound_calls=0,
        sampler_cell_draws=0,sampler_rejected=0,sampler_empty_prefix_calls=0,
        heuristic_violations=0,refine_edge_evaluations=0,refine_validity_checks=0;
};

class RpObjective final : public ob::OptimizationObjective {
public:
    RpObjective(const ob::SpaceInformationPtr &si,const Terrain &map,std::shared_ptr<const SafetyPolicy> safety,
                Point start,Point goal,const RpBitOptions &options,PolicyCounters &policy,RpCounters &counters);
    ob::Cost stateCost(const ob::State*) const override { return identityCost(); }
    ob::Cost motionCost(const ob::State *a,const ob::State *b) const override;
    ob::Cost motionCostHeuristic(const ob::State *a,const ob::State *b) const override;
    ob::InformedSamplerPtr allocInformedStateSampler(const ob::ProblemDefinitionPtr &p,unsigned int attempts) const override;
    // g/h: admissible bounds of a path start->q and q->goal (not rounded down);
    // ps/pg: the two potentials alone (zero when the potential is disabled).
    struct Bounds { bool has_data=false; double z=0.,ps=0.,pg=0.,g=0.,h=0.; };
    Bounds bounds(Point q) const;
    double lowerBoundStartGoal() const { return bounds(start_).h; }
    const RpEdgeIntegrator &integrator() const { return integrator_; }
    const SafetyField &field() const { return integrator_.field(); }
    const RiskPotential *potentialFromStart() const { return from_start_.get(); }
    const RiskPotential *potentialToGoal() const { return to_goal_.get(); }
    double lambda() const { return options_.lambda; }
private:
    const Bounds &memo(const ob::State *state) const;
    const Terrain &map_;
    std::shared_ptr<const SafetyPolicy> safety_;
    Point start_,goal_;
    RpBitOptions options_;
    PolicyCounters &policy_;
    RpCounters &counters_;
    RpEdgeIntegrator integrator_;
    std::unique_ptr<RiskPotential> from_start_,to_goal_;
    std::unique_ptr<RiskBoxMin> box_;
    double start_z_=0.,goal_z_=0.;
    std::uint64_t token_=0;
};

struct RpProblem {
    ob::StateSpacePtr space;
    ob::SpaceInformationPtr si;
    ob::ProblemDefinitionPtr pdef;
    std::shared_ptr<RpObjective> objective;
    std::shared_ptr<ompl::geometric::BITstar> planner;
};
RpProblem makeRpProblem(const Terrain &map,std::shared_ptr<const SafetyPolicy> safety,Point start,Point goal,
                        const RpBitOptions &options,PolicyCounters &policy,RpCounters &counters);

struct RpPathCost {
    bool valid=false;
    double objective=0.,length=0.,length_xy=0.,exposure=0.;
    unsigned sharp_turns=0;
};
// Fast-integral decomposition of a polyline; valid only if every edge is defined and SEM-supported.
RpPathCost rpPathCost(const RpEdgeIntegrator &integrator,double lambda,const std::vector<Point> &path);
// Heading changes above 30 degrees after removing consecutive duplicate points.
unsigned rpSharpTurns(const std::vector<Point> &path);

struct RpRefineOptions {
    bool shortcut=true,relocate=true,round_turns=true;
    double max_edge_m=1.5,step_m=.4,min_step_m=.05,turn_limit_rad=.5235987755982988,
        min_chord_m=.3,round_cost_ratio=.005;
    unsigned max_sweeps=3;
    std::uint64_t max_edge_evaluations=20000;
};
struct RpRefineResult {
    std::vector<Point> path;
    RpPathCost cost;
    std::uint64_t edge_evaluations=0,validity_checks=0,shortcuts=0,moves=0,rounded=0;
    unsigned kept_sharp=0;
};
// Deterministic, declared post-processing of one incumbent. Shortcuts and vertex
// moves are accepted only if J does not increase and the new edges pass the full
// safety policy; rounding all corners may raise J by at most round_cost_ratio in total.
RpRefineResult rpRefine(const SafetyPolicy &policy,const RpEdgeIntegrator &integrator,double lambda,
                        std::vector<Point> path,const RpRefineOptions &options);

struct RpRunOptions {
    RpBitOptions planner;
    bool refine=true;
    RpRefineOptions refinement;
    double budget_s=20.,stop_relative_eps=1e-3;
    unsigned stop_batches=5;         // 0 disables the convergence stop
    unsigned test_max_iterations=0;  // fixed-work limit for reproducibility tests
    std::uint64_t planner_seed=11;
};
struct RpPublication {
    std::size_t id=0;
    std::string origin;
    double elapsed_s=0.,bitstar_cost=0.;
    unsigned batch=0;
    std::vector<Point> path;
    RpPathCost cost;
};
struct RpRunResult {
    std::string status="NO_SOLUTION",termination="UNKNOWN_NATIVE_PLANNER_STOP";
    double actual_wall_s=0.,setup_s=0.,refine_s=0.,lower_bound=0.;
    unsigned iterations=0,batches=0,batch_transitions=0,raw_incumbents=0,refinements=0;
    std::vector<RpPublication> publications;
    PolicyCounters counters;
    RpCounters rp;
};
// The clock starts before the safety policy, potentials and planner are created.
RpRunResult runRpBit(const Terrain &map,std::shared_ptr<const SafetyField> field,Point start,Point goal,
                     const RpRunOptions &options);
}
