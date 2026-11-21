#pragma once
#include "terrain25d/RpBit.hpp"

// A5: A4 / RP-BIT* made faster without touching A4's objective, bounds, sampler,
// safety policy or refinement. Every mechanism has its own switch:
//   C1 batch_growth   the convergence window counts samples and uses few large batches;
//   C2 fast_validity  BIT*'s validity checks skip the exact swept-slope computation
//                     where a conservative slope envelope already certifies it;
//   C3 lazy_refine    an incumbent is refined when its raw cost has dropped by eps since
//                     the last refined one, and always before the window may close;
//   C4 seed           a lattice path from the risk potential is refined and published
//                     first; BIT* then samples only where a better path can pass.
// With every switch off, batch_min = 200 and window_samples = 200 * M, a run is the A4 run.
namespace terrain25d {

inline constexpr std::uint64_t RP_A5_SEED_EXPANSIONS=60000;   // C4: node expansions allowed for the seed search

struct RpA5Options {
    RpBitOptions planner;
    bool refine=true;
    RpRefineOptions refinement;
    double budget_s=20.,stop_relative_eps=1e-3;
    unsigned window_samples=1000;   // stale informed samples before the stop; 0 disables it
    unsigned batch_min=200;         // batch size after the first solution (and the smallest later one)
    bool batch_growth=false,fast_validity=false,lazy_refine=false,seed=false;
    unsigned test_max_iterations=0; // fixed-work limit for reproducibility tests
    std::uint64_t planner_seed=11;
};

// Upper bound of the triangle gradient magnitude around each terrain cell, built from
// Terrain::height at the grid nodes. certifies() is true only if every triangle within
// the footprint radius of the segment has a slope strictly below the cap by a margin,
// so SafetyPolicy's exact swept-slope test would pass; false means "not proven".
class RpA5SlopeEnvelope {
public:
    explicit RpA5SlopeEnvelope(const Terrain &terrain);
    bool certifies(Point a,Point b,double slopeCapDeg) const;
    int reach() const { return reach_; }   // window half-width in cells
private:
    int nx_=0,ny_=0,reach_=0;
    double x0_=0.,y0_=0.,dx_=0.,dy_=0.,step_=0.;
    mutable double cap_=-1.,limit_=0.;   // gradient limit of the last cap asked for
    std::vector<double> envelope_;       // per cell; +infinity where data is missing nearby
};

struct RpA5Counters {
    std::uint64_t slope_certified=0,slope_exact=0;   // C2: BIT* checks settled by the envelope / by the exact computation
    std::uint64_t refine_deferred=0;                 // C3: batch boundaries that left a new incumbent unrefined
    std::uint64_t bounded_samples=0;                 // C4: random samples drawn under the published-J bound
    std::uint64_t seed_expansions=0;                 // C4: nodes expanded by the seed search
};
// The reason SafetyPolicy::rejectionReason(a,b) returns, with the same test order
// (SEM cells, footprint geometry, swept slope); only the slope computation may be skipped.
std::string rpA5RejectionReason(const SafetyPolicy &policy,const Terrain &terrain,const RpA5SlopeEnvelope &envelope,
                                Point a,Point b,RpA5Counters &counters);

// The planner SEM field of a request: the text exchange read by SafetyField::load, or the same cells in
// the binary container SAFE_SEM_FIELD_B1 of tools/rp_bit_a5_field.py (little-endian hosts). Either way
// SafetyField's constructor validates every cell.
SafetyField rpA5LoadField(const std::string &path);

// Next batch's random-sample count: batch_min, then as many as are already stale or
// drawn (doubling), never beyond the window.
unsigned rpA5NextBatch(std::uint64_t stale,unsigned drawn,const RpA5Options &options);

// C4: path over the centres of open SEM cells from start to goal, guided by the goal
// potential. Empty when there is none within the expansion limit. Not validated here.
std::vector<Point> rpA5SeedPath(const SafetyPolicy &policy,const RpObjective &objective,const Terrain &terrain,
                                Point start,Point goal,std::uint64_t maxExpansions,std::uint64_t &expansions);

struct RpA5Batch {   // state at the moment BIT* has just opened batch `batch`
    unsigned batch=0,random_samples=0,iterations=0;
    double elapsed_s=0.,bitstar_cost=0.,refine_s=0.,published=0.;
    std::uint64_t stale_samples=0;
};
struct RpA5RunResult {
    RpRunResult run;   // same fields and meaning as an A4 run
    double policy_s=0.,problem_s=0.,envelope_s=0.,seed_s=0.;
    std::string seed_status="disabled";
    std::vector<RpA5Batch> batches;
    RpA5Counters a5;
};
// The clock starts before the safety policy, potentials and planner are created, as in A4.
RpA5RunResult runRpBitA5(const Terrain &map,std::shared_ptr<const SafetyField> field,Point start,Point goal,
                         const RpA5Options &options);
}
