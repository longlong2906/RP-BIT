// SPDX-License-Identifier: GPL-3.0-only
// Controlled headless port of PUTN/PF-RRT*, see external/putn/PROVENANCE.json.
#pragma once
#include "terrain25d/Terrain.hpp"
#include <cstdint>
#include <functional>
#include <memory>

namespace terrain25d {
struct PutnPlane {
    bool available = false;
    Point support{};
    Point3 center_of_mass{}, normal{};
    double traversability = 0, flatness = 0, slope_deg = 0, sparsity = 0;
    std::size_t point_count = 0;
};
struct PutnStats {
    std::uint64_t iterations = 0, plane_fits = 0, model_rejections = 0,
        common_rejections = 0, edge_cost_calls = 0, rewires = 0,
        improvements = 0, nodes = 0;
    double conversion_seconds = 0, preprocessing_seconds = 0, solve_seconds = 0;
    bool tree_consistent = true;
    std::uint64_t informed_proposals = 0, informed_rejections = 0;
};
struct PutnSnapshot {
    double elapsed_seconds = 0, planner_objective = 0;
    std::vector<Point> path;
    std::vector<Point3> raw_com_path;
    std::vector<PutnPlane> native_planes; // paper-v2 snapshot evidence, not recomputed in callback
};
// Outcome of the optional convergence stop of the last solve.
struct PutnConvergence {
    bool converged = false;
    double reference_cost = 0;             // published cost at the last reset (valid when has_reference)
    std::uint64_t reference_iteration = 0; // iterations completed at that reset
    bool has_reference = false;
};
class PutnStatic {
public:
    // The terrain must outlive the planner. Source map1.yaml parameters are fixed.
    explicit PutnStatic(const Terrain &terrain, const std::vector<Rect> &obstacles = {});
    // Opt-in paper-first policy with a pre-generated canonical master cloud. The slope term of tau is
    // slope_deg / tau_slope_cap_deg: 10 reproduces the source weight 0.1 bit for bit (declared A4
    // adaptation: 35, the protocol slope cap). Other source parameters stay fixed.
    PutnStatic(const Terrain &terrain, const std::vector<Point3> &master_cloud, bool paper_v2,
               double tau_slope_cap_deg = 10.);
    ~PutnStatic();
    PutnStatic(const PutnStatic &) = delete;
    PutnStatic &operator=(const PutnStatic &) = delete;
    PutnPlane fitPlane(Point q);
    bool nativeMotion(const PutnPlane &a, const PutnPlane &b) const;
    double nativeCost(const PutnPlane &a, const PutnPlane &b) const;
    // 1: exact path published; 0: no solution within budget; -2: invalid endpoints.
    // fixed_work > 0 disables wall-clock termination, exclusively for replay tests.
    int solve(Point start, Point goal, std::uint64_t seed, double budget_seconds,
              bool own_cost, std::uint64_t fixed_work,
              const std::function<void(const PutnSnapshot &)> &callback);
    const PutnStats &stats() const;
    const PutnSnapshot &lastSnapshot() const;
    // Optional stop on convergence, the SEM-RRT* rule (A4 rule per iteration); off when the window is 0
    // (default). At each iteration boundary after the first solution: if the published cost is below
    // reference * (1 - relative_eps), it becomes the reference; otherwise, once `window_iterations`
    // iterations have passed since the last reset, the search stops. It reads only the published cost,
    // so a stopped search is an exact prefix of the same search without the stop.
    void setConvergenceStop(double relative_eps, std::uint64_t window_iterations);
    const PutnConvergence &convergence() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
