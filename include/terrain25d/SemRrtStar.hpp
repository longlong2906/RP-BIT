#pragma once
// C++ port of the project's static SEM-RRT* reconstruction (tools/sem_native_planner.py).
// Not an official implementation of the SEM-RRT* paper. Given the same native raster, seed and
// work budget it reproduces the Python search bit for bit: numpy's PCG64/SeedSequence stream,
// CPython's math.dist/hypot/fsum, the same operation order and the same tie-breaking. The native
// raster (Python/scipy preprocessing) is passed in; hard geometry is the shared terrain25d check.
#include "terrain25d/Terrain.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace terrain25d {

__extension__ typedef unsigned __int128 Uint128;  // GCC/Clang; PCG64 state

// np.random.default_rng(seed): PCG64 seeded through SeedSequence(seed).
class NumpyPcg64 {
public:
    explicit NumpyPcg64(std::uint64_t seed);
    std::uint64_t next64();
    double random();                       // Generator.random()
    double uniform(double low, double high); // Generator.uniform(low, high), scalar call
private:
    Uint128 state_ = 0, inc_ = 0;
    void step();
};

// CPython 3.12 math semantics (vector_norm and msum), finite inputs.
double pyHypot(double x, double y);
double pyDist(Point a, Point b);
double pyFsum(const double *values, std::size_t count);
// sem_native_planner._turn: max(0, acos(clamped cosine) - pi/6), zero for a repeated vertex.
double semTurnCost(Point a, Point b, Point c);
// cos(x), sin(x), atan2(y, x), acos(z) from the C library CPython's math module uses.
void semLibm(double x, double y, double z, double out[4]);

struct SemRaster {
    double xmin = 0, ymin = 0, xmax = 0, ymax = 0, resolution = .1;
    int nx = 0, ny = 0;
    std::vector<unsigned char> valid;  // row-major [y][x]
    std::vector<double> risk;          // row-major [y][x], NaN where invalid
};

struct SemCounters {
    std::uint64_t iterations = 0, sample_proposals = 0, sampler_failures = 0, tree_insertions = 0,
        edge_evaluations = 0, hard_rejections = 0, native_rejections = 0, parent_candidates = 0,
        rewires = 0, descendant_cost_updates = 0, start_connection_checks = 0, incumbents = 0,
        late_candidates = 0;
};

struct SemIncumbent {
    double elapsed_seconds = 0, cost = 0;
    std::array<double, 3> totals{};       // length_grid, turn_excess_rad, risk_integral_grid
    std::vector<Point> path;              // start ... goal
    std::vector<double> heights;          // terrain height of every path vertex
};

enum class SemStatus { ExactSolution = 0, Timeout = 1, InvalidHardEndpoints = 2, InvalidNativeEndpoints = 3, Aborted = 4 };

// Outcome of the optional convergence stop of the last solve.
struct SemConvergence {
    bool converged = false;
    double reference_cost = 0;            // published cost at the last reset (valid when has_reference)
    std::uint64_t reference_iteration = 0; // iterations completed at that reset
    bool has_reference = false;
};

class SemRrtStar {
public:
    static constexpr double kStep = 1., kNeighbor = 3., kStartConnection = 3., kRiskThreshold = 40.;
    static constexpr double kWeights[3] = {.48, .16, .36};

    SemRrtStar(const Terrain &terrain, SemRaster raster);
    // Returns false from the callback to abort. `already_elapsed` is charged to the budget
    // (planner construction under the pri_portfolio_v1 clock policy).
    SemStatus solve(Point start, Point goal, std::uint64_t seed, double budget_seconds,
                    std::uint64_t work_iterations, double already_elapsed,
                    const std::function<bool(const SemIncumbent &)> &callback);
    const SemCounters &counters() const { return counters_; }
    std::size_t nodeCount() const { return nodes_.size(); }
    // Model queries without counters, for equivalence tests.
    bool modelEdge(Point a, Point b, double &length, double &risk) const;
    bool modelValidState(Point q) const;
    // Test only: cross-check every grid query against an exhaustive scan.
    void setExhaustiveCheck(bool enabled) { exhaustive_check_ = enabled; }
    // Optional stop on convergence, the A4 rule per iteration instead of per batch; off when the window
    // is 0 (default). At each iteration boundary after the first solution: if the published cost is below
    // reference * (1 - relative_eps), it becomes the reference; otherwise, once `window_iterations`
    // iterations have passed since the last reset, the search stops. It reads only the published cost,
    // so a stopped search is an exact prefix of the same search without the stop.
    void setConvergenceStop(double relative_eps, std::uint64_t window_iterations);
    const SemConvergence &convergence() const { return convergence_; }

private:
    struct Node {
        Point point;
        long parent = -1;
        std::vector<long> children;
        std::array<double, 2> edge_base{};
        std::array<double, 3> local{}, totals{};
        double cost = 0;
    };
    struct Candidate { double cost; std::array<double, 3> local, totals; };

    const Terrain &terrain_;
    SemRaster raster_;
    std::vector<Node> nodes_;
    SemCounters counters_;
    double best_cost_ = 0;
    bool has_best_ = false, exhaustive_check_ = false;
    double stop_eps_ = 0;
    std::uint64_t stop_window_ = 0;
    SemConvergence convergence_;
    // Uniform grid over the bounds; each cell lists node indices in insertion (ascending) order.
    double cell_ = kNeighbor;
    int gx_ = 1, gy_ = 1;
    std::vector<std::vector<long>> grid_;
    mutable std::vector<double> scratch_;  // Bresenham cell risks of one edge

    bool index(Point q, int &ix, int &iy) const;
    bool edge(Point a, Point b, std::array<double, 2> &base);  // counted, hard + native
    Candidate candidate(Point q, long parent, const std::array<double, 2> &base) const;
    bool isAncestor(long ancestor, long child) const;
    long insert(Point q, long parent, const std::array<double, 2> &base);
    void reparent(long index, long parent, const std::array<double, 2> &base);
    bool sample(NumpyPcg64 &rng, Point start, Point goal, Point &q);
    void gridCell(Point q, int &cx, int &cy) const;
    void gridInsert(long index);
    long nearest(Point q, double &distance) const;
    std::vector<long> within(Point q, double radius) const;
};

}  // namespace terrain25d
