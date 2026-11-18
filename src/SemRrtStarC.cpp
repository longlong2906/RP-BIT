#include "terrain25d/SemRrtStarC.h"
#include "terrain25d/SemRrtStar.hpp"
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;
struct Context {
    terrain25d::Terrain terrain;
    std::unique_ptr<terrain25d::SemRrtStar> planner;
    explicit Context(const char *path) : terrain(terrain25d::Terrain::load(path)) { terrain.enableCollisionIndex(); }
};
Context &ctx(void *p) {
    if (!p) throw std::invalid_argument("null SEM planner handle");
    return *static_cast<Context *>(p);
}
template <class F> int guarded(F f) {
    last_error.clear();
    try { return f(); }
    catch (const std::exception &e) { last_error = e.what(); return -1; }
    catch (...) { last_error = "unknown C++ exception"; return -1; }
}
template <class F> double guardedValue(F f) {
    last_error.clear();
    try { return f(); }
    catch (const std::exception &e) { last_error = e.what(); return std::numeric_limits<double>::quiet_NaN(); }
}
}  // namespace

extern "C" {
const char *sem_error(void) { return last_error.c_str(); }
const char *sem_version(void) { return "sem_rrtstar_cpp_port_v1"; }

void *sem_create(const char *path, const double *bounds, int nx, int ny, double resolution,
                 const unsigned char *valid, const double *risk) {
    last_error.clear();
    try {
        if (!path || !bounds || !valid || !risk || nx <= 0 || ny <= 0) throw std::invalid_argument("invalid SEM create arguments");
        terrain25d::SemRaster raster;
        raster.xmin = bounds[0]; raster.ymin = bounds[1]; raster.xmax = bounds[2]; raster.ymax = bounds[3];
        raster.resolution = resolution;
        raster.nx = nx;
        raster.ny = ny;
        std::size_t size = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
        raster.valid.assign(valid, valid + size);
        raster.risk.assign(risk, risk + size);
        for (unsigned char v : raster.valid)
            if (v > 1) throw std::invalid_argument("SEM valid mask must hold 0/1 bytes");
        auto c = std::make_unique<Context>(path);
        c->planner = std::make_unique<terrain25d::SemRrtStar>(c->terrain, std::move(raster));
        return c.release();
    } catch (const std::exception &e) { last_error = e.what(); return nullptr; }
    catch (...) { last_error = "unknown C++ exception"; return nullptr; }
}
void sem_destroy(void *p) { delete static_cast<Context *>(p); }

int sem_solve(void *p, double sx, double sy, double gx, double gy, uint64_t seed, double budget, uint64_t work,
              double already, sem_callback cb, void *user) {
    return guarded([&] {
        auto status = ctx(p).planner->solve({sx, sy}, {gx, gy}, seed, budget, work, already,
            [&](const terrain25d::SemIncumbent &s) {
                if (!cb) return true;
                std::vector<double> xy;
                xy.reserve(2 * s.path.size());
                for (auto q : s.path) { xy.push_back(q.x); xy.push_back(q.y); }
                return cb(s.elapsed_seconds, xy.data(), s.heights.data(), s.path.size(), s.cost, s.totals.data(), user) == 0;
            });
        return static_cast<int>(status);
    });
}
int sem_counters(void *p, uint64_t *out) {
    return guarded([&] {
        if (!out) throw std::invalid_argument("null output");
        auto &planner = *ctx(p).planner;
        const auto &c = planner.counters();
        const uint64_t values[14] = {c.iterations, c.sample_proposals, c.sampler_failures, c.tree_insertions,
            c.edge_evaluations, c.hard_rejections, c.native_rejections, c.parent_candidates, c.rewires,
            c.descendant_cost_updates, c.start_connection_checks, c.incumbents, c.late_candidates, planner.nodeCount()};
        for (int i = 0; i < 14; ++i) out[i] = values[i];
        return 1;
    });
}
int sem_set_exhaustive_check(void *p, int enabled) {
    return guarded([&] { ctx(p).planner->setExhaustiveCheck(enabled != 0); return 1; });
}
int sem_set_convergence_stop(void *p, double relative_eps, uint64_t window_iterations) {
    return guarded([&] { ctx(p).planner->setConvergenceStop(relative_eps, window_iterations); return 1; });
}
int sem_convergence(void *p, double *out) {
    return guarded([&] {
        if (!out) throw std::invalid_argument("null output");
        const auto &c = ctx(p).planner->convergence();
        out[0] = c.has_reference ? c.reference_cost : std::numeric_limits<double>::quiet_NaN();
        out[1] = static_cast<double>(c.reference_iteration);
        return c.converged ? 1 : 0;
    });
}
int sem_model_edge(void *p, double ax, double ay, double bx, double by, double *out) {
    return guarded([&] {
        if (!out) throw std::invalid_argument("null output");
        return ctx(p).planner->modelEdge({ax, ay}, {bx, by}, out[0], out[1]) ? 1 : 0;
    });
}
int sem_model_valid_state(void *p, double x, double y) {
    return guarded([&] { return ctx(p).planner->modelValidState({x, y}) ? 1 : 0; });
}
int sem_debug_random(uint64_t seed, size_t count, double *out) {
    return guarded([&] {
        if (!out && count) throw std::invalid_argument("null output");
        terrain25d::NumpyPcg64 rng(seed);
        for (size_t i = 0; i < count; ++i) out[i] = rng.random();
        return 1;
    });
}
int sem_debug_uniform(uint64_t seed, double low, double high, size_t count, double *out) {
    return guarded([&] {
        if (!out && count) throw std::invalid_argument("null output");
        terrain25d::NumpyPcg64 rng(seed);
        for (size_t i = 0; i < count; ++i) out[i] = rng.uniform(low, high);
        return 1;
    });
}
double sem_debug_hypot(double x, double y) { return terrain25d::pyHypot(x, y); }
double sem_debug_dist(double ax, double ay, double bx, double by) { return terrain25d::pyDist({ax, ay}, {bx, by}); }
double sem_debug_fsum(const double *values, size_t count) {
    return guardedValue([&] { return terrain25d::pyFsum(values, count); });
}
double sem_debug_turn(double ax, double ay, double bx, double by, double cx, double cy) {
    return terrain25d::semTurnCost({ax, ay}, {bx, by}, {cx, cy});
}
int sem_debug_libm(double x, double y, double z, double *out) {
    return guarded([&] {
        if (!out) throw std::invalid_argument("null output");
        terrain25d::semLibm(x, y, z, out);
        return 1;
    });
}
}
