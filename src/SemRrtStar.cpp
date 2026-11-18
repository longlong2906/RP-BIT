#include "terrain25d/SemRrtStar.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef near
#undef far
#endif

namespace terrain25d {
namespace {
// Transcendentals from the C library CPython's math module calls: the Universal CRT on Windows
// (MinGW would otherwise link its own cos/sin/atan2, which differ in the last bit), the system
// libm elsewhere. Calls go through pointers so GCC cannot fold them into builtins or sincos().
struct Libm {
    double (*cos)(double), (*sin)(double), (*acos)(double), (*atan2)(double, double);
};
const Libm &libm() {
    static const Libm table = [] {
#if defined(_WIN32)
        HMODULE ucrt = GetModuleHandleA("ucrtbase.dll");
        if (!ucrt) ucrt = LoadLibraryA("ucrtbase.dll");
        if (!ucrt) throw std::runtime_error("ucrtbase.dll unavailable for CPython-compatible libm");
        auto unary = [&](const char *name) {
            auto f = reinterpret_cast<double (*)(double)>(reinterpret_cast<void (*)()>(GetProcAddress(ucrt, name)));
            if (!f) throw std::runtime_error(std::string("ucrtbase.dll lacks ") + name);
            return f;
        };
        auto atan2 = reinterpret_cast<double (*)(double, double)>(reinterpret_cast<void (*)()>(GetProcAddress(ucrt, "atan2")));
        if (!atan2) throw std::runtime_error("ucrtbase.dll lacks atan2");
        return Libm{unary("cos"), unary("sin"), unary("acos"), atan2};
#else
        double (*volatile c)(double) = ::cos, (*volatile s)(double) = ::sin, (*volatile a)(double) = ::acos;
        double (*volatile t)(double, double) = ::atan2;
        return Libm{c, s, a, t};
#endif
    }();
    return table;
}
// ---- numpy SeedSequence / PCG64 constants ----
constexpr std::uint32_t kInitA = 0x43b0d7e5u, kMultA = 0x931e8875u, kInitB = 0x8b51f9ddu, kMultB = 0x58f38dedu,
    kMixMultL = 0xca01f9ddu, kMixMultR = 0x4973f715u;
constexpr int kXShift = 16;
const Uint128 kPcgMultiplier = (static_cast<Uint128>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;

std::uint32_t hashmix(std::uint32_t value, std::uint32_t &hash_const) {
    value ^= hash_const;
    hash_const *= kMultA;
    value *= hash_const;
    value ^= value >> kXShift;
    return value;
}
std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t result = kMixMultL * x - kMixMultR * y;
    result ^= result >> kXShift;
    return result;
}

// ---- CPython 3.12 vector_norm (math.hypot / math.dist) ----
struct DoubleLength { double hi, lo; };
DoubleLength dlFastSum(double a, double b) { double x = a + b; return {x, (a - x) + b}; }
void veltkamp(double x, double &hi, double &lo) { double t = x * 134217729.0; hi = t - (t - x); lo = x - hi; }
DoubleLength dlMul(double x, double y) {
    // fl(x*y) and its exact rounding error: the pair CPython obtains with fma(x, y, -z).
    double z = x * y, xh, xl, yh, yl;
    veltkamp(x, xh, xl);
    veltkamp(y, yh, yl);
    return {z, ((xh * yh - z) + xh * yl + xl * yh) + xl * yl};
}
double vectorNorm(double *vec, int n, double max) {
    if (std::isinf(max)) return max;
    if (max == 0.0 || n <= 1) return max;
    int max_e = 0;
    std::frexp(max, &max_e);
    if (max_e < -1023) {
        for (int i = 0; i < n; ++i) vec[i] /= DBL_MIN;
        return DBL_MIN * vectorNorm(vec, n, max / DBL_MIN);
    }
    double scale = std::ldexp(1.0, -max_e), csum = 1.0, frac1 = 0.0, frac2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = vec[i] * scale;
        DoubleLength pr = dlMul(x, x), sm = dlFastSum(csum, pr.hi);
        csum = sm.hi;
        frac1 += pr.lo;
        frac2 += sm.lo;
    }
    double h = std::sqrt(csum - 1.0 + (frac1 + frac2));
    DoubleLength pr = dlMul(-h, h), sm = dlFastSum(csum, pr.hi);
    csum = sm.hi;
    frac1 += pr.lo;
    frac2 += sm.lo;
    double x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    return h / scale;
}

constexpr double kPi = 3.141592653589793;  // math.pi
constexpr double kTurnThreshold = kPi / 6;
using Clock = std::chrono::steady_clock;
bool same(Point a, Point b) { return a.x == b.x && a.y == b.y; }
}  // namespace

// ---- NumpyPcg64 ----
NumpyPcg64::NumpyPcg64(std::uint64_t seed) {
    std::vector<std::uint32_t> entropy;  // SeedSequence: little-endian 32-bit words, 0 -> [0]
    if (seed == 0) entropy.push_back(0);
    for (std::uint64_t n = seed; n; n >>= 32) entropy.push_back(static_cast<std::uint32_t>(n & 0xffffffffu));
    std::array<std::uint32_t, 4> pool{};
    std::uint32_t hash_const = kInitA;
    for (std::size_t i = 0; i < pool.size(); ++i) pool[i] = hashmix(i < entropy.size() ? entropy[i] : 0u, hash_const);
    for (std::size_t src = 0; src < pool.size(); ++src)
        for (std::size_t dst = 0; dst < pool.size(); ++dst)
            if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], hash_const));
    for (std::size_t src = pool.size(); src < entropy.size(); ++src)
        for (std::size_t dst = 0; dst < pool.size(); ++dst) pool[dst] = mix(pool[dst], hashmix(entropy[src], hash_const));
    // generate_state(4, uint64): eight words, paired little-endian into four uint64.
    std::uint32_t words[8];
    std::uint32_t hash_b = kInitB;
    for (int i = 0; i < 8; ++i) {
        std::uint32_t value = pool[static_cast<std::size_t>(i) % pool.size()];
        value ^= hash_b;
        hash_b *= kMultB;
        value *= hash_b;
        value ^= value >> kXShift;
        words[i] = value;
    }
    std::uint64_t s[4];
    for (int i = 0; i < 4; ++i) s[i] = static_cast<std::uint64_t>(words[2 * i]) | (static_cast<std::uint64_t>(words[2 * i + 1]) << 32);
    // pcg64_set_seed: seed[0] and inc[0] are the high halves; pcg_setseq_128_srandom_r.
    Uint128 initstate = (static_cast<Uint128>(s[0]) << 64) | s[1];
    Uint128 initseq = (static_cast<Uint128>(s[2]) << 64) | s[3];
    state_ = 0;
    inc_ = (initseq << 1u) | 1u;
    step();
    state_ += initstate;
    step();
}
void NumpyPcg64::step() { state_ = state_ * kPcgMultiplier + inc_; }
std::uint64_t NumpyPcg64::next64() {
    step();  // XSL-RR 128/64 output of the advanced state
    std::uint64_t value = static_cast<std::uint64_t>(state_ >> 64) ^ static_cast<std::uint64_t>(state_);
    unsigned rot = static_cast<unsigned>(state_ >> 122);
    return (value >> rot) | (value << ((-rot) & 63u));
}
double NumpyPcg64::random() { return static_cast<double>(next64() >> 11) * (1.0 / 9007199254740992.0); }
double NumpyPcg64::uniform(double low, double high) { double range = high - low; return low + range * random(); }

// ---- CPython math ----
double pyHypot(double x, double y) {
    double v[2] = {std::fabs(x), std::fabs(y)}, max = 0.0;
    for (double value : v) {
        if (std::isnan(value)) return std::numeric_limits<double>::quiet_NaN();
        if (value > max) max = value;
    }
    return vectorNorm(v, 2, max);
}
double pyDist(Point a, Point b) { return pyHypot(a.x - b.x, a.y - b.y); }
double pyFsum(const double *values, std::size_t count) {
    // CPython math_fsum (Shewchuk partials, half-even correction), finite inputs only.
    // Nonoverlapping binary64 partials never exceed ~40, so a fixed buffer suffices.
    double p[64];
    std::size_t size = 0;
    double x, y, t, hi, yr, lo = 0.0;
    for (std::size_t k = 0; k < count; ++k) {
        x = values[k];
        if (!std::isfinite(x)) throw std::invalid_argument("fsum: non-finite input");
        std::size_t i = 0;
        for (std::size_t j = 0; j < size; ++j) {
            y = p[j];
            if (std::fabs(x) < std::fabs(y)) { t = x; x = y; y = t; }
            hi = x + y;
            yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) p[i++] = lo;
            x = hi;
        }
        size = i;
        if (x != 0.0) {
            if (!std::isfinite(x)) throw std::overflow_error("fsum: intermediate overflow");
            if (size == 64) throw std::logic_error("fsum: partial buffer exhausted");
            p[size++] = x;
        }
    }
    hi = 0.0;
    std::size_t n = size;
    if (n > 0) {
        hi = p[--n];
        while (n > 0) {
            x = hi;
            y = p[--n];
            hi = x + y;
            yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
            y = lo * 2.0;
            x = hi + y;
            yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return hi;
}
void semLibm(double x, double y, double z, double out[4]) {
    const Libm &m = libm();
    out[0] = m.cos(x);
    out[1] = m.sin(x);
    out[2] = m.atan2(y, x);
    out[3] = m.acos(z);
}
double semTurnCost(Point a, Point b, Point c) {
    double ux = b.x - a.x, uy = b.y - a.y, vx = c.x - b.x, vy = c.y - b.y;
    double norm = pyHypot(ux, uy) * pyHypot(vx, vy);
    if (norm == 0) return 0.;
    double value = (ux * vx + uy * vy) / norm;
    value = value < 1. ? value : 1.;    // min(1., value)
    value = value > -1. ? value : -1.;  // max(-1., value)
    double turn = libm().acos(value) - kTurnThreshold;
    return turn > 0. ? turn : 0.;       // max(0., turn)
}

// ---- planner ----
SemRrtStar::SemRrtStar(const Terrain &terrain, SemRaster raster) : terrain_(terrain), raster_(std::move(raster)) {
    const auto &r = raster_;
    if (r.nx <= 0 || r.ny <= 0 || r.valid.size() != static_cast<std::size_t>(r.nx) * r.ny || r.risk.size() != r.valid.size())
        throw std::invalid_argument("invalid SEM raster shape");
    if (!std::isfinite(r.xmin) || !std::isfinite(r.ymin) || !std::isfinite(r.xmax) || !std::isfinite(r.ymax)
        || r.xmax <= r.xmin || r.ymax <= r.ymin || r.resolution != .1)
        throw std::invalid_argument("invalid SEM raster geometry");
    auto b = terrain_.bounds();
    if (std::fabs(b.xmin - r.xmin) > 1e-9 || std::fabs(b.ymin - r.ymin) > 1e-9 || std::fabs(b.xmax - r.xmax) > 1e-9
        || std::fabs(b.ymax - r.ymax) > 1e-9)
        throw std::invalid_argument("terrain and SEM bounds differ");
    gx_ = std::max(1, static_cast<int>(std::ceil((r.xmax - r.xmin) / cell_)));
    gy_ = std::max(1, static_cast<int>(std::ceil((r.ymax - r.ymin) / cell_)));
}

bool SemRrtStar::index(Point q, int &ix, int &iy) const {
    const auto &r = raster_;
    if (!(r.xmin <= q.x && q.x <= r.xmax) || !(r.ymin <= q.y && q.y <= r.ymax)) return false;
    long long fx = static_cast<long long>(std::floor((q.x - r.xmin) / r.resolution));
    long long fy = static_cast<long long>(std::floor((q.y - r.ymin) / r.resolution));
    ix = static_cast<int>(std::min<long long>(r.nx - 1, fx));
    iy = static_cast<int>(std::min<long long>(r.ny - 1, fy));
    return true;
}

bool SemRrtStar::modelValidState(Point q) const {
    int ix, iy;
    if (!index(q, ix, iy)) return false;
    std::size_t k = static_cast<std::size_t>(iy) * raster_.nx + ix;
    return raster_.valid[k] && raster_.risk[k] <= kRiskThreshold;
}

bool SemRrtStar::modelEdge(Point a, Point b, double &length, double &risk) const {
    int x0, y0, x1, y1;
    if (!index(a, x0, y0) || !index(b, x1, y1)) return false;
    if (std::make_pair(x1, y1) < std::make_pair(x0, y0)) { std::swap(x0, x1); std::swap(y0, y1); }
    int x = x0, y = y0, dx = std::abs(x1 - x), dy = -std::abs(y1 - y);
    int sx = x < x1 ? 1 : -1, sy = y < y1 ? 1 : -1, error = dx + dy;
    std::vector<double> &values = scratch_;
    values.clear();
    while (true) {
        std::size_t k = static_cast<std::size_t>(y) * raster_.nx + x;
        double value = raster_.risk[k];
        if (!raster_.valid[k] || !std::isfinite(value) || value > kRiskThreshold) return false;
        values.push_back(value);
        if (x == x1 && y == y1) break;
        int twice = 2 * error;
        if (twice >= dy) { error += dy; x += sx; }
        if (twice <= dx) { error += dx; y += sy; }
    }
    double ddx = std::fabs(b.x - a.x) / raster_.resolution, ddy = std::fabs(b.y - a.y) / raster_.resolution;
    length = pyHypot(ddx, ddy);
    double major = ddy > ddx ? ddy : ddx;  // max(dx, dy)
    risk = length / (major + 1) * pyFsum(values.data(), values.size());
    return true;
}

bool SemRrtStar::edge(Point a, Point b, std::array<double, 2> &base) {
    ++counters_.edge_evaluations;
    if (!terrain_.validMotion(a, b)) { ++counters_.hard_rejections; return false; }
    double length, risk;
    if (!modelEdge(a, b, length, risk)) { ++counters_.native_rejections; return false; }
    base = {length, risk};
    return true;
}

SemRrtStar::Candidate SemRrtStar::candidate(Point q, long parent, const std::array<double, 2> &base) const {
    const Node &node = nodes_[static_cast<std::size_t>(parent)];
    double turn = node.parent < 0 ? 0. : semTurnCost(q, node.point, nodes_[static_cast<std::size_t>(node.parent)].point);
    Candidate out;
    out.local = {base[0], turn, base[1]};
    double weighted[3];
    for (int i = 0; i < 3; ++i) {
        out.totals[i] = out.local[i] + node.totals[i];
        weighted[i] = kWeights[i] * out.totals[i];
    }
    out.cost = pyFsum(weighted, 3);
    return out;
}

bool SemRrtStar::isAncestor(long ancestor, long child) const {
    std::size_t steps = 0;
    while (child >= 0) {
        if (++steps > nodes_.size()) throw std::runtime_error("cycle in SEM tree");
        if (child == ancestor) return true;
        child = nodes_[static_cast<std::size_t>(child)].parent;
    }
    return false;
}

long SemRrtStar::insert(Point q, long parent, const std::array<double, 2> &base) {
    Candidate c = candidate(q, parent, base);
    long index = static_cast<long>(nodes_.size());
    Node node;
    node.point = q;
    node.parent = parent;
    node.edge_base = base;
    node.local = c.local;
    node.totals = c.totals;
    node.cost = c.cost;
    nodes_.push_back(std::move(node));
    nodes_[static_cast<std::size_t>(parent)].children.push_back(index);
    ++counters_.tree_insertions;
    gridInsert(index);
    return index;
}

void SemRrtStar::reparent(long index, long parent, const std::array<double, 2> &base) {
    Node &node = nodes_[static_cast<std::size_t>(index)];
    if (node.parent < 0 || isAncestor(index, parent)) throw std::logic_error("rewire would change root or create a cycle");
    auto &siblings = nodes_[static_cast<std::size_t>(node.parent)].children;
    auto found = std::find(siblings.begin(), siblings.end(), index);
    if (found == siblings.end()) throw std::logic_error("inconsistent SEM tree children");
    siblings.erase(found);
    nodes_[static_cast<std::size_t>(parent)].children.push_back(index);
    node.parent = parent;
    node.edge_base = base;
    std::vector<long> queue{index};
    for (std::size_t k = 0; k < queue.size(); ++k) {
        long current = queue[k];
        Node &n = nodes_[static_cast<std::size_t>(current)];
        Candidate c = candidate(n.point, n.parent, n.edge_base);
        n.cost = c.cost;
        n.local = c.local;
        n.totals = c.totals;
        queue.insert(queue.end(), n.children.begin(), n.children.end());
        if (current != index) ++counters_.descendant_cost_updates;
    }
    ++counters_.rewires;
}

bool SemRrtStar::sample(NumpyPcg64 &rng, Point start, Point goal, Point &q) {
    const auto &r = raster_;
    double direct = pyDist(start, goal);
    for (int attempt = 0; attempt < 100; ++attempt) {
        ++counters_.sample_proposals;
        Point c;
        if (has_best_) {
            // Native length is XY/res; every other native term is nonnegative.
            double bound = r.resolution * best_cost_ / kWeights[0];
            if (bound < direct - 1e-10) throw std::runtime_error("incumbent violates informed lower bound");
            double major = (direct > bound ? direct : bound) / 2;  // max(bound, direct)
            double inner = (bound - direct) * (bound + direct);
            double minor = std::sqrt(inner > 0. ? inner : 0.) / 2;
            double radius = std::sqrt(rng.random());
            double theta = 2 * kPi * rng.random();
            const Libm &m = libm();
            double x = major * radius * m.cos(theta), y = minor * radius * m.sin(theta);
            double angle = m.atan2(goal.y - start.y, goal.x - start.x);
            c.x = (start.x + goal.x) / 2 + x * m.cos(angle) - y * m.sin(angle);
            c.y = (start.y + goal.y) / 2 + x * m.sin(angle) + y * m.cos(angle);
        } else {
            c.x = rng.uniform(r.xmin, r.xmax);
            c.y = rng.uniform(r.ymin, r.ymax);
        }
        if (r.xmin <= c.x && c.x <= r.xmax && r.ymin <= c.y && c.y <= r.ymax) { q = c; return true; }
    }
    ++counters_.sampler_failures;
    return false;
}

void SemRrtStar::gridCell(Point q, int &cx, int &cy) const {
    double fx = std::floor((q.x - raster_.xmin) / cell_), fy = std::floor((q.y - raster_.ymin) / cell_);
    cx = static_cast<int>(std::min<double>(gx_ - 1, std::max(0., fx)));
    cy = static_cast<int>(std::min<double>(gy_ - 1, std::max(0., fy)));
}

void SemRrtStar::gridInsert(long index) {
    int cx, cy;
    gridCell(nodes_[static_cast<std::size_t>(index)].point, cx, cy);
    grid_[static_cast<std::size_t>(cy) * gx_ + cx].push_back(index);
}

long SemRrtStar::nearest(Point q, double &distance) const {
    // Exact argmin of math.dist with the first index on ties (np.argmin). A node in Chebyshev
    // ring k+1 around the query cell is at least k cells away, so rings can stop early.
    int cx, cy;
    gridCell(q, cx, cy);
    long best = -1;
    distance = std::numeric_limits<double>::infinity();
    int rings = std::max(gx_, gy_);
    for (int ring = 0; ring <= rings; ++ring) {
        for (int j = std::max(0, cy - ring); j <= std::min(gy_ - 1, cy + ring); ++j)
            for (int i = std::max(0, cx - ring); i <= std::min(gx_ - 1, cx + ring); ++i) {
                if (std::max(std::abs(i - cx), std::abs(j - cy)) != ring) continue;
                for (long k : grid_[static_cast<std::size_t>(j) * gx_ + i]) {
                    double d = pyDist(nodes_[static_cast<std::size_t>(k)].point, q);
                    if (d < distance || (d == distance && k < best)) { distance = d; best = k; }
                }
            }
        if (best >= 0 && distance < ring * cell_ - 1e-9) break;
    }
    if (exhaustive_check_) {
        long check = -1;
        double value = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < nodes_.size(); ++k) {
            double d = pyDist(nodes_[k].point, q);
            if (d < value) { value = d; check = static_cast<long>(k); }
        }
        if (check != best || value != distance) throw std::logic_error("grid nearest differs from exhaustive scan");
    }
    return best;
}

std::vector<long> SemRrtStar::within(Point q, double radius) const {
    int i0, j0, i1, j1;
    gridCell({q.x - radius - 1e-9, q.y - radius - 1e-9}, i0, j0);
    gridCell({q.x + radius + 1e-9, q.y + radius + 1e-9}, i1, j1);
    std::vector<long> out;
    for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i)
            for (long k : grid_[static_cast<std::size_t>(j) * gx_ + i])
                if (pyDist(nodes_[static_cast<std::size_t>(k)].point, q) <= radius) out.push_back(k);
    std::sort(out.begin(), out.end());
    if (exhaustive_check_) {
        std::vector<long> check;
        for (std::size_t k = 0; k < nodes_.size(); ++k)
            if (pyDist(nodes_[k].point, q) <= radius) check.push_back(static_cast<long>(k));
        if (check != out) throw std::logic_error("grid radius query differs from exhaustive scan");
    }
    return out;
}

void SemRrtStar::setConvergenceStop(double relative_eps, std::uint64_t window_iterations) {
    if (window_iterations && !(std::isfinite(relative_eps) && relative_eps > 0 && relative_eps < 1))
        throw std::invalid_argument("convergence stop needs 0 < relative_eps < 1");
    stop_eps_ = window_iterations ? relative_eps : 0;
    stop_window_ = window_iterations;
}

SemStatus SemRrtStar::solve(Point start, Point goal, std::uint64_t seed, double budget, std::uint64_t work,
                            double already_elapsed, const std::function<bool(const SemIncumbent &)> &callback) {
    if (!std::isfinite(budget) || budget <= 0) throw std::invalid_argument("positive finite budget required");
    if (!std::isfinite(already_elapsed) || already_elapsed < 0) throw std::invalid_argument("invalid algorithm clock origin");
    if (!std::isfinite(start.x) || !std::isfinite(start.y) || !std::isfinite(goal.x) || !std::isfinite(goal.y))
        throw std::invalid_argument("expected two finite XY coordinates");
    const auto began = Clock::now() - std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(already_elapsed));
    auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - began).count(); };
    counters_ = SemCounters{};
    convergence_ = SemConvergence{};
    nodes_.clear();
    grid_.assign(static_cast<std::size_t>(gx_) * gy_, {});
    Node root;
    root.point = goal;
    nodes_.push_back(root);
    gridInsert(0);
    has_best_ = false;
    best_cost_ = std::numeric_limits<double>::infinity();
    if (!terrain_.validMotion(start, start) || !terrain_.validMotion(goal, goal)) return SemStatus::InvalidHardEndpoints;
    if (!modelValidState(start) || !modelValidState(goal)) return SemStatus::InvalidNativeEndpoints;
    NumpyPcg64 rng(seed);
    std::vector<std::pair<long, std::array<double, 2>>> connections;  // insertion order, as the dict
    auto connect = [&](long index) {
        Point q = nodes_[static_cast<std::size_t>(index)].point;
        if (pyDist(start, q) <= kStartConnection) {
            ++counters_.start_connection_checks;
            std::array<double, 2> base;
            if (edge(start, q, base)) connections.emplace_back(index, base);
        }
    };
    auto publish = [&]() -> bool {
        bool found = false;
        double selected_cost = 0;
        long selected = -1;
        std::array<double, 3> selected_totals{};
        for (const auto &[index, base] : connections) {
            const Node &node = nodes_[static_cast<std::size_t>(index)];
            double cost;
            std::array<double, 3> totals;
            if (same(start, node.point)) { cost = node.cost; totals = node.totals; }
            else { Candidate c = candidate(start, index, base); cost = c.cost; totals = c.totals; }
            if (cost < best_cost_ && (!found || cost < selected_cost || (cost == selected_cost && index < selected))) {
                found = true; selected_cost = cost; selected = index; selected_totals = totals;
            }
        }
        if (!found) return true;
        SemIncumbent incumbent;
        incumbent.path.push_back(start);
        std::size_t steps = 0;
        for (long c = selected; c >= 0; c = nodes_[static_cast<std::size_t>(c)].parent) {
            if (++steps > nodes_.size()) throw std::runtime_error("cycle in solution path");
            Point p = nodes_[static_cast<std::size_t>(c)].point;
            if (!same(p, incumbent.path.back())) incumbent.path.push_back(p);
        }
        for (Point p : incumbent.path) {
            double z;
            if (!terrain_.height(p, z)) throw std::runtime_error("height unavailable on SEM incumbent");
            incumbent.heights.push_back(z);
        }
        double now = elapsed();
        if (now >= budget) { ++counters_.late_candidates; return true; }
        best_cost_ = selected_cost;
        has_best_ = true;
        ++counters_.incumbents;
        incumbent.elapsed_seconds = now;
        incumbent.cost = selected_cost;
        incumbent.totals = selected_totals;
        return callback ? callback(incumbent) : true;
    };
    connect(0);
    if (!publish()) return SemStatus::Aborted;
    if (same(start, goal)) return has_best_ ? SemStatus::ExactSolution : SemStatus::Timeout;
    struct Neighbor { long index; bool evaluated, valid; std::array<double, 2> base; };
    while (elapsed() < budget && (!work || counters_.iterations < work)) {
        if (stop_window_ && has_best_) {
            if (!convergence_.has_reference || best_cost_ < convergence_.reference_cost * (1. - stop_eps_)) {
                convergence_.has_reference = true;
                convergence_.reference_cost = best_cost_;
                convergence_.reference_iteration = counters_.iterations;
            } else if (counters_.iterations - convergence_.reference_iteration >= stop_window_) {
                convergence_.converged = true;
                break;
            }
        }
        ++counters_.iterations;
        Point s;
        if (!sample(rng, start, goal, s)) continue;
        double distance;
        long closest = nearest(s, distance);
        if (distance == 0) continue;
        Point a = nodes_[static_cast<std::size_t>(closest)].point;
        double ratio = kStep / distance, fraction = ratio < 1. ? ratio : 1.;  // min(1., STEP/distance)
        Point q{a.x + fraction * (s.x - a.x), a.y + fraction * (s.y - a.y)};
        if (!modelValidState(q)) { ++counters_.native_rejections; continue; }
        std::vector<Neighbor> neighbors;
        for (long k : within(q, kNeighbor)) neighbors.push_back({k, false, false, {}});
        bool any = false;
        double best_cost = 0;
        std::size_t best_slot = 0;
        for (std::size_t slot = 0; slot < neighbors.size(); ++slot) {
            if (elapsed() >= budget) break;
            ++counters_.parent_candidates;
            Neighbor &n = neighbors[slot];
            n.evaluated = true;
            n.valid = edge(q, nodes_[static_cast<std::size_t>(n.index)].point, n.base);
            if (!n.valid) continue;
            double cost = candidate(q, n.index, n.base).cost;
            if (!any || cost < best_cost || (cost == best_cost && n.index < neighbors[best_slot].index)) {
                any = true; best_cost = cost; best_slot = slot;  // min((cost, index))
            }
        }
        if (!any || elapsed() >= budget) continue;
        long added = insert(q, neighbors[best_slot].index, neighbors[best_slot].base);
        connect(added);
        for (const Neighbor &n : neighbors) {
            if (elapsed() >= budget) break;
            if (!n.evaluated || !n.valid) continue;
            const Node &node = nodes_[static_cast<std::size_t>(n.index)];
            if (node.parent < 0 || isAncestor(n.index, added)) continue;
            double cost = candidate(node.point, added, n.base).cost;
            // Complete the mutation and its dependent costs even if this crosses the deadline.
            if (cost < node.cost) reparent(n.index, added, n.base);
        }
        if (!publish()) return SemStatus::Aborted;
    }
    return has_best_ ? SemStatus::ExactSolution : SemStatus::Timeout;
}

}  // namespace terrain25d
