#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace terrain25d {
struct Point { double x, y; };
struct Point3 { double x, y, z; };
struct Rect { double xmin, ymin, xmax, ymax; };
struct Counters {
    std::uint64_t height_queries = 0, state_validity_calls = 0,
        motion_validity_calls = 0, full_surface_cost_calls = 0, triangle_visits = 0;
    // Candidates count rectangles passed to exact capsule distance, after the
    // selected broad phase; narrow checks count actual distance evaluations.
    std::uint64_t bvh_node_visits = 0, collision_candidates = 0, collision_narrow_checks = 0;
};
struct SurfaceEdge {
    bool has_data = false;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<Point3> points;
    std::string reason;
};
struct EdgeEvaluation {
    bool valid;
    double cost;
    std::size_t faces;
    std::string reason;
};
struct TraversabilityConfig {
    double lambda = 0, slope_weight = .5, clearance_weight = .5, clearance_comfort_m = .4;
    std::string feature_version = "cell_envelope_v1";
};
struct TraversabilityCell {
    bool available = false;
    double slope_penalty = 0, clearance_penalty = 0;
};
// Build policies change only how the immutable cell_envelope_v1 field is
// computed, never its definition or the planning objective.
enum class TraversabilityBuildPolicy { Legacy, SkipDominated };
struct TraversabilityBuildStats {
    std::uint64_t available_cells = 0, candidate_cells = 0, candidate_triangles = 0;
    std::uint64_t slope_distance_calls = 0, dominated_cell_skips = 0,
        dominated_triangle_skips = 0, slope_saturated_cells = 0;
    std::uint64_t obstacle_distance_calls = 0, clearance_saturated_cells = 0;
    double slope_extraction_seconds = 0, slope_envelope_seconds = 0,
        clearance_seconds = 0, total_seconds = 0;
};
struct EdgeCostBreakdown {
    bool has_data = false;
    double cost_terrain_m = std::numeric_limits<double>::infinity();
    double risk_slope_integral_m = std::numeric_limits<double>::infinity();
    double risk_clearance_integral_m = std::numeric_limits<double>::infinity();
    double risk_integral_m = std::numeric_limits<double>::infinity();
    double objective_value = std::numeric_limits<double>::infinity();
};

// Immutable after construction. Missing data is NaN internally, never a height of zero.
// Every cell touching missing data, or containing an over-slope triangle, is forbidden.
class Terrain {
public:
    Terrain(int nx, int ny, double x0, double y0, double dx, double dy,
            std::vector<double> heights, std::vector<int> cell_mask,
            std::vector<Rect> obstacles, double robot_radius, double margin, double slope_deg);
    static Terrain load(const std::string &filename);
    bool height(Point q, double &z) const;
    bool validState(Point q) const;
    bool validMotion(Point a, Point b) const;
    // Maximum triangle slope (degrees) touched by the complete swept disk.
    // Uses segment/triangle distance, not a conservative cell feature envelope.
    // Outside-map or missing footprint support returns +infinity.
    double maximumSweptSlope(Point a, Point b) const;
    // Optional cut parameters correspond to the lifted points. They retain
    // the original edge parameterization even if the lifted XY rounds.
    SurfaceEdge surfaceEdge(Point a, Point b, std::vector<double> *cut_parameters = nullptr) const;
    // Same traversal and summation as surfaceEdge, without materializing a polyline.
    // Returns +infinity when either endpoint or an interior segment has no data.
    double surfaceCost(Point a, Point b) const;
    // Optional immutable risk field. Hard validity and legacy surface costs are unchanged.
    // Only explicit obstacle rectangles contribute to the obstacle-clearance feature.
    void enableTraversability(const TraversabilityConfig &config,
                              TraversabilityBuildPolicy policy = TraversabilityBuildPolicy::Legacy);
    bool traversabilityEnabled() const { return traversability_enabled_; }
    const TraversabilityConfig &traversabilityConfig() const;
    double traversabilityPreprocessingSeconds() const { return traversability_preprocessing_seconds_; }
    TraversabilityBuildPolicy traversabilityBuildPolicy() const { return traversability_build_policy_; }
    const TraversabilityBuildStats &traversabilityBuildStats() const { return traversability_build_stats_; }
    // Input-data predicate for the deliberately narrow flat_direct_v1 proof.
    // Computed while validating all input heights; NaN/NoData always makes it false.
    bool hasCompleteConstantHeight() const { return complete_constant_height_; }
    bool hasExplicitObstacles() const { return !obstacles_.empty(); }
    TraversabilityCell featureCell(int i, int j) const;
    // Exact integral of the declared cell field along the same lifted connector.
    // Grid-line intervals use componentwise maxima of incident available cells;
    // a grid line is the exact represented origin + index * spacing coordinate.
    // Without the optional field, risk is zero and objective equals surface length.
    EdgeCostBreakdown edgeCostBreakdown(Point a, Point b,
                                       std::vector<Point3> *surface_points = nullptr) const;
    // Optional preprocessing; the original bounding-box scan remains the default.
    // Idempotent and independent of the query counters; model data never changes.
    void enableCollisionIndex();
    bool collisionIndexEnabled() const { return collision_index_enabled_; }
    double collisionIndexBuildSeconds() const { return collision_index_build_seconds_; }
    EdgeEvaluation evaluate(Point a, Point b) const;
    double lowerBound(Point a, Point b, bool lifted) const;
    double solutionLowerBound(Point start, Point q, Point goal, bool lifted) const;
    Rect bounds() const { return {x0_, y0_, x0_ + (nx_-1)*dx_, y0_ + (ny_-1)*dy_}; }
    double radius() const { return radius_; }
    double preprocessingSeconds() const { return preprocessing_seconds_; }
    const Counters &counters() const { return counters_; }
    void resetCounters() const { counters_ = {}; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }
private:
    struct CollisionNode {
        Rect box;
        std::size_t begin = 0, end = 0;
        int left = -1, right = -1;
    };
    bool capsuleClear(Point a, Point b) const;
    bool indexedCapsuleClear(int node, Point a, Point b, Rect query, double radius) const;
    int buildCollisionNode(std::size_t begin, std::size_t end);
    bool surfaceCuts(Point a, Point b, std::vector<double> &cuts, std::string &reason) const;
    bool inBounds(Point p) const;
    bool cellHasData(int i, int j) const;
    TraversabilityCell intervalFeature(Point a, Point b, Point midpoint) const;
    double h(int i, int j) const { return heights_[j*nx_+i]; }
    int nx_, ny_;
    double x0_, y0_, dx_, dy_, radius_, slope_limit_deg_;
    std::vector<double> heights_;
    std::vector<int> forbidden_;
    std::vector<Rect> obstacles_;
    std::vector<Rect> collision_rectangles_;
    std::vector<CollisionNode> collision_nodes_;
    bool collision_index_enabled_ = false;
    double collision_index_build_seconds_ = 0;
    bool traversability_enabled_ = false;
    bool complete_constant_height_ = false;
    TraversabilityBuildPolicy traversability_build_policy_ = TraversabilityBuildPolicy::Legacy;
    TraversabilityBuildStats traversability_build_stats_;
    TraversabilityConfig traversability_config_;
    std::vector<TraversabilityCell> traversability_cells_;
    double traversability_preprocessing_seconds_ = 0;
    mutable Counters counters_;
    double preprocessing_seconds_ = 0;
};
double distanceXY(Point a, Point b);
double segmentRectangleDistance(Point a, Point b, Rect rect);
}
