#pragma once
#include "terrain25d/Terrain.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace terrain25d {

struct SafetyEdgeSummary {
    bool supported = false;
    // Unavailable, never zero-filled, if any touched cell has no SEM support.
    std::optional<double> maximum_risk;
    std::string reason;
    std::uint64_t touched_cells = 0;
};

// Immutable planner input. Construction and traversal are independent of the
// Python posthoc oracle; unavailable cells carry an explicit support mask.
class SafetyField {
public:
    SafetyField(int nx, int ny, double x0, double y0, double resolution,
                double xmax, double ymax, std::vector<int> support,
                std::vector<double> risk);
    static SafetyField load(const std::string &path);
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    double resolution() const { return resolution_; }
    Rect bounds() const { return {x0_, y0_, xmax_, ymax_}; }
    bool supported(int i, int j) const;
    double risk(int i, int j) const;
    Rect cellBounds(int i, int j) const;
    // Every closed cell touched by the complete segment is considered, with
    // conservative 16-ulp boundary ties in raster coordinates.
    std::string edgeRejectionReason(Point a, Point b, double riskCap) const;
    // Complete risk summary; geometry/footprint checks remain separate.
    SafetyEdgeSummary edgeSummary(Point a, Point b) const;
    // Unique row-major closed-cell IDs, including zero-length boundary contacts.
    // Empty means invalid/outside geometry. This is not a footprint traversal.
    std::vector<std::size_t> touchedCells(Point a, Point b) const;
    // Integral of the piecewise-constant SEM risk along the lifted connector.
    // Grid-line intervals use the maximum risk of incident closed cells.
    std::optional<double> edgeExposure(const Terrain &terrain, Point a, Point b) const;
private:
    std::size_t index(int i, int j) const;
    int nx_, ny_;
    double x0_, y0_, resolution_, xmax_, ymax_;
    std::vector<int> support_;
    std::vector<double> risk_;
};

class SafetyPolicy {
public:
    SafetyPolicy(const Terrain &terrain, std::shared_ptr<const SafetyField> field,
                 double riskCap = 40., double slopeCap = 35., std::vector<int> corridorMask = {});
    bool validState(Point point) const;
    bool validMotion(Point a, Point b) const;
    std::string rejectionReason(Point a, Point b) const;
    double atlasArea() const { return atlas_area_; }
    Point sampleAtlas(double areaUnit, double xUnit, double yUnit) const;
    double riskCap() const { return risk_cap_; }
    double slopeCap() const { return slope_cap_; }
    bool appliesTo(const Terrain &terrain) const { return &terrain_ == &terrain; }
    const std::shared_ptr<const SafetyField> &field() const { return field_; }
    const std::vector<int> &corridorMask() const { return corridor_mask_; }
private:
    const Terrain &terrain_;
    std::shared_ptr<const SafetyField> field_;
    double risk_cap_, slope_cap_, atlas_area_ = 0.;
    std::vector<std::size_t> atlas_cells_;
    std::vector<double> atlas_cumulative_area_;
    std::vector<int> corridor_mask_;
};

} // namespace terrain25d
