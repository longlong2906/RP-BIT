#pragma once
#include "terrain25d/Safety.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace terrain25d {

struct RiskCalibrationOptions {
    double grid_spacing_m = .1;
    double endpoint_connection_radius_m = .2;
    double risk_ceiling = 40.;
    double slope_cap_deg = 35.;
    double risk_slack = .5;
};

struct RiskCalibrationResult {
    std::string status = "CALIBRATION_NO_CONNECTION";
    std::optional<double> b_graph, effective_risk_cap;
    std::vector<Point> witness;
    std::uint64_t graph_nodes = 0, settled_nodes = 0, state_evaluations = 0,
        edge_evaluations = 0, edge_cache_hits = 0;
    double preprocessing_s = 0.;
    bool connected() const { return status == "CALIBRATED"; }
};

// Deterministic finite-graph minimax calibration. The witness is only evidence
// for the threshold, never a planner solution or an externally seeded incumbent.
// Grid nodes are origin + integer * spacing, with eight-neighbour grid edges.
// Explicit endpoints connect to every grid node within the declared radius
// (16-ulp ties at the coordinate magnitude) and to one another. All edges check
// complete support, capsule and swept slope.
// Missing graph connection does not prove continuous-space infeasibility.
RiskCalibrationResult calibrateRisk(const Terrain &terrain, const SafetyField &field,
                                   Point start, Point goal,
                                   const RiskCalibrationOptions &options = {});

} // namespace terrain25d
