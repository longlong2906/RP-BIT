#pragma once
#include "terrain25d/Safety.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace terrain25d {
struct CorridorOptions {
    double alpha=1.10;
    unsigned label_bins=16, dilation_cells=1;
    std::uint64_t max_labels=2000000, max_memory_bytes=536870912;
    double grid_spacing_m=.1, endpoint_connection_radius_m=.2;
    // V4 only. V3 ignores these fields and retains its original algorithm.
    std::string selector="preview_pri";
    std::string beam_scheduler="legacy_pri_first";
    unsigned representatives_per_bin=4, max_preview_candidates=64;
    std::uint64_t max_preview_edge_checks=2000000;
};
struct CorridorPathStats {
    bool supported=false;
    std::optional<double> length,pri,exposure,risk_sum;
    std::uint64_t risk_count=0;
    // Linear interpolated quantiles q0,q25,q50,q75,q100.
    std::vector<double> risk_quantiles;
    std::vector<std::size_t> cells;
};
struct CorridorCandidate {
    std::string source;
    std::vector<std::uint32_t> raw_node_ids;
    std::vector<Point> raw,preview;
    CorridorPathStats raw_stats,preview_stats;
    std::uint64_t preview_edge_checks=0;
    bool selected=false;
};
CorridorPathStats corridorPathStats(const Terrain&,const SafetyField&,const std::vector<Point>&);
struct CorridorResult {
    std::string status="GUIDANCE_NO_CONNECTION";
    std::optional<double> L0,guide_length,guide_pri,guide_exposure,risk_sum;
    std::vector<Point> guide;
    std::vector<int> mask;
    std::uint64_t risk_count=0;
    std::vector<double> risk_quantiles;
    std::uint64_t graph_nodes=0,state_evaluations=0,edge_evaluations=0,edge_cache_hits=0,
        labels_allocated=0,peak_memory_bytes=0;
    double graph_s=0,guidance_s=0,mask_s=0,total_s=0;
    double preview_s=0;
    std::uint64_t preview_edge_checks=0,beam_rejected=0,non_simple_rejected=0;
    std::uint64_t beam_labels_allocated=0,beam_replacement_labels=0,beam_role_replacements=0,
        beam_equal_score_retains=0,beam_retired=0,beam_expansions=0,beam_stale_pops=0;
    std::vector<Point> raw_guide,preview;
    CorridorPathStats raw_guide_stats,preview_stats;
    std::vector<CorridorCandidate> candidate_diagnostics;
    bool connected() const { return status=="GUIDANCE_READY"; }
};
// Finite-graph, bounded approximate guidance, not a replacement planner.
// The single selected guide only fixes a centerline mask; never seeds BIT*.
CorridorResult computeCorridor(const Terrain&,const SafetyField&,Point start,Point goal,
                              const CorridorOptions &options={});
CorridorResult computeCorridorV4(const Terrain&,const SafetyField&,Point start,Point goal,
                                const CorridorOptions &options={});
// Conservative exact binary64 segment predicates; adjacent shared endpoints
// are allowed, repeated vertices, overlap and non-adjacent contacts are not.
bool corridorSimplePath(const std::vector<Point>&);
} // namespace terrain25d
