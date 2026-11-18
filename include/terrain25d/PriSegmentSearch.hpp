#pragma once
#include "terrain25d/PriPortfolio.hpp"
#include <functional>

namespace terrain25d {
// A bounded path-proposal operator. This is not a replacement OMPL planner,
// and its finite beam does not establish a globally optimal path.
struct PriSegmentSearchOptions {
    double pri_cap=0.,length_cap=0.;
    std::uint64_t seed=0;
    // Beam width applies per endpoint per depth. Expansion work is capped
    // across the whole search; max_depth bounds replacement edge count.
    std::size_t max_nodes=24,beam_width=8,max_expansions=128,max_depth=8;
    std::function<bool()> cancel;
    // A scheduling slice may end between expansions while keeping an already
    // fully scored result. Only cancel is a hard publication deadline.
    std::function<bool()> stop;
    // Explicit deterministic test seam; these nodes use the same checks.
    std::vector<Point> extra_nodes;
    // Production retains evenly spaced parent knots in at most one third of
    // the remaining slots; new uniform/low-risk samples fill the rest.
    bool generate_nodes=true;
};
struct PriSegmentSearchCounters {
    std::uint64_t nodes_proposed=0,nodes_retained=0,edge_evaluations=0,edge_cache_hits=0;
    std::uint64_t expansions=0,labels_created=0,beam_discarded=0,distinct_sets_same_node=0;
    std::uint64_t prefix_above_pri_kept=0,length_rejected=0,invalid_edges=0,non_simple_rejected=0;
    std::uint64_t completed_paths=0,pri_rejected=0,full_recomputations=0,cancel_checks=0,soft_stops=0;
};
struct PriSegmentSearchResult {
    std::string status="NO_CANDIDATE";
    std::string termination="WORK_COMPLETED";
    double pri_cap=0.,length_cap=0.;
    std::size_t first=0,last=0;
    // Both are empty unless a fully checked, in-budget candidate is returned.
    std::vector<Point> path,replacement;
    PriPathStats stats;
    std::vector<std::size_t> outside_cells;
    PriSegmentSearchCounters counters;
};
PriSegmentSearchResult runPriSegmentSearch(const Terrain&,const SafetyPolicy&,
    const std::vector<Point>& parent,std::size_t first,std::size_t last,
    const PriSegmentSearchOptions&);
// Convenience overload. Production should reuse its existing policy, since
// policy construction and its atlas also consume the caller's deadline.
PriSegmentSearchResult runPriSegmentSearch(const Terrain&,std::shared_ptr<const SafetyField>,
    const std::vector<Point>& parent,std::size_t first,std::size_t last,
    const PriSegmentSearchOptions&);
} // namespace terrain25d
