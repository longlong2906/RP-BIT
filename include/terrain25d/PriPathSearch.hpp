#pragma once
#include "terrain25d/PriPortfolio.hpp"
#include "terrain25d/PriSegmentSearch.hpp"
#include <array>
#include <map>

namespace terrain25d {
struct PriPathSearchOptions {
    double budget_s=20.;
    std::string mode="full"; // length | shortcut | full | segment (opt-in v2)
    std::uint64_t planner_seed=11;
    // Explicit test-only work limits; the wall deadline is NEVER disabled.
    unsigned test_anchor_iterations=0,test_iterations_per_slice=0;
    unsigned test_optimizer_proposals_per_slice=0,test_cycles=0;
    std::uint64_t max_archive_bytes=134217728,max_edge_cache_bytes=67108864;
};
struct PriPathSearchCandidate {
    std::size_t id=0;
    std::uint64_t proposal_id=0;
    unsigned callback_id=0;
    std::vector<std::size_t> parent_ids;
    std::string move_type,reason;
    double callback_s=0.,score_completed_s=0.;
    bool anchor_phase=false,pool_admitted=false;
    std::array<bool,4> published{{false,false,false,false}};
    PriPathStats stats;
    std::vector<Point> path;
    std::optional<double> segment_pri_cap,segment_length_cap;
    std::size_t segment_first=0,segment_last=0;
};
struct PriPathSearchSlice {
    unsigned id=0,iterations=0,batch_transitions=0;
    std::string phase,status;
    double began_s=0.,finished_s=0.;
    std::uint64_t proposals=0;
};
struct PriPathSearchPublication {
    std::size_t candidate_id=0;
    unsigned cap_index=0;
    double elapsed_s=0.;
    std::string reason;
};
struct PriPathSearchCounters {
    std::uint64_t proposals=0,shortcut_proposals=0,deform_proposals=0,splice_proposals=0;
    std::uint64_t segment_proposals=0;
    std::uint64_t accepted=0,pool_evictions=0,edge_cache_hits=0,edge_evaluations=0;
    std::uint64_t incidence_adds=0,incidence_removes=0,full_recomputations=0;
    std::uint64_t current_pool_size=0,peak_pool_size=0,peak_edge_cache_bytes=0,archive_bytes=0;
    std::map<std::string,std::uint64_t> rejected;
};
struct PriSegmentAttempt {
    std::uint64_t proposal_id=0;
    std::size_t parent_id=0,first=0,last=0;
    std::string move_type,status,termination;
    double pri_cap=0.,length_cap=0.,began_s=0.,finished_s=0.;
    PriSegmentSearchCounters counters;
};
struct PriPathSearchResult {
    std::string status="NO_SOLUTION",stage="NO_ANCHOR",mode;
    std::optional<std::size_t> anchor;
    std::array<std::optional<std::size_t>,4> selected{};
    std::array<double,4> cap_ratios{{1.,1.025,1.05,1.10}};
    std::array<std::optional<double>,4> length_caps{};
    double actual_wall_s=0.,anchor_deadline_s=0.,setup_completed_s=0.;
    std::optional<double> anchor_locked_s;
    bool optimizer_resource_limit=false;
    unsigned planner_instances=0;
    PolicyCounters counters;
    PriPathSearchCounters search_counters;
    std::vector<PriPathSearchCandidate> candidates;
    std::vector<PriPathSearchSlice> slices;
    std::vector<PriPathSearchPublication> publications;
    std::vector<PriSegmentAttempt> segment_attempts;
};
// Pure selector: strict cap/deadline, then PRI, L_H and stable candidate ID.
bool priPathSearchBetter(const PriPathSearchCandidate&,const PriPathSearchCandidate&);
std::array<std::optional<std::size_t>,4> selectPriPathSearchCandidates(
    const std::vector<PriPathSearchCandidate>&,double anchorLength,double deadline);
// Public test seam for the same incremental add/remove machinery used by search.
// replacement includes oldPath[first] and oldPath[last]. No external path is
// accepted by the production planner through this diagnostic API.
PriPathStats priPathSearchReplacementStats(const Terrain&,std::shared_ptr<const SafetyField>,
    const std::vector<Point>&oldPath,std::size_t first,std::size_t last,
    const std::vector<Point>&replacement,PriPathSearchCounters *counters=nullptr);
PriPathSearchResult runPriPathSearch(const Terrain&,std::shared_ptr<const SafetyField>,
    Point start,Point goal,const PriPathSearchOptions&options={});
} // namespace terrain25d
