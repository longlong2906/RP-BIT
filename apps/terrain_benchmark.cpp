#include "terrain25d/OmplIntegration.hpp"
#include "terrain25d/FlatDirectCertificate.hpp"
#include "terrain25d/RiskCalibration.hpp"
#include "terrain25d/Corridor.hpp"
#include "terrain25d/SemExposure.hpp"
#include "terrain25d/PriPortfolio.hpp"
#include "terrain25d/PriPathSearch.hpp"
#include "terrain25d/RpBit.hpp"
#include <ompl/config.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/util/Console.h>
#include <boost/json.hpp>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <regex>
#include <thread>

namespace bj=boost::json;
namespace fs=std::filesystem;
using namespace terrain25d;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
bj::value number(double x) {return std::isfinite(x)?bj::value(x):bj::value(nullptr);}
std::uint64_t unsignedProgress(const ob::Planner &planner,const std::string &name) {
    const auto &properties=planner.getPlannerProgressProperties();auto found=properties.find(name);
    if(found==properties.end())throw std::runtime_error("missing planner progress property: "+name);
    return std::stoull(found->second());
}
void saveJson(const fs::path &file,const bj::value &value) {
    std::ofstream out(file);if(!out)throw std::runtime_error("cannot write "+file.string());out<<bj::serialize(value)<<'\n';
}
// A calibration witness is evidence for a fixed domain, never a planner path.
bj::object calibrationJson(const RiskCalibrationResult &calibration,
                           const RiskCalibrationOptions &options,const fs::path &output) {
    bj::array witness;
    for(const auto p:calibration.witness)witness.push_back(bj::array{p.x,p.y});
    bj::value witnessFile=nullptr;
    if(calibration.connected()) {
        witnessFile="calibration_witness_xy.csv";
        std::ofstream file(output/"calibration_witness_xy.csv",std::ios::binary);
        if(!file)throw std::runtime_error("cannot write calibration witness");
        file<<std::setprecision(17)<<"x,y\n";
        for(const auto p:calibration.witness)file<<p.x<<','<<p.y<<'\n';
        file.close();
        if(!file)throw std::runtime_error("cannot flush calibration witness");
    }
    return {{"status",calibration.status},
        {"b_graph",calibration.b_graph?number(*calibration.b_graph):bj::value(nullptr)},
        {"effective_risk_cap",calibration.effective_risk_cap?number(*calibration.effective_risk_cap):bj::value(nullptr)},
        {"risk_slack",options.risk_slack},{"risk_ceiling",options.risk_ceiling},
        {"grid_spacing_m",options.grid_spacing_m},{"endpoint_radius_m",options.endpoint_connection_radius_m},
        {"endpoint_radius_tie_ulps",16},
        {"endpoint_radius_tie_scale","max(1,abs(endpoint_xy),abs(grid_node_xy),radius)"},
        {"slope_cap_deg",options.slope_cap_deg},{"witness_xy",witness},{"witness_file",witnessFile},
        {"witness_hash_policy","sha256_of_csv_recorded_by_runner_in_calibration_receipt"},
        {"edge_checks",calibration.edge_evaluations},{"edge_cache_hits",calibration.edge_cache_hits},
        {"graph_nodes",calibration.graph_nodes},{"settled_nodes",calibration.settled_nodes},
        {"state_evaluations",calibration.state_evaluations},{"preprocessing_s",calibration.preprocessing_s},
        {"bottleneck_scope","finite_calibration_graph_not_continuous_optimum"},
        {"witness_used_as_incumbent",false}};
}
std::string utcTimestamp() {
    const auto now=std::chrono::system_clock::now();
    const auto milliseconds=std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    const auto whole=std::chrono::system_clock::to_time_t(now);
    std::tm value=*std::gmtime(&whole);std::ostringstream out;
    out<<std::put_time(&value,"%Y-%m-%dT%H:%M:%S")<<'.'
       <<std::setw(3)<<std::setfill('0')<<(milliseconds%1000)<<'Z';return out.str();
}
// The guide defines an immutable search domain; it is never a planner incumbent.
bj::object corridorJson(const CorridorResult &c,const CorridorOptions &o,const fs::path &output,bool v4=false) {
    bj::array guide,mask,quantiles;
    for(const auto p:c.guide)guide.push_back(bj::array{p.x,p.y});
    for(const auto value:c.mask)mask.push_back(value);
    for(const auto value:c.risk_quantiles)quantiles.push_back(number(value));
    if(c.connected()) {
        std::ofstream file(output/"corridor_guide_xy.csv",std::ios::binary);
        if(!file)throw std::runtime_error("cannot write corridor guide");
        file<<std::setprecision(17)<<"x,y\n";
        for(const auto p:c.guide)file<<p.x<<','<<p.y<<'\n';
        file.close();if(!file)throw std::runtime_error("cannot flush corridor guide");
    }
    auto optionalNumber=[](const std::optional<double> &v)->bj::value{return v?number(*v):bj::value(nullptr);};
    bj::object result{{"status",c.status},{"L0",optionalNumber(c.L0)},
        {"guide_length",optionalNumber(c.guide_length)},{"guide_pri",optionalNumber(c.guide_pri)},
        {"guide_exposure",optionalNumber(c.guide_exposure)},{"risk_sum",optionalNumber(c.risk_sum)},
        {"risk_count",c.risk_count},{"risk_quantiles",quantiles},{"guide_xy",guide},{"mask",mask},
        {"guide_file",c.connected()?bj::value("corridor_guide_xy.csv"):bj::value(nullptr)},
        {"guide_used_as_incumbent",false},{"graph_s",c.graph_s},{"guidance_s",c.guidance_s},
        {"mask_s",c.mask_s},{"total_s",c.total_s},
        {"alpha",o.alpha},{"label_bins",o.label_bins},{"dilation_cells",o.dilation_cells},
        {"grid_spacing_m",o.grid_spacing_m},{"endpoint_radius_m",o.endpoint_connection_radius_m},
        {"max_labels",o.max_labels},{"max_memory_bytes",o.max_memory_bytes},
        {"hash_policy","sha256_canonical_json_mask_and_csv_guide_in_corridor_receipt"},
        {"scope","approximate_guidance_on_finite_graph_not_continuous_or_PRI_optimum"},
        {"counters",{{"graph_nodes",c.graph_nodes},{"state_evaluations",c.state_evaluations},
            {"edge_evaluations",c.edge_evaluations},{"edge_cache_hits",c.edge_cache_hits},
            {"labels_allocated",c.labels_allocated},{"peak_memory_bytes",c.peak_memory_bytes}}}};
    if(v4) {
        const auto xy=[](const std::vector<Point> &path){bj::array out;for(const auto p:path)out.push_back(bj::array{p.x,p.y});return out;};
        const auto stats=[&](const CorridorPathStats &s)->bj::value {
            if(!s.supported)return nullptr;
            bj::array q;for(const auto v:s.risk_quantiles)q.push_back(number(v));
            return bj::object{{"length",optionalNumber(s.length)},{"pri",optionalNumber(s.pri)},
                {"exposure",optionalNumber(s.exposure)},{"risk_sum",optionalNumber(s.risk_sum)},
                {"risk_count",s.risk_count},{"risk_quantiles",q}};
        };
        const auto pathFile=[&](const std::vector<Point> &path,const char *name)->bj::value {
            if(path.empty()||!c.connected())return nullptr;
            std::ofstream file(output/name,std::ios::binary);if(!file)throw std::runtime_error("cannot write v4 guide");
            file<<std::setprecision(17)<<"x,y\n";for(const auto p:path)file<<p.x<<','<<p.y<<'\n';
            file.close();if(!file)throw std::runtime_error("cannot flush v4 guide");return name;
        };
        result["raw_guide_xy"]=xy(c.raw_guide);result["preview_xy"]=xy(c.preview);
        result["raw_guide_file"]=pathFile(c.raw_guide,"corridor_raw_guide_xy.csv");
        result["preview_file"]=pathFile(c.preview,"corridor_preview_xy.csv");
        result["raw_guide_stats"]=stats(c.raw_guide_stats);result["preview_stats"]=stats(c.preview_stats);
        result["selector"]=o.selector;result["representatives_per_bin"]=o.representatives_per_bin;
        result["max_preview_candidates"]=o.max_preview_candidates;result["max_preview_edge_checks"]=o.max_preview_edge_checks;
        result["preview_s"]=c.preview_s;result["preview_used_as_incumbent"]=false;
        result["preview_policy"]="single_pass_farthest_visible_strict_surface_decrease_provisional_dilation1";
        result["preview_length_decrease_ulps"]=64;
        result["preview_length_decrease_scale"]="max(1,abs(old),abs(new))";
        result["guide_policy"]="unique_cell_four_role_beam_no_safe_cross_path_dominance";
        result["beam_scheduler"]=o.beam_scheduler;
        auto &counter=result["counters"].as_object();counter["preview_edge_checks"]=c.preview_edge_checks;
        counter["beam_rejected"]=c.beam_rejected;counter["non_simple_rejected"]=c.non_simple_rejected;
        counter["beam_labels_allocated"]=c.beam_labels_allocated;
        counter["beam_replacement_labels"]=c.beam_replacement_labels;
        counter["beam_role_replacements"]=c.beam_role_replacements;
        counter["beam_equal_score_retains"]=c.beam_equal_score_retains;
        counter["beam_retired"]=c.beam_retired;
        counter["beam_expansions"]=c.beam_expansions;
        counter["beam_stale_pops"]=c.beam_stale_pops;
        bj::array diagnostics;
        for(const auto &candidate:c.candidate_diagnostics) {
            bj::array ids;for(const auto id:candidate.raw_node_ids)ids.push_back(id);
            diagnostics.push_back(bj::object{{"source",candidate.source},{"raw_node_ids",ids},
                {"raw_xy",xy(candidate.raw)},{"preview_xy",xy(candidate.preview)},
                {"raw_stats",stats(candidate.raw_stats)},{"preview_stats",stats(candidate.preview_stats)},
                {"preview_edge_checks",candidate.preview_edge_checks},{"selected",candidate.selected}});
        }
        result["candidate_diagnostics"]=diagnostics;
    }
    return result;
}
void savePhase(const fs::path &output,const char *phase) {
    bj::object value{{"phase",phase},{"updated_utc",utcTimestamp()}};
    if(std::string(phase)=="SOLVING")value["solve_started_utc"]=value["updated_utc"];
    const auto temporary=output/"phase.json.tmp",target=output/"phase.json";
    saveJson(temporary,value);
    // Only a diagnostic control file is replaced here; raw results are separate.
    std::error_code error;
    for(int attempt=0;attempt<10;++attempt) {
#ifdef _WIN32
        // A parent reader can briefly hold this control file open on Windows.
        fs::remove(target,error);
        if(error) {
            if(attempt==9)throw fs::filesystem_error("replace phase control file",target,error);
            std::this_thread::sleep_for(std::chrono::milliseconds(50*(attempt+1)));continue;
        }
#endif
        // POSIX rename replaces the control file atomically, with no missing gap.
        fs::rename(temporary,target,error);
        if(!error)return;
        if(attempt==9)throw fs::filesystem_error("commit phase control file",temporary,target,error);
        std::this_thread::sleep_for(std::chrono::milliseconds(50*(attempt+1)));
    }
}
bj::object counterJson(const Counters &c,const PolicyCounters &p) {
    return {{"height_queries",c.height_queries},{"state_validity_calls",c.state_validity_calls},
        {"motion_cost_calls",p.motion_cost_calls},{"traversability_motion_cost_calls",p.traversability_motion_cost_calls},
        {"sem_exposure_motion_cost_calls",p.sem_exposure_motion_cost_calls},
        {"sem_exposure_integral_calls",p.sem_exposure_integral_calls},
        {"motion_validity_calls",c.motion_validity_calls},{"full_surface_cost_calls",c.full_surface_cost_calls},
        {"triangle_visits",c.triangle_visits},{"motion_heuristic_calls",p.motion_heuristic_calls},
        {"bvh_node_visits",c.bvh_node_visits},{"collision_candidates",c.collision_candidates},
        {"collision_narrow_checks",c.collision_narrow_checks},
        {"height_cache_hits",p.height_cache_hits},{"height_cache_misses",p.height_cache_misses},
        {"fixed_heuristic_cache_hits",p.fixed_heuristic_cache_hits},
        {"fixed_heuristic_cache_misses",p.fixed_heuristic_cache_misses},
        {"sampler_query_cache_hits",p.sampler_query_cache_hits},{"sampler_query_cache_rebuilds",p.sampler_query_cache_rebuilds},
        {"sampler_bound_cache_hits",p.sampler_bound_cache_hits},{"sampler_bound_cache_rebuilds",p.sampler_bound_cache_rebuilds},
        {"sampler_invalid_interval_calls",p.sampler_invalid_interval_calls},
        {"sampler_lower_bound_rejected_calls",p.sampler_lower_bound_rejected_calls},
        {"sampler_proposal_limit_calls",p.sampler_proposal_limit_calls},
        {"safety_state_checks",p.safety_state_checks},{"safety_motion_checks",p.safety_motion_checks},
        {"safety_geometry_rejections",p.safety_geometry_rejections},
        {"safety_support_rejections",p.safety_support_rejections},{"safety_risk_rejections",p.safety_risk_rejections},
        {"safety_slope_rejections",p.safety_slope_rejections},
        {"safety_corridor_rejections",p.safety_corridor_rejections},
        {"atlas_sample_proposals",p.atlas_sample_proposals},{"ellipse_sample_proposals",p.ellipse_sample_proposals},
        {"from_start_heuristic_calls",p.from_start_calls},{"to_goal_heuristic_calls",p.to_goal_calls},
        {"cost_to_go_calls",p.cost_to_go_calls},{"sampler_allocations",p.sampler_allocations},
        {"sampler_one_bound_calls",p.sampler_one_bound_calls},{"sampler_interval_calls",p.sampler_interval_calls},
        {"sampler_finite_bound_calls",p.sampler_finite_bound_calls},{"sample_proposals",p.sample_proposals},
        {"samples_returned_before_validity",p.samples_returned},{"rejected_bounds",p.rejected_bounds},
        {"rejected_nodata",p.rejected_nodata},{"rejected_heuristic",p.rejected_heuristic},
        {"sampler_exhausted_calls",p.exhausted_calls},{"informed_measure_calls",p.measure_calls},
        {"finite_bound_sample_proposals",p.finite_bound_sample_proposals},
        {"finite_bound_samples_returned",p.finite_bound_samples_returned},
        {"finite_bound_rejected_bounds",p.finite_bound_rejected_bounds},
        {"finite_bound_rejected_nodata",p.finite_bound_rejected_nodata},
        {"finite_bound_rejected_heuristic",p.finite_bound_rejected_heuristic}};
}
void markNativeSamplerCountersUnavailable(bj::object &c) {
    for(const char *key:{"sample_proposals","samples_returned_before_validity","rejected_bounds","rejected_nodata",
        "rejected_heuristic","sampler_exhausted_calls","finite_bound_sample_proposals",
        "finite_bound_samples_returned","finite_bound_rejected_bounds","finite_bound_rejected_nodata",
        "finite_bound_rejected_heuristic"})c[key]=nullptr;
    c["native_sampler_counter_unavailable_reason"]="OMPL_PathLengthDirectInfSampler_does_not_expose_proposal_counters";
}
void savePath(const fs::path &directory,const std::vector<Point> &path,const Terrain &map) {
    fs::create_directories(directory);
    std::ofstream xy(directory/"path_xy.csv"),surface(directory/"path_surface.csv");
    xy<<std::setprecision(17)<<"x_m,y_m\n";
    surface<<std::setprecision(17)<<"x_m,y_m,z_m\n";
    for(auto q:path)xy<<q.x<<','<<q.y<<'\n';
    if(path.size()==1){double z=0;if(!map.height(path[0],z))throw std::runtime_error("path NoData");
        surface<<path[0].x<<','<<path[0].y<<','<<z<<'\n';}
    for(std::size_t i=1;i<path.size();++i){auto edge=map.surfaceEdge(path[i-1],path[i]);
        if(!edge.has_data)throw std::runtime_error("path contains NoData");
        for(std::size_t k=i==1?0:1;k<edge.points.size();++k){auto p=edge.points[k];surface<<p.x<<','<<p.y<<','<<p.z<<'\n';}}
}
struct Event {double elapsed,cost;std::vector<Point> path;bj::object counters;};
TraversabilityConfig parseTraversability(const bj::object &value) {
    const std::vector<std::string> keys{"feature_version","lambda","slope_weight","clearance_weight","clearance_comfort_m"};
    if(value.size()!=keys.size())throw std::runtime_error("traversability: missing or unknown field");
    for(const auto &key:keys)if(!value.if_contains(key))throw std::runtime_error("traversability: missing "+key);
    TraversabilityConfig config;
    config.feature_version=bj::value_to<std::string>(value.at("feature_version"));
    config.lambda=bj::value_to<double>(value.at("lambda"));
    config.slope_weight=bj::value_to<double>(value.at("slope_weight"));
    config.clearance_weight=bj::value_to<double>(value.at("clearance_weight"));
    config.clearance_comfort_m=bj::value_to<double>(value.at("clearance_comfort_m"));
    return config;
}
bj::object traversabilityJson(const TraversabilityConfig &config) {
    return {{"feature_version",config.feature_version},{"lambda",config.lambda},
        {"slope_weight",config.slope_weight},{"clearance_weight",config.clearance_weight},
        {"clearance_comfort_m",config.clearance_comfort_m}};
}
bj::object pathCostBreakdown(const std::vector<Point> &path,const Terrain &map) {
    double length=0,xy=0,slope=0,clearance=0,risk=0,objective=0;
    for(std::size_t i=1;i<path.size();++i) {
        const auto edge=map.edgeCostBreakdown(path[i-1],path[i]);
        if(!edge.has_data)throw std::runtime_error("snapshot has undefined traversability cost");
        length+=edge.cost_terrain_m;xy+=distanceXY(path[i-1],path[i]);
        slope+=edge.risk_slope_integral_m;clearance+=edge.risk_clearance_integral_m;
        risk+=edge.risk_integral_m;objective+=edge.objective_value;
    }
    return {{"cost_terrain_m",number(length)},{"best_cost_terrain_m",number(length)},
        {"length_xy_m",number(xy)},{"risk_slope_integral_m",number(slope)},
        {"risk_clearance_integral_m",number(clearance)},{"risk_integral_m",number(risk)},
        {"objective_value",number(objective)}};
}
SemExposureConfig parseSemExposure(const bj::object &value) {
    for(const auto &entry:value)
        if(entry.key()!="lambda" && entry.key()!="risk_scale")
            throw std::runtime_error("unknown sem_exposure field: "+std::string(entry.key()));
    SemExposureConfig config;
    if(const auto *weight=value.if_contains("lambda"))config.lambda=bj::value_to<double>(*weight);
    if(const auto *scale=value.if_contains("risk_scale"))config.risk_scale=bj::value_to<double>(*scale);
    if(!std::isfinite(config.lambda) || config.lambda<0. || config.risk_scale!=40.)
        throw std::runtime_error("sem_exposure requires finite nonnegative lambda and risk_scale40");
    return config;
}
bj::object semExposurePathCostBreakdown(const std::vector<Point> &path,const Terrain &map,
                                      const PlannerOptions &options) {
    if(path.empty() || !options.sem_exposure || !options.safety)
        throw std::runtime_error("SEM exposure snapshot/config unavailable");
    double length=0.,xy=0.,exposure=0.;
    const std::size_t edgeCount=path.size()>1?path.size()-1:1;
    for(std::size_t i=0;i<edgeCount;++i) {
        const auto a=path[i],b=path.size()>1?path[i+1]:a;
        const double cost=map.surfaceCost(a,b);
        const auto integral=semExposureCanonical(map,*options.safety->field(),a,b);
        if(!std::isfinite(cost) || !integral)throw std::runtime_error("snapshot has undefined SEM exposure cost");
        length+=cost;xy+=distanceXY(a,b);exposure+=*integral;
    }
    const double normalized=exposure/options.sem_exposure->risk_scale;
    return {{"cost_terrain_m",number(length)},{"best_cost_terrain_m",number(length)},
        {"length_xy_m",number(xy)},{"sem_exposure_risk_m",number(exposure)},
        {"sem_exposure_normalized_m",number(normalized)},
        {"objective_value",number(length+options.sem_exposure->lambda*normalized)}};
}
void addMetrics(bj::object &target,const bj::object &metrics) {
    for(const auto &field:metrics)target[field.key()]=field.value();
}
bj::object buildStatsJson(const TraversabilityBuildStats &s) {
    return {{"available_cells",s.available_cells},{"candidate_cells",s.candidate_cells},
        {"candidate_triangles",s.candidate_triangles},{"slope_distance_calls",s.slope_distance_calls},
        {"dominated_cell_skips",s.dominated_cell_skips},{"dominated_triangle_skips",s.dominated_triangle_skips},
        {"slope_saturated_cells",s.slope_saturated_cells},{"obstacle_distance_calls",s.obstacle_distance_calls},
        {"clearance_saturated_cells",s.clearance_saturated_cells},
        {"slope_extraction_seconds",s.slope_extraction_seconds},{"slope_envelope_seconds",s.slope_envelope_seconds},
        {"clearance_seconds",s.clearance_seconds},{"total_seconds",s.total_seconds}};
}
bj::object certificateConditions(const FlatDirectCertificate &c) {
    return {{"complete_constant_height",c.complete_constant_height},{"direct_endpoint_path",c.direct_endpoint_path},
        {"zero_active_risk",c.zero_active_risk},{"hard_validity_checked",c.hard_validity_checked},{"hard_valid",c.hard_valid}};
}
int runPriPortfolioWorker(const bj::object &request,const fs::path &output,bool cloudHandshake) {
    const auto requireString=[&](const char *key,const char *expected){if(!request.if_contains(key)||bj::value_to<std::string>(request.at(key))!=expected)throw std::runtime_error(std::string("PRI portfolio requires ")+key+"="+expected);};
    requireString("planner_id","bitstar_a3");requireString("objective_type","surface_length");
    for(const char *key:{"traversability","sem_exposure"})if(request.if_contains(key))throw std::runtime_error("PRI portfolio cannot mix objectives");
    for(const char *key:{"state_cache","scalar_cost","sampler_query_cache","collision_index"})
        if(!request.if_contains(key)||!bj::value_to<bool>(request.at(key)))throw std::runtime_error(std::string("PRI portfolio requires ")+key);
    if(request.if_contains("exact_flat_certificate")&&bj::value_to<bool>(request.at("exact_flat_certificate")))throw std::runtime_error("PRI portfolio does not support a length certificate");
    const auto &registered=request.at("pri_portfolio").as_object();
    if(registered.size()!=1||!registered.if_contains("length_cap_ratio"))throw std::runtime_error("pri_portfolio: missing or unknown field");
    PriPortfolioOptions options;options.length_cap_ratio=bj::value_to<double>(registered.at("length_cap_ratio"));options.budget_s=bj::value_to<double>(request.at("planning_budget_s"));
    if(const auto *work=request.if_contains("work_iterations")){
        options.work_iterations_per_episode=bj::value_to<unsigned>(*work);
        if(options.work_iterations_per_episode&&(!request.if_contains("test_only_fixed_work")||!bj::value_to<bool>(request.at("test_only_fixed_work"))))
            throw std::runtime_error("PRI portfolio work limit is available only with test_only_fixed_work=true");
    }
    if(bj::value_to<unsigned>(request.at("samples_per_batch"))!=100||bj::value_to<unsigned>(request.at("batch_after_first_solution"))!=200||
       bj::value_to<double>(request.at("rewire_factor"))!=1.1||bj::value_to<unsigned>(request.at("sample_attempts"))!=100)
        throw std::runtime_error("PRI portfolio requires locked BIT* batch/rewire/sampling configuration");
    const auto &safety=request.at("safe_length").as_object();
    for(const auto &entry:safety){const auto key=entry.key();if(key!="field_path"&&key!="risk_cap"&&key!="slope_cap_deg"&&key!="atlas_sampling"&&key!="policy_id"&&
            key!="field_sha256"&&key!="cloud_sha256"&&key!="builder_source_sha256")throw std::runtime_error("PRI portfolio unknown safety field");}
    if(bj::value_to<std::string>(safety.at("policy_id"))!="a3t_safe_length_v1"||bj::value_to<double>(safety.at("risk_cap"))!=40.||
       bj::value_to<double>(safety.at("slope_cap_deg"))!=35.||!bj::value_to<bool>(safety.at("atlas_sampling")))throw std::runtime_error("PRI portfolio safety contract mismatch");
    const auto pointFrom=[&](const char *key){const auto &a=request.at(key).as_array();if(a.size()!=2)throw std::runtime_error("PRI portfolio XY endpoints required");Point p{bj::value_to<double>(a[0]),bj::value_to<double>(a[1])};if(!std::isfinite(p.x)||!std::isfinite(p.y))throw std::runtime_error("nonfinite endpoint");return p;};
    const auto start=pointFrom("start"),goal=pointFrom("goal");
    const auto beforeLoad=Clock::now();auto map=Terrain::load(bj::value_to<std::string>(request.at("map_path")));map.enableCollisionIndex();
    auto field=std::make_shared<const SafetyField>(SafetyField::load(bj::value_to<std::string>(safety.at("field_path"))));
    const double preprocessing=seconds(beforeLoad);
    ompl::RNG::setSeed(bj::value_to<unsigned>(request.at("planner_seed")));
    const auto beganUtc=utcTimestamp();if(cloudHandshake)savePhase(output,"SOLVING");map.resetCounters();
    auto portfolio=runPriPortfolio(map,field,start,goal,options);const auto solveCounters=counterJson(map.counters(),portfolio.counters);
    if(cloudHandshake)savePhase(output,"SERIALIZING");const auto serializationStart=Clock::now();
    bj::object effective{{"planner_class","ompl::geometric::BITstar"},{"motion_objective","surface_length"},{"edge_lower_bound","d_phi"},
        {"cost_to_go_lower_bound","d_phi"},{"informed_sampling","I_phi"},{"informed_cost_bound","episode_valid_surface_length_incumbent"},
        {"k_nearest",true},{"jit",false},{"pruning",true},{"samples_per_batch",100},{"batch_after_first_solution",200},{"safety_atlas_sampling",true},
        {"nearest_neighbor_metric","euclidean_xy"},{"external_length_cap_in_sampler",false},{"anchor_injected_as_incumbent",false}};
    bj::object result{{"schema_version",1},{"backend_id","a3t_pri_portfolio_v1"},{"planner_id",request.at("variant_id")},{"backend_planner_id","bitstar_a3"},
        {"ompl_version",OMPL_VERSION},{"compiler",__VERSION__},{"state_dimension",2},{"objective_type","surface_length"},
        {"requested_policy","d_phi/I_phi"},{"requested_pri_portfolio",registered},{"requested_safe_length",safety},
        {"effective_pri_portfolio",{{"length_cap_ratio",options.length_cap_ratio},{"anchor_fraction",.25},{"episode_fraction",.25},{"maximum_mask_episodes",3},
            {"selection","exact_pri_then_surface_length_then_episode_then_callback"},{"cap_source","frozen_shortest_anchor"},{"sem_reference_used",false},
            {"mask_thresholds","max(anchor_PRI,linear_quantile_90_or_75)"},{"fresh_planner_per_mask",true},{"deadline_after_scoring",true}}},
        {"effective_safe_length",{{"policy_id","a3t_safe_length_v1"},{"risk_cap",40.},{"slope_cap_deg",35.},{"atlas_sampling",true},{"thresholds_immutable_during_solve",true}}},
        {"effective_policy",effective},{"planner_status",portfolio.status},{"execution_status","COMPLETED"},{"portfolio_stage",portfolio.stage},
        {"termination_reason",options.work_iterations_per_episode?"WORK_OR_TIME_BUDGET":"TIME_BUDGET_OR_NATIVE_STOP"},
        {"time_budget_s",options.budget_s},{"actual_solve_wall_s",portfolio.actual_wall_s},{"solve_started_utc",beganUtc},
        {"preprocessing_s",preprocessing},{"load_preprocess_s",preprocessing},{"mask_build_s",portfolio.mask_build_s},
        {"timing_origin","before_initial_policy_and_planner_creation"},{"work_iterations_per_episode",options.work_iterations_per_episode},
        {"counters",solveCounters},{"path_available",portfolio.selected.has_value()},
        {"length_cap_m",portfolio.length_cap?number(*portfolio.length_cap):bj::value(nullptr)},
        {"anchor_snapshot_id",portfolio.anchor?bj::value("candidate_"+std::to_string(*portfolio.anchor)):bj::value(nullptr)},
        {"selected_snapshot_id",portfolio.selected?bj::value("candidate_"+std::to_string(*portfolio.selected)):bj::value(nullptr)},
        {"candidate_archive","portfolio_candidates.jsonl"},{"mask_archive","portfolio_masks.json"},{"best_cost_terrain_m",nullptr},{"length_xy_m",nullptr}};
    bj::array masks;
    for(const auto &m:portfolio.masks){bj::array excluded;for(auto id:m.excluded)excluded.push_back(id);
        bj::value maskFile=nullptr;if(!m.allowed.empty()){fs::create_directories(output/"masks");const auto name="masks/mask_"+std::to_string(m.episode_id)+".bin";
            std::ofstream out(output/name,std::ios::binary);for(int b:m.allowed){const char value=static_cast<char>(b);out.write(&value,1);}if(!out)throw std::runtime_error("cannot write portfolio mask");maskFile=name;}
        masks.push_back(bj::object{{"episode_id",m.episode_id},{"recipe",m.recipe},{"threshold",m.threshold},{"status",m.status},
            {"mask_sha256",m.sha256.empty()?bj::value(nullptr):bj::value(m.sha256)},{"excluded_cells",excluded},{"mask_file",maskFile}});
    }
    saveJson(output/"portfolio_masks.json",bj::object{{"nx",field->nx()},{"ny",field->ny()},{"encoding","uint8_row_major_allowed"},{"masks",masks}});
    bj::array episodes;bool called=false;
    for(const auto &e:portfolio.episodes){called=called||e.solve_called;episodes.push_back(bj::object{{"episode_id",e.episode_id},{"planner_instance_id",e.planner_instance_id},
        {"fresh_planner",e.fresh_planner},{"anchor_injected",false},{"solve_called",e.solve_called},{"mask_sha256",e.mask_sha256},
        {"began_s",e.began_s},{"setup_completed_s",e.setup_completed_s},{"finished_s",e.finished_s},{"iterations",e.iterations},
        {"batch_transition_count",e.batch_transitions},{"status",e.status}});}
    result["episodes"]=episodes;result["planner_solve_called"]=called;
    bj::array snapshots;std::ofstream archive(output/"portfolio_candidates.jsonl"),progress(output/"progress.jsonl");
    for(const auto &c:portfolio.candidates){const auto id="candidate_"+std::to_string(c.id);savePath(output/"snapshots"/id,c.path,map);
        bj::array xy;for(const auto p:c.path)xy.push_back(bj::array{p.x,p.y});
        double exposure=0.;bool exposureAvailable=c.stats.valid;
        if(exposureAvailable)for(std::size_t i=0;i<std::max<std::size_t>(1,c.path.size()-1);++i){const auto integral=semExposureCanonical(map,*field,c.path[i],c.path.size()>1?c.path[i+1]:c.path[i]);
            if(!integral){exposureAvailable=false;break;}exposure+=*integral;}
        bj::object value{{"snapshot_id",id},{"event_type","improvement"},{"event_origin","portfolio_registered_selection"},{"elapsed_s",c.score_completed_s},
            {"callback_s",c.callback_s},{"score_completed_s",c.score_completed_s},{"episode_id",c.episode_id},{"callback_id",c.callback_id},
            {"anchor_phase",c.anchor_phase},{"eligible",c.eligible},{"published",c.published},{"reason",c.reason},{"feasible",c.stats.valid},
            {"cost_terrain_m",number(c.stats.length)},{"best_cost_terrain_m",number(c.stats.length)},{"length_xy_m",number(c.stats.length_xy)},
            {"objective_value",number(c.stats.length)},{"pri",number(c.stats.pri)},{"risk_sum",number(c.stats.risk_sum)},{"risk_count",c.stats.cells.size()},
            {"path_xy",xy},{"sem_exposure_risk_m",exposureAvailable?number(exposure):bj::value(nullptr)},
            {"sem_exposure_normalized_m",exposureAvailable?number(exposure/40.):bj::value(nullptr)}};
        archive<<bj::serialize(value)<<'\n';
        if(c.published){snapshots.push_back(value);progress<<bj::serialize(value)<<'\n';}
        if(portfolio.selected&&*portfolio.selected==c.id){savePath(output,c.path,map);for(const char *key:{"cost_terrain_m","best_cost_terrain_m","length_xy_m","objective_value","pri","risk_sum","risk_count","sem_exposure_risk_m","sem_exposure_normalized_m"})result[key]=value.at(key);}
    }
    if(!archive||!progress)throw std::runtime_error("cannot write PRI portfolio trace");
    result["snapshots"]=snapshots;result["candidate_count"]=portfolio.candidates.size();result["serialize_s"]=seconds(serializationStart);
    saveJson(output/"worker.json",result);if(cloudHandshake)savePhase(output,"FINISHED");return 0;
}
#include "PriPathSearchWorker.inc"
#include "RpBitWorker.inc"
#include "RpRefineReplayWorker.inc"
#include "RpBitA5Worker.inc"
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("usage: terrain_benchmark REQUEST.json OUTPUT_DIRECTORY");
        std::ifstream input(argv[1]);std::stringstream buffer;buffer<<input.rdbuf();
        auto request=bj::parse(buffer.str()).as_object();
        fs::path output(argv[2]);fs::create_directories(output);
        const bool cloudHandshake=request.if_contains("cloud_phase_handshake") &&
            bj::value_to<bool>(request.at("cloud_phase_handshake"));
        if(cloudHandshake)savePhase(output,"SETUP");
        auto logname=(output/"ompl.log").string();ompl::msg::OutputHandlerFile omplLog(logname.c_str());
        ompl::msg::useOutputHandler(&omplLog);ompl::msg::setLogLevel(ompl::msg::LOG_INFO);
        if(request.if_contains("rp_refine_replay"))return runRpRefineReplayWorker(request,output,cloudHandshake);
        if(request.if_contains("rp_bit_a5"))return runRpBitA5Worker(request,output,cloudHandshake);
        if(request.if_contains("rp_bit"))return runRpBitWorker(request,output,cloudHandshake);
        if(request.if_contains("pri_pathsearch"))return runPriPathSearchWorker(request,output,cloudHandshake);
        if(request.if_contains("pri_portfolio"))return runPriPortfolioWorker(request,output,cloudHandshake);
        const std::string planner=bj::value_to<std::string>(request.at("planner_id"));
        const std::string variant=request.if_contains("variant_id")?
            bj::value_to<std::string>(request.at("variant_id")):planner;
        PlannerOptions options;
        const bool safeLength=request.if_contains("safe_length")!=nullptr;
        options.state_cache=request.if_contains("state_cache") && bj::value_to<bool>(request.at("state_cache"));
        options.scalar_cost=request.if_contains("scalar_cost") && bj::value_to<bool>(request.at("scalar_cost"));
        options.sampler_query_cache=request.if_contains("sampler_query_cache") && bj::value_to<bool>(request.at("sampler_query_cache"));
        const bool exactFlatCertificate=request.if_contains("exact_flat_certificate") && bj::value_to<bool>(request.at("exact_flat_certificate"));
        const std::string buildPolicy=request.if_contains("traversability_build_policy")?
            bj::value_to<std::string>(request.at("traversability_build_policy")):"legacy";
        if(buildPolicy!="legacy" && buildPolicy!="skip_dominated")throw std::runtime_error("unknown traversability build policy");
        options.batch_after_first_solution=request.if_contains("batch_after_first_solution")?
            bj::value_to<unsigned int>(request.at("batch_after_first_solution")):0;
        const bool collisionIndex=request.if_contains("collision_index") && bj::value_to<bool>(request.at("collision_index"));
        if(variant.empty() || ((options.state_cache || options.scalar_cost || collisionIndex || options.sampler_query_cache ||
            exactFlatCertificate || safeLength || buildPolicy!="legacy" ||
            options.batch_after_first_solution) && variant==planner))
            throw std::runtime_error("optimized implementation requires a distinct variant_id");
        if(options.batch_after_first_solution && planner!="bitstar_a3")
            throw std::runtime_error("after-first-solution batch policy requires BITstar A3");
        if(options.batch_after_first_solution && options.batch_after_first_solution==
            bj::value_to<unsigned int>(request.at("samples_per_batch")))
            throw std::runtime_error("after-first-solution batch must differ from initial batch");
        if(planner!="bitstar_a0"&&planner!="bitstar_a3"&&planner!="abitstar"&&planner!="aitstar"&&
           planner!="lazyprmstar_hauser2015"&&
           planner!="sem_rrtstar_common_v1"&&planner!="sem_rrtstar_common_t_v1"&&
           planner!="rrtstar"&&planner!="informed_rrtstar")
            throw std::runtime_error("unsupported planner");
        const bool traversability=request.if_contains("traversability")!=nullptr;
        const bool semExposure=request.if_contains("sem_exposure")!=nullptr;
        if(semExposure)options.sem_exposure=parseSemExposure(request.at("sem_exposure").as_object());
        const bool compositeObjective=traversability||semExposure;
        if((options.sampler_query_cache || exactFlatCertificate) &&
           ((!traversability && !safeLength) || planner!="bitstar_a3"))
            throw std::runtime_error("A3-T performance policies require traversability BITstar A3");
        if(safeLength && (traversability || exactFlatCertificate ||
            (planner!="bitstar_a3" && planner!="sem_rrtstar_common_v1")))
            throw std::runtime_error("safe length requires BITstar A3 or matched SEM, surface length and no flat certificate");
        if(buildPolicy!="legacy" && (!traversability ||
           (planner!="bitstar_a3" && planner!="sem_rrtstar_common_t_v1")))
            throw std::runtime_error("optimized field builder requires a supported traversability variant");
        const std::string objectiveType=request.if_contains("objective_type")?
            bj::value_to<std::string>(request.at("objective_type")):"surface_length";
        if(semExposure && (!safeLength || traversability || exactFlatCertificate || planner!="bitstar_a3" ||
            variant==planner || bj::value_to<unsigned int>(request.at("samples_per_batch"))!=100 ||
            options.batch_after_first_solution!=200))
            throw std::runtime_error("SEM exposure requires distinct safe A3, batch100-to-200 and no traversability/certificate");
        if(objectiveType!=(semExposure?"surface_length_sem_exposure":traversability?"surface_length_traversability":"surface_length"))
            throw std::runtime_error("objective_type and traversability config mismatch");
        if(traversability && ((planner!="bitstar_a0" && planner!="bitstar_a3" && planner!="sem_rrtstar_common_t_v1") || variant==planner ||
            options.batch_after_first_solution))
            throw std::runtime_error("traversability v1 requires distinct BITstar A0/A3 variant and fixed batch");
        bool lifted=planner=="bitstar_a3"||planner=="informed_rrtstar";
        const bool lazy=planner=="lazyprmstar_hauser2015";
        const bool sem=planner=="sem_rrtstar_common_v1"||planner=="sem_rrtstar_common_t_v1";
        const bool uniformRrt=planner=="rrtstar";
        const bool uniform=uniformRrt||lazy;
        if(sem) {
            auto requireLockedDouble=[&](const char *field,double expected) {
                const auto *value=request.if_contains(field);
                if(value==nullptr)throw std::runtime_error(std::string("missing locked SEM field: ")+field);
                const double actual=bj::value_to<double>(*value);
                if(!std::isfinite(actual)||std::abs(actual-expected)>1e-12)
                    throw std::runtime_error(std::string("SEM field does not match locked implementation: ")+field);
            };
            requireLockedDouble("sem_range_m",SemRRTstarCommon::STEER_RANGE_M);
            requireLockedDouble("sem_neighbor_radius_m",SemRRTstarCommon::NEAR_RADIUS_M);
        }
        auto getPoint=[&](const char *name){auto a=request.at(name).as_array();return Point{bj::value_to<double>(a[0]),bj::value_to<double>(a[1])};};
        Point start=getPoint("start"),goal=getPoint("goal");
        double budget=bj::value_to<double>(request.at("planning_budget_s"));
        if(!std::isfinite(budget)||budget<0)throw std::runtime_error("invalid time budget");
        auto beforeLoad=Clock::now();auto map=Terrain::load(bj::value_to<std::string>(request.at("map_path")));
        if(traversability)map.enableTraversability(parseTraversability(request.at("traversability").as_object()),
            buildPolicy=="legacy"?TraversabilityBuildPolicy::Legacy:TraversabilityBuildPolicy::SkipDominated);
        if(collisionIndex)map.enableCollisionIndex();
        double safetyPreprocess=0.;
        std::string safetyPolicyId="a3t_safe_length_v1";
        RiskCalibrationOptions calibrationOptions;
        RiskCalibrationResult calibration;
        CorridorOptions corridorOptions;
        CorridorResult corridor;
        if(safeLength) {
            const auto beganSafety=Clock::now();
            const auto &policy=request.at("safe_length").as_object();
            for(const auto &entry:policy) {
                const std::string key(entry.key());
                if(key!="field_path" && key!="risk_cap" && key!="slope_cap_deg" && key!="atlas_sampling" &&
                   key!="field_sha256" && key!="policy_id" && key!="cloud_sha256" && key!="builder_source_sha256" &&
                   key!="risk_slack" && key!="corridor_policy")
                    throw std::runtime_error("unknown safe_length field: "+key);
            }
            if(const auto *id=policy.if_contains("policy_id"))safetyPolicyId=bj::value_to<std::string>(*id);
            if(semExposure && (safetyPolicyId!="a3t_safe_length_v1" ||
                bj::value_to<double>(policy.at("risk_cap"))!=40. ||
                bj::value_to<double>(policy.at("slope_cap_deg"))!=35. ||
                !bj::value_to<bool>(policy.at("atlas_sampling")) || policy.if_contains("corridor_policy")))
                throw std::runtime_error("SEM exposure requires full safe-length-v1 cap40/slope35 atlas domain without corridor");
            if(safetyPolicyId!="a3t_safe_length_v1" && safetyPolicyId!="a3t_safe_length_v2" &&
               safetyPolicyId!="a3t_corridor_v3" && safetyPolicyId!="a3t_corridor_v4")
                throw std::runtime_error("unsupported safety policy_id");
            const bool calibrated=safetyPolicyId=="a3t_safe_length_v2";
            const bool corridorV4=safetyPolicyId=="a3t_corridor_v4";
            const bool corridorPolicy=safetyPolicyId=="a3t_corridor_v3"||corridorV4;
            if(corridorPolicy!=bool(policy.if_contains("corridor_policy")))
                throw std::runtime_error("corridor_policy requires a3t_corridor_v3 or a3t_corridor_v4");
            if((calibrated && (policy.if_contains("risk_cap") || !policy.if_contains("risk_slack"))) ||
               (!calibrated && policy.if_contains("risk_slack")))
                throw std::runtime_error("safety policy requires either v1 risk_cap or v2 risk_slack");
            auto field=std::make_shared<const SafetyField>(SafetyField::load(bj::value_to<std::string>(policy.at("field_path"))));
            if(corridorPolicy) {
                if(bj::value_to<double>(policy.at("risk_cap"))!=40. ||
                   bj::value_to<double>(policy.at("slope_cap_deg"))!=35.)
                    throw std::runtime_error("corridor v3 requires risk40 and slope35");
                const auto &config=policy.at("corridor_policy").as_object();
                for(const auto &entry:config) {
                    const std::string key(entry.key());
                    if(key!="alpha" && key!="label_bins" && key!="dilation_cells" &&
                       key!="max_labels" && key!="max_memory_bytes" &&
                       !(corridorV4&&(key=="selector"||key=="representatives_per_bin"||
                           key=="max_preview_candidates"||key=="max_preview_edge_checks"||key=="beam_scheduler")))
                        throw std::runtime_error("unknown corridor policy field: "+key);
                }
                corridorOptions.alpha=bj::value_to<double>(config.at("alpha"));
                corridorOptions.label_bins=bj::value_to<unsigned int>(config.at("label_bins"));
                corridorOptions.dilation_cells=bj::value_to<unsigned int>(config.at("dilation_cells"));
                corridorOptions.max_labels=bj::value_to<std::uint64_t>(config.at("max_labels"));
                corridorOptions.max_memory_bytes=bj::value_to<std::uint64_t>(config.at("max_memory_bytes"));
                if(corridorV4) {
                    corridorOptions.selector=bj::value_to<std::string>(config.at("selector"));
                    corridorOptions.representatives_per_bin=bj::value_to<unsigned>(config.at("representatives_per_bin"));
                    corridorOptions.max_preview_candidates=bj::value_to<unsigned>(config.at("max_preview_candidates"));
                    corridorOptions.max_preview_edge_checks=bj::value_to<std::uint64_t>(config.at("max_preview_edge_checks"));
                    if(const auto *scheduler=config.if_contains("beam_scheduler"))
                        corridorOptions.beam_scheduler=bj::value_to<std::string>(*scheduler);
                }
                if(corridorOptions.max_labels>2000000 || corridorOptions.max_memory_bytes>536870912)
                    throw std::runtime_error("corridor resources exceed registered ceilings");
                corridor=corridorV4?computeCorridorV4(map,*field,start,goal,corridorOptions):computeCorridor(map,*field,start,goal,corridorOptions);
                if(corridor.connected())options.safety=std::make_shared<const SafetyPolicy>(map,field,40.,35.,corridor.mask);
            } else if(calibrated) {
                calibrationOptions.risk_slack=bj::value_to<double>(policy.at("risk_slack"));
                if(bj::value_to<double>(policy.at("slope_cap_deg"))!=35.)
                    throw std::runtime_error("v2 calibration requires fixed slope cap 35 degrees");
                calibration=calibrateRisk(map,*field,start,goal,calibrationOptions);
                if(calibration.connected())options.safety=std::make_shared<const SafetyPolicy>(map,field,
                    *calibration.effective_risk_cap,calibrationOptions.slope_cap_deg);
            } else options.safety=std::make_shared<const SafetyPolicy>(map,field,
                bj::value_to<double>(policy.at("risk_cap")),bj::value_to<double>(policy.at("slope_cap_deg")));
            options.safety_atlas_sampling=bj::value_to<bool>(policy.at("atlas_sampling"));
            if(options.safety_atlas_sampling && planner!="bitstar_a3")
                throw std::runtime_error("matched SEM control does not support the A3 atlas sampler");
            safetyPreprocess=seconds(beganSafety);
        }
        double loadTime=seconds(beforeLoad);
        ompl::RNG::setSeed(bj::value_to<unsigned int>(request.at("planner_seed")));
        bj::object result{{"schema_version",2},{"planner_id",variant},{"backend_planner_id",planner},
            {"variant_id",variant},{"ompl_version",OMPL_VERSION},
            {"requested_optimizations",{{"state_cache",options.state_cache},{"scalar_cost",options.scalar_cost},
                {"sampler_query_cache",options.sampler_query_cache},{"traversability_build_policy",buildPolicy},
                {"exact_flat_certificate",exactFlatCertificate},
                {"collision_index",collisionIndex},{"batch_after_first_solution",options.batch_after_first_solution}}},
            {"compiler",__VERSION__},{"state_dimension",2},{"requested_policy",lazy?"d_xy/D":sem?"d_xy/D_then_E_xy":uniformRrt?"d_phi/D":lifted?"d_phi/I_phi":"d_xy/E_xy"},
            {"effective_policy",{{"edge_lower_bound",lifted?"d_phi":"d_xy"},
                {"start_lower_bound",lifted?"d_phi":"d_xy"},{"goal_lower_bound",lifted?"d_phi":"d_xy"},
                {"sample_domain",sem?"D_then_E_xy":uniform?"D":lifted?"I_phi":"E_xy"},{"nearest_neighbor_metric","euclidean_xy"},
                {"informed_sampling",!uniform},{"pruning",sem?false:!uniform},{"k_nearest",!sem},{"jit",false},{"has_exact_informed_measure",false},
                {"informed_measure_fallback","full_domain_upper_bound"},{"simplification",false}}},
            {"time_budget_s",budget},{"load_preprocess_s",loadTime},
            {"load_s",loadTime-map.preprocessingSeconds()-map.collisionIndexBuildSeconds()-map.traversabilityPreprocessingSeconds()-safetyPreprocess},
            {"preprocessing_s",map.preprocessingSeconds()+map.collisionIndexBuildSeconds()+map.traversabilityPreprocessingSeconds()+safetyPreprocess},
            {"collision_index_build_s",map.collisionIndexBuildSeconds()},{"path_available",false},
            {"independent_valid",nullptr},{"success",false},{"best_cost_terrain_m",nullptr},{"length_xy_m",nullptr},
            {"first_solution_time_s",nullptr},{"peak_rss_mb",nullptr}};
        if(safeLength) {
            result["objective_type"]="surface_length";
            result["requested_safe_length"]=request.at("safe_length");
            result["effective_safe_length"]={{"policy_id",safetyPolicyId},
                {"risk_cap",options.safety?number(options.safety->riskCap()):bj::value(nullptr)},
                {"slope_cap_deg",options.safety?options.safety->slopeCap():calibrationOptions.slope_cap_deg},
                {"atlas_sampling",options.safety_atlas_sampling},
                {"atlas_proposal_area_m2",options.safety?number(options.safety->atlasArea()):bj::value(nullptr)},
                {"risk_rasterization","all_closed_cells_touched_by_centerline_16ulp_boundary_ties"},
                {"slope_policy","maximum_all_triangles_touched_by_swept_footprint"},
                {"thresholds_immutable_during_solve",true}};
            result["safety_preprocessing_s"]=safetyPreprocess;
            if(safetyPolicyId=="a3t_safe_length_v2") {
                result["calibration"]=calibrationJson(calibration,calibrationOptions,output);
                result["calibration_preprocessing_s"]=calibration.preprocessing_s;
                result["planner_solve_called"]=false;
                result["effective_safe_length"].as_object()["witness_used_as_incumbent"]=false;
            }
            result["first_safe_solution_time_s"]=nullptr;
            if(safetyPolicyId=="a3t_corridor_v3"||safetyPolicyId=="a3t_corridor_v4") {
                result["corridor"]=corridorJson(corridor,corridorOptions,output,safetyPolicyId=="a3t_corridor_v4");
                result["corridor_preprocessing_s"]=corridor.total_s;
                result["planner_solve_called"]=false;
                result["effective_safe_length"].as_object()["guide_used_as_incumbent"]=false;
                result["effective_safe_length"].as_object()["domain"]="immutable_centerline_corridor";
            }
            auto &policy=result["effective_policy"].as_object();
            policy["motion_objective"]="surface_length";policy["informed_cost_bound"]="valid_surface_length_L_best";
            policy["safety_gate"]="state_and_motion_before_tree_acceptance";
            policy["variant"]=safetyPolicyId;
            policy["safety_atlas_sampling"]=options.safety_atlas_sampling;
        }
        if(traversability) {
            result["objective_type"]=objectiveType;
            result["requested_traversability"]=request.at("traversability");
            result["effective_traversability"]=traversabilityJson(map.traversabilityConfig());
            result["traversability_preprocessing_s"]=map.traversabilityPreprocessingSeconds();
            result["traversability_build_stats"]=buildStatsJson(map.traversabilityBuildStats());
            result["risk_interpretation"]="normalized_geometric_exposure_not_accident_probability";
            result["feature_boundary_policy"]="componentwise_max_incident_cells";
            result["clearance_cost_obstacles"]="explicit_rectangles_only_not_generic_mask";
            for(const char *key:{"objective_value","cost_terrain_m","risk_slope_integral_m",
                "risk_clearance_integral_m","risk_integral_m"})result[key]=nullptr;
            result["effective_policy"].as_object()["informed_cost_bound"]="objective_value_J";
            result["effective_policy"].as_object()["motion_objective"]="surface_length_traversability";
        }
        if(semExposure) {
            result["objective_type"]=objectiveType;
            result["requested_sem_exposure"]=request.at("sem_exposure");
            result["effective_sem_exposure"]={{"lambda",options.sem_exposure->lambda},
                {"risk_scale",options.sem_exposure->risk_scale},
                {"integral","piecewise_constant_SEM_risk_ds_surface"},
                {"density_version","exact_canonical_raster_v1"},
                {"boundary_policy","max_incident_supported_cells_on_positive_length_intervals"},
                {"risk_interpretation","additive_SEM_exposure_proxy_not_unique_cell_PRI"}};
            for(const char *key:{"objective_value","cost_terrain_m","sem_exposure_risk_m","sem_exposure_normalized_m"})
                result[key]=nullptr;
            result["effective_policy"].as_object()["motion_objective"]=objectiveType;
            result["effective_policy"].as_object()["informed_cost_bound"]="objective_value_J";
            result["effective_safe_length"].as_object()["domain"]="full_safe_cap40_no_corridor";
        }
        PolicyCounters counters;
        // Even an invalid query records an explicitly unclaimed certificate.
        result["exact_certificate"]={{"policy","flat_direct_v1"},{"enabled",exactFlatCertificate},
            {"certified",false},{"triggered_termination",false},{"snapshot_id",nullptr},
            {"check_wall_s",0.},{"checks",0},{"conditions",certificateConditions({})},
            {"reason","NOT_APPLICABLE"},{"proof_scope","declared_binary64_input_heightfield_model_not_all_machine_roundoff"}};
        if(safeLength && ((safetyPolicyId=="a3t_safe_length_v2" && !calibration.connected()) ||
                         ((safetyPolicyId=="a3t_corridor_v3"||safetyPolicyId=="a3t_corridor_v4") && !corridor.connected()))) {
            const auto &status=safetyPolicyId!="a3t_safe_length_v2"?corridor.status:calibration.status;
            result["planner_status"]=status=="INVALID_START"?"INVALID_START":
                status=="INVALID_GOAL"?"INVALID_GOAL":"NO_SOLUTION";
            result["execution_status"]="COMPLETED";
            result["termination_reason"]=status;
            result["actual_solve_wall_s"]=0.;
            result["setup_s"]=0.;
            map.resetCounters();result["counters"]=counterJson(map.counters(),counters);
            saveJson(output/"worker.json",result);std::ofstream(output/"progress.jsonl");
            if(cloudHandshake)savePhase(output,"FINISHED");
            ompl::msg::restorePreviousOutputHandler();return 0;
        }
        const bool validStart=options.safety?options.safety->validState(start):map.validState(start);
        const bool validGoal=options.safety?options.safety->validState(goal):map.validState(goal);
        if(!validStart||!validGoal) {
            result["planner_status"]=!validStart?"INVALID_START":"INVALID_GOAL";
            if(options.safety)result["invalid_query_reason"]=options.safety->rejectionReason(!validStart?start:goal,!validStart?start:goal);
            result["execution_status"]="COMPLETED";result["termination_reason"]="INVALID_QUERY";
            result["actual_solve_wall_s"]=0;result["counters"]=counterJson(map.counters(),counters);
            saveJson(output/"worker.json",result);std::ofstream(output/"progress.jsonl");
            if(cloudHandshake)savePhase(output,"FINISHED");
            ompl::msg::restorePreviousOutputHandler();return 0;
        }
        auto beforeSetup=Clock::now();auto problem=makePlannerProblem(map,planner,start,goal,counters,
            bj::value_to<unsigned int>(request.at("samples_per_batch")),bj::value_to<double>(request.at("rewire_factor")),
            bj::value_to<unsigned int>(request.at("sample_attempts")),options);
        result["setup_s"]=seconds(beforeSetup);
        // Parent-object insertions during solve setup can relocate its values.
        // Keep a value here; never hold a nested reference across those inserts.
        bj::object effective=result["effective_policy"].as_object();
        effective["state_cache"]=problem.objective->stateCacheEnabled();
        effective["scalar_cost"]=problem.objective->scalarCostEnabled();
        effective["collision_index"]=map.collisionIndexEnabled();
        effective["sampler_query_cache"]=problem.objective->samplerQueryCacheEnabled();
        effective["traversability_build_policy"]=map.traversabilityBuildPolicy()==TraversabilityBuildPolicy::Legacy?"legacy":"skip_dominated";
        effective["exact_flat_certificate"]=exactFlatCertificate;
        effective["batch_policy"]=options.batch_after_first_solution?"after_first_solution":"fixed";
        effective["batch_after_first_solution"]=problem.batch_after_first_solution;
        if(problem.objective->stateCacheEnabled()!=options.state_cache ||
           problem.objective->scalarCostEnabled()!=options.scalar_cost ||
           problem.objective->samplerQueryCacheEnabled()!=options.sampler_query_cache ||
           bj::value_to<std::string>(effective.at("traversability_build_policy"))!=buildPolicy ||
           map.collisionIndexEnabled()!=collisionIndex ||
           problem.batch_after_first_solution!=options.batch_after_first_solution)
            throw std::runtime_error("requested implementation variant is not effective");
        if(bool(problem.objective->semExposureConfig())!=semExposure ||
           (semExposure && (problem.objective->semExposureConfig()->lambda!=options.sem_exposure->lambda ||
                            problem.objective->semExposureConfig()->risk_scale!=options.sem_exposure->risk_scale)))
            throw std::runtime_error("requested SEM exposure objective is not effective");
        const bool nativeEuclideanSampler=static_cast<bool>(problem.abit)||static_cast<bool>(problem.ait);
        effective["has_exact_informed_measure"]=nativeEuclideanSampler;
        effective["informed_measure_fallback"]=nativeEuclideanSampler?
            bj::value("OMPL_PathLengthDirectInfSampler_E_xy_measure"):uniform?
            bj::value("not_applicable_uniform_sampling"):bj::value("full_domain_upper_bound");
        effective["sampler_class"]=nativeEuclideanSampler?
            bj::value("ompl::base::PathLengthDirectInfSampler"):uniform?
            bj::value("ompl::base::StateSampler_uniform_xy"):bj::value("terrain25d::TerrainInformedSampler");
        effective["samples_per_batch"]=problem.bit?bj::value(problem.bit->getSamplesPerBatch()):
            problem.ait?bj::value(problem.ait->getBatchSize()):bj::value(nullptr);
        effective["rewire_factor"]=problem.bit?problem.bit->getRewireFactor():
            problem.ait?problem.ait->getRewireFactor():problem.lazy?bj::value(1.1):problem.sem?bj::value(nullptr):bj::value(problem.rrt->getRewireFactor());
        effective["planner_class"]=problem.abit?"ompl::geometric::ABITstar":problem.ait?
            "ompl::geometric::AITstar":problem.lazy?"terrain25d::HauserLazyPRMstar":problem.sem?"terrain25d::SemRRTstarCommon":problem.bit?"ompl::geometric::BITstar":planner=="rrtstar"?
            "ompl::geometric::RRTstar":"ompl::geometric::InformedRRTstar";
        effective["range_m"]=problem.rrt?bj::value(problem.rrt->getRange()):problem.sem?bj::value(SemRRTstarCommon::STEER_RANGE_M):bj::value(nullptr);
        effective["range_policy"]=problem.rrt?bj::value("OMPL_auto_0.2_maximum_extent"):problem.sem?bj::value("fixed_steer_range_1m"):bj::value(nullptr);
        effective["goal_bias"]=problem.rrt?bj::value(problem.rrt->getGoalBias()):problem.sem?bj::value(SemRRTstarCommon::GOAL_BIAS):bj::value(nullptr);
        effective["delay_collision_checking"]=problem.rrt?bj::value(problem.rrt->getDelayCC()):bj::value(nullptr);
        effective["pruning"]=problem.bit?problem.bit->getPruning():problem.ait?
            problem.ait->isPruningEnabled():(problem.lazy||problem.sem)?false:problem.rrt->getTreePruning();
        effective["prune_threshold"]=problem.bit?bj::value(problem.bit->getPruneThresholdFraction()):
            problem.rrt?bj::value(problem.rrt->getPruneThreshold()):bj::value(nullptr);
        effective["pruned_measure"]=problem.rrt?bj::value(problem.rrt->getPrunedMeasure()):bj::value(nullptr);
        effective["informed_sampling"]=static_cast<bool>(problem.bit)||static_cast<bool>(problem.ait)||
            static_cast<bool>(problem.sem)||(problem.rrt&&problem.rrt->getInformedSampling());
        effective["sample_domain"]=sem?"D_then_E_xy":uniform?"D":lifted?"I_phi":"E_xy";
        effective["sample_rejection"]=problem.rrt?bj::value(problem.rrt->getSampleRejection()):bj::value(nullptr);
        effective["new_state_rejection"]=problem.rrt?bj::value(problem.rrt->getNewStateRejection()):bj::value(nullptr);
        effective["admissible_cost_to_come"]=problem.rrt?bj::value(problem.rrt->getAdmissibleCostToCome()):bj::value(nullptr);
        effective["ordered_sampling"]=problem.rrt?bj::value(problem.rrt->getOrderedSampling()):bj::value(nullptr);
        if(problem.lazy) {
            effective["sample_attempts"]=nullptr;
            effective["sample_attempts_status"]="NOT_APPLICABLE_ONE_UNIFORM_PROPOSAL_PER_ITERATION";
        } else {
            effective["sample_attempts"]=bj::value_to<unsigned int>(request.at("sample_attempts"));
        }
        effective["edge_cost_cache"]=false;effective["threads"]=1;
        effective["stopping_policy"]=exactFlatCertificate?"flat_direct_v1_or_deadline_or_native_planner_stop":
            "deadline_upper_budget_or_native_planner_stop";
        effective["motion_cost_nodata_contract"]="positive_infinity_internal_json_null";
        if(problem.rrt)effective["jit"]=nullptr;
        if(problem.abit) {
            effective["initial_inflation_factor"]=problem.abit->getInitialInflationFactor();
            effective["inflation_scaling_parameter"]=problem.abit->getInflationScalingParameter();
            effective["truncation_scaling_parameter"]=problem.abit->getTruncationScalingParameter();
            effective["algorithm_profile"]="ABITstar_paper_schedule_matched_controls";
        }
        if(problem.ait) {
            effective["track_approximate_solutions"]=problem.ait->areApproximateSolutionsTracked();
            effective["max_number_of_goals"]=problem.ait->getMaxNumberOfGoals();
            effective["algorithm_profile"]="AITstar_paper_core_patched_matched_controls";
        }
        if(problem.lazy) {
            effective["connection_rule"]="floor(1.1*e*(1+1/2)*ln(n)), clamped to n-1";
            effective["edge_selector"]="forward_first_invalid";
            effective["shortest_path_update"]="Hauser_2015_dynamic_c_pi_decrease_increase_propagation";
            effective["algorithm_profile"]="LazyPRMstar_Hauser_2015_Algorithms_1_2_5_core";
            effective["bidirectional_path_precheck"]=false;
            effective["section_III_E_roadmap_pruning"]=false;
            effective["implementation_scope_note"]=
                "paper_core_with_dynamic_c_pi_updates;optional_bidirectional_precheck_and_section_III_E_pruning_omitted";
        }
        if(problem.sem) {
            effective["algorithm_profile"]="SEM_RRTstar_common_v1_controlled_reproduction";
            effective["official_sem_rrtstar_implementation"]=false;
            effective["tree_direction"]="goal_to_start";
            effective["solution_output_order"]="start_to_goal";
            effective["sample_domain_before_incumbent"]="uniform_D";
            effective["sample_domain_after_incumbent"]="d_xy_E_xy";
            effective["steer_range_m"]=SemRRTstarCommon::STEER_RANGE_M;
            effective["near_radius_m"]=SemRRTstarCommon::NEAR_RADIUS_M;
            effective["rewire_radius_m"]=SemRRTstarCommon::NEAR_RADIUS_M;
            effective["exact_start_connection_radius_m"]=SemRRTstarCommon::START_CONNECTION_RADIUS_M;
            effective["goal_bias"]=SemRRTstarCommon::GOAL_BIAS;
            effective["sample_attempts"]=SemRRTstarCommon::INFORMED_ATTEMPTS;
            effective["k_nearest"]=false;
            effective["deterministic_tie_break"]="lowest_insertion_id";
            effective["descendant_cost_propagation"]=true;
            effective["risk_assessment_enabled"]=false;
            effective["turn_constraint_enabled"]=false;
            effective["rolling_constraint_enabled"]=false;
            effective["comparison_scope_note"]="common_xy_surface_length_core_not_official_SEM_RRTstar";
        }
        auto currentCounters=[&]() {
            auto value=counterJson(map.counters(),counters);
            if(problem.lazy) {
                value["sample_proposals"]=problem.lazy->numSampleProposals();
                value["samples_returned_before_validity"]=problem.lazy->numSampleProposals();
                value["uniform_state_rejections"]=problem.lazy->numStateRejections();
                value["uniform_states_accepted"]=problem.lazy->numSampleProposals()-problem.lazy->numStateRejections();
            }
            return value;
        };
        std::vector<Event> events;std::vector<bj::object> progress;bj::array batchTransitions;
        FlatDirectCertificate certificate;
        bool certificateTriggered=false;
        std::size_t certificateSnapshot=0;
        std::uint64_t certificateChecks=0;
        double certificateTime=0;
        Clock::time_point began;double best=INFINITY,lastProgress=-INFINITY,callbackTime=0.;
        problem.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &path,const ob::Cost cost){
            const auto entered=Clock::now();
            const double elapsed=std::chrono::duration<double>(entered-began).count();
            if(cost.value()>best+1e-8)throw std::runtime_error("incumbent increased");
            Event event{elapsed,cost.value(),problem.snapshot(path),currentCounters()};
            events.push_back(std::move(event));best=cost.value();
            if(exactFlatCertificate && !certificate.certified) {
                const auto checkBegan=Clock::now();++certificateChecks;
                certificate=certifyFlatDirect(map,events.back().path,start,goal);
                certificateTime+=seconds(checkBegan);
                if(certificate.certified)certificateSnapshot=events.size()-1;
            }
            if(problem.onFirstSolution())batchTransitions.push_back({{"elapsed_s",elapsed},
                {"after_batch",problem.bit->numBatches()},
                {"samples_per_batch",problem.bit->getSamplesPerBatch()},
                {"effective_from","next_native_batch"}});
            callbackTime+=seconds(entered);
        });
        const unsigned workLimit=request.if_contains("work_iterations")?bj::value_to<unsigned>(request.at("work_iterations")):0;
        const double interval=bj::value_to<double>(request.at("progress_interval_s"));
        auto ptc=ob::PlannerTerminationCondition([&](){
            double elapsed=seconds(began);
            if(elapsed-lastProgress>=interval){progress.push_back({{"event_type","poll"},{"elapsed_s",elapsed},
                {"feasible",std::isfinite(best)},{"best_cost_terrain_m",compositeObjective?bj::value(nullptr):number(best)},
                {"counters",currentCounters()},{"snapshot_id",nullptr}});lastProgress=elapsed;}
            if(certificate.certified) {certificateTriggered=true;return true;}
            return (workLimit && problem.iterations()>=workLimit) || elapsed>=budget;
        });
        map.resetCounters();auto setupPolicyCounters=counters;counters={};
        result["solve_started_utc"]=utcTimestamp();
        if(cloudHandshake)savePhase(output,"SOLVING");
        began=Clock::now();
        const bool identityQuery=problem.pdef->getGoal()->isSatisfied(problem.pdef->getStartState(0));
        ob::PlannerStatus status;
        if(identityQuery && !problem.rrt) {
            auto path=std::make_shared<ompl::geometric::PathGeometric>(problem.si);
            path->append(problem.pdef->getStartState(0));
            problem.pdef->addSolutionPath(path,false,0.,problem.planner->getName()+"_identity_query");
            const auto &callback=problem.pdef->getIntermediateSolutionCallback();
            if(callback)callback(problem.planner.get(),{problem.pdef->getStartState(0)},problem.objective->identityCost());
            status=ob::PlannerStatus::EXACT_SOLUTION;
        } else {
            if(safeLength && (safetyPolicyId=="a3t_safe_length_v2" || safetyPolicyId=="a3t_corridor_v3"||safetyPolicyId=="a3t_corridor_v4"))
                result["planner_solve_called"]=true;
            status=problem.planner->solve(ptc);
        }
        double solveTime=seconds(began);
        effective["samples_per_batch_final"]=problem.bit?bj::value(problem.bit->getSamplesPerBatch()):
            problem.ait?bj::value(problem.ait->getBatchSize()):bj::value(nullptr);
        result["effective_policy"]=std::move(effective);
        result["batch_transition_count"]=problem.batch_transition_count;
        result["batch_transitions"]=std::move(batchTransitions);
        if(cloudHandshake)savePhase(output,"SERIALIZING");
        auto solveCounters=currentCounters();
        if(nativeEuclideanSampler)markNativeSamplerCountersUnavailable(solveCounters);
        auto beforeSerialization=Clock::now();
        result["planner_status"]=status==ob::PlannerStatus::EXACT_SOLUTION?"EXACT_SOLUTION":
            status==ob::PlannerStatus::APPROXIMATE_SOLUTION?"APPROXIMATE_SOLUTION":"NO_SOLUTION";
        result["execution_status"]=status==ob::PlannerStatus::EXACT_SOLUTION?"COMPLETED":"TIMEOUT";
        const bool identity=identityQuery;
        result["termination_reason"]=identity?"ANALYTIC_IDENTITY_QUERY":
            certificateTriggered?"EXACT_FLAT_DIRECT_CERTIFICATE":
            workLimit && problem.iterations()>=workLimit?"WORK_BUDGET":solveTime>=budget?"TIME_BUDGET":
            problem.objective->isSatisfied(ob::Cost(best))?"OBJECTIVE_THRESHOLD":
            "UNKNOWN_NATIVE_PLANNER_STOP";
        result["exact_certificate"]={{"policy","flat_direct_v1"},{"enabled",exactFlatCertificate},
            {"certified",certificate.certified},{"triggered_termination",certificateTriggered},
            {"snapshot_id",certificate.certified?bj::value("snapshot_"+std::to_string(certificateSnapshot)):bj::value(nullptr)},
            {"check_wall_s",certificateTime},{"checks",certificateChecks},{"conditions",certificateConditions(certificate)},
            {"reason",certificate.reason},{"proof_scope","declared_binary64_input_heightfield_model_not_all_machine_roundoff"}};
        result["actual_solve_wall_s"]=solveTime;result["overshoot_s"]=std::max(0.,solveTime-budget);
        result["counters"]=solveCounters;result["setup_policy_counters"]=counterJson({},setupPolicyCounters);
        result["iterations"]=problem.iterations();result["batches"]=problem.bit?bj::value(problem.bit->numBatches()):bj::value(nullptr);
        if(problem.ait)result["planner_specific_counters"]={{"iterations",unsignedProgress(*problem.ait,"iterations INTEGER")},
            {"state_collision_checks",unsignedProgress(*problem.ait,"state collision checks INTEGER")},
            {"edge_collision_checks",unsignedProgress(*problem.ait,"edge collision checks INTEGER")},
            {"nearest_neighbor_calls",unsignedProgress(*problem.ait,"nearest neighbour calls INTEGER")}};
        if(problem.lazy)result["planner_specific_counters"]={{"iterations",problem.lazy->numIterations()},
            {"sample_proposals",problem.lazy->numSampleProposals()},
            {"state_rejections",problem.lazy->numStateRejections()},
            {"motion_checks",problem.lazy->numMotionChecks()},
            {"candidate_path_searches",problem.lazy->numShortestPathSearches()},
            {"invalid_edges_removed",problem.lazy->numInvalidEdges()},
            {"undefined_cost_edges_skipped",problem.lazy->numUndefinedCostEdges()},
            {"milestones",problem.lazy->milestoneCount()},{"active_edges",problem.lazy->edgeCount()}};
        if(problem.sem)result["planner_specific_counters"]={{"iterations",problem.sem->numIterations()},
            {"sample_calls",problem.sem->numSampleCalls()},{"tree_insertions",problem.sem->numTreeInsertions()},
            {"state_rejections",problem.sem->numStateRejections()},{"motion_checks",problem.sem->numMotionChecks()},
            {"rewires",problem.sem->numRewires()},{"descendant_cost_updates",problem.sem->numDescendantCostUpdates()},
            {"start_connection_checks",problem.sem->numStartConnectionChecks()},
            {"start_connections_valid",problem.sem->numStartConnectionsValid()},
            {"sampler_failures",problem.sem->numSamplerFailures()},{"vertices",problem.sem->vertexCount()}};
        result["improvement_count"]=events.size();result["callback_timestamp_policy"]="steady_clock_at_callback_entry";
        result["callback_overhead_s"]=callbackTime;
        result["snapshot_adapter"]=problem.bit?"BIT_family_reverse_including_actual_endpoints":problem.ait?
            "AITstar_start_to_goal_including_actual_endpoints":problem.lazy?
            "HauserLazyPRMstar_start_to_goal_including_actual_endpoints":problem.sem?
            "SEM_RRTstar_common_backward_tree_reversed_start_to_goal":"RRTstar_observe_actual_incumbent_validate_reverse_interiors";
        result["event_origin"]=identity?"analytic_identity_query":"ompl_intermediate_solution_callback";
        result["goal_biased_proposals"]=nullptr;
        result["goal_biased_proposals_unavailable_reason"]=problem.rrt?"not_exposed_by_pinned_OMPL":"not_applicable_explicit_goal_vertex";
        result["custom_sampler_acceptance_denominator"]=nativeEuclideanSampler?
            bj::value(nullptr):problem.lazy?bj::value("exact_uniform_xy_proposals_one_per_iteration"):
            bj::value("sample_proposals_all_phases_including_preincumbent_excludes_goal_bias_and_post_steer_validity");
        result["informed_acceptance_denominator"]=nativeEuclideanSampler?
            bj::value(nullptr):bj::value("finite_bound_sample_proposals_excludes_goal_bias_and_post_steer_validity");
        if(nativeEuclideanSampler)result["sampler_counter_unavailable_reason"]=
            "native_OMPL_PathLengthDirectInfSampler_has_no_proposal_counter_hook";
        result["improvement_count_includes_first_solution"]=true;
        std::uint64_t snapshotPoints=0;for(const auto &event:events)snapshotPoints+=event.path.size();
        result["callback_snapshot_points"]=snapshotPoints;
        result["num_prune_calls"]=problem.rrt?bj::value(problem.rrt->getTreePruning()&&!identity?events.size():0):problem.sem?bj::value(0):bj::value(nullptr);
        result["num_prune_calls_provenance"]=problem.rrt?"RRTstar_solve_510_once_before_each_native_solution_callback":problem.sem?"not_applicable_no_tree_pruning":"see_graph_prunings_from_native_log";
        result["states_removed"]=nullptr;result["branches_disconnected"]=nullptr;
        result["rewires"]=problem.sem?bj::value(problem.sem->numRewires()):bj::value(nullptr);
        if(problem.sem) {
            result["rewires_measurement"]="SemRRTstarCommon_successful_parent_changes";
            result["samples_accepted"]=problem.sem->numTreeInsertions();
            result["samples_accepted_measurement"]="SemRRTstarCommon_tree_insertions_after_state_and_parent_motion_validation";
        }
        result["unavailable_counter_reason"]=problem.rrt?"OMPL_1.7.0_RRTstar_does_not_expose_removed_states_or_actual_rewires":
            problem.lazy?"Hauser_LazyPRMstar_is_a_roadmap_without_tree_pruning_or_rewiring_counters":problem.sem?
            "SemRRTstarCommon_does_not_prune_or_disconnect_branches":"native_BITstar_pruning_and_rewiring_log_parsed_offline";
        if(planner=="rrtstar") {
            auto &c=result["counters"].as_object();c["sample_proposals"]=nullptr;c["samples_returned_before_validity"]=nullptr;
            c["uniform_sampling_unavailable_reason"]="OMPL_default_state_sampler_not_instrumented";
        }
        std::vector<bj::object> eventMetrics;
        for(std::size_t i=0;i<events.size();++i) {
            auto &event=events[i];std::string id="snapshot_"+std::to_string(i);
            savePath(output/"snapshots"/id,event.path,map);
            progress.push_back({{"event_type","improvement"},{"elapsed_s",event.elapsed},{"feasible",true},
                {"best_cost_terrain_m",event.cost},{"snapshot_id",id},{"counters",event.counters},
                {"event_origin",identity?"analytic_identity_query":"ompl_intermediate_solution_callback"}});
            if(compositeObjective) {
                auto metrics=semExposure?semExposurePathCostBreakdown(event.path,map,options):pathCostBreakdown(event.path,map);
                const double recomputed=bj::value_to<double>(metrics.at("objective_value"));
                if(std::abs(recomputed-event.cost)>1e-8+1e-8*std::max(recomputed,event.cost))
                    throw std::runtime_error("callback objective differs from snapshot integral");
                metrics["objective_value"]=event.cost;
                addMetrics(progress.back(),metrics);
                eventMetrics.push_back(std::move(metrics));
            }
        }
        // Reconstruct poll tuples from the same immutable incumbent snapshot.
        // No integration or serialization is added to the timed callback.
        if(compositeObjective)for(auto &poll:progress)if(poll.at("event_type")=="poll") {
            const double elapsed=bj::value_to<double>(poll.at("elapsed_s"));
            std::size_t count=0;
            while(count<events.size() && events[count].elapsed<=elapsed)++count;
            if(count) {
                addMetrics(poll,eventMetrics[count-1]);
                poll["snapshot_id"]="snapshot_"+std::to_string(count-1);
            } else {
                for(const char *key:{"objective_value","cost_terrain_m","length_xy_m"})poll[key]=nullptr;
                if(semExposure)for(const char *key:{"sem_exposure_risk_m","sem_exposure_normalized_m"})poll[key]=nullptr;
                else for(const char *key:{"risk_slope_integral_m","risk_clearance_integral_m","risk_integral_m"})poll[key]=nullptr;
            }
        }
        if(!events.empty()) {
            result["first_solution_time_s"]=events.front().elapsed;
            // Native safety admission timestamp; success still requires the independent oracle.
            if(safeLength)result["first_safe_solution_time_s"]=events.front().elapsed;
        }
        if(problem.pdef->hasSolution()) {
            auto path=std::dynamic_pointer_cast<ompl::geometric::PathGeometric>(problem.pdef->getSolutionPath());
            if(!path)throw std::runtime_error("non-geometric solution");
            std::vector<Point> xy;double cost=0,xyLength=0;
            for(const auto *state:path->getStates())xy.push_back(point(state));
            for(std::size_t i=1;i<xy.size();++i){cost+=map.surfaceEdge(xy[i-1],xy[i]).cost;xyLength+=distanceXY(xy[i-1],xy[i]);}
            savePath(output,xy,map);result["path_available"]=true;result["best_cost_terrain_m"]=number(cost);result["length_xy_m"]=xyLength;
            if(traversability)addMetrics(result,pathCostBreakdown(xy,map));
            if(semExposure)addMetrics(result,semExposurePathCostBreakdown(xy,map,options));
        }
        std::stable_sort(progress.begin(),progress.end(),[](const bj::object &a,const bj::object &b){return a.at("elapsed_s").as_double()<b.at("elapsed_s").as_double();});
        std::ofstream log(output/"progress.jsonl");for(auto &event:progress) {
            if(nativeEuclideanSampler)markNativeSamplerCountersUnavailable(event["counters"].as_object());
            if(planner=="rrtstar") {
                auto &c=event["counters"].as_object();
                c["sample_proposals"]=nullptr;c["samples_returned_before_validity"]=nullptr;
                c["uniform_sampling_unavailable_reason"]="OMPL_default_state_sampler_not_instrumented";
            }
            log<<bj::serialize(event)<<'\n';
        }
        result["serialize_s"]=seconds(beforeSerialization);
        if(request.if_contains("audit_tree") && bj::value_to<bool>(request.at("audit_tree"))) {
            auto auditStart=Clock::now();
            if(problem.rrt && !identity)dynamic_cast<RRTInspection&>(*problem.rrt).auditTree(map);
            if(problem.ait || problem.lazy) {
                result["tree_checkpoint"]={{"passed",nullptr},{"status","SKIPPED"},{"vertices",nullptr},
                    {"scope",problem.ait?
                        "AITstar_solution_paths_verified_separately;reverse_search_contains_lazy_edges":
                        "HauserLazyPRMstar_solution_paths_verified_separately;roadmap_contains_unchecked_lazy_edges"},
                    {"offline_audit_s",seconds(auditStart)}};
            } else {
                ob::PlannerData data(problem.si);problem.planner->getPlannerData(data);
                std::vector<unsigned> parents(data.numVertices());std::vector<int> color(data.numVertices());
                std::function<void(unsigned)> visit=[&](unsigned i) {
                    if(color[i]==1)throw std::runtime_error("cycle at final PlannerData checkpoint");
                    if(color[i]==2)return;color[i]=1;
                    std::vector<unsigned> edges;data.getEdges(i,edges);
                    for(auto j:edges) {
                        if(!map.validMotion(point(data.getVertex(i).getState()),point(data.getVertex(j).getState())))
                            throw std::runtime_error("invalid final PlannerData edge");
                        visit(j);
                    }
                    color[i]=2;
                };
                for(unsigned i=0;i<data.numVertices();++i){std::vector<unsigned> edges;data.getEdges(i,edges);for(auto j:edges)++parents[j];}
                for(unsigned i=0;i<data.numVertices();++i){if(parents[i]>1)throw std::runtime_error("multiple final tree parents");visit(i);}
                result["tree_checkpoint"]={{"passed",true},{"vertices",data.numVertices()},
                    {"scope",problem.rrt?"final_internal_RRT_parent_costs_and_exported_forest":problem.sem?
                        "final_exported_SEM_RRTstar_common_backward_forest;internal_costs_covered_by_sem_regression":
                        "final_exported_BIT_forest;internal_costs_covered_by_integration_fixture"},
                    {"offline_audit_s",seconds(auditStart)}};
                }
        }
        saveJson(output/"worker.json",result);
        if(cloudHandshake)savePhase(output,"FINISHED");
        std::cout<<bj::serialize(result)<<'\n';
        ompl::msg::restorePreviousOutputHandler();return 0;
    }catch(const std::exception &error){ompl::msg::noOutputHandler();std::cerr<<error.what()<<'\n';return 2;}
}
