// A5 worker. Included after RpBitWorker.inc; A4 and the legacy workers are unchanged.
// Output follows the A4 / SEM-exposure schema, so the independent validators apply as they are.
#include "terrain25d/RpBitA5.hpp"
int runRpBitA5Worker(const bj::object &request,const fs::path &output,bool cloudHandshake) {
    const auto requireString=[&](const char *key,const char *expected){
        if(!request.if_contains(key)||bj::value_to<std::string>(request.at(key))!=expected)
            throw std::runtime_error(std::string("RP-BIT* A5 requires ")+key+"="+expected);
    };
    requireString("planner_id","bitstar_a3");requireString("objective_type","surface_length_sem_exposure");
    for(const char *key:{"traversability","pri_portfolio","pri_pathsearch","work_iterations","rp_bit","rp_refine_replay"})
        if(request.if_contains(key))throw std::runtime_error("RP-BIT* A5 incompatible request field: "+std::string(key));
    for(const char *key:{"state_cache","scalar_cost","sampler_query_cache","collision_index"})
        if(!request.if_contains(key)||!bj::value_to<bool>(request.at(key)))
            throw std::runtime_error(std::string("RP-BIT* A5 requires ")+key);
    if(request.if_contains("exact_flat_certificate")&&bj::value_to<bool>(request.at("exact_flat_certificate")))
        throw std::runtime_error("RP-BIT* A5 does not support length-only certificates");
    const std::string variant=bj::value_to<std::string>(request.at("variant_id"));
    if(variant.empty()||variant=="bitstar_a3")throw std::runtime_error("RP-BIT* A5 requires a distinct variant_id");
    const auto exposure=parseSemExposure(request.at("sem_exposure").as_object());
    const auto &registered=request.at("rp_bit_a5").as_object();
    RpA5Options options;
    // Every switch is registered explicitly: no arm relies on a default.
    for(const char *key:{"refine","window_samples","batch_min","batch_growth","fast_validity","lazy_refine","seed"})
        if(!registered.if_contains(key))throw std::runtime_error(std::string("rp_bit_a5: missing ")+key);
    for(const auto &entry:registered) {
        const std::string key(entry.key());
        if(key=="refine")options.refine=bj::value_to<bool>(entry.value());
        else if(key=="window_samples")options.window_samples=bj::value_to<unsigned>(entry.value());
        else if(key=="batch_min")options.batch_min=bj::value_to<unsigned>(entry.value());
        else if(key=="batch_growth")options.batch_growth=bj::value_to<bool>(entry.value());
        else if(key=="fast_validity")options.fast_validity=bj::value_to<bool>(entry.value());
        else if(key=="lazy_refine")options.lazy_refine=bj::value_to<bool>(entry.value());
        else if(key=="seed")options.seed=bj::value_to<bool>(entry.value());
        else if(key=="stop_relative_eps")options.stop_relative_eps=bj::value_to<double>(entry.value());
        else if(key=="test_max_iterations") {
            if(!request.if_contains("test_only_fixed_work")||!bj::value_to<bool>(request.at("test_only_fixed_work")))
                throw std::runtime_error("rp_bit_a5 work limit requires test_only_fixed_work=true");
            options.test_max_iterations=bj::value_to<unsigned>(entry.value());
        } else throw std::runtime_error("rp_bit_a5: unknown field "+key);
    }
    if(options.batch_min==0)throw std::runtime_error("rp_bit_a5: batch_min must be positive");
    if(bj::value_to<unsigned>(request.at("samples_per_batch"))!=100||
       bj::value_to<unsigned>(request.at("batch_after_first_solution"))!=options.batch_min||
       bj::value_to<double>(request.at("rewire_factor"))!=1.1||bj::value_to<unsigned>(request.at("sample_attempts"))!=100)
        throw std::runtime_error("registered BIT* batch/rewire/sampling configuration required");
    const auto &safety=request.at("safe_length").as_object();
    for(const auto &entry:safety){const auto key=entry.key();if(key!="field_path"&&key!="risk_cap"&&key!="slope_cap_deg"&&key!="atlas_sampling"&&
            key!="policy_id"&&key!="field_sha256"&&key!="cloud_sha256"&&key!="builder_source_sha256")throw std::runtime_error("unknown RP-BIT* A5 safety field");}
    if(bj::value_to<std::string>(safety.at("policy_id"))!="a3t_safe_length_v1"||bj::value_to<double>(safety.at("risk_cap"))!=40.||
       bj::value_to<double>(safety.at("slope_cap_deg"))!=35.||!bj::value_to<bool>(safety.at("atlas_sampling")))
        throw std::runtime_error("RP-BIT* A5 safety contract mismatch");
    options.planner.lambda=exposure.lambda;
    options.planner.batch_after_first_solution=options.batch_min;
    options.budget_s=bj::value_to<double>(request.at("planning_budget_s"));
    options.planner_seed=bj::value_to<unsigned>(request.at("planner_seed"));
    const auto pointFrom=[&](const char *key){const auto &a=request.at(key).as_array();
        if(a.size()!=2)throw std::runtime_error("XY endpoints required");
        Point p{bj::value_to<double>(a[0]),bj::value_to<double>(a[1])};
        if(!std::isfinite(p.x)||!std::isfinite(p.y))throw std::runtime_error("nonfinite endpoint");return p;};
    const auto start=pointFrom("start"),goal=pointFrom("goal");
    const auto beforeLoad=Clock::now();auto map=Terrain::load(bj::value_to<std::string>(request.at("map_path")));map.enableCollisionIndex();
    auto field=std::make_shared<const SafetyField>(rpA5LoadField(bj::value_to<std::string>(safety.at("field_path"))));
    const double preprocessing=seconds(beforeLoad);
    const auto beganUtc=utcTimestamp();if(cloudHandshake)savePhase(output,"SOLVING");map.resetCounters();
    const auto a5=runRpBitA5(map,field,start,goal,options);
    const auto &run=a5.run;
    const auto solveCounters=counterJson(map.counters(),run.counters);
    if(cloudHandshake)savePhase(output,"SERIALIZING");const auto serializationStart=Clock::now();
    // Reported metrics come from the canonical (rational) integral, never from the search-time integral.
    const auto canonical=[&](const std::vector<Point> &path){
        double length=0.,xy=0.,integral=0.;
        const std::size_t edges=path.size()>1?path.size()-1:1;
        for(std::size_t i=0;i<edges;++i){const auto a=path[i],b=path.size()>1?path[i+1]:a;
            const double cost=map.surfaceCost(a,b);const auto value=semExposureCanonical(map,*field,a,b);
            if(!std::isfinite(cost)||!value)throw std::runtime_error("published path has undefined canonical exposure");
            length+=cost;xy+=distanceXY(a,b);integral+=*value;}
        return bj::object{{"cost_terrain_m",number(length)},{"best_cost_terrain_m",number(length)},{"length_xy_m",number(xy)},
            {"sem_exposure_risk_m",number(integral)},{"sem_exposure_normalized_m",number(integral/40.)},
            {"objective_value",number(length+exposure.lambda*integral/40.)}};
    };
    const RpRefineOptions refinement=options.refinement;
    bj::object effective{{"planner_class","ompl::geometric::BITstar"},{"motion_objective","surface_length_sem_exposure"},
        {"informed_cost_bound","objective_value_J"},
        {"edge_lower_bound","max(d_phi*(1+lambda*min_box_risk/40),potential_differences)"},
        {"cost_to_go_lower_bound","max(d_phi,risk_potential)"},
        {"informed_sampling",options.seed?"uniform_over_risk_potential_sublevel_cells_exact_rejection_capped_by_published_J":
            "uniform_over_risk_potential_sublevel_cells_exact_rejection"},
        {"k_nearest",true},{"jit",false},{"pruning",true},{"samples_per_batch",100},{"batch_after_first_solution",options.batch_min},
        {"safety_atlas_sampling",true},{"nearest_neighbor_metric","euclidean_xy"},{"rewire_factor",1.1},{"sample_attempts",100},
        {"simplification",false},{"declared_refinement",options.refine}};
    bj::array batches;
    for(const auto &b:a5.batches)
        batches.push_back(bj::object{{"batch",b.batch},{"random_samples",b.random_samples},
            {"elapsed_s",b.elapsed_s},{"iterations",b.iterations},{"bitstar_cost",number(b.bitstar_cost)},{"refine_s",b.refine_s},
            {"published_J",number(b.published)},{"stale_samples",b.stale_samples}});
    bj::object result{{"schema_version",2},{"backend_id","a5_rpbitstar_v1"},{"planner_id",variant},{"backend_planner_id","bitstar_a3"},
        {"variant_id",variant},{"ompl_version",OMPL_VERSION},{"compiler",__VERSION__},{"state_dimension",2},
        {"objective_type","surface_length_sem_exposure"},{"requested_policy","risk_potential/I_J"},
        {"requested_sem_exposure",request.at("sem_exposure")},
        {"effective_sem_exposure",{{"lambda",exposure.lambda},{"risk_scale",exposure.risk_scale},
            {"integral","piecewise_constant_SEM_risk_ds_surface"},{"density_version","exact_canonical_raster_v1"},
            {"search_integral","double_single_pass_merge_v1_audited_against_canonical_at_serialization"},
            {"boundary_policy","max_incident_supported_cells_on_positive_length_intervals"},
            {"risk_interpretation","additive_SEM_exposure_proxy_not_unique_cell_PRI"}}},
        {"requested_safe_length",safety},
        {"effective_safe_length",{{"policy_id","a3t_safe_length_v1"},{"risk_cap",40.},{"slope_cap_deg",35.},{"atlas_sampling",true},
            {"thresholds_immutable_during_solve",true},{"domain","full_safe_cap40_no_corridor"}}},
        {"requested_rp_bit_a5",registered},
        {"effective_rp_bit_a5",{{"refine",options.refine},{"window_samples",options.window_samples},{"batch_min",options.batch_min},
            {"batch_growth",options.batch_growth},{"fast_validity",options.fast_validity},{"lazy_refine",options.lazy_refine},
            {"seed",options.seed},{"stop_relative_eps",options.stop_relative_eps},
            {"stop_rule","published_J_not_lower_by_eps_over_window_samples_of_exhausted_batches"},
            {"batch_schedule",options.batch_growth?"batch_min_then_max(batch_min,stale+drawn)_capped_by_window":"batch_min"},
            {"validity","sem_cells_then_footprint_then_swept_slope_exact_policy_order"},
            {"slope_shortcut",options.fast_validity?"exact_swept_slope_skipped_only_where_gradient_envelope_is_below_the_cap":"none"},
            {"refine_rule",options.lazy_refine?"first_incumbent_or_raw_cost_lower_by_eps_than_last_refined_or_window_about_to_close":
                "every_batch_boundary_with_a_new_incumbent"},
            {"seed_search",options.seed?"astar_over_open_sem_cell_centres_guided_by_goal_potential":"none"},
            {"sampling_bound",options.seed?"published_J":"bitstar_cost"},
            {"seed_expansion_limit",options.seed?bj::value(RP_A5_SEED_EXPANSIONS):bj::value(nullptr)},
            {"potential_lattice","sem_cell_corners_8_connected_kappa_0.92387953251_four_side_lipschitz_extension"},
            {"refinement",{{"shortcut_window",8},{"round_turns",refinement.round_turns},
                {"max_edge_m",refinement.max_edge_m},{"step_m",refinement.step_m},
                {"min_step_m",refinement.min_step_m},{"turn_limit_deg",30.},{"min_chord_m",refinement.min_chord_m},
                {"round_cost_ratio",refinement.round_cost_ratio},{"max_sweeps",refinement.max_sweeps},
                {"max_edge_evaluations",refinement.max_edge_evaluations}}},
            {"publication","refined_incumbent_at_batch_boundary_if_J_improves_by_1e-6"},{"sem_reference_used",false}}},
        {"effective_policy",effective},{"planner_status",run.status},{"execution_status","COMPLETED"},
        {"termination_reason",run.termination},{"time_budget_s",options.budget_s},{"actual_solve_wall_s",run.actual_wall_s},
        {"overshoot_s",std::max(0.,run.actual_wall_s-options.budget_s)},{"solve_started_utc",beganUtc},
        {"preprocessing_s",preprocessing},{"load_preprocess_s",preprocessing},{"setup_s",run.setup_s},
        {"timing_origin","before_safety_policy_potential_and_planner_creation"},
        {"iterations",run.iterations},{"batches",run.batches},{"batch_transition_count",run.batch_transitions},
        {"raw_incumbent_count",run.raw_incumbents},{"improvement_count",run.publications.size()},
        {"refinement_count",run.refinements},{"refine_s",run.refine_s},
        {"lower_bound_J",number(run.lower_bound)},{"counters",solveCounters},
        {"rp_counters",{{"fast_integral_calls",run.rp.fast_integral_calls},{"potential_evaluations",run.rp.potential_evaluations},
            {"box_bound_calls",run.rp.box_bound_calls},{"sampler_cell_draws",run.rp.sampler_cell_draws},
            {"sampler_rejected",run.rp.sampler_rejected},{"sampler_empty_prefix_calls",run.rp.sampler_empty_prefix_calls},
            {"heuristic_violations",run.rp.heuristic_violations},{"refine_edge_evaluations",run.rp.refine_edge_evaluations},
            {"refine_validity_checks",run.rp.refine_validity_checks}}},
        {"a5_timing",{{"policy_s",a5.policy_s},{"problem_s",a5.problem_s},{"envelope_s",a5.envelope_s},{"seed_s",a5.seed_s},
            {"refine_s",run.refine_s},{"setup_s",run.setup_s}}},
        {"a5_counters",{{"slope_certified",a5.a5.slope_certified},{"slope_exact",a5.a5.slope_exact},
            {"refine_deferred",a5.a5.refine_deferred},{"bounded_samples",a5.a5.bounded_samples},
            {"seed_expansions",a5.a5.seed_expansions}}},
        {"seed_status",a5.seed_status},{"batch_log",batches},
        {"path_available",!run.publications.empty()},{"first_solution_time_s",nullptr},{"first_safe_solution_time_s",nullptr}};
    for(const char *key:{"objective_value","cost_terrain_m","best_cost_terrain_m","length_xy_m","sem_exposure_risk_m","sem_exposure_normalized_m"})
        result[key]=nullptr;
    if(request.if_contains("test_only_fixed_work"))result["test_only_fixed_work"]=request.at("test_only_fixed_work");
    std::ofstream progress(output/"progress.jsonl");
    double previous=std::numeric_limits<double>::infinity(),worst=0.;
    for(const auto &publication:run.publications) {
        const std::string id="snapshot_"+std::to_string(publication.id);
        savePath(output/"snapshots"/id,publication.path,map);
        auto metrics=canonical(publication.path);
        const double value=bj::value_to<double>(metrics.at("objective_value")),search=publication.cost.objective;
        const double error=std::abs(value-search)/std::max(1.,std::abs(value));
        worst=std::max(worst,error);
        if(error>1e-8)throw std::runtime_error("search-time integral differs from the canonical exposure");
        if(value>previous+1e-9)throw std::runtime_error("published objective increased");
        previous=value;
        bj::object event{{"event_type","improvement"},{"snapshot_id",id},{"elapsed_s",publication.elapsed_s},{"feasible",true},
            {"event_origin",publication.origin},{"bitstar_cost",number(publication.bitstar_cost)},{"batch",publication.batch},
            {"search_objective_value",number(search)},{"sharp_turn_count",publication.cost.sharp_turns},
            {"vertex_count",publication.path.size()}};
        addMetrics(event,metrics);
        progress<<bj::serialize(event)<<'\n';
        if(&publication==&run.publications.back()) {
            savePath(output,publication.path,map);addMetrics(result,metrics);
            result["final_sharp_turn_count"]=publication.cost.sharp_turns;
        }
    }
    if(!progress)throw std::runtime_error("cannot write RP-BIT* A5 progress");
    if(!run.publications.empty()) {
        result["first_solution_time_s"]=run.publications.front().elapsed_s;
        result["first_safe_solution_time_s"]=run.publications.front().elapsed_s;
        const double final=run.publications.back().cost.objective;
        result["certificate_gap"]=run.lower_bound>0.?number(final/run.lower_bound-1.):bj::value(nullptr);
    }
    result["fast_integral_max_rel_error"]=worst;
    result["serialize_s"]=seconds(serializationStart);
    saveJson(output/"worker.json",result);if(cloudHandshake)savePhase(output,"FINISHED");return 0;
}
