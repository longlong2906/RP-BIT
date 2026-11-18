// Included after the common worker helpers, keeping legacy workers unchanged.
int runPriPathSearchWorker(const bj::object &request,const fs::path &output,bool cloudHandshake) {
    const auto requireString=[&](const char *key,const char *expected){
        if(!request.if_contains(key)||bj::value_to<std::string>(request.at(key))!=expected)
            throw std::runtime_error(std::string("PRI path search requires ")+key+"="+expected);
    };
    requireString("planner_id","bitstar_a3");requireString("objective_type","surface_length");
    for(const char *key:{"traversability","sem_exposure","pri_portfolio","work_iterations"})
        if(request.if_contains(key))throw std::runtime_error("PRI path search incompatible request field: "+std::string(key));
    for(const char *key:{"state_cache","scalar_cost","sampler_query_cache","collision_index"})
        if(!request.if_contains(key)||!bj::value_to<bool>(request.at(key)))
            throw std::runtime_error(std::string("PRI path search requires ")+key);
    if(request.if_contains("exact_flat_certificate")&&bj::value_to<bool>(request.at("exact_flat_certificate")))
        throw std::runtime_error("PRI path search does not support early length certificates");
    const auto &registered=request.at("pri_pathsearch").as_object();
    if(registered.size()!=1||!registered.if_contains("mode"))throw std::runtime_error("pri_pathsearch: missing or unknown field");
    PriPathSearchOptions options;
    options.mode=bj::value_to<std::string>(registered.at("mode"));
    if(options.mode!="length"&&options.mode!="shortcut"&&options.mode!="full"&&options.mode!="segment")throw std::runtime_error("unregistered path search mode");
    if(options.mode=="segment")requireString("variant_id","a3t_pri_segment_v2");
    options.budget_s=bj::value_to<double>(request.at("planning_budget_s"));
    if(!std::isfinite(options.budget_s)||options.budget_s<=0.)throw std::runtime_error("invalid planning budget");
    options.planner_seed=bj::value_to<unsigned>(request.at("planner_seed"));
    if(const auto *test=request.if_contains("pathsearch_test_work")) {
        if(!request.if_contains("test_only_fixed_work")||!bj::value_to<bool>(request.at("test_only_fixed_work")))
            throw std::runtime_error("pathsearch work limits require test_only_fixed_work=true");
        const auto &work=test->as_object();
        if(work.size()!=4)throw std::runtime_error("test work missing or unknown field");
        options.test_anchor_iterations=bj::value_to<unsigned>(work.at("anchor_iterations"));
        options.test_iterations_per_slice=bj::value_to<unsigned>(work.at("iterations_per_slice"));
        options.test_optimizer_proposals_per_slice=bj::value_to<unsigned>(work.at("optimizer_proposals_per_slice"));
        options.test_cycles=bj::value_to<unsigned>(work.at("cycles"));
        if(!options.test_anchor_iterations||!options.test_iterations_per_slice||!options.test_optimizer_proposals_per_slice||!options.test_cycles)
            throw std::runtime_error("positive test work limits required");
    }
    if(bj::value_to<unsigned>(request.at("samples_per_batch"))!=100||bj::value_to<unsigned>(request.at("batch_after_first_solution"))!=200||
       bj::value_to<double>(request.at("rewire_factor"))!=1.1||bj::value_to<unsigned>(request.at("sample_attempts"))!=100)
        throw std::runtime_error("locked BIT* batch/rewire/sampling configuration required");
    const auto &safety=request.at("safe_length").as_object();
    for(const auto &entry:safety){const auto key=entry.key();if(key!="field_path"&&key!="risk_cap"&&key!="slope_cap_deg"&&key!="atlas_sampling"&&key!="policy_id"&&
            key!="field_sha256"&&key!="cloud_sha256"&&key!="builder_source_sha256")throw std::runtime_error("unknown path search safety field");}
    if(bj::value_to<std::string>(safety.at("policy_id"))!="a3t_safe_length_v1"||bj::value_to<double>(safety.at("risk_cap"))!=40.||
       bj::value_to<double>(safety.at("slope_cap_deg"))!=35.||!bj::value_to<bool>(safety.at("atlas_sampling")))
        throw std::runtime_error("path search safety contract mismatch");
    const auto pointFrom=[&](const char *key){const auto &a=request.at(key).as_array();
        if(a.size()!=2)throw std::runtime_error("XY endpoints required");
        Point p{bj::value_to<double>(a[0]),bj::value_to<double>(a[1])};
        if(!std::isfinite(p.x)||!std::isfinite(p.y))throw std::runtime_error("nonfinite endpoint");return p;};
    const auto start=pointFrom("start"),goal=pointFrom("goal");
    const auto beforeLoad=Clock::now();auto map=Terrain::load(bj::value_to<std::string>(request.at("map_path")));map.enableCollisionIndex();
    auto field=std::make_shared<const SafetyField>(SafetyField::load(bj::value_to<std::string>(safety.at("field_path"))));
    const double preprocessing=seconds(beforeLoad);
    ompl::RNG::setSeed(static_cast<unsigned>(options.planner_seed));
    const auto beganUtc=utcTimestamp();if(cloudHandshake)savePhase(output,"SOLVING");map.resetCounters();
    auto search=runPriPathSearch(map,field,start,goal,options);
    const auto solveCounters=counterJson(map.counters(),search.counters);
    if(cloudHandshake)savePhase(output,"SERIALIZING");const auto serializationStart=Clock::now();
    const auto sid=[](std::size_t id){return "candidate_"+std::to_string(id);};
    const auto optionalSid=[&](const std::optional<std::size_t>&id)->bj::value{return id?bj::value(sid(*id)):bj::value(nullptr);};
    bj::array ratios;for(auto ratio:search.cap_ratios)ratios.push_back(ratio);
    bj::object effective{{"planner_class","ompl::geometric::BITstar"},{"motion_objective","surface_length"},{"edge_lower_bound","d_phi"},
        {"cost_to_go_lower_bound","d_phi"},{"informed_sampling","I_phi"},{"informed_cost_bound","own_valid_surface_length_incumbent"},
        {"k_nearest",true},{"jit",false},{"pruning",true},{"samples_per_batch",100},{"batch_after_first_solution",200},{"safety_atlas_sampling",true},
        {"nearest_neighbor_metric","euclidean_xy"},{"external_length_cap_in_sampler",false},{"anchor_injected_as_incumbent",false}};
    bj::object result{{"schema_version",1},{"backend_id",options.mode=="segment"?"a3t_pri_segment_v2":"a3t_pri_pathsearch_v1"},{"planner_id",request.at("variant_id")},{"backend_planner_id","bitstar_a3"},
        {"ompl_version",OMPL_VERSION},{"compiler",__VERSION__},{"state_dimension",2},{"objective_type","surface_length"},
        {"requested_policy","d_phi/I_phi"},{"requested_pri_pathsearch",registered},{"requested_safe_length",safety},
        {"effective_pri_pathsearch",{{"mode",options.mode},{"anchor_fraction",.25},{"bitstar_slice_fraction",.005},{"optimizer_slice_fraction",.015},
            {"internal_length_cap_ratio",1.1},{"output_cap_ratios",ratios},{"length_bins",16},{"paths_per_bin",4},{"pool_capacity",64},
            {"single_persistent_planner",true},{"sem_reference_used",false},{"deadline_after_scoring",true},
            {"selection",options.mode=="length"?"shortest_surface_length_then_PRI_then_candidate_id":"exact_pri_then_surface_length_then_candidate_id"},
            {"cap_source",options.mode=="length"?"none_length_baseline":"frozen_shortest_anchor"},
            {"max_archive_bytes",options.max_archive_bytes},{"max_edge_cache_bytes",options.max_edge_cache_bytes}}},
        {"effective_safe_length",{{"policy_id","a3t_safe_length_v1"},{"risk_cap",40.},{"slope_cap_deg",35.},{"atlas_sampling",true},{"thresholds_immutable_during_solve",true}}},
        {"effective_policy",effective},{"planner_status",search.status},{"execution_status","COMPLETED"},{"pathsearch_stage",search.stage},
        {"termination_reason",search.optimizer_resource_limit?"RESOURCE_LIMIT":options.test_cycles?"WORK_OR_TIME_BUDGET":"TIME_BUDGET_OR_NATIVE_STOP"},
        {"time_budget_s",options.budget_s},{"actual_solve_wall_s",search.actual_wall_s},{"solve_started_utc",beganUtc},
        {"preprocessing_s",preprocessing},{"load_preprocess_s",preprocessing},{"setup_completed_s",search.setup_completed_s},
        {"timing_origin","before_initial_policy_and_planner_creation"},{"anchor_deadline_s",search.anchor_deadline_s},
        {"anchor_locked_s",search.anchor_locked_s?number(*search.anchor_locked_s):bj::value(nullptr)},
        {"planner_instances",search.planner_instances},{"optimizer_resource_limit",search.optimizer_resource_limit},
        {"counters",solveCounters},{"path_available",search.selected[3].has_value()},
        {"anchor_snapshot_id",optionalSid(search.anchor)},{"selected_snapshot_id",optionalSid(search.selected[3])},
        {"primary_output_alpha",1.1},{"candidate_archive","pathsearch_candidates.jsonl"},
        {"best_cost_terrain_m",nullptr},{"length_xy_m",nullptr}};
    if(request.if_contains("pathsearch_test_work"))result["pathsearch_test_work"]=request.at("pathsearch_test_work");
    const auto &sc=search.search_counters;bj::object rejected;for(const auto &r:sc.rejected)rejected[r.first]=r.second;
    result["search_counters"]=bj::object{{"proposals",sc.proposals},{"shortcut_proposals",sc.shortcut_proposals},{"deform_proposals",sc.deform_proposals},
        {"splice_proposals",sc.splice_proposals},{"accepted",sc.accepted},{"pool_evictions",sc.pool_evictions},{"edge_cache_hits",sc.edge_cache_hits},
        {"edge_evaluations",sc.edge_evaluations},{"incidence_adds",sc.incidence_adds},{"incidence_removes",sc.incidence_removes},
        {"full_recomputations",sc.full_recomputations},{"current_pool_size",sc.current_pool_size},{"peak_pool_size",sc.peak_pool_size},
        {"peak_edge_cache_bytes",sc.peak_edge_cache_bytes},{"archive_bytes",sc.archive_bytes},{"rejected",rejected}};
    if(options.mode=="segment"){
        const PriSegmentSearchOptions defaults;
        result["effective_pri_pathsearch"].as_object()["segment_search"]=bj::object{
            {"operator","contextual_bounded_beam"},{"schedule",bj::array{"shortcut","deform","segment_reduce","segment_contract"}},
            {"max_nodes",defaults.max_nodes},{"beam_width",defaults.beam_width},{"max_expansions",defaults.max_expansions},{"max_depth",defaults.max_depth},
            {"risk_target_ratio",.99},{"target_cap_indices",bj::array{0,1,2,2,3}},
            {"objective","min_surface_length_subject_to_whole_path_PRI_cap"},{"prefix_PRI_pruning",false},{"shared_outside_cells",true}};
        result["search_counters"].as_object()["segment_proposals"]=sc.segment_proposals;
        bj::array attempts;
        for(const auto&a:search.segment_attempts){const auto&c=a.counters;
            attempts.push_back(bj::object{{"proposal_id",a.proposal_id},{"parent_id",sid(a.parent_id)},{"move_type",a.move_type},
                {"first",a.first},{"last",a.last},{"pri_cap",a.pri_cap},{"length_cap",a.length_cap},{"began_s",a.began_s},{"finished_s",a.finished_s},{"status",a.status},{"termination",a.termination},
                {"counters",bj::object{{"nodes_proposed",c.nodes_proposed},{"nodes_retained",c.nodes_retained},{"edge_evaluations",c.edge_evaluations},
                {"edge_cache_hits",c.edge_cache_hits},{"expansions",c.expansions},{"labels_created",c.labels_created},{"beam_discarded",c.beam_discarded},
                {"distinct_sets_same_node",c.distinct_sets_same_node},{"prefix_above_pri_kept",c.prefix_above_pri_kept},{"length_rejected",c.length_rejected},
                {"invalid_edges",c.invalid_edges},{"non_simple_rejected",c.non_simple_rejected},{"completed_paths",c.completed_paths},
                {"pri_rejected",c.pri_rejected},{"full_recomputations",c.full_recomputations},{"cancel_checks",c.cancel_checks},{"soft_stops",c.soft_stops}}}});
        }
        result["segment_attempts"]=attempts;
    }
    bj::array episodes;bool called=false;
    for(const auto &e:search.slices){const bool solve=e.phase=="anchor"||e.phase=="bitstar"||e.phase=="length"||e.phase=="no_anchor_continue";called=called||solve;
        episodes.push_back(bj::object{{"episode_id",e.id},{"planner_instance_id",1},{"fresh_planner",e.id==0},{"anchor_injected",false},
            {"phase",e.phase},{"solve_called",solve},{"began_s",e.began_s},{"finished_s",e.finished_s},{"iterations",e.iterations},
            {"batch_transition_count",e.batch_transitions},{"proposals",e.proposals},{"status",e.status}});}
    result["episodes"]=episodes;result["planner_solve_called"]=called;
    bj::array snapshots;std::map<std::size_t,bj::object> metrics;
    std::ofstream archive(output/"pathsearch_candidates.jsonl"),progress(output/"progress.jsonl");
    for(const auto &c:search.candidates){
        const auto id=sid(c.id);savePath(output/"snapshots"/id,c.path,map);
        bj::array xy,parents,published;for(const auto p:c.path)xy.push_back(bj::array{p.x,p.y});
        for(const auto p:c.parent_ids)parents.push_back(bj::value(sid(p)));
        for(unsigned k=0;k<4;++k)if(c.published[k])published.push_back(search.cap_ratios[k]);
        const bool selected=std::any_of(search.selected.begin(),search.selected.end(),[&](const auto &v){return v&&*v==c.id;});
        double exposure=0.;bool exposureAvailable=c.stats.valid&&(!published.empty()||selected);
        if(exposureAvailable)for(std::size_t i=0;i<std::max<std::size_t>(1,c.path.size()-1);++i){
            const auto integral=semExposureCanonical(map,*field,c.path[i],c.path.size()>1?c.path[i+1]:c.path[i]);
            if(!integral){exposureAvailable=false;break;}exposure+=*integral;}
        bj::object value{{"snapshot_id",id},{"candidate_id",c.id},{"event_type","improvement"},{"event_origin","pathsearch_registered_selection"},
            {"elapsed_s",c.score_completed_s},{"callback_s",c.callback_s},{"score_completed_s",c.score_completed_s},
            {"callback_id",c.callback_id},{"proposal_id",c.proposal_id},{"parent_ids",parents},{"move_type",c.move_type},
            {"anchor_phase",c.anchor_phase},{"accepted",true},{"pool_admitted",c.pool_admitted},{"published",!published.empty()},
            {"published_alphas",published},{"reason",c.reason},{"feasible",c.stats.valid},
            {"cost_terrain_m",number(c.stats.length)},{"best_cost_terrain_m",number(c.stats.length)},{"length_xy_m",number(c.stats.length_xy)},
            {"objective_value",number(c.stats.length)},{"pri",number(c.stats.pri)},{"risk_sum",number(c.stats.risk_sum)},{"risk_count",c.stats.cells.size()},
            {"path_xy",xy},{"sem_exposure_risk_m",exposureAvailable?number(exposure):bj::value(nullptr)},
            {"sem_exposure_normalized_m",exposureAvailable?number(exposure/40.):bj::value(nullptr)},
            {"exposure_unavailable_reason",exposureAvailable?bj::value(nullptr):bj::value(published.empty()&&!selected?"not_used_for_selection":"undefined_diagnostic_exposure")}};
        if(c.segment_pri_cap){value["segment_pri_cap"]=*c.segment_pri_cap;value["segment_length_cap"]=*c.segment_length_cap;
            value["segment_first"]=c.segment_first;value["segment_last"]=c.segment_last;}
        archive<<bj::serialize(value)<<'\n';
        if(!published.empty()||selected)metrics.emplace(c.id,value);
        if(search.selected[3]&&*search.selected[3]==c.id)savePath(output,c.path,map);
    }
    for(std::size_t i=0;i<search.publications.size();++i){
        const auto &publication=search.publications[i];auto event=metrics.at(publication.candidate_id);
        event["event_id"]="publication_"+std::to_string(i);event["alpha"]=search.cap_ratios[publication.cap_index];
        event["elapsed_s"]=publication.elapsed_s;event["publication_reason"]=publication.reason;
        snapshots.push_back(event);progress<<bj::serialize(event)<<'\n';
    }
    bj::array selectors;
    for(unsigned k=0;k<4;++k){bj::object entry{{"alpha",search.cap_ratios[k]},{"selected_snapshot_id",optionalSid(search.selected[k])},
        {"length_cap_m",search.length_caps[k]?number(*search.length_caps[k]):bj::value(nullptr)},{"path_available",search.selected[k].has_value()}};
        if(search.selected[k]){const auto &m=metrics.at(*search.selected[k]);
            for(const char *key:{"cost_terrain_m","best_cost_terrain_m","length_xy_m","objective_value","pri","risk_sum","risk_count","sem_exposure_risk_m","sem_exposure_normalized_m"}){
                entry[key]=m.at(key);if(k==3)result[key]=m.at(key);}}
        selectors.push_back(entry);}
    if(!archive||!progress)throw std::runtime_error("cannot write PRI path search trace");
    result["selectors"]=selectors;result["snapshots"]=snapshots;result["candidate_count"]=search.candidates.size();
    result["serialize_s"]=seconds(serializationStart);
    saveJson(output/"worker.json",result);if(cloudHandshake)savePhase(output,"FINISHED");return 0;
}
