// SEM-RRT* + rpRefine replay for the refine-attribution experiment. Included after RpBitWorker.inc;
// RP-BIT* itself is unchanged. It replays the published incumbents of a finished SEM-RRT* trial on
// SEM's own solve clock and refines the incumbent standing at each registered time (and at SEM's
// stop) with the same rpRefine and safety policy as A4. Every measured step is added to that clock,
// as if it had run inline in SEM's thread. Unsafe SEM incumbents are neither published nor refined.
#include <cstring>
namespace replay_detail {
// Coordinates travel as IEEE-754 bit patterns: Boost.JSON 1.82 does not parse decimals exactly.
double fromBits(const bj::value &value) {
    const auto text=bj::value_to<std::string>(value);
    if(text.size()!=16||text.find_first_not_of("0123456789abcdef")!=std::string::npos)
        throw std::runtime_error("rp_refine_replay: coordinates must be 16 lowercase hex digits");
    const std::uint64_t bits=std::stoull(text,nullptr,16);double x;std::memcpy(&x,&bits,sizeof x);
    if(!std::isfinite(x))throw std::runtime_error("rp_refine_replay: nonfinite coordinate");
    return x;
}
bool member(const std::string &key,std::initializer_list<const char*> keys) {
    for(const char *k:keys)if(key==k)return true;
    return false;
}
}

int runRpRefineReplayWorker(const bj::object &request,const fs::path &output,bool cloudHandshake) {
    using replay_detail::fromBits;using replay_detail::member;
    const auto requireString=[&](const char *key,const char *expected){
        if(!request.if_contains(key)||bj::value_to<std::string>(request.at(key))!=expected)
            throw std::runtime_error(std::string("rp_refine_replay requires ")+key+"="+expected);
    };
    requireString("planner_id","rp_refine_replay");requireString("objective_type","surface_length_sem_exposure");
    for(const auto &entry:request)
        if(!member(std::string(entry.key()),{"planner_id","variant_id","planner_seed","objective_type","sem_exposure","map_path",
                "start","goal","planning_budget_s","collision_index","cloud_phase_handshake","safe_length","rp_refine_replay"}))
            throw std::runtime_error("rp_refine_replay: unexpected request field "+std::string(entry.key()));
    if(!bj::value_to<bool>(request.at("collision_index")))throw std::runtime_error("rp_refine_replay requires collision_index");
    const std::string variant=bj::value_to<std::string>(request.at("variant_id"));
    if(variant.empty())throw std::runtime_error("rp_refine_replay requires a variant_id");
    const auto exposure=parseSemExposure(request.at("sem_exposure").as_object());
    const double lambda=exposure.lambda,budget=bj::value_to<double>(request.at("planning_budget_s"));
    if(!std::isfinite(budget)||budget<=0.)throw std::runtime_error("rp_refine_replay: invalid budget");
    // Same safety contract as RpBitWorker.inc.
    const auto &safety=request.at("safe_length").as_object();
    for(const auto &entry:safety){const auto key=entry.key();if(key!="field_path"&&key!="risk_cap"&&key!="slope_cap_deg"&&key!="atlas_sampling"&&
            key!="policy_id"&&key!="field_sha256"&&key!="cloud_sha256"&&key!="builder_source_sha256")throw std::runtime_error("unknown replay safety field");}
    if(bj::value_to<std::string>(safety.at("policy_id"))!="a3t_safe_length_v1"||bj::value_to<double>(safety.at("risk_cap"))!=40.||
       bj::value_to<double>(safety.at("slope_cap_deg"))!=35.||!bj::value_to<bool>(safety.at("atlas_sampling")))
        throw std::runtime_error("rp_refine_replay safety contract mismatch");
    const auto pointFrom=[&](const char *key){const auto &a=request.at(key).as_array();
        if(a.size()!=2)throw std::runtime_error("XY endpoints required");
        Point p{bj::value_to<double>(a[0]),bj::value_to<double>(a[1])};
        if(!std::isfinite(p.x)||!std::isfinite(p.y))throw std::runtime_error("nonfinite endpoint");return p;};
    const auto start=pointFrom("start"),goal=pointFrom("goal");

    const auto &spec=request.at("rp_refine_replay").as_object();
    for(const char *key:{"source_trial_id","source_worker_sha256","refine_at_s","source_solve_s","source_preprocessing_s",
                         "source_planner_status","source_termination_reason","incumbents"})
        if(!spec.if_contains(key))throw std::runtime_error("rp_refine_replay: missing "+std::string(key));
    if(spec.size()!=8)throw std::runtime_error("rp_refine_replay: unknown replay field");
    const double stop=bj::value_to<double>(spec.at("source_solve_s")),sourcePrep=bj::value_to<double>(spec.at("source_preprocessing_s"));
    if(!std::isfinite(stop)||stop<0.||!std::isfinite(sourcePrep)||sourcePrep<0.)throw std::runtime_error("rp_refine_replay: invalid source timing");
    std::vector<double> refineAt;
    for(const auto &value:spec.at("refine_at_s").as_array()) {
        const double r=bj::value_to<double>(value);
        if(!std::isfinite(r)||r<=0.||(!refineAt.empty()&&r<=refineAt.back()))
            throw std::runtime_error("rp_refine_replay: refine_at_s must be positive and strictly increasing");
        refineAt.push_back(r);
    }
    struct Incumbent{std::string id;double t;std::vector<Point> path;};
    std::vector<Incumbent> incumbents;
    for(const auto &value:spec.at("incumbents").as_array()) {
        const auto &item=value.as_object();
        if(item.size()!=3||!item.if_contains("snapshot_id")||!item.if_contains("elapsed_s")||!item.if_contains("path_bits"))
            throw std::runtime_error("rp_refine_replay: incumbent needs exactly snapshot_id, elapsed_s, path_bits");
        Incumbent next{bj::value_to<std::string>(item.at("snapshot_id")),bj::value_to<double>(item.at("elapsed_s")),{}};
        if(next.id.empty()||!std::isfinite(next.t)||next.t<0.||next.t>stop||(!incumbents.empty()&&next.t<incumbents.back().t))
            throw std::runtime_error("rp_refine_replay: incumbents must be ordered in time within the source solve");
        for(const auto &point:item.at("path_bits").as_array()) {
            const auto &xy=point.as_array();
            if(xy.size()!=2)throw std::runtime_error("rp_refine_replay: XY coordinates required");
            next.path.push_back({fromBits(xy[0]),fromBits(xy[1])});
        }
        // Endpoints are compared with a tolerance because the request endpoints are decimal JSON.
        const auto near=[](Point a,Point b){return std::abs(a.x-b.x)<=1e-9&&std::abs(a.y-b.y)<=1e-9;};
        if(next.path.size()<2||!near(next.path.front(),start)||!near(next.path.back(),goal))
            throw std::runtime_error("rp_refine_replay: incumbent endpoints differ from start/goal");
        incumbents.push_back(std::move(next));
    }

    const auto beforeLoad=Clock::now();auto map=Terrain::load(bj::value_to<std::string>(request.at("map_path")));map.enableCollisionIndex();
    auto field=std::make_shared<const SafetyField>(SafetyField::load(bj::value_to<std::string>(safety.at("field_path"))));
    const double loadSeconds=seconds(beforeLoad);
    if(cloudHandshake)savePhase(output,"SOLVING");
    const auto replayStart=Clock::now();
    auto policy=std::make_shared<const SafetyPolicy>(map,field,40.,35.);
    const bool validQuery=policy->validState(start)&&policy->validState(goal);
    const std::string invalidStatus=!policy->validState(start)?"INVALID_START":"INVALID_GOAL";
    const RpEdgeIntegrator integrator(map,*field);
    const RpRefineOptions refinement{};
    const double setup=seconds(replayStart);
    const auto safe=[&](const std::vector<Point> &path) {
        for(std::size_t i=1;i<path.size();++i)if(!policy->validMotion(path[i-1],path[i]))return false;
        return path.size()>=2;
    };

    struct Published{std::string origin,source;double stamp,semAt;std::vector<Point> path;RpPathCost cost;};
    std::vector<Published> published;
    bj::array steps;
    double offset=setup,best=std::numeric_limits<double>::infinity(),refineSeconds=0.;
    unsigned refinements=0,rawRejected=0;
    std::uint64_t edgeEvaluations=0,validityChecks=0;
    std::vector<bool> rawOk(incumbents.size(),false),refined(incumbents.size(),false);
    // A candidate is published by A4's rule: in budget and J lower by more than 1e-6 relative.
    const auto offer=[&](bj::object &step,const char *origin,const std::string &source,double at,std::vector<Point> path,const RpPathCost &cost) {
        const double stamp=at+offset;
        step["stamp_s"]=stamp;step["objective"]=number(cost.objective);
        const bool improves=!std::isfinite(best)||cost.objective<best-1e-6*std::max(1.,best);
        if(stamp<=budget&&improves) {
            step["published"]=true;step["snapshot_id"]="snapshot_"+std::to_string(published.size());
            published.push_back({origin,source,stamp,at,std::move(path),cost});best=cost.objective;
        }
    };
    const auto baseStep=[](const char *kind,double at){
        return bj::object{{"kind",kind},{"at_s",at},{"source_snapshot_id",nullptr},{"skipped",nullptr},{"duration_s",0.},
            {"valid",false},{"safe",false},{"objective",nullptr},{"stamp_s",nullptr},{"published",false},{"snapshot_id",nullptr}};
    };
    const auto raw=[&](std::size_t i) {
        auto step=baseStep("raw",incumbents[i].t);step["source_snapshot_id"]=incumbents[i].id;
        const auto began=Clock::now();
        const auto cost=rpPathCost(integrator,lambda,incumbents[i].path);
        const bool ok=cost.valid&&safe(incumbents[i].path);
        const double duration=seconds(began);offset+=duration;
        step["duration_s"]=duration;step["valid"]=cost.valid;step["safe"]=ok;
        rawOk[i]=ok;
        if(ok)offer(step,"sem_incumbent",incumbents[i].id,incumbents[i].t,incumbents[i].path,cost);
        else ++rawRejected;
        steps.push_back(std::move(step));
    };
    const auto refine=[&](double at,std::size_t done) {
        auto step=baseStep("refine",at);
        if(!done)step["skipped"]="no_incumbent";
        else {
            const std::size_t i=done-1;step["source_snapshot_id"]=incumbents[i].id;
            if(!rawOk[i])step["skipped"]="unsafe_source";
            else if(refined[i])step["skipped"]="already_refined";
            else {
                refined[i]=true;
                const auto began=Clock::now();
                auto out=rpRefine(*policy,integrator,lambda,incumbents[i].path,refinement);
                refineSeconds+=seconds(began);++refinements;
                edgeEvaluations+=out.edge_evaluations;validityChecks+=out.validity_checks;
                const bool ok=out.cost.valid&&safe(out.path);
                const double duration=seconds(began);offset+=duration;
                step["duration_s"]=duration;step["valid"]=out.cost.valid;step["safe"]=ok;
                if(ok)offer(step,"rp_refinement",incumbents[i].id,at,std::move(out.path),out.cost);
            }
        }
        steps.push_back(std::move(step));
    };
    if(validQuery) {
        // Registered times before SEM's stop, then once at the stop; raw events first on equal times.
        std::vector<double> points;
        for(const double r:refineAt)if(r<stop)points.push_back(r);
        points.push_back(stop);
        std::size_t i=0,k=0;
        while(i<incumbents.size()||k<points.size()) {
            if(i<incumbents.size()&&(k==points.size()||incumbents[i].t<=points[k]))raw(i++);
            else refine(points[k++],i);
        }
    }
    const double replayWall=seconds(replayStart);
    const std::string status=!validQuery?invalidStatus:published.empty()?"NO_SOLUTION":"EXACT_SOLUTION";
    if(cloudHandshake)savePhase(output,"SERIALIZING");const auto serializationStart=Clock::now();
    // Reported metrics come from the canonical (rational) integral; mirrors RpBitWorker.inc.
    const auto canonical=[&](const std::vector<Point> &path){
        double length=0.,xy=0.,integral=0.;
        const std::size_t edges=path.size()>1?path.size()-1:1;
        for(std::size_t i=0;i<edges;++i){const auto a=path[i],b=path.size()>1?path[i+1]:a;
            const double cost=map.surfaceCost(a,b);const auto value=semExposureCanonical(map,*field,a,b);
            if(!std::isfinite(cost)||!value)throw std::runtime_error("published path has undefined canonical exposure");
            length+=cost;xy+=distanceXY(a,b);integral+=*value;}
        return bj::object{{"cost_terrain_m",number(length)},{"best_cost_terrain_m",number(length)},{"length_xy_m",number(xy)},
            {"sem_exposure_risk_m",number(integral)},{"sem_exposure_normalized_m",number(integral/40.)},
            {"objective_value",number(length+lambda*integral/40.)}};
    };
    bj::object requested;
    for(const auto &entry:spec)if(entry.key()!="incumbents")requested[entry.key()]=entry.value();
    bj::object result{{"schema_version",2},{"backend_id","a4_rpbitstar_v1"},{"planner_id",variant},{"backend_planner_id","rp_refine_replay"},
        {"variant_id",variant},{"ompl_version",OMPL_VERSION},{"compiler",__VERSION__},{"state_dimension",2},
        {"objective_type","surface_length_sem_exposure"},{"requested_policy","sem_native_incumbents_plus_rp_refine"},
        {"requested_sem_exposure",request.at("sem_exposure")},
        {"effective_sem_exposure",{{"lambda",exposure.lambda},{"risk_scale",exposure.risk_scale},
            {"integral","piecewise_constant_SEM_risk_ds_surface"},{"density_version","exact_canonical_raster_v1"},
            {"search_integral","double_single_pass_merge_v1_audited_against_canonical_at_serialization"},
            {"boundary_policy","max_incident_supported_cells_on_positive_length_intervals"},
            {"risk_interpretation","additive_SEM_exposure_proxy_not_unique_cell_PRI"}}},
        {"requested_safe_length",safety},
        {"effective_safe_length",{{"policy_id","a3t_safe_length_v1"},{"risk_cap",40.},{"slope_cap_deg",35.},{"atlas_sampling",true},
            {"thresholds_immutable_during_solve",true},{"domain","full_safe_cap40_no_corridor"}}},
        {"requested_rp_refine_replay",requested},{"source_incumbent_count",incumbents.size()},
        {"effective_refinement",{{"shortcut_window",8},{"round_turns",refinement.round_turns},
            {"max_edge_m",refinement.max_edge_m},{"step_m",refinement.step_m},
            {"min_step_m",refinement.min_step_m},{"turn_limit_deg",30.},{"min_chord_m",refinement.min_chord_m},
            {"round_cost_ratio",refinement.round_cost_ratio},{"max_sweeps",refinement.max_sweeps},
            {"max_edge_evaluations",refinement.max_edge_evaluations}}},
        {"effective_policy",{{"planner_class","rp_refine_replay_of_sem_native"},{"motion_objective","surface_length_sem_exposure"},
            {"source_planner","sem_native"},{"publication","a4_rule_J_improves_by_1e-6_and_stamp_within_budget"},
            {"raw_publication","sem_incumbent_if_valid_safe_and_J_improves"},
            {"refine_schedule","incumbent_standing_at_each_registered_time_before_stop_then_at_stop_once_per_incumbent"},
            {"timing","sem_solve_clock_plus_inline_replay_overhead"},{"unsafe_source","neither_published_nor_refined"}}},
        {"planner_status",status},{"execution_status","COMPLETED"},
        {"termination_reason",validQuery?bj::value(spec.at("source_termination_reason")):bj::value("INVALID_QUERY")},
        {"time_budget_s",budget},{"source_solve_s",stop},{"actual_solve_wall_s",stop+offset},
        {"overshoot_s",std::max(0.,stop+offset-budget)},{"replay_overhead_s",offset},{"replay_wall_s",replayWall},
        {"preprocessing_s",sourcePrep+loadSeconds},{"source_preprocessing_s",sourcePrep},{"replay_load_s",loadSeconds},
        {"setup_s",setup},{"timing_origin","sem_solve_clock_plus_replay_safety_policy_setup"},
        {"improvement_count",published.size()},{"refinement_count",refinements},{"refine_s",refineSeconds},
        {"raw_rejected_count",rawRejected},{"replay_steps",steps},
        {"rp_counters",{{"refine_edge_evaluations",edgeEvaluations},{"refine_validity_checks",validityChecks}}},
        {"path_available",!published.empty()},{"first_solution_time_s",nullptr},{"first_safe_solution_time_s",nullptr}};
    for(const char *key:{"objective_value","cost_terrain_m","best_cost_terrain_m","length_xy_m","sem_exposure_risk_m","sem_exposure_normalized_m"})
        result[key]=nullptr;
    std::ofstream progress(output/"progress.jsonl");
    double previous=std::numeric_limits<double>::infinity(),worst=0.;
    for(std::size_t n=0;n<published.size();++n) {
        const auto &item=published[n];
        const std::string id="snapshot_"+std::to_string(n);
        savePath(output/"snapshots"/id,item.path,map);
        auto metrics=canonical(item.path);
        const double value=bj::value_to<double>(metrics.at("objective_value")),search=item.cost.objective;
        const double error=std::abs(value-search)/std::max(1.,std::abs(value));
        worst=std::max(worst,error);
        if(error>1e-8)throw std::runtime_error("search-time integral differs from the canonical exposure");
        if(value>previous+1e-9)throw std::runtime_error("published objective increased");
        previous=value;
        bj::object event{{"event_type","improvement"},{"snapshot_id",id},{"elapsed_s",item.stamp},{"feasible",true},
            {"event_origin",item.origin},{"source_snapshot_id",item.source},{"sem_elapsed_s",item.semAt},
            {"search_objective_value",number(search)},{"sharp_turn_count",item.cost.sharp_turns},{"vertex_count",item.path.size()}};
        addMetrics(event,metrics);
        progress<<bj::serialize(event)<<'\n';
        if(n+1==published.size()) {
            savePath(output,item.path,map);addMetrics(result,metrics);
            result["final_sharp_turn_count"]=item.cost.sharp_turns;
        }
    }
    if(!progress)throw std::runtime_error("cannot write replay progress");
    if(!published.empty()) {
        result["first_solution_time_s"]=published.front().stamp;
        result["first_safe_solution_time_s"]=published.front().stamp;
    }
    result["fast_integral_max_rel_error"]=worst;
    result["serialize_s"]=seconds(serializationStart);
    saveJson(output/"worker.json",result);if(cloudHandshake)savePhase(output,"FINISHED");return 0;
}
