#include "terrain25d/RpBitA5.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <queue>

namespace terrain25d {
namespace {
const double INF=std::numeric_limits<double>::infinity();
// A4's potential is KAPPA * (8-connected lattice distance). Dividing by it gives a
// lattice-scale cost-to-go that only guides the seed search; no optimality is claimed.
const double KAPPA=.92387953251;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
bool same(Point a,Point b) { return a.x==b.x&&a.y==b.y; }

// C4: A4's sampler under an extra upper bound, the J of the path already published.
// Every path better than that only visits states with g_hat + h_hat below it.
class A5Sampler final : public ob::InformedSampler {
public:
    A5Sampler(const ob::ProblemDefinitionPtr &p,unsigned attempts,ob::InformedSamplerPtr inner,const double &bound,std::uint64_t &drawn)
        :ob::InformedSampler(p,attempts),inner_(std::move(inner)),bound_(bound),drawn_(drawn){}
    bool sampleUniform(ob::State *state,const ob::Cost &maxCost) override { return sampleUniform(state,ob::Cost(0.),maxCost); }
    bool sampleUniform(ob::State *state,const ob::Cost &minCost,const ob::Cost &maxCost) override {
        const bool ok=inner_->sampleUniform(state,minCost,ob::Cost(std::min(maxCost.value(),bound_)));
        if(ok&&bound_<maxCost.value())++drawn_;
        return ok;
    }
    bool hasInformedMeasure() const override { return inner_->hasInformedMeasure(); }
    double getInformedMeasure(const ob::Cost &cost) const override { return inner_->getInformedMeasure(cost); }
    double getInformedMeasure(const ob::Cost &min,const ob::Cost &max) const override { return inner_->getInformedMeasure(min,max); }
private:
    ob::InformedSamplerPtr inner_;
    const double &bound_;
    std::uint64_t &drawn_;
};
// Costs and bounds are A4's, unchanged; only the sampler is wrapped.
class A5Objective final : public ob::OptimizationObjective {
public:
    A5Objective(const ob::SpaceInformationPtr &si,std::shared_ptr<RpObjective> inner,const double &bound,std::uint64_t &drawn)
        :ob::OptimizationObjective(si),inner_(std::move(inner)),bound_(bound),drawn_(drawn) { description_=inner_->getDescription(); }
    ob::Cost stateCost(const ob::State *state) const override { return inner_->stateCost(state); }
    ob::Cost motionCost(const ob::State *a,const ob::State *b) const override { return inner_->motionCost(a,b); }
    ob::Cost motionCostHeuristic(const ob::State *a,const ob::State *b) const override { return inner_->motionCostHeuristic(a,b); }
    ob::InformedSamplerPtr allocInformedStateSampler(const ob::ProblemDefinitionPtr &p,unsigned int attempts) const override {
        return std::make_shared<A5Sampler>(p,attempts,inner_->allocInformedStateSampler(p,attempts),bound_,drawn_);
    }
private:
    std::shared_ptr<RpObjective> inner_;
    const double &bound_;
    std::uint64_t &drawn_;
};

void recordRejection(PolicyCounters &c,const std::string &reason) {
    if(reason.empty())return;
    if(reason=="missing_sem_support")++c.safety_support_rejections;
    else if(reason=="sem_risk_above_cap")++c.safety_risk_rejections;
    else if(reason=="swept_slope_above_cap")++c.safety_slope_rejections;
    else if(reason=="outside_corridor")++c.safety_corridor_rejections;
    else ++c.safety_geometry_rejections;
}
// C2: TerrainMotionValidator with the slope shortcut; same verdicts and counters.
class A5MotionValidator final : public ob::MotionValidator {
public:
    A5MotionValidator(const ob::SpaceInformationPtr &si,const Terrain &map,std::shared_ptr<const SafetyPolicy> safety,
                      const RpA5SlopeEnvelope &envelope,PolicyCounters &policy,RpA5Counters &counters)
        :ob::MotionValidator(si),map_(map),safety_(std::move(safety)),envelope_(envelope),policy_(policy),counters_(counters){}
    bool checkMotion(const ob::State *a,const ob::State *b) const override {
        ++policy_.safety_motion_checks;
        const auto reason=rpA5RejectionReason(*safety_,map_,envelope_,point(a),point(b),counters_);
        recordRejection(policy_,reason);
        const bool ok=reason.empty();
        if(ok)++valid_;else ++invalid_;
        return ok;
    }
    bool checkMotion(const ob::State *a,const ob::State *b,std::pair<ob::State*,double> &last) const override {
        if(checkMotion(a,b))return true;
        const Point u=point(a),v=point(b);
        double lo=0.,hi=1.;
        for(int k=0;k<55;++k) {
            const double mid=(lo+hi)/2.;
            if(safety_->validMotion(u,{u.x+mid*(v.x-u.x),u.y+mid*(v.y-u.y)}))lo=mid;else hi=mid;
        }
        last.second=lo;
        if(last.first)setPoint(last.first,{u.x+lo*(v.x-u.x),u.y+lo*(v.y-u.y)});
        return false;
    }
private:
    const Terrain &map_;
    std::shared_ptr<const SafetyPolicy> safety_;
    const RpA5SlopeEnvelope &envelope_;
    PolicyCounters &policy_;
    RpA5Counters &counters_;
};
}

RpA5SlopeEnvelope::RpA5SlopeEnvelope(const Terrain &terrain) {
    const auto box=terrain.bounds();
    nx_=terrain.nx()-1;ny_=terrain.ny()-1;
    x0_=box.xmin;y0_=box.ymin;dx_=(box.xmax-box.xmin)/nx_;dy_=(box.ymax-box.ymin)/ny_;
    const double nan=std::numeric_limits<double>::quiet_NaN();
    std::vector<double> z(static_cast<std::size_t>(nx_+1)*(ny_+1),nan);
    for(int j=0;j<=ny_;++j)for(int i=0;i<=nx_;++i) {
        double height=0.;
        if(terrain.height({i==nx_?box.xmax:x0_+i*dx_,j==ny_?box.ymax:y0_+j*dy_},height))z[static_cast<std::size_t>(j)*(nx_+1)+i]=height;
    }
    // Largest gradient of the two triangles of each cell (SW-SE-NE and SW-NE-NW), as in
    // Terrain::maximumSweptSlope; a node without a height makes its cells unknown.
    std::vector<double> gradient(static_cast<std::size_t>(nx_)*ny_,INF);
    for(int j=0;j<ny_;++j)for(int i=0;i<nx_;++i) {
        const double sw=z[static_cast<std::size_t>(j)*(nx_+1)+i],se=z[static_cast<std::size_t>(j)*(nx_+1)+i+1],
            nw=z[static_cast<std::size_t>(j+1)*(nx_+1)+i],ne=z[static_cast<std::size_t>(j+1)*(nx_+1)+i+1];
        if(std::isnan(sw)||std::isnan(se)||std::isnan(nw)||std::isnan(ne))continue;
        const double value=std::max(std::hypot((se-sw)/dx_,(ne-se)/dy_),std::hypot((ne-nw)/dx_,(nw-sw)/dy_));
        if(std::isfinite(value))gradient[static_cast<std::size_t>(j)*nx_+i]=value;
    }
    // A triangle within the footprint radius of a point lies in a cell at most
    // floor(radius/spacing)+1 indices away from the point's cell; certifies() samples the
    // segment at half-cell steps, which can miss the point's cell by one more index.
    const double spacing=std::min(dx_,dy_);
    reach_=static_cast<int>(std::ceil(terrain.radius()/spacing))+2;
    step_=.5*spacing;
    std::vector<double> rows(gradient.size());
    for(int j=0;j<ny_;++j)for(int i=0;i<nx_;++i) {
        double value=0.;
        for(int k=std::max(0,i-reach_);k<=std::min(nx_-1,i+reach_);++k)value=std::max(value,gradient[static_cast<std::size_t>(j)*nx_+k]);
        rows[static_cast<std::size_t>(j)*nx_+i]=value;
    }
    envelope_.resize(gradient.size());
    for(int j=0;j<ny_;++j)for(int i=0;i<nx_;++i) {
        double value=0.;
        for(int k=std::max(0,j-reach_);k<=std::min(ny_-1,j+reach_);++k)value=std::max(value,rows[static_cast<std::size_t>(k)*nx_+i]);
        envelope_[static_cast<std::size_t>(j)*nx_+i]=value;
    }
}
bool RpA5SlopeEnvelope::certifies(Point a,Point b,double slopeCapDeg) const {
    // Strictly inside the cap: the envelope is computed from re-read heights, not bit-equal to Terrain's.
    if(slopeCapDeg!=cap_){cap_=slopeCapDeg;limit_=std::tan(slopeCapDeg*std::acos(-1.)/180.)*(1.-1e-9);}
    const double limit=limit_;
    const int parts=std::max(1,static_cast<int>(std::ceil(std::hypot(b.x-a.x,b.y-a.y)/step_)));
    for(int k=0;k<=parts;++k) {
        const double t=static_cast<double>(k)/parts;
        const double x=a.x+t*(b.x-a.x),y=a.y+t*(b.y-a.y);
        const double gx=(x-x0_)/dx_,gy=(y-y0_)/dy_;
        if(!(gx>=0.&&gy>=0.&&gx<=nx_&&gy<=ny_))return false;
        const int i=std::min(nx_-1,static_cast<int>(gx)),j=std::min(ny_-1,static_cast<int>(gy));
        if(!(envelope_[static_cast<std::size_t>(j)*nx_+i]<=limit))return false;
    }
    return true;
}
std::string rpA5RejectionReason(const SafetyPolicy &policy,const Terrain &terrain,const RpA5SlopeEnvelope &envelope,
                                Point a,Point b,RpA5Counters &counters) {
    if(!policy.appliesTo(terrain)||!policy.corridorMask().empty())return policy.rejectionReason(a,b);
    auto reason=policy.field()->edgeRejectionReason(a,b,policy.riskCap());
    if(!reason.empty())return reason;
    const bool geometry=same(a,b)?terrain.validState(a):terrain.validMotion(a,b);
    if(!geometry)return "invalid_geometry";
    if(envelope.certifies(a,b,policy.slopeCap())){++counters.slope_certified;return {};}
    ++counters.slope_exact;
    const double slope=terrain.maximumSweptSlope(a,b);
    if(!std::isfinite(slope))return "invalid_geometry";
    return slope>policy.slopeCap()?"swept_slope_above_cap":std::string{};
}

SafetyField rpA5LoadField(const std::string &path) {
    std::ifstream stream(path,std::ios::binary);
    char magic[8];
    if(!stream.read(magic,sizeof magic)||std::memcmp(magic,"SEMFLDB1",sizeof magic)!=0)return SafetyField::load(path);
    std::int32_t size[2];
    double grid[5];   // x0, y0, resolution, xmax, ymax
    if(!stream.read(reinterpret_cast<char*>(size),sizeof size)||!stream.read(reinterpret_cast<char*>(grid),sizeof grid)||
       size[0]<=0||size[1]<=0||static_cast<long long>(size[0])*size[1]>100000000)
        throw std::runtime_error("invalid binary safety field header: "+path);
    const std::size_t cells=static_cast<std::size_t>(size[0])*size[1];
    std::vector<unsigned char> support(cells);
    std::vector<double> risk(cells);
    if(!stream.read(reinterpret_cast<char*>(support.data()),static_cast<std::streamsize>(cells))||
       !stream.read(reinterpret_cast<char*>(risk.data()),static_cast<std::streamsize>(cells*sizeof(double)))||
       stream.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("truncated or oversized binary safety field: "+path);
    return SafetyField(size[0],size[1],grid[0],grid[1],grid[2],grid[3],grid[4],
                       std::vector<int>(support.begin(),support.end()),std::move(risk));
}

unsigned rpA5NextBatch(std::uint64_t stale,unsigned drawn,const RpA5Options &options) {
    const std::uint64_t used=stale+drawn;
    if(!options.batch_growth||used>=options.window_samples)return options.batch_min;
    return static_cast<unsigned>(std::min<std::uint64_t>(std::max<std::uint64_t>(options.batch_min,used),options.window_samples-used));
}

std::vector<Point> rpA5SeedPath(const SafetyPolicy &policy,const RpObjective &objective,const Terrain &terrain,
                                Point start,Point goal,std::uint64_t maxExpansions,std::uint64_t &expansions) {
    expansions=0;
    const auto *toGoal=objective.potentialToGoal();
    if(!toGoal)return {};
    const auto &field=*policy.field();
    const int nx=field.nx(),ny=field.ny();
    const auto box=field.bounds();
    const double res=field.resolution(),cap=policy.riskCap(),lambda=objective.lambda();
    const auto open=[&](int i,int j){return i>=0&&j>=0&&i<nx&&j<ny&&field.supported(i,j)&&field.risk(i,j)<=cap;};
    const auto centre=[&](int cell){return Point{box.xmin+(cell%nx+.5)*res,box.ymin+(cell/nx+.5)*res};};
    const auto cellOf=[&](Point q) {
        return std::clamp(static_cast<int>(std::floor((q.y-box.ymin)/res)),0,ny-1)*nx+
               std::clamp(static_cast<int>(std::floor((q.x-box.xmin)/res)),0,nx-1);
    };
    const int source=cellOf(start),target=cellOf(goal);
    if(!open(source%nx,source/nx)||!open(target%nx,target/nx))return {};
    const std::size_t count=static_cast<std::size_t>(nx)*ny;
    const double nan=std::numeric_limits<double>::quiet_NaN();
    std::vector<double> cost(count,INF),bound(count,nan),height(count,nan);
    std::vector<int> parent(count,-1);
    // Lazily cached per cell: terrain height and lattice-scale cost-to-go at the centre (+inf: unusable).
    const auto prepare=[&](int cell) {
        if(!std::isnan(bound[cell]))return std::isfinite(bound[cell]);
        const Point q=centre(cell);
        double z=0.;
        const double value=toGoal->at(q);
        bound[cell]=std::isfinite(value)&&terrain.height(q,z)?value/KAPPA:INF;
        height[cell]=z;
        return std::isfinite(bound[cell]);
    };
    struct Node { double f,g;int cell; };
    const auto later=[](const Node &a,const Node &b){return a.f>b.f||(a.f==b.f&&a.g<b.g);};   // smallest f, then deepest
    std::priority_queue<Node,std::vector<Node>,decltype(later)> queue(later);
    if(!prepare(source)||!prepare(target))return {};
    cost[source]=0.;queue.push({bound[source],0.,source});
    static const int steps[8][2]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    while(!queue.empty()) {
        const Node node=queue.top();queue.pop();
        if(node.g>cost[node.cell])continue;
        if(node.cell==target)break;
        if(++expansions>maxExpansions)return {};
        const int i=node.cell%nx,j=node.cell/nx;
        for(const auto &step:steps) {
            const int ni=i+step[0],nj=j+step[1];
            if(!open(ni,nj))continue;
            // A diagonal passes through the shared corner, which touches both side cells.
            const bool diagonal=step[0]!=0&&step[1]!=0;
            if(diagonal&&(!open(ni,j)||!open(i,nj)))continue;
            const int next=nj*nx+ni;
            if(!prepare(next))continue;
            const double length=std::hypot(diagonal?std::sqrt(2.)*res:res,height[next]-height[node.cell]);
            const double value=node.g+length*(1.+lambda/40.*.5*(field.risk(i,j)+field.risk(ni,nj)));
            if(value<cost[next]){cost[next]=value;parent[next]=node.cell;queue.push({value+bound[next],value,next});}
        }
    }
    if(!std::isfinite(cost[target]))return {};
    std::vector<Point> path{goal};
    for(int cell=target;cell!=-1;cell=parent[cell])if(!same(path.back(),centre(cell)))path.push_back(centre(cell));
    if(!same(path.back(),start))path.push_back(start);
    std::reverse(path.begin(),path.end());
    return path;
}

RpA5RunResult runRpBitA5(const Terrain &map,std::shared_ptr<const SafetyField> field,Point start,Point goal,
                         const RpA5Options &options) {
    const auto began=Clock::now();
    RpA5RunResult out;
    RpRunResult &result=out.run;
    if(!std::isfinite(options.budget_s)||options.budget_s<=0.||!std::isfinite(options.stop_relative_eps)||
       options.stop_relative_eps<0.||!std::isfinite(options.refinement.round_cost_ratio)||options.refinement.round_cost_ratio<0.||
       !std::isfinite(options.refinement.min_chord_m)||options.refinement.min_chord_m<=0.||options.batch_min==0)
        throw std::runtime_error("invalid RP-BIT* A5 run options");
    if((options.lazy_refine||options.seed)&&!options.refine)throw std::runtime_error("A5 lazy refinement and seed need refinement");
    if(options.seed&&!options.planner.potential)throw std::runtime_error("the A5 seed needs the risk potential");
    auto safety=std::make_shared<const SafetyPolicy>(map,std::move(field),40.,35.);
    if(!safety->validState(start)){result.status="INVALID_START";result.termination="INVALID_QUERY";return out;}
    if(!safety->validState(goal)){result.status="INVALID_GOAL";result.termination="INVALID_QUERY";return out;}
    ompl::RNG::setSeed(static_cast<unsigned>(options.planner_seed));
    out.policy_s=seconds(began);
    // Both outlive the planner that refers to them.
    double samplingBound=INF;   // C4: J of the published path; no sample is drawn at or above it
    std::unique_ptr<RpA5SlopeEnvelope> envelope;
    auto problem=makeRpProblem(map,safety,start,goal,options.planner,result.counters,result.rp);
    const auto &integrator=problem.objective->integrator();
    const double lambda=options.planner.lambda;
    result.lower_bound=problem.objective->lowerBoundStartGoal();
    out.problem_s=seconds(began)-out.policy_s;
    if(options.fast_validity) {
        const auto envelopeStart=Clock::now();
        envelope=std::make_unique<RpA5SlopeEnvelope>(map);
        const RpA5SlopeEnvelope &bound=*envelope;
        problem.si->setStateValidityChecker([safety,&map,&bound,&result,&out](const ob::State *state) {
            ++result.counters.safety_state_checks;const auto q=point(state);
            return rpA5RejectionReason(*safety,map,bound,q,q,out.a5).empty();
        });
        problem.si->setMotionValidator(std::make_shared<A5MotionValidator>(problem.si,map,safety,bound,result.counters,out.a5));
        problem.si->setup();
        out.envelope_s=seconds(envelopeStart);
    }
    if(options.seed) {
        // Same planner object and settings; only the objective (hence the sampler) is replaced.
        problem.pdef->setOptimizationObjective(std::make_shared<A5Objective>(problem.si,problem.objective,samplingBound,out.a5.bounded_samples));
        problem.planner->clear();
        problem.planner->setup();
        if(!problem.planner->getUseKNearest()||problem.planner->getJustInTimeSampling()||!problem.planner->getPruning()||
           problem.planner->getSamplesPerBatch()!=options.planner.samples_per_batch||
           problem.planner->getRewireFactor()!=options.planner.rewire_factor)
            throw std::runtime_error("requested RP-BIT* policy is not effective after the A5 objective was installed");
    }
    result.setup_s=seconds(began);

    std::vector<Point> pending;
    double pendingCost=INF,pendingElapsed=0.,published=INF,reference=INF;
    bool hasPending=false,converged=false,transitioned=false;
    unsigned lastBatch=0;
    // Random samples of the batch BIT* will open next, has open, and has just exhausted.
    unsigned current=options.planner.samples_per_batch,drawn=0,completed=0;
    std::uint64_t stale=0;
    double refinedRaw=INF;   // C3: lowest unrefined cost among the paths refined so far
    const auto safe=[&](const std::vector<Point> &path) {
        if(path.size()==1)return safety->validState(path[0]);
        for(std::size_t i=1;i<path.size();++i)if(!safety->validMotion(path[i-1],path[i]))return false;
        return !path.empty();
    };
    const auto setBatch=[&](unsigned random){current=random;problem.planner->setSamplesPerBatch(random);};
    const auto publish=[&](std::vector<Point> path,const RpPathCost &cost,const char *origin,double stamp,double raw) {
        if(std::isfinite(published)&&!(cost.objective<published-1e-6*std::max(1.,published)))return;
        RpPublication item;
        item.id=result.publications.size();item.origin=origin;item.elapsed_s=stamp;item.bitstar_cost=raw;
        item.batch=problem.planner->numBatches();item.path=std::move(path);item.cost=cost;
        published=cost.objective;
        if(options.seed)samplingBound=published;
        result.publications.push_back(std::move(item));
    };
    const auto countRefine=[&](const RpRefineResult &refined,Clock::time_point since) {
        result.rp.refine_edge_evaluations+=refined.edge_evaluations;result.rp.refine_validity_checks+=refined.validity_checks;
        result.refine_s+=seconds(since);++result.refinements;
    };
    // Publish the pending BIT* incumbent, refined when requested and in time. With lazy refinement
    // an incumbent whose raw cost has not dropped by eps since the last refined one waits, unless
    // `closing`: then the window would fill at this boundary and the run could stop on an unrefined path.
    const auto process=[&](bool refine,bool closing) {
        if(!hasPending)return;
        if(refine&&options.refine&&options.lazy_refine&&!closing&&!result.publications.empty()&&
           !(pendingCost<refinedRaw*(1.-options.stop_relative_eps))){++out.a5.refine_deferred;return;}
        hasPending=false;
        std::vector<Point> path=pending;
        RpPathCost cost=rpPathCost(integrator,lambda,path);
        const char *origin="ompl_intermediate_solution_callback";
        double stamp=pendingElapsed;
        if(refine&&options.refine) {
            const auto refineStart=Clock::now();
            auto refined=rpRefine(*safety,integrator,lambda,pending,options.refinement);
            countRefine(refined,refineStart);
            refinedRaw=std::min(refinedRaw,pendingCost);
            const double now=seconds(began);
            if(refined.cost.valid&&now<=options.budget_s&&safe(refined.path)) {
                path=std::move(refined.path);cost=refined.cost;origin="rp_refinement";stamp=now;
            }
        }
        if(!cost.valid||!safe(path))throw std::logic_error("BIT* incumbent failed the safety or objective contract");
        publish(std::move(path),cost,origin,stamp,pendingCost);
    };
    if(same(start,goal)) {
        pending={start};pendingCost=0.;pendingElapsed=seconds(began);hasPending=true;process(false,true);
        result.status="EXACT_SOLUTION";result.termination="ANALYTIC_IDENTITY_QUERY";result.actual_wall_s=seconds(began);
        return out;
    }
    if(options.seed) {
        const auto seedStart=Clock::now();
        out.seed_status="no_lattice_path";
        auto lattice=rpA5SeedPath(*safety,*problem.objective,map,start,goal,RP_A5_SEED_EXPANSIONS,out.a5.seed_expansions);
        if(lattice.size()>=2) {
            // Merge the lattice steps with A4's own validated shortcut pass, then refine as usual.
            RpRefineOptions merge=options.refinement;merge.relocate=false;merge.round_turns=false;
            const auto mergeStart=Clock::now();
            auto merged=rpRefine(*safety,integrator,lambda,std::move(lattice),merge);
            countRefine(merged,mergeStart);
            out.seed_status="lattice_path_undefined";
            if(merged.cost.valid) {
                const auto refineStart=Clock::now();
                auto refined=rpRefine(*safety,integrator,lambda,merged.path,options.refinement);
                countRefine(refined,refineStart);
                --result.refinements;   // the merge and the refinement of the seed count as one
                out.seed_status="seed_unsafe";
                if(refined.cost.valid&&safe(refined.path)) {
                    out.seed_status="published";
                    publish(std::move(refined.path),refined.cost,"a5_lattice_seed_refined",seconds(began),INF);
                    refinedRaw=merged.cost.objective;   // the seed's own unrefined cost
                    transitioned=true;++result.batch_transitions;setBatch(options.batch_min);
                }
            }
        }
        out.seed_s=seconds(seedStart);
    }
    problem.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &states,const ob::Cost cost) {
        pending.clear();
        for(auto it=states.rbegin();it!=states.rend();++it)pending.push_back(point(*it));
        if(pending.empty()||!same(pending.front(),start)||!same(pending.back(),goal))
            throw std::runtime_error("BIT* callback endpoint mismatch");
        pendingCost=cost.value();pendingElapsed=seconds(began);hasPending=true;
        // Runtime audit on every real incumbent (states run goal -> start), as in A4.
        const auto exceeds=[](double bound,double value){return bound>value*(1.+1e-9)+1e-9;};
        std::vector<double> toGoal(states.size(),0.);
        for(std::size_t i=1;i<states.size();++i) {
            const auto edge=integrator.evaluate(point(states[i]),point(states[i-1]));
            const double value=edge.length+lambda*edge.exposure/40.;
            toGoal[i]=toGoal[i-1]+value;
            if(exceeds(problem.objective->motionCostHeuristic(states[i],states[i-1]).value(),value))++result.rp.heuristic_violations;
        }
        for(std::size_t i=0;i<states.size();++i) {
            const auto bound=problem.objective->bounds(point(states[i]));
            if(exceeds(bound.h,toGoal[i])||exceeds(bound.g,toGoal.back()-toGoal[i]))++result.rp.heuristic_violations;
        }
        ++result.raw_incumbents;
        if(!transitioned){transitioned=true;++result.batch_transitions;setBatch(options.batch_min);}
        if(!options.refine)process(false,true);
    });
    const auto status=problem.planner->solve(ob::PlannerTerminationCondition([&]() {
        // Work-based triggers. When numBatches() changes, BIT* has exhausted the previous
        // batch and already drawn the new one; a size set now applies to the batch after.
        const unsigned batch=problem.planner->numBatches();
        if(batch!=lastBatch) {
            lastBatch=batch;
            completed=drawn;drawn=current;
            const double refineBefore=result.refine_s;
            process(true,options.window_samples&&stale+completed>=options.window_samples);
            if(options.window_samples&&!result.publications.empty()) {
                if(published<reference*(1.-options.stop_relative_eps)){reference=published;stale=0;}
                else if((stale+=completed)>=options.window_samples)converged=true;
            }
            if(transitioned)setBatch(rpA5NextBatch(stale,drawn,options));
            RpA5Batch entry;
            entry.batch=batch;entry.random_samples=drawn;entry.iterations=problem.planner->numIterations();
            entry.elapsed_s=seconds(began);entry.bitstar_cost=problem.planner->bestCost().value();
            entry.refine_s=result.refine_s-refineBefore;entry.published=published;entry.stale_samples=stale;
            out.batches.push_back(entry);
        }
        if(converged)return true;
        if(options.test_max_iterations&&problem.planner->numIterations()>=options.test_max_iterations)return true;
        return seconds(began)>=options.budget_s;
    }));
    process(seconds(began)<options.budget_s,true);
    result.actual_wall_s=seconds(began);
    result.iterations=problem.planner->numIterations();result.batches=problem.planner->numBatches();
    result.status=!result.publications.empty()?"EXACT_SOLUTION":"NO_SOLUTION";
    (void)status;
    result.termination=converged?"CONVERGED_NO_IMPROVEMENT":
        options.test_max_iterations&&result.iterations>=options.test_max_iterations?"WORK_BUDGET":
        result.actual_wall_s>=options.budget_s?"TIME_BUDGET":
        !result.publications.empty()&&published<=result.lower_bound*(1.+1e-9)+1e-9?"LOWER_BOUND_REACHED":
        "UNKNOWN_NATIVE_PLANNER_STOP";
    return out;
}
}
