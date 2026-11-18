#include "terrain25d/OmplIntegration.hpp"
#include "terrain25d/SemExposure.hpp"
#include <ompl/base/ScopedState.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

namespace terrain25d {
namespace {
struct StateMemo {
    std::uint64_t token=0,x_bits=0,y_bits=0;
    bool height_ready=false,has_data=false,from_start_ready=false,to_goal_ready=false;
    double height=0,from_start=0,to_goal=0;
};
// Metadata is derived from XY, not an additional state-space dimension. Native
// OMPL samplers/interpolation still write the same RealVector values array.
class CachedXYStateSpace final : public ob::RealVectorStateSpace {
public:
    struct StateType final : ob::RealVectorStateSpace::StateType {
        mutable StateMemo memo;
    };
    CachedXYStateSpace():ob::RealVectorStateSpace(2){}
    ob::State *allocState() const override {
        auto *state=new StateType;
        state->values=new double[2];
        return state;
    }
    void freeState(ob::State *state) const override {
        auto *cached=state->as<StateType>();
        delete[] cached->values;
        delete cached;
    }
    void copyState(ob::State *destination,const ob::State *source) const override {
        ob::RealVectorStateSpace::copyState(destination,source);
        // Copy only the state. No metadata may leak across objective lifetimes.
        destination->as<StateType>()->memo={};
    }
};
std::uint64_t bits(double value) {
    std::uint64_t result;
    static_assert(sizeof(result)==sizeof(value),"64-bit coordinates required");
    std::memcpy(&result,&value,sizeof(result));
    return result;
}
StateMemo &memoFor(const ob::State *state,std::uint64_t token) {
    auto *cached=state->as<CachedXYStateSpace::StateType>();
    auto &memo=cached->memo;
    const auto x=bits(cached->values[0]),y=bits(cached->values[1]);
    // This check is mandatory at every lookup: upstream can mutate values
    // directly, deserialize, sample or interpolate without calling setPoint.
    if(memo.token!=token || memo.x_bits!=x || memo.y_bits!=y) {
        memo={};memo.token=token;memo.x_bits=x;memo.y_bits=y;
    }
    return memo;
}
std::uint64_t nextCacheToken() {
    static std::atomic<std::uint64_t> next{1};
    const auto token=next.fetch_add(1,std::memory_order_relaxed);
    if(token==0)throw std::runtime_error("state cache token exhausted");
    return token;
}
bool finitePoint(Point p) { return std::isfinite(p.x)&&std::isfinite(p.y); }
void recordSafetyRejection(PolicyCounters *c,const std::string &reason) {
    if(!c || reason.empty())return;
    if(reason=="missing_sem_support")++c->safety_support_rejections;
    else if(reason=="sem_risk_above_cap")++c->safety_risk_rejections;
    else if(reason=="swept_slope_above_cap")++c->safety_slope_rejections;
    else if(reason=="outside_corridor")++c->safety_corridor_rejections;
    else ++c->safety_geometry_rejections;
}
}
Point point(const ob::State *state) {
    auto s=state->as<ob::RealVectorStateSpace::StateType>();return {s->values[0],s->values[1]};
}
void setPoint(ob::State *state,Point p) {
    auto s=state->as<ob::RealVectorStateSpace::StateType>();s->values[0]=p.x;s->values[1]=p.y;
}
bool SingletonGoal::isSatisfied(const ob::State *s) const {return isSatisfied(s,nullptr);}
bool SingletonGoal::isSatisfied(const ob::State *s,double *distance) const {
    Point a=point(s),b=point(getState());
    if(distance) *distance=distanceXY(a,b);
    return a.x==b.x && a.y==b.y;
}
bool TerrainMotionValidator::checkMotion(const ob::State *a,const ob::State *b) const {
    bool ok;
    if(safety_) {
        if(counters_)++counters_->safety_motion_checks;
        const auto reason=safety_->rejectionReason(point(a),point(b));
        recordSafetyRejection(counters_,reason);ok=reason.empty();
    } else ok=map_.validMotion(point(a),point(b));
    if(ok) ++valid_;else ++invalid_;
    return ok;
}
bool TerrainMotionValidator::checkMotion(const ob::State *a,const ob::State *b,std::pair<ob::State*,double> &last) const {
    // On success the API requires both output arguments to remain untouched.
    if(checkMotion(a,b)) return true;
    Point u=point(a),v=point(b);
    double lo=0,hi=1;
    // Monotone predicate is validity of the ENTIRE prefix capsule, not just a point.
    for(int k=0;k<55;++k) {
        double mid=(lo+hi)/2;
        Point end{u.x+mid*(v.x-u.x),u.y+mid*(v.y-u.y)};
        if(safety_?safety_->validMotion(u,end):map_.validMotion(u,end)) lo=mid;else hi=mid;
    }
    last.second=lo;
    if(last.first) setPoint(last.first,{u.x+lo*(v.x-u.x),u.y+lo*(v.y-u.y)});
    return false;
}
TerrainInformedSampler::TerrainInformedSampler(const ob::ProblemDefinitionPtr &p,unsigned int attempts,
    const Terrain &map,bool lifted,Point start,Point goal,PolicyCounters &counters,bool queryCache,
    std::shared_ptr<const SafetyPolicy> safety,bool atlas)
    :ob::InformedSampler(p,attempts),map_(map),lifted_(lifted),start_(start),goal_(goal),counters_(counters),
     query_cache_enabled_(queryCache),safety_(std::move(safety)),atlas_enabled_(atlas) {
    if(atlas_enabled_&&!safety_)throw std::runtime_error("safety atlas requires a safety policy");
    if(query_cache_enabled_) {
        cached_objective_=dynamic_cast<const TerrainObjective*>(opt_.get());
        cached_state_cache_enabled_=cached_objective_&&cached_objective_->stateCacheEnabled();
    }
}
bool TerrainInformedSampler::sampleUniform(ob::State *state,const ob::Cost &max) {
    ++counters_.sampler_one_bound_calls;
    return sampleUniform(state,ob::Cost(0),max);
}
bool TerrainInformedSampler::sampleUniform(ob::State *state,const ob::Cost &minCost,const ob::Cost &maxCost) {
    ++counters_.sampler_interval_calls;
    double lo=minCost.value(),hi=maxCost.value();
    auto fail=[&](std::uint64_t &reason){++reason;++counters_.exhausted_calls;return false;};
    // Invalid intervals must neither read nor replace the one-entry bound cache.
    if(std::isnan(lo)||std::isnan(hi)||lo<0||hi<=lo||hi<=0||!std::isfinite(lo))
        return fail(counters_.sampler_invalid_interval_calls);
    bool finite=std::isfinite(hi);
    if(finite) ++counters_.sampler_finite_bound_calls;
    const auto *objective=query_cache_enabled_?cached_objective_:dynamic_cast<const TerrainObjective*>(opt_.get());
    const bool cache=query_cache_enabled_?cached_state_cache_enabled_:(objective&&objective->stateCacheEnabled());
    Rect bounds{};double major=0,minor=0,cosTheta=1,sinTheta=0;bool ellipse=false;
    Point center{};
    if(query_cache_enabled_) {
        if(!query_geometry_.ready) {
            auto &query=query_geometry_;
            query.focal=distanceXY(start_,goal_);
            query.lower_bound=cache?objective->queryLowerBound():map_.lowerBound(start_,goal_,lifted_);
            query.bounds=map_.bounds();
            query.area=(query.bounds.xmax-query.bounds.xmin)*(query.bounds.ymax-query.bounds.ymin);
            query.cos_theta=query.focal>0?(goal_.x-start_.x)/query.focal:1;
            query.sin_theta=query.focal>0?(goal_.y-start_.y)/query.focal:0;
            query.center={(start_.x+goal_.x)/2,(start_.y+goal_.y)/2};
            query.ready=true;++counters_.sampler_query_cache_rebuilds;
        } else ++counters_.sampler_query_cache_hits;
        // This is the legacy numeric guard, not an optimality certificate.
        if(finite && hi<=query_geometry_.lower_bound)
            return fail(counters_.sampler_lower_bound_rejected_calls);
        const auto upperBits=bits(hi);
        if(!bound_geometry_.ready || bound_geometry_.upper_bits!=upperBits) {
            auto &bound=bound_geometry_;
            bound.upper_bits=upperBits;bound.major=hi/2;bound.minor=0;
            if(finite)bound.minor=.5*std::sqrt(std::max(0.,(hi-query_geometry_.focal)*(hi+query_geometry_.focal)));
            bound.ellipse=finite && std::acos(-1.)*bound.major*bound.minor<query_geometry_.area;
            bound.ready=true;++counters_.sampler_bound_cache_rebuilds;
        } else ++counters_.sampler_bound_cache_hits;
        bounds=query_geometry_.bounds;major=bound_geometry_.major;minor=bound_geometry_.minor;
        ellipse=bound_geometry_.ellipse;cosTheta=query_geometry_.cos_theta;sinTheta=query_geometry_.sin_theta;
        center=query_geometry_.center;
    } else {
        // Preserve the original proposal calculations, with no cached bounds or
        // changed arithmetic. In particular, never round the caller's upper bound.
        const double focal=distanceXY(start_,goal_);
        if(finite && hi<=(cache?objective->queryLowerBound():map_.lowerBound(start_,goal_,lifted_)))
            return fail(counters_.sampler_lower_bound_rejected_calls);
        bounds=map_.bounds();
        const double area=(bounds.xmax-bounds.xmin)*(bounds.ymax-bounds.ymin);
        major=hi/2;
        if(finite)minor=.5*std::sqrt(std::max(0.,(hi-focal)*(hi+focal)));
        ellipse=finite && std::acos(-1.)*major*minor<area;
        cosTheta=focal>0?(goal_.x-start_.x)/focal:1;
        sinTheta=focal>0?(goal_.y-start_.y)/focal:0;
        center={(start_.x+goal_.x)/2,(start_.y+goal_.y)/2};
    }
    // Atlas is only a proposal superset. Geometry and exact swept slope remain
    // the state validator's responsibility; do not discard partially valid cells.
    bool useAtlas=false;
    if(atlas_enabled_) {
        const double area=safety_->atlasArea();
        if(area<=0)return fail(counters_.sampler_proposal_limit_calls);
        const double domainArea=(bounds.xmax-bounds.xmin)*(bounds.ymax-bounds.ymin);
        const double ellipseArea=finite?std::acos(-1.)*major*minor:std::numeric_limits<double>::infinity();
        useAtlas=area<=std::min(domainArea,ellipseArea);
    }
    for(unsigned int attempt=0;attempt<numIters_;++attempt) {
        ++counters_.sample_proposals;
        if(finite)++counters_.finite_bound_sample_proposals;
        Point q{};
        if(useAtlas) {
            ++counters_.atlas_sample_proposals;
            // Named draws avoid unspecified C++ argument evaluation order.
            const double areaUnit=rng_.uniform01(),xUnit=rng_.uniform01(),yUnit=rng_.uniform01();
            q=safety_->sampleAtlas(areaUnit,xUnit,yUnit);
        } else if(ellipse) {
            ++counters_.ellipse_sample_proposals;
            double r=std::sqrt(rng_.uniform01()),angle=2*std::acos(-1.)*rng_.uniform01();
            double x=r*major*std::cos(angle),y=r*minor*std::sin(angle);
            q={center.x+cosTheta*x-sinTheta*y,
               center.y+sinTheta*x+cosTheta*y};
        } else q={rng_.uniformReal(bounds.xmin,bounds.xmax),rng_.uniformReal(bounds.ymin,bounds.ymax)};
        if(q.x<bounds.xmin||q.x>bounds.xmax||q.y<bounds.ymin||q.y>bounds.ymax) {
            ++counters_.rejected_bounds;if(finite)++counters_.finite_bound_rejected_bounds;continue;
        }
        double z=0;
        if(cache) {
            setPoint(state,q);
            if(!objective->stateHeight(state,z)) {++counters_.rejected_nodata;if(finite)++counters_.finite_bound_rejected_nodata;continue;}
        } else {
            if(!map_.height(q,z)) {++counters_.rejected_nodata;if(finite)++counters_.finite_bound_rejected_nodata;continue;}
            setPoint(state,q);
        }
        // Exercise the separate objective cost-to-go hook here. BIT* itself calls
        // motionCostHeuristic to each explicit goal; see the pinned source audit.
        const double estimate=opt_->motionCostHeuristic(probDefn_->getStartState(0),state).value()+
            opt_->costToGo(state,probDefn_->getGoal().get()).value();
        if(estimate<lo || estimate>=hi) {++counters_.rejected_heuristic;if(finite)++counters_.finite_bound_rejected_heuristic;continue;}
        ++counters_.samples_returned;
        if(finite)++counters_.finite_bound_samples_returned;
        // Footprint validity is checked once by ImplicitGraph::updateSamples.
        return true;
    }
    return fail(counters_.sampler_proposal_limit_calls);
}
double TerrainInformedSampler::getInformedMeasure(const ob::Cost&) const {
    ++counters_.measure_calls;
    return space_->getMeasure();
}
double TerrainInformedSampler::getInformedMeasure(const ob::Cost &min,const ob::Cost &max) const {
    ++counters_.measure_calls;
    // The interval upper bound is not the difference of equal fallback areas.
    return max.value()<=min.value()?0:space_->getMeasure();
}
TerrainObjective::TerrainObjective(const ob::SpaceInformationPtr &si,const Terrain &map,bool lifted,
    Point start,Point goal,unsigned int attempts,PolicyCounters &counters,bool nativeEuclideanSampler,PlannerOptions options)
    :ob::OptimizationObjective(si),map_(map),lifted_(lifted),start_(start),goal_(goal),
     attempts_(attempts),counters_(counters),nativeEuclideanSampler_(nativeEuclideanSampler),options_(options) {
    if(lifted_ && nativeEuclideanSampler_)throw std::runtime_error("native Euclidean sampler cannot implement I_phi");
    if(options_.sem_exposure) {
        const auto &config=*options_.sem_exposure;
        if(!std::isfinite(config.lambda) || config.lambda<0. || config.risk_scale!=40. ||
           !options_.safety || options_.safety->riskCap()!=40. || options_.safety->slopeCap()!=35. ||
           !options_.safety->corridorMask().empty() || map_.traversabilityEnabled() ||
           !lifted_ || nativeEuclideanSampler_)
            throw std::runtime_error("SEM exposure requires nonnegative finite lambda, scale40 and full safe cap40/slope35 A3 domain");
    }
    if(options_.safety && map_.traversabilityEnabled())
        throw std::runtime_error("safe length requires the surface-length objective, without soft risk penalties");
    if(options_.safety_atlas_sampling && (!options_.safety || !lifted_ || nativeEuclideanSampler_))
        throw std::runtime_error("safe atlas requires a lifted custom informed sampler and a safety policy");
    if(options_.state_cache) {
        if(dynamic_cast<const CachedXYStateSpace*>(si->getStateSpace().get())==nullptr)
            throw std::runtime_error("state cache requires CachedXYStateSpace allocation");
        cache_token_=nextCacheToken(); // Includes map, fixed query and lifted policy identity.
        if(lifted_) {
            start_has_data_=map_.height(start_,start_height_);
            goal_has_data_=map_.height(goal_,goal_height_);
            counters_.height_cache_misses+=2;
        }
    }
    if(map_.traversabilityEnabled() && nativeEuclideanSampler_)
        throw std::runtime_error("traversability requires the audited custom informed sampler");
    description_=options_.sem_exposure?"surface_length_sem_exposure":
        map_.traversabilityEnabled()?"surface_length_traversability":"piecewise_affine_surface_length";
    setCostToGoHeuristic([this](const ob::State *s,const ob::Goal*) {
        ++counters_.cost_to_go_calls;
        double value=options_.state_cache?fixedLowerBound(s,false):map_.lowerBound(point(s),goal_,lifted_);
        return ob::Cost(std::isfinite(value)&&value>0?std::nextafter(value,0.):value);
    });
}
bool TerrainObjective::stateHeight(const ob::State *state,double &height) const {
    if(!options_.state_cache)return map_.height(point(state),height);
    auto &memo=memoFor(state,cache_token_);
    if(memo.height_ready)++counters_.height_cache_hits;
    else {
        ++counters_.height_cache_misses;
        memo.has_data=map_.height(point(state),memo.height);
        memo.height_ready=true;
    }
    height=memo.height;
    return memo.has_data;
}
double TerrainObjective::queryLowerBound() const {
    if(!options_.state_cache)return map_.lowerBound(start_,goal_,lifted_);
    if(!finitePoint(start_)||!finitePoint(goal_))return std::numeric_limits<double>::infinity();
    if(!lifted_)return distanceXY(start_,goal_);
    if(!start_has_data_||!goal_has_data_)return std::numeric_limits<double>::infinity();
    return std::hypot(distanceXY(start_,goal_),start_height_-goal_height_);
}
double TerrainObjective::fixedLowerBound(const ob::State *state,bool fromStart) const {
    auto &memo=memoFor(state,cache_token_);
    auto &ready=fromStart?memo.from_start_ready:memo.to_goal_ready;
    auto &value=fromStart?memo.from_start:memo.to_goal;
    if(ready) {++counters_.fixed_heuristic_cache_hits;return value;}
    ++counters_.fixed_heuristic_cache_misses;
    const Point q=point(state),fixed=fromStart?start_:goal_;
    value=std::numeric_limits<double>::infinity();
    if(finitePoint(q)&&finitePoint(fixed)) {
        const double xy=fromStart?distanceXY(fixed,q):distanceXY(q,fixed);
        if(!lifted_)value=xy;
        else {
            double z=0;
            const bool fixedHasData=fromStart?start_has_data_:goal_has_data_;
            if(fixedHasData && stateHeight(state,z))
                value=std::hypot(xy,fromStart?start_height_-z:z-goal_height_);
        }
    }
    ready=true;
    return value;
}
double TerrainObjective::cachedLowerBound(const ob::State *a,const ob::State *b) const {
    const auto u=point(a),v=point(b);
    if(u.x==start_.x&&u.y==start_.y)return fixedLowerBound(b,true);
    if(v.x==goal_.x&&v.y==goal_.y)return fixedLowerBound(a,false);
    if(!finitePoint(u)||!finitePoint(v))return std::numeric_limits<double>::infinity();
    if(!lifted_)return distanceXY(u,v);
    double za=0,zb=0;
    if(!stateHeight(a,za)||!stateHeight(b,zb))return std::numeric_limits<double>::infinity();
    return std::hypot(distanceXY(u,v),za-zb);
}
ob::Cost TerrainObjective::motionCost(const ob::State *a,const ob::State *b) const {
    ++counters_.motion_cost_calls;
    if(options_.sem_exposure) {
        ++counters_.sem_exposure_motion_cost_calls;
        const auto &config=*options_.sem_exposure;
        // Lambda zero deliberately follows the exact legacy cost path below.
        // This preserves fixed-work samples and parent ordering for the control.
        if(config.lambda>0.) {
            const auto u=point(a),v=point(b);
            const double length=options_.scalar_cost?map_.surfaceCost(u,v):map_.surfaceEdge(u,v).cost;
            if(!std::isfinite(length))return infiniteCost();
            ++counters_.sem_exposure_integral_calls;
            const auto exposure=semExposureCanonical(map_,*options_.safety->field(),u,v);
            if(!exposure)return infiniteCost();
            const double cost=length+config.lambda*(*exposure/config.risk_scale);
            return std::isfinite(cost)?ob::Cost(cost):infiniteCost();
        }
    }
    if(map_.traversabilityEnabled()) {
        ++counters_.traversability_motion_cost_calls;
        // The same field integral drives both allocation policies. Materialize
        // the lifted polyline only in the original (non-scalar) policy.
        std::vector<Point3> surface;
        const auto edge=map_.edgeCostBreakdown(point(a),point(b),options_.scalar_cost?nullptr:&surface);
        return edge.has_data && std::isfinite(edge.objective_value)?ob::Cost(edge.objective_value):infiniteCost();
    }
    // RRT* delay-CC queries costs before validating candidate connections.
    // Undefined NoData cost is +infinity internally, never a fabricated height;
    // finite cost across an obstacle still does not imply collision validity.
    if(options_.scalar_cost) {
        const double cost=map_.surfaceCost(point(a),point(b));
        return std::isfinite(cost)?ob::Cost(cost):infiniteCost();
    }
    const auto edge=map_.surfaceEdge(point(a),point(b));
    return edge.has_data && std::isfinite(edge.cost)?ob::Cost(edge.cost):infiniteCost();
}
ob::Cost TerrainObjective::motionCostHeuristic(const ob::State *a,const ob::State *b) const {
    ++counters_.motion_heuristic_calls;
    auto u=point(a),v=point(b);
    if(u.x==start_.x&&u.y==start_.y) ++counters_.from_start_calls;
    if(v.x==goal_.x&&v.y==goal_.y) ++counters_.to_goal_calls;
    double value=options_.state_cache?cachedLowerBound(a,b):map_.lowerBound(u,v,lifted_);
    // One ulp downward protects exact lower-bound equalities from upward roundoff.
    return ob::Cost(std::isfinite(value)&&value>0?std::nextafter(value,0.):value);
}
ob::InformedSamplerPtr TerrainObjective::allocInformedStateSampler(const ob::ProblemDefinitionPtr &p,unsigned int attempts) const {
    ++counters_.sampler_allocations;
    if(nativeEuclideanSampler_)
        return std::make_shared<ob::PathLengthDirectInfSampler>(p,std::min(attempts,attempts_));
    return std::make_shared<TerrainInformedSampler>(p,std::min(attempts,attempts_),map_,lifted_,start_,goal_,counters_,
        options_.sampler_query_cache,options_.safety,options_.safety_atlas_sampling);
}
static PlanningProblem makeBaseProblem(const Terrain &map,bool lifted,Point start,Point goal,PolicyCounters &counters,
    unsigned int batch,double rewire,unsigned int attempts,bool nativeEuclideanSampler,PlannerOptions options) {
    if(batch==0||attempts==0||!std::isfinite(rewire)||rewire<=1) throw std::runtime_error("invalid planner parameters");
    if(options.safety && !options.safety->appliesTo(map))
        throw std::runtime_error("safety policy belongs to a different terrain instance");
    PlanningProblem p;
    std::shared_ptr<ob::RealVectorStateSpace> space;
    if(options.state_cache)space=std::make_shared<CachedXYStateSpace>();
    else space=std::make_shared<ob::RealVectorStateSpace>(2);
    auto box=map.bounds(); ob::RealVectorBounds bounds(2);
    bounds.setLow(0,box.xmin);bounds.setHigh(0,box.xmax);bounds.setLow(1,box.ymin);bounds.setHigh(1,box.ymax);
    space->setBounds(bounds);p.space=space;
    p.si=std::make_shared<ob::SpaceInformation>(space);
    p.si->setStateValidityChecker([&map,safety=options.safety,&counters](const ob::State *s){
        if(!safety)return map.validState(point(s));
        ++counters.safety_state_checks;
        const auto q=point(s);const auto reason=safety->rejectionReason(q,q);
        recordSafetyRejection(&counters,reason);return reason.empty();
    });
    p.si->setMotionValidator(std::make_shared<TerrainMotionValidator>(p.si,map,options.safety,&counters));p.si->setup();
    p.pdef=std::make_shared<ob::ProblemDefinition>(p.si);
    ob::ScopedState<> s(space),g(space);setPoint(s.get(),start);setPoint(g.get(),goal);
    p.pdef->addStartState(s.get());
    auto singleton=std::make_shared<SingletonGoal>(p.si);singleton->setState(g.get());
    singleton->setThreshold(1e-9);p.pdef->setGoal(singleton);
    p.objective=std::make_shared<TerrainObjective>(p.si,map,lifted,start,goal,attempts,counters,nativeEuclideanSampler,options);
    p.pdef->setOptimizationObjective(p.objective);
    return p;
}
PlanningProblem makeProblem(const Terrain &map,bool lifted,Point start,Point goal,PolicyCounters &counters,
    unsigned int batch,double rewire,unsigned int attempts,PlannerOptions options) {
    if(options.batch_after_first_solution)
        throw std::runtime_error("adaptive batch requires makePlannerProblem and onFirstSolution callback");
    auto p=makeBaseProblem(map,lifted,start,goal,counters,batch,rewire,attempts,false,options);
    p.planner=std::make_shared<ompl::geometric::BITstar>(p.si,lifted?"BITstar_A3":"BITstar_A0");
    p.planner->setSamplesPerBatch(batch);p.planner->setRewireFactor(rewire);
    p.planner->setUseKNearest(true);p.planner->setJustInTimeSampling(false);
    p.planner->setPruning(true);p.planner->setDropSamplesOnPrune(false);
    p.planner->setConsiderApproximateSolutions(false);
    p.planner->setProblemDefinition(p.pdef);p.planner->setup();
    if(!p.planner->getUseKNearest()||p.planner->getJustInTimeSampling()||!p.planner->getPruning()||
        p.planner->getSamplesPerBatch()!=batch||p.planner->getRewireFactor()!=rewire)
        throw std::runtime_error("requested BIT* policy is not effective");
    return p;
}
bool PlannerProblem::onFirstSolution() {
    if(batch_after_first_solution==0 || batch_transition_count!=0)return false;
    if(!bit || abit)throw std::runtime_error("adaptive batch requires BITstar A3");
    bit->setSamplesPerBatch(batch_after_first_solution);
    if(bit->getSamplesPerBatch()!=batch_after_first_solution)
        throw std::runtime_error("adaptive BIT* batch policy is not effective");
    ++batch_transition_count;
    return true;
}
unsigned int PlannerProblem::iterations() const {
    if(bit)return bit->numIterations();
    if(rrt)return rrt->numIterations();
    if(sem)return static_cast<unsigned int>(sem->numIterations());
    if(lazy)return static_cast<unsigned int>(lazy->numIterations());
    if(ait) {
        const auto &properties=ait->getPlannerProgressProperties();
        auto found=properties.find("iterations INTEGER");
        if(found==properties.end())throw std::runtime_error("AIT* iterations progress property missing");
        return static_cast<unsigned int>(std::stoull(found->second()));
    }
    throw std::runtime_error("planner does not expose iterations");
}
std::vector<Point> PlannerProblem::snapshot(const std::vector<const ob::State*> &path) const {
    if(rrt)return dynamic_cast<const RRTInspection&>(*rrt).snapshot(path);
    std::vector<Point> copied;
    if(bit) {
        for(auto it=path.rbegin();it!=path.rend();++it)copied.push_back(point(*it));
        if(copied.empty() || !pdef->getGoal()->isSatisfied(path.front()))
            throw std::runtime_error("BIT-family callback missing actual goal endpoint");
    } else {
        for(const auto *state:path)copied.push_back(point(state));
        if(copied.empty() || point(path.front()).x!=point(pdef->getStartState(0)).x ||
           point(path.front()).y!=point(pdef->getStartState(0)).y || !pdef->getGoal()->isSatisfied(path.back()))
            throw std::runtime_error("start-to-goal callback endpoint mismatch");
    }
    return copied;
}
PlannerProblem makePlannerProblem(const Terrain &map,const std::string &id,Point start,Point goal,
    PolicyCounters &counters,unsigned int batch,double rewire,unsigned int attempts,PlannerOptions options) {
    const bool bit=id=="bitstar_a0"||id=="bitstar_a3";
    const bool abit=id=="abitstar";
    const bool ait=id=="aitstar";
    const bool lazy=id=="lazyprmstar_hauser2015";
    const bool sem=id=="sem_rrtstar_common_v1"||id=="sem_rrtstar_common_t_v1";
    const bool informed=id=="informed_rrtstar";
    if(options.sem_exposure && id!="bitstar_a3")
        throw std::runtime_error("SEM exposure supports only the audited BITstar A3 backend");
    if(options.safety && id!="bitstar_a3" && id!="sem_rrtstar_common_v1")
        throw std::runtime_error("safe length supports BITstar A3 and the matched common SEM control only");
    if(map.traversabilityEnabled() && ((!bit && id!="sem_rrtstar_common_t_v1") || options.batch_after_first_solution))
        throw std::runtime_error("traversability v1 supports only fixed-batch BITstar A0/A3");
    if(id=="sem_rrtstar_common_t_v1" && !map.traversabilityEnabled())
        throw std::runtime_error("SEM common-T requires the traversability objective");
    if(!bit && !abit && !ait && !lazy && !sem && !informed && id!="rrtstar")throw std::runtime_error("unsupported planner: "+id);
    if(options.batch_after_first_solution && id!="bitstar_a3")
        throw std::runtime_error("adaptive batch is supported only for bitstar_a3");
    if(options.batch_after_first_solution && options.batch_after_first_solution==batch)
        throw std::runtime_error("adaptive batch must change the initial batch size");
    // A3 and Informed RRT* are the only configurations that receive the
    // terrain-lifted lower bound/sampler. ABIT* and AIT* intentionally use
    // their paper-profile Euclidean a-priori heuristic and E_xy sampler.
    const bool lifted=id=="bitstar_a3"||informed;
    auto base=makeBaseProblem(map,lifted,start,goal,counters,batch,rewire,attempts,abit||ait,options);
    PlannerProblem p;
    p.space=base.space;p.si=base.si;p.pdef=base.pdef;p.objective=base.objective;
    p.batch_after_first_solution=options.batch_after_first_solution;
    if(bit || abit) {
        if(abit) {
            p.abit=std::make_shared<ompl::geometric::ABITstar>(p.si,"ABITstar_paper_core");
            p.abit->setInitialInflationFactor(1000000.0);
            p.abit->setInflationScalingParameter(10.0);
            p.abit->setTruncationScalingParameter(5.0);
            p.bit=p.abit;
        } else p.bit=std::make_shared<ompl::geometric::BITstar>(p.si,lifted?"BITstar_A3":"BITstar_A0");
        p.bit->setSamplesPerBatch(batch);p.bit->setRewireFactor(rewire);
        p.bit->setUseKNearest(true);p.bit->setJustInTimeSampling(false);
        p.bit->setPruning(true);p.bit->setDropSamplesOnPrune(false);
        p.bit->setConsiderApproximateSolutions(false);
        p.bit->setProblemDefinition(p.pdef);p.bit->setup();p.planner=p.bit;
        if(!p.bit->getUseKNearest()||p.bit->getJustInTimeSampling()||!p.bit->getPruning()||
           p.bit->getSamplesPerBatch()!=batch||p.bit->getRewireFactor()!=rewire)
            throw std::runtime_error("requested BIT-family policy is not effective");
        if(abit && (p.abit->getInitialInflationFactor()!=1000000.0 ||
           p.abit->getInflationScalingParameter()!=10.0 || p.abit->getTruncationScalingParameter()!=5.0))
            throw std::runtime_error("requested ABIT* schedule is not effective");
        return p;
    }
    if(ait) {
        p.ait=std::make_shared<ompl::geometric::AITstar>(p.si);
        p.ait->setBatchSize(batch);p.ait->setRewireFactor(rewire);
        p.ait->setUseKNearest(true);p.ait->enablePruning(true);
        p.ait->trackApproximateSolutions(false);p.ait->setMaxNumberOfGoals(1u);
        p.ait->setProblemDefinition(p.pdef);p.ait->setup();p.planner=p.ait;
        if(p.ait->getBatchSize()!=batch||p.ait->getRewireFactor()!=rewire||
           !p.ait->getUseKNearest()||!p.ait->isPruningEnabled()||
           p.ait->areApproximateSolutionsTracked()||p.ait->getMaxNumberOfGoals()!=1u)
            throw std::runtime_error("requested AIT* policy is not effective");
        return p;
    }
    if(lazy) {
        if(std::abs(rewire-1.1)>1e-12)
            throw std::runtime_error("Hauser Lazy-PRM* comparison fixes the k(n) rewire factor at 1.1");
        p.lazy=std::make_shared<HauserLazyPRMstar>(p.si);
        p.lazy->setProblemDefinition(p.pdef);p.lazy->setup();p.planner=p.lazy;
        return p;
    }
    if(sem) {
        if(attempts!=SemRRTstarCommon::INFORMED_ATTEMPTS)
            throw std::runtime_error("sem_rrtstar_common_v1 fixes informed attempts at 100");
        p.sem=std::make_shared<SemRRTstarCommon>(p.si);
        p.sem->setProblemDefinition(p.pdef);p.sem->setup();p.planner=p.sem;
        return p;
    }
    if(informed)p.rrt=std::make_shared<ObservedRRTstar<ompl::geometric::InformedRRTstar>>(p.si);
    else p.rrt=std::make_shared<ObservedRRTstar<ompl::geometric::RRTstar>>(p.si);
    p.planner=p.rrt;
    // Locked OMPL defaults: range is auto-configured to 0.2*maximumExtent.
    p.rrt->setRange(0.);p.rrt->setGoalBias(.05);p.rrt->setKNearest(true);
    p.rrt->setRewireFactor(rewire);p.rrt->setDelayCC(true);
    p.rrt->setNumSamplingAttempts(attempts);
    p.rrt->setAdmissibleCostToCome(true);
    p.rrt->setTreePruning(informed);p.rrt->setInformedSampling(informed);
    // I_phi has no exact closed-form measure. Keep full-domain measure and
    // k-nearest rewiring; tree pruning and informed sampling remain enabled.
    p.rrt->setPrunedMeasure(false);
    p.rrt->setSampleRejection(false);p.rrt->setNewStateRejection(false);
    // In 1.7.0 even setOrderedSampling(false) emits an error for uniform RRT*;
    // its default is already false. Verify the effective setting below.
    if(informed)p.rrt->setOrderedSampling(false);
    p.rrt->setProblemDefinition(p.pdef);p.rrt->setup();
    if(p.rrt->getTreePruning()!=informed || p.rrt->getInformedSampling()!=informed ||
       p.rrt->getPrunedMeasure() || !p.rrt->getAdmissibleCostToCome() ||
       !p.rrt->getKNearest() || !p.rrt->getDelayCC() || p.rrt->getOrderedSampling() || p.rrt->getRange()<=0)
        throw std::runtime_error("requested RRT* policy is not effective");
    return p;
}
}
