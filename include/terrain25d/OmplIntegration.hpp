#pragma once
#include "terrain25d/HauserLazyPRMstar.hpp"
#include "terrain25d/SemRRTstarCommon.hpp"
#include "terrain25d/Terrain.hpp"
#include "terrain25d/Safety.hpp"
#include <ompl/base/MotionValidator.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/samplers/InformedStateSampler.h>
#include <ompl/base/samplers/informed/PathLengthDirectInfSampler.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/informedtrees/ABITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/util/RandomNumbers.h>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace terrain25d {
namespace ob=ompl::base;
struct PolicyCounters {
    std::uint64_t motion_cost_calls=0, traversability_motion_cost_calls=0;
    std::uint64_t sem_exposure_motion_cost_calls=0, sem_exposure_integral_calls=0;
    std::uint64_t motion_heuristic_calls=0, from_start_calls=0, to_goal_calls=0,
        cost_to_go_calls=0, sampler_allocations=0, sampler_one_bound_calls=0,
        sampler_interval_calls=0, sampler_finite_bound_calls=0,
        sample_proposals=0, samples_returned=0, rejected_bounds=0,
        rejected_nodata=0, rejected_heuristic=0, exhausted_calls=0, measure_calls=0,
        finite_bound_sample_proposals=0, finite_bound_samples_returned=0,
        finite_bound_rejected_bounds=0, finite_bound_rejected_nodata=0, finite_bound_rejected_heuristic=0,
        height_cache_hits=0, height_cache_misses=0,
        fixed_heuristic_cache_hits=0, fixed_heuristic_cache_misses=0,
        sampler_query_cache_hits=0, sampler_query_cache_rebuilds=0,
        sampler_bound_cache_hits=0, sampler_bound_cache_rebuilds=0,
        sampler_invalid_interval_calls=0, sampler_lower_bound_rejected_calls=0,
        sampler_proposal_limit_calls=0,
        safety_state_checks=0, safety_motion_checks=0, safety_geometry_rejections=0,
        safety_support_rejections=0, safety_risk_rejections=0, safety_slope_rejections=0,
        safety_corridor_rejections=0,
        atlas_sample_proposals=0, ellipse_sample_proposals=0;
};
// Additive SEM-risk exposure is distinct from unique-cell path PRI.
struct SemExposureConfig {
    double lambda=1.;
    double risk_scale=40.;
};
// All optimizations are opt-in. The original A3 remains the default control.
struct PlannerOptions {
    bool state_cache=false;
    bool scalar_cost=false;
    unsigned int batch_after_first_solution=0;
    bool sampler_query_cache=false;
    std::shared_ptr<const SafetyPolicy> safety;
    bool safety_atlas_sampling=false;
    std::optional<SemExposureConfig> sem_exposure;
};
Point point(const ob::State *state);
void setPoint(ob::State *state,Point p);

class SingletonGoal final : public ob::GoalState {
public:
    explicit SingletonGoal(const ob::SpaceInformationPtr &si) : ob::GoalState(si) {}
    bool isSatisfied(const ob::State *state) const override;
    bool isSatisfied(const ob::State *state,double *distance) const override;
};
class TerrainMotionValidator final : public ob::MotionValidator {
public:
    TerrainMotionValidator(const ob::SpaceInformationPtr &si,const Terrain &map,
        std::shared_ptr<const SafetyPolicy> safety={},PolicyCounters *counters=nullptr)
        :ob::MotionValidator(si),map_(map),safety_(std::move(safety)),counters_(counters){}
    bool checkMotion(const ob::State *a,const ob::State *b) const override;
    bool checkMotion(const ob::State *a,const ob::State *b,std::pair<ob::State*,double> &last) const override;
private:
    const Terrain &map_;
    std::shared_ptr<const SafetyPolicy> safety_;
    PolicyCounters *counters_;
};
class TerrainObjective;
class TerrainInformedSampler final : public ob::InformedSampler {
public:
    TerrainInformedSampler(const ob::ProblemDefinitionPtr &p,unsigned int attempts,
        const Terrain &map,bool lifted,Point start,Point goal,PolicyCounters &counters,
        bool queryCache=false,std::shared_ptr<const SafetyPolicy> safety={},bool atlas=false);
    bool sampleUniform(ob::State *state,const ob::Cost &max) override;
    bool sampleUniform(ob::State *state,const ob::Cost &min,const ob::Cost &max) override;
    bool hasInformedMeasure() const override { return false; }
    double getInformedMeasure(const ob::Cost &max) const override;
    double getInformedMeasure(const ob::Cost &min,const ob::Cost &max) const override;
private:
    // Test/replay code can set a local RNG seed without changing production RNG
    // allocation or adding instrumentation to the per-proposal hot path.
    friend struct TerrainSamplerReplayAccess;
    struct QueryGeometry {
        bool ready=false;
        Rect bounds{};
        double area=0,focal=0,cos_theta=1,sin_theta=0,lower_bound=0;
        Point center{};
    };
    struct BoundGeometry {
        bool ready=false,ellipse=false;
        std::uint64_t upper_bits=0;
        double major=0,minor=0;
    };
    const Terrain &map_;
    bool lifted_;
    Point start_,goal_;
    PolicyCounters &counters_;
    ompl::RNG rng_;
    bool query_cache_enabled_=false;
    const TerrainObjective *cached_objective_=nullptr;
    bool cached_state_cache_enabled_=false;
    QueryGeometry query_geometry_;
    BoundGeometry bound_geometry_;
    std::shared_ptr<const SafetyPolicy> safety_;
    bool atlas_enabled_=false;
};
class TerrainObjective final : public ob::OptimizationObjective {
public:
    TerrainObjective(const ob::SpaceInformationPtr &si,const Terrain &map,bool lifted,
        Point start,Point goal,unsigned int attempts,PolicyCounters &counters,
        bool nativeEuclideanSampler=false,PlannerOptions options={});
    ob::Cost stateCost(const ob::State*) const override {return identityCost();}
    ob::Cost motionCost(const ob::State *a,const ob::State *b) const override;
    ob::Cost motionCostHeuristic(const ob::State *a,const ob::State *b) const override;
    ob::InformedSamplerPtr allocInformedStateSampler(const ob::ProblemDefinitionPtr &p,unsigned int attempts) const override;
    bool stateHeight(const ob::State *state,double &height) const;
    double queryLowerBound() const;
    bool stateCacheEnabled() const { return options_.state_cache; }
    bool scalarCostEnabled() const { return options_.scalar_cost; }
    bool samplerQueryCacheEnabled() const { return options_.sampler_query_cache; }
    const std::optional<SemExposureConfig> &semExposureConfig() const { return options_.sem_exposure; }
private:
    double cachedLowerBound(const ob::State *a,const ob::State *b) const;
    double fixedLowerBound(const ob::State *state,bool fromStart) const;
    const Terrain &map_;
    bool lifted_;
    Point start_,goal_;
    unsigned int attempts_;
    PolicyCounters &counters_;
    bool nativeEuclideanSampler_;
    PlannerOptions options_;
    std::uint64_t cache_token_=0;
    double start_height_=0,goal_height_=0;
    bool start_has_data_=false,goal_has_data_=false;
};
struct PlanningProblem {
    ob::StateSpacePtr space;
    ob::SpaceInformationPtr si;
    ob::ProblemDefinitionPtr pdef;
    std::shared_ptr<TerrainObjective> objective;
    std::shared_ptr<ompl::geometric::BITstar> planner;
};
PlanningProblem makeProblem(const Terrain &map,bool lifted,Point start,Point goal,
    PolicyCounters &counters,unsigned int batch=200,double rewire=1.1,unsigned int attempts=100,
    PlannerOptions options={});

// Observation only: the algorithm remains the real OMPL Base implementation.
// RRTstar's callback omits both endpoints; recover the actual endpoint states
// from its current incumbent, rather than inventing a connection to nominal goal.
class RRTInspection {
public:
    virtual ~RRTInspection()=default;
    virtual std::vector<Point> snapshot(const std::vector<const ob::State*> &interior) const=0;
    virtual bool identitySolution() const=0;
    virtual void auditTree(const Terrain &map) const=0;
};
template<class Base> class ObservedRRTstar final : public Base, public RRTInspection {
public:
    explicit ObservedRRTstar(const ob::SpaceInformationPtr &si):Base(si){}
    ob::PlannerStatus solve(const ob::PlannerTerminationCondition &ptc) override {
        // Upstream RRTstar skips duplicate sampled states with callbacks enabled
        // and does not recognize a start already at the goal. The zero-edge
        // identity solution is handled explicitly, with its own event provenance.
        if(this->pdef_->getStartStateCount()==1 &&
           this->pdef_->getGoal()->isSatisfied(this->pdef_->getStartState(0)) &&
           this->si_->isValid(this->pdef_->getStartState(0))) {
            identity_=true;
            auto path=std::make_shared<ompl::geometric::PathGeometric>(this->si_);
            path->append(this->pdef_->getStartState(0));
            this->pdef_->addSolutionPath(path,false,0.,this->getName()+"_identity_query");
            const auto &cb=this->pdef_->getIntermediateSolutionCallback();
            if(cb)cb(this,{},this->pdef_->getOptimizationObjective()->identityCost());
            return ob::PlannerStatus::EXACT_SOLUTION;
        }
        return Base::solve(ptc);
    }
    bool identitySolution() const override {return identity_;}
    std::vector<Point> snapshot(const std::vector<const ob::State*> &interior) const override {
        if(identity_)return {point(this->pdef_->getStartState(0))};
        if(!this->bestGoalMotion_)throw std::runtime_error("RRT callback without incumbent");
        std::vector<Point> reverse;
        for(auto *m=this->bestGoalMotion_;m;m=m->parent)reverse.push_back(point(m->state));
        if(reverse.size()<2 || interior.size()!=reverse.size()-2)
            throw std::runtime_error("RRT callback interior size mismatch");
        for(std::size_t i=0;i<interior.size();++i) {
            auto q=point(interior[i]);
            if(q.x!=reverse[i+1].x || q.y!=reverse[i+1].y)
                throw std::runtime_error("RRT callback interior order mismatch");
        }
        return {reverse.rbegin(),reverse.rend()};
    }
    void auditTree(const Terrain &map) const override {
        std::vector<typename Base::Motion*> motions;
        this->nn_->list(motions);
        for(auto *m:motions) {
            std::vector<const typename Base::Motion*> ancestors;
            for(auto *a=m;a;a=a->parent) {
                if(std::find(ancestors.begin(),ancestors.end(),a)!=ancestors.end())
                    throw std::runtime_error("cycle at RRT checkpoint");
                ancestors.push_back(a);
            }
            if(m->parent) {
                if(!map.validMotion(point(m->parent->state),point(m->state)))
                    throw std::runtime_error("invalid RRT checkpoint edge");
                double edge=map.surfaceEdge(point(m->parent->state),point(m->state)).cost;
                double expected=m->parent->cost.value()+edge;
                if(std::abs(m->cost.value()-expected)>1e-8+1e-8*std::abs(expected) ||
                   std::abs(m->incCost.value()-edge)>1e-8+1e-8*std::abs(edge))
                    throw std::runtime_error("RRT parent/child cost mismatch at checkpoint");
            }
        }
    }
private:
    bool identity_=false;
};
struct PlannerProblem {
    ob::StateSpacePtr space;
    ob::SpaceInformationPtr si;
    ob::ProblemDefinitionPtr pdef;
    std::shared_ptr<TerrainObjective> objective;
    ob::PlannerPtr planner;
    std::shared_ptr<ompl::geometric::BITstar> bit;
    std::shared_ptr<ompl::geometric::ABITstar> abit;
    std::shared_ptr<ompl::geometric::AITstar> ait;
    std::shared_ptr<HauserLazyPRMstar> lazy;
    std::shared_ptr<SemRRTstarCommon> sem;
    std::shared_ptr<ompl::geometric::RRTstar> rrt;
    unsigned int batch_after_first_solution=0, batch_transition_count=0;
    bool onFirstSolution();
    unsigned int iterations() const;
    std::vector<Point> snapshot(const std::vector<const ob::State*> &path) const;
};
PlannerProblem makePlannerProblem(const Terrain &map,const std::string &planner,Point start,Point goal,
    PolicyCounters &counters,unsigned int batch=200,double rewire=1.1,unsigned int attempts=100,
    PlannerOptions options={});
}
