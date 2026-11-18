#include "terrain25d/OmplIntegration.hpp"
#include <ompl/base/ScopedState.h>
#include <ompl/geometric/planners/informedtrees/bitstar/CostHelper.h>
#include <ompl/geometric/planners/informedtrees/bitstar/SearchQueue.h>
#include <ompl/geometric/planners/informedtrees/bitstar/Vertex.h>
#include <iostream>

using namespace terrain25d;
using BIT=ompl::geometric::BITstar;
void check(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
void near(double a,double b) {check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-8+1e-8*std::max(a,b),"cost mismatch");}
Terrain fixture() {
    return Terrain(21,21,0,0,1,1,std::vector<double>(441),std::vector<int>(400),
        {{7,10.7,13,11.7}},.2,0,35);
}
void test(bool lifted,bool optimized) {
    auto map=fixture();TraversabilityConfig config;config.lambda=20;config.clearance_comfort_m=2;
    map.enableTraversability(config);if(optimized)map.enableCollisionIndex();
    const Point start{2,10},goal{18,10},detour{10,3};
    check(map.validMotion(start,goal)&&map.validMotion(start,detour)&&map.validMotion(detour,goal),"fixture paths must be safe");
    const auto direct=map.edgeCostBreakdown(start,goal);
    const double witness=map.edgeCostBreakdown(start,detour).objective_value+map.edgeCostBreakdown(detour,goal).objective_value;
    check(distanceXY(start,detour)+distanceXY(detour,goal)>direct.cost_terrain_m,"detour should be longer");
    check(witness<direct.objective_value,"risk objective should prefer safe longer witness");
    PolicyCounters counters;PlannerOptions options;options.state_cache=optimized;options.scalar_cost=optimized;
    auto p=makeProblem(map,lifted,start,goal,counters,100,1.1,100,options);
    ob::ScopedState<> a(p.space),b(p.space),state(p.space);
    setPoint(a.get(),start);setPoint(b.get(),goal);setPoint(state.get(),detour);
    near(p.si->distance(a.get(),b.get()),16);
    near(p.objective->motionCost(a.get(),b.get()).value(),direct.objective_value);
    near(p.objective->motionCostHeuristic(a.get(),b.get()).value(),16);
    near(p.objective->costToGo(state.get(),p.pdef->getGoal().get()).value(),distanceXY(detour,goal));

    // Real pinned OMPL helpers must retain a longer but improving position.
    BIT::ImplicitGraph graph([](){return "traversability";});BIT::CostHelper helper;
    BIT::SearchQueue queue([](){return "traversability";});
    helper.setup(p.objective,&graph);queue.setup(&helper,&graph);
    ob::PlannerInputStates inputs;inputs.use(p.pdef);
    graph.setup(p.si,p.pdef,&helper,&queue,p.planner.get(),inputs);
    auto generation=std::make_shared<const unsigned int>(0);
    auto vertex=std::make_shared<BIT::Vertex>(p.si,&helper,&queue,generation);
    setPoint(vertex->state(),detour);
    graph.registerSolutionCost(ob::Cost(direct.objective_value));
    queue.registerSolutionCost(ob::Cost(direct.objective_value));
    check(!graph.canSampleBePruned(vertex),"J-based pruning lost improving witness");
    check(queue.canPossiblyImproveCurrentSolution(vertex),"J-based queue lost improving witness");
    graph.registerSolutionCost(ob::Cost(direct.cost_terrain_m));
    check(graph.canSampleBePruned(vertex),"negative control: L-based pruning should lose the witness");
    graph.registerSolutionCost(ob::Cost(direct.objective_value));
    auto root=*graph.startVerticesBeginConst();
    vertex->addParent(root,ob::Cost(direct.objective_value+10));root->addChild(vertex);
    check(graph.canVertexBeDisconnected(vertex)&&!graph.canSampleBePruned(vertex),"tree cost incorrectly used as permanent bound");
    root->removeChild(vertex);vertex->removeParent(false);

    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);
    check(!sampler->hasInformedMeasure(),"informed measure falsely exact");
    near(sampler->getInformedMeasure(ob::Cost(16),ob::Cost(direct.objective_value)),400);
    unsigned accepted=0,beyondLength=0;
    for(unsigned k=0;k<300;++k)if(sampler->sampleUniform(state.get(),ob::Cost(16),ob::Cost(direct.objective_value))) {
        const double f=map.solutionLowerBound(start,point(state.get()),goal,lifted);
        check(f>=16-1e-9&&f<direct.objective_value+1e-9,"interval mismatch");
        ++accepted;if(f>direct.cost_terrain_m+1)++beyondLength;
    }
    check(accepted>100&&beyondLength>0,"sampler still uses geometric incumbent limit");
    check(sampler->sampleUniform(state.get(),ob::Cost(direct.objective_value)),"single-bound sampler failed");

    // Exercise true BIT* search, callbacks, cost integration and hard validation.
    unsigned callbacks=0;double best=INFINITY;
    p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &path,ob::Cost cost) {
        double actual=0;for(std::size_t i=1;i<path.size();++i) {
            const auto u=point(path[i-1]),v=point(path[i]);
            check(map.validMotion(u,v),"unsafe incumbent");actual+=map.edgeCostBreakdown(u,v).objective_value;
        }
        near(actual,cost.value());check(cost.value()<=best+1e-8,"J increased");best=cost.value();++callbacks;
    });
    unsigned calls=0;
    const auto status=p.planner->solve(ob::PlannerTerminationCondition([&](){return ++calls>=2000;}));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks,"native planner did not produce exact incumbent");
    check(counters.traversability_motion_cost_calls>0&&counters.from_start_calls>0&&counters.to_goal_calls>0&&
        counters.cost_to_go_calls>0&&counters.sampler_one_bound_calls>0&&counters.sampler_interval_calls>0,
        "objective/heuristic/sampler hooks not effective");
    check(p.planner->getPruning()&&p.planner->getUseKNearest()&&!p.planner->getJustInTimeSampling(),"native policy drift");
    std::cout<<"PASS "<<(lifted?"A3":"A0")<<(optimized?" optimized":" original")
        <<" L_direct="<<direct.cost_terrain_m<<" J_direct="<<direct.objective_value<<" J_witness="<<witness<<'\n';
}
int main() {
    try {
        ompl::RNG::setSeed(811);
        for(bool lifted:{false,true})for(bool optimized:{false,true})test(lifted,optimized);
        auto map=fixture();map.enableTraversability(TraversabilityConfig{});PolicyCounters counters;
        for(const std::string id:{"rrtstar","informed_rrtstar","abitstar","aitstar","lazyprmstar_hauser2015","sem_rrtstar_common_v1"}) {
            bool rejected=false;
            try {makePlannerProblem(map,id,{2,10},{18,10},counters);}
            catch(const std::runtime_error &){rejected=true;}
            check(rejected,"unsupported risk backend admitted by factory");
        }
        bool rejected=false;PlannerOptions options;options.batch_after_first_solution=400;
        try {makePlannerProblem(map,"bitstar_a3",{2,10},{18,10},counters,200,1.1,100,options);}
        catch(const std::runtime_error &){rejected=true;}
        check(rejected,"unsupported risk adaptive-batch policy admitted");
    }
    catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
    return 0;
}
