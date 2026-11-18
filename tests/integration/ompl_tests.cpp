#include "terrain25d/OmplIntegration.hpp"
#include <ompl/base/ScopedState.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/informedtrees/bitstar/CostHelper.h>
#include <ompl/geometric/planners/informedtrees/bitstar/SearchQueue.h>
#include <ompl/geometric/planners/informedtrees/bitstar/Vertex.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>

using namespace terrain25d;
using BIT=ompl::geometric::BITstar;
void check(bool value,const char *message) {if(!value)throw std::runtime_error(message);}
void near(double a,double b) {check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<1e-8+1e-8*std::max(a,b),"cost mismatch");}
Terrain hill() {
    std::vector<double> h;
    for(int j=0;j<=10;++j)for(int i=0;i<=10;++i)h.push_back(4-std::abs(i-5)*.8);
    return Terrain(11,11,0,0,1,1,h,std::vector<int>(100),{},.1,0,60);
}
void policyTest(bool lifted) {
    auto map=hill(); PolicyCounters counters;
    Point start{1,5},goal{9,5};auto p=makeProblem(map,lifted,start,goal,counters,40);
    ob::ScopedState<> state(p.space),a(p.space),b(p.space);
    setPoint(a.get(),start);setPoint(b.get(),goal);
    check(p.space->getDimension()==2,"state dimension");
    near(p.si->distance(a.get(),b.get()),8);
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);
    check(!sampler->hasInformedMeasure(),"false exact measure claim");
    near(sampler->getInformedMeasure(ob::Cost(9)),100);
    near(sampler->getInformedMeasure(ob::Cost(8.5),ob::Cost(9)),100);
    int success=0;
    for(int k=0;k<600;++k) if(sampler->sampleUniform(state.get(),ob::Cost(8.5),ob::Cost(10.5))) {
        double f=map.solutionLowerBound(start,point(state.get()),goal,lifted);
        check(f>=8.5-1e-10&&f<10.5+1e-10,"informed interval policy");++success;
    }
    check(success>100,"sampler never accepts");
    for(double bad:{-1.,0.,7.,std::numeric_limits<double>::quiet_NaN()})
        check(!sampler->sampleUniform(state.get(),ob::Cost(bad)),"invalid max bound accepted");
    check(!sampler->sampleUniform(state.get(),ob::Cost(12),ob::Cost(11)),"reversed interval accepted");
    check(sampler->sampleUniform(state.get(),ob::Cost(std::numeric_limits<double>::infinity())),"infinite bound");
    check(counters.finite_bound_sample_proposals>0 && counters.sample_proposals>counters.finite_bound_sample_proposals,
        "finite sampling phase must exclude preincumbent proposals");
    check(counters.finite_bound_sample_proposals==counters.finite_bound_samples_returned+
        counters.finite_bound_rejected_bounds+counters.finite_bound_rejected_nodata+counters.finite_bound_rejected_heuristic,
        "finite sampling acceptance denominator mismatch");
    // Real OMPL helper/graph instances: distinguish A0/A3 on f_xy < 9 < f_phi.
    BIT::ImplicitGraph graph([](){return "policy_test";});
    BIT::CostHelper helper; BIT::SearchQueue queue([](){return "policy_test";});
    helper.setup(p.objective,&graph);queue.setup(&helper,&graph);
    ob::PlannerInputStates inputs;inputs.use(p.pdef);
    graph.setup(p.si,p.pdef,&helper,&queue,p.planner.get(),inputs);
    auto approximation=std::make_shared<const unsigned int>(0);
    auto vertex=std::make_shared<BIT::Vertex>(p.si,&helper,&queue,approximation);
    setPoint(vertex->state(),{5,5});
    graph.registerSolutionCost(ob::Cost(9));queue.registerSolutionCost(ob::Cost(9));
    check(graph.canSampleBePruned(vertex)==lifted,"actual OMPL sample pruning policy wrong");
    check(queue.canPossiblyImproveCurrentSolution(vertex)!=lifted,"actual OMPL queue policy wrong");
    near(helper.lowerBoundHeuristicVertex(vertex).value(),map.solutionLowerBound(start,{5,5},goal,lifted));
    // A current tree detour must not permanently remove an improving position.
    graph.registerSolutionCost(ob::Cost(20));
    auto root=*graph.startVerticesBeginConst();
    vertex->addParent(root,ob::Cost(30));root->addChild(vertex);
    near(vertex->getCost().value(),root->getCost().value()+30);
    check(graph.canVertexBeDisconnected(vertex),"tree detour should disconnect");
    check(!graph.canSampleBePruned(vertex),"tree detour incorrectly permanently pruned");
    root->removeChild(vertex);vertex->removeParent(false);
    // Exercise real Vertex cost propagation during a parent replacement.
    auto child=std::make_shared<BIT::Vertex>(p.si,&helper,&queue,approximation);
    vertex->addParent(root,ob::Cost(5));root->addChild(vertex);
    child->addParent(vertex,ob::Cost(3));vertex->addChild(child);
    near(child->getCost().value(),8);
    root->removeChild(vertex);vertex->removeParent(true);
    vertex->addParent(root,ob::Cost(2));root->addChild(vertex);
    // Vertex::addParent cascades cost updates (pinned Vertex.cpp:264).
    near(vertex->getCost().value(),2);near(child->getCost().value(),5);
    vertex->removeChild(child);child->removeParent(false);
    root->removeChild(vertex);vertex->removeParent(false);
    // Dijkstra oracle on a finite graph (not a continuous optimality claim).
    std::vector<Point> points;
    for(int y=1;y<=9;y+=2)for(int x=1;x<=9;x+=2)points.push_back({double(x),double(y)});
    std::vector<double> dist(points.size(),INFINITY);dist.back()=0;
    using Item=std::pair<double,int>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>> q;q.push({0,int(points.size()-1)});
    while(!q.empty()) {auto [cost,i]=q.top();q.pop();if(cost>dist[i])continue;
        for(std::size_t j=0;j<points.size();++j) {double edge=map.surfaceEdge(points[i],points[j]).cost;
            if(cost+edge<dist[j]){dist[j]=cost+edge;q.push({dist[j],int(j)});}}}
    for(std::size_t i=0;i<points.size();++i)check(map.lowerBound(points[i],points.back(),lifted)<=dist[i]+1e-8,"inadmissible lower bound");
    std::cout<<"T11/T12 PASS "<<(lifted?"A3":"A0")<<" sampler, actual OMPL queue/pruning, detour, Dijkstra\n";
    // Both motion-validator overloads including no modification on success.
    setPoint(a.get(),{1,2});setPoint(b.get(),{2,2});
    std::pair<ob::State*,double> last{state.get(),.123};setPoint(state.get(),{3,3});
    check(p.si->getMotionValidator()->checkMotion(a.get(),b.get(),last),"valid motion");
    near(last.second,.123);near(point(state.get()).x,3);
    setPoint(b.get(),{11,2});
    check(!p.si->getMotionValidator()->checkMotion(a.get(),b.get(),last),"invalid motion");
    check(map.validMotion(point(a.get()),point(state.get()))&&last.second<1,"last valid prefix");
    setPoint(state.get(),{goal.x-1e-10,goal.y});check(!p.pdef->getGoal()->isSatisfied(state.get()),"near goal incorrectly exact");
    setPoint(state.get(),goal);check(p.pdef->getGoal()->isSatisfied(state.get()),"singleton goal missed");
    // Incumbents must never increase. PlannerData checks cover exported checkpoint,
    // not every transient internal rewiring.
    double best=INFINITY;int callbacks=0;
    p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &path,const ob::Cost cost){
        check(cost.value()<=best+1e-8,"incumbent increased");best=cost.value();++callbacks;
        check(point(path.front()).x==goal.x && point(path.back()).x==start.x,"callback orientation changed");
    });
    unsigned int calls=0;
    auto status=p.planner->solve(ob::PlannerTerminationCondition([&](){return ++calls>=4000;}));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks>0,"real BIT* did not solve");
    check(counters.sampler_allocations>0&&counters.motion_heuristic_calls>0&&
        counters.from_start_calls>0&&counters.to_goal_calls>0&&counters.cost_to_go_calls>0,"hooks not used");
    ob::PlannerData data(p.si);p.planner->getPlannerData(data);
    std::vector<int> color(data.numVertices());std::vector<unsigned> incoming(data.numVertices());
    std::function<void(unsigned)> visit=[&](unsigned i){check(color[i]!=1,"cycle in exported tree");if(color[i]==2)return;color[i]=1;
        std::vector<unsigned> edges;data.getEdges(i,edges);for(auto j:edges){
            check(map.validMotion(point(data.getVertex(i).getState()),point(data.getVertex(j).getState())),"invalid exported edge");visit(j);}color[i]=2;};
    for(unsigned i=0;i<data.numVertices();++i){std::vector<unsigned> edges;data.getEdges(i,edges);for(auto j:edges)++incoming[j];}
    for(unsigned i=0;i<data.numVertices();++i){check(incoming[i]<=1,"multiple tree parents");visit(i);}
    std::cout<<"T13 PASS monotone incumbent and final PlannerData forest (scope: exported checkpoint)\n";
}
void advancedNativeTest(const std::string &id) {
    auto map=hill();PolicyCounters counters;Point start{1,5},goal{9,5};
    auto p=makePlannerProblem(map,id,start,goal,counters,40,1.1,100);
    check(id=="abitstar"?static_cast<bool>(p.abit):static_cast<bool>(p.ait),"wrong native advanced planner class");
    check(!p.rrt,"advanced planner routed through RRT adapter");
    ob::ScopedState<> a(p.space),b(p.space),sample(p.space);setPoint(a.get(),start);setPoint(b.get(),goal);
    near(p.objective->motionCostHeuristic(a.get(),b.get()).value(),8.);
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);
    check(dynamic_cast<ob::PathLengthDirectInfSampler*>(sampler.get())!=nullptr,
        "advanced comparator reused A3 informed sampler");
    check(sampler->hasInformedMeasure(),"E_xy sampler measure unavailable");
    check(sampler->sampleUniform(sample.get(),ob::Cost(20.)),"E_xy sampler failed");
    int callbacks=0;double best=INFINITY;
    p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &raw,const ob::Cost cost){
        auto path=p.snapshot(raw);check(path.front().x==start.x&&path.front().y==start.y,"callback start mismatch");
        check(path.back().x==goal.x&&path.back().y==goal.y,"callback goal mismatch");
        check(cost.value()<=best+1e-8,"advanced incumbent increased");best=cost.value();++callbacks;
    });
    unsigned checks=0;auto status=p.planner->solve(ob::PlannerTerminationCondition([&](){return ++checks>30000;}));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks>0,"advanced native planner did not solve");
    check(p.iterations()>0,"advanced native planner iterations unavailable");
    std::cout<<"PASS native "<<id<<" class, d_xy/E_xy, callback orientation and exact solution\n";
}
int main() {try {ompl::RNG::setSeed(73129);policyTest(false);policyTest(true);
    advancedNativeTest("abitstar");advancedNativeTest("aitstar");
    auto map=hill();PolicyCounters counters;Point q{2,2};auto p=makeProblem(map,true,q,q,counters,20);
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);ob::ScopedState<> state(p.space);
    check(sampler->sampleUniform(state.get(),ob::Cost(1)),"start=goal sampler");
    unsigned n=0;auto status=p.planner->solve(ob::PlannerTerminationCondition([&](){return ++n>1000;}));
    check(status==ob::PlannerStatus::EXACT_SOLUTION,"start=goal BIT* solve");
    std::cout<<"PASS start=goal and near-goal\n";return 0;
}catch(const std::exception &error){std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}}
