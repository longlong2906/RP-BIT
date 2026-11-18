#include "terrain25d/OmplIntegration.hpp"
#include <ompl/base/ScopedState.h>
#include <ompl/util/Console.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstring>

using namespace terrain25d;
namespace {
void check(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
void near(double a,double b) {
    check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-12*(1+std::abs(b)),"independent plane oracle mismatch");
}
Terrain plane(double slope=.2,bool nodata=false) {
    std::vector<double> heights;
    for(int j=0;j<=10;++j)for(int i=0;i<=10;++i)heights.push_back(slope*i+.1*j);
    if(nodata)heights[5*11+5]=std::numeric_limits<double>::quiet_NaN();
    return Terrain(11,11,0,0,1,1,heights,std::vector<int>(100),{},.1,0,70);
}
Terrain hill(bool obstacle=false) {
    std::vector<double> heights;
    for(int j=0;j<=10;++j)for(int i=0;i<=10;++i)heights.push_back(4-std::abs(i-5)*.8);
    const std::vector<Rect> obstacles=obstacle?std::vector<Rect>{{4.6,4.3,5.4,5.7}}:std::vector<Rect>{};
    return Terrain(11,11,0,0,1,1,heights,std::vector<int>(100),obstacles,.1,0,70);
}
double planeDistance(Point a,Point b,double slope=.2) {
    return std::hypot(std::hypot(a.x-b.x,a.y-b.y),slope*(a.x-b.x)+.1*(a.y-b.y));
}
void lifecycle() {
    auto map=plane();PolicyCounters c;
    PlannerOptions options;options.state_cache=true;
    const Point start{1,2},goal{9,8};auto p=makeProblem(map,true,start,goal,c,50,1.1,100,options);
    check(p.space->getDimension()==2,"metadata changed state dimension");
    check(p.space->getSerializationLength()==2*sizeof(double),"metadata leaked into state serialization");
    ob::ScopedState<> s(p.space),a(p.space),b(p.space),copy(p.space);
    setPoint(a.get(),start);setPoint(b.get(),goal);setPoint(s.get(),{3.2,4.1});
    auto verify=[&](ob::State *state) {
        const auto q=point(state);
        near(p.objective->motionCostHeuristic(a.get(),state).value(),planeDistance(start,q));
        near(p.objective->costToGo(state,p.pdef->getGoal().get()).value(),planeDistance(q,goal));
    };
    verify(s.get());const auto queries=map.counters().height_queries;
    for(int k=0;k<100;++k)verify(s.get());
    check(map.counters().height_queries==queries,"repeated fixed heuristic did not reuse cached values");
    check(c.fixed_heuristic_cache_hits>=200,"fixed heuristic cache counters absent");
    p.space->copyState(copy.get(),s.get());verify(copy.get());
    setPoint(s.get(),{7.3,2.2});verify(s.get());verify(copy.get());
    p.space->interpolate(a.get(),b.get(),.3,s.get());verify(s.get());
    p.space->interpolate(s.get(),b.get(),.2,s.get());verify(s.get());
    std::vector<char> serialized(p.space->getSerializationLength());
    p.space->serialize(serialized.data(),copy.get());
    p.space->deserialize(s.get(),serialized.data());verify(s.get());
    // Coordinate-key equality must be bit-exact, including adjacent doubles.
    double z=0;p.objective->stateHeight(s.get(),z);
    const auto before=map.counters().height_queries;
    s->as<ob::RealVectorStateSpace::StateType>()->values[0]=std::nextafter(point(s.get()).x,10.);
    p.objective->stateHeight(s.get(),z);
    check(map.counters().height_queries==before+1,"nearby distinct coordinate was merged in cache");
    verify(s.get());
    auto native=p.space->allocDefaultStateSampler();
    for(int k=0;k<100;++k) {native->sampleUniform(s.get());verify(s.get());}
    // A state allocation reused for another query/map/policy must miss by token.
    setPoint(s.get(),{3,4});verify(s.get());
    TerrainObjective otherQuery(p.si,map,true,{2,1},{8,9},100,c,false,options);
    near(otherQuery.costToGo(s.get(),p.pdef->getGoal().get()).value(),planeDistance({3,4},{8,9}));
    verify(s.get());
    auto secondMap=plane(.4);
    TerrainObjective otherMap(p.si,secondMap,true,start,goal,100,c,false,options);
    near(otherMap.costToGo(s.get(),p.pdef->getGoal().get()).value(),planeDistance({3,4},goal,.4));
    verify(s.get());
    TerrainObjective xyPolicy(p.si,map,false,start,goal,100,c,false,options);
    near(xyPolicy.costToGo(s.get(),p.pdef->getGoal().get()).value(),std::hypot(6.,4.));
    verify(s.get());
    for(int k=0;k<100;++k) {
        auto *reused=p.space->allocState();setPoint(reused,{1.+k*.01,2});verify(reused);p.space->freeState(reused);
    }
    near(p.space->distance(a.get(),b.get()),10.);
    auto nd=plane(.2,true);PolicyCounters nc;auto np=makeProblem(nd,true,start,goal,nc,50,1.1,100,options);
    ob::ScopedState<> missing(np.space);setPoint(missing.get(),{5.2,5.2});
    check(!np.objective->stateHeight(missing.get(),z),"NoData height fabricated");
    const auto missingQueries=nd.counters().height_queries;
    check(!np.objective->stateHeight(missing.get(),z),"cached NoData fabricated");
    check(nd.counters().height_queries==missingQueries,"NoData result not cached");
    check(std::isinf(np.objective->costToGo(missing.get(),np.pdef->getGoal().get()).value()),"NoData heuristic finite");
    setPoint(missing.get(),{3,4});check(np.objective->stateHeight(missing.get(),z),"NoData poisoned mutated state");near(z,1.);
    setPoint(missing.get(),{INFINITY,4});check(!np.objective->stateHeight(missing.get(),z),"infinite coordinate height accepted");
    std::cout<<"PASS cache lifecycle, native writes, exact XY keys, map/query/policy identity and NoData\n";
}
void samplerAndSharedOptions() {
    auto map=plane();PolicyCounters c;PlannerOptions options;options.state_cache=true;options.scalar_cost=true;
    auto p=makePlannerProblem(map,"bitstar_a3",{1,2},{9,8},c,50,1.1,100,options);
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);ob::ScopedState<> sample(p.space);
    map.resetCounters();
    check(sampler->sampleUniform(sample.get(),ob::Cost(INFINITY)),"cached informed sampler failed");
    check(map.counters().height_queries==1,"sampler and objective hooks repeated height interpolation");
    check(c.motion_heuristic_calls>0&&c.cost_to_go_calls>0&&c.height_cache_hits>=2,"sampler bypassed objective hooks/cache");
    const Point q=point(sample.get());
    near(p.objective->costToGo(sample.get(),p.pdef->getGoal().get()).value(),planeDistance(q,{9,8}));
    for(const auto &id:{"abitstar","aitstar","lazyprmstar_hauser2015","rrtstar","informed_rrtstar","sem_rrtstar_common_v1","bitstar_a0"}) {
        PolicyCounters counters;auto other=makePlannerProblem(map,id,{1,2},{9,8},counters,50,1.1,100,options);
        check(other.objective->stateCacheEnabled()&&other.objective->scalarCostEnabled(),"shared optimization ignored");
        ob::ScopedState<> s(other.space),g(other.space);setPoint(s.get(),{1,2});setPoint(g.get(),{9,8});
        near(other.objective->motionCost(s.get(),g.get()).value(),planeDistance({1,2},{9,8}));
        if(std::string(id)=="abitstar"||std::string(id)=="aitstar") {
            auto native=other.objective->allocInformedStateSampler(other.pdef,100);
            check(dynamic_cast<ob::PathLengthDirectInfSampler*>(native.get()),"native comparator sampler replaced");
            check(native->sampleUniform(s.get(),ob::Cost(15)),"native sampler cannot write cache state");
            near(other.objective->motionCostHeuristic(s.get(),g.get()).value(),distanceXY(point(s.get()),point(g.get())));
        }
    }
    std::cout<<"PASS shared options, objective sampler cache reuse and native Euclidean sampler\n";
}
void batches() {
    auto map=hill(true);PolicyCounters c;PlannerOptions options;options.batch_after_first_solution=200;
    // A free direct start-goal edge can solve at batch zero, correctly switching
    // before any samples. Block that edge to observe BOTH actual batch sizes.
    check(!map.validMotion({1,5},{9,5}),"batch fixture must require sampled detour");
    auto p=makePlannerProblem(map,"bitstar_a3",{1,5},{9,5},c,50,1.1,100,options);
    std::uint64_t rejectedStates=0;
    p.si->setStateValidityChecker([&](const ob::State *state) {
        const bool valid=map.validState(point(state));
        if(!valid)++rejectedStates;
        return valid;
    });
    unsigned transitionBatch=0,callbacks=0;bool initialBatchSeen=false,nextBatchSeen=false;
    p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*>&,ob::Cost) {
        ++callbacks;
        if(p.onFirstSolution())transitionBatch=p.bit->numBatches();
        check(!p.onFirstSolution(),"batch transition is not one-shot");
    });
    std::uint64_t previousGenerated=0;unsigned calls=0;
    auto status=p.planner->solve(ob::PlannerTerminationCondition([&]() {
        const auto &properties=p.bit->getPlannerProgressProperties();
        // The pinned BITstar registers state collision checks, but comments out
        // its total-states-generated property. ImplicitGraph.cpp:965 increments
        // this public counter once for each returned sample before isValid;
        // subtract actual rejected states to observe accepted batch sizes.
        // Both fixed endpoints are valid, so neither contributes a rejection.
        const auto generated=std::stoull(properties.at("state collision checks INTEGER")())-rejectedStates;
        if(generated!=previousGenerated) {
            const auto delta=generated-previousGenerated;
            if(previousGenerated==0||p.batch_transition_count==0) {check(delta==50,"initial actual batch size mismatch");initialBatchSeen=true;}
            else if(p.bit->numBatches()>transitionBatch) {
                check(delta==200,"next actual OMPL batch did not consume updated setting");nextBatchSeen=true;
            }
            previousGenerated=generated;
        }
        return nextBatchSeen || ++calls>20000;
    }));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks>0,"adaptive BIT* did not reach exact goal");
    check(initialBatchSeen&&nextBatchSeen&&p.batch_transition_count==1,"adaptive next-batch effect unproven");
    for(const auto &id:{"bitstar_a0","abitstar","aitstar","rrtstar"}) {
        bool rejected=false;try {makePlannerProblem(map,id,{1,5},{9,5},c,50,1.1,100,options);}
        catch(const std::runtime_error&) {rejected=true;}check(rejected,"unsupported adaptive planner silently accepted");
    }
    bool rejectedNoop=false;
    try {makePlannerProblem(map,"bitstar_a3",{1,5},{9,5},c,200,1.1,100,options);}
    catch(const std::runtime_error&) {rejectedNoop=true;}
    check(rejectedNoop,"no-op adaptive batch silently accepted");
    std::cout<<"PASS actual 50-to-200 next batch, one-shot callback and unsupported-policy rejection\n";
}
void fixedWork(unsigned mode) {
    check(mode<=2,"invalid fixed-work mode");
    auto map=hill();PolicyCounters c;PlannerOptions options;options.state_cache=mode>0;options.scalar_cost=mode>1;
    auto p=makePlannerProblem(map,"bitstar_a3",{1,5},{9,5},c,50,1.1,100,options);
    std::ostringstream trace;trace<<std::hexfloat;unsigned improvements=0;
    p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &raw,ob::Cost cost) {
        ++improvements;trace<<cost.value()<<':';
        for(auto q:p.snapshot(raw))trace<<q.x<<','<<q.y<<';';
        trace<<'|';
    });
    unsigned calls=0;
    auto status=p.planner->solve(ob::PlannerTerminationCondition([&](){return ++calls>=5000;}));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&improvements>0,"fixed-work BIT* did not solve");
    trace<<" calls="<<calls<<" iterations="<<p.iterations()<<" batches="<<p.bit->numBatches()
         <<" hooks="<<c.motion_heuristic_calls<<','<<c.from_start_calls<<','<<c.to_goal_calls<<','<<c.cost_to_go_calls
         <<" samples="<<c.sample_proposals<<','<<c.samples_returned<<','<<c.rejected_bounds<<','<<c.rejected_nodata<<','<<c.rejected_heuristic
         <<" validity="<<map.counters().state_validity_calls<<','<<map.counters().motion_validity_calls;
    std::cout<<"TRACE "<<trace.str()<<'\n';
    std::cout<<"HEIGHT_QUERIES "<<map.counters().height_queries<<'\n';
}
}
int main(int argc,char **argv) {
    try {
        ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);ompl::RNG::setSeed(73129);
        if(argc==3&&std::string(argv[1])=="--fixed-work")fixedWork(static_cast<unsigned>(std::stoul(argv[2])));
        else {check(argc==1,"unknown test arguments");lifecycle();samplerAndSharedOptions();batches();}
        return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
