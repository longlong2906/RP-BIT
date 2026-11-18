#include "terrain25d/OmplIntegration.hpp"
#include "terrain25d/SemExposure.hpp"
#include <ompl/base/ScopedState.h>
#include <ompl/util/Console.h>
#include <iostream>
#include <limits>

using namespace terrain25d;
namespace {
void check(bool ok,const char *message) { if(!ok)throw std::runtime_error(message); }
void near(double a,double b) {
    check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-8+1e-8*std::max(std::abs(a),std::abs(b)),
          "SEM exposure analytic value differs");
}
Terrain terrain(double slope=0.,bool missing=false,std::vector<Rect> obstacles={}) {
    std::vector<double> heights;
    for(unsigned j=0;j<=10;++j)for(unsigned i=0;i<=10;++i)heights.push_back(slope*i);
    if(missing)heights[5*11+5]=std::numeric_limits<double>::quiet_NaN();
    return Terrain(11,11,0,0,1,1,heights,std::vector<int>(100),std::move(obstacles),.1,0,35);
}
std::shared_ptr<const SafetyField> field(std::vector<double> risks=std::vector<double>(100,20.),
                                         std::vector<int> support=std::vector<int>(100,1)) {
    return std::make_shared<const SafetyField>(10,10,0,0,1,10,10,std::move(support),std::move(risks));
}
PlannerOptions options(const Terrain &map,std::shared_ptr<const SafetyField> risks,double weight=1.) {
    PlannerOptions out;out.state_cache=true;out.scalar_cost=true;out.sampler_query_cache=true;
    out.safety=std::make_shared<const SafetyPolicy>(map,std::move(risks),40,35);
    out.safety_atlas_sampling=true;out.sem_exposure=SemExposureConfig{weight,40.};return out;
}
template<class Action> void rejects(Action action) {
    bool rejected=false;try {action();} catch(const std::exception &) {rejected=true;}
    check(rejected,"unsupported SEM exposure configuration accepted");
}
void analyticIntegralAndBounds() {
    auto map=terrain(.25);auto risks=field();auto opts=options(map,risks,2.);
    PolicyCounters counts;const Point start{1.5,1.5},goal{8.5,7.5},middle{5,4.5};
    auto p=makeProblem(map,true,start,goal,counts,100,1.1,100,opts);
    ob::ScopedState<> a(p.space),b(p.space),m(p.space);setPoint(a.get(),start);setPoint(b.get(),goal);setPoint(m.get(),middle);
    const double length=std::sqrt(49.+36.+1.75*1.75);
    near(*semExposureCanonical(map,*risks,start,goal),20.*length);
    near(p.objective->motionCost(a.get(),b.get()).value(),2.*length);
    near(p.objective->motionCost(b.get(),a.get()).value(),2.*length);
    near(p.objective->motionCost(a.get(),m.get()).value()+p.objective->motionCost(m.get(),b.get()).value(),2.*length);
    near(p.objective->motionCost(a.get(),a.get()).value(),0.);
    near(p.objective->motionCostHeuristic(a.get(),b.get()).value(),length);
    near(p.objective->costToGo(a.get(),p.pdef->getGoal().get()).value(),length);
    near(p.space->distance(a.get(),b.get()),std::sqrt(85.));
    check(p.space->getDimension()==2,"SEM exposure added a state dimension");
    check(counts.sem_exposure_motion_cost_calls==5&&counts.sem_exposure_integral_calls==5,"objective integral counter mismatch");
    opts.sem_exposure->lambda=0.;PolicyCounters zeroCounts;
    auto zero=makeProblem(map,true,start,goal,zeroCounts,100,1.1,100,opts);
    ob::ScopedState<> za(zero.space),zb(zero.space);setPoint(za.get(),start);setPoint(zb.get(),goal);
    opts.sem_exposure.reset();PolicyCounters legacyCounts;
    auto legacy=makeProblem(map,true,start,goal,legacyCounts,100,1.1,100,opts);
    ob::ScopedState<> la(legacy.space),lb(legacy.space);setPoint(la.get(),start);setPoint(lb.get(),goal);
    check(zero.objective->motionCost(za.get(),zb.get()).value()==legacy.objective->motionCost(la.get(),lb.get()).value(),
          "lambda0 changed the legacy cost bits");
    check(zeroCounts.sem_exposure_motion_cost_calls==1&&zeroCounts.sem_exposure_integral_calls==0,
          "lambda0 performed an unnecessary risk integral in solve");
}
void boundaryAndInvalidSemantics() {
    auto map=terrain();auto values=std::vector<double>(100,2.);
    for(int i=0;i<10;++i)values[5*10+i]=8.;
    auto risks=field(values);const Point a{1.5,5.},b{8.5,5.},m{4.25,5.};
    near(*semExposureCanonical(map,*risks,a,b),56.);
    near(*semExposureCanonical(map,*risks,b,a),56.);
    near(*semExposureCanonical(map,*risks,a,m)+*semExposureCanonical(map,*risks,m,b),56.);
    near(*semExposureCanonical(map,*risks,{5.,5.},{5.,5.}),0.);
    // An isolated high-risk corner contributes zero length but remains a hard safety contact.
    auto cornerValues=std::vector<double>(100,0.);cornerValues[4*10+5]=41.;
    auto corner=field(cornerValues);
    near(*semExposureCanonical(map,*corner,{4.5,4.5},{5.5,5.5}),0.);
    SafetyPolicy cornerPolicy(map,corner,40,35);
    check(!cornerPolicy.validMotion({4.5,4.5},{5.5,5.5}),"isolated risk41 contact bypassed hard safety");
    auto support=std::vector<int>(100,1);support[5*10+4]=0;values[5*10+4]=0.;auto unsupported=field(values,support);
    check(!semExposureCanonical(map,*unsupported,a,b),"missing SEM support was assigned zero risk");
    check(!semExposureCanonical(map,*unsupported,{4.5,5.5},{4.5,5.5}),"zero edge fabricated support");
    auto missingMap=terrain(0.,true);
    check(!semExposureCanonical(missingMap,*risks,{2.5,5.5},{7.5,5.5}),"NoData exposure fabricated");
    check(!semExposureCanonical(map,*risks,{-1.,5.},{8.,5.}),"outside exposure fabricated");
    PolicyCounters counters;auto opts=options(map,unsupported);
    auto p=makeProblem(map,true,{1.5,5.5},{8.5,5.5},counters,100,1.1,100,opts);
    ob::ScopedState<> s(p.space),g(p.space);setPoint(s.get(),{1.5,5.5});setPoint(g.get(),{8.5,5.5});
    check(std::isinf(p.objective->motionCost(s.get(),g.get()).value()),"unsupported edge objective was finite");
    auto obstacleMap=terrain(0.,false,{{4.99,0.,5.01,10.}});auto obstacleOpts=options(obstacleMap,field());
    PolicyCounters obstacleCounters;auto op=makeProblem(obstacleMap,true,{1.5,5.5},{8.5,5.5},obstacleCounters,100,1.1,100,obstacleOpts);
    ob::ScopedState<> os(op.space),og(op.space);setPoint(os.get(),{1.5,5.5});setPoint(og.get(),{8.5,5.5});
    check(std::isfinite(op.objective->motionCost(os.get(),og.get()).value()),"obstacle must not become an arbitrary risk penalty");
    check(!op.si->checkMotion(os.get(),og.get()),"finite soft cost bypassed thin obstacle");
}
void exactRasterBoundaryRegression() {
    auto map=terrain();auto values=std::vector<double>(100,2.);
    for(int i=0;i<10;++i)values[5*10+i]=8.;
    auto risks=field(values);const double ulp=std::nextafter(5.,INFINITY)-5.;
    for(const double rightX:{1.75,8.75}) {
        const Point a{1.25,5.-8.*ulp},b{rightX,5.-32.*ulp};
        const Point middle{(a.x+b.x)/2.,5.-20.*ulp};
        const auto whole=semExposureCanonical(map,*risks,a,b),reverse=semExposureCanonical(map,*risks,b,a);
        const auto first=semExposureCanonical(map,*risks,a,middle),second=semExposureCanonical(map,*risks,middle,b);
        check(whole&&reverse&&first&&second,"exact near-boundary integral unavailable");
        const double expected=2.*std::hypot(b.x-a.x,b.y-a.y);
        near(*whole,expected);near(*reverse,expected);near(*first+*second,expected);
    }
    const Point start{1.25,5.},end{1.75,std::nextafter(5.,-INFINITY)};
    check((start.y+end.y)/2.==5.,"one-ULP fixture does not round midpoint onto boundary");
    const auto sliver=semExposureCanonical(map,*risks,start,end),reverse=semExposureCanonical(map,*risks,end,start);
    check(sliver&&reverse,"one-ULP slanted edge unavailable");near(*sliver,1.);near(*reverse,1.);
    // Exact positive-length boundary contact still takes the incident maximum.
    near(*semExposureCanonical(map,*risks,{1.25,5.},{1.75,5.}),4.);
}
void roundedSurfaceEndpointRegression() {
    // Two edges from the failed Kaggle snapshots. The lifted polyline's
    // arithmetic endpoint is one ULP below or above the requested endpoint.
    // Use a complete analytic plane to isolate cut reconstruction from safety.
    std::vector<double> heights;
    for(int j=0;j<=40;++j)for(int i=0;i<=40;++i)heights.push_back(.25*i);
    auto map=Terrain(41,41,0,0,1,1,std::move(heights),std::vector<int>(1600),{},.1,0,35);
    SafetyField risks(40,40,0,0,1,40,40,std::vector<int>(1600,1),std::vector<double>(1600,20.));
    const std::pair<Point,Point> edges[]={
        {{4.413526356695283,22.16450639193167},{13.547450037716247,29.0779241759954}},
        {{4.0777092708496765,20.550916129387453},{15.092693098563915,14.069778126006993}},
    };
    for(const auto &[a,b]:edges) {
        const auto forward=semExposureCanonical(map,risks,a,b);
        const auto backward=semExposureCanonical(map,risks,b,a);
        check(forward&&backward,"rounded surface endpoint made supported SEM exposure undefined");
        const double expected=20.*std::hypot(b.x-a.x,b.y-a.y,.25*(b.x-a.x));
        near(*forward,expected);near(*backward,expected);
    }
}
void invalidConfiguration() {
    auto map=terrain();auto risks=field();auto good=options(map,risks);PolicyCounters c;
    const auto create=[&](const PlannerOptions &opts){makePlannerProblem(map,"bitstar_a3",{1.5,5.5},{8.5,5.5},c,100,1.1,100,opts);};
    for(const double weight:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        auto bad=good;bad.sem_exposure->lambda=weight;rejects([&]{create(bad);});
    }
    for(const double scale:{0.,1.,std::numeric_limits<double>::infinity()}) {auto bad=good;bad.sem_exposure->risk_scale=scale;rejects([&]{create(bad);});}
    auto bad=good;bad.safety.reset();rejects([&]{create(bad);});
    bad=good;bad.safety=std::make_shared<const SafetyPolicy>(map,risks,39.,35.);rejects([&]{create(bad);});
    bad=good;bad.safety=std::make_shared<const SafetyPolicy>(map,risks,40.,34.);rejects([&]{create(bad);});
    bad=good;bad.safety=std::make_shared<const SafetyPolicy>(map,risks,40.,35.,std::vector<int>(100,1));rejects([&]{create(bad);});
    for(const auto *id:{"bitstar_a0","sem_rrtstar_common_v1","rrtstar","abitstar"})
        rejects([&]{makePlannerProblem(map,id,{1.5,5.5},{8.5,5.5},c,100,1.1,100,good);});
    rejects([&]{makeProblem(map,false,{1.5,5.5},{8.5,5.5},c,100,1.1,100,good);});
}
void improvingPathAndRealHooks() {
    auto map=terrain();auto values=std::vector<double>(100,0.);
    for(int i=0;i<10;++i)values[5*10+i]=40.;
    auto risks=field(values);auto opts=options(map,risks);opts.batch_after_first_solution=200;
    PolicyCounters c;const Point start{1.5,5.5},goal{8.5,5.5};
    auto p=makePlannerProblem(map,"bitstar_a3",start,goal,c,100,1.1,100,opts);
    const auto nativeSetupSamplerAllocations=c.sampler_allocations;
    ob::ScopedState<> s(p.space),g(p.space),sample(p.space),detour(p.space);
    setPoint(s.get(),start);setPoint(g.get(),goal);setPoint(detour.get(),{5.,4.5});
    const double incumbentLength=7.,incumbentJ=14.;
    near(p.objective->motionCost(s.get(),g.get()).value(),incumbentJ);
    // The independently calculable 1+7+1 detour has E=40 and J=10, but L=9>7.
    const std::vector<Point> path{start,{1.5,4.5},{8.5,4.5},goal};
    double detourCost=0.;for(std::size_t i=1;i<path.size();++i) {
        setPoint(s.get(),path[i-1]);setPoint(g.get(),path[i]);detourCost+=p.objective->motionCost(s.get(),g.get()).value();
    }
    near(detourCost,10.);setPoint(s.get(),start);setPoint(g.get(),goal);
    const double f=p.objective->motionCostHeuristic(s.get(),detour.get()).value()+p.objective->costToGo(detour.get(),p.pdef->getGoal().get()).value();
    check(f>incumbentLength&&f<incumbentJ,"J-bound witness does not lie outside length bound");
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);
    bool outsideLength=false;
    for(unsigned i=0;i<300;++i) {
        check(sampler->sampleUniform(sample.get(),ob::Cost(incumbentLength),ob::Cost(incumbentJ)),"nonempty J interval exhausted");
        const double bound=map.lowerBound(start,point(sample.get()),true)+map.lowerBound(point(sample.get()),goal,true);
        check(bound>=incumbentLength-1e-10&&bound<incumbentJ+1e-10,"sampler violated J interval");
        outsideLength|=bound>incumbentLength+1e-6;
    }
    check(outsideLength,"sampler silently used L incumbent");
    // Manual objective/sampler exercises above must not satisfy the native hook assertions.
    c={};
    double last=INFINITY;unsigned callbacks=0,calls=0;
    p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*>&states,ob::Cost cost) {
        check(cost.value()<=last+1e-8,"J incumbent increased");last=cost.value();++callbacks;
        const auto xy=p.snapshot(states);check(xy.front().x==start.x&&xy.back().x==goal.x,"callback endpoint mismatch");
        p.onFirstSolution();
    });
    const auto status=p.planner->solve(ob::PlannerTerminationCondition([&]{return ++calls>4000;}));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks>0,"real BITstar did not solve exposure fixture");
    check(c.sem_exposure_motion_cost_calls>0&&c.sem_exposure_integral_calls>0&&c.motion_heuristic_calls>0&&
          c.from_start_calls>0&&c.to_goal_calls>0&&c.cost_to_go_calls>0&&
          (c.sampler_allocations>0||nativeSetupSamplerAllocations>0)&&
          c.sampler_finite_bound_calls>0,"required objective/heuristic/sampler hooks unused");
    check(p.batch_transition_count==1&&p.bit->getSamplesPerBatch()==200,"batch transition ineffective");
}
}
int main() {
    try {
        ompl::msg::setLogLevel(ompl::msg::LOG_NONE);ompl::RNG::setSeed(45231);
        analyticIntegralAndBounds();boundaryAndInvalidSemantics();exactRasterBoundaryRegression();
        roundedSurfaceEndpointRegression();
        invalidConfiguration();improvingPathAndRealHooks();
        std::cout<<"PASS SEM exposure integral, safety, lambda0, admissible J bounds and real BITstar hooks\n";return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
