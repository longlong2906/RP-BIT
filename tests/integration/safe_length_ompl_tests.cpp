#include "terrain25d/OmplIntegration.hpp"
#include "terrain25d/Safety.hpp"
#include <ompl/base/ScopedState.h>
#include <ompl/util/Console.h>
#include <array>
#include <iostream>
#include <limits>

using namespace terrain25d;
namespace {
void check(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
void near(double a,double b) {
    check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-8+1e-8*std::max(std::abs(a),std::abs(b)),
          "independent flat cost mismatch");
}
Terrain flat(int extent=20,double radius=.1) {
    return Terrain(extent+1,extent+1,0,0,1,1,std::vector<double>((extent+1)*(extent+1)),
                   std::vector<int>(extent*extent),{},radius,0,35);
}
std::shared_ptr<const SafetyField> stripField() {
    std::vector<int> support(400,1);std::vector<double> risk(400,1.);
    for(int j=7;j<14;++j)for(int i=9;i<12;++i)risk[j*20+i]=41.;
    return std::make_shared<SafetyField>(20,20,0,0,1,20,20,std::move(support),std::move(risk));
}
// Independent Liang--Barsky clipping against the explicitly known test hazard;
// no SafetyField traversal or planner validity routine is used as the oracle.
bool touchesStrip(Point a,Point b) {
    double low=0,high=1;
    for(int axis=0;axis<2;++axis) {
        const double start=axis?a.y:a.x,delta=axis?b.y-a.y:b.x-a.x;
        const double left=axis?7:9,right=axis?14:12;
        if(delta==0) {if(start<left||start>right)return false;continue;}
        double l=(left-start)/delta,h=(right-start)/delta;
        if(l>h)std::swap(l,h);
        low=std::max(low,l);high=std::min(high,h);
        if(low>high)return false;
    }
    return true;
}
void samplerIntervalsAndEmptyDomain() {
    auto map=flat();auto safety=std::make_shared<SafetyPolicy>(map,stripField(),40,35);
    PlannerOptions options;options.safety=safety;options.safety_atlas_sampling=true;
    options.state_cache=true;options.scalar_cost=true;options.sampler_query_cache=true;
    PolicyCounters counters;
    auto p=makePlannerProblem(map,"bitstar_a3",{2,10},{18,10},counters,100,1.1,100,options);
    ob::ScopedState<> state(p.space),a(p.space),b(p.space);
    setPoint(a.get(),{2,10});setPoint(b.get(),{18,10});
    near(p.space->distance(a.get(),b.get()),16);
    near(p.objective->motionCost(a.get(),b.get()).value(),16);
    near(p.objective->motionCostHeuristic(a.get(),b.get()).value(),16);
    check(!p.si->checkMotion(a.get(),b.get()),"unsafe cheap direct edge reached tree");
    ob::ScopedState<> lastState(p.space);
    std::pair<ob::State*,double> last{lastState.get(),.123};
    check(!p.si->getMotionValidator()->checkMotion(a.get(),b.get(),last),
          "last-valid overload accepted the hazard-crossing edge");
    check(last.second>.43&&last.second<.4375,"last-valid prefix did not stop at first risk cell");
    check(p.si->isValid(last.first)&&p.si->checkMotion(a.get(),last.first),
          "last-valid overload returned an unsafe prefix");
    setPoint(b.get(),{2,2});setPoint(lastState.get(),{.7,.8});last.second=.123;
    check(p.si->getMotionValidator()->checkMotion(a.get(),b.get(),last),"safe edge rejected");
    check(last.second==.123&&point(lastState.get()).x==.7&&point(lastState.get()).y==.8,
          "successful last-valid overload changed caller outputs");
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);
    check(!sampler->hasInformedMeasure(),"atlas advertised as exact informed measure");
    near(sampler->getInformedMeasure(ob::Cost(17),ob::Cost(25)),400);
    unsigned accepted=0,validStates=0;
    for(unsigned k=0;k<500;++k)if(sampler->sampleUniform(state.get(),ob::Cost(17),ob::Cost(25))) {
        const Point q=point(state.get());
        const double bound=distanceXY({2,10},q)+distanceXY(q,{18,10});
        check(bound>=17&&bound<25,"informed interval violated");
        // The sampler returns heuristic-valid proposals; OMPL applies hard
        // state validity once. An ellipse proposal may still hit a hazard.
        if(p.si->isValid(state.get())) {
            check(!touchesStrip(q,q),"state validator accepted a hazard proposal");
            check(q.x>.1&&q.y>.1&&q.x<19.9&&q.y<19.9,"state validator accepted invalid footprint bounds");
            ++validStates;
        }
        ++accepted;
    }
    check(accepted>400,"valid interval starved unexpectedly");
    check(validStates>300,"safe sample acceptance unexpectedly poor");
    check(!sampler->sampleUniform(state.get(),ob::Cost(18),ob::Cost(18)),"empty interval accepted");
    check(!sampler->sampleUniform(state.get(),ob::Cost(16)),"lower bound accepted");
    check(!sampler->sampleUniform(state.get(),ob::Cost(NAN)),"NaN bound accepted");
    check(counters.motion_heuristic_calls&&counters.cost_to_go_calls&&counters.sampler_interval_calls,
          "real informed sampler hooks bypassed");

    auto empty=std::make_shared<SafetyField>(20,20,0,0,1,20,20,std::vector<int>(400),std::vector<double>(400));
    options.safety=std::make_shared<SafetyPolicy>(map,empty,40,35);
    PolicyCounters emptyCounters;
    auto ep=makePlannerProblem(map,"bitstar_a3",{2,10},{18,10},emptyCounters,100,1.1,7,options);
    auto es=ep.objective->allocInformedStateSampler(ep.pdef,7);
    ob::ScopedState<> sample(ep.space);
    check(!es->sampleUniform(sample.get(),ob::Cost(INFINITY)),"empty atlas fabricated a sample");
    std::cout<<"PASS objective, kNN, interval, unknown measure, empty atlas and effective sampler hooks\n";
}
void atlasAreaAndPartialCells() {
    auto field=std::make_shared<SafetyField>(3,2,0,0,1,2.5,1.5,
                                            std::vector<int>(6,1),std::vector<double>(6));
    // Build an exactly matching map because the safety policy binds field bounds.
    Terrain clipped(6,4,0,0,.5,.5,std::vector<double>(24),std::vector<int>(15),{},.01,0,35);
    SafetyPolicy policy(clipped,field,40,35);
    near(policy.atlasArea(),3.75);
    std::array<unsigned,6> counts{};
    for(unsigned k=0;k<15000;++k) {
        const Point q=policy.sampleAtlas((k+.5)/15000.,.3,.7);
        const int i=static_cast<int>(std::floor(q.x)),j=static_cast<int>(std::floor(q.y));
        check(i>=0&&i<3&&j>=0&&j<2&&q.x<2.5&&q.y<1.5,"clipped atlas sample outside cell");
        ++counts[j*3+i];
    }
    const std::array<unsigned,6> expected{{4000,4000,2000,2000,2000,1000}};
    check(counts==expected,"atlas ignored clipped-cell area weights");

    Terrain obstacle(5,5,0,0,1,1,std::vector<double>(25),std::vector<int>(16),{{1,1,1.4,1.4}},.05,0,35);
    auto whole=std::make_shared<SafetyField>(4,4,0,0,1,4,4,std::vector<int>(16,1),std::vector<double>(16));
    auto partial=std::make_shared<SafetyPolicy>(obstacle,whole,40,35);
    near(partial->atlasArea(),16.); // Proposal must retain partially valid cells.
    PlannerOptions options;options.safety=partial;options.safety_atlas_sampling=true;
    PolicyCounters c;auto p=makePlannerProblem(obstacle,"bitstar_a3",{.5,.5},{3.5,3.5},c,100,1.1,100,options);
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);ob::ScopedState<> sample(p.space);
    unsigned inRetainedPart=0;
    for(unsigned k=0;k<2500;++k)if(sampler->sampleUniform(sample.get(),ob::Cost(INFINITY))) {
        const auto q=point(sample.get());
        if(q.x>1.6&&q.x<2&&q.y>1.6&&q.y<2)++inRetainedPart;
    }
    check(inRetainedPart>5,"atlas erased the valid part of a partially blocked cell");
    std::cout<<"PASS exact clipped-cell area weights and partially valid atlas cells\n";
}
void solveSafe(bool atlas,bool adaptive) {
    auto map=flat();auto safety=std::make_shared<SafetyPolicy>(map,stripField(),40,35);
    PlannerOptions options;options.safety=safety;options.safety_atlas_sampling=atlas;
    options.state_cache=true;options.scalar_cost=true;options.sampler_query_cache=true;
    if(adaptive)options.batch_after_first_solution=200;
    PolicyCounters counters;
    auto p=makePlannerProblem(map,"bitstar_a3",{2,10},{18,10},counters,adaptive?100:200,1.1,100,options);
    unsigned callbacks=0;double best=INFINITY;bool beforeSeen=false,afterSeen=false;
    unsigned transitionBatch=0;
    std::uint64_t rejectedStates=0,previousGenerated=0;
    p.si->setStateValidityChecker([&](const ob::State *state) {
        const bool valid=safety->validState(point(state));if(!valid)++rejectedStates;return valid;
    });
    p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &raw,ob::Cost cost) {
        const auto path=p.snapshot(raw);
        check(path.size()>=3,"unsafe direct edge became an incumbent");
        check(path.front().x==2&&path.front().y==10&&path.back().x==18&&path.back().y==10,
              "callback lost exact endpoints");
        double actual=0;
        for(std::size_t k=1;k<path.size();++k) {
            check(!touchesStrip(path[k-1],path[k]),"unsafe shorter improvement replaced incumbent");
            actual+=distanceXY(path[k-1],path[k]);
        }
        near(actual,cost.value());check(cost.value()<=best+1e-8,"safe incumbent length increased");
        best=cost.value();++callbacks;
        if(adaptive&&p.onFirstSolution())transitionBatch=p.bit->numBatches();
    });
    unsigned calls=0;
    const auto status=p.planner->solve(ob::PlannerTerminationCondition([&]() {
        const auto &properties=p.bit->getPlannerProgressProperties();
        const auto generated=std::stoull(properties.at("state collision checks INTEGER")())-rejectedStates;
        if(generated!=previousGenerated) {
            const auto delta=generated-previousGenerated;
            if(adaptive) {
                if(previousGenerated==0||p.batch_transition_count==0) {
                    check(delta==100,"actual initial safe batch is not 100");beforeSeen=true;
                } else if(p.bit->numBatches()>transitionBatch) {
                    check(delta==200,"actual next safe batch is not 200");afterSeen=true;
                }
            }
            previousGenerated=generated;
        }
        return ++calls>=4000;
    }));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks>0,"safe BITstar fixture failed");
    check(best>16,"hazard-crossing direct path survived safety gating");
    check(p.bit->getUseKNearest()&&!p.bit->getJustInTimeSampling()&&p.bit->getPruning(),"BITstar policy drift");
    if(adaptive)check(beforeSeen&&afterSeen&&p.batch_transition_count==1,"100-to-200 safe batch effect unproven");
    std::cout<<"PASS all safe incumbents, native BITstar atlas="<<atlas<<" adaptive="<<adaptive<<'\n';
}
void forbiddenObjectiveMixing() {
    auto map=flat();map.enableTraversability(TraversabilityConfig{});
    PlannerOptions options;options.safety=std::make_shared<SafetyPolicy>(map,stripField(),40,35);
    PolicyCounters counters;bool rejected=false;
    try {makePlannerProblem(map,"bitstar_a3",{2,10},{18,10},counters,200,1.1,100,options);}
    catch(const std::runtime_error &) {rejected=true;}
    check(rejected,"safe-length accidentally retained weighted risk objective");
    std::cout<<"PASS weighted-objective mixing rejected\n";
}
}
int main() {
    try {
        ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);ompl::RNG::setSeed(73129);
        samplerIntervalsAndEmptyDomain();atlasAreaAndPartialCells();
        solveSafe(false,false);solveSafe(true,false);solveSafe(true,true);forbiddenObjectiveMixing();
        return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
