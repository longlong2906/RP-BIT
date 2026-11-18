#include "terrain25d/RpBit.hpp"
#include "terrain25d/SemExposure.hpp"
#include <ompl/base/ScopedState.h>
#include <ompl/util/Console.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>

using namespace terrain25d;
namespace {
const double INF=std::numeric_limits<double>::infinity();
void check(bool ok,const char *message) { if(!ok)throw std::runtime_error(message); }
void near(double a,double b,double relative=1e-9) {
    check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=relative+relative*std::max(std::abs(a),std::abs(b)),
          "RP-BIT* analytic value differs");
}
// 10 m x 10 m, 1 m terrain cells; plane with the given x-slope.
Terrain coarseTerrain(double slope=0.) {
    std::vector<double> heights;
    for(unsigned j=0;j<=10;++j)for(unsigned i=0;i<=10;++i)heights.push_back(slope*i);
    return Terrain(11,11,0,0,1,1,heights,std::vector<int>(100),{},.1,0,35);
}
std::shared_ptr<const SafetyField> coarseField(std::vector<double> risks,std::vector<int> support=std::vector<int>(100,1)) {
    return std::make_shared<const SafetyField>(10,10,0,0,1,10,10,std::move(support),std::move(risks));
}
std::shared_ptr<const SafetyField> bandField() {
    auto values=std::vector<double>(100,0.);
    for(int i=0;i<10;++i)values[5*10+i]=40.;
    return coarseField(values);
}
// 8 m x 8 m undulating terrain (0.5 m cells) with a 0.1 m risk raster of random blocks.
Terrain fineTerrain() {
    std::vector<double> heights;
    for(unsigned j=0;j<=16;++j)for(unsigned i=0;i<=16;++i)heights.push_back(.3*std::sin(.25*i)*std::cos(.5*j/3.));
    return Terrain(17,17,0,0,.5,.5,heights,std::vector<int>(256),{},.1,0,35);
}
std::shared_ptr<const SafetyField> fineField(unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(0.,1.);
    std::vector<double> risks(6400);std::vector<int> support(6400,1);
    // 8x8-cell blocks: weight ratios up to about 1:30, plus a few blocked blocks.
    for(int bj=0;bj<10;++bj)for(int bi=0;bi<10;++bi) {
        const double roll=unit(rng);
        const double risk=roll<.12?41.:roll<.5?.2*unit(rng):39.*unit(rng);
        const bool missing=unit(rng)<.04&&!(bi<2&&bj<2)&&!(bi>7&&bj>7);
        for(int j=8*bj;j<8*bj+8;++j)for(int i=8*bi;i<8*bi+8;++i) {
            const bool corner=(bi<2&&bj<2)||(bi>7&&bj>7);
            risks[j*80+i]=missing?0.:corner?1.:risk;support[j*80+i]=missing?0:1;
        }
    }
    return std::make_shared<const SafetyField>(80,80,0,0,.1,8,8,std::move(support),std::move(risks));
}

void integratorMatchesCanonical() {
    auto flat=coarseTerrain();auto values=std::vector<double>(100,2.);
    for(int i=0;i<10;++i)values[5*10+i]=8.;
    auto risks=coarseField(values);RpEdgeIntegrator fast(flat,*risks);
    const Point a{1.5,5.},b{8.5,5.},m{4.25,5.};
    // A positive-length run on a raster line takes the incident maximum, in both directions.
    near(fast.evaluate(a,b).exposure,56.);near(fast.evaluate(b,a).exposure,56.);
    near(fast.evaluate(a,m).exposure+fast.evaluate(m,b).exposure,56.);
    near(fast.evaluate({1.25,5.},{1.75,5.}).exposure,4.);
    check(fast.evaluate({5.,5.},{5.,5.}).defined&&fast.evaluate({5.,5.},{5.,5.}).exposure==0.,"zero edge has nonzero exposure");
    const double ulp=std::nextafter(5.,INF)-5.;
    for(const double rightX:{1.75,8.75}) {
        const Point p{1.25,5.-8.*ulp},q{rightX,5.-32.*ulp};
        near(fast.evaluate(p,q).exposure,2.*std::hypot(q.x-p.x,q.y-p.y));
        near(fast.evaluate(q,p).exposure,2.*std::hypot(q.x-p.x,q.y-p.y));
    }
    near(fast.evaluate({1.25,5.},{1.75,std::nextafter(5.,-INF)}).exposure,1.);
    // An isolated corner contact has zero measure.
    auto cornerValues=std::vector<double>(100,0.);cornerValues[4*10+5]=41.;
    auto corner=coarseField(cornerValues);
    near(RpEdgeIntegrator(flat,*corner).evaluate({4.5,4.5},{5.5,5.5}).exposure,0.);
    auto support=std::vector<int>(100,1);support[5*10+4]=0;values[5*10+4]=0.;
    auto unsupported=coarseField(values,support);RpEdgeIntegrator gap(flat,*unsupported);
    check(gap.evaluate(a,b).defined&&!gap.evaluate(a,b).supported,"missing SEM support not reported");
    check(!gap.evaluate({4.5,5.5},{4.5,5.5}).supported,"zero edge fabricated support");
    check(!fast.evaluate({-1.,5.},{8.,5.}).defined,"outside exposure fabricated");
    std::vector<double> holed;
    for(unsigned j=0;j<=10;++j)for(unsigned i=0;i<=10;++i)holed.push_back(0.);
    holed[5*11+5]=std::numeric_limits<double>::quiet_NaN();
    Terrain missing(11,11,0,0,1,1,holed,std::vector<int>(100),{},.1,0,35);
    check(!RpEdgeIntegrator(missing,*risks).evaluate({2.5,5.5},{7.5,5.5}).defined,"NoData exposure fabricated");

    // Random edges on undulating terrain against the rational reference.
    auto map=fineTerrain();auto field=fineField(11);RpEdgeIntegrator integrator(map,*field);
    std::mt19937_64 rng(5);std::uniform_real_distribution<double> xy(.05,7.95),unit(0.,1.);
    unsigned compared=0;double worst=0.;
    for(unsigned k=0;k<4000;++k) {
        Point p{xy(rng),xy(rng)},q{xy(rng),xy(rng)};
        if(k%7==0){p.x=std::round(p.x*10.)/10.;q.x=p.x;}       // on a vertical raster line
        if(k%11==0){p.y=std::round(p.y*10.)/10.;q.y=p.y;}      // on a horizontal raster line
        if(k%13==0){q.x=p.x+(q.y-p.y);}                        // through raster corners when snapped
        if(k%5==0){p={std::round(p.x*10.)/10.,std::round(p.y*10.)/10.};}
        if(q.x<0.||q.x>8.||q.y<0.||q.y>8.)continue;
        const auto reference=semExposureCanonical(map,*field,p,q);
        const auto value=integrator.evaluate(p,q);
        if(!reference)continue;
        check(value.defined&&value.supported,"fast integral undefined where the canonical one exists");
        check(value.length==map.surfaceCost(p,q),"fast integral changed the surface length");
        const double error=std::abs(value.exposure-*reference)/std::max(1.,std::abs(*reference));
        worst=std::max(worst,error);++compared;
        const double t=.2+.6*unit(rng);const Point mid{p.x+t*(q.x-p.x),p.y+t*(q.y-p.y)};
        const auto first=integrator.evaluate(p,mid),second=integrator.evaluate(mid,q),reverse=integrator.evaluate(q,p);
        check(first.supported&&second.supported&&reverse.supported,"split or reversed edge lost support");
        near(first.exposure+second.exposure,value.exposure,1e-8);near(reverse.exposure,value.exposure,1e-9);
    }
    check(compared>2000&&worst<=1e-9,"fast integral deviates from the canonical exposure");
    // Speed: the reason this integrator exists.
    std::vector<std::pair<Point,Point>> edges;
    for(unsigned k=0;k<400;++k)edges.push_back({{xy(rng),xy(rng)},{xy(rng),xy(rng)}});
    const auto began=std::chrono::steady_clock::now();double sink=0.;
    for(const auto &edge:edges)sink+=integrator.evaluate(edge.first,edge.second).exposure;
    const auto middle=std::chrono::steady_clock::now();
    for(const auto &edge:edges){const auto v=semExposureCanonical(map,*field,edge.first,edge.second);if(v)sink+=*v;}
    const auto finished=std::chrono::steady_clock::now();
    const double quick=std::chrono::duration<double>(middle-began).count()/edges.size();
    const double exact=std::chrono::duration<double>(finished-middle).count()/edges.size();
    std::cout<<"fast integral "<<quick*1e6<<" us/edge, canonical "<<exact*1e6<<" us/edge, ratio "<<exact/quick
             <<", worst relative error "<<worst<<" (checksum "<<sink<<")\n";
    check(exact>20.*quick,"fast integral is not substantially faster than the canonical one");
}

void potentialProperties() {
    auto field=fineField(23);const Point source{.75,.65};const double lambda=30.;
    RiskPotential potential(*field,40.,lambda,source);
    check(potential.at(source)==0.,"potential at its source must be exactly zero");
    std::mt19937_64 rng(17);std::uniform_real_distribution<double> unit(0.,1.);
    unsigned open=0,blocked=0;
    for(unsigned k=0;k<60000;++k) {
        const int i=static_cast<int>(unit(rng)*80),j=static_cast<int>(unit(rng)*80);
        const bool usable=field->supported(i,j)&&field->risk(i,j)<=40.;
        const double w=1.+lambda/40.*field->risk(i,j);
        const Point p{(i+unit(rng))*.1,(j+unit(rng))*.1},q{(i+unit(rng))*.1,(j+unit(rng))*.1};
        const double a=potential.at(p),b=potential.at(q);
        check(!std::isnan(a)&&!std::isnan(b)&&a>=0.&&b>=0.,"potential is NaN or negative");
        if(!usable){check(std::isinf(a)&&std::isinf(b),"blocked cell has a finite potential");++blocked;continue;}
        if(std::isinf(a)){check(std::isinf(b)&&std::isinf(potential.cellLowerBound(i,j)),"open cell is half reachable");continue;}
        ++open;
        // Lipschitz with the cell's own weight, and never below the cell bound.
        check(std::abs(a-b)<=w*std::hypot(p.x-q.x,p.y-q.y)*(1.+1e-9)+3e-9,"potential is not Lipschitz inside a cell");
        check(potential.cellLowerBound(i,j)<=std::min(a,b)+1e-12,"cell lower bound exceeds the potential");
        // Continuity across the right and top sides into an open neighbour.
        for(const bool vertical:{true,false}) {
            const int ni=vertical?i+1:i,nj=vertical?j:j+1;
            if(ni>=80||nj>=80||!field->supported(ni,nj)||field->risk(ni,nj)>40.)continue;
            const double t=unit(rng)*.1,eps=1e-9;
            const Point inside=vertical?Point{(i+1)*.1-eps,j*.1+t}:Point{i*.1+t,(j+1)*.1-eps};
            const Point outside=vertical?Point{(i+1)*.1+eps,j*.1+t}:Point{i*.1+t,(j+1)*.1+eps};
            const double jump=std::abs(potential.at(inside)-potential.at(outside));
            check(jump<=2.*eps*(w+1.+lambda/40.*field->risk(ni,nj))+5e-9,"potential jumps across a cell side");
        }
    }
    check(open>20000&&blocked>2000,"potential fixture does not exercise open and blocked cells");
    // A full wall of blocked cells: the relaxed lattice is disconnected.
    auto walled=std::vector<double>(100,1.);for(int j=0;j<10;++j)walled[j*10+5]=41.;
    RiskPotential split(*coarseField(walled),40.,lambda,{1.5,5.5});
    check(std::isinf(split.at({8.5,5.5}))&&std::isfinite(split.at({3.5,2.5})),"blocked wall did not disconnect the lattice");
    bool rejected=false;
    try {RiskPotential(SafetyField(10,10,0,0,1,9.5,10,std::vector<int>(100,1),std::vector<double>(100,1.)),40.,lambda,{1.5,5.5});}
    catch(const std::exception &) {rejected=true;}
    check(rejected,"clipped raster accepted by the risk potential");
    // Uniform field: the bound is within cos(pi/8) of the exact straight-line cost.
    RiskPotential uniform(*coarseField(std::vector<double>(100,4.)),40.,lambda,{1.5,1.5});
    for(const Point q:{Point{8.5,1.5},Point{8.5,8.5},Point{5.2,8.1},Point{2.5,7.5}}) {
        const double exact=(1.+lambda/40.*4.)*std::hypot(q.x-1.5,q.y-1.5),bound=uniform.at(q);
        check(bound<=exact+1e-9&&bound>=.85*exact,"uniform-field potential is not a tight lower bound");
    }
    // Box minimum never exceeds any open cell an edge can touch.
    RiskBoxMin box(*field,40.);
    for(unsigned k=0;k<20000;++k) {
        const Point p{unit(rng)*8.,unit(rng)*8.},q{std::clamp(p.x+(unit(rng)-.5)*6.,0.,8.),std::clamp(p.y+(unit(rng)-.5)*6.,0.,8.)};
        const double least=box.minRisk(p,q);
        for(const auto cell:field->touchedCells(p,q)) {
            const int i=static_cast<int>(cell%80),j=static_cast<int>(cell/80);
            if(field->supported(i,j)&&field->risk(i,j)<=40.)check(least<=field->risk(i,j),"box minimum exceeds a touched open cell");
        }
    }
}

// Real BIT*: at every incumbent each bound must be below the real cost it estimates.
double bitstarRespectsBounds(const Terrain &map,std::shared_ptr<const SafetyField> risks,Point start,Point goal,
                             bool potential,unsigned seed,unsigned limit) {
    auto safety=std::make_shared<const SafetyPolicy>(map,risks,40.,35.);
    PolicyCounters policy;RpCounters counters;RpBitOptions options;options.lambda=30.;options.potential=potential;
    ompl::RNG::setSeed(seed);
    auto p=makeRpProblem(map,safety,start,goal,options,policy,counters);
    ob::ScopedState<> s(p.space),g(p.space);setPoint(s.get(),start);setPoint(g.get(),goal);
    check(p.objective->motionCostHeuristic(g.get(),g.get()).value()==0.,"h(goal) is not exactly zero");
    check(p.objective->motionCostHeuristic(s.get(),s.get()).value()==0.,"g(start) is not exactly zero");
    check(p.objective->costToGo(g.get(),p.pdef->getGoal().get()).value()==0.,"costToGo(goal) is not exactly zero");
    check(p.space->getDimension()==2,"RP-BIT* added a state dimension");
    const double lower=p.objective->lowerBoundStartGoal();
    unsigned callbacks=0,calls=0;double last=INF;
    p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &states,ob::Cost cost) {
        ++callbacks;check(cost.value()<=last+1e-9,"J incumbent increased");last=cost.value();
        const std::vector<const ob::State*> path(states.rbegin(),states.rend());
        std::vector<double> prefix(path.size(),0.);
        for(std::size_t i=1;i<path.size();++i)prefix[i]=prefix[i-1]+p.objective->motionCost(path[i-1],path[i]).value();
        const double total=prefix.back();
        check(std::abs(total-cost.value())<=1e-9*std::max(1.,total),"callback cost differs from the edge sum");
        check(lower<=total+1e-9,"start-goal lower bound exceeds a real path");
        for(std::size_t i=0;i<path.size();++i) {
            check(p.si->isValid(path[i]),"incumbent contains an invalid state");
            const double come=p.objective->motionCostHeuristic(s.get(),path[i]).value();
            const double go=p.objective->motionCostHeuristic(path[i],g.get()).value();
            check(come<=prefix[i]+1e-9&&go<=total-prefix[i]+1e-9,"path bound exceeds a real path cost");
            if(i)check(p.objective->motionCostHeuristic(path[i-1],path[i]).value()<=prefix[i]-prefix[i-1]+1e-9,
                       "edge bound exceeds the true edge cost");
        }
    });
    const auto status=p.planner->solve(ob::PlannerTerminationCondition([&]{return ++calls>limit;}));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks>0,"real BIT* did not solve the RP fixture");
    check(policy.motion_cost_calls>0&&policy.motion_heuristic_calls>0&&policy.from_start_calls>0&&policy.to_goal_calls>0&&
          policy.sampler_allocations>0&&policy.safety_motion_checks>0&&counters.fast_integral_calls>0,"objective hooks unused");
    // Each switch must have an observable effect.
    check(potential?(counters.potential_evaluations>0&&counters.box_bound_calls>0&&counters.sampler_cell_draws>0):
                    (counters.potential_evaluations==0&&counters.box_bound_calls==0&&counters.sampler_cell_draws==0&&
                     policy.atlas_sample_proposals+policy.ellipse_sample_proposals>0),"potential switch is not effective");
    return last;
}

void edgeBoundBelowTrueCost() {
    auto map=fineTerrain();auto field=fineField(31);
    auto safety=std::make_shared<const SafetyPolicy>(map,field,40.,35.);
    PolicyCounters policy;RpCounters counters;RpBitOptions options;options.lambda=30.;
    const Point start{.75,.65},goal{7.25,7.35};
    auto p=makeRpProblem(map,safety,start,goal,options,policy,counters);
    ob::ScopedState<> a(p.space),b(p.space),s(p.space),g(p.space);setPoint(s.get(),start);setPoint(g.get(),goal);
    std::mt19937_64 rng(3);std::uniform_real_distribution<double> xy(.15,7.85),unit(0.,1.);
    unsigned valid=0;
    for(unsigned k=0;k<60000;++k) {
        const Point u{xy(rng),xy(rng)};const double angle=6.283185307179586*unit(rng),reach=.03+2.5*unit(rng)*unit(rng);
        Point v{u.x+reach*std::cos(angle),u.y+reach*std::sin(angle)};
        if(k%9==0)v={std::round(v.x*10.)/10.,std::round(v.y*10.)/10.};
        if(v.x<.15||v.x>7.85||v.y<.15||v.y>7.85)continue;
        setPoint(a.get(),u);setPoint(b.get(),v);
        const double bound=p.objective->motionCostHeuristic(a.get(),b.get()).value();
        check(!std::isnan(bound)&&bound>=0.,"edge bound is NaN or negative");
        if(!safety->validMotion(u,v))continue;
        ++valid;
        const double cost=p.objective->motionCost(a.get(),b.get()).value();
        check(std::isfinite(cost)&&bound<=cost+1e-9,"edge bound exceeds the true cost of a valid edge");
        // Triangle-type consistency of the two path bounds along a real edge.
        const double hu=p.objective->motionCostHeuristic(a.get(),g.get()).value(),hv=p.objective->motionCostHeuristic(b.get(),g.get()).value();
        const double gu=p.objective->motionCostHeuristic(s.get(),a.get()).value(),gv=p.objective->motionCostHeuristic(s.get(),b.get()).value();
        check(hu<=cost+hv+1e-8&&gv<=gu+cost+1e-8,"path bound is inconsistent along a valid edge");
    }
    check(valid>8000,"edge-bound fixture has too few valid edges");
}

void samplerContract() {
    auto map=fineTerrain();auto field=fineField(31);
    auto safety=std::make_shared<const SafetyPolicy>(map,field,40.,35.);
    PolicyCounters policy;RpCounters counters;RpBitOptions options;options.lambda=30.;
    const Point start{.75,.65},goal{7.25,7.35};
    auto p=makeRpProblem(map,safety,start,goal,options,policy,counters);
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);
    check(!sampler->hasInformedMeasure(),"sampler claims an exact informed measure");
    ob::ScopedState<> state(p.space);
    const double lower=p.objective->lowerBoundStartGoal();
    check(std::isfinite(lower)&&lower>0.,"fixture has no finite lower bound");
    const auto before=counters.sampler_empty_prefix_calls;
    check(!sampler->sampleUniform(state.get(),ob::Cost(0.),ob::Cost(.5*lower)),"sampled below the start-goal lower bound");
    check(counters.sampler_empty_prefix_calls==before+1,"empty informed set was not detected immediately");
    check(!sampler->sampleUniform(state.get(),ob::Cost(2.),ob::Cost(1.)),"invalid interval accepted");
    std::vector<unsigned> columns(8,0);
    for(const double factor:{4.,2.,1.3,1.1}) {
        const double hi=factor*lower,lo=factor>1.2?0.:1.02*lower;
        for(unsigned k=0;k<400;++k) {
            if(!sampler->sampleUniform(state.get(),ob::Cost(lo),ob::Cost(hi)))continue;
            const auto value=p.objective->bounds(point(state.get()));
            check(value.g+value.h>=lo&&value.g+value.h<hi,"sampler violated the J interval");
            if(factor==4.)++columns[std::min(7,static_cast<int>(point(state.get()).x))];
        }
    }
    check(std::count_if(columns.begin(),columns.end(),[](unsigned n){return n>0;})>=6,"wide informed set is not covered");
    // An infinite bound (before the first solution) samples the reachable open cells.
    for(unsigned k=0;k<200;++k) {
        check(sampler->sampleUniform(state.get(),ob::Cost(0.),ob::Cost(INF)),"pre-solution sampling failed");
        const auto q=point(state.get());
        check(std::isfinite(p.objective->bounds(q).h),"pre-solution sample lies in a blocked or unreachable cell");
    }
}

void refinementContract() {
    auto map=coarseTerrain();auto values=std::vector<double>(100,1.);
    for(int j=3;j<7;++j)for(int i=3;i<7;++i)values[j*10+i]=41.;   // blocked square forces a corner
    auto field=coarseField(values);SafetyPolicy policy(map,field,40.,35.);RpEdgeIntegrator integrator(map,*field);
    const auto sound=[&](const RpRefineResult &result,const std::vector<Point> &input,double floor) {
        check(result.cost.valid&&result.path.front().x==input.front().x&&result.path.front().y==input.front().y&&
              result.path.back().x==input.back().x&&result.path.back().y==input.back().y,"refinement moved an endpoint");
        for(std::size_t i=1;i<result.path.size();++i) {
            check(policy.validMotion(result.path[i-1],result.path[i]),"refinement produced an unsafe edge");
            check(distanceXY(result.path[i-1],result.path[i])>=floor,"rounding hid a corner behind a micro-segment");
        }
    };
    // Rounding alone on a corner that hugs the blocked square: the arc must pass outside, at a small capped cost.
    RpRefineOptions rounding;rounding.shortcut=rounding.relocate=false;rounding.round_cost_ratio=.01;
    const std::vector<Point> hugging{{1.5,1.5},{7.02,2.98},{8.5,8.5}};
    const auto tight=rpPathCost(integrator,30.,hugging);
    check(tight.valid&&tight.sharp_turns==1,"hugging fixture must have exactly one sharp turn");
    const auto arc=rpRefine(policy,integrator,30.,hugging,rounding);
    sound(arc,hugging,.99*rounding.min_chord_m);
    check(arc.cost.sharp_turns==0&&arc.kept_sharp==0&&arc.rounded==1&&arc.path.size()>=4,"obstacle-hugging corner was not rounded");
    check(arc.cost.objective>tight.objective&&arc.cost.objective<=tight.objective*(1.+rounding.round_cost_ratio),
          "outward arc must cost a little and stay within the rounding cap");
    auto frozen=rounding;frozen.round_cost_ratio=0.;
    const auto kept=rpRefine(policy,integrator,30.,hugging,frozen);
    check(kept.rounded==0&&kept.kept_sharp==1&&kept.path.size()==3&&kept.cost.objective==tight.objective,
          "a zero rounding cap must leave the corner alone");
    // A corner in free space is cut on the inside, which shortens the path.
    const std::vector<Point> open{{1.5,1.5},{8.5,1.5},{8.5,8.5}};
    const auto wide=rpPathCost(integrator,30.,open);
    const auto inside=rpRefine(policy,integrator,30.,open,frozen);
    sound(inside,open,.99*rounding.min_chord_m);
    check(wide.sharp_turns==1&&inside.cost.sharp_turns==0&&inside.rounded>=1&&inside.cost.objective<wide.objective,
          "free corner was not rounded on the inside");
    // Full refinement of a clumsy path: safe, never worse than the cap allows, no more sharp turns than before.
    const std::vector<Point> corner{{1.5,1.5},{2.,1.5},{4.,1.6},{8.2,1.5},{8.5,2.},{8.4,5.},{8.5,8.5}};
    const auto before=rpPathCost(integrator,30.,corner);
    check(before.valid&&before.sharp_turns==2,"refinement fixture must have two sharp turns");
    RpRefineOptions options;
    const auto refined=rpRefine(policy,integrator,30.,corner,options);
    sound(refined,corner,0.);
    check(refined.cost.objective<.95*before.objective,"refinement did not tighten a clumsy path");
    check(refined.cost.sharp_turns<=before.sharp_turns&&refined.cost.sharp_turns==refined.kept_sharp,"sharp-turn accounting is inconsistent");
    std::cout<<"refinement: J "<<before.objective<<" -> "<<refined.cost.objective<<", sharp turns "<<before.sharp_turns<<" -> "
             <<refined.cost.sharp_turns<<", vertices "<<refined.path.size()<<", edge evaluations "<<refined.edge_evaluations<<"\n";
    const auto again=rpRefine(policy,integrator,30.,corner,options);
    check(again.path.size()==refined.path.size(),"refinement is not deterministic");
    for(std::size_t i=0;i<again.path.size();++i)
        check(again.path[i].x==refined.path[i].x&&again.path[i].y==refined.path[i].y,"refinement is not deterministic");
    RpRefineOptions plain;plain.round_turns=false;
    const auto unrounded=rpRefine(policy,integrator,30.,corner,plain);
    check(unrounded.cost.valid&&unrounded.cost.objective<=before.objective+1e-9&&unrounded.rounded==0,
          "shortcut/relocation increased J or rounded while disabled");
    RpRefineOptions off;off.shortcut=off.relocate=off.round_turns=false;
    const auto untouched=rpRefine(policy,integrator,30.,corner,off);
    check(untouched.path.size()==corner.size()&&untouched.validity_checks==0,"disabled refinement changed the path");
}

void plannerRuns() {
    auto map=coarseTerrain();auto risks=bandField();const Point start{1.5,5.5},goal{8.5,5.5};
    // Leaving the risk band costs 31/m; the infimum of J is 7 + sqrt(960) (analytic, lambda=30).
    const double optimum=7.+std::sqrt(960.);
    RpRunOptions options;options.planner.lambda=30.;options.budget_s=60.;options.stop_batches=0;
    options.test_max_iterations=12000;options.planner_seed=77;
    const auto first=runRpBit(map,risks,start,goal,options),second=runRpBit(map,risks,start,goal,options);
    check(first.status=="EXACT_SOLUTION"&&first.termination=="WORK_BUDGET"&&!first.publications.empty(),"fixed-work run failed");
    check(first.publications.size()==second.publications.size()&&first.iterations==second.iterations,"fixed-work run is not reproducible");
    SafetyPolicy policy(map,risks,40.,35.);
    double last=INF;
    for(std::size_t k=0;k<first.publications.size();++k) {
        const auto &a=first.publications[k],&b=second.publications[k];
        check(a.origin=="rp_refinement"&&a.cost.valid&&a.cost.objective<last,"publication is not a strictly improving refined path");
        last=a.cost.objective;
        check(a.path.size()==b.path.size(),"fixed-work publications differ");
        for(std::size_t i=0;i<a.path.size();++i)check(a.path[i].x==b.path[i].x&&a.path[i].y==b.path[i].y,"fixed-work publications differ");
        for(std::size_t i=1;i<a.path.size();++i)check(policy.validMotion(a.path[i-1],a.path[i]),"published an unsafe edge");
        check(a.path.front().x==start.x&&a.path.back().x==goal.x&&
              a.cost.objective<=a.bitstar_cost*(1.+options.refinement.round_cost_ratio)+1e-9,"publication contract violated");
    }
    check(first.lower_bound<=last+1e-9&&first.lower_bound>.8*optimum,"lower bound is invalid or uninformative");
    check(last<=optimum*1.01&&last>=optimum-1e-6,"refined RP-BIT* did not reach the analytic optimum");
    check(first.batch_transitions==1&&first.rp.heuristic_violations==0,"batch transition or bound audit failed");
    // Refinement off publishes raw BIT* incumbents only.
    auto raw=options;raw.refine=false;
    const auto unrefined=runRpBit(map,risks,start,goal,raw);
    check(!unrefined.publications.empty()&&unrefined.rp.refine_edge_evaluations==0,"refine switch is not effective");
    for(const auto &item:unrefined.publications)check(item.origin=="ompl_intermediate_solution_callback","unexpected refined publication");
    check(unrefined.publications.back().cost.objective>=optimum-1e-6,"unrefined run beat the analytic optimum");
    // The convergence stop ends the run long before the budget.
    auto stopping=options;stopping.test_max_iterations=0;stopping.stop_batches=3;stopping.budget_s=30.;
    const auto stopped=runRpBit(map,risks,start,goal,stopping);
    check(stopped.termination=="CONVERGED_NO_IMPROVEMENT"&&stopped.actual_wall_s<10.,"convergence stop did not trigger");
    check(stopped.publications.back().cost.objective<=optimum*1.05,"converged run is far from the optimum");
    std::cout<<"band fixture: optimum "<<optimum<<", lower bound "<<first.lower_bound<<", fixed work "<<last<<" (raw "
             <<unrefined.publications.back().cost.objective<<"), converged "<<stopped.publications.back().cost.objective
             <<" in "<<stopped.actual_wall_s<<" s after "<<stopped.batches<<" batches\n";
    // Invalid query and identity query.
    check(runRpBit(map,risks,{-1.,5.5},goal,options).status=="INVALID_START","invalid start accepted");
    const auto identity=runRpBit(map,risks,start,start,options);
    check(identity.status=="EXACT_SOLUTION"&&identity.publications.size()==1&&identity.publications[0].cost.objective==0.,"identity query failed");
}
}
int main() {
    try {
        ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
        integratorMatchesCanonical();potentialProperties();edgeBoundBelowTrueCost();samplerContract();
        const double with=bitstarRespectsBounds(coarseTerrain(),bandField(),{1.5,5.5},{8.5,5.5},true,45231,8000);
        const double without=bitstarRespectsBounds(coarseTerrain(),bandField(),{1.5,5.5},{8.5,5.5},false,45231,8000);
        const double optimum=7.+std::sqrt(960.);
        std::cout<<"raw BIT* on the band fixture: potential "<<with<<", d_phi control "<<without<<", optimum "<<optimum<<"\n";
        check(with<=1.08*optimum&&without<=1.08*optimum&&with>=optimum-1e-6&&without>=optimum-1e-6,
              "BIT* is far from (or below) the band-fixture optimum");
        bitstarRespectsBounds(fineTerrain(),fineField(31),{.75,.65},{7.25,7.35},true,9,6000);
        bitstarRespectsBounds(coarseTerrain(.25),bandField(),{1.5,5.5},{8.5,5.5},true,3,4000);
        refinementContract();plannerRuns();
        std::cout<<"PASS RP-BIT* fast integral, risk potential, bounds under real BITstar, sampler, refinement and runs\n";return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
