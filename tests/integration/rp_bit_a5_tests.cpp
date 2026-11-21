#include "terrain25d/RpBitA5.hpp"
#include <ompl/util/Console.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

using namespace terrain25d;
namespace {
const double INF=std::numeric_limits<double>::infinity();
void check(bool ok,const char *message) { if(!ok)throw std::runtime_error(message); }
bool samePath(const std::vector<Point> &a,const std::vector<Point> &b) {
    if(a.size()!=b.size())return false;
    for(std::size_t i=0;i<a.size();++i)if(a[i].x!=b[i].x||a[i].y!=b[i].y)return false;
    return true;
}
bool samePublications(const std::vector<RpPublication> &a,const std::vector<RpPublication> &b) {
    if(a.size()!=b.size())return false;
    for(std::size_t k=0;k<a.size();++k)
        if(a[k].origin!=b[k].origin||a[k].batch!=b[k].batch||a[k].bitstar_cost!=b[k].bitstar_cost||
           a[k].cost.objective!=b[k].cost.objective||!samePath(a[k].path,b[k].path))return false;
    return true;
}
// Fixtures of the A4 tests (tests/integration/rp_bit_tests.cpp), repeated because that file is locked.
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
// A5 fixture: 12 m x 12 m, 0.2 m cells, footprint radius 0.4 m and a 60-degree model limit, so slopes
// between the 35-degree policy cap and 60 degrees are driveable geometry that only the swept-slope test rejects.
// Gentle waves everywhere, a ramp of about 40 degrees for 5 <= x <= 7, a NoData hole and an obstacle.
Terrain steepTerrain() {
    const int n=61;
    std::vector<double> heights;
    for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        const double x=.2*i,y=.2*j;
        double z=.15*std::sin(.8*x)*std::cos(.6*y)+.839*std::clamp(x-5.,0.,2.);
        if(i>=44&&i<=46&&j>=12&&j<=14)z=std::numeric_limits<double>::quiet_NaN();
        heights.push_back(z);
    }
    return Terrain(n,n,0,0,.2,.2,heights,std::vector<int>((n-1)*(n-1)),{Rect{2.,8.,3.,9.}},.3,.1,60);
}
std::shared_ptr<const SafetyField> steepField() {
    std::vector<double> risks(14400);std::vector<int> support(14400,1);
    for(int j=0;j<120;++j)for(int i=0;i<120;++i) {
        risks[j*120+i]=1.+4.*((i/15+j/15)%3);
        if(i>=30&&i<36&&j>=40&&j<46){support[j*120+i]=0;risks[j*120+i]=0.;}
        if(i>=80&&i<84&&j>=90&&j<96)risks[j*120+i]=41.;
    }
    return std::make_shared<const SafetyField>(120,120,0,0,.1,12,12,std::move(support),std::move(risks));
}
void publicationsAreSafe(const RpRunResult &run,const SafetyPolicy &policy,Point start,Point goal) {
    double last=INF;
    for(const auto &item:run.publications) {
        check(item.cost.valid&&item.cost.objective<last,"publication is not strictly improving");
        last=item.cost.objective;
        check(item.path.front().x==start.x&&item.path.front().y==start.y&&item.path.back().x==goal.x&&item.path.back().y==goal.y,
              "publication does not join start and goal");
        for(std::size_t i=1;i<item.path.size();++i)check(policy.validMotion(item.path[i-1],item.path[i]),"published an unsafe edge");
    }
    check(run.rp.heuristic_violations==0,"a bound exceeded the true cost on a real incumbent");
}

// C1: the schedule function.
void scheduleFunction() {
    RpA5Options options;options.batch_growth=true;
    const auto sizes=[&]() {
        // The batch opened at the transition has batch_min samples; later ones follow the formula.
        std::vector<unsigned> out;std::uint64_t stale=0;unsigned drawn=options.batch_min;
        for(;;) {
            out.push_back(drawn);
            const unsigned next=rpA5NextBatch(stale,drawn,options);
            stale+=drawn;
            if(stale>=options.window_samples)return out;
            drawn=next;
        }
    };
    options.batch_min=500;options.window_samples=4000;
    check(sizes()==std::vector<unsigned>({500,500,1000,2000}),"4000-sample window must use batches 500, 500, 1000, 2000");
    options.window_samples=1000;
    check(sizes()==std::vector<unsigned>({500,500}),"1000-sample window must use batches 500, 500");
    for(const unsigned window:{600u,1000u,1700u,4000u,10000u})for(const unsigned least:{100u,200u,500u}) {
        options.window_samples=window;options.batch_min=least;
        std::uint64_t total=0;
        for(const unsigned size:sizes()){check(size>0,"empty verification batch");total+=size;}
        check(total==std::max(window,least),"growing batches overshoot or undershoot the window");
    }
    options.batch_growth=false;options.batch_min=200;options.window_samples=4000;
    check(sizes()==std::vector<unsigned>(20,200u),"without growth the window is 20 batches of 200, as in A4");
    check(rpA5NextBatch(0,200,options)==200&&rpA5NextBatch(123456,200,options)==200,"growth switch is not effective");
}

// With every A5 switch off the run is the A4 run, bit for bit.
void a4Equivalence() {
    struct Case { Terrain map;std::shared_ptr<const SafetyField> field;Point start,goal;unsigned work; };
    const Case cases[]={{coarseTerrain(),bandField(),{1.5,5.5},{8.5,5.5},6000},{fineTerrain(),fineField(31),{.75,.65},{7.25,7.35},5000}};
    for(const auto &item:cases) {
        RpRunOptions a4;a4.planner.lambda=30.;a4.budget_s=60.;a4.stop_batches=0;a4.test_max_iterations=item.work;a4.planner_seed=77;
        RpA5Options a5;a5.planner=a4.planner;a5.budget_s=60.;a5.window_samples=0;a5.batch_min=200;
        a5.test_max_iterations=item.work;a5.planner_seed=77;
        const auto compare=[&](const char *message) {
            const auto reference=runRpBit(item.map,item.field,item.start,item.goal,a4);
            const auto mine=runRpBitA5(item.map,item.field,item.start,item.goal,a5);
            check(!reference.publications.empty()&&samePublications(reference.publications,mine.run.publications)&&
                  reference.iterations==mine.run.iterations&&reference.batches==mine.run.batches&&
                  reference.termination==mine.run.termination&&reference.raw_incumbents==mine.run.raw_incumbents&&
                  reference.refinements==mine.run.refinements&&reference.batch_transitions==mine.run.batch_transitions,message);
            check(mine.a5.bounded_samples==0&&mine.a5.refine_deferred==0&&mine.a5.slope_certified==0&&mine.seed_status=="disabled",
                  "a disabled A5 mechanism left a trace");
            return reference;
        };
        check(compare("A5 with every switch off differs from A4 at fixed work").termination=="WORK_BUDGET","fixed-work label");
        a4.refine=a5.refine=false;
        compare("A5 with every switch off differs from A4 with refinement off");
        a4.refine=a5.refine=true;
        // Three stale batches of 200 samples are a 600-sample window.
        a4.test_max_iterations=a5.test_max_iterations=0;a4.budget_s=a5.budget_s=40.;a4.stop_batches=3;a5.window_samples=600;
        check(compare("A5's sample window differs from A4's batch window").termination=="CONVERGED_NO_IMPROVEMENT",
              "convergence stop did not trigger");
    }
}

// C1: growing batches follow the formula, stop on the window, and need far fewer batches.
void batchGrowth() {
    const auto map=fineTerrain();const auto field=fineField(31);const Point start{.75,.65},goal{7.25,7.35};
    RpA5Options options;options.planner.lambda=30.;options.budget_s=60.;options.planner_seed=5;
    options.batch_min=200;options.window_samples=4000;
    const auto flat=runRpBitA5(map,field,start,goal,options);
    options.batch_growth=true;options.batch_min=500;
    const auto grown=runRpBitA5(map,field,start,goal,options);
    SafetyPolicy policy(map,field,40.,35.);
    for(const auto *run:{&flat,&grown}) {
        check(run->run.termination=="CONVERGED_NO_IMPROVEMENT","window stop did not trigger");
        publicationsAreSafe(run->run,policy,start,goal);
        const auto &log=run->batches;
        check(log.size()>=3&&log.back().stale_samples>=options.window_samples&&
              log[log.size()-2].stale_samples<options.window_samples,"run did not stop exactly when the window filled");
    }
    for(std::size_t k=0;k+1<grown.batches.size();++k)
        if(std::isfinite(grown.batches[k].published))
            check(grown.batches[k+1].random_samples==rpA5NextBatch(grown.batches[k].stale_samples,grown.batches[k].random_samples,options),
                  "a batch size does not follow the registered schedule");
    for(std::size_t k=0;k+1<flat.batches.size();++k)
        if(std::isfinite(flat.batches[k].published))check(flat.batches[k+1].random_samples==200,"flat schedule changed a batch size");
    check(grown.run.batches+8<flat.run.batches,"growing batches did not reduce the number of batches");
    std::cout<<"window 4000: flat "<<flat.run.batches<<" batches, "<<flat.run.iterations<<" iterations, J "
             <<flat.run.publications.back().cost.objective<<"; growing "<<grown.run.batches<<" batches, "<<grown.run.iterations
             <<" iterations, J "<<grown.run.publications.back().cost.objective<<"\n";
}

// C2: the slope shortcut never changes a verdict or its reason.
void slopeShortcut() {
    const auto map=steepTerrain();const auto field=steepField();
    SafetyPolicy policy(map,field,40.,35.);RpA5SlopeEnvelope envelope(map);RpA5Counters counters;
    check(envelope.reach()==4,"0.4 m footprint on 0.2 m cells needs a 4-cell envelope reach");
    std::mt19937_64 rng(2026);std::uniform_real_distribution<double> xy(-.1,12.1),unit(0.,1.);
    std::uint64_t compared=0,valid=0,steep=0,geometry=0,sem=0;
    for(unsigned k=0;k<80000;++k) {
        Point a{xy(rng),xy(rng)};
        const double angle=6.283185307179586*unit(rng),length=k%4==0?0.:2.5*unit(rng)*unit(rng);
        Point b{a.x+length*std::cos(angle),a.y+length*std::sin(angle)};
        if(k%7==0){a.x=std::round(a.x*10.)/10.;b.x=k%14==0?a.x:b.x;}       // on SEM and terrain lines
        if(k%11==0){a.y=std::round(a.y*5.)/5.;b.y=k%22==0?a.y:b.y;}
        if(k%13==0){b={std::round(b.x*5.)/5.,std::round(b.y*5.)/5.};}
        const auto exact=policy.rejectionReason(a,b),fast=rpA5RejectionReason(policy,map,envelope,a,b,counters);
        check(exact==fast,"slope shortcut changed a validity verdict or its reason");
        ++compared;
        if(exact.empty())++valid;else if(exact=="swept_slope_above_cap")++steep;
        else if(exact=="invalid_geometry")++geometry;else ++sem;
    }
    check(valid>10000&&steep>500&&geometry>1000&&sem>100,"validity fixture does not exercise every verdict");
    check(counters.slope_certified>5000&&counters.slope_exact>2000,"validity fixture does not exercise both slope paths");
    std::cout<<"slope shortcut: "<<compared<<" checks identical; valid "<<valid<<", steep "<<steep<<", geometry "<<geometry
             <<", SEM "<<sem<<"; certified "<<counters.slope_certified<<", exact "<<counters.slope_exact<<"\n";
    // A whole run is unchanged by the shortcut, on gentle and on steep terrain.
    struct Case { const Terrain &map;std::shared_ptr<const SafetyField> field;Point start,goal;bool mixed; };
    const auto gentle=fineTerrain();
    const Case cases[]={{gentle,fineField(31),{.75,.65},{7.25,7.35},false},{map,field,{1.,1.},{4.2,11.},true}};
    for(const auto &item:cases) {
        RpA5Options options;options.planner.lambda=30.;options.budget_s=60.;options.window_samples=0;
        options.test_max_iterations=5000;options.planner_seed=19;
        const auto exact=runRpBitA5(item.map,item.field,item.start,item.goal,options);
        options.fast_validity=true;
        const auto fast=runRpBitA5(item.map,item.field,item.start,item.goal,options);
        check(!exact.run.publications.empty()&&samePublications(exact.run.publications,fast.run.publications)&&
              exact.run.iterations==fast.run.iterations&&exact.run.batches==fast.run.batches&&
              exact.run.counters.safety_motion_checks==fast.run.counters.safety_motion_checks&&
              exact.run.counters.safety_state_checks==fast.run.counters.safety_state_checks&&
              exact.run.counters.safety_geometry_rejections==fast.run.counters.safety_geometry_rejections&&
              exact.run.counters.safety_slope_rejections==fast.run.counters.safety_slope_rejections,
              "the slope shortcut changed a run");
        check(exact.a5.slope_certified==0&&fast.a5.slope_certified>1000&&(!item.mixed||fast.a5.slope_exact>0),
              "the slope shortcut switch is not effective");
        std::cout<<"slope shortcut run: "<<fast.a5.slope_certified<<" certified, "<<fast.a5.slope_exact<<" exact of "
                 <<fast.run.counters.safety_motion_checks+fast.run.counters.safety_state_checks<<" BIT* checks\n";
    }
}

// C3: an incumbent waits for refinement until its raw cost has dropped by eps or the window is about to close.
void lazyRefine() {
    struct Case { Terrain map;std::shared_ptr<const SafetyField> field;Point start,goal; };
    const Case cases[]={{coarseTerrain(),bandField(),{1.5,5.5},{8.5,5.5}},{fineTerrain(),fineField(31),{.75,.65},{7.25,7.35}}};
    for(const auto &item:cases) {
        SafetyPolicy policy(item.map,item.field,40.,35.);
        RpA5Options options;options.planner.lambda=30.;options.budget_s=60.;options.planner_seed=23;
        options.batch_growth=true;options.batch_min=500;options.window_samples=4000;
        const auto eager=runRpBitA5(item.map,item.field,item.start,item.goal,options);
        options.lazy_refine=true;
        const auto lazy=runRpBitA5(item.map,item.field,item.start,item.goal,options);
        const auto again=runRpBitA5(item.map,item.field,item.start,item.goal,options);
        check(samePublications(lazy.run.publications,again.run.publications)&&lazy.run.iterations==again.run.iterations,
              "a lazy-refinement run is not reproducible");
        check(lazy.run.termination=="CONVERGED_NO_IMPROVEMENT"&&eager.run.termination=="CONVERGED_NO_IMPROVEMENT","window stop did not trigger");
        publicationsAreSafe(lazy.run,policy,item.start,item.goal);
        const double best=lazy.run.publications.back().cost.objective,reference=eager.run.publications.back().cost.objective;
        std::cout<<"lazy refinement: refinements "<<eager.run.refinements<<" -> "<<lazy.run.refinements<<" (deferred at "
                 <<lazy.a5.refine_deferred<<" boundaries), batches "<<eager.run.batches<<" -> "<<lazy.run.batches<<", J "<<reference
                 <<" -> "<<best<<"\n";
        // On the band fixture BIT*'s raw cost creeps down; on the block fixture it keeps dropping by more than eps.
        const bool creeping=&item==&cases[0];
        check(eager.a5.refine_deferred==0&&lazy.run.refinements<=eager.run.refinements&&
              (!creeping||(lazy.a5.refine_deferred>0&&lazy.run.refinements<eager.run.refinements)),
              "lazy refinement switch is not effective");
        check(best<=reference*1.01,"lazy refinement lost more than 1% of J on a fixture");
        // The run never stops on an incumbent it has not refined: BIT*'s cost at the stop is the one last refined.
        const auto &log=lazy.batches;
        std::size_t refined=log.size();
        for(std::size_t k=0;k<log.size();++k)if(log[k].refine_s>0.)refined=k;
        check(refined<log.size()&&log.back().bitstar_cost==log[refined].bitstar_cost,"the run stopped on an unrefined incumbent");
    }
    // With eps = 0 every new incumbent has dropped "by eps": lazy refinement is then eager refinement, bit for bit.
    const auto &item=cases[1];
    RpA5Options options;options.planner.lambda=30.;options.budget_s=60.;options.planner_seed=23;options.stop_relative_eps=0.;
    options.batch_growth=true;options.batch_min=500;options.window_samples=0;options.test_max_iterations=6000;
    const auto eager=runRpBitA5(item.map,item.field,item.start,item.goal,options);
    options.lazy_refine=true;
    const auto lazy=runRpBitA5(item.map,item.field,item.start,item.goal,options);
    check(samePublications(eager.run.publications,lazy.run.publications)&&eager.run.refinements==lazy.run.refinements&&
          lazy.a5.refine_deferred==0,"lazy refinement with eps = 0 differs from eager refinement");
}

// C4: the lattice seed is published before BIT* runs; an unusable seed falls back to the A4 start.
void seed() {
    const auto map=fineTerrain();const auto field=fineField(31);const Point start{.75,.65},goal{7.25,7.35};
    SafetyPolicy policy(map,field,40.,35.);
    RpA5Options options;options.planner.lambda=30.;options.budget_s=60.;options.planner_seed=41;
    options.batch_growth=true;options.batch_min=500;options.window_samples=4000;options.lazy_refine=true;
    const auto unseeded=runRpBitA5(map,field,start,goal,options);
    options.seed=true;
    const auto seeded=runRpBitA5(map,field,start,goal,options),again=runRpBitA5(map,field,start,goal,options);
    check(samePublications(seeded.run.publications,again.run.publications)&&seeded.run.iterations==again.run.iterations,
          "a seeded run is not reproducible");
    check(seeded.seed_status=="published"&&seeded.a5.seed_expansions>0&&seeded.a5.bounded_samples>0&&
          unseeded.seed_status=="disabled"&&unseeded.a5.seed_expansions==0&&unseeded.a5.bounded_samples==0,"seed switch is not effective");
    publicationsAreSafe(seeded.run,policy,start,goal);
    const auto &first=seeded.run.publications.front();
    check(first.origin=="a5_lattice_seed_refined"&&first.batch==0&&!std::isfinite(first.bitstar_cost),
          "the seed must be the first publication, before any BIT* batch");
    for(std::size_t k=1;k<seeded.run.publications.size();++k)
        check(seeded.run.publications[k].origin!="a5_lattice_seed_refined","a second seed publication");
    for(const auto &item:unseeded.run.publications)check(item.origin!="a5_lattice_seed_refined","seed publication with the seed off");
    check(seeded.run.batch_transitions==1&&seeded.run.termination=="CONVERGED_NO_IMPROVEMENT","seeded run did not converge");
    std::cout<<"seed: first J "<<first.cost.objective<<" (unseeded "<<unseeded.run.publications.front().cost.objective<<"), final J "
             <<seeded.run.publications.back().cost.objective<<" (unseeded "<<unseeded.run.publications.back().cost.objective
             <<"), expansions "<<seeded.a5.seed_expansions<<", refinements "<<seeded.run.refinements<<" vs "<<unseeded.run.refinements
             <<", batches "<<seeded.run.batches<<" vs "<<unseeded.run.batches<<"\n";
    check(first.cost.objective<=unseeded.run.publications.front().cost.objective*1.5,"the seed is far worse than BIT*'s first path");
    check(seeded.run.publications.back().cost.objective<=unseeded.run.publications.back().cost.objective*1.01,
          "the seeded run ends more than 1% above the unseeded run");
    // The lattice path itself: start to goal through the centres of open cells.
    PolicyCounters counters;RpCounters rp;RpBitOptions planner;planner.lambda=30.;
    auto safety=std::make_shared<const SafetyPolicy>(map,field,40.,35.);
    auto problem=makeRpProblem(map,safety,start,goal,planner,counters,rp);
    std::uint64_t expansions=0;
    const auto lattice=rpA5SeedPath(*safety,*problem.objective,map,start,goal,RP_A5_SEED_EXPANSIONS,expansions);
    check(lattice.size()>20&&lattice.front().x==start.x&&lattice.back().y==goal.y&&expansions>0,"no lattice path on the seed fixture");
    for(const auto q:lattice) {
        const int i=std::min(79,static_cast<int>(q.x/.1)),j=std::min(79,static_cast<int>(q.y/.1));
        check(field->supported(i,j)&&field->risk(i,j)<=40.,"lattice path leaves the open cells");
    }
    check(rpA5SeedPath(*safety,*problem.objective,map,start,goal,3,expansions).empty(),"expansion limit is not enforced");
    // An obstacle the SEM field does not know: the lattice path crosses it, so there is no seed.
    Terrain walled(41,41,0,0,.2,.2,std::vector<double>(1681,0.),std::vector<int>(1600),{Rect{3.6,1.,4.4,7.}},.3,.1,35);
    auto open=std::make_shared<const SafetyField>(80,80,0,0,.1,8,8,std::vector<int>(6400,1),std::vector<double>(6400,1.));
    const Point west{1.5,4.},east{6.5,4.};
    const auto blocked=runRpBitA5(walled,open,west,east,options);
    check(blocked.seed_status=="seed_unsafe"&&blocked.run.status=="EXACT_SOLUTION"&&blocked.run.batch_transitions==1,
          "an unsafe seed did not fall back to the A4 start");
    check(blocked.run.publications.front().origin=="rp_refinement","fallback must publish a refined BIT* incumbent first");
    SafetyPolicy walledPolicy(walled,open,40.,35.);
    publicationsAreSafe(blocked.run,walledPolicy,west,east);
    // No lattice path at all: an unsupported wall across the field.
    auto support=std::vector<int>(6400,1);auto risks=std::vector<double>(6400,1.);
    for(int j=0;j<80;++j)for(int i=38;i<42;++i){support[j*80+i]=0;risks[j*80+i]=0.;}
    auto split=std::make_shared<const SafetyField>(80,80,0,0,.1,8,8,support,risks);
    Terrain flat(41,41,0,0,.2,.2,std::vector<double>(1681,0.),std::vector<int>(1600),{},.3,.1,35);
    auto quick=options;quick.budget_s=1.;
    const auto none=runRpBitA5(flat,split,west,east,quick);
    check(none.seed_status=="no_lattice_path"&&none.run.status=="NO_SOLUTION","a disconnected field must give neither seed nor solution");
}

// The binary field container holds the cells of the text exchange bit for bit; malformed files are rejected.
void fieldExchange() {
    const auto reference=fineField(31);
    const int nx=reference->nx(),ny=reference->ny();
    std::ostringstream text;
    text<<std::setprecision(17)<<"SAFE_SEM_FIELD_V1\n"<<nx<<' '<<ny<<" 0 0 0.10000000000000001 8 8\n";
    std::string binary("SEMFLDB1");
    const std::int32_t size[2]={nx,ny};
    const double grid[5]={0.,0.,.1,8.,8.};
    binary.append(reinterpret_cast<const char*>(size),sizeof size).append(reinterpret_cast<const char*>(grid),sizeof grid);
    std::string risks;
    for(int j=0;j<ny;++j)for(int i=0;i<nx;++i) {
        const double risk=reference->risk(i,j);
        text<<(reference->supported(i,j)?1:0)<<' '<<risk<<'\n';
        binary.push_back(reference->supported(i,j)?1:0);
        risks.append(reinterpret_cast<const char*>(&risk),sizeof risk);
    }
    binary+=risks;
    const auto folder=std::filesystem::temp_directory_path()/("rp_bit_a5_field_"+std::to_string(std::random_device{}()));
    std::filesystem::create_directories(folder);
    const auto save=[&](const char *name,const std::string &bytes) {
        const auto path=(folder/name).string();
        std::ofstream(path,std::ios::binary)<<bytes;
        return path;
    };
    const auto rejects=[&](const std::string &bytes) {
        try{rpA5LoadField(save("bad.bin",bytes));}catch(const std::runtime_error&){return true;}
        return false;
    };
    try {
        const auto fromText=rpA5LoadField(save("field.txt",text.str())),fromBinary=rpA5LoadField(save("field.bin",binary));
        for(const auto *field:{&fromText,&fromBinary}) {
            check(field->nx()==nx&&field->ny()==ny&&field->resolution()==reference->resolution()&&
                  field->bounds().xmax==reference->bounds().xmax&&field->bounds().ymax==reference->bounds().ymax,
                  "field exchange changed the grid");
            for(int j=0;j<ny;++j)for(int i=0;i<nx;++i) {
                const double a=field->risk(i,j),b=reference->risk(i,j);
                check(field->supported(i,j)==reference->supported(i,j)&&std::memcmp(&a,&b,sizeof a)==0,
                      "field exchange changed a cell");
            }
        }
        const std::size_t cells=static_cast<std::size_t>(nx)*ny,header=8+sizeof size+sizeof grid;
        auto bad=binary;bad.pop_back();check(rejects(bad),"truncated binary field accepted");
        bad=binary;bad.push_back(0);check(rejects(bad),"binary field with trailing data accepted");
        bad=binary;bad[header]=2;check(rejects(bad),"binary field with a support value other than 0 or 1 accepted");
        bad=binary;bad[8]=0;bad[9]=0;check(rejects(bad),"binary field with an empty grid accepted");
        // An unsupported cell must carry the placeholder risk 0: give one a risk instead.
        std::size_t hole=0;
        while(hole<cells&&binary[header+hole])++hole;
        check(hole<cells,"field fixture has no unsupported cell");
        const double one=1.;
        bad=binary;bad.replace(header+cells+hole*sizeof one,sizeof one,reinterpret_cast<const char*>(&one),sizeof one);
        check(rejects(bad),"binary field with a risk on an unsupported cell accepted");
    } catch(...) {std::filesystem::remove_all(folder);throw;}
    std::filesystem::remove_all(folder);
}

void fullRun() {
    const auto map=steepTerrain();const auto field=steepField();const Point start{1.,1.},goal{4.2,11.};
    SafetyPolicy policy(map,field,40.,35.);
    RpA5Options options;options.planner.lambda=33.;options.budget_s=60.;options.planner_seed=7;
    options.batch_growth=options.fast_validity=options.lazy_refine=options.seed=true;options.batch_min=500;options.window_samples=4000;
    const auto run=runRpBitA5(map,field,start,goal,options);
    check(run.run.status=="EXACT_SOLUTION"&&run.run.termination=="CONVERGED_NO_IMPROVEMENT","full A5 run did not converge");
    publicationsAreSafe(run.run,policy,start,goal);
    auto work=options;work.window_samples=0;work.test_max_iterations=4000;
    const auto first=runRpBitA5(map,field,start,goal,work),second=runRpBitA5(map,field,start,goal,work);
    check(first.run.termination=="WORK_BUDGET"&&samePublications(first.run.publications,second.run.publications)&&
          first.run.iterations==second.run.iterations&&first.batches.size()==second.batches.size(),"fixed-work A5 run is not reproducible");
    std::cout<<"full A5 on the steep fixture: "<<run.run.actual_wall_s<<" s, "<<run.run.batches<<" batches, "<<run.run.refinements
             <<" refinements, J "<<run.run.publications.back().cost.objective<<", seed "<<run.seed_status<<"\n";
    // Invalid option combinations and queries.
    const auto rejects=[&](RpA5Options bad) {
        try{runRpBitA5(map,field,start,goal,bad);}catch(const std::runtime_error&){return true;}
        return false;
    };
    auto bad=options;bad.refine=false;check(rejects(bad),"seed and lazy refinement without refinement accepted");
    bad=options;bad.seed=false;bad.refine=false;check(rejects(bad),"lazy refinement without refinement accepted");
    bad=options;bad.batch_min=0;check(rejects(bad),"empty batch accepted");
    bad=options;bad.planner.potential=false;check(rejects(bad),"seed without the risk potential accepted");
    check(runRpBitA5(map,field,{-1.,1.},goal,options).run.status=="INVALID_START","invalid start accepted");
    const auto identity=runRpBitA5(map,field,start,start,options);
    check(identity.run.status=="EXACT_SOLUTION"&&identity.run.publications.size()==1&&identity.run.publications[0].cost.objective==0.,
          "identity query failed");
}
}
int main() {
    try {
        ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
        scheduleFunction();fieldExchange();a4Equivalence();batchGrowth();slopeShortcut();lazyRefine();seed();fullRun();
        std::cout<<"PASS A5: sample window and batch schedule, field exchange, A4 equivalence, slope shortcut, lazy refinement, seed\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
