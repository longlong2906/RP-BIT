#include "terrain25d/PriPortfolio.hpp"
#include "terrain25d/SemExposure.hpp"
#include <ompl/util/Console.h>
#include <iostream>
#include <limits>
#include <set>

using namespace terrain25d;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
void near(double a,double b){check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-9,"numeric result differs");}
Terrain terrain(std::vector<Rect> obstacles={},bool missing=false,double slope=0.){
    std::vector<double>h;for(int j=0;j<=10;++j)for(int i=0;i<=10;++i)h.push_back(slope*i);
    if(missing)h[5*11+5]=std::numeric_limits<double>::quiet_NaN();
    return Terrain(11,11,0,0,1,1,h,std::vector<int>(100),std::move(obstacles),.1,0.,35.);
}
std::shared_ptr<const SafetyField> field(std::vector<double>r=std::vector<double>(100,1.),std::vector<int>s=std::vector<int>(100,1)){
    return std::make_shared<const SafetyField>(10,10,0,0,1,10,10,std::move(s),std::move(r));
}
void uniqueCellsAndGeometry(){
    auto t=terrain();auto r=std::vector<double>(100,1.);r[5*10+4]=10.;r[5*10+5]=20.;r[5*10+6]=30.;auto f=field(r);SafetyPolicy p(t,f);
    const std::vector<Point>direct{{1.5,5.5},{8.5,5.5}},split{{1.5,5.5},{4.,5.5},{8.5,5.5}},reverse{{8.5,5.5},{1.5,5.5}},revisit{{1.5,5.5},{8.5,5.5},{1.5,5.5}};
    const auto a=priPathStats(t,p,direct),b=priPathStats(t,p,split),c=priPathStats(t,p,reverse),d=priPathStats(t,p,revisit);
    check(a.valid&&b.valid&&c.valid&&d.valid,"valid path rejected");check(a.cells.size()==8,"unique row traversal count");near(a.risk_sum,65.);near(a.pri,65./8.);
    check(a.cells==b.cells&&a.cells==c.cells&&a.cells==d.cells,"split/reverse/revisit changed unique cell set");near(a.length,7.);near(b.length,7.);near(d.length,14.);near(d.pri,a.pri);
    const auto corner=priPathStats(t,p,{{5.,5.}});check(corner.valid&&corner.cells.size()==4,"singleton corner must touch four closed cells");
    auto thin=terrain({{4.99,0.,5.01,10.}});check(!priPathStats(thin,SafetyPolicy(thin,f),direct).valid,"whole edge crossed thin obstacle");
    auto noData=terrain({},true);check(!priPathStats(noData,SafetyPolicy(noData,f),direct).valid,"whole edge crossed NoData");
    auto support=std::vector<int>(100,1);support[5*10+4]=0;r[5*10+4]=0.;auto unsupported=field(r,support);
    check(!priPathStats(t,SafetyPolicy(t,unsupported),direct).valid,"missing SEM support accepted");
    auto steep=terrain({},false,1.);check(!priPathStats(steep,SafetyPolicy(steep,f),direct).valid,"slope cap bypassed");
    check(!priPathStats(t,p,{{.05,5.5},{8.5,5.5}}).valid,"footprint boundary ignored");
}
void masksAndHash(){
    check(priMaskSha256({})=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256 empty vector");
    check(priMaskSha256({0})=="6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d","SHA256 zero byte");
    auto t=terrain();auto r=std::vector<double>(100,1.);r[5*10+4]=10.;r[5*10+5]=20.;r[5*10+6]=30.;auto f=field(r);
    const Point s{1.5,5.5},g{8.5,5.5};auto stats=priPathStats(t,SafetyPolicy(t,f),{s,g});const auto masks=priPortfolioMasks(*f,stats,s,g);
    check(masks.size()==3,"three registered mask recipes required");check(masks[0].status=="READY"&&masks[1].status=="READY"&&masks[2].status=="SKIPPED_DUPLICATE","mask readiness/dedup changed");
    check(masks[0].excluded==std::vector<std::size_t>{56},"q90 must exclude highest-risk component");
    check(masks[1].excluded==std::vector<std::size_t>({55,56}),"q75 must exclude top connected component");
    for(const auto &m:masks)if(!m.allowed.empty()){check(m.allowed[51]&&m.allowed[58],"mask removed endpoint");check(m.sha256==priMaskSha256(m.allowed),"mask hash mismatch");}
    r=std::vector<double>(100,1.);r[5*10+8]=30.;f=field(r);stats=priPathStats(t,SafetyPolicy(t,f),{s,g});
    for(const auto &m:priPortfolioMasks(*f,stats,s,g))check(m.excluded.empty(),"endpoint-touching component was excluded");
    f=field(std::vector<double>(100,0.));stats=priPathStats(t,SafetyPolicy(t,f),{s,g});near(stats.pri,0.);
    for(const auto &m:priPortfolioMasks(*f,stats,s,g))check(m.status=="SKIPPED_NO_COMPONENT","constant zero risk created artificial mask");
    for(const auto &m:priPortfolioMasks(*f,stats,s,g,[]{return true;}))check(m.status=="SKIPPED_DEADLINE","mask generation ignored deadline");
}
void acceptance(){
    check(priCandidateWithinBudget(10.,10.,2.,2.),"cap/deadline equality must pass");
    check(!priCandidateWithinBudget(std::nextafter(10.,INFINITY),10.,1.,2.),"length cap was relaxed");
    check(!priCandidateWithinBudget(9.,10.,std::nextafter(2.,INFINITY),2.),"late scoring was backdated");
    PriCandidate a,b;a.stats.pri=1.;a.stats.length=10.;b.stats.pri=2.;b.stats.length=9.;check(priCandidateBetter(a,b),"selector used length instead of exact PRI");
    a.stats.pri=b.stats.pri;check(!priCandidateBetter(a,b),"PRI tie did not minimize length");a.stats.length=b.stats.length;a.episode_id=1;b.episode_id=2;check(priCandidateBetter(a,b),"episode tie break changed");
    PriPortfolioResult result;std::optional<std::size_t>anchor;
    PriCandidate straddling;straddling.id=0;straddling.anchor_phase=true;straddling.stats.valid=true;straddling.stats.length=10.;straddling.stats.pri=2.;straddling.score_completed_s=1.01;
    considerPriCandidate(result,straddling,anchor,1.,4.);
    check(!anchor&&result.selected==0&&straddling.eligible&&straddling.published&&straddling.reason=="NO_ANCHOR_FALLBACK","quarter-straddling first solution was lost instead of preserved as fallback");
    result.candidates.push_back(straddling);
    PriCandidate late=straddling;late.id=1;late.reason.clear();late.eligible=late.published=false;late.score_completed_s=4.01;
    considerPriCandidate(result,late,anchor,1.,4.);check(!late.eligible&&!late.published&&result.selected==0,"late scoring replaced the valid fallback");
}
void realPlanner(){
    auto t=terrain();auto r=std::vector<double>(100,1.);r[5*10+4]=10.;r[5*10+5]=20.;r[5*10+6]=30.;auto f=field(r);
    PriPortfolioOptions o;o.budget_s=60.;o.length_cap_ratio=1.1;o.work_iterations_per_episode=1200;
    auto result=runPriPortfolio(t,f,{1.5,5.5},{8.5,5.5},o);
    check(result.anchor&&result.selected&&result.length_cap,"real BIT* did not provide anchor/final");
    const auto &selected=result.candidates[*result.selected];check(selected.stats.valid&&selected.stats.length<=*result.length_cap,"selected path violates validity/cap");
    check(result.counters.motion_cost_calls&&result.counters.motion_heuristic_calls&&result.counters.cost_to_go_calls&&result.counters.sampler_allocations,"real objective/heuristic/sampler hooks not called");
    check(!result.counters.sem_exposure_motion_cost_calls&&!result.counters.traversability_motion_cost_calls,"portfolio changed motionCost away from length");
    std::set<unsigned>instances;
    for(const auto &episode:result.episodes)if(episode.fresh_planner){check(instances.insert(episode.planner_instance_id).second,"mask reused existing planner");}
    check(instances.size()>=2,"test did not exercise a fresh masked planner");
    for(const auto &c:result.candidates)if(c.eligible){check(c.score_completed_s<=o.budget_s,"selected a late callback");check(c.stats.length<=*result.length_cap,"eligible candidate exceeds cap");}
    auto zero=field(std::vector<double>(100,0.));auto identity=runPriPortfolio(t,zero,{2.5,2.5},{2.5,2.5},o);
    check(identity.selected.has_value(),"identity query lost");near(identity.candidates[*identity.selected].stats.length,0.);
    auto tiny=o;tiny.budget_s=1e-12;tiny.work_iterations_per_episode=0;auto late=runPriPortfolio(t,f,{1.5,5.5},{8.5,5.5},tiny);
    check(!late.selected,"expired initial setup fabricated a solution");
}
}
int main(){try{ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);ompl::RNG::setSeed(11);uniqueCellsAndGeometry();masksAndHash();acceptance();realPlanner();std::cout<<"PRI portfolio core PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
