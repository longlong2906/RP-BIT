#include "terrain25d/PriPathSearch.hpp"
#include "terrain25d/Corridor.hpp"
#include <ompl/util/Console.h>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

using namespace terrain25d;
namespace {
void check(bool value,const char*message){if(!value)throw std::runtime_error(message);}
void near(double a,double b){check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-8+1e-8*std::max(std::abs(a),std::abs(b)),"numeric mismatch");}
Terrain terrain(std::vector<Rect>obstacles={},bool missing=false){
    std::vector<double>h(121,0.);if(missing)h[5*11+5]=std::numeric_limits<double>::quiet_NaN();
    return Terrain(11,11,0,0,1,1,h,std::vector<int>(100),std::move(obstacles),.1,0.,35.);
}
std::shared_ptr<const SafetyField>field(bool high=true){
    std::vector<double>risk(100,1.);if(high)risk[5*10+4]=35.;
    return std::make_shared<const SafetyField>(10,10,0,0,1,10,10,std::vector<int>(100,1),risk);
}
void compare(const PriPathStats&a,const PriPathStats&b){
    check(a.valid==b.valid,"validity differs");if(!a.valid)return;
    check(a.cells==b.cells,"unique touched cells differ");near(a.length,b.length);near(a.length_xy,b.length_xy);near(a.pri,b.pri);near(a.risk_sum,b.risk_sum);
}
void replacementIncidence(){
    auto t=terrain();auto f=field();SafetyPolicy policy(t,f);PriPathSearchCounters counters;
    const Point a{1.5,5.5},b{8.5,5.5},c{4.5,6.5};
    // The removed forward edge and retained reverse edge share all risky cells.
    // Removing only the forward edge must NOT remove their risk from the union.
    const auto changed=priPathSearchReplacementStats(t,f,{a,b,a,b},0,1,{a,c,b},&counters);
    compare(changed,priPathStats(t,policy,{a,c,b,a,b}));
    check(std::find(changed.cells.begin(),changed.cells.end(),54)!=changed.cells.end(),"retained edge lost its risk cell");
    check(counters.incidence_adds&&counters.incidence_removes&&counters.edge_cache_hits&&counters.full_recomputations,"incremental/cache path not exercised");
    const std::vector<Point>base{{1.5,5.},{8.5,5.}};
    for(const double y:{5.,std::nextafter(5.,0.),std::nextafter(5.,10.)}){
        const Point mid{4.5,y};compare(priPathSearchReplacementStats(t,f,base,0,1,{base[0],mid,base[1]}),
                                     priPathStats(t,policy,{base[0],mid,base[1]}));
    }
    // Exact-bit memo keys must not collapse nearby chords at a raster boundary.
    const Point low{4.5,std::nextafter(5.,0.)},high{4.5,std::nextafter(5.,10.)};
    compare(priPathSearchReplacementStats(t,f,{a,low,b},0,2,{a,high,b}),priPathStats(t,policy,{a,high,b}));
    const std::vector<Point>detour{a,{1.5,7.5},{8.5,7.5},b};
    auto thin=terrain({{4.99,5.,5.01,6.}});
    check(!priPathSearchReplacementStats(thin,f,detour,0,3,{a,b}).valid,"shortcut crossed thin obstacle");
    auto missing=terrain({},true);
    check(!priPathSearchReplacementStats(missing,f,detour,0,3,{a,b}).valid,"shortcut crossed NoData");
    check(!corridorSimplePath({{1.,1.},{3.,3.},{1.,3.},{3.,1.}}),"nonadjacent crossing accepted");
    check(!corridorSimplePath({{1.,1.},{3.,1.},{2.,1.}}),"overlap accepted");
    check(corridorSimplePath({{1.,1.},{2.,1.},{3.,2.}}),"ordinary corner rejected");
}
void selectors(){
    std::vector<PriPathSearchCandidate>c(5);
    for(std::size_t i=0;i<c.size();++i){c[i].id=i;c[i].stats.valid=true;c[i].stats.pri=10.-i;c[i].score_completed_s=1.;}
    c[0].stats.length=10.;c[1].stats.length=10.25;c[2].stats.length=10.5;c[3].stats.length=11.;
    c[4].stats.length=9.;c[4].score_completed_s=std::nextafter(2.,INFINITY);
    const auto selected=selectPriPathSearchCandidates(c,10.,2.);
    for(unsigned i=0;i<4;++i)check(selected[i]&&*selected[i]==i,"strict cap/deadline selector changed");
    c[1].stats.length=std::nextafter(10.25,INFINITY);
    check(selectPriPathSearchCandidates(c,10.,2.)[1]==0,"cap was relaxed by tolerance");
    c[1].stats.length=10.;c[1].stats.pri=10.;check(priPathSearchBetter(c[0],c[1]),"stable ID tie changed");
}
PriPathSearchOptions workOptions(const std::string&mode="full"){
    PriPathSearchOptions o;o.budget_s=60.;o.mode=mode;o.planner_seed=719;
    o.test_anchor_iterations=1500;o.test_iterations_per_slice=200;o.test_optimizer_proposals_per_slice=300;o.test_cycles=3;return o;
}
std::string signature(const PriPathSearchResult&r){
    std::ostringstream out;out.precision(17);out<<r.stage<<'|'<<r.candidates.size()<<'|'<<r.search_counters.proposals;
    for(const auto&c:r.candidates){out<<'|'<<c.id<<','<<c.move_type<<','<<c.stats.length<<','<<c.stats.pri;
        for(const auto&p:c.path)out<<','<<p.x<<','<<p.y;}
    return out.str();
}
void verifyResult(const Terrain&t,std::shared_ptr<const SafetyField>f,const PriPathSearchResult&r,const PriPathSearchOptions&o){
    check(r.planner_instances==1,"optimizer restarted BIT* instead of retaining it");
    check(r.counters.motion_cost_calls&&r.counters.motion_heuristic_calls,"real BIT* hooks missing");
    check(!r.counters.sem_exposure_motion_cost_calls&&!r.counters.traversability_motion_cost_calls,"BIT* motionCost changed from length");
    check(r.search_counters.peak_pool_size<=64,"pool exceeds 16x4 bound");
    SafetyPolicy policy(t,f);
    for(const auto&c:r.candidates){
        check(c.score_completed_s<=o.budget_s,"late score retained");check(corridorSimplePath(c.path),"non-simple accepted path");
        compare(c.stats,priPathStats(t,policy,c.path));for(auto id:c.parent_ids)check(id<c.id,"invalid lineage");
    }
    std::array<std::optional<std::size_t>,4>last{};double previous=0.;
    for(const auto&e:r.publications){check(e.elapsed_s>=previous&&e.elapsed_s<=o.budget_s,"publication outside time order/deadline");
        check(e.elapsed_s>=r.candidates[e.candidate_id].score_completed_s,"publication backdated score");previous=e.elapsed_s;last[e.cap_index]=e.candidate_id;}
    check(last==r.selected,"final selection differs from last publication");
    if(r.anchor){
        check(r.anchor_locked_s.has_value(),"anchor lacks actual lock timestamp");
        check(r.selected==selectPriPathSearchCandidates(r.candidates,r.candidates[*r.anchor].stats.length,o.budget_s),"selection lost an evicted/first-phase candidate");
        for(unsigned i=0;i<4;++i)check(r.selected[i]&&r.candidates[*r.selected[i]].stats.length<=*r.length_caps[i],"selected path violates cap");
    }
}
void realPlanner(){
    auto t=terrain();auto f=field();const Point s{1.5,5.5},g{8.5,5.5};auto o=workOptions();
    const auto r=runPriPathSearch(t,f,s,g,o);verifyResult(t,f,r,o);
    check(r.anchor&&r.selected[3],"BIT* failed to supply an anchor");
    check(r.search_counters.shortcut_proposals&&r.search_counters.deform_proposals&&r.search_counters.splice_proposals,"registered operators not used");
    check(r.search_counters.proposals==o.test_cycles*o.test_optimizer_proposals_per_slice,"fixed work did not terminate at exact work count");
    check(r.candidates[*r.selected[3]].stats.pri<r.candidates[*r.anchor].stats.pri,"optimizer failed to improve exact PRI after BIT* length optimum");
    check(r.candidates[*r.selected[3]].move_type!="bitstar","new improvement incorrectly attributed to BIT*");
    bool shortenedLongerParent=false;
    for(const auto&c:r.candidates)if(c.move_type!="bitstar")for(auto parent:c.parent_ids){
        const auto&p=r.candidates[parent];
        if(p.stats.length>r.candidates[*r.anchor].stats.length&&c.stats.length<p.stats.length&&
           c.stats.pri<r.candidates[*r.anchor].stats.pri)shortenedLongerParent=true;
    }
    check(shortenedLongerParent,"optimizer did not retain a temporary longer path and then shorten its descendant");
    const auto replay=runPriPathSearch(t,f,s,g,o);verifyResult(t,f,replay,o);
    check(signature(r)==signature(replay),"fixed-work optimizer replay changed");
    auto shortcut=o;shortcut.mode="shortcut";const auto sh=runPriPathSearch(t,f,s,g,shortcut);verifyResult(t,f,sh,shortcut);
    check(sh.search_counters.shortcut_proposals&&!sh.search_counters.deform_proposals&&!sh.search_counters.splice_proposals,"shortcut ablation not effective");
    auto length=o;length.mode="length";const auto pure=runPriPathSearch(t,f,s,g,length);verifyResult(t,f,pure,length);
    check(pure.selected[0]&&!pure.search_counters.proposals&&!pure.anchor,"length-only mode changed behavior");
    for(const auto&cap:pure.length_caps)check(!cap,"pure length unexpectedly capped output");
    auto tiny=o;tiny.budget_s=1e-12;const auto late=runPriPathSearch(t,f,s,g,tiny);check(!late.selected[0]&&late.publications.empty(),"setup after deadline fabricated solution");
    auto memory=o;memory.max_archive_bytes=1;const auto limited=runPriPathSearch(t,f,s,g,memory);
    check(limited.stage=="RESOURCE_LIMIT"&&limited.optimizer_resource_limit&&!limited.selected[0],"archive resource limit hidden");
    const auto identity=runPriPathSearch(t,f,{2.5,2.5},{2.5,2.5},o);
    check(identity.selected[0]&&identity.candidates[*identity.selected[0]].stats.length==0.,"identity path lost");
}
}
int main(){try{
    ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);ompl::RNG::setSeed(11);
    replacementIncidence();selectors();realPlanner();std::cout<<"PRI path search core PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
