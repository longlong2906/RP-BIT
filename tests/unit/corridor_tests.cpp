#include "terrain25d/Corridor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace terrain25d;
namespace {
void check(bool condition,const char *message) { if (!condition) throw std::runtime_error(message); }
void near(double a,double b) { check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<1e-9,"numeric disagreement"); }
Terrain flat(double radius=.1,std::vector<Rect> obstacles={}) {
    Terrain map(13,13,0,0,.5,.5,std::vector<double>(169),std::vector<int>(144),std::move(obstacles),radius,0,35);
    map.enableCollisionIndex();return map;
}
std::shared_ptr<const SafetyField> field(std::vector<double> risks=std::vector<double>(36,2.),
                                        std::vector<int> support=std::vector<int>(36,1)) {
    return std::make_shared<const SafetyField>(6,6,0,0,1,6,6,std::move(support),std::move(risks));
}
CorridorOptions options() { CorridorOptions opts;opts.grid_spacing_m=.5;opts.endpoint_connection_radius_m=.75;return opts; }
void cellAndExposureInvariants() {
    auto risks=std::vector<double>(36,2.);
    for (int i=0;i<6;++i) risks[2*6+i]=8.;
    const auto f=field(risks);const auto map=flat();
    const std::vector<Point> original{{.5,2.},{5.5,2.}},split{{.5,2.},{2.25,2.},{2.25,2.},{5.5,2.}},reverse{{5.5,2.},{.5,2.}};
    const auto a=corridorPathStats(map,*f,original),b=corridorPathStats(map,*f,split),c=corridorPathStats(map,*f,reverse);
    check(a.supported&&b.supported&&c.supported,"supported path unavailable");
    check(a.cells==b.cells&&a.cells==c.cells,"unique cells changed on reversal/split/duplicate point");
    near(*a.pri,5.);near(*a.pri,*b.pri);near(*a.pri,*c.pri);
    near(*a.risk_sum,60.);check(a.risk_count==12,"closed grid line missed incident cells");
    near(*a.exposure,40.);near(*a.exposure,*b.exposure);near(*a.exposure,*c.exposure);
    auto support=std::vector<int>(36,1);support[2*6+3]=0;risks[2*6+3]=0.;
    check(!corridorPathStats(map,*field(risks,support),original).supported,"missing support became zero risk");
    // An isolated corner touches four cells for PRI, but has zero integral measure.
    const auto singleton=corridorPathStats(map,*f,{{2.,2.}});
    check(singleton.risk_count==4,"closed vertex did not include four cells");near(*singleton.exposure,0.);
    Terrain plane(13,13,0,0,.5,.5,[](){std::vector<double> h;for(int j=0;j<13;++j)for(int i=0;i<13;++i)h.push_back(i*.125);return h;}(),
        std::vector<int>(144),{},.1,0,35);
    const auto constant=field();const auto planeStats=corridorPathStats(plane,*constant,{{.5,.5},{5.5,4.5}});
    near(*planeStats.exposure,2.*std::sqrt(25.+16.+1.5625));
}
void maskAndFootprint() {
    const auto map=flat(.6);const auto f=field();std::vector<int> mask(36);mask[2*6+2]=1;mask[2*6+4]=1;
    const SafetyPolicy p(map,f,40,35,mask);
    check(p.validState({2.5,2.5}),"corridor was incorrectly inflated by footprint");
    check(p.validState({4.5,2.5}),"safe second endpoint rejected");
    check(p.rejectionReason({2.5,2.5},{4.5,2.5})=="outside_corridor","middle of edge escaped mask");
    check(!p.validState({3.,2.5}),"closed-cell mask boundary not enforced");
    near(p.atlasArea(),2.);
    for (unsigned k=0;k<100;++k) {
        const auto q=p.sampleAtlas((k+.5)/100.,.5,.5);
        check(mask[static_cast<int>(q.y)*6+static_cast<int>(q.x)]==1,"atlas proposed outside mask");
    }
    const auto obstacleMap=flat(.1,{{2.99,0,3.01,6}});
    const SafetyPolicy full(obstacleMap,f,40,35,std::vector<int>(36,1));
    check(!full.validMotion({2.5,2.5},{4.5,2.5}),"thin physical obstacle escaped full-edge check");
    bool rejected=false;try { SafetyPolicy malformed(map,f,40,35,{1}); } catch (const std::invalid_argument &) { rejected=true; }
    check(rejected,"malformed corridor mask accepted");
}
void guidanceAndLimits() {
    const auto map=flat();auto risks=std::vector<double>(36,1.);
    for (int i=0;i<6;++i) risks[2*6+i]=20.;
    const auto f=field(risks);auto opts=options();opts.alpha=1.3;opts.dilation_cells=0;
    const Point s{.5,2.5},g{5.5,2.5};
    const auto result=computeCorridor(map,*f,s,g,opts);
    check(result.connected(),"guide generation failed");near(*result.L0,5.);
    check(*result.guide_length<=opts.alpha**result.L0,"guide exceeded length budget");
    check(*result.guide_pri<20.,"same-peak low-risk detour was not explored");
    const auto stats=corridorPathStats(map,*f,result.guide);
    near(*stats.pri,*result.guide_pri);near(*stats.exposure,*result.guide_exposure);
    check(result.peak_memory_bytes<=opts.max_memory_bytes,"memory accounting exceeded budget");
    for (std::size_t i=0;i<result.guide.size();++i) for (std::size_t j=0;j<i;++j)
        check(result.guide[i].x!=result.guide[j].x||result.guide[i].y!=result.guide[j].y,"guide contains repeated vertex loop");
    const SafetyPolicy policy(map,f,40,35,result.mask);
    for (std::size_t k=1;k<result.guide.size();++k)
        check(policy.validMotion(result.guide[k-1],result.guide[k]),"selected guide not inside final corridor");
    const auto again=computeCorridor(map,*f,s,g,opts);
    check(again.mask==result.mask&&again.guide.size()==result.guide.size(),"deterministic guide changed");
    for (std::size_t k=0;k<result.guide.size();++k)
        check(again.guide[k].x==result.guide[k].x&&again.guide[k].y==result.guide[k].y,"deterministic tie changed");
    auto exactBudget=opts;exactBudget.alpha=1.;
    const auto exactGuide=computeCorridor(map,*f,s,g,exactBudget);
    check(exactGuide.connected(),"alpha=1 discarded L0 candidate through roundoff");
    near(*exactGuide.guide_length,*exactGuide.L0);
    opts.max_labels=1;
    check(computeCorridor(map,*f,s,g,opts).status=="GUIDANCE_RESOURCE_LIMIT","label limit did not fail closed");
    opts=options();opts.max_memory_bytes=100;
    check(computeCorridor(map,*f,s,g,opts).status=="GUIDANCE_RESOURCE_LIMIT","memory limit did not cover graph allocations");
    opts=options();
    check(computeCorridor(map,*f,{0.,0.},g,opts).status=="INVALID_START","invalid start not distinguished");
    const auto same=computeCorridor(map,*f,s,s,opts);
    check(same.connected()&&same.guide.size()==1,"singleton goal did not produce one-point guide");near(*same.L0,0.);near(*same.guide_length,0.);
    const auto wall=flat(.1,{{2.99,0,3.01,6}});
    check(computeCorridor(wall,*f,s,g,opts).status=="GUIDANCE_NO_CONNECTION","finite graph disconnection not reported");
    opts.alpha=.99;bool rejected=false;
    try { computeCorridor(map,*f,s,g,opts); } catch (const std::invalid_argument &) { rejected=true; }
    check(rejected,"invalid alpha accepted");
}
void v4GeometryAndCandidates() {
    check(corridorSimplePath({{0,0},{1,0},{2,0}}),"forward collinear path rejected");
    check(!corridorSimplePath({{0,0},{2,2},{0,2},{2,0}}),"crossing guide accepted");
    check(!corridorSimplePath({{0,0},{2,0},{1,0}}),"adjacent overlap accepted");
    check(!corridorSimplePath({{0,0},{1,0},{1,1},{0,0}}),"repeated endpoint loop accepted");
    check(!corridorSimplePath({{0,0},{2,0},{2,2},{1,0}}),"nonadjacent touch accepted");
    const double above=std::nextafter(0.,1.);
    check(corridorSimplePath({{0,0},{2,0},{2,2},{1,above}}),"one-ULP noncontact treated as contact");
    const auto map=flat();auto risks=std::vector<double>(36,1.);
    for(int i=0;i<6;++i)risks[2*6+i]=20.;
    const auto f=field(risks);auto opts=options();opts.alpha=1.3;
    const Point start{.5,2.5},goal{5.5,2.5};
    const auto legacy=computeCorridor(map,*f,start,goal,opts);
    for(const auto selector:{"raw_pri","preview_pri"})for(unsigned dilation=0;dilation<=1;++dilation) {
        if(std::string(selector)=="raw_pri"&&dilation==0)continue;
        opts.selector=selector;opts.dilation_cells=dilation;
        const auto result=computeCorridorV4(map,*f,start,goal,opts);
        check(result.connected(),"v4 beam failed on connected fixture");
        check(result.candidate_diagnostics.size()<=66,"too many retained v4 candidates");
        check(result.peak_memory_bytes<=opts.max_memory_bytes,"v4 memory budget exceeded");
        check(result.labels_allocated<=opts.max_labels,"v4 shared label budget exceeded");
        check(result.preview_edge_checks<=opts.max_preview_edge_checks,"v4 preview budget exceeded");
        check(corridorSimplePath(result.guide),"v4 selected nonsimple guide");
        const auto stats=corridorPathStats(map,*f,result.guide);near(*stats.pri,*result.guide_pri);
        const SafetyPolicy policy(map,f,40.,35.,result.mask);
        for(std::size_t k=1;k<result.guide.size();++k)check(policy.validMotion(result.guide[k-1],result.guide[k]),"v4 final guide outside mask");
        unsigned selected=0;
        for(const auto &candidate:result.candidate_diagnostics) {
            selected+=candidate.selected;
            check(corridorSimplePath(candidate.raw),"beam retained nonsimple raw guide");
            check(*candidate.raw_stats.length<=opts.alpha**result.L0,"beam raw guide exceeded budget");
            if(opts.selector=="preview_pri") {
                check(corridorSimplePath(candidate.preview),"preview introduced crossing");
                check(*candidate.preview_stats.length<=*candidate.raw_stats.length,"preview increased terrain cost");
                std::size_t cursor=0;
                for(const auto p:candidate.preview) {
                    while(cursor<candidate.raw.size()&&(candidate.raw[cursor].x!=p.x||candidate.raw[cursor].y!=p.y))++cursor;
                    check(cursor<candidate.raw.size(),"preview added or reordered vertex");++cursor;
                }
                if(candidate.selected)near(*candidate.preview_stats.pri,*result.guide_pri);
            } else check(candidate.preview.empty()&&!candidate.preview_stats.supported,"raw selector secretly previewed");
        }
        check(selected==1,"v4 does not identify exactly one candidate");
        const auto again=computeCorridorV4(map,*f,start,goal,opts);
        check(again.mask==result.mask&&again.raw_guide.size()==result.raw_guide.size(),"v4 deterministic selection changed");
        near(*again.guide_pri,*result.guide_pri);
    }
    const auto legacyAgain=computeCorridor(map,*f,start,goal,options());
    check(legacy.connected()&&legacyAgain.connected(),"v3 compatibility lost");
    opts.selector="preview_pri";opts.dilation_cells=0;opts.max_preview_edge_checks=1;
    check(computeCorridorV4(map,*f,start,goal,opts).status=="GUIDANCE_RESOURCE_LIMIT","v4 preview limit silently fell back");
    opts.max_preview_edge_checks=2000000;opts.max_labels=1;
    check(computeCorridorV4(map,*f,start,goal,opts).status=="GUIDANCE_RESOURCE_LIMIT","v4 shared label limit ignored");
    opts=options();opts.selector="raw_pri";opts.dilation_cells=0;
    bool rejected=false;try{computeCorridorV4(map,*f,start,goal,opts);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"unsupported v4 raw tight-mask ablation accepted");
    opts=options();const auto singleton=computeCorridorV4(map,*f,start,start,opts);
    check(singleton.connected()&&singleton.guide.size()==1&&singleton.candidate_diagnostics.size()==1,"v4 singleton failed");
}
void v4StableBeamPolicy() {
    const auto map=flat();
    const auto f=field();
    auto opts=options();
    opts.alpha=1.3;opts.selector="raw_pri";opts.beam_scheduler="length_first_stable";
    const Point start{.5,2.5},goal{5.5,2.5};
    const auto result=computeCorridorV4(map,*f,start,goal,opts);
    const auto repeated=computeCorridorV4(map,*f,start,goal,opts);
    check(result.connected()&&repeated.connected(),"stable beam lost a feasible guide");
    check(result.beam_equal_score_retains>0,"stable beam did not exercise an exact score tie");
    check(result.beam_labels_allocated<=result.labels_allocated,"shared beam budget omitted allocations");
    check(result.beam_replacement_labels<=result.beam_labels_allocated&&
          result.beam_role_replacements>=result.beam_replacement_labels,"replacement counters inconsistent");
    check(result.beam_expansions+result.beam_stale_pops<=result.beam_labels_allocated,"queue counters inconsistent");
    check(result.beam_labels_allocated==repeated.beam_labels_allocated&&
          result.beam_role_replacements==repeated.beam_role_replacements&&
          result.mask==repeated.mask,"stable beam order changed between runs");
    opts.max_labels=result.labels_allocated-1;
    const auto limited=computeCorridorV4(map,*f,start,goal,opts);
    check(limited.status=="GUIDANCE_RESOURCE_LIMIT"&&limited.labels_allocated==opts.max_labels&&
          limited.guide.empty(),"stable beam ignored shared label cap");
    opts.max_labels=2000001;
    bool rejected=false;
    try{computeCorridorV4(map,*f,start,goal,opts);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"v4 library accepted more than two million labels");
}
}
int main() {
    try { cellAndExposureInvariants();maskAndFootprint();guidanceAndLimits();v4GeometryAndCandidates();v4StableBeamPolicy();
        std::cout<<"PASS corridor v3/v4 unique-cell PRI, beam, preview, simple geometry, immutable mask, footprint and limits\n";return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}
