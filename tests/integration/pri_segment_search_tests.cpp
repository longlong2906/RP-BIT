#include "terrain25d/PriSegmentSearch.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

using namespace terrain25d;
namespace {
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void near(double a,double b){check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-9+1e-9*std::max(std::abs(a),std::abs(b)),"independent numeric oracle mismatch");}
Terrain makeTerrain(std::vector<Rect> obstacles={},bool missing=false,double slope=0.){
    std::vector<double> h;for(int j=0;j<=10;++j)for(int i=0;i<=10;++i)h.push_back(slope*i);
    if(missing)h[5*11+5]=std::numeric_limits<double>::quiet_NaN();
    return Terrain(11,11,0,0,1,1,h,std::vector<int>(100),std::move(obstacles),.1,0.,35.);
}
std::shared_ptr<const SafetyField> makeField(const std::vector<double>& r=std::vector<double>(100,1.),
                                          const std::vector<int>& support=std::vector<int>(100,1)){
    return std::make_shared<const SafetyField>(10,10,0,0,1,10,10,support,r);
}
// Independent exhaustive geometry: intersect every expanded closed rectangle
// using orientation predicates, not production slab traversal or touchedCells.
struct P {long double x,y;};
long double cross(P a,P b,P c){return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);}
bool intersect(P a,P b,P c,P d){
    if(std::max(a.x,b.x)<std::min(c.x,d.x)||std::max(c.x,d.x)<std::min(a.x,b.x)||
       std::max(a.y,b.y)<std::min(c.y,d.y)||std::max(c.y,d.y)<std::min(a.y,b.y))return false;
    return cross(a,b,c)*cross(a,b,d)<=0.&&cross(c,d,a)*cross(c,d,b)<=0.;
}
long double pad(int coordinate){const double v=std::max(1.,std::abs(static_cast<double>(coordinate)));
    return 16.L*(std::nextafter(v,INFINITY)-v);}
bool touches(Point aa,Point bb,int x,int y){
    const P a{aa.x,aa.y},b{bb.x,bb.y};
    const long double loX=x-pad(x),hiX=x+1+pad(x+1),loY=y-pad(y),hiY=y+1+pad(y+1);
    const auto inside=[&](P q){return q.x>=loX&&q.x<=hiX&&q.y>=loY&&q.y<=hiY;};
    const P sw{loX,loY},se{hiX,loY},ne{hiX,hiY},nw{loX,hiY};
    return inside(a)||inside(b)||intersect(a,b,sw,se)||intersect(a,b,se,ne)||intersect(a,b,ne,nw)||intersect(a,b,nw,sw);
}
std::set<std::size_t> cells(const std::vector<Point>& path){
    std::set<std::size_t> out;if(path.empty())return out;
    for(std::size_t k=0;k<std::max<std::size_t>(1,path.size()-1);++k)
        for(int y=0;y<10;++y)for(int x=0;x<10;++x)
            if(touches(path[k],path.size()>1?path[k+1]:path[k],x,y))out.insert(y*10+x);
    return out;
}
double analyticLength(const std::vector<Point>& path,double slope=0.){
    long double out=0.;for(std::size_t k=1;k<path.size();++k){const auto dx=path[k].x-path[k-1].x,dy=path[k].y-path[k-1].y;
        out+=std::sqrt(dx*dx+dy*dy+slope*slope*dx*dx);}return static_cast<double>(out);
}
bool simple(const std::vector<Point>& path){
    for(std::size_t i=1;i<path.size();++i)for(std::size_t j=i+2;j<path.size();++j)
        if(intersect({path[i-1].x,path[i-1].y},{path[i].x,path[i].y},
                     {path[j-1].x,path[j-1].y},{path[j].x,path[j].y}))return false;
    for(std::size_t i=1;i+1<path.size();++i){const P a{path[i-1].x,path[i-1].y},b{path[i].x,path[i].y},c{path[i+1].x,path[i+1].y};
        if(cross(a,b,c)==0.&&(a.x-b.x)*(c.x-b.x)+(a.y-b.y)*(c.y-b.y)>0.)return false;}
    return true;
}
void verify(const PriSegmentSearchResult& r,const std::vector<double>& risk,double slope=0.){
    check(r.status=="FOUND"&&r.stats.valid&&!r.path.empty(),"expected fully scored segment result");
    const auto expected=cells(r.path);check(std::vector<std::size_t>(expected.begin(),expected.end())==r.stats.cells,"independent unique-cell union differs");
    long double sum=0.;for(auto id:expected)sum+=risk[id];near(r.stats.risk_sum,static_cast<double>(sum));
    near(r.stats.pri,static_cast<double>(sum/expected.size()));near(r.stats.length,analyticLength(r.path,slope));
    near(r.stats.length_xy,analyticLength(r.path));check(simple(r.path),"accepted path self-intersects");
    check(r.stats.length<=r.length_cap&&r.stats.pri<=r.pri_cap,"strict output cap violated");
}
PriSegmentSearchOptions options(double length=30.,double pri=40.){
    PriSegmentSearchOptions o;o.length_cap=length;o.pri_cap=pri;o.seed=719;o.generate_nodes=false;
    o.max_nodes=24;o.beam_width=32;o.max_expansions=512;o.max_depth=8;return o;
}
void cellUnionAndBoundaries(){
    auto t=makeTerrain({},false,.25);std::vector<double> risk(100);for(int i=0;i<100;++i)risk[i]=1.+(i%7);auto f=makeField(risk);
    std::vector<std::vector<Point>> paths;
    for(double y:{5.,std::nextafter(5.,0.),std::nextafter(5.,10.)}){
        paths.push_back({{1.5,y},{8.5,y}});paths.push_back({{8.5,y},{1.5,y}});paths.push_back({{1.5,y},{4.,y},{8.5,y}});
    }
    paths.push_back({{1.,1.},{8.,8.}});paths.push_back({{8.,8.},{1.,1.}});paths.push_back({{1.,1.},{4.,4.},{8.,8.}});
    for(const auto& path:paths){const auto r=runPriSegmentSearch(t,f,path,0,path.size()-1,options());verify(r,risk,.25);}
    // Distinct non-crossing edges can share the same SEM cell. The outside
    // union must retain that cell when its middle edge is removed.
    const std::vector<Point> parent{{1.1,2.1},{4.1,2.1},{4.7,2.7},{6.1,2.1}};
    auto o=options();o.extra_nodes={{4.5,3.5}};auto r=runPriSegmentSearch(t,f,parent,1,2,o);verify(r,risk,.25);
    auto expected=cells({parent[0],parent[1]});const auto suffix=cells({parent[2],parent[3]});expected.insert(suffix.begin(),suffix.end());
    check(std::vector<std::size_t>(expected.begin(),expected.end())==r.outside_cells,"outside union incorrectly removed shared cells");
    check(expected.count(24)&&std::find(r.stats.cells.begin(),r.stats.cells.end(),24)!=r.stats.cells.end(),"shared risk cell disappeared");
}
void wholeSegmentAndLengthContraction(){
    auto t=makeTerrain();std::vector<double> risk(100,1.);for(int y=3;y<=6;++y)for(int x=3;x<=6;++x)risk[y*10+x]=35.;auto f=makeField(risk);
    const Point a{1.5,5.5},b{8.5,5.5};const std::vector<Point> parent{a,{1.5,8.5},{8.5,8.5},b};
    auto o=options(9.5,4.);o.extra_nodes={{2.5,7.5},{7.5,7.5}};
    const auto r=runPriSegmentSearch(t,f,parent,0,3,o);verify(r,risk);
    check(r.path.size()>=4,"fixture bypassed required multi-bend replacement");
    check(r.stats.length<analyticLength(parent),"low-PRI route was not contracted");
    long double directRisk=0.;auto directCells=cells({a,b});for(auto id:directCells)directRisk+=risk[id];
    check(directRisk/directCells.size()>o.pri_cap,"direct shortcut unexpectedly satisfies risk cap");
    auto replay=runPriSegmentSearch(t,f,parent,0,3,o);verify(replay,risk);
    check(r.path.size()==replay.path.size(),"fixed graph replay changed path length");
    for(std::size_t i=0;i<r.path.size();++i)check(r.path[i].x==replay.path[i].x&&r.path[i].y==replay.path[i].y,"fixed graph replay changed path");
    check(r.counters.expansions==replay.counters.expansions&&r.counters.edge_evaluations==replay.counters.edge_evaluations&&
          r.counters.labels_created==replay.counters.labels_created&&r.counters.beam_discarded==replay.counters.beam_discarded,"fixed graph replay changed counters");
}
void nonmonotonePrefixes(){
    auto t=makeTerrain();std::vector<double> risk(100,0.);risk[51]=35.;risk[52]=35.;auto f=makeField(risk);
    auto o=options(9.,10.);o.extra_nodes={{2.5,5.5},{4.5,5.5},{6.5,5.5}};
    auto r=runPriSegmentSearch(t,f,{{1.5,5.5},{8.5,5.5}},0,1,o);verify(r,risk);
    check(r.counters.prefix_above_pri_kept>0,"high-PRI prefixes were never retained");
    // No final path can satisfy PRI=0 in this constant positive field. This
    // leaves multiple genuinely different prefix cell sets at the same node
    // available for inspection, without incumbent-length pruning masking it.
    auto diverse=options(25.,0.);diverse.extra_nodes={{2.5,3.5},{4.5,7.5},{6.5,3.5}};diverse.max_depth=4;
    const auto labels=runPriSegmentSearch(t,makeField(),{{1.5,5.5},{8.5,5.5}},0,1,diverse);
    check(labels.path.empty()&&labels.counters.distinct_sets_same_node>0,"different prefix cell sets were merged at endpoint");
    // Equal scalar prefix labels need not stay equal after the same suffix:
    // only one already contains the suffix's high-risk cell.
    const std::set<int> left{1,2},right{3,4},suffix{2};const std::vector<double> risks{0.,0.,20.,0.,20.};
    auto average=[&](std::set<int> ids){ids.insert(suffix.begin(),suffix.end());double sum=0.;for(int id:ids)sum+=risks[id];return sum/ids.size();};
    check(average(left)<average(right),"set-overlap counterexample invalid");
}
void safetyAndCancellation(){
    std::vector<double> risk(100,1.);auto f=makeField(risk);
    const std::vector<Point> detour{{1.5,5.5},{1.5,7.5},{8.5,7.5},{8.5,5.5}};
    for(const auto obstacle:std::vector<Rect>{{4.99,4.9,5.01,6.1},{4.99,5.59,5.01,6.1}}){
        auto t=makeTerrain({obstacle});auto r=runPriSegmentSearch(t,f,detour,0,3,options());verify(r,risk);
        check(r.counters.invalid_edges>0,"thin obstacle/footprint fixture did not reject edges");
        check(r.path.size()>2,"shortcut crossed obstacle or footprint clearance");
    }
    auto missing=makeTerrain({},true);auto r=runPriSegmentSearch(missing,f,detour,0,3,options());verify(r,risk);
    check(r.counters.invalid_edges>0&&r.path.size()>2,"NoData shortcut was accepted");
    auto t=makeTerrain();auto support=std::vector<int>(100,1);support[54]=0;risk[54]=0.;auto unsupported=makeField(risk,support);
    r=runPriSegmentSearch(t,unsupported,detour,0,3,options());verify(r,risk);check(r.counters.invalid_edges>0,"missing support bypassed");
    auto invalid=runPriSegmentSearch(t,f,{{1.,1.},{3.,3.},{1.,3.},{3.,1.}},0,3,options());
    check(invalid.status=="INVALID_PARENT"&&invalid.path.empty(),"self-cut parent accepted");
    auto identity=runPriSegmentSearch(t,f,{{2.5,2.5}},0,0,options());check(identity.path.empty()&&identity.status=="IDENTITY_PATH","identity operator fabricated move");
    auto small=options(6.,40.);auto bound=runPriSegmentSearch(t,f,{{1.5,5.5},{8.5,5.5}},0,1,small);
    check(bound.path.empty(),"length lower bound ignored");
    auto base=options();base.extra_nodes={{2.5,6.5},{7.5,6.5}};unsigned calls=0;
    auto cancelled=base;cancelled.cancel=[&](){++calls;return true;};
    auto stopped=runPriSegmentSearch(t,f,detour,0,3,cancelled);check(stopped.status=="CANCELLED"&&stopped.path.empty(),"initial cancellation ignored");
    // Sweep cancellation checks through completed scoring as well as graph
    // construction. Any canceled computation must publish no candidate.
    auto complete=runPriSegmentSearch(t,f,detour,0,3,base);verify(complete,std::vector<double>(100,1.));
    const auto last=complete.counters.cancel_checks;check(last>0,"deadline checks not instrumented");
    for(std::uint64_t cutoff:{std::uint64_t(2),std::max<std::uint64_t>(2,last/2),last}){
        calls=0;cancelled=base;cancelled.cancel=[&](){return ++calls>=cutoff;};
        stopped=runPriSegmentSearch(t,f,detour,0,3,cancelled);
        check(stopped.status=="CANCELLED"&&stopped.path.empty()&&stopped.replacement.empty(),"late canceled score escaped");
    }
    // A slice end differs from the trial deadline. Direct completion is
    // already scored when expansion reaches the collinear alternative nodes;
    // soft stopping there preserves that complete path.
    auto sliced=options();sliced.extra_nodes={{3.5,5.5},{5.5,5.5}};unsigned stopCalls=0;
    sliced.stop=[&](){return ++stopCalls>=6;};
    const auto soft=runPriSegmentSearch(t,f,detour,0,3,sliced);verify(soft,std::vector<double>(100,1.));
    check(soft.termination=="SLICE_STOP"&&soft.counters.soft_stops==1,"soft slice end discarded scored path");
}
}
int main(){try{cellUnionAndBoundaries();wholeSegmentAndLengthContraction();nonmonotonePrefixes();safetyAndCancellation();
    std::cout<<"PRI segment independent geometry/context/replay PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
