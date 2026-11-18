#include "terrain25d/PutnStatic.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace terrain25d;
void require(bool value,const char *why){if(!value)throw std::runtime_error(why);}
void near(double a,double b,double e=1e-8){require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<e*(1+std::abs(b)),"numeric mismatch");}
Terrain make(std::function<double(double,double)> f,std::vector<Rect> rect={}) {
    std::vector<double> h;for(int j=0;j<=40;++j)for(int i=0;i<=40;++i)h.push_back(f(i*.2,j*.2));
    Terrain t(41,41,0,0,.2,.2,h,std::vector<int>(1600),rect,.3,.1,35);
    TraversabilityConfig c;c.lambda=1;t.enableTraversability(c,TraversabilityBuildPolicy::SkipDominated);t.enableCollisionIndex();return t;
}
int main(){try{
    auto flat=make([](double,double){return 0.;});PutnStatic model(flat);
    auto a=model.fitPlane({2.137,2.249}),b=model.fitPlane({2.737,2.249});
    require(a.available&&a.point_count>3,"source plane fitting absent");near(a.normal.z,1);near(a.center_of_mass.z,.7);
    near(a.support.x,2.137);near(a.support.y,2.249);require(model.nativeMotion(a,b),"flat native collision rejected");
    near(model.nativeCost(a,b),.6*(1+.1*(1/(1.0001-a.traversability)+1/(1.0001-b.traversability))));
    auto tilted=make([](double x,double){return .1*x;});PutnStatic tilted_model(tilted);
    auto p=tilted_model.fitPlane({4.137,4.249});require(p.available&&p.normal.z>0&&p.normal.x<-.01&&p.slope_deg>1,"slope model inactive");
    // Common cost exact plane oracle: slope envelope is constant on a planar map.
    double alpha=std::atan(.1),expected=std::sqrt(.6*.6+.06*.06)*(1+.5*std::pow(alpha/(35*3.14159265358979323846/180),2));
    std::vector<PutnSnapshot> common,own;
    require(tilted_model.solve({2.137,3.249},{2.737,3.249},11,.2,false,20,[&](auto s){common.push_back(s);})==1,"common goal failed");
    near(common.front().planner_objective,expected);
    require(tilted_model.stats().edge_cost_calls>0,"cost hook inactive");
    require(tilted_model.solve({2.137,3.249},{2.737,3.249},11,.2,true,20,[&](auto s){own.push_back(s);})==1,"own goal failed");
    require(own.front().planner_objective>common.front().planner_objective,"native objective ignored");
    // Same-work replay must retain parent choice, rewiring and incumbent path.
    std::vector<PutnSnapshot> first,second;
    require(model.solve({1,1},{7,7},111,1,false,650,[&](auto s){first.push_back(s);})==1,"fixed-work search has no solution");
    auto stats=model.stats();require(stats.rewires>0&&stats.tree_consistent,"rewiring missing or inconsistent");
    require(model.solve({1,1},{7,7},111,1,false,650,[&](auto s){second.push_back(s);})==1,"replay search failed");
    require(first.size()==second.size(),"replay event count");
    for(std::size_t k=0;k<first.size();++k){require(first[k].path.size()==second[k].path.size(),"replay path size");
        near(first[k].planner_objective,second[k].planner_objective,1e-14);double sum=0;
        for(std::size_t i=0;i<first[k].path.size();++i){auto q=first[k].path[i];auto r=second[k].path[i];
            require(q.x==r.x&&q.y==r.y,"replay coordinates");if(i){auto prev=first[k].path[i-1];require(flat.validMotion(prev,q),"invalid accepted edge");sum+=distanceXY(prev,q);}}
        near(sum,first[k].planner_objective);if(k)require(first[k].planner_objective<first[k-1].planner_objective,"non-improving callback");
    }
    auto wall=make([](double,double){return 0.;},{{3.99,0,4.01,8}});PutnStatic wall_model(wall,{{3.99,0,4.01,8}});
    int events=0;require(wall_model.solve({2,4},{6,4},11,1,false,150,[&](auto){++events;})==0&&events==0,"thin wall crossed");
    auto hole=make([](double x,double y){return (x==4&&y==4)?std::numeric_limits<double>::quiet_NaN():0.;});PutnStatic hole_model(hole);
    require(!hole_model.fitPlane({4,4}).available,"NoData model accepted");
    require(hole_model.solve({4,4},{6,4},11,.01,false,1,{})==-2,"NoData endpoint accepted");
    auto steep=make([](double x,double){return x;});PutnStatic steep_model(steep);require(!steep_model.fitPlane({4,4}).available,"hard slope ignored");
    // The terminal connector crosses a thin obstacle although both endpoints are valid.
    auto terminal=make([](double,double){return 0.;},{{3.99,0,4.01,8}});PutnStatic terminal_model(terminal);
    require(terminal_model.solve({3.58,4},{4.42,4},7,.01,false,1,{})==0,"final connector not checked");
    std::cout<<"PASS PUTN source plane/voxel model, native/common objective, rewiring replay, whole-edge hard safety\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
