#include "terrain25d/PutnStatic.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace terrain25d;
void check(bool x,const char *why){if(!x)throw std::runtime_error(why);}
void close(double a,double b){check(std::abs(a-b)<1e-8*(1+std::abs(b)),"paper numeric mismatch");}
Terrain terrain(double slope=0){std::vector<double> h;for(int j=0;j<=40;++j)for(int i=0;i<=40;++i)h.push_back(slope*i*.2);
    Terrain t(41,41,0,0,.2,.2,h,std::vector<int>(1600),{},.3,.1,35);t.enableCollisionIndex();return t;}
std::vector<Point3> cloud(double slope=0){std::vector<Point3> p;
    for(int j=0;j<160;++j)for(int i=0;i<160;++i){double x=(i+.5)*.05,y=(j+.5)*.05;p.push_back({x,y,slope*x});}return p;}
int main(){try{
    auto t=terrain();PutnStatic p(t,cloud(),true);
    auto a=p.fitPlane({2.1,2.1}),b=p.fitPlane({2.7,2.1});
    check(a.available&&b.available,"dense flat paper plane rejected");
    close(a.sparsity,0);close(a.traversability,0);close(a.flatness,0);close(a.normal.z,1);
    close(p.nativeCost(a,b),.6);check(p.nativeMotion(a,b),"flat body check failed");
    a.traversability=.2;b.traversability=.3;
    close(p.nativeCost(a,b),.6*(1+.1*(1/.8+1/.7-2)));
    a.traversability=1;check(!std::isfinite(p.nativeCost(a,b)),"singular tau was regularized");
    std::vector<PutnSnapshot> snapshots;
    check(p.solve({2.1,2.1},{2.7,2.1},11,1,true,30,[&](const auto &s){snapshots.push_back(s);})==1,"paper solve failed");
    check(!snapshots.empty()&&snapshots[0].native_planes.size()==snapshots[0].path.size(),"missing native feature evidence");
    close(snapshots[0].planner_objective,.6);
    check(p.stats().informed_proposals==30&&p.stats().informed_rejections==30,"COM informed hook inactive");
    check(p.stats().tree_consistent,"paper tree corrupt");
    // Rise crosses the 0.1 m voxel layers inside each 1 m fitting cube.
    auto gentle=terrain(.1);PutnStatic g(gentle,cloud(.1),true);
    std::vector<PutnSnapshot> safe_cost_events;
    check(g.solve({2.1,2.1},{2.7,2.1},11,1,true,20,
        [&](const auto &event){safe_cost_events.push_back(event);})==1,"gentle paper solve failed");
    check(!safe_cost_events.empty(),"gentle paper missing incumbent");
    const auto &event=safe_cost_events.front();
    double native_sum=0,com_length=0;
    for(std::size_t i=1;i<event.native_planes.size();++i){
        const auto &u=event.native_planes[i-1],&v=event.native_planes[i];
        check(u.traversability>0&&v.traversability>0,"gentle risk did not reach planner");
        native_sum+=g.nativeCost(u,v);
        const auto &a=u.center_of_mass,&b=v.center_of_mass;
        com_length+=std::hypot(std::hypot(a.x-b.x,a.y-b.y),a.z-b.z);
    }
    close(event.planner_objective,native_sum);
    check(event.planner_objective>com_length+1e-4,"planner ignored paper safety cost");
    auto steep=terrain(.3);PutnStatic s(steep,cloud(.3),true);
    check(!s.fitPlane({4,4}).available,"paper tau=1 did not reject steep plane");
    check(s.solve({4,4},{5,4},11,.1,true,1,{})==-2,"native endpoint rejection missing");
    bool denied=false;try{p.solve({2,2},{3,2},1,.1,false,1,{});}catch(const std::invalid_argument &){denied=true;}
    check(denied,"paper model accepted common-cost label");
    std::cout<<"PASS paper PUTN Eq1/8, dense occupancy, singular guard, native endpoint and COM informed hook\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
