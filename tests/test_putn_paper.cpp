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
    // A4 options. Slope cap 10 deg is the source weight bit for bit; 35 deg admits the 16.7 deg plane.
    PutnStatic ten(steep,cloud(.3),true,10.),cap35(steep,cloud(.3),true,35.);
    for(Point q:{Point{4,4},Point{2.3,5.1}}){auto x=s.fitPlane(q),y=ten.fitPlane(q);
        check(x.available==y.available&&x.traversability==y.traversability&&x.slope_deg==y.slope_deg,"cap 10 is not the source");}
    auto wide=cap35.fitPlane({4,4});
    check(wide.available&&wide.slope_deg>16&&wide.traversability<1,"cap 35 rejected a 16.7 deg plane");
    close(wide.traversability,1000*wide.flatness+wide.slope_deg/35.+wide.sparsity);
    check(cap35.solve({4,4},{5,4},11,1,true,400,{})==1,"cap 35 endpoints or search failed");
    bool capped=false;try{PutnStatic bad(steep,cloud(.3),true,0.);}catch(const std::invalid_argument &){capped=true;}
    check(capped,"zero slope cap accepted");
    // Convergence stop (SEM-RRT* rule): stop one window after the last 0.1% improvement, and the
    // stopped search is an exact prefix of the same search without the stop.
    PutnStatic plain(t,cloud(),true),stopping(t,cloud(),true);
    std::vector<PutnSnapshot> full,cut;
    check(plain.solve({1,1},{7,7},5,1,true,6000,[&](const auto &e){full.push_back(e);})==1,"unstopped solve failed");
    check(!plain.convergence().converged,"stop active by default");
    stopping.setConvergenceStop(1e-3,300);
    check(stopping.solve({1,1},{7,7},5,1,true,6000,[&](const auto &e){cut.push_back(e);})==1,"stopped solve failed");
    const auto &c=stopping.convergence();
    check(c.converged&&c.has_reference&&stopping.stats().iterations==c.reference_iteration+300,"stop point");
    check(!cut.empty()&&cut.size()<=full.size(),"stopped events");
    for(std::size_t k=0;k<cut.size();++k){
        check(cut[k].path.size()==full[k].path.size()&&cut[k].planner_objective==full[k].planner_objective,"not a prefix");
        for(std::size_t i=0;i<cut[k].path.size();++i)
            check(cut[k].path[i].x==full[k].path[i].x&&cut[k].path[i].y==full[k].path[i].y,"prefix path differs");
    }
    check(cut.back().planner_objective>=c.reference_cost*(1-1e-3)&&cut.back().planner_objective<=c.reference_cost,
          "improvement inside the window");
    bool rejected=false;try{stopping.setConvergenceStop(0.,10);}catch(const std::invalid_argument &){rejected=true;}
    check(rejected,"zero epsilon accepted");
    stopping.setConvergenceStop(0.,0);
    check(stopping.solve({1,1},{7,7},5,1,true,500,{})==1&&!stopping.convergence().converged,"stop not cleared");
    std::cout<<"PASS paper PUTN Eq1/8, dense occupancy, singular guard, native endpoint and COM informed hook;"
               " A4 slope cap and convergence stop\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
