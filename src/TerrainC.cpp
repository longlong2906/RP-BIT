#include "terrain25d/TerrainC.h"
#include "terrain25d/PutnStatic.hpp"
#include <chrono>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace {
thread_local std::string last_error;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
struct Context {
    terrain25d::Terrain terrain;
    std::vector<terrain25d::Rect> obstacles;
    double load_seconds = 0;
    explicit Context(const char *path):terrain(terrain25d::Terrain::load(path)) {
        // The full parser above validates the file. Only explicit rectangles are copied.
        std::ifstream in(path); std::string token;
        while(in >> token) if(token == "rectangles") {
            std::size_t n; in >> n;
            for(std::size_t k=0;k<n;++k) { terrain25d::Rect r{}; in>>r.xmin>>r.ymin>>r.xmax>>r.ymax; obstacles.push_back(r); }
            break;
        }
    }
};
Context &ctx(void *p) { if(!p) throw std::invalid_argument("null terrain handle"); return *static_cast<Context *>(p); }
terrain25d::PutnStatic &planner(void *p) { if(!p) throw std::invalid_argument("null planner handle"); return *static_cast<terrain25d::PutnStatic *>(p); }
template<class F> int guarded(F f) {
    last_error.clear();
    try { return f(); } catch(const std::exception &e) { last_error=e.what(); return -1; }
    catch(...) { last_error="unknown C++ exception"; return -1; }
}
}
extern "C" {
void *tc_create(const char *path,int risk) {
    last_error.clear();
    try {
        if(!path || (risk!=0 && risk!=1)) throw std::invalid_argument("invalid create arguments");
        auto start=Clock::now(); auto c=std::make_unique<Context>(path); c->load_seconds=seconds(start);
        if(risk) { terrain25d::TraversabilityConfig config; config.lambda=1;
            c->terrain.enableTraversability(config,terrain25d::TraversabilityBuildPolicy::SkipDominated); }
        c->terrain.enableCollisionIndex(); return c.release();
    } catch(const std::exception &e) { last_error=e.what(); return nullptr; }
    catch(...) { last_error="unknown C++ exception"; return nullptr; }
}
void tc_destroy(void *p) { delete static_cast<Context *>(p); }
const char *tc_error() { return last_error.c_str(); }
int tc_valid_motion(void *p,double ax,double ay,double bx,double by) {
    return guarded([&]{return ctx(p).terrain.validMotion({ax,ay},{bx,by})?1:0;});
}
int tc_edge_cost(void *p,double ax,double ay,double bx,double by,double *out) {
    return guarded([&]{ if(!out) throw std::invalid_argument("null output");
        auto c=ctx(p).terrain.edgeCostBreakdown({ax,ay},{bx,by});
        if(!c.has_data) return 0;
        out[0]=c.cost_terrain_m;out[1]=c.risk_slope_integral_m;out[2]=c.risk_clearance_integral_m;
        out[3]=c.risk_integral_m;out[4]=c.objective_value;return 1;});
}
int tc_height(void *p,double x,double y,double *z) {
    return guarded([&]{if(!z)throw std::invalid_argument("null output");return ctx(p).terrain.height({x,y},*z)?1:0;});
}
int tc_bounds(void *p,double *out) {
    return guarded([&]{if(!out)throw std::invalid_argument("null output");auto b=ctx(p).terrain.bounds();
        out[0]=b.xmin;out[1]=b.ymin;out[2]=b.xmax;out[3]=b.ymax;return 1;});
}
int tc_timings(void *p,double *out) {
    return guarded([&]{if(!out)throw std::invalid_argument("null output");auto &c=ctx(p);
        out[0]=c.load_seconds;out[1]=c.terrain.traversabilityPreprocessingSeconds();out[2]=c.terrain.collisionIndexBuildSeconds();return 1;});
}
void *tc_putn_create(void *p) {
    last_error.clear();try {auto &c=ctx(p);return new terrain25d::PutnStatic(c.terrain,c.obstacles);}
    catch(const std::exception &e){last_error=e.what();return nullptr;}
    catch(...){last_error="unknown C++ exception";return nullptr;}
}
void *tc_putn_create_paper(void *p,const double *xyz,size_t count) {
    last_error.clear();try {
        if(!xyz||!count||count>100000000)throw std::invalid_argument("invalid paper master cloud");
        std::vector<terrain25d::Point3> cloud;cloud.reserve(count);
        for(size_t i=0;i<count;++i)cloud.push_back({xyz[3*i],xyz[3*i+1],xyz[3*i+2]});
        return new terrain25d::PutnStatic(ctx(p).terrain,cloud,true);
    }catch(const std::exception &e){last_error=e.what();return nullptr;}
    catch(...){last_error="unknown paper model exception";return nullptr;}
}
int tc_putn_snapshot_features(void *p,double *out,size_t capacity){return guarded([&]{
    const auto &planes=planner(p).lastSnapshot().native_planes;
    if(!out)return static_cast<int>(planes.size());
    if(capacity<planes.size())throw std::invalid_argument("snapshot feature output too small");
    for(size_t i=0;i<planes.size();++i){const auto &f=planes[i];double *row=out+11*i;
        row[0]=f.center_of_mass.x;row[1]=f.center_of_mass.y;row[2]=f.center_of_mass.z;
        row[3]=f.normal.x;row[4]=f.normal.y;row[5]=f.normal.z;row[6]=f.traversability;
        row[7]=f.flatness;row[8]=f.slope_deg;row[9]=f.sparsity;row[10]=static_cast<double>(f.point_count);}
    return static_cast<int>(planes.size());
});}
int tc_putn_fit_plane(void *p,double x,double y,double *out){return guarded([&]{
    if(!out)throw std::invalid_argument("null plane output");
    const auto f=planner(p).fitPlane({x,y});
    out[0]=f.center_of_mass.x;out[1]=f.center_of_mass.y;out[2]=f.center_of_mass.z;
    out[3]=f.normal.x;out[4]=f.normal.y;out[5]=f.normal.z;out[6]=f.traversability;
    out[7]=f.flatness;out[8]=f.slope_deg;out[9]=f.sparsity;out[10]=static_cast<double>(f.point_count);
    out[11]=f.available?1.:0.;return 1;
});}
int tc_putn_paper_stats(void *p,double *out){return guarded([&]{
    if(!out)throw std::invalid_argument("null paper stats output");const auto &s=planner(p).stats();
    out[0]=static_cast<double>(s.informed_proposals);out[1]=static_cast<double>(s.informed_rejections);return 1;
});}
void tc_putn_destroy(void *p){delete static_cast<terrain25d::PutnStatic *>(p);}
int tc_putn_solve(void *p,double sx,double sy,double gx,double gy,uint64_t seed,
                  double budget,int own,uint64_t fixed,tc_putn_callback cb,void *user) {
    return guarded([&]{if(own!=0&&own!=1)throw std::invalid_argument("invalid cost mode");
        return planner(p).solve({sx,sy},{gx,gy},seed,budget,own!=0,fixed,[&](const terrain25d::PutnSnapshot &s){
            if(!cb)return;std::vector<double> xy,xyz;xy.reserve(2*s.path.size());xyz.reserve(3*s.raw_com_path.size());
            for(auto q:s.path){xy.push_back(q.x);xy.push_back(q.y);}
            for(auto q:s.raw_com_path){xyz.push_back(q.x);xyz.push_back(q.y);xyz.push_back(q.z);}
            cb(s.elapsed_seconds,xy.data(),xyz.data(),s.path.size(),s.planner_objective,user);});});
}
int tc_putn_stats(void *p,double *out){return guarded([&]{if(!out)throw std::invalid_argument("null output");
    auto s=planner(p).stats();out[0]=s.iterations;out[1]=s.plane_fits;out[2]=s.model_rejections;out[3]=s.common_rejections;
    out[4]=s.edge_cost_calls;out[5]=s.rewires;out[6]=s.improvements;out[7]=s.nodes;
    out[8]=s.preprocessing_seconds;out[9]=s.solve_seconds;out[10]=s.tree_consistent?1:0;out[11]=s.conversion_seconds;return 1;});}
}
