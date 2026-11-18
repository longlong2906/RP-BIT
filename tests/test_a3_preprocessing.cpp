#include "terrain25d/Terrain.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

using namespace terrain25d;
namespace {
void require(bool value,const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
bool sameBits(double a,double b) {
    std::uint64_t aa=0,bb=0;
    static_assert(sizeof(aa)==sizeof(a),"binary64 is required for this bitwise regression");
    std::memcpy(&aa,&a,sizeof(a));std::memcpy(&bb,&b,sizeof(b));return aa==bb;
}
struct Fixture {
    int nx=17,ny=15;
    double x0=0,y0=0,dx=.2,dy=.2,radius=.4,slope=35;
    std::vector<double> heights=std::vector<double>(17*15);
    std::vector<int> mask=std::vector<int>(16*14);
    std::vector<Rect> obstacles;
    Terrain make() const {
        return Terrain(nx,ny,x0,y0,dx,dy,heights,mask,obstacles,radius,0,slope);
    }
    Point point(double i,double j) const {return {x0+i*dx,y0+j*dy};}
};
void sameEdge(const Terrain &legacy,const Terrain &optimized,Point a,Point b) {
    require(legacy.validMotion(a,b)==optimized.validMotion(a,b),"hard validity changed");
    std::vector<Point3> ap,bp;
    const auto aa=legacy.edgeCostBreakdown(a,b,&ap),bb=optimized.edgeCostBreakdown(a,b,&bp);
    require(aa.has_data==bb.has_data,"edge availability changed");
    const std::array<double,5> av{{aa.cost_terrain_m,aa.risk_slope_integral_m,
        aa.risk_clearance_integral_m,aa.risk_integral_m,aa.objective_value}};
    const std::array<double,5> bv{{bb.cost_terrain_m,bb.risk_slope_integral_m,
        bb.risk_clearance_integral_m,bb.risk_integral_m,bb.objective_value}};
    for (std::size_t k=0;k<av.size();++k) require(sameBits(av[k],bv[k]),"edge tuple changed bits");
    require(ap.size()==bp.size(),"polyline size changed");
    for (std::size_t k=0;k<ap.size();++k)
        require(sameBits(ap[k].x,bp[k].x) && sameBits(ap[k].y,bp[k].y) &&
            sameBits(ap[k].z,bp[k].z),"polyline coordinates changed bits");
    const auto scalar=optimized.edgeCostBreakdown(a,b);
    require(sameBits(scalar.objective_value,bb.objective_value) &&
        sameBits(scalar.risk_integral_m,bb.risk_integral_m),"scalar and polyline differ");
}
void compare(const Fixture &fixture,double lambda,double comfort) {
    auto legacy=fixture.make(),optimized=fixture.make();
    TraversabilityConfig config;config.lambda=lambda;config.clearance_comfort_m=comfort;
    legacy.enableTraversability(config);
    optimized.enableTraversability(config,TraversabilityBuildPolicy::SkipDominated);
    require(legacy.counters().height_queries==0 && optimized.counters().height_queries==0,
        "builder changed query counters");
    for (int j=0;j<fixture.ny-1;++j) for (int i=0;i<fixture.nx-1;++i) {
        const auto a=legacy.featureCell(i,j),b=optimized.featureCell(i,j);
        require(a.available==b.available && sameBits(a.slope_penalty,b.slope_penalty) &&
            sameBits(a.clearance_penalty,b.clearance_penalty),"feature field changed bits");
    }
    const auto &old=legacy.traversabilityBuildStats(),&now=optimized.traversabilityBuildStats();
    require(now.available_cells==old.available_cells && now.slope_distance_calls<=old.slope_distance_calls &&
        now.obstacle_distance_calls<=old.obstacle_distance_calls,"unexpected geometry count regression");
    const double phase_seconds=now.slope_extraction_seconds+now.slope_envelope_seconds+now.clearance_seconds;
    require(now.total_seconds==optimized.traversabilityPreprocessingSeconds() &&
        now.total_seconds>=phase_seconds*(1.-8*std::numeric_limits<double>::epsilon()),
        "inconsistent preprocessing timing");
    optimized.enableTraversability(config,TraversabilityBuildPolicy::SkipDominated);
    require(now.total_seconds==optimized.traversabilityPreprocessingSeconds(),"build not idempotent");
    bool rejected=false;
    try {optimized.enableTraversability(config);} catch (const std::logic_error &) {rejected=true;}
    require(rejected,"immutable build policy unexpectedly changed");
    std::mt19937_64 rng(73129);
    std::uniform_real_distribution<double> ix(.2,fixture.nx-1.2),iy(.2,fixture.ny-1.2);
    for (unsigned k=0;k<32;++k) {
        const auto a=fixture.point(ix(rng),iy(rng)),b=fixture.point(ix(rng),iy(rng));
        sameEdge(legacy,optimized,a,b);sameEdge(legacy,optimized,b,a);
    }
    // Canonical boundaries, neighboring representable coordinates, diagonals,
    // and zero-length connectors exercise interval field lookup too.
    for (double x:{fixture.point(3,0).x,
                   std::nextafter(fixture.point(3,0).x,-std::numeric_limits<double>::infinity()),
                   std::nextafter(fixture.point(3,0).x,std::numeric_limits<double>::infinity())})
        sameEdge(legacy,optimized,{x,fixture.point(0,2.3).y},{x,fixture.point(0,11.7).y});
    sameEdge(legacy,optimized,fixture.point(2.3,3),fixture.point(12.7,3));
    sameEdge(legacy,optimized,fixture.point(2,2),fixture.point(11,11));
    sameEdge(legacy,optimized,fixture.point(3,3),fixture.point(3,3));
}
}
int main() {
    try {
        unsigned fixtures=0;
        for (double scale:{std::ldexp(1.,-10),1.,std::ldexp(1.,12)})
            for (bool translated:{false,true}) for (unsigned terrain_case=0;terrain_case<7;++terrain_case) {
                Fixture f;
                f.x0=translated?1048576.:0.;f.y0=translated?-1048576.:0.;
                f.dx=.2*scale;f.dy=(terrain_case==5?.3:.2)*scale;f.radius=.4*scale;
                if (terrain_case==6) f.radius=std::nextafter(f.radius,0.);
                const double z0=translated?1048576.:0.;
                for (int j=0;j<f.ny;++j) for (int i=0;i<f.nx;++i) {
                    double value=0;
                    if (terrain_case==1 || terrain_case==5 || terrain_case==6) value=.125*i*f.dx+.25*j*f.dy;
                    if (terrain_case==2) value=4*i*f.dx+2*j*f.dy; // saturated slope
                    if (terrain_case==3 && i==8 && j==7) value=3*scale; // local steep triangle
                    if (terrain_case==4) value=(i%3)*.017*scale;
                    f.heights[j*f.nx+i]=z0+value;
                }
                if (terrain_case==4) {
                    f.heights[5*f.nx+9]=std::numeric_limits<double>::quiet_NaN();
                    f.mask[8*(f.nx-1)+6]=1; // not a physical obstacle
                }
                if (terrain_case>=3) {
                    const auto a=f.point(4,3),b=f.point(4.01,10);
                    f.obstacles={{a.x,a.y,b.x,b.y},
                        {f.point(11,3).x,f.point(11,3).y,f.point(12,5).x,f.point(12,5).y},
                        {f.point(13,10).x,f.point(13,10).y,f.point(14,11).x,f.point(14,11).y}};
                }
                compare(f,terrain_case%3==0?0.:terrain_case%3==1?1.:4.,.4*scale);
                ++fixtures;
            }
        // Exact radius equality and both neighboring representable values.
        for (double radius:{std::nextafter(.4,0.),.4,std::nextafter(.4,1.)}) {
            Fixture f;f.radius=radius;f.heights[7*f.nx+8]=.1;
            f.obstacles={{1.2,.3,1.200001,1.7},{2.2,1.1,2.4,1.9}};
            compare(f,1.,.4);++fixtures;
        }
        // Clearance subtraction is deliberately not rearranged into a distance
        // threshold. Probe both ULP neighbors of its cancellation boundary at
        // different coordinate scales, with later obstacles still in the list.
        for (double scale:{std::ldexp(1.,-10),1.,std::ldexp(1.,12)})
            for (bool translated:{false,true}) for (int side:{-1,0,1}) {
                Fixture f;f.dx=.2*scale;f.dy=.2*scale;f.radius=.4*scale;
                f.x0=translated?1048576.:0.;f.y0=translated?-1048576.:0.;
                const auto box=f.make().bounds();
                const double pad=32*std::numeric_limits<double>::epsilon()*
                    std::max({1.,std::abs(box.xmin),std::abs(box.xmax),std::abs(box.ymin),
                              std::abs(box.ymax),f.radius});
                double edge=f.point(4,0).x+f.radius+pad;
                if (side) edge=std::nextafter(edge,side<0 ? -std::numeric_limits<double>::infinity() :
                                                                  std::numeric_limits<double>::infinity());
                f.obstacles={{edge,f.point(0,2).y,edge+.01*scale,f.point(0,10).y},
                    {f.point(11,3).x,f.point(11,3).y,f.point(12,5).x,f.point(12,5).y}};
                compare(f,1.,.4*scale);++fixtures;
            }
        Fixture flat;
        auto optimized=flat.make();
        require(optimized.hasCompleteConstantHeight() && !optimized.hasExplicitObstacles(),
            "complete flat input was not recognized");
        optimized.enableTraversability({},TraversabilityBuildPolicy::SkipDominated);
        require(optimized.traversabilityBuildStats().slope_distance_calls==0 &&
            optimized.traversabilityBuildStats().dominated_cell_skips>0,
            "flat builder should need no slope-distance evaluations");
        Fixture overlapped;
        overlapped.obstacles={{0,0,3.2,2.8},{.5,.5,.6,.6},{1.1,1.1,1.2,1.2}};
        auto clearance=overlapped.make();
        clearance.enableTraversability({},TraversabilityBuildPolicy::SkipDominated);
        require(clearance.traversabilityBuildStats().clearance_saturated_cells==16*14 &&
            clearance.traversabilityBuildStats().obstacle_distance_calls==16*14,
            "clearance saturation did not skip later obstacle distances");
        flat.heights[7]=std::nextafter(0.,1.);
        require(!flat.make().hasCompleteConstantHeight(),"one ULP height change recognized as constant");
        flat.heights[7]=std::numeric_limits<double>::quiet_NaN();
        require(!flat.make().hasCompleteConstantHeight(),"NoData recognized as complete constant height");
        bool rejected=false;
        try {auto map=flat.make();map.enableTraversability({},static_cast<TraversabilityBuildPolicy>(99));}
        catch (const std::invalid_argument &) {rejected=true;}
        require(rejected,"invalid build policy accepted");
        std::cout<<"PASS "<<fixtures<<" bitwise field/edge fixtures; flat geometry eliminated; exact input predicate\n";
        return 0;
    } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
