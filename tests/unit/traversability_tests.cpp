#include "terrain25d/Terrain.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace terrain25d;
namespace {
void require(bool condition,const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double a,double b) {
    require(std::isfinite(a) && std::isfinite(b) &&
        std::abs(a-b)<=1e-10+1e-10*std::max(std::abs(a),std::abs(b)),"cost mismatch");
}
template<class Function> void rejects(Function function) {
    bool rejected=false;
    try { function(); } catch (const std::exception &) { rejected=true; }
    require(rejected,"invalid configuration or access accepted");
}
Terrain make(std::function<double(double,double)> height,std::vector<Rect> obstacles={},
             double radius=.1,std::vector<int> mask=std::vector<int>(100)) {
    std::vector<double> heights;
    for (int j=0;j<=10;++j) for (int i=0;i<=10;++i) heights.push_back(height(i,j));
    return Terrain(11,11,0,0,1,1,std::move(heights),std::move(mask),std::move(obstacles),radius,0,35);
}
void bounded(const EdgeCostBreakdown &cost,double lambda) {
    require(cost.has_data,"missing breakdown on known terrain");
    require(cost.risk_integral_m>=0 && cost.risk_integral_m<=cost.cost_terrain_m+1e-10,
        "risk outside declared normalized bounds");
    require(cost.objective_value>=cost.cost_terrain_m &&
        cost.objective_value<=(1+lambda)*cost.cost_terrain_m+1e-10,"objective bounds violated");
}
void analyticNear(double actual,long double expected,double scale,unsigned int terms,const char *message) {
    // An absolute term scales with the map units; the relative term explicitly
    // accounts for the number of double-precision positive additions. Unlike a
    // fixed 1e-8 absolute tolerance, this remains useful on millimetre-scale maps.
    const long double epsilon=std::numeric_limits<double>::epsilon();
    const long double tolerance=4096.L*epsilon*scale+64.L*epsilon*terms*std::abs(expected);
    if (!std::isfinite(actual) || std::abs(static_cast<long double>(actual)-expected)>tolerance) {
        std::cerr<<message<<": actual="<<actual<<" expected="<<expected
                 <<" tolerance="<<tolerance<<" scale="<<scale<<" terms="<<terms<<'\n';
        throw std::runtime_error(message);
    }
}
void scaledAnalyticIntegrals() {
    // The encoded surface is exactly H(x,y)=height_origin+(x-x_origin).
    // Its slope is 45 degrees and the configured limit is 60 degrees, giving
    // p_s=(45/60)^2=9/16 without reusing the implementation's atan/gradient code.
    // With no obstacles p_c=0, R=9*L/32, and J=(1+9*lambda/32)*L.
    constexpr unsigned int pieces=1024;
    unsigned int cases=0;
    for (double scale:{std::ldexp(1.,-12),1.,std::ldexp(1.,12)})
        for (bool translated:{false,true}) for (double lambda:{0.,4.}) {
            const double origin_x=translated?std::ldexp(1.,20):0.;
            const double origin_y=translated?-std::ldexp(1.,20):0.;
            const double height_origin=translated?std::ldexp(1.,20):0.;
            std::vector<double> heights;
            for (int j=0;j<=32;++j) for (int i=0;i<=32;++i)
                heights.push_back(height_origin+i*scale);
            Terrain terrain(33,33,origin_x,origin_y,scale,scale,std::move(heights),
                std::vector<int>(32*32),{},scale/8.,0.,60.);
            TraversabilityConfig config;config.lambda=lambda;config.clearance_comfort_m=scale/2.;
            terrain.enableTraversability(config);
            const Point a{origin_x+4*scale,origin_y+4*scale};
            const Point b{origin_x+28*scale,origin_y+20*scale};
            require(terrain.validMotion(a,b),"scaled analytic fixture must be hard-valid");
            const long double dx=static_cast<long double>(b.x)-a.x;
            const long double dy=static_cast<long double>(b.y)-a.y;
            const long double length=std::sqrt(2.L*dx*dx+dy*dy);
            const long double slope=9.L*length/16.L,risk=9.L*length/32.L;
            const long double objective=length+static_cast<long double>(lambda)*risk;
            const auto whole=terrain.edgeCostBreakdown(a,b);
            analyticNear(whole.cost_terrain_m,length,scale,128,"scaled analytic length");
            analyticNear(whole.risk_slope_integral_m,slope,scale,128,"scaled analytic slope exposure");
            analyticNear(whole.risk_clearance_integral_m,0.L,scale,128,"empty-obstacle exposure");
            analyticNear(whole.risk_integral_m,risk,scale,128,"scaled analytic weighted risk");
            analyticNear(whole.objective_value,objective,scale,128,"scaled analytic objective");
            const double lower=terrain.lowerBound(a,b,true);
            analyticNear(lower,length,scale,1,"scaled lifted lower bound");
            // Equality is exact mathematically on this plane. Allow only the
            // declared arithmetic tolerance, never a heuristic inflation factor.
            require(lower<=whole.cost_terrain_m+4096*std::numeric_limits<double>::epsilon()*length,
                    "lifted lower bound exceeds analytic surface cost");
            double sum_length=0,sum_slope=0,sum_clearance=0,sum_risk=0,sum_objective=0;
            Point previous=a;
            for (unsigned int k=1;k<=pieces;++k) {
                // Dyadic fractions keep the test inputs exact even at translated
                // origins; numerical error comes from traversal/summation itself.
                const double fraction=static_cast<double>(k)/pieces;
                const Point next{a.x+(b.x-a.x)*fraction,a.y+(b.y-a.y)*fraction};
                const auto edge=terrain.edgeCostBreakdown(previous,next);
                require(edge.has_data,"small subedge unexpectedly lost terrain data");
                sum_length+=edge.cost_terrain_m;sum_slope+=edge.risk_slope_integral_m;
                sum_clearance+=edge.risk_clearance_integral_m;sum_risk+=edge.risk_integral_m;
                sum_objective+=edge.objective_value;previous=next;
            }
            analyticNear(sum_length,length,scale,pieces,"1024-edge analytic length accumulation");
            analyticNear(sum_slope,slope,scale,pieces,"1024-edge analytic slope accumulation");
            analyticNear(sum_clearance,0.L,scale,pieces,"1024-edge empty-obstacle accumulation");
            analyticNear(sum_risk,risk,scale,pieces,"1024-edge analytic risk accumulation");
            analyticNear(sum_objective,objective,scale,pieces,"1024-edge analytic objective accumulation");
            analyticNear(sum_length,whole.cost_terrain_m,scale,pieces+128,"many-edge length split invariance");
            analyticNear(sum_risk,whole.risk_integral_m,scale,pieces+128,"many-edge risk split invariance");
            analyticNear(sum_objective,whole.objective_value,scale,pieces+128,"many-edge objective split invariance");
            if (lambda==0.) {
                require(whole.objective_value==whole.cost_terrain_m && sum_objective==sum_length,
                    "scaled lambda zero changed objective arithmetic");
            }
            ++cases;
        }
    std::cout<<"PASS "<<cases<<" analytic scale/origin/height/lambda cases and "
             <<pieces<<"-edge accumulation; long_double_digits="
             <<std::numeric_limits<long double>::digits<<'\n';
}
}
int main() {
    try {
        const auto flat=[](double,double){return 0.;};
        TraversabilityConfig config;
        config.lambda=3.; config.clearance_comfort_m=3.;
        auto terrain=make(flat,{{4.6,.2,5.1,1.2}});
        const Point a{1.3,2.4},b{8.2,7.6};
        const auto legacy=terrain.surfaceEdge(a,b);
        const bool valid_before=terrain.validMotion(a,b);
        const auto disabled=terrain.edgeCostBreakdown(a,b);
        near(disabled.cost_terrain_m,legacy.cost); near(disabled.objective_value,legacy.cost);
        near(disabled.risk_integral_m,0);
        rejects([&] {terrain.traversabilityConfig();});
        rejects([&] {terrain.featureCell(0,0);});
        const auto calls=terrain.counters().full_surface_cost_calls;
        terrain.enableTraversability(config);
        const double preprocessing=terrain.traversabilityPreprocessingSeconds();
        terrain.enableTraversability(config);
        require(terrain.traversabilityEnabled() && preprocessing>=0 &&
            preprocessing==terrain.traversabilityPreprocessingSeconds(),"field is not idempotent");
        require(calls==terrain.counters().full_surface_cost_calls,"preprocessing changed query counters");
        require(terrain.validMotion(a,b)==valid_before,"risk field changed hard validity");
        near(terrain.surfaceCost(a,b),legacy.cost);
        rejects([&] {auto changed=config;changed.lambda=4;terrain.enableTraversability(changed);});
        rejects([&] {terrain.featureCell(-1,0);});
        rejects([&] {terrain.featureCell(0,10);});
        // Analytical distances for two incident cells: sqrt(.6^2+.8^2)-.1=.9,
        // and .8-.1=.7. The grid-line interval must take the larger penalty.
        near(terrain.featureCell(3,2).clearance_penalty,std::pow(1.-.9/3.,2));
        near(terrain.featureCell(4,2).clearance_penalty,std::pow(1.-.7/3.,2));
        const auto boundary=terrain.edgeCostBreakdown({4,2.2},{4,2.8});
        near(boundary.risk_clearance_integral_m,.6*std::pow(1.-.7/3.,2));
        near(boundary.risk_slope_integral_m,0);
        const double before_boundary=std::nextafter(4.,0.),after_boundary=std::nextafter(4.,10.);
        near(terrain.edgeCostBreakdown({before_boundary,2.2},{before_boundary,2.8}).risk_clearance_integral_m,
             .6*terrain.featureCell(3,2).clearance_penalty);
        near(terrain.edgeCostBreakdown({after_boundary,2.2},{after_boundary,2.8}).risk_clearance_integral_m,
             .6*terrain.featureCell(4,2).clearance_penalty);
        auto decimal_grid=Terrain(11,11,0,0,.2,.2,std::vector<double>(121),std::vector<int>(100),
            {{.75,.02,.95,.25}},.01,0,35);
        auto decimal_config=config;decimal_config.clearance_comfort_m=.4;
        decimal_grid.enableTraversability(decimal_config);
        const double canonical=.2*3;
        const double left=decimal_grid.featureCell(2,1).clearance_penalty;
        const double right=decimal_grid.featureCell(3,1).clearance_penalty;
        require(left<right,"decimal boundary fixture lacks distinct incident penalties");
        near(decimal_grid.edgeCostBreakdown({canonical,.30},{canonical,.35}).risk_clearance_integral_m,
             .05*right);
        const double canonical_before=std::nextafter(canonical,0.),canonical_after=std::nextafter(canonical,2.);
        near(decimal_grid.edgeCostBreakdown({canonical_before,.30},{canonical_before,.35}).risk_clearance_integral_m,
             .05*left);
        near(decimal_grid.edgeCostBreakdown({canonical_after,.30},{canonical_after,.35}).risk_clearance_integral_m,
             .05*right);
        std::vector<std::pair<Point,Point>> edges{{a,b},{{4,2.2},{4,8.7}},
            {{1.2,3},{8.8,3}},{{2,2},{8,8}},{{3,3},{3,3}}};
        for (const auto &edge:edges) {
            const Point p=edge.first,q=edge.second;
            const Point mid{.37*p.x+.63*q.x,.37*p.y+.63*q.y};
            std::vector<Point3> materialized;
            terrain.resetCounters();
            const auto full=terrain.edgeCostBreakdown(p,q,&materialized);
            const auto full_counters=terrain.counters();
            terrain.resetCounters();
            const auto scalar=terrain.edgeCostBreakdown(p,q);
            const auto scalar_counters=terrain.counters();
            require(full.objective_value==scalar.objective_value &&
                full.cost_terrain_m==scalar.cost_terrain_m &&
                full.risk_integral_m==scalar.risk_integral_m,"scalar/polyline changed summation");
            require(full_counters.height_queries==scalar_counters.height_queries &&
                full_counters.triangle_visits==scalar_counters.triangle_visits &&
                full_counters.full_surface_cost_calls==1 && scalar_counters.full_surface_cost_calls==1,
                "scalar/polyline query counters differ");
            const auto surface=terrain.surfaceEdge(p,q);
            require(materialized.size()==surface.points.size(),"optional polyline lost breakpoints");
            for (std::size_t k=0;k<materialized.size();++k) {
                require(materialized[k].x==surface.points[k].x && materialized[k].y==surface.points[k].y &&
                    materialized[k].z==surface.points[k].z,"optional polyline changed a breakpoint");
            }
            const auto reverse=terrain.edgeCostBreakdown(q,p);
            const auto first=terrain.edgeCostBreakdown(p,mid),second=terrain.edgeCostBreakdown(mid,q);
            near(full.cost_terrain_m,surface.cost);
            near(full.objective_value,reverse.objective_value);
            near(full.risk_slope_integral_m,reverse.risk_slope_integral_m);
            near(full.risk_clearance_integral_m,reverse.risk_clearance_integral_m);
            near(full.objective_value,first.objective_value+second.objective_value);
            near(full.risk_clearance_integral_m,first.risk_clearance_integral_m+second.risk_clearance_integral_m);
            bounded(full,config.lambda);
        }
        std::cout<<"PASS exact integration, boundary, reverse/split and scalar/polyline agreement\n";

        auto empty=make(flat);
        empty.enableTraversability(config);
        near(empty.edgeCostBreakdown(a,b).risk_integral_m,0);
        near(empty.featureCell(0,0).clearance_penalty,0); // Outside map is not a soft obstacle.
        auto masked=make(flat,{},.1,std::vector<int>(100,1));
        masked.enableTraversability(config);
        require(!masked.validMotion(a,b),"generic mask stopped being a hard constraint");
        near(masked.edgeCostBreakdown(a,b).risk_clearance_integral_m,0);
        auto thin=make(flat,{{4.99,0,5.01,10}});
        thin.enableTraversability(config);
        require(!thin.validMotion({1,5},{9,5}),"risk made forbidden shortcut valid");

        auto off_center=make([](double,double y){return .5*std::max(0.,y-3.);},{},.7);
        off_center.enableTraversability(config);
        const Point p{2.2,2.5},q{7.8,2.5};
        require(off_center.validMotion(p,q),"safe off-center slope fixture invalid");
        const auto off=off_center.edgeCostBreakdown(p,q);
        near(off.cost_terrain_m,distanceXY(p,q)); // The centerline is entirely flat.
        require(off.risk_slope_integral_m>0,"footprint slope outside centerline was missed");
        const double slope_penalty=std::pow(std::atan(.5)/(35.*std::acos(-1.)/180.),2);
        near(off.risk_slope_integral_m,distanceXY(p,q)*slope_penalty);
        auto zero_weight=make([](double x,double){return .1*x;},{{4.6,.2,5.1,1.2}});
        auto zero_config=config;zero_config.lambda=0;
        zero_weight.enableTraversability(zero_config);
        const auto zero=zero_weight.edgeCostBreakdown(a,b);
        require(zero.risk_integral_m>0,"lambda-zero regression lacked risk exposure");
        require(zero.objective_value==zero_weight.surfaceCost(a,b),"lambda zero did not restore exact objective");

        auto missing=make([](double x,double y){return x==5 && y==5 ?
            std::numeric_limits<double>::quiet_NaN() : 0.;});
        missing.enableTraversability(config);
        require(!missing.featureCell(4,4).available && !missing.featureCell(5,5).available,
            "whole-cell NoData policy lost");
        require(missing.featureCell(3,4).available,"missing neighbor invalidated known center cell");
        require(!missing.validMotion({1,5},{9,5}),"NoData no longer forbidden");
        std::vector<Point3> absent{{1,2,3}};
        const auto no_data=missing.edgeCostBreakdown({1,5},{9,5},&absent);
        require(!no_data.has_data && std::isinf(no_data.objective_value) && absent.empty(),
            "NoData returned finite cost or a partial polyline");
        std::cout<<"PASS safety separation, footprint slope, empty obstacles, NoData and lambda zero\n";

        auto corridor=make(flat,{{3,2.9,7,3.5}});
        auto trade_config=config;trade_config.lambda=10;trade_config.clearance_comfort_m=1.5;
        corridor.enableTraversability(trade_config);
        const Point start{1,4},goal{9,4},via1{1,6},via2{9,6};
        require(corridor.validMotion(start,goal) && corridor.validMotion(start,via1) &&
            corridor.validMotion(via1,via2) && corridor.validMotion(via2,goal),"tradeoff routes are not safe");
        const auto shortcut=corridor.edgeCostBreakdown(start,goal);
        double detour_length=0,detour_risk=0,detour_objective=0;
        for (const auto &edge:std::vector<std::pair<Point,Point>>{{start,via1},{via1,via2},{via2,goal}}) {
            const auto cost=corridor.edgeCostBreakdown(edge.first,edge.second);
            detour_length+=cost.cost_terrain_m;detour_risk+=cost.risk_integral_m;
            detour_objective+=cost.objective_value;
        }
        require(shortcut.cost_terrain_m<detour_length && shortcut.risk_integral_m>detour_risk &&
            shortcut.objective_value>detour_objective,"weighted objective failed known safe-route tradeoff");
        std::cout<<"PASS shorter/higher-risk versus longer/lower-risk route tradeoff\n";

        for (int invalid=0;invalid<8;++invalid) {
            auto fresh=make(flat);auto bad=config;
            if (invalid==0) bad.lambda=-1;
            if (invalid==1) bad.lambda=std::numeric_limits<double>::infinity();
            if (invalid==2) bad.slope_weight=-.5;
            if (invalid==3) bad.clearance_weight=.2;
            if (invalid==4) bad.clearance_comfort_m=0;
            if (invalid==5) bad.clearance_comfort_m=std::numeric_limits<double>::quiet_NaN();
            if (invalid==6) bad.slope_weight=std::numeric_limits<double>::quiet_NaN();
            if (invalid==7) bad.feature_version="unimplemented";
            rejects([&] {fresh.enableTraversability(bad);});
            require(!fresh.traversabilityEnabled(),"failed configuration partially enabled field");
        }
        auto zero_radius=make(flat,{},0);
        zero_radius.enableTraversability(config); // Explicit positive comfort remains well-defined.
        auto zero_slope=Terrain(2,2,0,0,1,1,{0,0,0,0},{0},{},0,0,0);
        rejects([&] {zero_slope.enableTraversability(config);});
        std::cout<<"PASS invalid configuration, immutable field and explicit zero-radius comfort\n";
        scaledAnalyticIntegrals();
        return 0;
    } catch (const std::exception &error) {
        std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;
    }
}
