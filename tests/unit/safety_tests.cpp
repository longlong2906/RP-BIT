#include "terrain25d/Safety.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace terrain25d;
namespace {
void require(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
void near(double actual,double expected) { require(std::isfinite(actual) && std::abs(actual-expected)<=1e-10,"numeric mismatch"); }
template<class F> void rejects(F fn) {
    bool rejected=false;
    try { fn(); } catch (const std::exception &) { rejected=true; }
    require(rejected,"malformed input was accepted");
}
Terrain terrain(std::function<double(int,int)> height=[](int,int){return 0.;},double radius=.1) {
    std::vector<double> values;
    for (int j=0;j<=10;++j) for (int i=0;i<=10;++i) values.push_back(height(i,j));
    return Terrain(11,11,0,0,1,1,std::move(values),std::vector<int>(100),{},radius,0,35);
}
std::shared_ptr<const SafetyField> field(std::vector<int> support=std::vector<int>(100,1),
                                          std::vector<double> risk=std::vector<double>(100,10.)) {
    return std::make_shared<const SafetyField>(10,10,0,0,1,10,10,std::move(support),std::move(risk));
}
void supportAndRisk() {
    const auto map=terrain();
    auto support=std::vector<int>(100,1);auto risk=std::vector<double>(100,10.);
    support[55]=0;risk[55]=0;
    SafetyPolicy policy(map,field(support,risk));
    map.resetCounters();
    require(policy.validState({2.5,2.5}),"safe state rejected");
    require(map.counters().state_validity_calls==1 && map.counters().motion_validity_calls==0,
            "state safety check was counted as a geometry motion check");
    require(policy.validMotion({2.5,2.5},{3.5,2.5}),"safe motion rejected");
    require(map.counters().motion_validity_calls==1,"geometry motion counter missing");
    require(policy.validState({2.5,5.5}) && policy.validState({8.5,5.5}),"endpoints should be safe");
    require(policy.rejectionReason({2.5,5.5},{8.5,5.5})=="missing_sem_support","missed mid-edge unsupported cell");
    require(!policy.validState({5.,5.}),"closed corner missed unsupported cell");
    require(!policy.validMotion({5.,3.5},{5.,7.5}),"closed grid line missed unsupported cell");
    require(policy.validState({4.5,4.5}),"support mask expanded beyond touched cells");
    const Point a{2.5,5.5},b{8.5,5.5},m{4.25,5.5};
    require(policy.validMotion(a,b)==policy.validMotion(b,a),"reverse validity changed");
    require(policy.validMotion(a,b)==(policy.validMotion(a,m)&&policy.validMotion(m,b)),"split validity changed");
    support[55]=1;risk[55]=40.;
    SafetyPolicy inclusive(map,field(support,risk));
    require(inclusive.validMotion(a,b),"risk 40 should be inclusive");
    risk[55]=std::nextafter(40.,std::numeric_limits<double>::infinity());
    SafetyPolicy high(map,field(support,risk));
    require(high.rejectionReason(a,b)=="sem_risk_above_cap","risk immediately above cap accepted");
    require(high.rejectionReason({-1,2},{2,2})=="invalid_geometry","outside point accepted");
    // 16-ulp raster tie includes the adjacent cell; a clearly separate point does not.
    risk[55]=10.;support[55]=0;risk[55]=0;
    SafetyPolicy ties(map,field(support,risk));
    require(!ties.validState({std::nextafter(5.,0.),5.5}),"near boundary did not include closed tie");
    require(ties.validState({5.-1e-9,5.5}),"boundary tolerance became a physical-width strip");
}
void sweptSlope() {
    const auto plane=terrain([](int i,int){return .25*i;});
    const double slope=std::atan(.25)*180./std::acos(-1.);
    near(plane.maximumSweptSlope({2,2},{8,8}),slope);
    SafetyPolicy inclusive(plane,field(),40,slope);
    require(inclusive.validMotion({2,2},{8,8}),"slope cap should be inclusive");
    SafetyPolicy strict(plane,field(),40,std::nextafter(slope,0.));
    require(strict.rejectionReason({2,2},{8,8})=="swept_slope_above_cap","slope cap not enforced");
    // The centerline remains on a flat strip; its disk reaches the adjacent ramp.
    const auto ramp=terrain([](int,int j){return j>=5?.5*(j-5):0.;},.4);
    near(ramp.maximumSweptSlope({2,4.7},{8,4.7}),std::atan(.5)*180./std::acos(-1.));
    near(ramp.maximumSweptSlope({2,4.5},{8,4.5}),0.);
    SafetyPolicy tight(ramp,field(),40,20.);
    require(!tight.validMotion({2,4.7},{8,4.7}),"off-center ramp escaped footprint check");
    require(tight.validMotion({2,4.5},{8,4.5}),"whole-cell envelope incorrectly rejected flat footprint");
    const auto grazing=terrain([](int,int j){return j>=5?.5*(j-5):0.;},.5);
    const double rampSlope=std::atan(.5)*180./std::acos(-1.);
    near(grazing.maximumSweptSlope({2,4.5},{8,4.5}),rampSlope);
    const double below=std::nextafter(4.5,0.),above=std::nextafter(4.5,10.);
    near(grazing.maximumSweptSlope({2,below},{8,below}),0.);
    near(grazing.maximumSweptSlope({2,above},{8,above}),rampSlope);
    // Grid translation and decimal spacing must not change broad-phase coverage.
    std::vector<double> shiftedHeights;
    for (int j=0;j<=10;++j) for (int i=0;i<=10;++i) shiftedHeights.push_back(.125*i);
    const Terrain shifted(11,11,1048576.,-1048576.,.2,.2,std::move(shiftedHeights),
        std::vector<int>(100),{},.05,0,35);
    near(shifted.maximumSweptSlope({1048576.4,-1048575.6},{1048577.6,-1048574.4}),
         std::atan(.625)*180./std::acos(-1.));
    const auto missing=terrain([](int i,int j){return i==5&&j==5?std::numeric_limits<double>::quiet_NaN():0.;});
    require(!std::isfinite(missing.maximumSweptSlope({4.5,4.5},{5.5,5.5})),"missing footprint support fabricated slope");
}
void atlas() {
    // Domain is 2.5 by 2.5; the last row/column are clipped cells.
    const Terrain map(6,6,0,0,.5,.5,std::vector<double>(36),std::vector<int>(25),{},.1,0,35);
    auto support=std::vector<int>(9,1);auto risk=std::vector<double>(9,10.);
    support[0]=0;risk[0]=0;risk[8]=41;
    const auto f=std::make_shared<const SafetyField>(3,3,0,0,1,2.5,2.5,support,risk);
    const SafetyPolicy p(map,f);
    near(p.atlasArea(),5.);
    const auto first=p.sampleAtlas(0,.25,.75);
    near(first.x,1.25);near(first.y,.75);
    const auto last=p.sampleAtlas(std::nextafter(1.,0.),.5,.5);
    near(last.x,1.5);near(last.y,2.25);
    // Area-weighted sampling selects full cells twice as frequently as half cells.
    unsigned full=0,half=0;
    for (unsigned k=0;k<1000;++k) {
        const auto q=p.sampleAtlas((k+.5)/1000.,.5,.5);
        if (q.x<2 && q.y<2) ++full; else ++half;
    }
    require(full==600&&half==400,"atlas cells were not weighted by clipped area");
    rejects([&](){p.sampleAtlas(1.,0.,0.);});
    const auto empty=std::make_shared<const SafetyField>(3,3,0,0,1,2.5,2.5,std::vector<int>(9),std::vector<double>(9));
    SafetyPolicy e(map,empty);near(e.atlasArea(),0.);rejects([&](){e.sampleAtlas(.5,.5,.5);});
}
void malformed() {
    rejects([](){SafetyField(1,1,0,0,1,2,1,{1},{0});});
    rejects([](){SafetyField(1,1,0,0,1,1,1,{0},{1});});
    rejects([](){SafetyField(1,1,0,0,1,1,1,{1},{-1});});
    rejects([](){SafetyField(1,1,0,0,1,1,1,{1},{std::numeric_limits<double>::quiet_NaN()});});
    const auto map=terrain();
    rejects([&](){SafetyPolicy p(map,field(),41,35);});
    rejects([&](){SafetyPolicy p(map,field(),40,36);});
    const auto file=std::filesystem::temp_directory_path()/"terrain25d_safety_field_unit.txt";
    const std::vector<std::string> bad={"BAD 1 1 0 0 1 1 1 1 0", "SAFE_SEM_FIELD_V1 1 1 0 0 1 1 1 1",
        "SAFE_SEM_FIELD_V1 1 1 0 0 1 1 1 1 nan", "SAFE_SEM_FIELD_V1 1 1 0 0 1 1 1 1 0 trailing"};
    for (const auto &data:bad) {
        {std::ofstream out(file);out<<data;}
        rejects([&](){SafetyField::load(file.string());});
    }
    {std::ofstream out(file);out<<"SAFE_SEM_FIELD_V1 1 1 0 0 1 1 1 1 40\n";}
    near(SafetyField::load(file.string()).risk(0,0),40.);
    std::filesystem::remove(file);
}
}
int main() {
    try {
        supportAndRisk();sweptSlope();atlas();malformed();
        std::cout<<"PASS safety closed-cell support/risk, exact swept slope, area-weighted atlas and malformed input\n";
        return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}
