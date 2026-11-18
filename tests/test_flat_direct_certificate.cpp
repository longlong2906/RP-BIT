#include "terrain25d/FlatDirectCertificate.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace terrain25d;
void require(bool value,const char *why) {if(!value)throw std::runtime_error(why);}
Terrain fixture(std::vector<double> heights={},std::vector<Rect> obstacles={},
                std::vector<int> mask={}) {
    if(heights.empty())heights.assign(121,0.);
    if(mask.empty())mask.assign(100,0);
    return Terrain(11,11,0,0,1,1,std::move(heights),std::move(mask),std::move(obstacles),.3,.1,35);
}
int main() {
    try {
        const Point s{1,5},g{9,5};const std::vector<Point> direct{s,g};
        for(double lambda:{0.,1.,4.}) {
            auto map=fixture();TraversabilityConfig config;config.lambda=lambda;map.enableTraversability(config);
            require(certifyFlatDirect(map,direct,s,g).certified,"flat direct certificate missing");
            require(certifyFlatDirect(map,{g,s},s,g).certified,"reverse path certificate missing");
            require(!certifyFlatDirect(map,{s,{5,5},g},s,g).certified,"multiwaypoint must not certify");
            require(!certifyFlatDirect(map,{s,{5,6},g},s,g).certified,"corner must not certify");
            require(!certifyFlatDirect(map,{s},s,s).certified,"identity has separate policy");
            require(!certifyFlatDirect(map,{s,{8,5}},s,g).certified,"wrong endpoint");
        }
        auto h=std::vector<double>(121,0.);h[60]=-0.;
        auto zero=fixture(h);require(certifyFlatDirect(zero,direct,s,g).certified,"signed zero");
        h[60]=std::nextafter(0.,1.);
        auto ulp=fixture(h);require(!certifyFlatDirect(ulp,direct,s,g).certified,"one ULP height mismatch");
        h[60]=std::numeric_limits<double>::quiet_NaN();
        auto missing=fixture(h);require(!certifyFlatDirect(missing,direct,s,g).certified,"NoData cannot certify");
        h.assign(121,0.);h[0]=std::numeric_limits<double>::quiet_NaN();
        auto remoteMissing=fixture(h);require(!certifyFlatDirect(remoteMissing,direct,s,g).certified,"remote NoData outside narrow certificate");
        auto obstacle=fixture({},{{4.9,4.9,5.1,5.1}});
        TraversabilityConfig config;config.lambda=0;obstacle.enableTraversability(config);
        auto invalid=certifyFlatDirect(obstacle,direct,s,g);
        require(!invalid.certified && invalid.hard_validity_checked && !invalid.hard_valid,"lambda zero must not bypass safety");
        auto offCenter=fixture({},{{4.9,5.35,5.1,5.37}});offCenter.enableTraversability(config);
        require(!certifyFlatDirect(offCenter,direct,s,g).certified,"footprint obstacle missed");
        auto mask=std::vector<int>(100,0);mask[54]=1;
        auto blocked=fixture({}, {},mask);require(!certifyFlatDirect(blocked,direct,s,g).certified,"hard mask ignored");
        for(double lambda:{1.,std::numeric_limits<double>::denorm_min()}) {
            auto distant=fixture({},{{4,8,5,9}});config.lambda=lambda;config.clearance_weight=.5;
            distant.enableTraversability(config);
            require(!certifyFlatDirect(distant,direct,s,g).certified,"positive risk coefficient cannot certify from rounded zero");
        }
        auto tinyWeight=fixture({},{{4,8,5,9}});config.lambda=1;config.clearance_weight=std::numeric_limits<double>::denorm_min();config.slope_weight=1;
        tinyWeight.enableTraversability(config);
        require(!certifyFlatDirect(tinyWeight,direct,s,g).certified,"tiny active clearance weight");
        auto zeroWeight=fixture({},{{4,8,5,9}});config.clearance_weight=0;zeroWeight.enableTraversability(config);
        require(certifyFlatDirect(zeroWeight,direct,s,g).certified,"zero clearance coefficient");
        std::cout<<"flat_direct_v1 certificate tests passed\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
