#include "terrain25d/FlatDirectCertificate.hpp"
#include <cmath>

namespace terrain25d {
FlatDirectCertificate certifyFlatDirect(const Terrain &map,
    const std::vector<Point> &path,Point start,Point goal) {
    FlatDirectCertificate result;
    auto equal=[](Point a,Point b){return a.x==b.x && a.y==b.y;};
    result.complete_constant_height=map.hasCompleteConstantHeight();
    result.direct_endpoint_path=path.size()==2 && !equal(start,goal) &&
        ((equal(path[0],start)&&equal(path[1],goal)) ||
         (equal(path[0],goal)&&equal(path[1],start)));
    // Avoid lambda*R==0 or field==0: a positive quantity can underflow.
    result.zero_active_risk=!map.traversabilityEnabled() ||
        map.traversabilityConfig().lambda==0. ||
        (result.complete_constant_height &&
         (map.traversabilityConfig().clearance_weight==0. || !map.hasExplicitObstacles()));
    if(!result.complete_constant_height || !result.direct_endpoint_path || !result.zero_active_risk)
        return result;
    result.hard_validity_checked=true;
    result.hard_valid=map.validMotion(path[0],path[1]);
    if(!result.hard_valid) {result.reason="INVALID_DIRECT_CONNECTOR";return result;}
    // Complete constant H implies p_s=0 and L_H=d_xy=d_Phi. The logical
    // condition above proves lambda*R=0 mathematically. Every feasible Q has
    // J(Q)>=d_Phi, so this actual feasible incumbent attains a global bound.
    // No numerical objective is overwritten, and no sqrt equality is tested.
    result.certified=true;result.reason="PROVEN_FLAT_DIRECT";
    return result;
}
}
