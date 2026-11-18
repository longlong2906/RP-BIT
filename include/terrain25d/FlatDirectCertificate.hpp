#pragma once
#include "terrain25d/Terrain.hpp"

namespace terrain25d {
// Algebraic certificate in the declared input heightfield model, not a
// floating-point comparison between a rounded incumbent and lower bound.
struct FlatDirectCertificate {
    bool certified=false;
    bool complete_constant_height=false;
    bool direct_endpoint_path=false;
    bool zero_active_risk=false;
    bool hard_validity_checked=false;
    bool hard_valid=false;
    const char *reason="NOT_APPLICABLE";
};
FlatDirectCertificate certifyFlatDirect(const Terrain &map,
    const std::vector<Point> &path,Point start,Point goal);
}
