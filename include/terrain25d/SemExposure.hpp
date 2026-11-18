#pragma once
#include "terrain25d/Safety.hpp"

namespace terrain25d {
// Additive line integral for the opt-in SEM objective. Density is evaluated on
// exact canonical raster cells, independently of hard safety's 16-ULP contacts.
// Defined in terrain_ompl so core-only terrain25d gains no Boost dependency.
std::optional<double> semExposureCanonical(const Terrain &terrain,const SafetyField &field,
                                          Point a,Point b);
}
