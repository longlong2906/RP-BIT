#include "terrain25d/SemExposure.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace terrain25d {
namespace {
using Rational=boost::multiprecision::cpp_rational;
struct SurfaceInterval { Rational lo,hi;long double length; };

// Midpoint coordinates remain exact rationals. Binary search uses the same
// canonical binary64 boundaries as cellBounds, including coincident boundaries
// at extreme origin/spacing scales. No rounded midpoint is used for density.
std::pair<int,int> incidentRange(const Rational &coordinate,int count,double origin,
                                 double spacing,double maximum) {
    int lo=0,hi=count;
    while(lo<hi) {
        const int mid=lo+(hi-lo)/2;
        const double upper=std::min(maximum,origin+(mid+1)*spacing);
        if(Rational(upper)<coordinate)lo=mid+1;else hi=mid;
    }
    const int first=lo;lo=0;hi=count;
    while(lo<hi) {
        const int mid=lo+(hi-lo)/2;
        if(Rational(origin+mid*spacing)<=coordinate)lo=mid+1;else hi=mid;
    }
    return {first,lo};
}
std::optional<double> canonicalDensity(const SafetyField &field,const Rational &x,const Rational &y) {
    const auto box=field.bounds();
    const auto columns=incidentRange(x,field.nx(),box.xmin,field.resolution(),box.xmax);
    const auto rows=incidentRange(y,field.ny(),box.ymin,field.resolution(),box.ymax);
    bool found=false;double risk=0.;
    for(int j=rows.first;j<rows.second;++j)
        for(int i=columns.first;i<columns.second;++i) {
            if(!field.supported(i,j))return {};
            found=true;risk=std::max(risk,field.risk(i,j));
        }
    return found?std::optional<double>(risk):std::nullopt;
}
}

std::optional<double> semExposureCanonical(const Terrain &terrain,const SafetyField &field,Point a,Point b) {
    // Preserve the existing full-edge support guard; numerical safety contact
    // tolerance must not silently change the density over positive intervals.
    if(!field.edgeSummary(a,b).supported)return {};
    std::vector<double> surfaceParameters;
    const auto surface=terrain.surfaceEdge(a,b,&surfaceParameters);
    if(!surface.has_data||!std::isfinite(surface.cost))return {};
    if(a.x==b.x&&a.y==b.y)return 0.;
    if(surfaceParameters.size()!=surface.points.size())return {};
    const Rational ax(a.x),ay(a.y),dx=Rational(b.x)-ax,dy=Rational(b.y)-ay;
    std::vector<Rational> cuts{Rational(0),Rational(1)};
    std::vector<SurfaceInterval> intervals;
    intervals.reserve(surface.points.size());
    for(std::size_t i=1;i<surface.points.size();++i) {
        const auto &p=surface.points[i-1],&q=surface.points[i];
        // Lifted XY may round away from the requested endpoint at t=1,
        // or two cuts may round onto the same coordinate. Integrate on
        // the original cut parameters, which still cover the full edge.
        const Rational lo(surfaceParameters[i-1]),hi(surfaceParameters[i]);
        const long double length=std::hypot(std::hypot(static_cast<long double>(q.x)-p.x,
            static_cast<long double>(q.y)-p.y),static_cast<long double>(q.z)-p.z);
        if(lo<0||hi>1||hi<=lo)return {};
        cuts.push_back(lo);cuts.push_back(hi);intervals.push_back({lo,hi,length});
    }
    if(intervals.empty()||intervals.front().lo!=0||intervals.back().hi!=1)return {};
    for(std::size_t i=1;i<intervals.size();++i)if(intervals[i-1].hi!=intervals[i].lo)return {};
    const auto box=field.bounds();
    const double starts[]={a.x,a.y},ends[]={b.x,b.y},origins[]={box.xmin,box.ymin};
    const int sizes[]={field.nx(),field.ny()};
    for(unsigned axis=0;axis<2;++axis) {
        const Rational &delta=axis==0?dx:dy;
        const Rational &start=axis==0?ax:ay;
        if(delta==0)continue;
        const int first=std::max(1,static_cast<int>(std::floor((std::min(starts[axis],ends[axis])-origins[axis])/field.resolution())));
        const int last=std::min(sizes[axis]-1,static_cast<int>(std::ceil((std::max(starts[axis],ends[axis])-origins[axis])/field.resolution())));
        for(int index=first;index<=last;++index) {
            // Match canonical field cellBounds: binary64 origin + index*spacing.
            const double boundary=origins[axis]+index*field.resolution();
            const Rational t=(Rational(boundary)-start)/delta;
            if(t>0&&t<1)cuts.push_back(t);
        }
    }
    std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
    long double exposure=0.;std::size_t surfaceIndex=0;
    for(std::size_t i=1;i<cuts.size();++i) {
        const Rational &lo=cuts[i-1],&hi=cuts[i];
        const Rational midpoint=(lo+hi)/2;
        while(surfaceIndex<intervals.size()&&midpoint>=intervals[surfaceIndex].hi)++surfaceIndex;
        if(surfaceIndex==intervals.size()||lo<intervals[surfaceIndex].lo||hi>intervals[surfaceIndex].hi)return {};
        const auto risk=canonicalDensity(field,ax+midpoint*dx,ay+midpoint*dy);
        if(!risk)return {};
        const auto &piece=intervals[surfaceIndex];
        const Rational fraction=(hi-lo)/(piece.hi-piece.lo);
        exposure+=static_cast<long double>(*risk)*piece.length*fraction.convert_to<long double>();
    }
    const double value=static_cast<double>(exposure);
    return std::isfinite(value)?std::optional<double>(value):std::nullopt;
}
}
