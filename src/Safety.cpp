#include "terrain25d/Safety.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace terrain25d {
namespace {
bool finite(Point p) { return std::isfinite(p.x) && std::isfinite(p.y); }
double number(std::istream &stream) {
    std::string token;
    if (!(stream >> token)) throw std::runtime_error("truncated safety field");
    std::size_t used = 0;
    const double value = std::stod(token, &used);
    if (used != token.size() || !std::isfinite(value))
        throw std::runtime_error("invalid safety field number");
    return value;
}
int integer(std::istream &stream) {
    const double value = number(stream);
    if (value != std::floor(value) || value < 0 || value > 100000000)
        throw std::runtime_error("invalid safety field integer");
    return static_cast<int>(value);
}
long double boundaryPad(long double coordinate) {
    const double magnitude = std::max(1., std::abs(static_cast<double>(coordinate)));
    return 16.L * (std::nextafter(magnitude, std::numeric_limits<double>::infinity()) - magnitude);
}
// Slab clipping is deliberately different from the oracle's sorted crossing
// parameters and midpoint enumeration. Expanded slabs preserve boundary ties.
bool clip(long double origin, long double delta, long double low, long double high,
          long double &enter, long double &leave) {
    if (delta == 0) return origin >= low && origin <= high;
    long double a = (low-origin)/delta, b = (high-origin)/delta;
    if (a > b) std::swap(a,b);
    enter = std::max(enter,a); leave = std::min(leave,b);
    return enter <= leave;
}
bool unit(double value) { return std::isfinite(value) && value >= 0 && value < 1; }
}

SafetyField::SafetyField(int nx, int ny, double x0, double y0, double resolution,
                         double xmax, double ymax, std::vector<int> support,
                         std::vector<double> risk)
    : nx_(nx), ny_(ny), x0_(x0), y0_(y0), resolution_(resolution), xmax_(xmax), ymax_(ymax),
      support_(std::move(support)), risk_(std::move(risk)) {
    if (nx <= 0 || ny <= 0 || nx > 100000 || ny > 100000 ||
        static_cast<long long>(nx)*ny > 100000000 ||
        support_.size() != static_cast<std::size_t>(nx)*ny || risk_.size() != support_.size() ||
        !std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(resolution) || resolution <= 0 ||
        !std::isfinite(xmax) || !std::isfinite(ymax) ||
        !std::isfinite(x0+nx*resolution) || !std::isfinite(y0+ny*resolution) ||
        xmax <= x0+(nx-1)*resolution || ymax <= y0+(ny-1)*resolution ||
        xmax > x0+nx*resolution || ymax > y0+ny*resolution)
        throw std::runtime_error("invalid safety field dimensions/bounds");
    for (std::size_t k=0; k<support_.size(); ++k) {
        if ((support_[k] != 0 && support_[k] != 1) || !std::isfinite(risk_[k]) || risk_[k] < 0 ||
            (!support_[k] && risk_[k] != 0))
            throw std::runtime_error("invalid safety field support/risk");
    }
}

SafetyField SafetyField::load(const std::string &path) {
    std::ifstream stream(path);
    std::string magic;
    if (!stream || !(stream >> magic) || magic != "SAFE_SEM_FIELD_V1")
        throw std::runtime_error("invalid safety field header: "+path);
    const int nx=integer(stream), ny=integer(stream);
    if (nx<=0 || ny<=0 || nx>100000 || ny>100000 || static_cast<long long>(nx)*ny>100000000)
        throw std::runtime_error("invalid safety field dimensions");
    const double x0=number(stream), y0=number(stream), resolution=number(stream),
        xmax=number(stream), ymax=number(stream);
    std::vector<int> support(static_cast<std::size_t>(nx)*ny);
    std::vector<double> risk(support.size());
    for (std::size_t k=0; k<support.size(); ++k) { support[k]=integer(stream); risk[k]=number(stream); }
    if (stream >> magic) throw std::runtime_error("trailing safety field data");
    return SafetyField(nx,ny,x0,y0,resolution,xmax,ymax,std::move(support),std::move(risk));
}
std::size_t SafetyField::index(int i,int j) const {
    if (i<0 || j<0 || i>=nx_ || j>=ny_) throw std::out_of_range("safety cell outside field");
    return static_cast<std::size_t>(j)*nx_+i;
}
bool SafetyField::supported(int i,int j) const { return support_[index(i,j)] != 0; }
double SafetyField::risk(int i,int j) const { return risk_[index(i,j)]; }
Rect SafetyField::cellBounds(int i,int j) const {
    index(i,j);
    return {x0_+i*resolution_,y0_+j*resolution_,
            std::min(xmax_,x0_+(i+1)*resolution_),std::min(ymax_,y0_+(j+1)*resolution_)};
}
std::string SafetyField::edgeRejectionReason(Point a,Point b,double riskCap) const {
    if (!std::isfinite(riskCap) || riskCap<0 || riskCap>40) throw std::invalid_argument("invalid risk cap");
    if (!finite(a) || !finite(b) || std::min(a.x,b.x)<x0_ || std::max(a.x,b.x)>xmax_ ||
        std::min(a.y,b.y)<y0_ || std::max(a.y,b.y)>ymax_)
        return "invalid_geometry";
    // Convert using binary64, as specified by the field raster convention.
    const long double ax=(a.x-x0_)/resolution_, ay=(a.y-y0_)/resolution_,
        bx=(b.x-x0_)/resolution_, by=(b.y-y0_)/resolution_;
    const long double dx=bx-ax, dy=by-ay;
    const int firstRow=std::max(0,static_cast<int>(std::floor(std::min(ay,by)))-1);
    const int lastRow=std::min(ny_-1,static_cast<int>(std::floor(std::max(ay,by)))+1);
    bool touched=false;
    for (int j=firstRow; j<=lastRow; ++j) {
        long double enter=0,leave=1;
        const long double low=j-boundaryPad(j), high=j+1+boundaryPad(j+1);
        if (!clip(ay,dy,low,high,enter,leave)) continue;
        const long double x1=ax+enter*dx, x2=ax+leave*dx;
        const int firstColumn=std::max(0,static_cast<int>(std::floor(std::min(x1,x2)))-1);
        const int lastColumn=std::min(nx_-1,static_cast<int>(std::floor(std::max(x1,x2)))+1);
        for (int i=firstColumn; i<=lastColumn; ++i) {
            long double entry=enter,exit=leave;
            if (!clip(ax,dx,i-boundaryPad(i),i+1+boundaryPad(i+1),entry,exit)) continue;
            touched=true;
            const auto k=static_cast<std::size_t>(j)*nx_+i;
            if (!support_[k]) return "missing_sem_support";
            if (risk_[k]>riskCap) return "sem_risk_above_cap";
        }
    }
    return touched ? std::string{} : "missing_sem_support";
}

SafetyEdgeSummary SafetyField::edgeSummary(Point a,Point b) const {
    SafetyEdgeSummary result;
    if (!finite(a) || !finite(b) || std::min(a.x,b.x)<x0_ || std::max(a.x,b.x)>xmax_ ||
        std::min(a.y,b.y)<y0_ || std::max(a.y,b.y)>ymax_) {
        result.reason="invalid_geometry";
        return result;
    }
    const long double ax=(a.x-x0_)/resolution_, ay=(a.y-y0_)/resolution_,
        bx=(b.x-x0_)/resolution_, by=(b.y-y0_)/resolution_;
    const long double dx=bx-ax, dy=by-ay;
    const int firstRow=std::max(0,static_cast<int>(std::floor(std::min(ay,by)))-1);
    const int lastRow=std::min(ny_-1,static_cast<int>(std::floor(std::max(ay,by)))+1);
    bool missing=false;
    double maximum=0.;
    for (int j=firstRow; j<=lastRow; ++j) {
        long double enter=0,leave=1;
        if (!clip(ay,dy,j-boundaryPad(j),j+1+boundaryPad(j+1),enter,leave)) continue;
        const long double x1=ax+enter*dx, x2=ax+leave*dx;
        const int firstColumn=std::max(0,static_cast<int>(std::floor(std::min(x1,x2)))-1);
        const int lastColumn=std::min(nx_-1,static_cast<int>(std::floor(std::max(x1,x2)))+1);
        for (int i=firstColumn; i<=lastColumn; ++i) {
            long double entry=enter,exit=leave;
            if (!clip(ax,dx,i-boundaryPad(i),i+1+boundaryPad(i+1),entry,exit)) continue;
            ++result.touched_cells;
            const auto k=static_cast<std::size_t>(j)*nx_+i;
            if (!support_[k]) missing=true;
            else maximum=std::max(maximum,risk_[k]);
        }
    }
    result.supported=result.touched_cells>0 && !missing;
    if (result.supported) result.maximum_risk=maximum;
    else result.reason="missing_sem_support";
    return result;
}

std::vector<std::size_t> SafetyField::touchedCells(Point a,Point b) const {
    std::vector<std::size_t> result;
    if (!finite(a) || !finite(b) || std::min(a.x,b.x)<x0_ || std::max(a.x,b.x)>xmax_ ||
        std::min(a.y,b.y)<y0_ || std::max(a.y,b.y)>ymax_) return result;
    const long double ax=(a.x-x0_)/resolution_,ay=(a.y-y0_)/resolution_,
        bx=(b.x-x0_)/resolution_,by=(b.y-y0_)/resolution_;
    const long double dx=bx-ax,dy=by-ay;
    const int firstRow=std::max(0,static_cast<int>(std::floor(std::min(ay,by)))-1);
    const int lastRow=std::min(ny_-1,static_cast<int>(std::floor(std::max(ay,by)))+1);
    for (int j=firstRow;j<=lastRow;++j) {
        long double enter=0,leave=1;
        if (!clip(ay,dy,j-boundaryPad(j),j+1+boundaryPad(j+1),enter,leave)) continue;
        const auto x1=ax+enter*dx,x2=ax+leave*dx;
        const int first=std::max(0,static_cast<int>(std::floor(std::min(x1,x2)))-1);
        const int last=std::min(nx_-1,static_cast<int>(std::floor(std::max(x1,x2)))+1);
        for (int i=first;i<=last;++i) {
            auto entry=enter,exit=leave;
            if (clip(ax,dx,i-boundaryPad(i),i+1+boundaryPad(i+1),entry,exit))
                result.push_back(static_cast<std::size_t>(j)*nx_+i);
        }
    }
    return result;
}

std::optional<double> SafetyField::edgeExposure(const Terrain &terrain,Point a,Point b) const {
    const auto summary=edgeSummary(a,b);
    if (!summary.supported) return {};
    const auto surface=terrain.surfaceEdge(a,b);
    if (!surface.has_data || !std::isfinite(surface.cost)) return {};
    if (a.x==b.x && a.y==b.y) return 0.;
    std::vector<double> cuts{0.,1.};
    const double dx=b.x-a.x,dy=b.y-a.y;
    const bool xAxis=std::abs(dx)>=std::abs(dy);
    for (const auto &p:surface.points) {
        const double t=xAxis?(p.x-a.x)/dx:(p.y-a.y)/dy;
        if (t>0 && t<1) cuts.push_back(t);
    }
    const double starts[]={a.x,a.y},deltas[]={dx,dy},origins[]={x0_,y0_};
    for (int axis=0;axis<2;++axis) if (deltas[axis]!=0) {
        const auto first=static_cast<long long>(std::ceil((std::min(starts[axis],starts[axis]+deltas[axis])-origins[axis])/resolution_));
        const auto last=static_cast<long long>(std::floor((std::max(starts[axis],starts[axis]+deltas[axis])-origins[axis])/resolution_));
        for (auto i=first;i<=last;++i) {
            const double t=(origins[axis]+i*resolution_-starts[axis])/deltas[axis];
            if (t>0 && t<1) cuts.push_back(t);
        }
    }
    std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
    long double exposure=0;
    for (std::size_t i=1;i<cuts.size();++i) {
        const double lo=cuts[i-1],hi=cuts[i],mid=lo+(hi-lo)/2;
        const Point p{a.x+lo*dx,a.y+lo*dy},q{a.x+hi*dx,a.y+hi*dy},m{a.x+mid*dx,a.y+mid*dy};
        const auto riskSummary=edgeSummary(m,m);
        if (!riskSummary.supported || !riskSummary.maximum_risk) return {};
        double zp=0,zq=0;
        if (!terrain.height(p,zp) || !terrain.height(q,zq)) return {};
        const double xy=std::hypot(q.x-p.x,q.y-p.y);
        exposure+=static_cast<long double>(*riskSummary.maximum_risk)*std::hypot(xy,zq-zp);
    }
    const double value=static_cast<double>(exposure);
    return std::isfinite(value)?std::optional<double>(value):std::nullopt;
}

SafetyPolicy::SafetyPolicy(const Terrain &terrain,std::shared_ptr<const SafetyField> field,
                           double riskCap,double slopeCap,std::vector<int> corridorMask)
    : terrain_(terrain),field_(std::move(field)),risk_cap_(riskCap),slope_cap_(slopeCap),
      corridor_mask_(std::move(corridorMask)) {
    if (!field_ || !std::isfinite(riskCap) || riskCap<0 || riskCap>40 ||
        !std::isfinite(slopeCap) || slopeCap<0 || slopeCap>35)
        throw std::invalid_argument("invalid safety policy");
    const auto a=terrain.bounds(),b=field_->bounds();
    if (a.xmin!=b.xmin || a.ymin!=b.ymin || a.xmax!=b.xmax || a.ymax!=b.ymax)
        throw std::invalid_argument("safety field/terrain bounds mismatch");
    if (!corridor_mask_.empty() && corridor_mask_.size()!=static_cast<std::size_t>(field_->nx())*field_->ny())
        throw std::invalid_argument("corridor mask dimensions mismatch");
    for (const int value:corridor_mask_) if (value!=0 && value!=1)
        throw std::invalid_argument("corridor mask must be binary");
    for (int j=0; j<field_->ny(); ++j) for (int i=0; i<field_->nx(); ++i) {
        if (!corridor_mask_.empty() && !corridor_mask_[static_cast<std::size_t>(j)*field_->nx()+i]) continue;
        if (!field_->supported(i,j) || field_->risk(i,j)>riskCap) continue;
        const auto box=field_->cellBounds(i,j);
        const double area=(box.xmax-box.xmin)*(box.ymax-box.ymin);
        atlas_area_+=area;
        if (!std::isfinite(atlas_area_) || area<=0) throw std::runtime_error("invalid safety atlas area");
        atlas_cells_.push_back(static_cast<std::size_t>(j)*field_->nx()+i);
        atlas_cumulative_area_.push_back(atlas_area_);
    }
}
std::string SafetyPolicy::rejectionReason(Point a,Point b) const {
    auto reason=field_->edgeRejectionReason(a,b,risk_cap_);
    if (!reason.empty()) return reason;
    if (!corridor_mask_.empty()) for (const auto cell:field_->touchedCells(a,b))
        if (!corridor_mask_[cell]) return "outside_corridor";
    const bool geometry=a.x==b.x && a.y==b.y ? terrain_.validState(a) : terrain_.validMotion(a,b);
    if (!geometry) return "invalid_geometry";
    const double slope=terrain_.maximumSweptSlope(a,b);
    if (!std::isfinite(slope)) return "invalid_geometry";
    return slope>slope_cap_ ? "swept_slope_above_cap" : std::string{};
}
bool SafetyPolicy::validState(Point p) const { return rejectionReason(p,p).empty(); }
bool SafetyPolicy::validMotion(Point a,Point b) const { return rejectionReason(a,b).empty(); }
Point SafetyPolicy::sampleAtlas(double areaUnit,double xUnit,double yUnit) const {
    if (!unit(areaUnit) || !unit(xUnit) || !unit(yUnit)) throw std::invalid_argument("atlas random values must be in [0,1)");
    if (atlas_cells_.empty()) throw std::runtime_error("empty safety atlas");
    const double target=std::min(areaUnit*atlas_area_,std::nextafter(atlas_area_,0.));
    const auto it=std::upper_bound(atlas_cumulative_area_.begin(),atlas_cumulative_area_.end(),target);
    const auto k=atlas_cells_.at(static_cast<std::size_t>(it-atlas_cumulative_area_.begin()));
    const auto box=field_->cellBounds(static_cast<int>(k%field_->nx()),static_cast<int>(k/field_->nx()));
    return {box.xmin+xUnit*(box.xmax-box.xmin),box.ymin+yUnit*(box.ymax-box.ymin)};
}
} // namespace terrain25d
