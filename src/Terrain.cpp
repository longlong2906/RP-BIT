#include "terrain25d/Terrain.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <fstream>
#include <stdexcept>

namespace terrain25d {
namespace {
bool finite(Point p) { return std::isfinite(p.x) && std::isfinite(p.y); }
Point mix(Point a, Point b, double t) { return {a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t}; }
double pointSegment(Point p, Point a, Point b) {
    const double dx=b.x-a.x, dy=b.y-a.y, d=dx*dx+dy*dy;
    const double t=d==0 ? 0 : std::clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/d, 0., 1.);
    return distanceXY(p, mix(a,b,t));
}
double pointRect(Point p, Rect r) {
    return std::hypot(std::max({r.xmin-p.x,0.,p.x-r.xmax}),
                      std::max({r.ymin-p.y,0.,p.y-r.ymax}));
}
bool intersects(Point a, Point b, Rect r) {
    double lo=0, hi=1;
    const double starts[2]={a.x,a.y}, delta[2]={b.x-a.x,b.y-a.y};
    const double mins[2]={r.xmin,r.ymin}, maxs[2]={r.xmax,r.ymax};
    for (int k=0;k<2;++k) {
        if (delta[k]==0) { if (starts[k]<mins[k] || starts[k]>maxs[k]) return false; }
        else {
            double u=(mins[k]-starts[k])/delta[k], v=(maxs[k]-starts[k])/delta[k];
            if (u>v) std::swap(u,v);
            lo=std::max(lo,u); hi=std::min(hi,v);
            if (lo>hi) return false;
        }
    }
    return true;
}
bool overlaps(Rect a, Rect b) {
    return a.xmin<=b.xmax && a.xmax>=b.xmin && a.ymin<=b.ymax && a.ymax>=b.ymin;
}
double rectangleDistance(Rect a, Rect b) {
    return std::hypot(std::max({a.xmin-b.xmax,b.xmin-a.xmax,0.}),
                      std::max({a.ymin-b.ymax,b.ymin-a.ymax,0.}));
}
bool pointInTriangle(Point p,const std::array<Point,3> &triangle) {
    bool nonnegative=true,nonpositive=true;
    for (std::size_t k=0;k<3;++k) {
        const Point a=triangle[k],b=triangle[(k+1)%3];
        const double cross=(b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);
        nonnegative=nonnegative && cross>=0;
        nonpositive=nonpositive && cross<=0;
    }
    return nonnegative || nonpositive;
}
double segmentTriangleDistance(Point a,Point b,const std::array<Point,3> &triangle) {
    if (pointInTriangle(a,triangle) || pointInTriangle(b,triangle)) return 0.;
    double minimum=std::numeric_limits<double>::infinity();
    for (std::size_t k=0;k<3;++k) {
        const Point c=triangle[k],d=triangle[(k+1)%3];
        const long double ux=b.x-a.x,uy=b.y-a.y,vx=d.x-c.x,vy=d.y-c.y;
        const long double denominator=ux*vy-uy*vx;
        if (denominator!=0) {
            const long double wx=c.x-a.x,wy=c.y-a.y;
            const long double t=(wx*vy-wy*vx)/denominator,s=(wx*uy-wy*ux)/denominator;
            if (t>=0 && t<=1 && s>=0 && s<=1) return 0.;
        }
        minimum=std::min({minimum,pointSegment(a,c,d),pointSegment(b,c,d),
                          pointSegment(c,a,b),pointSegment(d,a,b)});
    }
    return minimum;
}
double rectangleTriangleDistance(Rect rect,const std::array<Point,3> &triangle) {
    // Edge distances alone miss a rectangle wholly inside a triangle.
    for (Point corner:std::array<Point,4>{{{rect.xmin,rect.ymin},{rect.xmax,rect.ymin},
                                         {rect.xmax,rect.ymax},{rect.xmin,rect.ymax}}})
        if (pointInTriangle(corner,triangle)) return 0.;
    return std::min({segmentRectangleDistance(triangle[0],triangle[1],rect),
                     segmentRectangleDistance(triangle[1],triangle[2],rect),
                     segmentRectangleDistance(triangle[2],triangle[0],rect)});
}
bool sameTraversabilityConfig(const TraversabilityConfig &a,const TraversabilityConfig &b) {
    return a.lambda==b.lambda && a.slope_weight==b.slope_weight &&
        a.clearance_weight==b.clearance_weight && a.clearance_comfort_m==b.clearance_comfort_m &&
        a.feature_version==b.feature_version;
}
double readNumber(std::istream &in, bool allow_nodata=false) {
    std::string token;
    if (!(in>>token)) throw std::runtime_error("truncated terrain input");
    if (allow_nodata && token=="nodata") return std::numeric_limits<double>::quiet_NaN();
    std::size_t used=0;
    double value=std::stod(token,&used);
    if (used!=token.size() || !std::isfinite(value)) throw std::runtime_error("nonfinite/invalid number");
    return value;
}
void expect(std::istream &in, const char *word) {
    std::string token;
    if (!(in>>token) || token!=word) throw std::runtime_error(std::string("expected ")+word);
}
}

double distanceXY(Point a, Point b) { return std::hypot(a.x-b.x,a.y-b.y); }
double segmentRectangleDistance(Point a, Point b, Rect r) {
    if (intersects(a,b,r)) return 0.;
    return std::min({pointRect(a,r),pointRect(b,r),
        pointSegment({r.xmin,r.ymin},a,b), pointSegment({r.xmax,r.ymin},a,b),
        pointSegment({r.xmax,r.ymax},a,b), pointSegment({r.xmin,r.ymax},a,b)});
}

Terrain::Terrain(int nx,int ny,double x0,double y0,double dx,double dy,
                 std::vector<double> heights,std::vector<int> cell_mask,
                 std::vector<Rect> obstacles,double robot_radius,double margin,double slope_deg)
    : nx_(nx),ny_(ny),x0_(x0),y0_(y0),dx_(dx),dy_(dy),radius_(robot_radius+margin),slope_limit_deg_(slope_deg),
      heights_(std::move(heights)),forbidden_(std::move(cell_mask)),obstacles_(std::move(obstacles)) {
    const auto began=std::chrono::steady_clock::now();
    if (nx<2 || ny<2 || nx>100000 || ny>100000 ||
        heights_.size()!=static_cast<std::size_t>(nx)*ny ||
        forbidden_.size()!=static_cast<std::size_t>(nx-1)*(ny-1) ||
        !std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(dx) || !std::isfinite(dy) ||
        dx<=0 || dy<=0 || !std::isfinite(robot_radius) || !std::isfinite(margin) ||
        robot_radius<0 || margin<0 || !std::isfinite(radius_) ||
        !std::isfinite(slope_deg) || slope_deg<0 || slope_deg>=90)
        throw std::runtime_error("invalid terrain dimensions/model");
    auto box=bounds();
    if (!std::isfinite(box.xmax) || !std::isfinite(box.ymax)) throw std::runtime_error("extent overflow");
    complete_constant_height_=std::isfinite(heights_.front());
    for (double z:heights_) {
        if (std::isinf(z)) throw std::runtime_error("infinite height");
        complete_constant_height_=complete_constant_height_ &&
            std::isfinite(z) && z==heights_.front();
    }
    for (const auto &r:obstacles_) if (!finite({r.xmin,r.ymin}) || !finite({r.xmax,r.ymax}) ||
        r.xmin>=r.xmax || r.ymin>=r.ymax) throw std::runtime_error("invalid rectangle");
    const double limit=std::tan(slope_deg*std::acos(-1.)/180.);
    for (int j=0;j<ny-1;++j) for (int i=0;i<nx-1;++i) {
        int &mask=forbidden_[j*(nx-1)+i];
        if (mask!=0 && mask!=1) throw std::runtime_error("cell mask must be 0 or 1");
        if (!cellHasData(i,j)) { mask=1; continue; }
        // SW-SE-NE and SW-NE-NW, sharing the SW-NE diagonal.
        double ax=(h(i+1,j)-h(i,j))/dx, by=(h(i+1,j+1)-h(i+1,j))/dy;
        double bx=(h(i+1,j+1)-h(i,j+1))/dx, ay=(h(i,j+1)-h(i,j))/dy;
        if (std::hypot(ax,by)>limit || std::hypot(bx,ay)>limit) mask=1;
    }
    preprocessing_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
}

Terrain Terrain::load(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("cannot open terrain: "+filename);
    expect(in,"T25D"); expect(in,"1"); expect(in,"units"); expect(in,"m");
    expect(in,"diagonal"); expect(in,"SW_NE"); expect(in,"grid");
    int nx=0,ny=0;
    if (!(in>>nx>>ny) || nx<2 || ny<2 || static_cast<long long>(nx)*ny>100000000)
        throw std::runtime_error("invalid grid dimensions");
    double x0=readNumber(in),y0=readNumber(in),dx=readNumber(in),dy=readNumber(in);
    expect(in,"model");
    double radius=readNumber(in),margin=readNumber(in),slope=readNumber(in);
    expect(in,"heights");
    std::vector<double> heights(static_cast<std::size_t>(nx)*ny);
    for (auto &z:heights) z=readNumber(in,true);
    expect(in,"mask");
    std::vector<int> mask(static_cast<std::size_t>(nx-1)*(ny-1));
    for (auto &m:mask) { if (!(in>>m) || (m!=0 && m!=1)) throw std::runtime_error("invalid mask"); }
    expect(in,"rectangles");
    int n=0;
    if (!(in>>n) || n<0 || n>1000000) throw std::runtime_error("invalid rectangle count");
    std::vector<Rect> rects;
    for (int k=0;k<n;++k) rects.push_back({readNumber(in),readNumber(in),readNumber(in),readNumber(in)});
    expect(in,"end");
    std::string extra;
    if (in>>extra) throw std::runtime_error("unexpected trailing data");
    return Terrain(nx,ny,x0,y0,dx,dy,std::move(heights),std::move(mask),std::move(rects),radius,margin,slope);
}

bool Terrain::cellHasData(int i,int j) const {
    return std::isfinite(h(i,j)) && std::isfinite(h(i+1,j)) &&
           std::isfinite(h(i,j+1)) && std::isfinite(h(i+1,j+1));
}
bool Terrain::inBounds(Point q) const {
    auto b=bounds();
    return finite(q) && q.x>=b.xmin && q.x<=b.xmax && q.y>=b.ymin && q.y<=b.ymax;
}
bool Terrain::height(Point q,double &z) const {
    ++counters_.height_queries;
    if (!inBounds(q)) return false;
    double gx=(q.x-x0_)/dx_, gy=(q.y-y0_)/dy_;
    int i=std::clamp(static_cast<int>(std::floor(gx)),0,nx_-2);
    int j=std::clamp(static_cast<int>(std::floor(gy)),0,ny_-2);
    if (!cellHasData(i,j)) return false;
    double u=gx-i,v=gy-j;
    if (v<=u) z=h(i,j)+(h(i+1,j)-h(i,j))*u+(h(i+1,j+1)-h(i+1,j))*v;
    else z=h(i,j)+(h(i+1,j+1)-h(i,j+1))*u+(h(i,j+1)-h(i,j))*v;
    return std::isfinite(z);
}
void Terrain::enableCollisionIndex() {
    if (collision_index_enabled_) return;
    const auto began=std::chrono::steady_clock::now();
    // Clear first so retrying after an allocation failure cannot duplicate entries.
    collision_rectangles_=obstacles_;
    collision_nodes_.clear();
    for (int j=0;j<ny_-1;++j) for (int i=0;i<nx_-1;++i)
        if (forbidden_[j*(nx_-1)+i])
            collision_rectangles_.push_back({x0_+i*dx_,y0_+j*dy_,x0_+(i+1)*dx_,y0_+(j+1)*dy_});
    if (!collision_rectangles_.empty()) buildCollisionNode(0,collision_rectangles_.size());
    collision_index_build_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    collision_index_enabled_=true;
}
int Terrain::buildCollisionNode(std::size_t begin,std::size_t end) {
    Rect box=collision_rectangles_[begin];
    for (std::size_t k=begin+1;k<end;++k) {
        const auto &r=collision_rectangles_[k];
        box.xmin=std::min(box.xmin,r.xmin); box.ymin=std::min(box.ymin,r.ymin);
        box.xmax=std::max(box.xmax,r.xmax); box.ymax=std::max(box.ymax,r.ymax);
    }
    const int index=static_cast<int>(collision_nodes_.size());
    collision_nodes_.push_back({box,begin,end,-1,-1});
    if (end-begin>8) {
        const bool split_x=box.xmax-box.xmin>=box.ymax-box.ymin;
        const auto middle=begin+(end-begin)/2;
        std::nth_element(collision_rectangles_.begin()+begin,collision_rectangles_.begin()+middle,
            collision_rectangles_.begin()+end,[split_x](Rect a,Rect b) {
                return split_x ? .5*a.xmin+.5*a.xmax<.5*b.xmin+.5*b.xmax
                               : .5*a.ymin+.5*a.ymax<.5*b.ymin+.5*b.ymax;
            });
        // Recursion can reallocate the vector: write by index after both calls.
        const int left=buildCollisionNode(begin,middle), right=buildCollisionNode(middle,end);
        collision_nodes_[index].left=left; collision_nodes_[index].right=right;
    }
    return index;
}
bool Terrain::indexedCapsuleClear(int index,Point a,Point b,Rect query,double r) const {
    ++counters_.bvh_node_visits;
    const auto &node=collision_nodes_[index];
    if (!overlaps(node.box,query)) return true;
    if (node.left>=0)
        return indexedCapsuleClear(node.left,a,b,query,r) && indexedCapsuleClear(node.right,a,b,query,r);
    for (std::size_t k=node.begin;k<node.end;++k) {
        const auto &rect=collision_rectangles_[k];
        if (!overlaps(rect,query)) continue;
        ++counters_.collision_candidates;
        ++counters_.collision_narrow_checks;
        if (segmentRectangleDistance(a,b,rect)<=r) return false;
    }
    return true;
}
bool Terrain::capsuleClear(Point a,Point b) const {
    if (!inBounds(a) || !inBounds(b)) return false;
    auto box=bounds();
    // Roundoff padding only makes collision tests more conservative (never shrinks F).
    const double pad=32*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(box.xmin),
        std::abs(box.xmax),std::abs(box.ymin),std::abs(box.ymax),radius_});
    const double r=radius_+pad;
    if (std::min(a.x,b.x)-r<=box.xmin || std::max(a.x,b.x)+r>=box.xmax ||
        std::min(a.y,b.y)-r<=box.ymin || std::max(a.y,b.y)+r>=box.ymax) return false;
    if (collision_index_enabled_) {
        if (collision_nodes_.empty()) return true;
        // Outward rounding protects the broad phase from discarding a rectangle
        // that the unchanged padded, exact capsule predicate would reject.
        const double infinity=std::numeric_limits<double>::infinity();
        Rect query{std::nextafter(std::min(a.x,b.x)-r,-infinity),
                   std::nextafter(std::min(a.y,b.y)-r,-infinity),
                   std::nextafter(std::max(a.x,b.x)+r,infinity),
                   std::nextafter(std::max(a.y,b.y)+r,infinity)};
        return indexedCapsuleClear(0,a,b,query,r);
    }
    int imin=std::max(0,static_cast<int>(std::floor((std::min(a.x,b.x)-r-x0_)/dx_)));
    int imax=std::min(nx_-2,static_cast<int>(std::floor((std::max(a.x,b.x)+r-x0_)/dx_)));
    int jmin=std::max(0,static_cast<int>(std::floor((std::min(a.y,b.y)-r-y0_)/dy_)));
    int jmax=std::min(ny_-2,static_cast<int>(std::floor((std::max(a.y,b.y)+r-y0_)/dy_)));
    for (int j=jmin;j<=jmax;++j) for (int i=imin;i<=imax;++i)
        if (forbidden_[j*(nx_-1)+i]) {
            ++counters_.collision_candidates;
            ++counters_.collision_narrow_checks;
            if (segmentRectangleDistance(a,b,
                {x0_+i*dx_,y0_+j*dy_,x0_+(i+1)*dx_,y0_+(j+1)*dy_})<=r) return false;
        }
    for (const auto &rect:obstacles_) {
        ++counters_.collision_candidates;
        ++counters_.collision_narrow_checks;
        if (segmentRectangleDistance(a,b,rect)<=r) return false;
    }
    return true;
}
bool Terrain::validState(Point q) const { ++counters_.state_validity_calls; return capsuleClear(q,q); }
bool Terrain::validMotion(Point a,Point b) const { ++counters_.motion_validity_calls; return capsuleClear(a,b); }

double Terrain::maximumSweptSlope(Point a,Point b) const {
    const double infinity=std::numeric_limits<double>::infinity();
    if (!inBounds(a) || !inBounds(b)) return infinity;
    const auto box=bounds();
    if (std::min(a.x,b.x)-radius_<box.xmin || std::max(a.x,b.x)+radius_>box.xmax ||
        std::min(a.y,b.y)-radius_<box.ymin || std::max(a.y,b.y)+radius_>box.ymax) return infinity;
    // One extra index includes a closed triangle touching the lower search edge.
    const int jmin=std::max(0,static_cast<int>(std::floor((std::min(a.y,b.y)-radius_-y0_)/dy_))-1);
    const int jmax=std::min(ny_-2,static_cast<int>(std::floor((std::max(a.y,b.y)+radius_-y0_)/dy_))+1);
    double maximum=0.;
    bool touched=false;
    // Clip against each row expanded by the disk radius, then expand its X
    // interval. This is a conservative broad phase of the capsule, avoiding a
    // scan of every cell in a long diagonal edge's bounding rectangle.
    for (int j=jmin;j<=jmax;++j) {
        long double begin=0,end=1;
        const long double delta=static_cast<long double>(b.y)-a.y;
        // Use the same canonical coordinates as the represented triangles.
        // Outward rounding only affects candidate selection, never the final
        // segment/triangle distance or the declared footprint radius.
        const double rowLow=y0_+j*dy_,rowHigh=y0_+(j+1)*dy_;
        const long double low=std::nextafter(rowLow-radius_,-infinity);
        const long double high=std::nextafter(rowHigh+radius_,infinity);
        if (delta==0) { if (a.y<low || a.y>high) continue; }
        else {
            long double u=(low-a.y)/delta,v=(high-a.y)/delta;
            if (u>v) std::swap(u,v);
            begin=std::max(begin,u);end=std::min(end,v);
            if (begin>end) continue;
        }
        const long double x1=a.x+begin*(static_cast<long double>(b.x)-a.x);
        const long double x2=a.x+end*(static_cast<long double>(b.x)-a.x);
        const int imin=std::max(0,static_cast<int>(std::floor((std::min(x1,x2)-radius_-x0_)/dx_))-1);
        const int imax=std::min(nx_-2,static_cast<int>(std::floor((std::max(x1,x2)+radius_-x0_)/dx_))+1);
        for (int i=imin;i<=imax;++i) {
        const Point sw{x0_+i*dx_,y0_+j*dy_},se{x0_+(i+1)*dx_,sw.y},
            ne{se.x,y0_+(j+1)*dy_},nw{sw.x,ne.y};
        const std::array<std::array<Point,3>,2> triangles{{{{sw,se,ne}},{{sw,ne,nw}}}};
        for (std::size_t k=0;k<2;++k) {
            if (segmentTriangleDistance(a,b,triangles[k])>radius_) continue;
            touched=true;
            if (!cellHasData(i,j)) return infinity;
            const double gx=k==0?(h(i+1,j)-h(i,j))/dx_:(h(i+1,j+1)-h(i,j+1))/dx_;
            const double gy=k==0?(h(i+1,j+1)-h(i+1,j))/dy_:(h(i,j+1)-h(i,j))/dy_;
            const double slope=std::atan(std::hypot(gx,gy))*180./std::acos(-1.);
            if (!std::isfinite(slope)) return infinity;
            maximum=std::max(maximum,slope);
        }
        }
    }
    return touched?maximum:infinity;
}

bool Terrain::surfaceCuts(Point a,Point b,std::vector<double> &cuts,std::string &reason) const {
    if (!inBounds(a) || !inBounds(b)) { reason="outside_bounds"; return false; }
    std::vector<double> grid{0.,1.};
    auto addCrossings=[&](double p,double q,double origin,double spacing,int n) {
        if (p==q) return;
        int first=std::max(1,static_cast<int>(std::ceil((std::min(p,q)-origin)/spacing)));
        int last=std::min(n-2,static_cast<int>(std::floor((std::max(p,q)-origin)/spacing)));
        for (int i=first;i<=last;++i) {
            double t=(origin+i*spacing-p)/(q-p);
            if (t>0 && t<1) grid.push_back(t);
        }
    };
    addCrossings(a.x,b.x,x0_,dx_,nx_); addCrossings(a.y,b.y,y0_,dy_,ny_);
    std::sort(grid.begin(),grid.end());
    grid.erase(std::unique(grid.begin(),grid.end()),grid.end());
    cuts=grid;
    const double du=(b.x-a.x)/dx_, dv=(b.y-a.y)/dy_;
    for (std::size_t k=1;k<grid.size();++k) {
        Point mid=mix(a,b,(grid[k-1]+grid[k])/2);
        int i=std::clamp(static_cast<int>(std::floor((mid.x-x0_)/dx_)),0,nx_-2);
        int j=std::clamp(static_cast<int>(std::floor((mid.y-y0_)/dy_)),0,ny_-2);
        if (!cellHasData(i,j)) { reason="nodata"; return false; }
        if (du!=dv) {
            double t=((a.y-y0_)/dy_-(a.x-x0_)/dx_+i-j)/(du-dv);
            if (t>grid[k-1] && t<grid[k]) cuts.push_back(t);
        }
    }
    std::sort(cuts.begin(),cuts.end());
    cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
    return true;
}
SurfaceEdge Terrain::surfaceEdge(Point a,Point b,std::vector<double> *cut_parameters) const {
    ++counters_.full_surface_cost_calls;
    SurfaceEdge out;
    if(cut_parameters)cut_parameters->clear();
    std::vector<double> cuts;
    if (!surfaceCuts(a,b,cuts,out.reason)) return out;
    out.cost=0;
    for (double t:cuts) {
        Point p=mix(a,b,t); double z=0;
        if (!height(p,z)) { out.has_data=false; out.cost=std::numeric_limits<double>::infinity();
            out.points.clear(); out.reason="nodata"; return out; }
        if (!out.points.empty()) {
            auto prev=out.points.back();
            out.cost+=std::hypot(distanceXY(p,{prev.x,prev.y}),z-prev.z);
            ++counters_.triangle_visits;
        }
        out.points.push_back({p.x,p.y,z});
    }
    out.has_data=true; out.reason="ok";
    if(cut_parameters)*cut_parameters=std::move(cuts);
    return out;
}
double Terrain::surfaceCost(Point a,Point b) const {
    ++counters_.full_surface_cost_calls;
    std::vector<double> cuts;
    std::string reason;
    if (!surfaceCuts(a,b,cuts,reason)) return std::numeric_limits<double>::infinity();
    double cost=0;
    Point3 previous{};
    bool has_previous=false;
    for (double t:cuts) {
        Point p=mix(a,b,t); double z=0;
        if (!height(p,z)) return std::numeric_limits<double>::infinity();
        if (has_previous) {
            cost+=std::hypot(distanceXY(p,{previous.x,previous.y}),z-previous.z);
            ++counters_.triangle_visits;
        }
        previous={p.x,p.y,z}; has_previous=true;
    }
    return cost;
}
void Terrain::enableTraversability(const TraversabilityConfig &config,TraversabilityBuildPolicy policy) {
    if (policy!=TraversabilityBuildPolicy::Legacy && policy!=TraversabilityBuildPolicy::SkipDominated)
        throw std::invalid_argument("invalid traversability build policy");
    if (!std::isfinite(config.lambda) || config.lambda<0 ||
        !std::isfinite(config.slope_weight) || config.slope_weight<0 ||
        !std::isfinite(config.clearance_weight) || config.clearance_weight<0 ||
        config.slope_weight+config.clearance_weight!=1. ||
        !std::isfinite(config.clearance_comfort_m) || config.clearance_comfort_m<=0 ||
        slope_limit_deg_<=0 || config.feature_version!="cell_envelope_v1")
        throw std::invalid_argument("invalid traversability configuration");
    if (traversability_enabled_) {
        if (!sameTraversabilityConfig(config,traversability_config_) || policy!=traversability_build_policy_)
            throw std::logic_error("traversability configuration and build policy are immutable");
        return;
    }
    const auto began=std::chrono::steady_clock::now();
    const bool skip_dominated=policy==TraversabilityBuildPolicy::SkipDominated;
    TraversabilityBuildStats stats;
    const std::size_t count=static_cast<std::size_t>(nx_-1)*(ny_-1);
    std::vector<TraversabilityCell> cells(count);
    std::vector<std::array<double,2>> slopes(count);
    std::vector<double> xs,ys;
    if (skip_dominated) {
        xs.resize(nx_);ys.resize(ny_);
        // Preserve the canonical arithmetic used by the legacy builder; do not
        // replace it with an accumulated coordinate or a local-coordinate stencil.
        for (int i=0;i<nx_;++i) xs[i]=x0_+i*dx_;
        for (int j=0;j<ny_;++j) ys[j]=y0_+j*dy_;
    }
    for (int j=0;j<ny_-1;++j) for (int i=0;i<nx_-1;++i) {
        const auto index=static_cast<std::size_t>(j)*(nx_-1)+i;
        cells[index].available=cellHasData(i,j);
        if (!cells[index].available) continue;
        ++stats.available_cells;
        slopes[index]={{std::atan(std::hypot((h(i+1,j)-h(i,j))/dx_,
                                            (h(i+1,j+1)-h(i+1,j))/dy_)),
                        std::atan(std::hypot((h(i+1,j+1)-h(i,j+1))/dx_,
                                            (h(i,j+1)-h(i,j))/dy_))}};
    }
    auto phase=std::chrono::steady_clock::now();
    stats.slope_extraction_seconds=std::chrono::duration<double>(phase-began).count();
    const auto map_box=bounds();
    const double pad=32*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(map_box.xmin),
        std::abs(map_box.xmax),std::abs(map_box.ymin),std::abs(map_box.ymax),radius_});
    const double slope_limit=slope_limit_deg_*std::acos(-1.)/180.;
    // Clamp in floating point before conversion; even a huge finite footprint
    // must not overflow an integer candidate-cell index.
    auto cellIndex=[](double coordinate,double origin,double spacing,int last) {
        return static_cast<int>(std::clamp(std::floor((coordinate-origin)/spacing),0.,double(last)));
    };
    for (int j=0;j<ny_-1;++j) for (int i=0;i<nx_-1;++i) {
        auto &feature=cells[static_cast<std::size_t>(j)*(nx_-1)+i];
        if (!feature.available) continue;
        const Rect cell=skip_dominated ? Rect{xs[i],ys[j],xs[i+1],ys[j+1]} :
            Rect{x0_+i*dx_,y0_+j*dy_,x0_+(i+1)*dx_,y0_+(j+1)*dy_};
        const auto index=static_cast<std::size_t>(j)*(nx_-1)+i;
        // Both own triangles intersect this entire cell. Their known distance
        // zero permits the seed without any geometric membership approximation.
        double maximum_slope=skip_dominated ? std::max(slopes[index][0],slopes[index][1]) : 0.;
        const int imin=cellIndex(cell.xmin-radius_-pad,x0_,dx_,nx_-2);
        const int imax=cellIndex(cell.xmax+radius_+pad,x0_,dx_,nx_-2);
        const int jmin=cellIndex(cell.ymin-radius_-pad,y0_,dy_,ny_-2);
        const int jmax=cellIndex(cell.ymax+radius_+pad,y0_,dy_,ny_-2);
        for (int tj=jmin;tj<=jmax && !(skip_dominated && maximum_slope>=slope_limit);++tj)
            for (int ti=imin;ti<=imax && !(skip_dominated && maximum_slope>=slope_limit);++ti) {
            ++stats.candidate_cells;
            const auto other=static_cast<std::size_t>(tj)*(nx_-1)+ti;
            // A triangle in a cell with ANY missing corner is unavailable,
            // matching the existing whole-cell NoData model.
            if (!cells[other].available) continue;
            stats.candidate_triangles+=2;
            if (skip_dominated && slopes[other][0]<=maximum_slope && slopes[other][1]<=maximum_slope) {
                ++stats.dominated_cell_skips;stats.dominated_triangle_skips+=2;continue;
            }
            const Point sw=skip_dominated ? Point{xs[ti],ys[tj]} : Point{x0_+ti*dx_,y0_+tj*dy_};
            const Point se{skip_dominated ? xs[ti+1] : x0_+(ti+1)*dx_,sw.y};
            const Point ne{se.x,skip_dominated ? ys[tj+1] : y0_+(tj+1)*dy_},nw{sw.x,ne.y};
            const std::array<std::array<Point,3>,2> triangles{{{{sw,se,ne}},{{sw,ne,nw}}}};
            for (std::size_t k=0;k<2;++k) {
                if (skip_dominated && maximum_slope>=slope_limit) break;
                if (skip_dominated && slopes[other][k]<=maximum_slope) {
                    ++stats.dominated_triangle_skips;continue;
                }
                ++stats.slope_distance_calls;
                if (rectangleTriangleDistance(cell,triangles[k])<=radius_+pad)
                    maximum_slope=std::max(maximum_slope,slopes[other][k]);
            }
        }
        const double normalized_slope=std::min(1.,maximum_slope/slope_limit);
        feature.slope_penalty=normalized_slope*normalized_slope;
        if (skip_dominated && maximum_slope>=slope_limit) ++stats.slope_saturated_cells;
    }
    const auto envelope_end=std::chrono::steady_clock::now();
    stats.slope_envelope_seconds=std::chrono::duration<double>(envelope_end-phase).count();
    phase=envelope_end;
    for (int j=0;j<ny_-1;++j) for (int i=0;i<nx_-1;++i) {
        auto &feature=cells[static_cast<std::size_t>(j)*(nx_-1)+i];
        if (!feature.available) continue;
        const Rect cell=skip_dominated ? Rect{xs[i],ys[j],xs[i+1],ys[j+1]} :
            Rect{x0_+i*dx_,y0_+j*dy_,x0_+(i+1)*dx_,y0_+(j+1)*dy_};
        // Generic masks, NoData, excessive slope and map boundaries remain hard
        // constraints, but are not silently relabelled as physical obstacles.
        double minimum_distance=std::numeric_limits<double>::infinity();
        for (const Rect &obstacle:obstacles_) {
            ++stats.obstacle_distance_calls;
            minimum_distance=std::min(minimum_distance,rectangleDistance(cell,obstacle));
            // Evaluate exactly the legacy expression. A rearranged threshold
            // can differ at a floating-point boundary and is not equivalent.
            if (skip_dominated && std::max(0.,minimum_distance-radius_-pad)==0.) {
                ++stats.clearance_saturated_cells;break;
            }
        }
        if (!obstacles_.empty()) {
            const double clearance=std::max(0.,minimum_distance-radius_-pad);
            const double normalized_penalty=std::max(0.,1.-clearance/config.clearance_comfort_m);
            feature.clearance_penalty=normalized_penalty*normalized_penalty;
        }
    }
    stats.clearance_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase).count();
    // Commit only after all allocations and preprocessing succeed.
    traversability_cells_=std::move(cells);
    traversability_config_=config;
    traversability_build_policy_=policy;
    traversability_preprocessing_seconds_=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-began).count();
    stats.total_seconds=traversability_preprocessing_seconds_;
    traversability_build_stats_=stats;
    traversability_enabled_=true;
}
const TraversabilityConfig &Terrain::traversabilityConfig() const {
    if (!traversability_enabled_) throw std::logic_error("traversability is not enabled");
    return traversability_config_;
}
TraversabilityCell Terrain::featureCell(int i,int j) const {
    if (!traversability_enabled_) throw std::logic_error("traversability is not enabled");
    if (i<0 || i>=nx_-1 || j<0 || j>=ny_-1) throw std::out_of_range("feature cell outside map");
    return traversability_cells_[static_cast<std::size_t>(j)*(nx_-1)+i];
}
TraversabilityCell Terrain::intervalFeature(Point a,Point b,Point midpoint) const {
    auto incident=[](double value,double origin,double spacing,int nodes,bool constant) {
        const double coordinate=(value-origin)/spacing;
        const int containing=std::clamp(static_cast<int>(std::floor(coordinate)),0,nodes-2);
        std::array<int,2> result{{containing,containing}};
        if (constant) {
            const double grid=std::round(coordinate);
            const double grid_value=origin+grid*spacing;
            // A boundary is the represented grid coordinate, not an epsilon-wide
            // strip. Comparing the quotient to an integer misses canonical
            // values such as .2*3 because division need not invert multiplication.
            if (grid>0 && grid<nodes-1 && value==grid_value)
                result={{static_cast<int>(grid)-1,static_cast<int>(grid)}};
        }
        return result;
    };
    const auto columns=incident(midpoint.x,x0_,dx_,nx_,a.x==b.x);
    const auto rows=incident(midpoint.y,y0_,dy_,ny_,a.y==b.y);
    TraversabilityCell result;
    for (int j:rows) for (int i:columns) {
        const auto &feature=traversability_cells_[static_cast<std::size_t>(j)*(nx_-1)+i];
        if (!feature.available) continue;
        result.available=true;
        result.slope_penalty=std::max(result.slope_penalty,feature.slope_penalty);
        result.clearance_penalty=std::max(result.clearance_penalty,feature.clearance_penalty);
    }
    return result;
}
EdgeCostBreakdown Terrain::edgeCostBreakdown(Point a,Point b,std::vector<Point3> *surface_points) const {
    ++counters_.full_surface_cost_calls;
    if (surface_points) surface_points->clear();
    std::vector<double> cuts;
    std::string reason;
    if (!surfaceCuts(a,b,cuts,reason)) return {};
    EdgeCostBreakdown out;
    out.cost_terrain_m=out.risk_slope_integral_m=out.risk_clearance_integral_m=0.;
    Point3 previous{};
    bool has_previous=false;
    double previous_t=0;
    for (double t:cuts) {
        Point p=mix(a,b,t); double z=0;
        if (!height(p,z)) { if (surface_points) surface_points->clear(); return {}; }
        if (has_previous) {
            const double length=std::hypot(distanceXY(p,{previous.x,previous.y}),z-previous.z);
            out.cost_terrain_m+=length;
            ++counters_.triangle_visits;
            if (traversability_enabled_) {
                const auto feature=intervalFeature(a,b,mix(a,b,(previous_t+t)/2));
                if (!feature.available) { if (surface_points) surface_points->clear(); return {}; }
                out.risk_slope_integral_m+=length*feature.slope_penalty;
                out.risk_clearance_integral_m+=length*feature.clearance_penalty;
            }
        }
        previous={p.x,p.y,z}; has_previous=true; previous_t=t;
        if (surface_points) surface_points->push_back(previous);
    }
    out.risk_integral_m=traversability_enabled_
        ? traversability_config_.slope_weight*out.risk_slope_integral_m+
          traversability_config_.clearance_weight*out.risk_clearance_integral_m : 0.;
    out.objective_value=out.cost_terrain_m+
        (traversability_enabled_ ? traversability_config_.lambda*out.risk_integral_m : 0.);
    out.has_data=true;
    return out;
}
EdgeEvaluation Terrain::evaluate(Point a,Point b) const {
    bool valid=validMotion(a,b); auto edge=surfaceEdge(a,b);
    return {valid && edge.has_data, edge.cost,edge.points.empty()?0:edge.points.size()-1,
        !edge.has_data?edge.reason:valid?"ok":"forbidden_capsule"};
}
double Terrain::lowerBound(Point a,Point b,bool lifted) const {
    if (!finite(a) || !finite(b)) return std::numeric_limits<double>::infinity();
    if (!lifted) return distanceXY(a,b);
    double za=0,zb=0;
    if (!height(a,za) || !height(b,zb)) return std::numeric_limits<double>::infinity();
    return std::hypot(distanceXY(a,b),za-zb);
}
double Terrain::solutionLowerBound(Point start,Point q,Point goal,bool lifted) const {
    return lowerBound(start,q,lifted)+lowerBound(q,goal,lifted);
}
}
