// SPDX-License-Identifier: GPL-3.0-only
// Controlled port of PUTN_classes.cpp and the global PFRRTStar search methods.
// Copyright original PUTN authors. Adaptations and source hashes: external/putn/.
#include "terrain25d/PutnStatic.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace terrain25d {
namespace {
using V3=Eigen::Vector3d;
using Clock=std::chrono::steady_clock;
constexpr double resolution=.1, cloud_spacing=.2, fit_radius=.5, com_height=.7;
constexpr double step_size=.4, neighbor_radius=1., goal_threshold=1., goal_bias=.15;
constexpr double pi=3.1415926535897932384626433832795;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
V3 vec(Point3 p){return {p.x,p.y,p.z};}
Point3 point(V3 p){return {p.x(),p.y(),p.z()};}
struct Index {int x,y,z;bool operator==(const Index &o)const{return x==o.x&&y==o.y&&z==o.z;}};
struct IndexHash {std::size_t operator()(Index i)const {
    return (static_cast<std::size_t>(i.x)*73856093u)^(static_cast<std::size_t>(i.y)*19349663u)^(static_cast<std::size_t>(i.z)*83492791u);}};
}
struct PutnStatic::Impl {
    const Terrain &terrain;
    Rect bounds;
    V3 lower,upper;
    Eigen::Vector3i dimensions;
    std::unordered_set<Index,IndexHash> occupied;
    PutnStats stats;
    PutnSnapshot last;
    bool paper=false;
    double slope_weight=.1,stop_eps=0;  // tau slope coefficient per degree; convergence-stop epsilon
    std::uint64_t stop_window=0;
    PutnConvergence convergence;
    struct Node {PutnPlane plane;std::size_t parent=0;double cost=0,edge=0;std::vector<std::size_t> children;};
    std::vector<Node> tree;
    explicit Impl(const Terrain &t,const std::vector<Rect> &obstacles):terrain(t),bounds(t.bounds()) {
        const auto began=Clock::now();
        std::vector<V3> cloud;
        const int nx=static_cast<int>(std::floor((bounds.xmax-bounds.xmin)/cloud_spacing+.5));
        const int ny=static_cast<int>(std::floor((bounds.ymax-bounds.ymin)/cloud_spacing+.5));
        double zmin=std::numeric_limits<double>::infinity(),zmax=-zmin;
        for(int j=0;j<=ny;++j)for(int i=0;i<=nx;++i){
            Point q{bounds.xmin+i*cloud_spacing,bounds.ymin+j*cloud_spacing};double z;
            if(terrain.height(q,z)){cloud.emplace_back(q.x,q.y,z);zmin=std::min(zmin,z);zmax=std::max(zmax,z);}
        }
        if(cloud.empty())throw std::invalid_argument("PUTN ground cloud has no data");
        lower={bounds.xmin,bounds.ymin,zmin};upper={bounds.xmax,bounds.ymax,zmax+3.};
        dimensions=((upper-lower)/resolution).cast<int>()+Eigen::Vector3i::Ones();
        // Static representation: declared rectangles extruded 2 m above H.
        // Filling their voxel volume prevents treating their top as traversable support.
        for(const auto &r:obstacles){
            int xlo=std::max(0,static_cast<int>(std::floor((r.xmin-lower.x())/resolution)));
            int xhi=std::min(dimensions.x()-1,static_cast<int>(std::floor((r.xmax-lower.x())/resolution)));
            int ylo=std::max(0,static_cast<int>(std::floor((r.ymin-lower.y())/resolution)));
            int yhi=std::min(dimensions.y()-1,static_cast<int>(std::floor((r.ymax-lower.y())/resolution)));
            for(int i=xlo;i<=xhi;++i)for(int j=ylo;j<=yhi;++j){
                V3 p=center({i,j,0});double z;
                Point q{std::clamp(p.x(),bounds.xmin,bounds.xmax),std::clamp(p.y(),bounds.ymin,bounds.ymax)};
                if(!terrain.height(q,z))continue;
                int lo=std::max(0,index({p.x(),p.y(),z}).z), hi=std::min(dimensions.z()-1,index({p.x(),p.y(),z+2.}).z);
                for(int k=lo;k<=hi;++k)cloud.push_back(center({i,j,k}));
            }
        }
        stats.conversion_seconds=elapsed(began);
        const auto model_began=Clock::now();
        for(const auto &p:cloud)setOccupied(p);
        stats.preprocessing_seconds=elapsed(model_began);
    }
    Impl(const Terrain &t,const std::vector<Point3> &cloud,bool enabled,double cap):terrain(t),bounds(t.bounds()),paper(enabled) {
        if(!paper||cloud.empty())throw std::invalid_argument("paper PUTN requires a nonempty master cloud");
        if(!std::isfinite(cap)||cap<=0||cap>90)throw std::invalid_argument("PUTN tau slope cap must be in (0, 90] degrees");
        // IEEE division is correctly rounded, so the source cap of 10 degrees gives exactly the source weight.
        static_assert(1./10.==.1,"source slope weight must be reproduced exactly");
        slope_weight=1./cap;
        auto began=Clock::now();double zmin=std::numeric_limits<double>::infinity(),zmax=-zmin;
        for(auto p:cloud){if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))
                throw std::invalid_argument("nonfinite master cloud");
            if(p.x<bounds.xmin||p.x>bounds.xmax||p.y<bounds.ymin||p.y>bounds.ymax)
                throw std::invalid_argument("master cloud outside terrain bounds");
            zmin=std::min(zmin,p.z);zmax=std::max(zmax,p.z);}
        lower={bounds.xmin,bounds.ymin,zmin};upper={bounds.xmax,bounds.ymax,zmax+3.};
        const auto span=(upper-lower)/resolution;
        if(span.maxCoeff()>static_cast<double>(std::numeric_limits<int>::max()-1))
            throw std::invalid_argument("master cloud voxel dimensions overflow");
        dimensions=span.cast<int>()+Eigen::Vector3i::Ones();
        stats.conversion_seconds=elapsed(began);began=Clock::now();
        for(auto p:cloud)setOccupied(vec(p));
        stats.preprocessing_seconds=elapsed(began);
    }
    Index index(const V3 &p)const {auto v=((p-lower)/resolution).array().floor().cast<int>();return {v.x(),v.y(),v.z()};}
    bool inside(Index i)const{return i.x>=0&&i.y>=0&&i.z>=0&&i.x<dimensions.x()&&i.y<dimensions.y()&&i.z<dimensions.z();}
    V3 center(Index i)const{return lower+resolution*(V3(i.x,i.y,i.z)+V3::Constant(.5));}
    void setOccupied(const V3 &p){auto i=index(p);if(inside(i))occupied.insert(i);}
    bool free(const V3 &p)const {auto i=index(p);return inside(i)&&occupied.find(i)==occupied.end();}
    PutnPlane fit(Point q){
        ++stats.plane_fits;PutnPlane result;result.support=q;double z;
        if(!terrain.validState(q)||!terrain.height(q,z)){++stats.common_rejections;return result;}
        // Unlike sparse voxel-column projection, exact H preserves singleton XY.
        V3 surface(q.x,q.y,z),ball=center(index(surface));
        const int num=static_cast<int>(fit_radius/resolution),side=2*num+1;
        std::vector<V3> points;std::vector<std::pair<int,int>> vacant;
        for(int i=-num;i<=num;++i)for(int j=-num;j<=num;++j){
            bool found=false;
            const int vertical=paper?num:3;
            for(int k=-vertical;k<=vertical;++k){V3 p=ball+resolution*V3(i,j,k);auto ix=index(p);
                if(inside(ix)&&occupied.find(ix)!=occupied.end()){points.push_back(p);found=true;}}
            if(!found)vacant.emplace_back(i+num,j+num);
        }
        result.point_count=points.size();
        if(points.size()<3){++stats.model_rejections;return result;}
        V3 center_point=V3::Zero();for(const auto &p:points)center_point+=p;center_point/=points.size();
        Eigen::MatrixXd A(points.size(),3);
        for(std::size_t k=0;k<points.size();++k)A.row(k)=(points[k]-center_point).transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullV);
        if(svd.singularValues()(1)<=std::numeric_limits<double>::epsilon()){
            ++stats.model_rejections;return result;
        }
        V3 normal=svd.matrixV().col(2);if(normal.z()<0)normal=-normal;
        normal.normalize();
        double flatness=0;for(std::size_t k=0;k<points.size();++k){double d=normal.dot(A.row(k));flatness+=d*d*d*d;}
        flatness/=(paper?static_cast<double>(points.size()):1.+points.size());
        double slope=std::acos(std::clamp(normal.z(),-1.,1.))*180./pi,sparsity=0;
        if(!vacant.empty()){
            // Algebraic reduction of source M_vac centering and covariance trace.
            // Source centers each (i,j) column, not each row; preserve that policy.
            double ratio=static_cast<double>(vacant.size())/(side*side);
            if(paper){
                Eigen::Vector2d mean=Eigen::Vector2d::Zero();
                for(auto v:vacant)mean+=Eigen::Vector2d(v.first,v.second);mean/=vacant.size();
                Eigen::Matrix2d covariance=Eigen::Matrix2d::Zero();
                for(auto v:vacant){Eigen::Vector2d d=Eigen::Vector2d(v.first,v.second)-mean;covariance+=d*d.transpose();}
                covariance/=vacant.size();double trace=(covariance.transpose()*covariance).trace();
                if(ratio>.05)sparsity=1;
                else if(ratio>=.02&&ratio<=.05&&trace<1./.1152)sparsity=(ratio-.02)/(.05-.02);
            }else{
                double sumsq=0,first=0;
                for(std::size_t k=0;k<vacant.size();++k){double a=.5*(vacant[k].first-vacant[k].second);
                    if(k==0)first=a;sumsq+=a*a;}
                double trace=first*first*sumsq;
                if(ratio>.05)sparsity=1;
                else if(ratio>.02&&ratio<.05&&(trace==0||1./trace>.1152))sparsity=(ratio-.02)/(.05-.02);
            }
        }
        result.available=true;result.normal=point(normal);result.center_of_mass=point(surface+com_height*normal);
        result.flatness=flatness;result.slope_deg=slope;result.sparsity=sparsity;
        result.traversability=std::clamp(1000.*flatness+slope_weight*slope+sparsity,0.,1.);
        if(paper&&result.traversability>=1.){result.available=false;++stats.model_rejections;}
        return result;
    }
    bool native(const PutnPlane &a,const PutnPlane &b)const {
        if(!a.available||!b.available)return false;
        V3 ap=vec(a.center_of_mass),bp=vec(b.center_of_mass),diff=bp-ap;
        for(int i=0;i<=20;++i){double t=i/20.;V3 ez=(1-t)*vec(a.normal)+t*vec(b.normal);
            if(ez.norm()==0)return false;ez.normalize();
            // Source intended tangent frame, with corrected projection onto plane.
            V3 ex=diff-diff.dot(ez)*ez;
            if(ex.norm()<1e-12)ex=ez.unitOrthogonal();else ex.normalize();
            V3 ey=ez.cross(ex),c=ap+t*diff;
            for(int y=-2;y<=2;++y)for(int z=-2;z<=2;++z)
                if(!free(c+.15*y*ey+.1*z*ez))return false;
        }
        return true;
    }
    double nativeCost(const PutnPlane &a,const PutnPlane &b)const {
        if(paper){
            if(!a.available||!b.available||a.traversability<0||b.traversability<0||
                a.traversability>=1||b.traversability>=1)return std::numeric_limits<double>::infinity();
            return (vec(a.center_of_mass)-vec(b.center_of_mass)).norm()*
                (1.+.1*(1./(1.-a.traversability)+1./(1.-b.traversability)-2.));
        }
        return (vec(a.center_of_mass)-vec(b.center_of_mass)).norm()*
            (1.+.1*(1./(1.0001-a.traversability)+1./(1.0001-b.traversability)));
    }
    bool edge(const PutnPlane &a,const PutnPlane &b,bool own,double &cost){
        if(!terrain.validMotion(a.support,b.support)){++stats.common_rejections;return false;}
        if(!native(a,b)){++stats.model_rejections;return false;}
        ++stats.edge_cost_calls;
        cost=own?nativeCost(a,b):terrain.edgeCostBreakdown(a.support,b.support).objective_value;
        return std::isfinite(cost)&&cost>=0;
    }
    void updateDescendants(std::size_t root){
        std::vector<std::size_t> queue{root};
        for(std::size_t p=0;p<queue.size();++p)for(auto c:tree[queue[p]].children){
            tree[c].cost=tree[queue[p]].cost+tree[c].edge;queue.push_back(c);
        }
    }
    bool isAncestor(std::size_t possible,std::size_t node)const {
        for(std::size_t n=0;n<=tree.size();++n){if(node==possible)return true;if(node==0)return false;node=tree[node].parent;}
        throw std::runtime_error("PUTN tree cycle");
    }
    bool consistent()const {
        if(tree.empty())return true;
        for(std::size_t i=1;i<tree.size();++i){auto p=tree[i].parent;if(p>=tree.size())return false;
            if(std::abs(tree[i].cost-(tree[p].cost+tree[i].edge))>1e-9*(1.+tree[i].cost))return false;
            std::size_t c=i,steps=0;while(c!=0&&steps<=tree.size()){c=tree[c].parent;++steps;}if(steps>tree.size())return false;
        }return true;
    }
};
PutnStatic::PutnStatic(const Terrain &t,const std::vector<Rect> &obs):impl_(new Impl(t,obs)){}
PutnStatic::PutnStatic(const Terrain &t,const std::vector<Point3> &cloud,bool paper,double cap):impl_(new Impl(t,cloud,paper,cap)){}
PutnStatic::~PutnStatic()=default;
PutnPlane PutnStatic::fitPlane(Point p){return impl_->fit(p);}
bool PutnStatic::nativeMotion(const PutnPlane &a,const PutnPlane &b)const{return impl_->native(a,b);}
double PutnStatic::nativeCost(const PutnPlane &a,const PutnPlane &b)const{return impl_->nativeCost(a,b);}
const PutnStats &PutnStatic::stats()const{return impl_->stats;}
const PutnSnapshot &PutnStatic::lastSnapshot()const{return impl_->last;}
void PutnStatic::setConvergenceStop(double relative_eps,std::uint64_t window_iterations){
    if(window_iterations&&!(std::isfinite(relative_eps)&&relative_eps>0&&relative_eps<1))
        throw std::invalid_argument("convergence stop needs 0 < relative_eps < 1");
    impl_->stop_eps=window_iterations?relative_eps:0;
    impl_->stop_window=window_iterations;
}
const PutnConvergence &PutnStatic::convergence()const{return impl_->convergence;}
int PutnStatic::solve(Point start,Point goal,std::uint64_t seed,double budget,bool own,std::uint64_t work,
                      const std::function<void(const PutnSnapshot &)> &callback){
    if(!std::isfinite(budget)||budget<=0)throw std::invalid_argument("positive finite PUTN budget required");
    if(impl_->paper&&!own)throw std::invalid_argument("paper PUTN requires native objective");
    auto &m=*impl_;double prep=m.stats.preprocessing_seconds,conversion=m.stats.conversion_seconds;
    m.stats={};m.stats.preprocessing_seconds=prep;m.stats.conversion_seconds=conversion;
    m.last={};m.tree.clear();m.convergence={};auto began=Clock::now();
    PutnPlane origin=m.fit(start),target=m.fit(goal);
    if(!origin.available||!target.available){m.stats.solve_seconds=elapsed(began);return -2;}
    m.tree.push_back({origin,0,0,0,{}});
    double best=std::numeric_limits<double>::infinity();
    std::vector<std::pair<std::size_t,double>> close;
    auto checkGoal=[&](std::size_t node){double cost;
        if(distanceXY(m.tree[node].plane.support,goal)<goal_threshold&&m.edge(m.tree[node].plane,target,own,cost))close.emplace_back(node,cost);};
    auto publish=[&](){
        std::size_t winner=0;double candidate=best;bool found=false;
        for(auto c:close)if(m.tree[c.first].cost+c.second<candidate){candidate=m.tree[c.first].cost+c.second;winner=c.first;found=true;}
        if(!found)return;
        std::vector<std::size_t> chain;
        for(std::size_t i=winner;;i=m.tree[i].parent){chain.push_back(i);if(i==0)break;
            if(chain.size()>m.tree.size())throw std::runtime_error("PUTN parent cycle");}
        std::reverse(chain.begin(),chain.end());
        PutnSnapshot snap;double actual=0;PutnPlane prev=origin;
        for(auto i:chain){auto &p=m.tree[i].plane;snap.path.push_back(p.support);snap.raw_com_path.push_back(p.center_of_mass);
            if(m.paper)snap.native_planes.push_back(p);
            actual+=own?m.nativeCost(prev,p):m.terrain.edgeCostBreakdown(prev.support,p.support).objective_value;prev=p;}
        if(distanceXY(prev.support,goal)>0){snap.path.push_back(goal);snap.raw_com_path.push_back(target.center_of_mass);
            if(m.paper)snap.native_planes.push_back(target);
            actual+=own?m.nativeCost(prev,target):m.terrain.edgeCostBreakdown(prev.support,goal).objective_value;}
        if(!(actual<best))return;
        snap.elapsed_seconds=elapsed(began);snap.planner_objective=actual;best=actual;m.last=std::move(snap);++m.stats.improvements;
        if(callback)callback(m.last);
    };
    checkGoal(0);publish();
    std::mt19937_64 rng(seed);std::uniform_real_distribution<double> uniform(0.,1.);
    double focal=distanceXY(start,goal),ux=focal>0?(goal.x-start.x)/focal:1.,uy=focal>0?(goal.y-start.y)/focal:0.;
    while(work?m.stats.iterations<work:elapsed(began)<budget){
        // Optional convergence stop, the SEM-RRT* rule; it reads only the published cost.
        if(m.stop_window&&std::isfinite(best)){
            auto &c=m.convergence;
            if(!c.has_reference||best<c.reference_cost*(1.-m.stop_eps)){
                c.has_reference=true;c.reference_cost=best;c.reference_iteration=m.stats.iterations;
            }else if(m.stats.iterations-c.reference_iteration>=m.stop_window){c.converged=true;break;}
        }
        ++m.stats.iterations;Point q;
        if(!own&&std::isfinite(best)&&best>focal){
            double angle=2*pi*uniform(rng),r=std::sqrt(uniform(rng));
            double a=best*.5,b=.5*std::sqrt(std::max(0.,(best-focal)*(best+focal)));
            double x=a*r*std::cos(angle),y=b*r*std::sin(angle);
            q={.5*(start.x+goal.x)+ux*x-uy*y,.5*(start.y+goal.y)+uy*x+ux*y};
        }else if(!std::isfinite(best)&&uniform(rng)<goal_bias)q=goal;
        else q={m.bounds.xmin+(m.bounds.xmax-m.bounds.xmin)*uniform(rng),m.bounds.ymin+(m.bounds.ymax-m.bounds.ymin)*uniform(rng)};
        if(m.paper&&std::isfinite(best)){
            ++m.stats.informed_proposals;const auto proposal=m.fit(q);
            // Conservative rounding guard: retaining extra proposals is safe;
            // a slightly rounded-up lower bound must not exclude an improvement.
            const double bound_guard=1e-12*(1.+best);
            if(!proposal.available||(vec(origin.center_of_mass)-vec(proposal.center_of_mass)).norm()+
                (vec(proposal.center_of_mass)-vec(target.center_of_mass)).norm()>=best+bound_guard){
                ++m.stats.informed_rejections;continue;}
        }
        std::size_t nearest=0;double nearest_d=std::numeric_limits<double>::infinity();
        for(std::size_t i=0;i<m.tree.size();++i){double d=distanceXY(m.tree[i].plane.support,q);if(d<nearest_d){nearest=i;nearest_d=d;}}
        Point near=m.tree[nearest].plane.support;
        if(nearest_d>step_size){double f=step_size/nearest_d;q={near.x+(q.x-near.x)*f,near.y+(q.y-near.y)*f};}
        if(distanceXY(q,near)==0)continue;
        PutnPlane fresh=m.fit(q);if(!fresh.available)continue;
        std::vector<std::pair<std::size_t,double>> neighbors;
        for(std::size_t i=0;i<m.tree.size();++i)if(distanceXY(m.tree[i].plane.support,q)<neighbor_radius){double c;
            if(m.edge(m.tree[i].plane,fresh,own,c))neighbors.emplace_back(i,c);}
        if(neighbors.empty())continue;
        auto parent=*std::min_element(neighbors.begin(),neighbors.end(),[&](auto a,auto b){return m.tree[a.first].cost+a.second<m.tree[b.first].cost+b.second;});
        std::size_t id=m.tree.size();m.tree.push_back({fresh,parent.first,m.tree[parent.first].cost+parent.second,parent.second,{}});
        m.tree[parent.first].children.push_back(id);
        for(auto n:neighbors)if(n.first!=0&&!m.isAncestor(n.first,id)&&m.tree[id].cost+n.second<m.tree[n.first].cost){
            auto &old=m.tree[m.tree[n.first].parent].children;old.erase(std::remove(old.begin(),old.end(),n.first),old.end());
            m.tree[n.first].parent=id;m.tree[n.first].edge=n.second;m.tree[n.first].cost=m.tree[id].cost+n.second;
            m.tree[id].children.push_back(n.first);m.updateDescendants(n.first);++m.stats.rewires;
        }
        checkGoal(id);publish();
    }
    m.stats.solve_seconds=elapsed(began);m.stats.nodes=m.tree.size();m.stats.tree_consistent=m.consistent();
    if(!m.stats.tree_consistent)throw std::runtime_error("PUTN tree invariant failed");
    return std::isfinite(best)?1:0;
}
}
