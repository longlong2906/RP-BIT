#include "terrain25d/RpBit.hpp"
#include <ompl/base/ScopedState.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <queue>

namespace terrain25d {
namespace {
const double INF=std::numeric_limits<double>::infinity();
// cos(pi/8) rounded down. An 8-connected lattice path is at most 1/cos(pi/8)
// longer than the Euclidean one, so KAPPA * lattice distance is a lower bound.
const double KAPPA=.92387953251;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
bool finitePoint(Point p) { return std::isfinite(p.x)&&std::isfinite(p.y); }
bool same(Point a,Point b) { return a.x==b.x&&a.y==b.y; }
double roundDown(double value) { return std::isfinite(value)&&value>0.?std::nextafter(value,0.):value; }
// Absorbs the rounding of ~1e3 Dijkstra additions; always moves the value down.
double slack(double value) { return value-std::abs(value)*1e-11-1e-9; }

// Cells whose closed interval contains v immediately after leaving it in direction d.
void startRange(double v,double d,const std::vector<double> &lines,int &lo,int &hi) {
    const int n=static_cast<int>(lines.size())-1;
    int i=std::clamp(static_cast<int>(std::floor((v-lines[0])/(lines[1]-lines[0]))),0,n-1);
    while(i>0&&v<lines[i])--i;
    while(i<n-1&&v>=lines[i+1])++i;
    lo=hi=i;
    if(i>0&&v==lines[i]) {
        if(d<0.)lo=hi=i-1;
        else if(d==0.)lo=i-1;  // travelling exactly on a raster line: both incident cells
    }
}
struct Axis {
    int lo=0,hi=0,line=0,step=0;
    double origin=0.,delta=0.,end=0.,next=INF;
    void init(double a,double b,const std::vector<double> &lines) {
        origin=a;end=b;delta=b-a;step=delta>0.?1:delta<0.?-1:0;
        startRange(a,delta,lines,lo,hi);
        line=step>0?lo+1:lo;
        advance(lines);
    }
    // Parameter of the next interior raster line strictly between the endpoints.
    void advance(const std::vector<double> &lines) {
        const int n=static_cast<int>(lines.size())-1;
        next=INF;
        if(step>0&&line<n&&lines[line]<end)next=(lines[line]-origin)/delta;
        if(step<0&&line>0&&lines[line]>end)next=(lines[line]-origin)/delta;
    }
    void cross(const std::vector<double> &lines) { lo=hi=lo+step;line+=step;advance(lines); }
};

std::uint64_t bits(double value) { std::uint64_t out;std::memcpy(&out,&value,sizeof(out));return out; }
std::uint64_t nextToken() { static std::atomic<std::uint64_t> next{1};return next.fetch_add(1,std::memory_order_relaxed); }
// XY state with derived, per-objective bounds. Still a two-dimensional RealVector space.
class RpStateSpace final : public ob::RealVectorStateSpace {
public:
    struct StateType final : ob::RealVectorStateSpace::StateType {
        mutable std::uint64_t token=0,x_bits=0,y_bits=0;
        mutable RpObjective::Bounds bounds;
    };
    RpStateSpace():ob::RealVectorStateSpace(2){}
    ob::State *allocState() const override { auto *state=new StateType;state->values=new double[2];return state; }
    void freeState(ob::State *state) const override { auto *s=state->as<StateType>();delete[] s->values;delete s; }
    void copyState(ob::State *destination,const ob::State *source) const override {
        ob::RealVectorStateSpace::copyState(destination,source);
        destination->as<StateType>()->token=0;
    }
};

class RpInformedSampler final : public ob::InformedSampler {
public:
    RpInformedSampler(const ob::ProblemDefinitionPtr &p,unsigned attempts,const RpObjective &objective,RpCounters &counters)
        :ob::InformedSampler(p,attempts),objective_(objective),counters_(counters) {
        const auto &field=objective.field();const auto box=field.bounds();
        nx_=field.nx();x0_=box.xmin;y0_=box.ymin;res_=field.resolution();
        const auto *from=objective.potentialFromStart(),*to=objective.potentialToGoal();
        bound_.assign(static_cast<std::size_t>(nx_)*field.ny(),INF);
        for(int j=0;j<field.ny();++j)for(int i=0;i<nx_;++i) {
            const double value=from->cellLowerBound(i,j)+to->cellLowerBound(i,j);
            if(std::isfinite(value)){bound_[static_cast<std::size_t>(j)*nx_+i]=value;cells_.push_back(static_cast<std::uint32_t>(j*nx_+i));}
        }
        active_=cells_.size();
    }
    bool sampleUniform(ob::State *state,const ob::Cost &maxCost) override { return sampleUniform(state,ob::Cost(0.),maxCost); }
    bool sampleUniform(ob::State *state,const ob::Cost &minCost,const ob::Cost &maxCost) override {
        const double lo=minCost.value(),hi=maxCost.value();
        if(std::isnan(lo)||std::isnan(hi)||lo<0.||hi<=lo)return false;
        if(hi!=threshold_) {
            // The cell bound never exceeds g_hat+h_hat inside the cell, so the
            // prefix is a superset of the informed set; rejection below is exact.
            if(hi>threshold_)active_=cells_.size();
            active_=static_cast<std::size_t>(std::partition(cells_.begin(),cells_.begin()+static_cast<std::ptrdiff_t>(active_),
                [&](std::uint32_t cell){return bound_[cell]<hi;})-cells_.begin());
            threshold_=hi;
        }
        if(active_==0){++counters_.sampler_empty_prefix_calls;return false;}
        for(unsigned attempt=0;attempt<numIters_;++attempt) {
            ++counters_.sampler_cell_draws;
            const auto cell=cells_[static_cast<std::size_t>(rng_.uniformInt(0,static_cast<int>(active_)-1))];
            const double ux=rng_.uniform01(),uy=rng_.uniform01();
            const Point q{x0_+(static_cast<int>(cell%nx_)+ux)*res_,y0_+(static_cast<int>(cell/nx_)+uy)*res_};
            const auto value=objective_.bounds(q);
            const double f=value.g+value.h;
            if(!(f>=lo&&f<hi)){++counters_.sampler_rejected;continue;}
            setPoint(state,q);
            return true;
        }
        return false;
    }
    bool hasInformedMeasure() const override { return false; }
    // Inert with k-nearest and JIT off; reported as the current proposal area.
    double getInformedMeasure(const ob::Cost&) const override { return static_cast<double>(std::max<std::size_t>(active_,1))*res_*res_; }
    double getInformedMeasure(const ob::Cost &min,const ob::Cost &max) const override {
        return max.value()<=min.value()?0.:getInformedMeasure(max);
    }
private:
    const RpObjective &objective_;
    RpCounters &counters_;
    std::vector<double> bound_;
    std::vector<std::uint32_t> cells_;
    std::size_t active_=0;
    double threshold_=INF,x0_=0.,y0_=0.,res_=0.;
    int nx_=0;
    ompl::RNG rng_;
};

double turnAngle(Point a,Point v,Point b) {
    const double ux=v.x-a.x,uy=v.y-a.y,wx=b.x-v.x,wy=b.y-v.y;
    const double norm=std::hypot(ux,uy)*std::hypot(wx,wy);
    if(norm==0.)return 0.;
    return std::acos(std::clamp((ux*wx+uy*wy)/norm,-1.,1.));
}

struct Refiner {
    const SafetyPolicy &policy;
    const RpEdgeIntegrator &integrator;
    const double lambda;
    const RpRefineOptions &options;
    RpRefineResult out;
    std::vector<Point> p;
    std::vector<double> c;  // c[i] = J(p[i],p[i+1])
    double J(Point a,Point b) {
        ++out.edge_evaluations;
        const auto edge=integrator.evaluate(a,b);
        return edge.defined&&edge.supported?edge.length+lambda*edge.exposure/40.:INF;
    }
    bool valid(Point a,Point b) { ++out.validity_checks;return policy.validMotion(a,b); }
    bool budget() const { return out.edge_evaluations<options.max_edge_evaluations; }
    double total() const { double sum=0.;for(const double value:c)sum+=value;return sum; }
    // Replace a chain by one straight edge when that does not increase J.
    void shortcutPass() {
        const std::size_t window=8;
        for(std::size_t i=0;i+2<p.size()&&budget();++i) {
            const std::size_t last=std::min(p.size()-1,i+window);
            std::vector<double> chain(last-i+1,0.);
            for(std::size_t k=i;k<last;++k)chain[k-i+1]=chain[k-i]+c[k];
            for(std::size_t j=last;j>=i+2&&budget();--j) {
                const double direct=J(p[i],p[j]),old=chain[j-i];
                if(direct<=old+1e-12*std::max(1.,old)&&valid(p[i],p[j])) {
                    p.erase(p.begin()+static_cast<std::ptrdiff_t>(i)+1,p.begin()+static_cast<std::ptrdiff_t>(j));
                    c.erase(c.begin()+static_cast<std::ptrdiff_t>(i),c.begin()+static_cast<std::ptrdiff_t>(j));
                    c.insert(c.begin()+static_cast<std::ptrdiff_t>(i),direct);
                    ++out.shortcuts;break;
                }
            }
        }
    }
    // Sub-segments of a valid edge are valid; only their costs are needed.
    void subdivide() {
        std::vector<Point> points{p.front()};std::vector<double> costs;
        for(std::size_t i=0;i+1<p.size();++i) {
            const double length=distanceXY(p[i],p[i+1]);
            const int parts=std::max(1,static_cast<int>(std::ceil(length/options.max_edge_m)));
            if(parts==1||!budget()){points.push_back(p[i+1]);costs.push_back(c[i]);continue;}
            Point previous=p[i];
            for(int k=1;k<=parts;++k) {
                const double t=static_cast<double>(k)/parts;
                const Point q=k==parts?p[i+1]:Point{p[i].x+t*(p[i+1].x-p[i].x),p[i].y+t*(p[i+1].y-p[i].y)};
                costs.push_back(J(previous,q));points.push_back(q);previous=q;
            }
        }
        p=std::move(points);c=std::move(costs);
    }
    // Pattern search on each interior vertex; a move must lower J and stay safe.
    void relocatePass() {
        static const double r=.7071067811865476;
        static const double directions[8][2]={{1,0},{r,r},{0,1},{-r,r},{-1,0},{-r,-r},{0,-1},{r,-r}};
        struct Candidate { double first,second;Point q; };
        for(double step=options.step_m;step>=options.min_step_m&&budget();step*=.5)
            for(unsigned pass=0;pass<2&&budget();++pass) {
                bool moved=false;
                for(std::size_t i=1;i+1<p.size()&&budget();++i) {
                    const double base=c[i-1]+c[i];
                    std::vector<Candidate> better;
                    for(const auto &direction:directions) {
                        const Point q{p[i].x+step*direction[0],p[i].y+step*direction[1]};
                        const double first=J(p[i-1],q);
                        if(!(first<base))continue;
                        const double second=J(q,p[i+1]);
                        if(first+second<base-1e-9*std::max(1.,base))better.push_back({first,second,q});
                    }
                    std::sort(better.begin(),better.end(),[](const Candidate &a,const Candidate &b){return a.first+a.second<b.first+b.second;});
                    for(const auto &candidate:better)
                        if(valid(p[i-1],candidate.q)&&valid(candidate.q,p[i+1])) {
                            p[i]=candidate.q;c[i-1]=candidate.first;c[i]=candidate.second;++out.moves;moved=true;break;
                        }
                }
                if(!moved)break;
            }
    }
    // Replace a corner above the turn limit by an arc of equal chords (or drop the vertex).
    // The arc may be shifted outward, away from whatever the corner wraps around. Every
    // new segment keeps a minimum length, so a sharp corner is never hidden behind
    // micro-segments. A replacement must lower the local number of sharp turns and pass
    // the full safety policy; the whole pass may raise J by at most round_cost_ratio.
    void roundPass() {
        static const double chords[]={1.,1.5,2.5,4.,7.},shifts[]={0.,.5,1.,1.5};
        struct Candidate { double delta;std::vector<Point> q;std::vector<double> costs; };
        const double limit=options.turn_limit_rad,least=options.min_chord_m;
        double cap=options.round_cost_ratio*total();
        for(unsigned iteration=0;iteration<3;++iteration) {
            bool changed=false;
            for(std::size_t i=p.size()>=3?p.size()-2:0;i>=1&&budget();--i) {
                const Point prev=p[i-1],v=p[i],next=p[i+1];
                const double theta=turnAngle(prev,v,next);
                if(theta<=limit)continue;
                const double before=distanceXY(prev,v),after=distanceXY(v,next),base=c[i-1]+c[i];
                std::vector<Point> chain;
                if(i>=2)chain.push_back(p[i-2]);
                chain.push_back(prev);
                const std::size_t head=chain.size();
                const auto sharpTurns=[&](const std::vector<Point> &middle) {
                    chain.resize(head);
                    chain.insert(chain.end(),middle.begin(),middle.end());
                    chain.push_back(next);
                    if(i+2<p.size())chain.push_back(p[i+2]);
                    // A small margin keeps accepted arcs clear of the limit for any evaluator's rounding.
                    unsigned count=0;
                    for(std::size_t m=1;m+1<chain.size();++m)if(turnAngle(chain[m-1],chain[m],chain[m+1])>limit-1e-6)++count;
                    return count;
                };
                const unsigned current=sharpTurns({v});
                std::vector<Candidate> candidates;
                const auto consider=[&](std::vector<Point> q) {
                    const Point first=q.empty()?next:q.front(),last=q.empty()?prev:q.back();
                    if(distanceXY(prev,first)<least||distanceXY(last,next)<least||sharpTurns(q)>=current)return;
                    Candidate item{0.,std::move(q),{}};
                    Point from=prev;
                    for(std::size_t m=0;m<=item.q.size();++m) {
                        const Point to=m<item.q.size()?item.q[m]:next;
                        item.costs.push_back(J(from,to));item.delta+=item.costs.back();from=to;
                    }
                    item.delta-=base;
                    if(item.delta<=cap)candidates.push_back(std::move(item));
                };
                consider({});
                // k arc vertices, each turning theta/k; one more vertex leaves room for a shifted arc.
                const int fewest=std::max(2,static_cast<int>(std::ceil(theta/limit-1e-9)));
                const double cosHalf=std::cos(.5*theta);
                const Point d1{(v.x-prev.x)/before,(v.y-prev.y)/before},d2{(next.x-v.x)/after,(next.y-v.y)/after};
                for(int k=fewest;k<=fewest+1&&k<=8&&cosHalf>.05;++k) {
                    const double psi=theta/k,turn=d1.x*d2.y-d1.y*d2.x<0.?-psi:psi,cs=std::cos(turn),sn=std::sin(turn);
                    for(const double scale:chords) {
                        const double chord=scale*least;
                        // Distance from the corner to the first and last arc vertex on the two segments.
                        const double x=chord*std::sin(.5*(k-1)*psi)/(2.*std::sin(.5*psi)*cosHalf);
                        if(x>before-least||x>after-least)continue;
                        std::vector<Point> arc{{v.x-x*d1.x,v.y-x*d1.y}};
                        Point direction=d1;
                        for(int m=1;m<k;++m) {
                            direction={direction.x*cs-direction.y*sn,direction.x*sn+direction.y*cs};
                            arc.push_back({arc.back().x+chord*direction.x,arc.back().y+chord*direction.y});
                        }
                        const Point apex=k%2?arc[k/2]:Point{.5*(arc[k/2-1].x+arc[k/2].x),.5*(arc[k/2-1].y+arc[k/2].y)};
                        for(const double shift:shifts) {
                            std::vector<Point> moved=arc;
                            for(auto &vertex:moved){vertex.x+=shift*(v.x-apex.x);vertex.y+=shift*(v.y-apex.y);}
                            consider(std::move(moved));
                        }
                    }
                }
                std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate &a,const Candidate &b){return a.delta<b.delta;});
                for(const auto &candidate:candidates) {
                    bool safe=true;
                    Point from=prev;
                    for(std::size_t m=0;safe&&m<=candidate.q.size();++m) {
                        const Point to=m<candidate.q.size()?candidate.q[m]:next;
                        safe=valid(from,to);from=to;
                    }
                    if(!safe)continue;
                    const auto at=static_cast<std::ptrdiff_t>(i);
                    p.erase(p.begin()+at);p.insert(p.begin()+at,candidate.q.begin(),candidate.q.end());
                    c.erase(c.begin()+at-1,c.begin()+at+1);c.insert(c.begin()+at-1,candidate.costs.begin(),candidate.costs.end());
                    cap-=candidate.delta;++out.rounded;changed=true;break;
                }
            }
            if(!changed)break;
        }
        for(std::size_t i=1;i+1<p.size();++i)if(turnAngle(p[i-1],p[i],p[i+1])>limit)++out.kept_sharp;
    }
};
}

RpEdgeIntegrator::RpEdgeIntegrator(const Terrain &terrain,const SafetyField &field):terrain_(terrain),field_(field) {
    for(int i=0;i<field.nx();++i)xs_.push_back(field.cellBounds(i,0).xmin);
    xs_.push_back(field.cellBounds(field.nx()-1,0).xmax);
    for(int j=0;j<field.ny();++j)ys_.push_back(field.cellBounds(0,j).ymin);
    ys_.push_back(field.cellBounds(0,field.ny()-1).ymax);
}
RpEdgeCost RpEdgeIntegrator::evaluate(Point a,Point b) const {
    RpEdgeCost out;
    const auto box=field_.bounds();
    if(!finitePoint(a)||!finitePoint(b)||std::min(a.x,b.x)<box.xmin||std::max(a.x,b.x)>box.xmax||
       std::min(a.y,b.y)<box.ymin||std::max(a.y,b.y)>box.ymax)return out;
    std::vector<double> t;
    const auto surface=terrain_.surfaceEdge(a,b,&t);
    if(!surface.has_data||!std::isfinite(surface.cost)||t.size()<2||t.size()!=surface.points.size())return out;
    out.defined=out.supported=true;out.length=surface.cost;
    Axis x,y;x.init(a.x,b.x,xs_);y.init(a.y,b.y,ys_);
    const std::size_t pieces=t.size()-1;
    const auto pieceLength=[&](std::size_t k) {
        const auto &p=surface.points[k],&q=surface.points[k+1];
        return std::hypot(std::hypot(q.x-p.x,q.y-p.y),q.z-p.z);
    };
    std::size_t k=0;
    double length=pieceLength(0),span=t[1]-t[0],lo=0.;
    long double exposure=0.;
    // Three monotone cut sequences: terrain triangles, SEM columns, SEM rows.
    for(;;) {
        const double pieceEnd=k+1<pieces?t[k+1]:1.;
        const double hi=std::min({pieceEnd,x.next,y.next,1.});
        if(hi>lo) {
            double risk=0.;bool supported=true;
            for(int j=y.lo;j<=y.hi;++j)for(int i=x.lo;i<=x.hi;++i) {
                if(!field_.supported(i,j))supported=false;
                else risk=std::max(risk,field_.risk(i,j));
            }
            if(!supported)out.supported=false;
            else exposure+=static_cast<long double>(risk)*length*((hi-lo)/span);
        }
        if(hi>=1.)break;
        if(pieceEnd<=hi){++k;length=pieceLength(k);span=t[k+1]-t[k];}
        if(x.next<=hi)x.cross(xs_);
        if(y.next<=hi)y.cross(ys_);
        lo=std::max(lo,hi);
    }
    out.exposure=static_cast<double>(exposure);
    return out;
}

RiskPotential::RiskPotential(const SafetyField &field,double riskCap,double lambda,Point source)
    :nx_(field.nx()),ny_(field.ny()) {
    const auto box=field.bounds();
    x0_=box.xmin;y0_=box.ymin;res_=field.resolution();
    if(!std::isfinite(lambda)||lambda<0.||!finitePoint(source)||source.x<box.xmin||source.x>box.xmax||
       source.y<box.ymin||source.y>box.ymax)throw std::invalid_argument("invalid risk potential request");
    // The cos(pi/8) lattice bound needs square cells: no clipped last row or column.
    if(std::abs(box.xmax-(x0_+nx_*res_))>1e-12*res_||std::abs(box.ymax-(y0_+ny_*res_))>1e-12*res_)
        throw std::invalid_argument("risk potential requires an unclipped square raster");
    weight_.assign(static_cast<std::size_t>(nx_)*ny_,INF);
    for(int j=0;j<ny_;++j)for(int i=0;i<nx_;++i)
        if(field.supported(i,j)&&field.risk(i,j)<=riskCap)
            weight_[static_cast<std::size_t>(j)*nx_+i]=1.+lambda/40.*field.risk(i,j);
    const int NX=nx_+1,NY=ny_+1;
    u_.assign(static_cast<std::size_t>(NX)*NY,INF);
    using Item=std::pair<double,int>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> queue;
    // Every corner of every closed cell containing the source starts at zero.
    std::vector<double> xs(NX),ys(NY);
    for(int i=0;i<NX;++i)xs[i]=x0_+i*res_;
    for(int j=0;j<NY;++j)ys[j]=y0_+j*res_;
    startRange(source.x,0.,xs,si0_,si1_);startRange(source.y,0.,ys,sj0_,sj1_);
    for(int j=sj0_;j<=sj1_+1;++j)for(int i=si0_;i<=si1_+1;++i){u_[static_cast<std::size_t>(j)*NX+i]=0.;queue.push({0.,j*NX+i});}
    const auto w=[&](int i,int j){return i<0||j<0||i>=nx_||j>=ny_?INF:weight_[static_cast<std::size_t>(j)*nx_+i];};
    const double side=res_,diagonal=std::sqrt(2.)*res_;
    while(!queue.empty()) {
        const double d=queue.top().first;const int n=queue.top().second;queue.pop();
        if(d>u_[n])continue;
        const int i=n%NX,j=n/NX;
        const auto relax=[&](int ni,int nj,double cost) {
            if(ni<0||nj<0||ni>=NX||nj>=NY||!std::isfinite(cost))return;
            const int m=nj*NX+ni;const double value=d+cost;
            if(value<u_[m]){u_[m]=value;queue.push({value,m});}
        };
        // A side keeps the cheaper incident open cell; a blocked cell has no diagonal.
        relax(i+1,j,side*std::min(w(i,j-1),w(i,j)));
        relax(i-1,j,side*std::min(w(i-1,j-1),w(i-1,j)));
        relax(i,j+1,side*std::min(w(i-1,j),w(i,j)));
        relax(i,j-1,side*std::min(w(i-1,j-1),w(i,j-1)));
        relax(i+1,j+1,diagonal*w(i,j));
        relax(i-1,j-1,diagonal*w(i-1,j-1));
        relax(i+1,j-1,diagonal*w(i,j-1));
        relax(i-1,j+1,diagonal*w(i-1,j));
    }
    for(double &value:u_)value*=KAPPA;
    const int si=std::clamp(static_cast<int>(std::floor((source.x-x0_)/res_)),0,nx_-1);
    const int sj=std::clamp(static_cast<int>(std::floor((source.y-y0_)/res_)),0,ny_-1);
    const double value=raw(source,si,sj);
    offset_=std::isfinite(value)?value:0.;
}
double RiskPotential::raw(Point q,int i,int j) const {
    const double w=weight_[static_cast<std::size_t>(j)*nx_+i];
    const int NX=nx_+1;
    const double u00=u_[static_cast<std::size_t>(j)*NX+i],u10=u_[static_cast<std::size_t>(j)*NX+i+1],
        u01=u_[static_cast<std::size_t>(j+1)*NX+i],u11=u_[static_cast<std::size_t>(j+1)*NX+i+1];
    if(!std::isfinite(w)||!std::isfinite(u00)||!std::isfinite(u10)||!std::isfinite(u01)||!std::isfinite(u11))return INF;
    const double px=std::clamp(q.x-(x0_+i*res_),0.,res_),py=std::clamp(q.y-(y0_+j*res_),0.,res_);
    // The source's cells have zero side data; the smallest extension dips to -w * distance to the sides.
    if(sourceCell(i,j))return -w*std::min({px,res_-px,py,res_-py});
    // Largest extension: inf over one side of [linear side data + w * distance], in closed form.
    const auto side=[&](double g0,double g1,double along,double away) {
        const double slope=(g1-g0)/res_;
        double t=along;
        if(away>0.){const double s=w*w-slope*slope;t=s>0.?along-away*slope/std::sqrt(s):(slope>0.?0.:res_);}
        t=std::clamp(t,0.,res_);
        return g0+slope*t+w*std::hypot(away,t-along);
    };
    return std::min({side(u00,u10,px,py),side(u01,u11,px,res_-py),side(u00,u01,py,px),side(u10,u11,py,res_-px)});
}
double RiskPotential::at(Point q) const {
    if(!finitePoint(q)||q.x<x0_||q.y<y0_||q.x>x0_+nx_*res_||q.y>y0_+ny_*res_)return INF;
    const int i=std::clamp(static_cast<int>(std::floor((q.x-x0_)/res_)),0,nx_-1);
    const int j=std::clamp(static_cast<int>(std::floor((q.y-y0_)/res_)),0,ny_-1);
    const double value=raw(q,i,j);
    return std::isfinite(value)?std::max(0.,slack(value-offset_)):INF;
}
double RiskPotential::cellLowerBound(int i,int j) const {
    const double w=weight_[static_cast<std::size_t>(j)*nx_+i];
    const int NX=nx_+1;
    const double least=std::min({u_[static_cast<std::size_t>(j)*NX+i],u_[static_cast<std::size_t>(j)*NX+i+1],
        u_[static_cast<std::size_t>(j+1)*NX+i],u_[static_cast<std::size_t>(j+1)*NX+i+1]});
    if(!std::isfinite(w)||!std::isfinite(least))return INF;
    // The largest extension never goes below the smallest corner value of its cell.
    return sourceCell(i,j)?0.:std::max(0.,slack(least-offset_));
}

RiskBoxMin::RiskBoxMin(const SafetyField &field,double riskCap):nx_(field.nx()),ny_(field.ny()),global_(INF) {
    const auto box=field.bounds();
    x0_=box.xmin;y0_=box.ymin;res_=field.resolution();
    std::vector<double> base(static_cast<std::size_t>(nx_)*ny_,INF);
    for(int j=0;j<ny_;++j)for(int i=0;i<nx_;++i)
        if(field.supported(i,j)&&field.risk(i,j)<=riskCap) {
            base[static_cast<std::size_t>(j)*nx_+i]=field.risk(i,j);global_=std::min(global_,field.risk(i,j));
        }
    levels_.push_back(std::move(base));
    for(int k=1;k<=6&&(1<<k)<=std::min(nx_,ny_);++k) {
        const int half=1<<(k-1),size=1<<k,cols=nx_-size+1,rows=ny_-size+1,previousCols=nx_-half+1;
        const auto &previous=levels_.back();
        std::vector<double> level(static_cast<std::size_t>(cols)*rows);
        for(int j=0;j<rows;++j)for(int i=0;i<cols;++i)
            level[static_cast<std::size_t>(j)*cols+i]=std::min({previous[static_cast<std::size_t>(j)*previousCols+i],
                previous[static_cast<std::size_t>(j)*previousCols+i+half],
                previous[static_cast<std::size_t>(j+half)*previousCols+i],
                previous[static_cast<std::size_t>(j+half)*previousCols+i+half]});
        levels_.push_back(std::move(level));
    }
}
double RiskBoxMin::minRisk(Point a,Point b) const {
    // One extra cell on each side covers closed-cell boundary contacts.
    const int i0=std::clamp(static_cast<int>(std::floor((std::min(a.x,b.x)-x0_)/res_))-1,0,nx_-1);
    const int i1=std::clamp(static_cast<int>(std::floor((std::max(a.x,b.x)-x0_)/res_))+1,0,nx_-1);
    const int j0=std::clamp(static_cast<int>(std::floor((std::min(a.y,b.y)-y0_)/res_))-1,0,ny_-1);
    const int j1=std::clamp(static_cast<int>(std::floor((std::max(a.y,b.y)-y0_)/res_))+1,0,ny_-1);
    const int extent=std::max(i1-i0,j1-j0)+1;
    int k=0;
    while(k+1<static_cast<int>(levels_.size())&&(2<<k)<=extent)++k;
    const int size=1<<k;
    if(2*size<extent)return global_;  // larger than the pyramid: any superset minimum is valid
    const int cols=nx_-size+1,rows=ny_-size+1;
    const int ia=std::min(i0,cols-1),ib=std::clamp(i1-size+1,0,cols-1),ja=std::min(j0,rows-1),jb=std::clamp(j1-size+1,0,rows-1);
    const auto &level=levels_[static_cast<std::size_t>(k)];
    return std::min({level[static_cast<std::size_t>(ja)*cols+ia],level[static_cast<std::size_t>(ja)*cols+ib],
        level[static_cast<std::size_t>(jb)*cols+ia],level[static_cast<std::size_t>(jb)*cols+ib]});
}

RpObjective::RpObjective(const ob::SpaceInformationPtr &si,const Terrain &map,std::shared_ptr<const SafetyPolicy> safety,
    Point start,Point goal,const RpBitOptions &options,PolicyCounters &policy,RpCounters &counters)
    :ob::OptimizationObjective(si),map_(map),safety_(std::move(safety)),start_(start),goal_(goal),options_(options),
     policy_(policy),counters_(counters),integrator_(map,*safety_->field()),token_(nextToken()) {
    if(dynamic_cast<const RpStateSpace*>(si->getStateSpace().get())==nullptr)
        throw std::runtime_error("RP objective requires the RP state space");
    if(!map_.height(start_,start_z_)||!map_.height(goal_,goal_z_))throw std::runtime_error("RP objective endpoints have no terrain data");
    if(options_.potential) {
        from_start_=std::make_unique<RiskPotential>(*safety_->field(),safety_->riskCap(),options_.lambda,start_);
        to_goal_=std::make_unique<RiskPotential>(*safety_->field(),safety_->riskCap(),options_.lambda,goal_);
        box_=std::make_unique<RiskBoxMin>(*safety_->field(),safety_->riskCap());
    }
    description_="surface_length_sem_exposure";
    setCostToGoHeuristic([this](const ob::State *state,const ob::Goal*) {
        ++policy_.cost_to_go_calls;
        return ob::Cost(roundDown(memo(state).h));
    });
}
RpObjective::Bounds RpObjective::bounds(Point q) const {
    Bounds out;
    out.has_data=map_.height(q,out.z);
    double g=INF,h=INF;
    if(out.has_data) {
        g=std::hypot(distanceXY(start_,q),out.z-start_z_);
        h=std::hypot(distanceXY(q,goal_),out.z-goal_z_);
    }
    if(from_start_) {
        ++counters_.potential_evaluations;
        out.ps=from_start_->at(q);out.pg=to_goal_->at(q);
        g=std::max(g,out.ps);h=std::max(h,out.pg);
    }
    // BIT* disconnects the incumbent chain unless these are exactly zero.
    out.g=same(q,start_)?0.:g;out.h=same(q,goal_)?0.:h;
    return out;
}
const RpObjective::Bounds &RpObjective::memo(const ob::State *state) const {
    const auto *cached=state->as<RpStateSpace::StateType>();
    const auto x=bits(cached->values[0]),y=bits(cached->values[1]);
    if(cached->token!=token_||cached->x_bits!=x||cached->y_bits!=y) {
        cached->bounds=bounds(point(state));
        cached->token=token_;cached->x_bits=x;cached->y_bits=y;
    }
    return cached->bounds;
}
ob::Cost RpObjective::motionCost(const ob::State *a,const ob::State *b) const {
    ++policy_.motion_cost_calls;++counters_.fast_integral_calls;
    // Finite over supported cells above the cap: the safety gate rejects (and
    // BIT* blacklists) such an edge; a penalty never legalises it.
    const auto cost=integrator_.evaluate(point(a),point(b));
    if(!cost.defined)return infiniteCost();
    const double value=cost.length+options_.lambda*cost.exposure/40.;
    return std::isfinite(value)?ob::Cost(value):infiniteCost();
}
ob::Cost RpObjective::motionCostHeuristic(const ob::State *a,const ob::State *b) const {
    ++policy_.motion_heuristic_calls;
    const Point u=point(a),v=point(b);
    // BIT* uses the same hook for path bounds from the start and to the goal.
    if(same(u,start_)){++policy_.from_start_calls;return ob::Cost(roundDown(memo(b).g));}
    if(same(v,goal_)){++policy_.to_goal_calls;return ob::Cost(roundDown(memo(a).h));}
    const auto &p=memo(a),&q=memo(b);
    if(!p.has_data||!q.has_data)return infiniteCost();
    double bound=std::hypot(distanceXY(u,v),p.z-q.z);
    if(from_start_) {
        if(bound>0.) {
            // A valid straight edge only touches open cells of its bounding box.
            ++counters_.box_bound_calls;
            const double least=box_->minRisk(u,v);
            if(!std::isfinite(least))return infiniteCost();
            bound*=1.+options_.lambda/40.*least;
        }
        const auto gap=[](double x,double y) {
            if(std::isinf(x)&&std::isinf(y))return 0.;
            if(std::isinf(x)||std::isinf(y))return INF;
            return std::max(0.,std::abs(x-y)-2e-9);
        };
        bound=std::max({bound,gap(p.pg,q.pg),gap(p.ps,q.ps)});
    }
    return ob::Cost(roundDown(bound));
}
ob::InformedSamplerPtr RpObjective::allocInformedStateSampler(const ob::ProblemDefinitionPtr &p,unsigned int attempts) const {
    ++policy_.sampler_allocations;
    const unsigned limit=std::min(attempts,options_.sample_attempts);
    if(!from_start_)
        return std::make_shared<TerrainInformedSampler>(p,limit,map_,true,start_,goal_,policy_,false,safety_,true);
    return std::make_shared<RpInformedSampler>(p,limit,*this,counters_);
}

RpProblem makeRpProblem(const Terrain &map,std::shared_ptr<const SafetyPolicy> safety,Point start,Point goal,
                        const RpBitOptions &options,PolicyCounters &policy,RpCounters &counters) {
    if(!safety||!safety->appliesTo(map)||safety->riskCap()!=40.||safety->slopeCap()!=35.||!safety->corridorMask().empty())
        throw std::runtime_error("RP-BIT* requires the full safe cap40/slope35 domain");
    if(!std::isfinite(options.lambda)||options.lambda<0.||options.samples_per_batch==0||options.sample_attempts==0||
       !std::isfinite(options.rewire_factor)||options.rewire_factor<=1.)throw std::runtime_error("invalid RP-BIT* options");
    RpProblem p;
    auto space=std::make_shared<RpStateSpace>();
    const auto box=map.bounds();ob::RealVectorBounds bounds(2);
    bounds.setLow(0,box.xmin);bounds.setHigh(0,box.xmax);bounds.setLow(1,box.ymin);bounds.setHigh(1,box.ymax);
    space->setBounds(bounds);p.space=space;
    p.si=std::make_shared<ob::SpaceInformation>(space);
    p.si->setStateValidityChecker([safety,&policy](const ob::State *state) {
        ++policy.safety_state_checks;const auto q=point(state);return safety->rejectionReason(q,q).empty();
    });
    p.si->setMotionValidator(std::make_shared<TerrainMotionValidator>(p.si,map,safety,&policy));p.si->setup();
    p.pdef=std::make_shared<ob::ProblemDefinition>(p.si);
    ob::ScopedState<> s(space),g(space);setPoint(s.get(),start);setPoint(g.get(),goal);
    p.pdef->addStartState(s.get());
    auto singleton=std::make_shared<SingletonGoal>(p.si);singleton->setState(g.get());
    singleton->setThreshold(1e-9);p.pdef->setGoal(singleton);
    p.objective=std::make_shared<RpObjective>(p.si,map,safety,start,goal,options,policy,counters);
    p.pdef->setOptimizationObjective(p.objective);
    p.planner=std::make_shared<ompl::geometric::BITstar>(p.si,"BITstar_RP");
    p.planner->setSamplesPerBatch(options.samples_per_batch);p.planner->setRewireFactor(options.rewire_factor);
    p.planner->setUseKNearest(true);p.planner->setJustInTimeSampling(false);
    p.planner->setPruning(true);p.planner->setDropSamplesOnPrune(false);
    p.planner->setConsiderApproximateSolutions(false);
    p.planner->setProblemDefinition(p.pdef);p.planner->setup();
    if(!p.planner->getUseKNearest()||p.planner->getJustInTimeSampling()||!p.planner->getPruning()||
       p.planner->getSamplesPerBatch()!=options.samples_per_batch||p.planner->getRewireFactor()!=options.rewire_factor)
        throw std::runtime_error("requested RP-BIT* policy is not effective");
    return p;
}

unsigned rpSharpTurns(const std::vector<Point> &path) {
    std::vector<Point> points;
    for(const auto q:path)if(points.empty()||!same(points.back(),q))points.push_back(q);
    unsigned count=0;
    for(std::size_t i=1;i+1<points.size();++i)if(turnAngle(points[i-1],points[i],points[i+1])>.5235987755982988)++count;
    return count;
}
RpPathCost rpPathCost(const RpEdgeIntegrator &integrator,double lambda,const std::vector<Point> &path) {
    RpPathCost out;
    if(path.empty())return out;
    out.valid=true;
    const std::size_t edges=path.size()>1?path.size()-1:1;
    for(std::size_t i=0;i<edges;++i) {
        const Point a=path[i],b=path.size()>1?path[i+1]:a;
        const auto edge=integrator.evaluate(a,b);
        if(!edge.defined||!edge.supported){out.valid=false;return out;}
        out.length+=edge.length;out.exposure+=edge.exposure;out.length_xy+=distanceXY(a,b);
    }
    out.objective=out.length+lambda*out.exposure/40.;
    out.sharp_turns=rpSharpTurns(path);
    return out;
}

RpRefineResult rpRefine(const SafetyPolicy &policy,const RpEdgeIntegrator &integrator,double lambda,
                        std::vector<Point> path,const RpRefineOptions &options) {
    Refiner refiner{policy,integrator,lambda,options,{},std::move(path),{}};
    auto &p=refiner.p;
    bool defined=p.size()>=2;
    for(std::size_t i=0;defined&&i+1<p.size();++i) {
        refiner.c.push_back(refiner.J(p[i],p[i+1]));
        defined=std::isfinite(refiner.c.back());
    }
    if(defined) {
        double previous=refiner.total();
        for(unsigned sweep=0;sweep<options.max_sweeps&&refiner.budget();++sweep) {
            if(options.shortcut)refiner.shortcutPass();
            if(options.relocate){refiner.subdivide();refiner.relocatePass();}
            const double now=refiner.total();
            if(!(now<previous*(1.-1e-4)))break;
            previous=now;
        }
        if(options.shortcut)refiner.shortcutPass();
        if(options.round_turns)refiner.roundPass();
    }
    refiner.out.path=std::move(refiner.p);
    refiner.out.cost=rpPathCost(integrator,lambda,refiner.out.path);
    return std::move(refiner.out);
}

RpRunResult runRpBit(const Terrain &map,std::shared_ptr<const SafetyField> field,Point start,Point goal,
                     const RpRunOptions &options) {
    const auto began=Clock::now();
    RpRunResult result;
    if(!std::isfinite(options.budget_s)||options.budget_s<=0.||!std::isfinite(options.stop_relative_eps)||
       options.stop_relative_eps<0.||!std::isfinite(options.refinement.round_cost_ratio)||options.refinement.round_cost_ratio<0.||
       !std::isfinite(options.refinement.min_chord_m)||options.refinement.min_chord_m<=0.)
        throw std::runtime_error("invalid RP-BIT* run options");
    auto safety=std::make_shared<const SafetyPolicy>(map,std::move(field),40.,35.);
    if(!safety->validState(start)){result.status="INVALID_START";result.termination="INVALID_QUERY";return result;}
    if(!safety->validState(goal)){result.status="INVALID_GOAL";result.termination="INVALID_QUERY";return result;}
    ompl::RNG::setSeed(static_cast<unsigned>(options.planner_seed));
    auto problem=makeRpProblem(map,safety,start,goal,options.planner,result.counters,result.rp);
    const auto &integrator=problem.objective->integrator();
    const double lambda=options.planner.lambda;
    result.lower_bound=problem.objective->lowerBoundStartGoal();
    result.setup_s=seconds(began);

    std::vector<Point> pending;
    double pendingCost=INF,pendingElapsed=0.,published=INF,reference=INF;
    bool hasPending=false,converged=false;
    unsigned lastBatch=0,stale=0;
    const auto safe=[&](const std::vector<Point> &path) {
        if(path.size()==1)return safety->validState(path[0]);
        for(std::size_t i=1;i<path.size();++i)if(!safety->validMotion(path[i-1],path[i]))return false;
        return !path.empty();
    };
    // Publish the pending BIT* incumbent, refined when requested and in time.
    const auto process=[&](bool refine) {
        if(!hasPending)return;
        hasPending=false;
        std::vector<Point> path=pending;
        RpPathCost cost=rpPathCost(integrator,lambda,path);
        const char *origin="ompl_intermediate_solution_callback";
        double stamp=pendingElapsed;
        if(refine&&options.refine) {
            const auto refineStart=Clock::now();
            auto refined=rpRefine(*safety,integrator,lambda,pending,options.refinement);
            result.rp.refine_edge_evaluations+=refined.edge_evaluations;
            result.rp.refine_validity_checks+=refined.validity_checks;
            result.refine_s+=seconds(refineStart);++result.refinements;
            const double now=seconds(began);
            if(refined.cost.valid&&now<=options.budget_s&&safe(refined.path)) {
                path=std::move(refined.path);cost=refined.cost;origin="rp_refinement";stamp=now;
            }
        }
        if(!cost.valid||!safe(path))throw std::logic_error("BIT* incumbent failed the safety or objective contract");
        if(std::isfinite(published)&&!(cost.objective<published-1e-6*std::max(1.,published)))return;
        RpPublication item;
        item.id=result.publications.size();item.origin=origin;item.elapsed_s=stamp;item.bitstar_cost=pendingCost;
        item.batch=problem.planner->numBatches();item.path=std::move(path);item.cost=cost;
        published=cost.objective;
        result.publications.push_back(std::move(item));
    };
    if(same(start,goal)) {
        pending={start};pendingCost=0.;pendingElapsed=seconds(began);hasPending=true;process(false);
        result.status="EXACT_SOLUTION";result.termination="ANALYTIC_IDENTITY_QUERY";result.actual_wall_s=seconds(began);
        return result;
    }
    problem.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &states,const ob::Cost cost) {
        pending.clear();
        for(auto it=states.rbegin();it!=states.rend();++it)pending.push_back(point(*it));
        if(pending.empty()||!same(pending.front(),start)||!same(pending.back(),goal))
            throw std::runtime_error("BIT* callback endpoint mismatch");
        pendingCost=cost.value();pendingElapsed=seconds(began);hasPending=true;
        // Runtime audit on every real incumbent (states run goal -> start): each bound
        // must stay below the true cost it estimates, or BIT* could cut the incumbent chain.
        const auto exceeds=[](double bound,double value){return bound>value*(1.+1e-9)+1e-9;};
        std::vector<double> toGoal(states.size(),0.);
        for(std::size_t i=1;i<states.size();++i) {
            const auto edge=integrator.evaluate(point(states[i]),point(states[i-1]));
            const double value=edge.length+lambda*edge.exposure/40.;
            toGoal[i]=toGoal[i-1]+value;
            if(exceeds(problem.objective->motionCostHeuristic(states[i],states[i-1]).value(),value))++result.rp.heuristic_violations;
        }
        for(std::size_t i=0;i<states.size();++i) {
            const auto bound=problem.objective->bounds(point(states[i]));
            if(exceeds(bound.h,toGoal[i])||exceeds(bound.g,toGoal.back()-toGoal[i]))++result.rp.heuristic_violations;
        }
        if(++result.raw_incumbents==1&&options.planner.batch_after_first_solution&&
           options.planner.batch_after_first_solution!=options.planner.samples_per_batch) {
            problem.planner->setSamplesPerBatch(options.planner.batch_after_first_solution);
            ++result.batch_transitions;
        }
        if(!options.refine)process(false);
    });
    const auto status=problem.planner->solve(ob::PlannerTerminationCondition([&]() {
        // Work-based triggers: refinement and the convergence test run at batch boundaries.
        const unsigned batch=problem.planner->numBatches();
        if(batch!=lastBatch) {
            lastBatch=batch;
            process(true);
            if(options.stop_batches&&!result.publications.empty()) {
                if(published<reference*(1.-options.stop_relative_eps)){reference=published;stale=0;}
                else if(++stale>=options.stop_batches)converged=true;
            }
        }
        if(converged)return true;
        if(options.test_max_iterations&&problem.planner->numIterations()>=options.test_max_iterations)return true;
        return seconds(began)>=options.budget_s;
    }));
    process(seconds(began)<options.budget_s);
    result.actual_wall_s=seconds(began);
    result.iterations=problem.planner->numIterations();result.batches=problem.planner->numBatches();
    result.status=!result.publications.empty()?"EXACT_SOLUTION":"NO_SOLUTION";
    (void)status;
    result.termination=converged?"CONVERGED_NO_IMPROVEMENT":
        options.test_max_iterations&&result.iterations>=options.test_max_iterations?"WORK_BUDGET":
        result.actual_wall_s>=options.budget_s?"TIME_BUDGET":
        !result.publications.empty()&&published<=result.lower_bound*(1.+1e-9)+1e-9?"LOWER_BOUND_REACHED":
        "UNKNOWN_NATIVE_PLANNER_STOP";
    return result;
}
}
