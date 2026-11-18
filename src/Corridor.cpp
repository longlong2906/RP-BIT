#include "terrain25d/Corridor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory_resource>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <boost/multiprecision/cpp_int.hpp>

namespace terrain25d {
namespace {
using Clock=std::chrono::steady_clock;
using Node=std::uint32_t;
constexpr Node none=std::numeric_limits<Node>::max();
constexpr double inf=std::numeric_limits<double>::infinity();
double elapsed(Clock::time_point begin) { return std::chrono::duration<double>(Clock::now()-begin).count(); }
struct ResourceLimit {};

// All long-lived graph/queue/label allocations use this exact live-byte budget.
// A conservative external reservation covers result vectors, path statistics,
// traversal scratch and the backing allocator's container-node overhead.
class Budget final:public std::pmr::memory_resource {
public:
    explicit Budget(std::uint64_t limit):limit_(limit) {}
    void reserveExternal(std::uint64_t bytes) {
        if (bytes>limit_-live_) throw ResourceLimit{};
        live_+=bytes;peak_=std::max(peak_,live_);
    }
    std::uint64_t peak() const { return peak_; }
private:
    void *do_allocate(std::size_t bytes,std::size_t alignment) override {
        // Charge 64 bytes of allocation overhead in addition to requested bytes.
        if (bytes>limit_-live_ || 64>limit_-live_-bytes) throw ResourceLimit{};
        auto *p=std::pmr::new_delete_resource()->allocate(bytes,alignment);
        live_+=bytes+64;peak_=std::max(peak_,live_);return p;
    }
    void do_deallocate(void *p,std::size_t bytes,std::size_t alignment) override {
        std::pmr::new_delete_resource()->deallocate(p,bytes,alignment);live_-=bytes+64;
    }
    bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override { return this==&other; }
    std::uint64_t limit_,live_=0,peak_=0;
};
struct Edge { double length=inf,exposure=inf; };
struct Label { double length,exposure;Node node,parent;bool active=true; };
struct BeamLabel {
    double length,exposure,maximum;
    long double risk_sum;
    std::uint64_t risk_count;
    Node node,parent;
    bool active=true;
};
bool samePoint(Point a,Point b) {return a.x==b.x&&a.y==b.y;}
int orientation(Point a,Point b,Point c) {
    const long double x=static_cast<long double>(b.x)-a.x,y=static_cast<long double>(b.y)-a.y;
    const long double u=static_cast<long double>(c.x)-a.x,v=static_cast<long double>(c.y)-a.y;
    const auto p=x*v,q=y*u,d=p-q;
    if (std::abs(d)>32*std::numeric_limits<long double>::epsilon()*(std::abs(p)+std::abs(q))) return d>0?1:-1;
    using Exact=boost::multiprecision::cpp_rational;
    const Exact exact=(Exact(b.x)-Exact(a.x))*(Exact(c.y)-Exact(a.y))-
                      (Exact(b.y)-Exact(a.y))*(Exact(c.x)-Exact(a.x));
    return exact==0?0:exact>0?1:-1;
}
bool onSegment(Point a,Point b,Point c) {
    return c.x>=std::min(a.x,b.x)&&c.x<=std::max(a.x,b.x)&&
           c.y>=std::min(a.y,b.y)&&c.y<=std::max(a.y,b.y)&&orientation(a,b,c)==0;
}
bool intersects(Point a,Point b,Point c,Point d) {
    if (std::max(a.x,b.x)<std::min(c.x,d.x)||std::max(c.x,d.x)<std::min(a.x,b.x)||
        std::max(a.y,b.y)<std::min(c.y,d.y)||std::max(c.y,d.y)<std::min(a.y,b.y)) return false;
    const int x=orientation(a,b,c),y=orientation(a,b,d),u=orientation(c,d,a),v=orientation(c,d,b);
    return (x*y<0&&u*v<0)||(!x&&onSegment(a,b,c))||(!y&&onSegment(a,b,d))||
           (!u&&onSegment(c,d,a))||(!v&&onSegment(c,d,b));
}
// Check just an appended edge against the existing simple prefix.
bool appendSimple(const std::vector<Point> &path,Point next) {
    if (path.empty()) return true;
    for (const auto p:path) if (samePoint(p,next)) return false;
    const auto a=path.back();
    if (path.size()>1&&orientation(path[path.size()-2],a,next)==0&&
        onSegment(a,next,path[path.size()-2])) return false;
    if (path.size()>1&&orientation(path[path.size()-2],a,next)==0&&
        onSegment(path[path.size()-2],a,next)) return false;
    for (std::size_t k=1;k+1<path.size();++k)
        if (intersects(path[k-1],path[k],a,next)) return false;
    return true;
}
using DItem=std::pair<double,Node>;
using LItem=std::tuple<double,double,Node>;
template<class T> using PVector=std::pmr::vector<T>;

class Graph {
public:
    Graph(const Terrain &terrain,const SafetyField &field,Point start,Point goal,
          const CorridorOptions &options,CorridorResult &result,Budget &budget,bool v4=false)
        :terrain_(terrain),field_(field),options_(options),result_(result),budget_(budget),
         bounds_(field.bounds()),points_(&budget),states_(&budget),start_links_(&budget),goal_links_(&budget),
         start_adj_(&budget),goal_adj_(&budget),edges_(&budget),edge_cells_(&budget),v4_(v4) {
        const auto dimension=[&](double extent) {
            const double n=std::floor(extent/options.grid_spacing_m);
            if (!std::isfinite(n) || n<0 || n>2000000) throw ResourceLimit{};
            return static_cast<Node>(n)+1;
        };
        nx_=dimension(bounds_.xmax-bounds_.xmin);ny_=dimension(bounds_.ymax-bounds_.ymin);
        const std::uint64_t count=static_cast<std::uint64_t>(nx_)*ny_;
        if (count>2000000) throw ResourceLimit{};
        // Conservative bounds for two candidate paths, cell union/risk sorting,
        // final mask, and independent std::vector scratch in surface traversal.
        budget.reserveExternal(count*128+static_cast<std::uint64_t>(field.nx())*field.ny()*40+
            (static_cast<std::uint64_t>(terrain.nx())+terrain.ny()+field.nx()+field.ny()+8)*512);
        start_=static_cast<Node>(count);goal_=start_+1;
        points_.reserve(count+2);
        for (Node j=0;j<ny_;++j) for (Node i=0;i<nx_;++i)
            points_.push_back({bounds_.xmin+i*options.grid_spacing_m,bounds_.ymin+j*options.grid_spacing_m});
        points_.push_back(start);points_.push_back(goal);
        states_.assign(count+2,-1);start_adj_.assign(count,0);goal_adj_.assign(count,0);
        endpointLinks(start,start_links_);endpointLinks(goal,goal_links_);
        for (const auto n:start_links_) start_adj_[n]=1;
        for (const auto n:goal_links_) goal_adj_[n]=1;
        result_.graph_nodes=points_.size();
    }
    void solve() {
        const auto graphBegin=Clock::now();
        if (!valid(start_)) { result_.status="INVALID_START";return; }
        if (!valid(goal_)) { result_.status="INVALID_GOAL";return; }
        if (points_[start_].x==points_[goal_].x && points_[start_].y==points_[goal_].y) {
            result_.L0=0.;result_.graph_s=elapsed(graphBegin);
            if(v4_) {addCandidate({start_},"shortest");selectV4();} else finish({start_});
            return;
        }
        PVector<double> fromStart(points_.size(),inf,&budget_),toGoal(points_.size(),inf,&budget_);
        PVector<Node> parents(points_.size(),none,&budget_);
        dijkstra(start_,fromStart,&parents);
        if (!std::isfinite(fromStart[goal_])) { result_.graph_s=elapsed(graphBegin);return; }
        result_.L0=fromStart[goal_];
        dijkstra(goal_,toGoal,nullptr);
        result_.graph_s=elapsed(graphBegin);
        std::vector<Node> shortest;
        for (Node n=goal_;n!=none;n=parents[n]) {
            shortest.push_back(n);
            if (shortest.size()>points_.size()) throw std::logic_error("cyclic shortest guide");
        }
        std::reverse(shortest.begin(),shortest.end());
        // Re-sum the selected shortest route with the same long-double path
        // accumulation used for guide validation. This retains L0 at alpha=1
        // without introducing a physical-width or risk tolerance.
        std::vector<Point> shortestPoints;shortestPoints.reserve(shortest.size());
        for (const auto n:shortest) shortestPoints.push_back(points_[n]);
        const auto shortestStats=corridorPathStats(terrain_,field_,shortestPoints);
        if (!shortestStats.length) throw std::logic_error("shortest guide lacks length");
        result_.L0=shortestStats.length;
        choose(shortest);
        const auto guidanceBegin=Clock::now();
        pareto(fromStart,toGoal);
        if(v4_) {
            addCandidate(shortest,"shortest");addCandidate(best_,"v3");
            beam(fromStart,toGoal);
        }
        result_.guidance_s=elapsed(guidanceBegin);
        if(v4_)selectV4();else finish(best_);
    }
private:
    void endpointLinks(Point p,PVector<Node> &links) {
        if (!std::isfinite(p.x)||!std::isfinite(p.y)||p.x<bounds_.xmin||p.x>bounds_.xmax||p.y<bounds_.ymin||p.y>bounds_.ymax) return;
        const double h=options_.grid_spacing_m,r=options_.endpoint_connection_radius_m;
        const int imin=std::max(0,static_cast<int>(std::floor((p.x-bounds_.xmin-r)/h)));
        const int imax=std::min(static_cast<int>(nx_)-1,static_cast<int>(std::ceil((p.x-bounds_.xmin+r)/h)));
        const int jmin=std::max(0,static_cast<int>(std::floor((p.y-bounds_.ymin-r)/h)));
        const int jmax=std::min(static_cast<int>(ny_)-1,static_cast<int>(std::ceil((p.y-bounds_.ymin+r)/h)));
        for (int j=jmin;j<=jmax;++j) for (int i=imin;i<=imax;++i) {
            const Node n=j*nx_+i;const auto q=points_[n];
            const double scale=std::max({1.,std::abs(p.x),std::abs(p.y),std::abs(q.x),std::abs(q.y),r});
            const double tie=16.*(std::nextafter(scale,inf)-scale);
            if (distanceXY(p,q)<=r+tie) links.push_back(n);
        }
    }
    template<class F> void neighbours(Node n,F fn) {
        if (n==start_ || n==goal_) {
            for (const auto other:n==start_?start_links_:goal_links_) fn(other);
            fn(n==start_?goal_:start_);return;
        }
        const int i=n%nx_,j=n/nx_;
        for (int dj=-1;dj<=1;++dj) for (int di=-1;di<=1;++di)
            if ((di||dj)&&i+di>=0&&j+dj>=0&&i+di<static_cast<int>(nx_)&&j+dj<static_cast<int>(ny_))
                fn(static_cast<Node>(j+dj)*nx_+i+di);
        if (start_adj_[n]) fn(start_);
        if (goal_adj_[n]) fn(goal_);
    }
    bool valid(Node n) {
        if (states_[n]>=0) return states_[n]!=0;
        ++result_.state_evaluations;states_[n]=0;
        const auto p=points_[n];const auto summary=field_.edgeSummary(p,p);
        if (summary.supported&&summary.maximum_risk&&*summary.maximum_risk<=40.&&terrain_.validState(p)) {
            const auto slope=terrain_.maximumSweptSlope(p,p);
            if (std::isfinite(slope)&&slope<=35.) states_[n]=1;
        }
        return states_[n]!=0;
    }
    Edge edge(Node a,Node b,bool exposure) {
        if (a>b) std::swap(a,b);
        const auto key=(static_cast<std::uint64_t>(a)<<32)|b;
        auto found=edges_.find(key);
        if (found==edges_.end()) {
            ++result_.edge_evaluations;Edge value;
            if (valid(a)&&valid(b)&&field_.edgeRejectionReason(points_[a],points_[b],40.).empty()&&
                terrain_.validMotion(points_[a],points_[b])) {
                const auto slope=terrain_.maximumSweptSlope(points_[a],points_[b]);
                if (std::isfinite(slope)&&slope<=35.) value.length=terrain_.surfaceCost(points_[a],points_[b]);
            }
            found=edges_.emplace(key,value).first;
        } else ++result_.edge_cache_hits;
        if (exposure && std::isfinite(found->second.length) && !std::isfinite(found->second.exposure)) {
            const auto integral=field_.edgeExposure(terrain_,points_[a],points_[b]);
            if (!integral) throw std::logic_error("valid graph edge lacks risk exposure");
            found->second.exposure=*integral;
        }
        return found->second;
    }
    void dijkstra(Node source,PVector<double> &distance,PVector<Node> *parent) {
        std::priority_queue<DItem,PVector<DItem>,std::greater<DItem>> queue(std::greater<DItem>{},PVector<DItem>(&budget_));
        double limit=result_.L0?options_.alpha**result_.L0:inf;
        const Node target=source==start_?goal_:start_;
        distance[source]=0.;queue.push({0.,source});
        while (!queue.empty()) {
            const auto [cost,n]=queue.top();queue.pop();
            if (cost!=distance[n]) continue;
            if (cost>limit) break;
            if (source==start_&&n==goal_) limit=options_.alpha*cost;
            // Euclidean XY is only an admissible envelope filter; edge costs
            // and all distances retained inside it remain surface lengths.
            if (cost+distanceXY(points_[n],points_[target])>std::nextafter(limit,inf)) continue;
            neighbours(n,[&](Node other) {
                const auto e=edge(n,other,false);const double next=cost+e.length;
                if (next<distance[other]) {
                    distance[other]=next;if (parent) (*parent)[other]=n;
                    queue.push({next,other});
                }
            });
        }
    }
    void choose(const std::vector<Node> &nodes) {
        std::vector<Node> compact;
        for (const auto n:nodes) if (compact.empty() || points_[compact.back()].x!=points_[n].x || points_[compact.back()].y!=points_[n].y)
            compact.push_back(n);
        std::vector<Node> ordered=compact;std::sort(ordered.begin(),ordered.end());
        if (std::adjacent_find(ordered.begin(),ordered.end())!=ordered.end()) return;
        std::vector<Point> path;path.reserve(compact.size());
        for (const auto n:compact) path.push_back(points_[n]);
        auto pointOrder=path;
        std::sort(pointOrder.begin(),pointOrder.end(),[](Point a,Point b){return std::tie(a.x,a.y)<std::tie(b.x,b.y);});
        if (std::adjacent_find(pointOrder.begin(),pointOrder.end(),[](Point a,Point b){return a.x==b.x&&a.y==b.y;})!=pointOrder.end()) return;
        const auto stats=corridorPathStats(terrain_,field_,path);
        if (!stats.supported||!stats.length||!stats.pri) return;
        if (*stats.length>options_.alpha**result_.L0) return;
        if (best_.empty() || *stats.pri<best_pri_ || (*stats.pri==best_pri_&&
            (*stats.length<best_length_||(*stats.length==best_length_&&compact<best_)))) {
            best_=compact;best_pri_=*stats.pri;best_length_=*stats.length;
        }
    }
    void pareto(const PVector<double> &fromStart,const PVector<double> &toGoal) {
        const double limit=options_.alpha**result_.L0,slack=(options_.alpha-1.)**result_.L0;
        PVector<Node> bins(points_.size()*options_.label_bins,none,&budget_);
        PVector<Label> labels(&budget_);
        std::priority_queue<LItem,PVector<LItem>,std::greater<LItem>> queue(std::greater<LItem>{},PVector<LItem>(&budget_));
        const auto insert=[&](Node node,double length,double exposure,Node parent) {
            if (!std::isfinite(toGoal[node])||length+toGoal[node]>limit) return;
            auto first=bins.begin()+static_cast<std::size_t>(node)*options_.label_bins;
            for (unsigned b=0;b<options_.label_bins;++b) if (first[b]!=none) {
                const auto &old=labels[first[b]];
                if (old.length<=length&&old.exposure<=exposure) return;
            }
            const double used=std::max(0.,length-fromStart[node]);
            const auto b=slack>0?std::min(options_.label_bins-1,static_cast<unsigned>(std::floor(used/slack*options_.label_bins))):0;
            if (first[b]!=none) {
                const auto &old=labels[first[b]];
                if (old.exposure<exposure||(old.exposure==exposure&&old.length<=length)) return;
            }
            if (labels.size()>=options_.max_labels) throw ResourceLimit{};
            // Retired labels remain immutable: descendants still reference them.
            for (unsigned k=0;k<options_.label_bins;++k) if (first[k]!=none) {
                auto &old=labels[first[k]];
                if (k==b||(length<=old.length&&exposure<=old.exposure)) { old.active=false;first[k]=none; }
            }
            const Node id=static_cast<Node>(labels.size());
            labels.push_back({length,exposure,node,parent,true});first[b]=id;
            ++result_.labels_allocated;queue.push({exposure,length,id});
        };
        insert(start_,0.,0.,none);
        while (!queue.empty()) {
            const auto [exposure,length,id]=queue.top();queue.pop();(void)exposure;(void)length;
            const auto current=labels[id];if (!current.active) continue;
            if (current.node==goal_) {
                std::vector<Node> route;
                for (Node at=id;at!=none;at=labels[at].parent) {
                    route.push_back(labels[at].node);
                    if (route.size()>points_.size()) break;
                }
                std::reverse(route.begin(),route.end());choose(route);continue;
            }
            neighbours(current.node,[&](Node other) {
                // Do not spend expensive exposure integration outside the length envelope.
                const auto e0=edge(current.node,other,false);
                if (!std::isfinite(e0.length)||current.length+e0.length+toGoal[other]>limit) return;
                const auto e=edge(current.node,other,true);
                insert(other,current.length+e.length,current.exposure+e.exposure,id);
            });
        }
    }
    const PVector<std::size_t> &edgeCells(Node a,Node b) {
        if(a>b)std::swap(a,b);
        const auto key=(static_cast<std::uint64_t>(a)<<32)|b;
        auto found=edge_cells_.find(key);
        if(found==edge_cells_.end()) {
            const auto cells=field_.touchedCells(points_[a],points_[b]);
            PVector<std::size_t> stored(cells.begin(),cells.end(),&budget_);
            found=edge_cells_.emplace(key,std::move(stored)).first;
        }
        return found->second;
    }
    std::vector<Node> compactRoute(const std::vector<Node> &nodes) const {
        std::vector<Node> out;out.reserve(nodes.size());
        for(const auto n:nodes)if(out.empty()||!samePoint(points_[out.back()],points_[n]))out.push_back(n);
        return out;
    }
    void addCandidate(const std::vector<Node> &nodes,const char *source) {
        const auto compact=compactRoute(nodes);if(compact.empty())return;
        for(const auto &c:result_.candidate_diagnostics)if(c.raw_node_ids==compact)return;
        std::vector<Point> path;path.reserve(compact.size());for(const auto n:compact)path.push_back(points_[n]);
        if(!corridorSimplePath(path)){++result_.non_simple_rejected;return;}
        auto stats=corridorPathStats(terrain_,field_,path);
        if(!stats.supported||!stats.length||!stats.pri||*stats.length>options_.alpha**result_.L0)return;
        // Result-owned standard containers are charged before retaining them.
        budget_.reserveExternal(compact.size()*(sizeof(Node)+sizeof(Point))+1024);
        stats.cells.clear();stats.cells.shrink_to_fit();
        CorridorCandidate candidate;candidate.source=source;candidate.raw_node_ids=compact;
        candidate.raw=std::move(path);candidate.raw_stats=std::move(stats);
        result_.candidate_diagnostics.push_back(std::move(candidate));
    }
    void beam(const PVector<double> &fromStart,const PVector<double> &toGoal) {
        const double limit=options_.alpha**result_.L0,slack=(options_.alpha-1.)**result_.L0;
        constexpr unsigned roles=4;
        const bool stable=options_.beam_scheduler=="length_first_stable";
        PVector<Node> slots(points_.size()*options_.label_bins*roles,none,&budget_);
        PVector<BeamLabel> labels(&budget_);
        PVector<std::uint32_t> marks(static_cast<std::size_t>(field_.nx())*field_.ny(),0,&budget_);
        std::uint32_t generation=0;
        std::priority_queue<LItem,PVector<LItem>,std::greater<LItem>> queue(std::greater<LItem>{},PVector<LItem>(&budget_));
        const auto pri=[](const BeamLabel &a){return a.risk_count?static_cast<double>(a.risk_sum/a.risk_count):inf;};
        const auto route=[&](Node id) {
            std::vector<Node> path;
            for(Node at=id;at!=none;at=labels[at].parent)path.push_back(labels[at].node);
            std::reverse(path.begin(),path.end());return path;
        };
        const auto better=[&](const BeamLabel &a,const BeamLabel &b,unsigned role) {
            const double av=role==0?pri(a):role==1?a.exposure:role==2?a.length:a.maximum;
            const double bv=role==0?pri(b):role==1?b.exposure:role==2?b.length:b.maximum;
            if(av!=bv)return av<bv;
            const auto ak=std::make_tuple(pri(a),a.exposure,a.length,a.maximum);
            const auto bk=std::make_tuple(pri(b),b.exposure,b.length,b.maximum);
            if(ak!=bk)return ak<bk;
            if(stable) {
                ++result_.beam_equal_score_retains;
                return false;
            }
            auto ap=route(a.parent),bp=route(b.parent);ap.push_back(a.node);bp.push_back(b.node);
            return ap<bp;
        };
        const auto insert=[&](BeamLabel candidate) {
            if(!std::isfinite(toGoal[candidate.node])||candidate.length+toGoal[candidate.node]>limit)return;
            const double used=std::max(0.,candidate.length-fromStart[candidate.node]);
            const auto bin=slack>0?std::min(options_.label_bins-1,static_cast<unsigned>(std::floor(used/slack*options_.label_bins))):0;
            auto first=slots.begin()+(static_cast<std::size_t>(candidate.node)*options_.label_bins+bin)*roles;
            bool selected=false, wins[roles];
            for(unsigned role=0;role<roles;++role) {
                wins[role]=first[role]==none||better(candidate,labels[first[role]],role);
                selected=selected||wins[role];
            }
            if(!selected){++result_.beam_rejected;return;}
            if(result_.labels_allocated>=options_.max_labels)throw ResourceLimit{};
            const Node id=static_cast<Node>(labels.size());
            labels.push_back(candidate);
            ++result_.labels_allocated;
            ++result_.beam_labels_allocated;
            bool replaced=false;
            Node old[roles];for(unsigned role=0;role<roles;++role) {
                old[role]=first[role];
                if(wins[role]) {
                    if(old[role]!=none) {++result_.beam_role_replacements;replaced=true;}
                    first[role]=id;
                }
            }
            if(replaced)++result_.beam_replacement_labels;
            for(const auto prior:old)if(prior!=none&&labels[prior].active&&
                std::find(first,first+roles,prior)==first+roles) {
                labels[prior].active=false;
                ++result_.beam_retired;
            }
            if(stable)queue.push({candidate.length,pri(candidate),id});
            else queue.push({pri(candidate),candidate.length,id});
        };
        long double initialSum=0;double initialMax=0;
        const auto &initialCells=edgeCells(start_,start_);
        for(const auto cell:initialCells){const auto risk=field_.risk(cell%field_.nx(),cell/field_.nx());initialSum+=risk;initialMax=std::max(initialMax,risk);}
        insert({0.,0.,initialMax,initialSum,initialCells.size(),start_,none,true});
        while(!queue.empty()) {
            const Node id=std::get<2>(queue.top());queue.pop();const auto current=labels[id];
            if(!current.active){++result_.beam_stale_pops;continue;}
            if(current.node==goal_)continue;
            ++result_.beam_expansions;
            if(++generation==0){std::fill(marks.begin(),marks.end(),0);++generation;}
            auto nodes=route(id);std::vector<Point> path;path.reserve(nodes.size());
            for(const auto n:nodes)if(path.empty()||!samePoint(path.back(),points_[n]))path.push_back(points_[n]);
            for(Node at=id;at!=none;at=labels[at].parent) {
                const auto &label=labels[at];const auto parent=label.parent==none?label.node:labels[label.parent].node;
                for(const auto cell:edgeCells(parent,label.node))marks[cell]=generation;
            }
            neighbours(current.node,[&](Node other) {
                const auto e0=edge(current.node,other,false);
                if(!std::isfinite(e0.length)||current.length+e0.length+toGoal[other]>limit)return;
                // Endpoint/grid aliases are zero edges; compact them while
                // retaining the true query endpoint as the terminal node.
                const bool alias=samePoint(points_[current.node],points_[other]);
                if(alias&&other!=goal_&&current.node!=start_)return;
                if(!alias&&!appendSimple(path,points_[other])){++result_.non_simple_rejected;return;}
                for(const auto n:nodes)if(n==other)return;
                auto sum=current.risk_sum;auto count=current.risk_count;auto maximum=current.maximum;
                for(const auto cell:edgeCells(current.node,other))if(marks[cell]!=generation) {
                    const auto risk=field_.risk(cell%field_.nx(),cell/field_.nx());sum+=risk;++count;maximum=std::max(maximum,risk);
                }
                const auto e=edge(current.node,other,true);
                insert({current.length+e.length,current.exposure+e.exposure,maximum,sum,count,other,id,true});
            });
        }
        std::vector<Node> terminal;
        auto first=slots.begin()+static_cast<std::size_t>(goal_)*options_.label_bins*roles;
        for(unsigned k=0;k<options_.label_bins*roles;++k)if(first[k]!=none&&
            std::find(terminal.begin(),terminal.end(),first[k])==terminal.end())terminal.push_back(first[k]);
        if(terminal.size()>options_.max_preview_candidates)throw ResourceLimit{};
        // Deterministic full-path node order, independent of queue retirement.
        std::sort(terminal.begin(),terminal.end(),[&](Node a,Node b){return route(a)<route(b);});
        for(const auto id:terminal)addCandidate(route(id),"beam");
    }
    std::vector<int> makeMask(const std::vector<Point> &path,unsigned dilation) const {
        std::vector<int> mask(static_cast<std::size_t>(field_.nx())*field_.ny(),0);
        const auto stats=corridorPathStats(terrain_,field_,path);const int d=static_cast<int>(dilation);
        for(const auto cell:stats.cells) {
            const int i=cell%field_.nx(),j=cell/field_.nx();
            for(int y=std::max(0,j-d);y<=std::min(field_.ny()-1,j+d);++y)
                for(int x=std::max(0,i-d);x<=std::min(field_.nx()-1,i+d);++x)mask[static_cast<std::size_t>(y)*field_.nx()+x]=1;
        }
        return mask;
    }
    bool chordValid(Point a,Point b,const std::vector<int> &mask) const {
        if(!field_.edgeRejectionReason(a,b,40.).empty())return false;
        for(const auto cell:field_.touchedCells(a,b))if(!mask[cell])return false;
        if(!terrain_.validMotion(a,b))return false;
        const auto slope=terrain_.maximumSweptSlope(a,b);return std::isfinite(slope)&&slope<=35.;
    }
    std::vector<Point> preview(const std::vector<Point> &raw) {
        if(raw.size()<3)return raw;
        const auto mask=makeMask(raw,1);
        std::vector<long double> prefix(raw.size(),0.);
        for(std::size_t k=1;k<raw.size();++k)prefix[k]=prefix[k-1]+terrain_.surfaceCost(raw[k-1],raw[k]);
        std::vector<Point> out{raw.front()};std::size_t i=0;
        while(i+1<raw.size()) {
            std::size_t chosen=i+1;
            for(std::size_t j=raw.size()-1;j>i+1;--j) {
                if(result_.preview_edge_checks>=options_.max_preview_edge_checks)throw ResourceLimit{};
                ++result_.preview_edge_checks;
                if(!chordValid(raw[i],raw[j],mask))continue;
                const double oldLength=static_cast<double>(prefix[j]-prefix[i]);
                const double newLength=terrain_.surfaceCost(raw[i],raw[j]);
                const double scale=std::max({1.,std::abs(oldLength),std::abs(newLength)});
                const double tie=64.*(std::nextafter(scale,inf)-scale);
                if(!(oldLength-newLength>tie))continue;
                auto replacement=out;replacement.insert(replacement.end(),raw.begin()+j,raw.end());
                if(!corridorSimplePath(replacement))continue;
                chosen=j;break;
            }
            out.push_back(raw[chosen]);i=chosen;
        }
        return out;
    }
    void selectV4() {
        if(result_.candidate_diagnostics.empty())return;
        const auto began=Clock::now();std::size_t best=0;
        for(std::size_t i=0;i<result_.candidate_diagnostics.size();++i) {
            auto &candidate=result_.candidate_diagnostics[i];
            if(options_.selector=="preview_pri") {
                const auto before=result_.preview_edge_checks;
                budget_.reserveExternal(candidate.raw.size()*sizeof(Point)+1024);
                candidate.preview=preview(candidate.raw);
                candidate.preview_edge_checks=result_.preview_edge_checks-before;
                candidate.preview_stats=corridorPathStats(terrain_,field_,candidate.preview);
                candidate.preview_stats.cells.clear();candidate.preview_stats.cells.shrink_to_fit();
            }
            const auto &stats=options_.selector=="preview_pri"?candidate.preview_stats:candidate.raw_stats;
            const auto &previous=result_.candidate_diagnostics[best];
            const auto &old=options_.selector=="preview_pri"?previous.preview_stats:previous.raw_stats;
            if(!stats.supported||!stats.pri||!stats.length)throw std::logic_error("candidate preview lost support");
            if(i&&std::make_tuple(*stats.pri,*stats.length,candidate.raw_node_ids)<
                  std::make_tuple(*old.pri,*old.length,previous.raw_node_ids))best=i;
        }
        result_.preview_s=elapsed(began);auto &selected=result_.candidate_diagnostics[best];selected.selected=true;
        budget_.reserveExternal((selected.raw.size()+selected.preview.size()*2)*sizeof(Point)+4096);
        result_.raw_guide=selected.raw;result_.preview=selected.preview;
        result_.raw_guide_stats=selected.raw_stats;result_.preview_stats=selected.preview_stats;
        const auto &chosen=options_.selector=="preview_pri"?selected.preview:selected.raw;
        const auto maskBegin=Clock::now();result_.guide=chosen;
        const auto stats=corridorPathStats(terrain_,field_,chosen);
        result_.guide_length=stats.length;result_.guide_pri=stats.pri;result_.guide_exposure=stats.exposure;
        result_.risk_sum=stats.risk_sum;result_.risk_count=stats.risk_count;result_.risk_quantiles=stats.risk_quantiles;
        result_.mask=makeMask(chosen,options_.dilation_cells);
        for(std::size_t k=1;k<chosen.size();++k)if(!chordValid(chosen[k-1],chosen[k],result_.mask))
            throw std::logic_error("v4 final guide escaped immutable mask");
        result_.mask_s=elapsed(maskBegin);result_.status="GUIDANCE_READY";
    }
    void finish(const std::vector<Node> &nodes) {
        if (nodes.empty()) return;
        const auto begin=Clock::now();result_.guide.reserve(nodes.size());
        for (const auto n:nodes) result_.guide.push_back(points_[n]);
        const auto stats=corridorPathStats(terrain_,field_,result_.guide);
        if (!stats.supported) throw std::logic_error("selected guide lost support");
        result_.guide_length=stats.length;result_.guide_pri=stats.pri;result_.guide_exposure=stats.exposure;
        result_.risk_sum=stats.risk_sum;result_.risk_count=stats.risk_count;result_.risk_quantiles=stats.risk_quantiles;
        result_.mask.assign(static_cast<std::size_t>(field_.nx())*field_.ny(),0);
        const int d=static_cast<int>(options_.dilation_cells);
        for (const auto cell:stats.cells) {
            const int i=cell%field_.nx(),j=cell/field_.nx();
            for (int y=std::max(0,j-d);y<=std::min(field_.ny()-1,j+d);++y)
                for (int x=std::max(0,i-d);x<=std::min(field_.nx()-1,i+d);++x)
                    result_.mask[static_cast<std::size_t>(y)*field_.nx()+x]=1;
        }
        result_.mask_s=elapsed(begin);result_.status="GUIDANCE_READY";
    }
    const Terrain &terrain_;const SafetyField &field_;const CorridorOptions &options_;CorridorResult &result_;Budget &budget_;
    Rect bounds_;Node nx_=0,ny_=0,start_=0,goal_=0;
    PVector<Point> points_;PVector<signed char> states_;PVector<Node> start_links_,goal_links_;
    PVector<unsigned char> start_adj_,goal_adj_;std::pmr::unordered_map<std::uint64_t,Edge> edges_;
    std::pmr::unordered_map<std::uint64_t,PVector<std::size_t>> edge_cells_;
    bool v4_=false;
    std::vector<Node> best_;double best_pri_=inf,best_length_=inf;
};
}

bool corridorSimplePath(const std::vector<Point> &path) {
    std::vector<Point> prefix;prefix.reserve(path.size());
    for(const auto point:path) {
        if(!std::isfinite(point.x)||!std::isfinite(point.y)||!appendSimple(prefix,point))return false;
        prefix.push_back(point);
    }
    return !path.empty();
}

CorridorPathStats corridorPathStats(const Terrain &terrain,const SafetyField &field,const std::vector<Point> &path) {
    CorridorPathStats out;if (path.empty()) return out;
    std::vector<unsigned char> visited(static_cast<std::size_t>(field.nx())*field.ny(),0);
    long double length=0,exposure=0,sum=0;
    const std::size_t edges=path.size()>1?path.size()-1:1;
    for (std::size_t k=0;k<edges;++k) {
        const Point a=path[k],b=path.size()>1?path[k+1]:a;
        const auto cells=field.touchedCells(a,b);if (cells.empty()) return out;
        for (const auto cell:cells) {
            const int i=cell%field.nx(),j=cell/field.nx();
            if (!field.supported(i,j)) return out;
            visited[cell]=1;
        }
        const double cost=terrain.surfaceCost(a,b);const auto integral=field.edgeExposure(terrain,a,b);
        if (!std::isfinite(cost)||!integral) return out;
        length+=cost;exposure+=*integral;
    }
    std::vector<double> values;
    for (std::size_t k=0;k<visited.size();++k) if (visited[k]) {
        out.cells.push_back(k);const auto risk=field.risk(k%field.nx(),k/field.nx());
        values.push_back(risk);sum+=risk;
    }
    if (values.empty()) return out;
    std::sort(values.begin(),values.end());
    for (unsigned k=0;k<=4;++k) {
        const double index=(values.size()-1)*(k/4.);const auto lo=static_cast<std::size_t>(std::floor(index));
        const auto hi=static_cast<std::size_t>(std::ceil(index));
        out.risk_quantiles.push_back(values[lo]+(values[hi]-values[lo])*(index-lo));
    }
    out.supported=true;out.length=static_cast<double>(length);out.exposure=static_cast<double>(exposure);
    out.risk_sum=static_cast<double>(sum);out.risk_count=values.size();out.pri=static_cast<double>(sum/values.size());return out;
}

static CorridorResult compute(const Terrain &terrain,const SafetyField &field,Point start,Point goal,const CorridorOptions &options,bool v4) {
    if (!std::isfinite(options.alpha)||options.alpha<1.||!options.label_bins||options.label_bins>1024||
        options.dilation_cells>100000||!options.max_labels||options.max_labels>=none||!options.max_memory_bytes||
        !std::isfinite(options.grid_spacing_m)||options.grid_spacing_m<=0||
        !std::isfinite(options.endpoint_connection_radius_m)||options.endpoint_connection_radius_m<0)
        throw std::invalid_argument("invalid corridor options");
    if(v4&&(options.label_bins!=16||options.representatives_per_bin!=4||options.max_labels>2000000||
       (options.beam_scheduler!="legacy_pri_first"&&options.beam_scheduler!="length_first_stable")||
       options.max_preview_candidates!=64||!options.max_preview_edge_checks||options.max_preview_edge_checks>2000000||
       (options.selector!="raw_pri"&&options.selector!="preview_pri")||options.dilation_cells>1||
       (options.selector=="raw_pri"&&options.dilation_cells!=1)))
        throw std::invalid_argument("invalid v4 corridor options");
    const auto a=terrain.bounds(),b=field.bounds();
    if (a.xmin!=b.xmin||a.ymin!=b.ymin||a.xmax!=b.xmax||a.ymax!=b.ymax)
        throw std::invalid_argument("corridor field/terrain bounds mismatch");
    const auto begin=Clock::now();CorridorResult result;Budget budget(options.max_memory_bytes);
    double construction=0.;
    const auto failed=[&]() {
        const double spent=elapsed(begin);
        if (result.graph_s==0.) result.graph_s=spent;
        else {
            result.graph_s+=construction;
            if (result.guidance_s==0.) result.guidance_s=std::max(0.,spent-result.graph_s);
            else result.mask_s=std::max(0.,spent-result.graph_s-result.guidance_s);
        }
        result.status="GUIDANCE_RESOURCE_LIMIT";result.guide.clear();result.mask.clear();
        result.raw_guide.clear();result.preview.clear();
    };
    try { Graph graph(terrain,field,start,goal,options,result,budget,v4);construction=elapsed(begin);graph.solve();result.graph_s+=construction; }
    catch (const ResourceLimit &) { failed(); }
    catch (const std::bad_alloc &) { failed(); }
    result.peak_memory_bytes=budget.peak();result.total_s=elapsed(begin);return result;
}
CorridorResult computeCorridor(const Terrain &terrain,const SafetyField &field,Point start,Point goal,const CorridorOptions &options) {
    return compute(terrain,field,start,goal,options,false);
}
CorridorResult computeCorridorV4(const Terrain &terrain,const SafetyField &field,Point start,Point goal,const CorridorOptions &options) {
    return compute(terrain,field,start,goal,options,true);
}
} // namespace terrain25d
