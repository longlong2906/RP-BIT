#include "terrain25d/PriSegmentSearch.hpp"
#include "terrain25d/Corridor.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>

namespace terrain25d {
namespace {
bool same(Point a,Point b){return a.x==b.x&&a.y==b.y;}
std::uint64_t bits(double x){std::uint64_t b;std::memcpy(&b,&x,sizeof b);return b;}
using Key=std::array<std::uint64_t,4>;
struct Edge {bool valid=false;double length=0.;std::vector<std::size_t>cells;};
struct Label {
    std::uint64_t id=0;
    std::vector<std::size_t>nodes,cells;
    long double length=0.,risk=0.;
};
double separation(const std::vector<std::size_t>&a,const std::vector<std::size_t>&b){
    std::size_t i=0,j=0,common=0;
    while(i<a.size()&&j<b.size()){
        if(a[i]<b[j])++i;else if(b[j]<a[i])++j;else{++common;++i;++j;}
    }
    const auto count=a.size()+b.size()-common;
    return count?1.-static_cast<double>(common)/count:0.;
}
class SegmentSearch {
public:
    SegmentSearch(const Terrain&t,const SafetyPolicy&p,const std::vector<Point>&path,
                  std::size_t first,std::size_t last,const PriSegmentSearchOptions&o)
        :terrain(t),policy(p),field(*p.field()),parent(path),options(o),rng(o.seed){
        out.first=first;out.last=last;out.pri_cap=o.pri_cap;out.length_cap=o.length_cap;
    }
    bool cancelled(){
        if(stopped)return true;
        ++out.counters.cancel_checks;
        if(options.cancel&&options.cancel()){stopped=true;return true;}
        return false;
    }
    bool softStop(){
        if(options.stop&&options.stop()){
            ++out.counters.soft_stops;out.termination="SLICE_STOP";return true;
        }
        return false;
    }
    PriSegmentSearchResult finish(const std::string&status){
        if(cancelled()){
            out.status="CANCELLED";out.termination="HARD_CANCEL";out.path.clear();out.replacement.clear();out.stats=PriPathStats{};
        }else out.status=out.path.empty()?status:"FOUND";
        return std::move(out);
    }
    // Bounds are only a conservative rejection filter. Final caps below are
    // strict, without tolerance; rounding slack here cannot publish a long path.
    bool aboveLength(long double bound)const{
        const long double cap=out.path.empty()?options.length_cap:
            std::min(options.length_cap,out.stats.length);
        const long double slack=64.*std::numeric_limits<double>::epsilon()*(1.+std::abs(bound)+std::abs(cap));
        return bound>cap+slack;
    }
    const Edge*edge(Point a,Point b){
        if(cancelled())return nullptr;
        const Key key{{bits(a.x),bits(a.y),bits(b.x),bits(b.y)}};
        const auto found=cache.find(key);
        if(found!=cache.end()){++out.counters.edge_cache_hits;return &found->second;}
        ++out.counters.edge_evaluations;Edge e;
        const auto reason=policy.rejectionReason(a,b);
        if(cancelled())return nullptr;
        if(reason.empty()){
            e.cells=field.touchedCells(a,b);
            if(cancelled())return nullptr;
            e.length=terrain.surfaceCost(a,b);
            if(cancelled())return nullptr;
            e.valid=!e.cells.empty()&&std::isfinite(e.length);
        }
        if(!e.valid)++out.counters.invalid_edges;
        return &cache.emplace(key,std::move(e)).first->second;
    }
    long double sumRisk(const std::vector<std::size_t>&cells)const{
        long double risk=0.;
        for(auto cell:cells)risk+=field.risk(static_cast<int>(cell%field.nx()),static_cast<int>(cell/field.nx()));
        return risk;
    }
    std::vector<std::size_t>unite(const std::vector<std::size_t>&a,const std::vector<std::size_t>&b)const{
        std::vector<std::size_t>result;result.reserve(a.size()+b.size());
        std::set_union(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(result));return result;
    }
    std::vector<Point>replacement(const Label&label,bool goal=false)const{
        std::vector<Point>path;path.reserve(label.nodes.size()+static_cast<unsigned>(goal));
        for(auto node:label.nodes)path.push_back(nodes[node]);
        if(goal)path.push_back(nodes[1]);return path;
    }
    std::vector<Point>fullPath(const std::vector<Point>&middle,bool suffix=true)const{
        std::vector<Point>path;path.reserve(parent.size()+middle.size());
        path.insert(path.end(),parent.begin(),parent.begin()+out.first);
        path.insert(path.end(),middle.begin(),middle.end());
        if(suffix)path.insert(path.end(),parent.begin()+out.last+1,parent.end());
        return path;
    }
    bool addNode(Point q){
        ++out.counters.nodes_proposed;
        if(nodes.size()>=options.max_nodes||!std::isfinite(q.x)||!std::isfinite(q.y)||cancelled())return false;
        for(auto p:nodes)if(same(p,q))return false;
        const auto a=terrain.lowerBound(nodes[0],q,true),b=terrain.lowerBound(q,nodes[1],true);
        if(!std::isfinite(a)||!std::isfinite(b)||aboveLength(outside_length+a+b))return false;
        const bool valid=policy.validState(q);if(cancelled()||!valid)return false;
        nodes.push_back(q);++out.counters.nodes_retained;return true;
    }
    bool buildNodes(){
        nodes={parent[out.first],parent[out.last]};out.counters.nodes_retained=2;
        for(auto q:options.extra_nodes){if(cancelled()||softStop())return false;addNode(q);}
        const auto interior=out.last-out.first-1;
        // Dense parent paths must not consume every proposal node. Reserve at
        // least two thirds of the remaining slots for new mixed samples.
        // Test-only fixed graphs keep every parent knot that fits the node cap.
        const auto parentSlots=options.generate_nodes?
            std::min(interior,(options.max_nodes-nodes.size())/3):interior;
        for(std::size_t k=0;k<parentSlots;++k){
            if(cancelled()||softStop())return false;
            const auto i=out.first+1+k*interior/parentSlots;addNode(parent[i]);
        }
        if(!options.generate_nodes||nodes.size()>=options.max_nodes)return !cancelled();
        const double residual=static_cast<double>(options.length_cap-outside_length);
        const Point a=nodes[0],b=nodes[1],center{(a.x+b.x)*.5,(a.y+b.y)*.5};
        const double chord=distanceXY(a,b),major=std::max(0.,residual*.5);
        if(!(chord>0.)||major<chord*.5)return !cancelled();
        const double minor=std::sqrt(std::max(0.,major*major-chord*chord*.25));
        const Point tangent{(b.x-a.x)/chord,(b.y-a.y)/chord},normal{-tangent.y,tangent.x};
        struct Sample {Point q;double risk;std::uint64_t id;};
        std::vector<Sample>samples;const auto target=(options.max_nodes-nodes.size())*4;
        for(std::size_t i=0;i<target*8&&samples.size()<target;++i){
            if(cancelled()||softStop())return false;
            const double u=2.*std::generate_canonical<double,53>(rng)-1.,v=2.*std::generate_canonical<double,53>(rng)-1.;
            if(u*u+v*v>1.)continue;
            const Point q{center.x+major*u*tangent.x+minor*v*normal.x,center.y+major*u*tangent.y+minor*v*normal.y};
            const double lo=terrain.lowerBound(a,q,true)+terrain.lowerBound(q,b,true);
            if(!std::isfinite(lo)||aboveLength(outside_length+lo))continue;
            const auto cells=field.touchedCells(q,q);if(cells.empty())continue;
            bool supported=true;long double risk=0.;
            for(auto cell:cells){const int x=static_cast<int>(cell%field.nx()),y=static_cast<int>(cell/field.nx());
                if(!field.supported(x,y)||!std::isfinite(field.risk(x,y))||field.risk(x,y)>40.){supported=false;break;}risk+=field.risk(x,y);}
            if(supported)samples.push_back({q,static_cast<double>(risk/cells.size()),i});
        }
        // Half remain uniform XY samples. The other half prefer low risk,
        // with seeded sample order as a deterministic tie breaker.
        const auto uniformTarget=nodes.size()+(options.max_nodes-nodes.size()+1)/2;
        for(const auto&s:samples){if(nodes.size()>=uniformTarget)break;addNode(s.q);if(cancelled()||softStop())return false;}
        std::stable_sort(samples.begin(),samples.end(),[](const auto&a,const auto&b){return std::tie(a.risk,a.id)<std::tie(b.risk,b.id);});
        for(const auto&s:samples){if(nodes.size()>=options.max_nodes)break;addNode(s.q);if(cancelled()||softStop())return false;}
        return true;
    }
    void complete(const Label&label){
        const auto*last=edge(nodes[label.nodes.back()],nodes[1]);
        if(!last||!last->valid||cancelled())return;
        if(aboveLength(outside_length+label.length+last->length)){++out.counters.length_rejected;return;}
        const auto cells=unite(label.cells,last->cells);const auto risk=sumRisk(cells);
        if(cancelled())return;
        ++out.counters.completed_paths;
        if(cells.empty()||static_cast<double>(risk/cells.size())>options.pri_cap){++out.counters.pri_rejected;return;}
        auto middle=replacement(label,true),path=fullPath(middle);
        if(!corridorSimplePath(path)){++out.counters.non_simple_rejected;return;}
        if(cancelled())return;
        ++out.counters.full_recomputations;
        auto stats=priPathStats(terrain,policy,path);
        if(cancelled())return;
        if(!stats.valid||stats.cells!=cells||stats.risk_sum!=static_cast<double>(risk)||
           stats.pri!=static_cast<double>(risk/cells.size())){
            if(stats.valid)throw std::logic_error("segment whole-path union disagrees with full score");
            return;
        }
        if(stats.length>options.length_cap||stats.pri>options.pri_cap)return;
        if(!out.path.empty()&&std::tie(stats.length,stats.pri)>=std::tie(out.stats.length,out.stats.pri))return;
        out.path=std::move(path);out.replacement=std::move(middle);out.stats=std::move(stats);
        // Copy/selection costs also belong to the caller's deadline.
        cancelled();
    }
    bool retain(std::vector<Label>&bucket,Label candidate){
        const auto candidateId=candidate.id;
        for(const auto&prior:bucket)if(prior.cells!=candidate.cells){++out.counters.distinct_sets_same_node;break;}
        bucket.push_back(std::move(candidate));
        if(bucket.size()<=options.beam_width)return true;
        // Retention is deliberately heuristic, not scalar endpoint dominance.
        // Keep shortest, then lowest residual, then distinct touched-cell sets.
        std::vector<std::size_t>chosen;
        const auto shorter=[&](std::size_t a,std::size_t b){return std::tie(bucket[a].length,bucket[a].id)<std::tie(bucket[b].length,bucket[b].id);};
        std::size_t best=0;for(std::size_t i=1;i<bucket.size();++i)if(shorter(i,best))best=i;chosen.push_back(best);
        const auto residual=[&](const Label&l){return l.risk-options.pri_cap*l.cells.size();};
        while(chosen.size()<options.beam_width){
            std::size_t pick=bucket.size();double distance=-1.;
            for(std::size_t i=0;i<bucket.size();++i){
                if(std::find(chosen.begin(),chosen.end(),i)!=chosen.end())continue;
                double d=1.;for(auto prior:chosen)d=std::min(d,separation(bucket[i].cells,bucket[prior].cells));
                bool better=pick==bucket.size();
                if(!better&&chosen.size()==1)better=residual(bucket[i])<residual(bucket[pick])||
                    (residual(bucket[i])==residual(bucket[pick])&&shorter(i,pick));
                else if(!better&&chosen.size()>1)better=d>distance||(d==distance&&shorter(i,pick));
                if(better){pick=i;distance=d;}
            }
            chosen.push_back(pick);
        }
        std::vector<Label>kept;kept.reserve(chosen.size());for(auto i:chosen)kept.push_back(std::move(bucket[i]));
        out.counters.beam_discarded+=bucket.size()-kept.size();bucket=std::move(kept);
        return std::any_of(bucket.begin(),bucket.end(),[&](const Label&l){return l.id==candidateId;});
    }
    PriSegmentSearchResult run(){
        if(cancelled())return finish("CANCELLED");
        if(!corridorSimplePath(parent))return finish("INVALID_PARENT");
        std::map<std::size_t,unsigned>incidence;
        std::vector<const Edge*>parentEdges;
        for(std::size_t i=1;i<parent.size();++i){
            const auto*e=edge(parent[i-1],parent[i]);if(!e)return finish("CANCELLED");
            if(!e->valid)return finish("INVALID_PARENT");parentEdges.push_back(e);
            for(auto cell:e->cells)++incidence[cell];
            if(i-1<out.first||i-1>=out.last)outside_length+=e->length;
        }
        if(parent.size()==1){
            const auto*e=edge(parent.front(),parent.front());if(!e)return finish("CANCELLED");
            if(!e->valid)return finish("INVALID_PARENT");
        }
        if(out.first==out.last)return finish("IDENTITY_PATH");
        for(std::size_t i=out.first;i<out.last;++i)for(auto cell:parentEdges[i]->cells){
            auto found=incidence.find(cell);if(found==incidence.end()||!found->second)throw std::logic_error("segment incidence underflow");
            if(!--found->second)incidence.erase(found);
        }
        for(const auto&item:incidence)out.outside_cells.push_back(item.first);
        const double lower=terrain.lowerBound(parent[out.first],parent[out.last],true);
        if(!std::isfinite(lower)||aboveLength(outside_length+lower))return finish("LENGTH_BOUND");
        if(!buildNodes())return finish("NO_CANDIDATE");
        Label initial;initial.nodes={0};initial.cells=out.outside_cells;initial.risk=sumRisk(initial.cells);
        std::vector<Label>frontier{initial};std::uint64_t nextId=1;
        for(std::size_t depth=0;depth<options.max_depth&&!frontier.empty();++depth){
            std::vector<std::vector<Label>>next(nodes.size());
            // Scheduling is length-first with stable IDs breaking ties. No
            // risk criterion rejects an unfinished prefix.
            std::stable_sort(frontier.begin(),frontier.end(),[&](const Label&a,const Label&b){
                const long double al=a.length+terrain.lowerBound(nodes[a.nodes.back()],nodes[1],true);
                const long double bl=b.length+terrain.lowerBound(nodes[b.nodes.back()],nodes[1],true);
                return al<bl||(al==bl&&a.id<b.id);
            });
            for(const auto&label:frontier){
                if(cancelled())return finish("CANCELLED");
                if(softStop())return finish("NO_CANDIDATE");
                if(out.counters.expansions>=options.max_expansions){out.termination="EXPANSION_LIMIT";return finish("NO_CANDIDATE");}
                ++out.counters.expansions;complete(label);
                if(cancelled())return finish("CANCELLED");
                if(depth+1>=options.max_depth)continue;
                for(std::size_t node=2;node<nodes.size();++node){
                    if(cancelled())return finish("CANCELLED");
                    if(std::find(label.nodes.begin(),label.nodes.end(),node)!=label.nodes.end())continue;
                    const auto*e=edge(nodes[label.nodes.back()],nodes[node]);if(!e)return finish("CANCELLED");if(!e->valid)continue;
                    const auto bound=outside_length+label.length+e->length+terrain.lowerBound(nodes[node],nodes[1],true);
                    if(aboveLength(bound)){++out.counters.length_rejected;continue;}
                    Label child=label;child.id=nextId++;child.nodes.push_back(node);child.length+=e->length;
                    if(!corridorSimplePath(fullPath(replacement(child),false))){++out.counters.non_simple_rejected;continue;}
                    child.cells=unite(label.cells,e->cells);child.risk=sumRisk(child.cells);
                    const bool abovePri=!child.cells.empty()&&child.risk>options.pri_cap*child.cells.size();
                    ++out.counters.labels_created;
                    if(retain(next[node],std::move(child))&&abovePri)++out.counters.prefix_above_pri_kept;
                }
            }
            frontier.clear();for(auto&bucket:next)for(auto&label:bucket)frontier.push_back(std::move(label));
        }
        return finish("NO_CANDIDATE");
    }
private:
    const Terrain&terrain;const SafetyPolicy&policy;const SafetyField&field;
    const std::vector<Point>&parent;const PriSegmentSearchOptions&options;
    PriSegmentSearchResult out;std::mt19937_64 rng;bool stopped=false;
    long double outside_length=0.;std::vector<Point>nodes;std::map<Key,Edge>cache;
};
void validate(const Terrain&t,const SafetyPolicy&p,const std::vector<Point>&parent,std::size_t first,std::size_t last,
              const PriSegmentSearchOptions&o){
    if(!p.appliesTo(t)||!p.field()||t.traversabilityEnabled()||parent.empty()||first>last||last>=parent.size()||
       !std::isfinite(o.pri_cap)||o.pri_cap<0.||!std::isfinite(o.length_cap)||o.length_cap<0.||
       o.max_nodes<2||o.max_nodes>128||!o.beam_width||o.beam_width>64||!o.max_expansions||o.max_expansions>100000||
       !o.max_depth||o.max_depth>32||p.riskCap()!=40.||p.slopeCap()!=35.)
        throw std::invalid_argument("invalid constrained PRI segment search configuration");
}
} // namespace
PriSegmentSearchResult runPriSegmentSearch(const Terrain&t,const SafetyPolicy&p,const std::vector<Point>&path,
    std::size_t first,std::size_t last,const PriSegmentSearchOptions&o){
    validate(t,p,path,first,last,o);return SegmentSearch(t,p,path,first,last,o).run();
}
PriSegmentSearchResult runPriSegmentSearch(const Terrain&t,std::shared_ptr<const SafetyField>field,const std::vector<Point>&path,
    std::size_t first,std::size_t last,const PriSegmentSearchOptions&o){
    if(!field)throw std::invalid_argument("missing segment safety field");
    SafetyPolicy policy(t,std::move(field),40.,35.);return runPriSegmentSearch(t,policy,path,first,last,o);
}
} // namespace terrain25d
