#include "terrain25d/PriPathSearch.hpp"
#include "terrain25d/Corridor.hpp"
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <tuple>

namespace terrain25d {
namespace {
using Clock=std::chrono::steady_clock;
using EdgeKey=std::array<std::uint64_t,4>;
bool same(Point a,Point b){return a.x==b.x&&a.y==b.y;}
std::uint64_t bits(double value){std::uint64_t out;std::memcpy(&out,&value,sizeof out);return out;}
EdgeKey key(Point a,Point b){
    // Direction is part of the key: mathematical symmetry does not establish
    // bitwise symmetry of raster boundary arithmetic or summation order.
    return {{bits(a.x),bits(a.y),bits(b.x),bits(b.y)}};
}
struct Edge {
    bool valid=false;
    std::string reason;
    double length=0.,xy=0.;
    std::vector<std::size_t>cells;
};
struct Record {
    std::vector<Point>path;
    std::vector<std::shared_ptr<const Edge>>edges;
    std::map<std::size_t,unsigned>incidence;
    PriPathStats stats;
};
class Evaluator {
public:
    Evaluator(const Terrain&t,std::shared_ptr<const SafetyField>f,PriPathSearchCounters&c,
              std::uint64_t cacheLimit,const std::function<bool()>&cancel={})
        :terrain(t),field(std::move(f)),policy(t,field,40.,35.),counters(c),cache_limit(cacheLimit),cancel_(cancel){}
    std::shared_ptr<const Edge>edge(Point a,Point b){
        const auto k=key(a,b);const auto found=cache.find(k);
        if(found!=cache.end()){++counters.edge_cache_hits;return found->second;}
        auto e=std::make_shared<Edge>();++counters.edge_evaluations;
        if(cancel_&&cancel_()){e->reason="DEADLINE";return e;}
        e->reason=policy.rejectionReason(a,b);
        if(e->reason.empty()){
            e->cells=field->touchedCells(a,b);
            if(e->cells.empty())e->reason="missing_sem_support";
            else {e->length=terrain.surfaceCost(a,b);e->xy=distanceXY(a,b);
                if(!std::isfinite(e->length))e->reason="undefined_surface_cost";}
        }
        e->valid=e->reason.empty();
        const auto bytes=static_cast<std::uint64_t>(sizeof(Edge)+sizeof(EdgeKey)+64+e->cells.size()*sizeof(std::size_t));
        if(bytes<=cache_limit){
            if(cache_bytes+bytes>cache_limit){cache.clear();cache_bytes=0;}
            cache.emplace(k,e);cache_bytes+=bytes;
            counters.peak_edge_cache_bytes=std::max(counters.peak_edge_cache_bytes,cache_bytes);
        }
        return e;
    }
    void add(Record&r,const std::shared_ptr<const Edge>&e){
        for(const auto cell:e->cells){++r.incidence[cell];++counters.incidence_adds;}
    }
    void remove(Record&r,const std::shared_ptr<const Edge>&e){
        for(const auto cell:e->cells){const auto at=r.incidence.find(cell);
            if(at==r.incidence.end()||!at->second)throw std::logic_error("PRI incidence underflow");
            if(!--at->second)r.incidence.erase(at);++counters.incidence_removes;}
    }
    void score(Record&r){
        r.stats=PriPathStats{};
        if(r.path.empty()){r.stats.reason="empty_path";return;}
        long double length=0.,xy=0.,risk=0.;
        for(const auto&e:r.edges){if(!e->valid){r.stats.reason=e->reason;return;}length+=e->length;xy+=e->xy;}
        for(const auto&item:r.incidence){r.stats.cells.push_back(item.first);
            risk+=field->risk(static_cast<int>(item.first%field->nx()),static_cast<int>(item.first/field->nx()));}
        if(r.stats.cells.empty()){r.stats.reason="missing_sem_support";return;}
        r.stats.length=static_cast<double>(length);r.stats.length_xy=static_cast<double>(xy);
        r.stats.risk_sum=static_cast<double>(risk);r.stats.pri=static_cast<double>(risk/r.stats.cells.size());r.stats.valid=true;
    }
    Record build(const std::vector<Point>&path){
        Record r;r.path=path;
        if(!path.empty())for(std::size_t i=0;i<std::max<std::size_t>(1,path.size()-1);++i){
            auto e=edge(path[i],path.size()>1?path[i+1]:path[i]);r.edges.push_back(e);add(r,e);
            if(!e->valid)break;
        }
        score(r);return r;
    }
    Record replace(const Record&parent,std::size_t first,std::size_t last,const std::vector<Point>&middle){
        if(first>=last||last>=parent.path.size()||middle.size()<2||!same(middle.front(),parent.path[first])||!same(middle.back(),parent.path[last]))
            throw std::invalid_argument("invalid PRI path replacement endpoints");
        Record r;r.incidence=parent.incidence;
        r.path.insert(r.path.end(),parent.path.begin(),parent.path.begin()+first);
        r.path.insert(r.path.end(),middle.begin(),middle.end());
        r.path.insert(r.path.end(),parent.path.begin()+last+1,parent.path.end());
        r.edges.insert(r.edges.end(),parent.edges.begin(),parent.edges.begin()+first);
        for(std::size_t i=first;i<last;++i)remove(r,parent.edges[i]);
        for(std::size_t i=1;i<middle.size();++i){auto e=edge(middle[i-1],middle[i]);r.edges.push_back(e);add(r,e);}
        r.edges.insert(r.edges.end(),parent.edges.begin()+last,parent.edges.end());score(r);return r;
    }
    // Admission never trusts accumulated deltas. This independent full traversal
    // rebuilds the sorted unique-cell union and all lengths from the actual path.
    bool recompute(Record&r){
        ++counters.full_recomputations;const auto full=priPathStats(terrain,policy,r.path);
        if(!full.valid){r.stats=full;return false;}
        const auto near=[](double a,double b){return std::abs(a-b)<=1e-8+1e-8*std::max(std::abs(a),std::abs(b));};
        if(r.stats.cells!=full.cells||!near(r.stats.length,full.length)||!near(r.stats.length_xy,full.length_xy)||
           r.stats.risk_sum!=full.risk_sum||r.stats.pri!=full.pri)
            throw std::logic_error("incremental PRI score differs from complete recomputation");
        r.stats=full;return true;
    }
    const Terrain&terrain;
    std::shared_ptr<const SafetyField>field;
    SafetyPolicy policy;
private:
    PriPathSearchCounters&counters;
    std::uint64_t cache_limit,cache_bytes=0;
    std::function<bool()>cancel_;
    std::map<EdgeKey,std::shared_ptr<const Edge>>cache;
};
bool lengthBetter(const PriPathSearchCandidate&a,const PriPathSearchCandidate&b){
    return std::tie(a.stats.length,a.stats.pri,a.id)<std::tie(b.stats.length,b.stats.pri,b.id);
}
double jaccard(const std::vector<std::size_t>&a,const std::vector<std::size_t>&b){
    std::size_t i=0,j=0,intersection=0;
    while(i<a.size()&&j<b.size()){
        if(a[i]<b[j])++i;else if(b[j]<a[i])++j;else{++i;++j;++intersection;}
    }
    const auto count=a.size()+b.size()-intersection;return count?1.-static_cast<double>(intersection)/count:0.;
}
std::vector<double>arcs(const std::vector<Point>&path){
    std::vector<double>out(path.size(),0.);for(std::size_t i=1;i<path.size();++i)out[i]=out[i-1]+distanceXY(path[i-1],path[i]);return out;
}
Point interpolate(Point a,Point b,double t){return {(1.-t)*a.x+t*b.x,(1.-t)*a.y+t*b.y};}
struct Proposal {
    std::size_t parent=0,first=0,last=0;
    std::vector<std::size_t>parents;
    std::vector<Point>middle;
    std::string type,reason;
};
class Search {
public:
    Search(const Terrain&t,std::shared_ptr<const SafetyField>f,Point s,Point g,const PriPathSearchOptions&o)
        :terrain(t),field(std::move(f)),start(s),goal(g),options(o),began(Clock::now()),rng(o.planner_seed),
         evaluator(t,field,result.search_counters,o.max_edge_cache_bytes,[this](){return expired();}){
        result.mode=o.mode;result.anchor_deadline_s=o.budget_s*.25;
        fixed=o.test_cycles!=0;
    }
    double now()const{return std::chrono::duration<double>(Clock::now()-began).count();}
    bool expired()const{return now()>=options.budget_s||resource_stop;}
    void reject(const std::string&why){++result.search_counters.rejected[why];}
    std::vector<std::size_t>active()const{
        std::set<std::size_t>ids;for(const auto&bin:pool)ids.insert(bin.begin(),bin.end());
        if(result.anchor)ids.insert(*result.anchor);
        for(const auto&id:result.selected)if(id)ids.insert(*id);
        return {ids.begin(),ids.end()};
    }
    unsigned binFor(double length)const{
        const double lo=terrain.lowerBound(start,goal,true),hi=*result.length_caps.back();
        if(!(hi>lo))return 0;
        return static_cast<unsigned>(std::min(15.,std::max(0.,std::floor(16.*(length-lo)/(hi-lo)))));
    }
    std::vector<std::size_t>chooseBin(std::vector<std::size_t>ids,const PriPathSearchCandidate&candidate)const{
        const auto get=[&](std::size_t id)->const PriPathSearchCandidate&{return id==candidate.id?candidate:result.candidates.at(id);};
        std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return priPathSearchBetter(get(a),get(b));});
        std::vector<std::size_t>chosen;
        if(ids.empty())return chosen;chosen.push_back(ids.front());ids.erase(ids.begin());
        while(!ids.empty()&&chosen.size()<4){
            auto best=ids.begin();double bestDistance=-1.;
            for(auto at=ids.begin();at!=ids.end();++at){double d=1.;
                for(auto prior:chosen)d=std::min(d,jaccard(get(*at).stats.cells,get(prior).stats.cells));
                if(d>bestDistance||(d==bestDistance&&priPathSearchBetter(get(*at),get(*best)))){best=at;bestDistance=d;}
            }
            chosen.push_back(*best);ids.erase(best);
        }
        return chosen;
    }
    void publish(const std::array<std::optional<std::size_t>,4>&next,double when,const std::string&reason,bool force=false){
        if(when>options.budget_s)return;
        for(unsigned i=0;i<4;++i)if(next[i]&&(force||next[i]!=result.selected[i])){
            result.selected[i]=next[i];result.candidates[*next[i]].published[i]=true;
            result.publications.push_back({*next[i],i,when,reason});
        }
    }
    bool admission(Record record,PriPathSearchCandidate c,bool raw){
        if(expired()){reject("DEADLINE");return false;}
        if(!record.stats.valid){reject(record.stats.reason);return false;}
        if(record.path.empty()||!same(record.path.front(),start)||!same(record.path.back(),goal)){reject("ENDPOINT_MISMATCH");return false;}
        if(record.path.size()>2048){reject("PATH_VERTEX_LIMIT");return false;}
        if(!corridorSimplePath(record.path)){reject("NON_SIMPLE_PATH");return false;}
        if(!raw)for(auto id:active()){
            const auto&prior=result.candidates[id].path;
            if(prior.size()==record.path.size()&&std::equal(prior.begin(),prior.end(),record.path.begin(),same)){
                reject("DUPLICATE_PATH");return false;
            }
        }
        if(!raw&&result.length_caps.back()&&record.stats.length>*result.length_caps.back()){reject("EXPLORATION_LENGTH_CAP");return false;}
        c.id=result.candidates.size();c.stats=record.stats;c.path=record.path;
        // Fast path avoids a second expensive traversal for obviously unretainable proposals.
        std::vector<std::size_t>newBin;
        unsigned bin=0;
        bool output=false;
        if(result.anchor&&options.mode!="length"){
            if(c.stats.length<=*result.length_caps.back()){
                bin=binFor(c.stats.length);auto ids=pool[bin];ids.push_back(c.id);newBin=chooseBin(std::move(ids),c);
                c.pool_admitted=std::find(newBin.begin(),newBin.end(),c.id)!=newBin.end();
            }
            for(unsigned i=0;i<4;++i)if(c.stats.length<=*result.length_caps[i]&&
                (!result.selected[i]||priPathSearchBetter(c,result.candidates[*result.selected[i]])))output=true;
        }
        if(!raw&&!c.pool_admitted&&!output){reject("POOL_NOT_SELECTED");return false;}
        if(!evaluator.recompute(record)){reject(record.stats.reason);return false;}
        c.stats=record.stats;
        if(c.segment_pri_cap&&(c.stats.pri>*c.segment_pri_cap||c.stats.length>*c.segment_length_cap)){
            reject("SEGMENT_CONSTRAINT_RECHECK");return false;
        }
        if(c.move_type=="segment_contract"&&!(c.stats.length<result.candidates.at(c.parent_ids.at(0)).stats.length)){
            reject("SEGMENT_NOT_SHORTER");return false;
        }
        if(now()>options.budget_s){reject("SCORING_AFTER_DEADLINE");return false;}
        const auto bytes=static_cast<std::uint64_t>(sizeof(c)+c.path.size()*sizeof(Point)+c.stats.cells.size()*sizeof(std::size_t)+512);
        if(result.search_counters.archive_bytes+bytes>options.max_archive_bytes){
            resource_stop=true;result.optimizer_resource_limit=true;result.stage="RESOURCE_LIMIT";reject("ARCHIVE_RESOURCE_LIMIT");return false;
        }
        // Re-run decisions with fully recomputed metrics; no approximate score publishes a path.
        newBin.clear();c.pool_admitted=false;auto next=result.selected;
        if(result.anchor&&options.mode!="length"){
            if(c.stats.length<=*result.length_caps.back()){
                bin=binFor(c.stats.length);auto ids=pool[bin];ids.push_back(c.id);newBin=chooseBin(std::move(ids),c);
                c.pool_admitted=std::find(newBin.begin(),newBin.end(),c.id)!=newBin.end();
            }
            for(unsigned i=0;i<4;++i)if(c.stats.length<=*result.length_caps[i]&&
                (!result.selected[i]||priPathSearchBetter(c,result.candidates[*result.selected[i]])))next[i]=c.id;
        }else if(!result.selected[0]||lengthBetter(c,result.candidates[*result.selected[0]])){
            for(unsigned i=0;i<4;++i)next[i]=c.id;
        }
        c.reason=raw?"BITSTAR_EXACT":"OPTIMIZER_ACCEPTED";
        const auto id=c.id;records.emplace(id,std::move(record));result.candidates.push_back(std::move(c));
        // Record-copy and selection overhead belong to the algorithm clock too.
        const double completed=now();
        if(completed>options.budget_s){records.erase(id);result.candidates.pop_back();reject("SCORING_AFTER_DEADLINE");return false;}
        auto&stored=result.candidates.back();stored.score_completed_s=completed;
        if(raw&&anchor_phase&&(fixed||completed<=result.anchor_deadline_s)&&
           (!anchor_best||lengthBetter(stored,result.candidates[*anchor_best])))anchor_best=id;
        result.search_counters.archive_bytes+=bytes;++result.search_counters.accepted;
        publish(next,completed,result.anchor&&options.mode!="length"?"PRI_IMPROVEMENT":"LENGTH_FALLBACK");
        if(result.anchor&&options.mode!="length"&&!newBin.empty()){
            for(auto old:pool[bin])if(std::find(newBin.begin(),newBin.end(),old)==newBin.end())++result.search_counters.pool_evictions;
            pool[bin]=std::move(newBin);
        }
        cleanup();return true;
    }
    void cleanup(){
        // Archive owns serializable paths; detailed edge/incidence records only live for active parents.
        std::set<std::size_t>keep;const auto ids=active();keep.insert(ids.begin(),ids.end());
        if(anchor_best)keep.insert(*anchor_best);
        for(auto it=records.begin();it!=records.end();)if(!keep.count(it->first))it=records.erase(it);else ++it;
        std::uint64_t count=0;for(const auto&bin:pool)count+=bin.size();
        result.search_counters.current_pool_size=count;result.search_counters.peak_pool_size=std::max(result.search_counters.peak_pool_size,count);
    }
    void lockAnchor(){
        anchor_phase=false;
        if(!anchor_best){result.stage="NO_ANCHOR_CONTINUE";return;}
        if(expired()){result.stage="NO_ANCHOR_LOCK_BEFORE_DEADLINE";return;}
        result.anchor=anchor_best;const double length=result.candidates[*anchor_best].stats.length;
        for(unsigned i=0;i<4;++i)result.length_caps[i]=length*result.cap_ratios[i];
        const auto selected=selectPriPathSearchCandidates(result.candidates,length,options.budget_s);
        // Seed all first-phase paths, including those that were not the shortest anchor.
        for(const auto&c:result.candidates)if(c.stats.length<=*result.length_caps.back()){
            const auto b=binFor(c.stats.length);auto ids=pool[b];ids.push_back(c.id);pool[b]=chooseBin(std::move(ids),c);
        }
        // The first-phase candidates needed only their anchor/fallback records.
        // Rebuilding active edge records is charged before the lock publication.
        auto need=active();for(const auto&id:selected)if(id)need.push_back(*id);
        for(auto id:need)if(!records.count(id))records.emplace(id,evaluator.build(result.candidates[id].path));
        const double completed=now();
        if(completed>options.budget_s){
            result.anchor.reset();for(auto&cap:result.length_caps)cap.reset();for(auto&bin:pool)bin.clear();
            result.stage="NO_ANCHOR_LOCK_BEFORE_DEADLINE";return;
        }
        result.anchor_locked_s=completed;result.stage="ANCHOR_LOCKED";publish(selected,completed,"ANCHOR_LOCK",true);
        cleanup();
    }
    std::size_t draw(std::size_t n){return n?static_cast<std::size_t>(rng()%n):0;}
    Proposal proposal(){
        Proposal p;const auto ids=active();if(ids.empty()){p.reason="EMPTY_POOL";return p;}
        p.parent=ids[draw(ids.size())];p.parents={p.parent};const auto&path=result.candidates[p.parent].path;
        const auto kind=options.mode=="shortcut"?0:result.search_counters.proposals%(options.mode=="segment"?4:3);
        p.type=kind==0?"shortcut":kind==1?"deform":"splice";
        if(path.size()<2){p.reason="IDENTITY_PATH";return p;}
        if(kind==0){
            if(path.size()<3){p.reason="NO_SHORTCUT_PAIR";return p;}
            p.first=draw(path.size()-2);p.last=p.first+2+draw(path.size()-p.first-2);
            p.middle={path[p.first],path[p.last]};return p;
        }
        if(kind==1){
            const auto s=arcs(path);const double total=s.back();
            if(!(total>0.)){p.reason="ZERO_LENGTH_WINDOW";return p;}
            static constexpr double windows[]={.5,1.,2.,4.,8.,16.,-1.};
            const double width=windows[draw(7)];double center=0.;
            std::vector<std::pair<std::size_t,std::size_t>>riskLocations;
            const auto record=records.find(p.parent);
            if(record!=records.end())for(std::size_t i=0;i<record->second.edges.size();++i){
                for(auto cell:record->second.edges[i]->cells)
                    if(field->risk(static_cast<int>(cell%field->nx()),static_cast<int>(cell/field->nx()))>result.candidates[p.parent].stats.pri)
                        riskLocations.emplace_back(i,cell);
            }
            if(draw(2)&&!riskLocations.empty()){
                const auto location=riskLocations[draw(riskLocations.size())];const auto edge=location.first;
                const auto cell=field->cellBounds(static_cast<int>(location.second%field->nx()),static_cast<int>(location.second/field->nx()));
                const Point c{(cell.xmin+cell.xmax)*.5,(cell.ymin+cell.ymax)*.5},a=path[edge],b=path[edge+1];
                const double dx=b.x-a.x,dy=b.y-a.y,denominator=dx*dx+dy*dy;
                const double u=denominator>0.?std::clamp(((c.x-a.x)*dx+(c.y-a.y)*dy)/denominator,0.,1.):0.;
                center=s[edge]+u*(s[edge+1]-s[edge]);
            }else center=total*std::generate_canonical<double,53>(rng);
            const double beginArc=width<0.?0.:std::max(0.,center-width/2.);
            const double endArc=width<0.?total:std::min(total,center+width/2.);
            auto left=std::upper_bound(s.begin(),s.end(),beginArc);
            p.first=std::min(path.size()-2,left==s.begin()?0:static_cast<std::size_t>(left-s.begin()-1));
            auto right=std::lower_bound(s.begin(),s.end(),endArc);
            p.last=std::max(p.first+1,std::min(path.size()-1,static_cast<std::size_t>(right-s.begin())));
            const auto atArc=[&](double target){
                const auto exact=std::lower_bound(s.begin(),s.end(),target);
                if(exact!=s.end()&&*exact==target)return path[static_cast<std::size_t>(exact-s.begin())];
                const auto i=static_cast<std::size_t>(exact-s.begin()-1);
                return interpolate(path[i],path[i+1],(target-s[i])/(s[i+1]-s[i]));
            };
            // Physical window endpoints may lie inside existing edges. They are
            // explicit scored knots, never unverified "equivalent" resampling.
            const auto a=atArc(beginArc),b=atArc(endArc);const double chord=distanceXY(a,b);
            if(!(chord>0.)){p.reason="ZERO_CHORD";return p;}
            const double sign=draw(2)?1.:-1.,offset=sign*field->resolution()*static_cast<double>(1u<<draw(7));
            const Point normal{-(b.y-a.y)/chord,(b.x-a.x)/chord};
            p.middle.push_back(path[p.first]);if(!same(p.middle.back(),a))p.middle.push_back(a);
            std::vector<std::size_t>interior;
            for(std::size_t i=p.first+1;i<p.last;++i)if(s[i]>beginArc&&s[i]<endArc)interior.push_back(i);
            if(interior.empty()){
                // New knots are real mutations, never assumed raster-invariant.
                const unsigned knots=1+static_cast<unsigned>(draw(2));
                for(unsigned i=1;i<=knots;++i){const double u=static_cast<double>(i)/(knots+1);auto q=interpolate(a,b,u);
                    const double weight=1.-std::abs(2.*u-1.);q.x+=offset*weight*normal.x;q.y+=offset*weight*normal.y;p.middle.push_back(q);}
            }else for(const auto i:interior){
                const double u=(s[i]-beginArc)/(endArc-beginArc),weight=1.-std::abs(2.*u-1.);
                p.middle.push_back({path[i].x+offset*weight*normal.x,path[i].y+offset*weight*normal.y});
            }
            if(!same(p.middle.back(),b))p.middle.push_back(b);
            if(!same(p.middle.back(),path[p.last]))p.middle.push_back(path[p.last]);return p;
        }
        if(ids.size()<2){p.reason="NO_SPLICE_DONOR";return p;}
        auto donor=ids[draw(ids.size()-1)];if(donor==p.parent)donor=ids.back();
        if(donor==p.parent){p.reason="NO_SPLICE_DONOR";return p;}
        const auto&other=result.candidates[donor].path;p.parents.push_back(donor);
        if(other.size()<2){p.reason="IDENTITY_DONOR";return p;}
        p.first=draw(path.size()-1);p.last=p.first+1+draw(path.size()-p.first-1);
        std::pair<std::size_t,std::size_t>endpoints{};std::size_t matches=0;
        for(std::size_t i=0;i+1<other.size();++i){
            if(expired()){p.reason="DEADLINE";return p;}
            if(distanceXY(path[p.first],other[i])<=1.)
                for(std::size_t j=i+1;j<other.size();++j)if(distanceXY(path[p.last],other[j])<=1.){
                    ++matches;if(draw(matches)==0)endpoints={i,j};
                }
        }
        if(!matches){p.reason="NO_SPLICE_BRIDGES";return p;}
        p.middle.push_back(path[p.first]);
        for(std::size_t i=endpoints.first;i<=endpoints.second;++i)if(!same(p.middle.back(),other[i]))p.middle.push_back(other[i]);
        if(!same(p.middle.back(),path[p.last]))p.middle.push_back(path[p.last]);return p;
    }
    void segmentProposal(double deadline,std::uint64_t id,bool contract,double began){
        auto&count=result.search_counters.segment_proposals;
        static constexpr unsigned targetCaps[]={0,1,2,2,3};
        const unsigned capIndex=targetCaps[(count/2)%5];++count;
        PriSegmentAttempt attempt;attempt.proposal_id=id;attempt.began_s=began;
        attempt.move_type=contract?"segment_contract":"segment_reduce";
        const auto finish=[&](const std::string&status){attempt.status=status;attempt.finished_s=now();
            result.segment_attempts.push_back(attempt);
            result.search_counters.archive_bytes+=sizeof(PriSegmentAttempt)+512;
            if(result.search_counters.archive_bytes>options.max_archive_bytes){
                resource_stop=true;result.stage="RESOURCE_LIMIT";result.optimizer_resource_limit=true;reject("ARCHIVE_RESOURCE_LIMIT");
            }};
        if(!result.anchor||!result.selected[capIndex]){reject("SEGMENT_NO_PARENT");finish("NO_PARENT");return;}
        const auto parent=*result.selected[capIndex];attempt.parent_id=parent;
        const auto&source=result.candidates[parent];const auto&path=source.path;
        attempt.pri_cap=source.stats.pri*(contract?1.:.99);
        attempt.length_cap=contract?std::min(source.stats.length,*result.length_caps[capIndex]):*result.length_caps[capIndex];
        if(path.size()<2||(!contract&&!(source.stats.pri>0.))){reject("SEGMENT_IDENTITY");finish("IDENTITY");return;}
        const auto&record=records.at(parent);const auto s=arcs(path);
        static constexpr double windows[]={.5,1.,2.,4.,8.,16.,-1.};
        const double width=windows[draw(7)];
        std::size_t first=0,last=path.size()-1;
        if(width>0.){
            if(contract){
                first=draw(path.size()-1);
                last=std::max(first+1,static_cast<std::size_t>(std::lower_bound(s.begin(),s.end(),s[first]+width)-s.begin()));
                last=std::min(last,path.size()-1);
            }else {
                long double best=-1.;
                for(std::size_t a=0;a+1<path.size();++a){
                    if(expired()||now()>=deadline){reject("SEGMENT_WINDOW_DEADLINE");finish("CANCELLED");return;}
                    const auto b=std::min(path.size()-1,std::max(a+1,static_cast<std::size_t>(
                        std::lower_bound(s.begin(),s.end(),s[a]+width)-s.begin())));
                    std::map<std::size_t,unsigned>removed;
                    for(auto j=a;j<b;++j)for(auto cell:record.edges[j]->cells)++removed[cell];
                    long double surplus=0.;
                    for(const auto&cell:removed)if(cell.second==record.incidence.at(cell.first))
                        surplus+=std::max(0.,field->risk(static_cast<int>(cell.first%field->nx()),static_cast<int>(cell.first/field->nx()))-source.stats.pri);
                    if(surplus>best){best=surplus;first=a;last=b;}
                }
            }
        }
        attempt.first=first;attempt.last=last;
        PriSegmentSearchOptions settings;settings.pri_cap=attempt.pri_cap;settings.length_cap=attempt.length_cap;settings.seed=rng();
        settings.cancel=[this](){return expired();};
        settings.stop=[this,deadline](){return now()>=deadline;};
        auto found=runPriSegmentSearch(terrain,evaluator.policy,path,first,last,settings);attempt.counters=found.counters;attempt.termination=found.termination;
        if(found.status!="FOUND"){reject("SEGMENT_"+found.status);finish(found.status);return;}
        if(expired()){reject("SEGMENT_RETURN_AFTER_DEADLINE");finish("CANCELLED");return;}
        auto changed=evaluator.replace(record,first,last,found.replacement);
        PriPathSearchCandidate c;c.proposal_id=id;c.move_type=attempt.move_type;c.parent_ids={parent};c.callback_s=now();
        c.segment_pri_cap=attempt.pri_cap;c.segment_length_cap=attempt.length_cap;c.segment_first=first;c.segment_last=last;
        const bool accepted=admission(std::move(changed),std::move(c),false);
        finish(accepted?"ACCEPTED":"ADMISSION_REJECTED");
    }
    void optimize(double deadline,unsigned limit){
        unsigned work=0;
        while(!expired()&&now()<deadline&&(!limit||work<limit)){
            if(options.mode=="segment"&&result.search_counters.proposals%4>=2){
                const double began=now();if(began>=deadline||began>=options.budget_s||resource_stop)break;
                const bool contract=result.search_counters.proposals%4==3;
                ++work;const auto id=++result.search_counters.proposals;segmentProposal(deadline,id,contract,began);continue;
            }
            auto p=proposal();++work;const auto id=++result.search_counters.proposals;
            if(p.type=="shortcut")++result.search_counters.shortcut_proposals;
            if(p.type=="deform")++result.search_counters.deform_proposals;
            if(p.type=="splice")++result.search_counters.splice_proposals;
            if(!p.reason.empty()){reject(p.reason);continue;}
            auto parent=records.find(p.parent);
            if(parent==records.end())parent=records.emplace(p.parent,evaluator.build(result.candidates[p.parent].path)).first;
            auto record=evaluator.replace(parent->second,p.first,p.last,p.middle);
            PriPathSearchCandidate c;c.proposal_id=id;c.move_type=p.type;c.parent_ids=p.parents;c.callback_s=now();
            admission(std::move(record),std::move(c),false);
        }
    }
    PriPathSearchResult run(){
        if(!evaluator.policy.validState(start)||!evaluator.policy.validState(goal)){
            result.status=!evaluator.policy.validState(start)?"INVALID_START":"INVALID_GOAL";result.actual_wall_s=now();return std::move(result);
        }
        PlannerOptions p;p.state_cache=true;p.scalar_cost=true;p.sampler_query_cache=true;p.batch_after_first_solution=200;
        p.safety=std::make_shared<const SafetyPolicy>(terrain,field,40.,35.);p.safety_atlas_sampling=true;
        auto problem=makePlannerProblem(terrain,"bitstar_a3",start,goal,result.counters,100,1.1,100,p);
        result.planner_instances=1;result.setup_completed_s=now();unsigned callbacks=0;
        problem.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*>&path,const ob::Cost){
            PriPathSearchCandidate c;c.move_type="bitstar";c.callback_id=callbacks++;c.callback_s=now();c.anchor_phase=anchor_phase;
            auto copy=problem.snapshot(path);auto record=evaluator.build(copy);admission(std::move(record),std::move(c),true);problem.onFirstSolution();
        });
        const auto solve=[&](double deadline,unsigned work,const std::string&phase){
            PriPathSearchSlice slice;slice.id=result.slices.size();slice.phase=phase;slice.began_s=now();
            const auto before=problem.iterations(),batches=problem.batch_transition_count;
            if(!expired()&&now()<deadline){
                if(same(start,goal)){
                    if(result.candidates.empty()){PriPathSearchCandidate c;c.move_type="bitstar";c.callback_s=now();c.anchor_phase=anchor_phase;
                        admission(evaluator.build({start}),std::move(c),true);}slice.status="EXACT_SOLUTION";
                }else {
                    const ob::PlannerTerminationCondition ptc([&](){return expired()||now()>=deadline||(work&&problem.iterations()-before>=work);});
                    slice.status=problem.planner->solve(ptc).asString();
                }
            }else slice.status="SKIPPED_DEADLINE";
            slice.iterations=problem.iterations()-before;slice.batch_transitions=problem.batch_transition_count-batches;
            slice.finished_s=now();result.slices.push_back(std::move(slice));
        };
        if(options.mode=="length"){
            solve(options.budget_s,fixed?options.test_anchor_iterations+options.test_cycles*options.test_iterations_per_slice:0,"length");
            anchor_phase=false;result.stage="LENGTH_BASELINE";
        }else {
            solve(fixed?options.budget_s:result.anchor_deadline_s,fixed?options.test_anchor_iterations:0,"anchor");
            if(!resource_stop)lockAnchor();
            if(!result.anchor)solve(options.budget_s,fixed?options.test_cycles*options.test_iterations_per_slice:0,"no_anchor_continue");
            else for(unsigned cycle=0;!expired()&&(!fixed||cycle<options.test_cycles);++cycle){
                solve(fixed?options.budget_s:std::min(options.budget_s,now()+.005*options.budget_s),
                      fixed?options.test_iterations_per_slice:0,"bitstar");
                if(expired())break;
                PriPathSearchSlice slice;slice.id=result.slices.size();slice.phase="optimizer";slice.began_s=now();
                const auto before=result.search_counters.proposals;
                optimize(fixed?options.budget_s:std::min(options.budget_s,now()+.015*options.budget_s),
                         fixed?options.test_optimizer_proposals_per_slice:0);
                slice.proposals=result.search_counters.proposals-before;slice.finished_s=now();slice.status=resource_stop?"RESOURCE_LIMIT":"COMPLETED";
                result.slices.push_back(std::move(slice));
            }
        }
        if(resource_stop)result.stage="RESOURCE_LIMIT";
        result.actual_wall_s=now();result.status=result.selected[0]?"EXACT_SOLUTION":"NO_SOLUTION";return std::move(result);
    }
private:
    const Terrain&terrain;
    std::shared_ptr<const SafetyField>field;
    Point start,goal;
    PriPathSearchOptions options;
    Clock::time_point began;
    PriPathSearchResult result;
    std::mt19937_64 rng;
    Evaluator evaluator;
    bool fixed=false,anchor_phase=true,resource_stop=false;
    std::optional<std::size_t>anchor_best;
    std::array<std::vector<std::size_t>,16>pool;
    std::map<std::size_t,Record>records;
};
} // namespace
bool priPathSearchBetter(const PriPathSearchCandidate&a,const PriPathSearchCandidate&b){
    return std::tie(a.stats.pri,a.stats.length,a.id)<std::tie(b.stats.pri,b.stats.length,b.id);
}
std::array<std::optional<std::size_t>,4>selectPriPathSearchCandidates(
    const std::vector<PriPathSearchCandidate>&candidates,double anchorLength,double deadline){
    std::array<std::optional<std::size_t>,4>out{};const double ratios[]={1.,1.025,1.05,1.10};
    for(const auto&c:candidates)if(c.stats.valid&&std::isfinite(c.stats.pri)&&c.score_completed_s>=0.&&c.score_completed_s<=deadline)
        for(unsigned i=0;i<4;++i)if(std::isfinite(c.stats.length)&&c.stats.length<=ratios[i]*anchorLength&&
            (!out[i]||priPathSearchBetter(c,candidates[*out[i]])))out[i]=c.id;
    return out;
}
PriPathStats priPathSearchReplacementStats(const Terrain&t,std::shared_ptr<const SafetyField>field,
    const std::vector<Point>&path,std::size_t first,std::size_t last,const std::vector<Point>&middle,PriPathSearchCounters*counters){
    PriPathSearchCounters local;Evaluator evaluator(t,std::move(field),counters?*counters:local,67108864);
    auto original=evaluator.build(path);if(!original.stats.valid)return original.stats;
    auto result=evaluator.replace(original,first,last,middle);if(result.stats.valid)evaluator.recompute(result);return result.stats;
}
PriPathSearchResult runPriPathSearch(const Terrain&t,std::shared_ptr<const SafetyField>field,Point start,Point goal,const PriPathSearchOptions&o){
    const bool fixed=o.test_cycles!=0;
    if(!field||!std::isfinite(o.budget_s)||o.budget_s<=0.||t.traversabilityEnabled()||
       (o.mode!="length"&&o.mode!="shortcut"&&o.mode!="full"&&o.mode!="segment")||!o.max_archive_bytes||!o.max_edge_cache_bytes||
       (fixed&&(!o.test_anchor_iterations||!o.test_iterations_per_slice||!o.test_optimizer_proposals_per_slice))||
       (!fixed&&(o.test_anchor_iterations||o.test_iterations_per_slice||o.test_optimizer_proposals_per_slice)))
        throw std::invalid_argument("invalid PRI path search configuration");
    return Search(t,std::move(field),start,goal,o).run();
}
} // namespace terrain25d
