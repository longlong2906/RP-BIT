#include "terrain25d/PriPortfolio.hpp"
#include <array>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

namespace terrain25d {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
double quantile(std::vector<double> values,double p) {
    std::sort(values.begin(),values.end());
    const double x=p*(values.size()-1);const auto i=static_cast<std::size_t>(x);
    return values[i]+(x-i)*(values[std::min(i+1,values.size()-1)]-values[i]);
}
std::uint32_t rotr(std::uint32_t x,unsigned n) {return (x>>n)|(x<<(32-n));}
}
// SHA-256 of exactly one byte (0 or 1) per row-major cell. This encoding is
// independent of int width/endianness and is checked against hashlib in tests.
std::string priMaskSha256(const std::vector<int>&mask) {
    static constexpr std::uint32_t k[]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::vector<unsigned char> bytes;bytes.reserve(mask.size()+72);
    for(int b:mask){if(b!=0&&b!=1)throw std::invalid_argument("nonbinary PRI mask");bytes.push_back(static_cast<unsigned char>(b));}
    const auto bits=static_cast<std::uint64_t>(bytes.size())*8;bytes.push_back(0x80);
    while(bytes.size()%64!=56)bytes.push_back(0);
    for(int i=7;i>=0;--i)bytes.push_back(static_cast<unsigned char>(bits>>(8*i)));
    std::array<std::uint32_t,8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for(std::size_t offset=0;offset<bytes.size();offset+=64){
        std::array<std::uint32_t,64>w{};
        for(unsigned i=0;i<16;++i)for(unsigned j=0;j<4;++j)w[i]=(w[i]<<8)|bytes[offset+4*i+j];
        for(unsigned i=16;i<64;++i){const auto s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3),s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for(unsigned i=0;i<64;++i){const auto t1=hh+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            const auto t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto x:h)out<<std::setw(8)<<x;return out.str();
}
PriPathStats priPathStats(const Terrain&terrain,const SafetyPolicy&policy,const std::vector<Point>&path) {
    PriPathStats out;if(path.empty()){out.reason="empty_path";return out;}
    std::set<std::size_t> unique;long double length=0.,xy=0.,risk=0.;
    for(std::size_t i=0;i<std::max<std::size_t>(1,path.size()-1);++i){
        const auto a=path[i],b=path.size()>1?path[i+1]:a;
        out.reason=policy.rejectionReason(a,b);if(!out.reason.empty())return out;
        const auto cells=policy.field()->touchedCells(a,b);if(cells.empty()){out.reason="missing_sem_support";return out;}
        unique.insert(cells.begin(),cells.end());
        const double cost=terrain.surfaceCost(a,b);if(!std::isfinite(cost)){out.reason="undefined_surface_cost";return out;}
        length+=cost;xy+=distanceXY(a,b);
    }
    out.cells.assign(unique.begin(),unique.end());
    for(auto id:out.cells)risk+=policy.field()->risk(static_cast<int>(id%policy.field()->nx()),static_cast<int>(id/policy.field()->nx()));
    out.length=static_cast<double>(length);out.length_xy=static_cast<double>(xy);out.risk_sum=static_cast<double>(risk);
    out.pri=static_cast<double>(risk/out.cells.size());out.valid=true;return out;
}
std::vector<PriMask> priPortfolioMasks(const SafetyField&field,const PriPathStats&anchor,Point start,Point goal,
                                     const std::function<bool()>&cancel) {
    if(!anchor.valid||anchor.cells.empty())throw std::invalid_argument("PRI masks require a valid anchor");
    const auto size=static_cast<std::size_t>(field.nx())*field.ny();
    std::vector<double> values;std::vector<bool> onAnchor(size,false),endpoint(size,false);
    for(auto cell:anchor.cells){if(cell>=size)throw std::invalid_argument("anchor cell outside field");onAnchor[cell]=true;values.push_back(field.risk(static_cast<int>(cell%field.nx()),static_cast<int>(cell/field.nx())));}
    for(auto cell:field.touchedCells(start,start))endpoint[cell]=true;
    for(auto cell:field.touchedCells(goal,goal))endpoint[cell]=true;
    struct Component{std::vector<std::size_t>cells;long double surplus=0.;};
    const auto components=[&](double threshold){
        std::vector<Component> out;std::vector<bool>seen(size,false);std::vector<std::size_t> queue;
        const auto active=[&](std::size_t cell){const int i=static_cast<int>(cell%field.nx()),j=static_cast<int>(cell/field.nx());return field.supported(i,j)&&field.risk(i,j)<=40.&&field.risk(i,j)>threshold;};
        for(std::size_t root=0;root<size;++root){
            if((root&1023)==0&&cancel&&cancel())return std::vector<Component>{};
            if(seen[root]||!active(root))continue;
            Component c;bool touchesEndpoint=false,touchesAnchor=false;seen[root]=true;queue={root};
            for(std::size_t head=0;head<queue.size();++head){
                if((head&1023)==0&&cancel&&cancel())return std::vector<Component>{};
                const auto id=queue[head];c.cells.push_back(id);touchesEndpoint=touchesEndpoint||endpoint[id];
                const int x=static_cast<int>(id%field.nx()),y=static_cast<int>(id/field.nx());
                if(onAnchor[id]){touchesAnchor=true;c.surplus+=field.risk(x,y)-anchor.pri;}
                for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){const int i=x+dx,j=y+dy;
                    if((dx==0&&dy==0)||i<0||i>=field.nx()||j<0||j>=field.ny())continue;
                    const auto next=static_cast<std::size_t>(j)*field.nx()+i;
                    if(!seen[next]&&active(next)){seen[next]=true;queue.push_back(next);}
                }
            }
            if(touchesAnchor&&!touchesEndpoint){std::sort(c.cells.begin(),c.cells.end());out.push_back(std::move(c));}
        }
        std::sort(out.begin(),out.end(),[](const auto&a,const auto&b){return a.surplus!=b.surplus?a.surplus>b.surplus:a.cells.front()<b.cells.front();});return out;
    };
    const double t90=std::max(anchor.pri,quantile(values,.9)),t75=std::max(anchor.pri,quantile(values,.75));
    const auto c90=components(t90),c75=components(t75);std::vector<PriMask> masks;
    for(unsigned n=1;n<=3;++n){PriMask m;m.episode_id=n;m.recipe=n==1?"q90_top1":n==2?"q75_top1":"q75_top2";m.threshold=n==1?t90:t75;
        if(cancel&&cancel()){m.status="SKIPPED_DEADLINE";masks.push_back(std::move(m));continue;}
        const auto &list=n==1?c90:c75;
        if(list.empty()){m.status="SKIPPED_NO_COMPONENT";masks.push_back(std::move(m));continue;}
        for(std::size_t i=0;i<std::min<std::size_t>(list.size(),n==3?2:1);++i)m.excluded.insert(m.excluded.end(),list[i].cells.begin(),list[i].cells.end());
        std::sort(m.excluded.begin(),m.excluded.end());m.allowed.assign(size,1);for(auto id:m.excluded)m.allowed[id]=0;
        m.sha256=priMaskSha256(m.allowed);m.status="READY";
        for(const auto&old:masks)if(old.allowed==m.allowed)m.status="SKIPPED_DUPLICATE";
        masks.push_back(std::move(m));
    }
    return masks;
}
bool priCandidateWithinBudget(double length,double cap,double finished,double deadline) {
    return std::isfinite(length)&&std::isfinite(cap)&&std::isfinite(finished)&&std::isfinite(deadline)&&length<=cap&&finished>=0.&&finished<=deadline;
}
bool priCandidateBetter(const PriCandidate&a,const PriCandidate&b) {
    return std::tie(a.stats.pri,a.stats.length,a.episode_id,a.callback_id)<std::tie(b.stats.pri,b.stats.length,b.episode_id,b.callback_id);
}
void considerPriCandidate(PriPortfolioResult&out,PriCandidate&c,std::optional<std::size_t>&anchorBest,
                          double anchorDeadline,double globalDeadline,bool fixedWork) {
    if(!c.reason.empty())return;
    if(!c.stats.valid){c.reason=c.stats.reason;return;}
    if(c.score_completed_s>globalDeadline){c.reason="SCORING_AFTER_DEADLINE";return;}
    if(c.anchor_phase&&(c.score_completed_s<=anchorDeadline||fixedWork)){
        c.reason="ANCHOR_CANDIDATE";
        if(!anchorBest||c.stats.length<out.candidates[*anchorBest].stats.length){anchorBest=c.id;out.selected=c.id;c.published=true;}
        return;
    }
    if(c.anchor_phase&&anchorBest){c.reason="SCORING_AFTER_ANCHOR_DEADLINE";return;}
    if(out.length_cap){
        if(!priCandidateWithinBudget(c.stats.length,*out.length_cap,c.score_completed_s,globalDeadline)){c.reason="LENGTH_CAP_EXCEEDED";return;}
        c.eligible=true;c.reason="ELIGIBLE";
        if(!out.selected||priCandidateBetter(c,out.candidates[*out.selected])){c.published=true;out.selected=c.id;}
    }else{
        // A callback may straddle the quarter deadline. It cannot become the
        // anchor, but remains a valid fallback if no earlier anchor exists.
        c.eligible=true;c.reason="NO_ANCHOR_FALLBACK";
        if(!out.selected||c.stats.length<out.candidates[*out.selected].stats.length){c.published=true;out.selected=c.id;}
    }
}
PriPortfolioResult runPriPortfolio(const Terrain&terrain,std::shared_ptr<const SafetyField>field,Point start,Point goal,
                                  const PriPortfolioOptions&options) {
    if(!field||!std::isfinite(options.length_cap_ratio)||options.length_cap_ratio<1.||options.length_cap_ratio>1.1||
       !std::isfinite(options.budget_s)||options.budget_s<=0.||terrain.traversabilityEnabled())throw std::invalid_argument("invalid PRI portfolio configuration");
    PriPortfolioResult out;const auto began=Clock::now();const double quarter=options.budget_s/4.;
    const auto now=[&](){return elapsed(began);};
    std::shared_ptr<const SafetyPolicy> basePolicy=std::make_shared<const SafetyPolicy>(terrain,field,40.,35.);
    if(!basePolicy->validState(start)||!basePolicy->validState(goal)){out.status=!basePolicy->validState(start)?"INVALID_START":"INVALID_GOAL";out.actual_wall_s=now();return out;}
    bool anchorPhase=true;unsigned activeEpisode=0,callback=0;std::optional<std::size_t> anchorBest;
    unsigned instances=0;
    const auto create=[&](std::shared_ptr<const SafetyPolicy>policy){
        PlannerOptions p;p.state_cache=true;p.scalar_cost=true;p.sampler_query_cache=true;p.batch_after_first_solution=200;p.safety=std::move(policy);p.safety_atlas_sampling=true;
        ++instances;return makePlannerProblem(terrain,"bitstar_a3",start,goal,out.counters,100,1.1,100,p);
    };
    auto base=create(basePolicy);const double baseSetupCompleted=now();
    const auto install=[&](PlannerProblem&problem,std::shared_ptr<const SafetyPolicy>policy){
        problem.pdef->setIntermediateSolutionCallback([&,policy](const ob::Planner*,const std::vector<const ob::State*>&path,const ob::Cost){
            PriCandidate c;c.id=out.candidates.size();c.episode_id=activeEpisode;c.callback_id=callback++;c.callback_s=now();c.anchor_phase=anchorPhase;
            c.path=problem.snapshot(path);c.stats=priPathStats(terrain,*policy,c.path);
            const bool exact=!c.path.empty()&&distanceXY(c.path.front(),start)<=1e-9&&distanceXY(c.path.back(),goal)<=1e-9;
            c.reason=exact?"":"endpoint_mismatch";c.score_completed_s=now();
            considerPriCandidate(out,c,anchorBest,quarter,options.budget_s,options.work_iterations_per_episode!=0);
            out.candidates.push_back(std::move(c));problem.onFirstSolution();
        });
    };
    const auto solve=[&](PlannerProblem&problem,unsigned episode,bool fresh,const std::string&mask,double deadline,unsigned work,double setupBegan,double setupCompleted){
        activeEpisode=episode;callback=0;PriEpisode e;e.episode_id=episode;e.planner_instance_id=fresh?instances:1;e.fresh_planner=fresh;e.mask_sha256=mask;e.began_s=setupBegan;e.setup_completed_s=setupCompleted;
        const auto beginIterations=problem.iterations();
        if(now()<deadline){
            if(start.x==goal.x&&start.y==goal.y){const auto &cb=problem.pdef->getIntermediateSolutionCallback();cb(problem.planner.get(),{problem.pdef->getStartState(0)},ob::Cost(0.));e.status="EXACT_SOLUTION";}
            else {e.solve_called=true;const auto ptc=ob::PlannerTerminationCondition([&](){return now()>=deadline||(work&&problem.iterations()-beginIterations>=work);});e.status=problem.planner->solve(ptc).asString();}
        }else e.status="SKIPPED_DEADLINE";
        e.iterations=problem.iterations()-beginIterations;e.batch_transitions=problem.batch_transition_count;e.finished_s=now();out.episodes.push_back(std::move(e));
    };
    install(base,basePolicy);
    solve(base,0,true,"",options.work_iterations_per_episode?options.budget_s:quarter,options.work_iterations_per_episode,0.,baseSetupCompleted);
    anchorPhase=false;
    if(anchorBest){out.anchor=anchorBest;out.selected=anchorBest;auto &anchor=out.candidates[*anchorBest];anchor.eligible=true;anchor.published=true;anchor.reason="ANCHOR_SELECTED";
        out.length_cap=options.length_cap_ratio*anchor.stats.length;out.stage="ANCHOR_LOCKED";
        const auto maskBegan=Clock::now();out.masks=priPortfolioMasks(*field,anchor.stats,start,goal,[&](){return now()>=options.budget_s;});out.mask_build_s=elapsed(maskBegan);
        for(const auto&m:out.masks){
            if(m.status!="READY")continue;
            const double episodeStart=now();if(episodeStart>=options.budget_s)break;
            auto policy=std::make_shared<const SafetyPolicy>(terrain,field,40.,35.,m.allowed);auto p=create(policy);install(p,policy);
            solve(p,m.episode_id,true,m.sha256,options.work_iterations_per_episode?options.budget_s:std::min(options.budget_s,episodeStart+quarter),options.work_iterations_per_episode,episodeStart,now());
        }
        // Unused episode slots return to the unchanged original planner.
        if(now()<options.budget_s){unsigned skipped=0;for(const auto&m:out.masks)if(m.status!="READY")++skipped;
            if(!options.work_iterations_per_episode||skipped){const double t=now();solve(base,4,false,"",options.budget_s,options.work_iterations_per_episode*skipped,t,t);}}
    }else{out.stage="NO_ANCHOR_CONTINUE";const double t=now();solve(base,4,false,"",options.budget_s,options.work_iterations_per_episode*3,t,t);}
    out.actual_wall_s=now();out.status=out.selected?"EXACT_SOLUTION":"NO_SOLUTION";return out;
}
} // namespace terrain25d
