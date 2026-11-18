#include "terrain25d/OmplIntegration.hpp"
#include <ompl/base/ScopedState.h>
#include <ompl/util/Console.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

// The friend is only implemented in this replay executable. Production sampler
// calls retain OMPL's normal RNG construction and incur no proposal observer.
namespace terrain25d {
struct TerrainSamplerReplayAccess {
    static void seed(TerrainInformedSampler &sampler,unsigned int seed) {sampler.rng_.setLocalSeed(seed);}
    static double peekUniform(const TerrainInformedSampler &sampler) {
        auto copy=sampler.rng_;return copy.uniform01();
    }
};
}
using namespace terrain25d;
namespace {
void check(bool condition,const char *message) {if(!condition)throw std::runtime_error(message);}
std::uint64_t bits(double value) {
    std::uint64_t result=0;std::memcpy(&result,&value,sizeof(result));return result;
}
bool identical(Point a,Point b) {return bits(a.x)==bits(b.x)&&bits(a.y)==bits(b.y);}
Terrain fixture(bool ridge=false,bool nodata=false,double origin=0,double spacing=1) {
    std::vector<double> heights;
    for(int j=0;j<=20;++j)for(int i=0;i<=20;++i)
        heights.push_back(ridge?(3.-.3*std::abs(i-10)):0.);
    if(nodata)heights[11*21+10]=std::numeric_limits<double>::quiet_NaN();
    return Terrain(21,21,origin,origin,spacing,spacing,heights,std::vector<int>(400),{},.1*spacing,0,70);
}
void sameCounters(const PolicyCounters &a,const PolicyCounters &b) {
#define SAME_COUNTER(name) check(a.name==b.name,"counter differs: " #name)
    SAME_COUNTER(motion_heuristic_calls);SAME_COUNTER(from_start_calls);SAME_COUNTER(to_goal_calls);
    SAME_COUNTER(cost_to_go_calls);SAME_COUNTER(sampler_one_bound_calls);SAME_COUNTER(sampler_interval_calls);
    SAME_COUNTER(sampler_finite_bound_calls);SAME_COUNTER(sample_proposals);SAME_COUNTER(samples_returned);
    SAME_COUNTER(rejected_bounds);SAME_COUNTER(rejected_nodata);SAME_COUNTER(rejected_heuristic);
    SAME_COUNTER(exhausted_calls);SAME_COUNTER(finite_bound_sample_proposals);
    SAME_COUNTER(finite_bound_samples_returned);SAME_COUNTER(finite_bound_rejected_bounds);
    SAME_COUNTER(finite_bound_rejected_nodata);SAME_COUNTER(finite_bound_rejected_heuristic);
    SAME_COUNTER(sampler_invalid_interval_calls);SAME_COUNTER(sampler_lower_bound_rejected_calls);
    SAME_COUNTER(sampler_proposal_limit_calls);
#undef SAME_COUNTER
    check(a.exhausted_calls==a.sampler_invalid_interval_calls+a.sampler_lower_bound_rejected_calls+
          a.sampler_proposal_limit_calls,"failure counters do not partition exhausted calls");
}
struct Interval {double lo,hi;bool one;};
void pairedTrace(bool lifted,bool stateCache,bool ridge,bool nodata,unsigned int attempts,
                 bool identity=false,double origin=0,double spacing=1) {
    auto map=fixture(ridge,nodata,origin,spacing);
    const Point start{origin+2*spacing,origin+10*spacing};
    const Point goal=identity?start:Point{origin+18*spacing,origin+10*spacing};
    PolicyCounters plainCounters,cachedCounters;
    PlannerOptions plainOptions;plainOptions.state_cache=stateCache;
    auto cachedOptions=plainOptions;cachedOptions.sampler_query_cache=true;
    auto plain=makeProblem(map,lifted,start,goal,plainCounters,200,1.1,attempts,plainOptions);
    auto cached=makeProblem(map,lifted,start,goal,cachedCounters,200,1.1,attempts,cachedOptions);
    check(!plain.objective->samplerQueryCacheEnabled()&&cached.objective->samplerQueryCacheEnabled(),
          "sampler option did not reach objective");
    auto a=plain.objective->allocInformedStateSampler(plain.pdef,attempts);
    auto b=cached.objective->allocInformedStateSampler(cached.pdef,attempts);
    auto *pa=dynamic_cast<TerrainInformedSampler*>(a.get());
    auto *pb=dynamic_cast<TerrainInformedSampler*>(b.get());
    check(pa&&pb,"expected actual terrain sampler");
    TerrainSamplerReplayAccess::seed(*pa,11);TerrainSamplerReplayAccess::seed(*pb,11);
    ob::ScopedState<> sa(plain.space),sb(cached.space);
    const double lower=plain.objective->queryLowerBound();
    const double positive=std::max(spacing,lower);
    const double infinity=std::numeric_limits<double>::infinity();
    const double nan=std::numeric_limits<double>::quiet_NaN();
    std::vector<Interval> intervals={
        {0,nan,true},{-1,positive,false},{infinity,infinity,false},{0,0,true},{0,-infinity,true},
        {positive,positive,false},{positive+1,positive,false},{0,lower,true},
        {0,std::nextafter(lower,0.),false},{0,std::nextafter(lower,infinity),true},
        {0,std::nextafter(std::nextafter(lower,infinity),infinity),false},
        {0,positive+spacing,true},{0,positive+spacing,false},
        {positive+.75*spacing,positive+spacing,false},{0,positive+spacing,true},
        {0,std::nextafter(positive+spacing,infinity),false},
        {0,infinity,true},{positive+spacing,infinity,false},{0,positive+6*spacing,false},{0,100*spacing,true},
        {0,std::numeric_limits<double>::max(),false},{-0.,positive+spacing,false}};
    for(unsigned int round=0;round<50;++round)for(const auto interval:intervals) {
        const Point sentinel{-123.5,-457.25};setPoint(sa.get(),sentinel);setPoint(sb.get(),sentinel);
        const auto cachedHits=cachedCounters.sampler_bound_cache_hits;
        const auto cachedBuilds=cachedCounters.sampler_bound_cache_rebuilds;
        const auto queryHits=cachedCounters.sampler_query_cache_hits;
        const auto queryBuilds=cachedCounters.sampler_query_cache_rebuilds;
        const auto invalidCalls=cachedCounters.sampler_invalid_interval_calls;
        const auto proposalsBefore=plainCounters.sample_proposals;
        const bool acceptedA=interval.one?a->sampleUniform(sa.get(),ob::Cost(interval.hi)):
            a->sampleUniform(sa.get(),ob::Cost(interval.lo),ob::Cost(interval.hi));
        const bool acceptedB=interval.one?b->sampleUniform(sb.get(),ob::Cost(interval.hi)):
            b->sampleUniform(sb.get(),ob::Cost(interval.lo),ob::Cost(interval.hi));
        check(acceptedA==acceptedB,"cache changed acceptance trace");
        check(identical(point(sa.get()),point(sb.get())),"cache changed proposal/state bits");
        check(bits(TerrainSamplerReplayAccess::peekUniform(*pa))==bits(TerrainSamplerReplayAccess::peekUniform(*pb)),
              "cache changed RNG advancement on accepted or rejected proposal");
        sameCounters(plainCounters,cachedCounters);
        if(cachedCounters.sampler_invalid_interval_calls!=invalidCalls) {
            check(cachedCounters.sampler_bound_cache_hits==cachedHits&&
                  cachedCounters.sampler_bound_cache_rebuilds==cachedBuilds&&
                  cachedCounters.sampler_query_cache_hits==queryHits&&
                  cachedCounters.sampler_query_cache_rebuilds==queryBuilds,
                  "invalid interval accessed a cache");
        }
        if(attempts==1)check(plainCounters.sample_proposals-proposalsBefore<=1,
                             "single-attempt replay hid intermediate proposals");
    }
    check(cachedCounters.sampler_query_cache_rebuilds==1,"query cache not built exactly once");
    check(cachedCounters.sampler_query_cache_hits>0&&cachedCounters.sampler_bound_cache_hits>0,
          "cache was not exercised");
    check(cachedCounters.sampler_bound_cache_rebuilds>100,"distinct upper bounds were merged");
    check(plainCounters.sampler_query_cache_hits==0&&plainCounters.sampler_query_cache_rebuilds==0&&
          plainCounters.sampler_bound_cache_hits==0&&plainCounters.sampler_bound_cache_rebuilds==0,
          "baseline silently enabled cache");
    check(!a->hasInformedMeasure()&&!b->hasInformedMeasure(),"cache changed informed measure claim");
    check(bits(a->getInformedMeasure(ob::Cost(lower),ob::Cost(infinity)))==
          bits(b->getInformedMeasure(ob::Cost(lower),ob::Cost(infinity))),"measure fallback changed");
}
void exactKeyAndLowerInterval() {
    auto map=fixture();PolicyCounters counters;PlannerOptions options;options.sampler_query_cache=true;
    auto p=makeProblem(map,true,{2,10},{18,10},counters,200,1.1,1,options);
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,1);ob::ScopedState<> state(p.space);
    const double upper=17,adjacent=std::nextafter(upper,INFINITY);
    sampler->sampleUniform(state.get(),ob::Cost(upper));
    check(counters.sampler_bound_cache_rebuilds==1,"first upper bound did not build");
    sampler->sampleUniform(state.get(),ob::Cost(16.5),ob::Cost(upper));
    check(counters.sampler_bound_cache_rebuilds==1&&counters.sampler_bound_cache_hits==1,
          "changing lower bound rebuilt upper geometry");
    sampler->sampleUniform(state.get(),ob::Cost(adjacent));
    check(counters.sampler_bound_cache_rebuilds==2,"adjacent-double upper bound reused stale geometry");
    sampler->sampleUniform(state.get(),ob::Cost(upper),ob::Cost(upper));
    sampler->sampleUniform(state.get(),ob::Cost(adjacent));
    check(counters.sampler_bound_cache_rebuilds==2&&counters.sampler_bound_cache_hits==2,
          "invalid interval replaced cached upper bound");
    sampler->sampleUniform(state.get(),ob::Cost(upper));
    check(counters.sampler_bound_cache_rebuilds==3,"cache is not keyed by current upper bits");
}
struct ReplayResult {PolicyCounters counters;double seconds=0;std::uint64_t trace=1469598103934665603ULL;};
ReplayResult block(const std::string &id,bool queryCache,unsigned calls) {
    auto map=fixture(id=="near_empty_hill");ReplayResult result;
    PlannerOptions options;options.state_cache=true;options.scalar_cost=true;options.sampler_query_cache=queryCache;
    auto p=makeProblem(map,true,{2,10},{18,10},result.counters,200,1.1,100,options);
    auto sampler=p.objective->allocInformedStateSampler(p.pdef,100);
    TerrainSamplerReplayAccess::seed(*dynamic_cast<TerrainInformedSampler*>(sampler.get()),11);
    ob::ScopedState<> state(p.space);setPoint(state.get(),{-123.5,-457.25});
    const double lower=p.objective->queryLowerBound();
    const double upper=id=="lower_bound_guard"?lower:
        (id=="near_empty_flat"||id=="near_empty_hill")?std::nextafter(lower,INFINITY):
        id=="ellipse"?lower+2:INFINITY;
    const auto begin=std::chrono::steady_clock::now();
    for(unsigned k=0;k<calls;++k) {
        const bool accepted=(k%2==0)?sampler->sampleUniform(state.get(),ob::Cost(upper)):
            sampler->sampleUniform(state.get(),ob::Cost(0),ob::Cost(upper));
        for(const auto value:{std::uint64_t(accepted),bits(point(state.get()).x),bits(point(state.get()).y)})
            result.trace=(result.trace^value)*1099511628211ULL;
    }
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    return result;
}
void replay(unsigned calls) {
    check(calls>0&&calls<=10000000,"replay calls outside [1,10000000]");
    std::cout<<std::setprecision(17)<<"{\"schema_version\":1,\"kind\":\"sampler_fixed_work_replay\","
        "\"seed\":11,\"calls_per_block\":"<<calls<<",\"clock_policy\":\"steady_clock_per_block\","
        "\"timing_scope\":\"sampling_calls_plus_trace_hash_and_loop_control\",\"rows\":[";
    bool first=true;
    for(const auto *id:{"lower_bound_guard","near_empty_flat","near_empty_hill","ellipse","full_domain"}) {
        const auto baseline=block(id,false,calls),cached=block(id,true,calls);
        check(baseline.trace==cached.trace,"replay trace hashes differ");sameCounters(baseline.counters,cached.counters);
        for(unsigned mode=0;mode<2;++mode) {
            const auto &row=mode?cached:baseline;
            if(!first)std::cout<<',';
            first=false;
            std::cout<<"{\"case_id\":\""<<id<<"\",\"sampler_query_cache\":"<<(mode?"true":"false")
                <<",\"calls\":"<<calls<<",\"elapsed_s\":"<<row.seconds<<",\"trace_hash\":\""<<row.trace<<'"';
#define WRITE_COUNTER(name) std::cout<<",\"" #name "\":"<<row.counters.name
            WRITE_COUNTER(sample_proposals);WRITE_COUNTER(samples_returned);
            WRITE_COUNTER(rejected_bounds);WRITE_COUNTER(rejected_nodata);WRITE_COUNTER(rejected_heuristic);
            WRITE_COUNTER(sampler_query_cache_hits);WRITE_COUNTER(sampler_query_cache_rebuilds);
            WRITE_COUNTER(sampler_bound_cache_hits);WRITE_COUNTER(sampler_bound_cache_rebuilds);
            WRITE_COUNTER(sampler_invalid_interval_calls);WRITE_COUNTER(sampler_lower_bound_rejected_calls);
            WRITE_COUNTER(sampler_proposal_limit_calls);WRITE_COUNTER(exhausted_calls);
#undef WRITE_COUNTER
            std::cout<<'}';
        }
    }
    std::cout<<"]}\n";
}
}
int main(int argc,char **argv) {
    try {
        ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);ompl::RNG::setSeed(73129);
        if(argc==3&&std::string(argv[1])=="--replay")replay(static_cast<unsigned>(std::stoul(argv[2])));
        else {
            check(argc==1,"usage: a3_sampler_cache_tests [--replay CALLS]");
            for(const bool lifted:{false,true})for(const bool stateCache:{false,true})
                for(const unsigned attempts:{1U,7U}) {
                    pairedTrace(lifted,stateCache,false,false,attempts);
                    pairedTrace(lifted,stateCache,true,true,attempts);
                }
            pairedTrace(true,true,false,false,1,true);
            pairedTrace(true,true,true,false,1,false,1e6,.125);
            pairedTrace(true,true,true,false,1,false,-100,10);
            exactKeyAndLowerInterval();
            std::cout<<"PASS sampler bitwise fixed-work trace, both overloads, interval guards, exact bound keys, "
                "failure partition, query lifetime, XY/A3, state cache combinations, NoData and scale/offset\n";
        }
        return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
