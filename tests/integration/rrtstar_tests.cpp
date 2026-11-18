#include "terrain25d/OmplIntegration.hpp"
#include <ompl/base/ScopedState.h>
#include <fstream>
#include <iostream>
#include <filesystem>
using namespace terrain25d;
namespace fs=std::filesystem;
void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}
void near(double a,double b){check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-8+1e-8*std::max(std::abs(a),std::abs(b)),"cost mismatch");}

class PruningProbe:public ompl::geometric::RRTstar {
public:
    using RRTstar::RRTstar;
    Motion *add(Point q,Motion *parent=nullptr) {
        auto *m=new Motion(si_);setPoint(m->state,q);m->parent=parent;
        m->incCost=parent?opt_->motionCost(parent->state,m->state):opt_->identityCost();
        m->cost=parent?opt_->combineCosts(parent->cost,m->incCost):opt_->identityCost();
        if(parent)parent->children.push_back(m);else startMotions_.push_back(m);
        nn_->add(m);return m;
    }
    double estimate(const Motion *m)const{return solutionHeuristic(m).value();}
    double oldExpression(const Motion *m)const {
        return opt_->motionCost(startMotions_.front()->state,m->state).value()+
            opt_->costToGo(m->state,pdef_->getGoal().get()).value();
    }
    void incumbent(Motion *m){bestGoalMotion_=m;m->inGoal=true;goalMotions_.push_back(m);bestCost_=m->cost;}
    int prune(){return pruneTree(bestCost_);}
    bool contains(const Motion *m)const{std::vector<Motion*> motions;nn_->list(motions);return std::find(motions.begin(),motions.end(),m)!=motions.end();}
};

void regression(const fs::path &output) {
    std::vector<double> heights(121,0.);heights[5*11+5]=4.;
    Terrain map(11,11,0,0,1,1,heights,std::vector<int>(100),{},.1,0,85);
    Point start{1,5},goal{9,8},q{9,5};PolicyCounters counters;
    auto p=makePlannerProblem(map,"informed_rrtstar",start,goal,counters,40);
    PruningProbe probe(p.si);probe.setProblemDefinition(p.pdef);probe.setTreePruning(true);
    probe.setPrunedMeasure(false);probe.setAdmissibleCostToCome(true);probe.setup();
    auto root=probe.add(start);auto a=probe.add({1,3},root);auto b=probe.add({9,3},a);
    auto incumbent=probe.add(goal,b);probe.incumbent(incumbent);
    auto candidate=probe.add(q,root);auto removable=probe.add({.2,.2},root);
    const double threshold=15.; // Exact analytical sum 2+8+5, on H=0.
    for(auto edge:std::vector<std::pair<Point,Point>>{{start,{1,3}},{{1,3},{9,3}},{{9,3},goal}})
        check(map.validMotion(edge.first,edge.second),"witness edge invalid");
    check(map.validMotion(start,q),"direct hill edge must remain valid");
    near(map.surfaceEdge(start,q).cost,6+2*std::sqrt(17.));
    check(probe.oldExpression(candidate)>threshold,"negative control must overestimate witness");
    near(probe.estimate(candidate),11.);
    check(probe.estimate(candidate)<threshold,"patched actual method not admissible");
    auto calls=counters.motion_heuristic_calls;int removed=probe.prune();
    check(counters.motion_heuristic_calls>calls,"real prune did not call heuristic hook");
    check(probe.contains(candidate),"real pruning incorrectly removed improving position");
    check(!probe.contains(removable)&&removed>0,"actual pruning did not remove over-bound leaf");
    fs::create_directories(output);
    std::ofstream mapFile(output/"map.t25d");
    mapFile<<"T25D 1\nunits m\ndiagonal SW_NE\ngrid 11 11 0 0 1 1\nmodel 0.1 0 85\nheights\n";
    for(int j=0;j<11;++j){for(int i=0;i<11;++i)mapFile<<heights[j*11+i]<<' ';mapFile<<'\n';}
    mapFile<<"mask\n";for(int j=0;j<10;++j){for(int i=0;i<10;++i)mapFile<<"0 ";mapFile<<'\n';}
    mapFile<<"rectangles 0\nend\n";
    std::ofstream(output/"witness_xy.csv")<<"x_m,y_m\n1,5\n1,3\n9,3\n9,8\n";
    std::ofstream report(output/"rrtstar_pruning_regression.json");
    report<<"{\"fixture\":true,\"threshold_source\":\"analytical_valid_witness_2_plus_8_plus_5\",\"incumbent_cost\":15,"
          <<"\"old_expression\":"<<probe.oldExpression(candidate)<<",\"actual_patched_expression\":"<<probe.estimate(candidate)
          <<",\"actual_removed\":"<<removed<<",\"candidate_retained\":true}\n";
    std::cout<<"PASS actual RRTstar solutionHeuristic + negative control + pruneTree; witness=15, old="<<probe.oldExpression(candidate)<<" patched="<<probe.estimate(candidate)<<" removed="<<removed<<'\n';
}

void costAndCallbacks() {
    for(const auto &id:{"rrtstar","informed_rrtstar"}) {
        for(bool tilted:{false,true}) {
            std::vector<double> heights;for(int j=0;j<=10;++j)for(int i=0;i<=10;++i)heights.push_back(tilted?.3*i+.2*j:0.);
            Terrain map(11,11,0,0,1,1,heights,std::vector<int>(100),{},.1,0,70);PolicyCounters counters;
            auto p=makePlannerProblem(map,id,{1,5},{9,5},counters,40);
            ob::ScopedState<> a(p.space),b(p.space);setPoint(a.get(),{1,5});setPoint(b.get(),{9,5});
            near(p.objective->motionCost(a.get(),b.get()).value(),std::hypot(8.,tilted?2.4:0.));
            check(!p.pdef->getGoal()->isSatisfied(a.get()),"goal acceptance wrong");
            setPoint(a.get(),{9-1e-10,5});check(!p.pdef->getGoal()->isSatisfied(a.get()),"near goal treated exact");
            int callbacks=0;double best=INFINITY;std::vector<std::vector<Point>> snapshots;
            p.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &path,ob::Cost cost) {
                auto snapshot=p.snapshot(path);check(snapshot.front().x==1&&snapshot.back().x==9,"callback endpoints");
                double recomputed=0;for(std::size_t k=1;k<snapshot.size();++k)recomputed+=map.surfaceEdge(snapshot[k-1],snapshot[k]).cost;
                near(recomputed,cost.value());check(cost.value()<=best+1e-8,"incumbent nonmonotone");best=cost.value();++callbacks;snapshots.push_back(snapshot);
            });
            auto status=p.planner->solve(ob::PlannerTerminationCondition([&]{return p.iterations()>=1000;}));
            check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks>1,"baseline multi-improvement callback coverage missing");
            dynamic_cast<RRTInspection&>(*p.rrt).auditTree(map);
            std::cout<<"PASS "<<id<<" "<<(tilted?"tilted":"flat")<<" cost callbacks="<<callbacks<<" final parent-cost checkpoint\n";
        }
    }
    // Undefined pre-validation cost must be infinity; obstacles may have finite
    // geometry cost and must still fail the common motion validator.
    auto h=std::vector<double>(121,0.);h[5*11+5]=NAN;
    Terrain nodata(11,11,0,0,1,1,h,std::vector<int>(100),{},.1,0,70);PolicyCounters c;
    auto p=makePlannerProblem(nodata,"informed_rrtstar",{1,5},{9,5},c,40);
    ob::ScopedState<> a(p.space),b(p.space);setPoint(a.get(),{1,5});setPoint(b.get(),{9,5});
    check(std::isinf(p.objective->motionCost(a.get(),b.get()).value()),"NoData cost must be infinity");
    check(!p.si->checkMotion(a.get(),b.get()),"NoData motion accepted");
    Terrain blocked(11,11,0,0,1,1,std::vector<double>(121),std::vector<int>(100),{{4,0,6,10}},.1,0,70);
    auto p2=makePlannerProblem(blocked,"rrtstar",{1,5},{9,5},c,40);
    near(p2.objective->motionCost(a.get(),b.get()).value(),8.);
    check(!p2.si->checkMotion(a.get(),b.get()),"finite cost incorrectly implies validity");
    std::cout<<"PASS pre-validation NoData/obstacle cost contract\n";
    // Adapter-only fixture with a goal region: the endpoint is (2,5), not the
    // nominal center (2.05,5). No informed/pruning policy is used in this test.
    Terrain flat(11,11,0,0,1,1,std::vector<double>(121),std::vector<int>(100),{},.1,0,70);
    auto region=makePlannerProblem(flat,"rrtstar",{1,5},{2.05,5},c,40);
    auto goal=std::make_shared<ob::GoalState>(region.si);ob::ScopedState<> center(region.space);
    setPoint(center.get(),{2.05,5});goal->setState(center.get());goal->setThreshold(.1);region.pdef->setGoal(goal);
    region.rrt->setGoalBias(1.);region.rrt->setRange(1.);int callbacks=0;
    std::vector<Point> saved;
    region.pdef->setIntermediateSolutionCallback([&](const ob::Planner*,const std::vector<const ob::State*> &interior,ob::Cost cost){
        check(interior.empty(),"expected empty intermediate callback");saved=region.snapshot(interior);
        check(saved.size()==2,"empty callback adapter must keep actual endpoints");near(saved.back().x,2.);near(cost.value(),1.);++callbacks;
    });
    auto status=region.planner->solve(ob::PlannerTerminationCondition([&]{return callbacks>0 || region.iterations()>=100;}));
    check(status==ob::PlannerStatus::EXACT_SOLUTION&&callbacks==1,"goal region callback fixture");
    region.planner->clear();near(saved.back().x,2.); // Snapshot owns its memory.
    std::cout<<"PASS empty-interior callback, actual goal-region endpoint, retained snapshot after clear\n";
}
int main(int argc,char **argv){try{ompl::RNG::setSeed(73129);regression(argc>1?argv[1]:".");costAndCallbacks();return 0;}
catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
