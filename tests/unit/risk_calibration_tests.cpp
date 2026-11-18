#include "terrain25d/RiskCalibration.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

using namespace terrain25d;
namespace {
void check(bool value,const char *message) { if (!value) throw std::runtime_error(message); }
void near(double a,double b) { check(std::isfinite(a)&&std::abs(a-b)<1e-10,"numeric disagreement"); }
template<class F> void rejects(F fn) {
    bool rejected=false;try { fn(); } catch (const std::exception &) { rejected=true; }
    check(rejected,"invalid calibration options accepted");
}
Terrain flat(int extent=6,std::vector<Rect> obstacles={}) {
    Terrain result(extent+1,extent+1,0,0,1,1,std::vector<double>((extent+1)*(extent+1)),
                   std::vector<int>(extent*extent),std::move(obstacles),.1,0,35);
    result.enableCollisionIndex();return result;
}

// Independent exact-cell test oracle: boundary parameter clipping against every
// rectangle, without the production row/slab enumeration or its helper calls.
bool touches(Point a,Point b,Rect box) {
    const double p[]={a.x,a.y},q[]={b.x,b.y},lo[]={box.xmin,box.ymin},hi[]={box.xmax,box.ymax};
    double lower=0.,upper=1.;
    for (int axis=0;axis<2;++axis) {
        if (p[axis]==q[axis]) { if (p[axis]<lo[axis]||p[axis]>hi[axis]) return false;continue; }
        const double t1=(lo[axis]-p[axis])/(q[axis]-p[axis]);
        const double t2=(hi[axis]-p[axis])/(q[axis]-p[axis]);
        lower=std::max(lower,std::min(t1,t2));upper=std::min(upper,std::max(t1,t2));
        if (lower>upper) return false;
    }
    return true;
}
double oracleRisk(Point a,Point b,int n,const std::vector<int> &support,const std::vector<double> &risk) {
    double maximum=0.;
    for (int j=0;j<n;++j) for (int i=0;i<n;++i) if (touches(a,b,{double(i),double(j),double(i+1),double(j+1)})) {
        if (!support[j*n+i]) return std::numeric_limits<double>::infinity();
        maximum=std::max(maximum,risk[j*n+i]);
    }
    return maximum;
}
bool insideFlat(Point p,int n) { return p.x>.1 && p.x<n-.1 && p.y>.1 && p.y<n-.1; }

// Exhaust every distinct possible threshold using unweighted connectivity, an
// independent oracle for finite-graph minimax (not a second Dijkstra routine).
double exhaustiveBottleneck(int n,Point start,Point goal,double spacing,double radius,
                           const std::vector<int> &support,const std::vector<double> &risk) {
    const int width=static_cast<int>(std::floor(n/spacing))+1;
    std::vector<Point> points;
    for (int j=0;j<width;++j) for (int i=0;i<width;++i) points.push_back({i*spacing,j*spacing});
    const int first=static_cast<int>(points.size()),last=first+1;
    points.push_back(start);points.push_back(goal);
    std::vector<std::vector<std::pair<int,double>>> edges(points.size());
    std::vector<double> thresholds{0.};
    for (int u=0;u<=last;++u) for (int v=u+1;v<=last;++v) {
        if (!insideFlat(points[u],n)||!insideFlat(points[v],n)) continue;
        bool linked;
        if (v<first) linked=std::abs(u%width-v%width)<=1 && std::abs(u/width-v/width)<=1;
        else if (u>=first) linked=true;
        else linked=std::hypot(points[u].x-points[v].x,points[u].y-points[v].y)<=std::nextafter(radius,INFINITY);
        if (!linked) continue;
        const double maximum=oracleRisk(points[u],points[v],n,support,risk);
        if (!std::isfinite(maximum)||maximum>40) continue;
        thresholds.push_back(maximum);edges[u].push_back({v,maximum});edges[v].push_back({u,maximum});
    }
    std::sort(thresholds.begin(),thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(),thresholds.end()),thresholds.end());
    for (const double cap:thresholds) {
        std::vector<bool> seen(points.size());std::queue<int> queue;seen[first]=true;queue.push(first);
        while (!queue.empty()) {
            const int at=queue.front();queue.pop();
            for (const auto &[other,weight]:edges[at]) if (weight<=cap&&!seen[other]) {
                seen[other]=true;queue.push(other);
            }
        }
        if (seen[last]) return cap;
    }
    return std::numeric_limits<double>::infinity();
}

void summaryOracle() {
    constexpr int n=6;
    std::vector<int> support(n*n,1);std::vector<double> risk(n*n);
    for (int i=0;i<n*n;++i) risk[i]=(i*17)%43;
    SafetyField field(n,n,0,0,1,n,n,support,risk);
    const std::vector<std::pair<Point,Point>> edges={{{.25,.75},{5.75,4.25}},{{1,1},{5,5}},
        {{2,1},{2,5}},{{1,3},{5,3}},{{3,3},{3,3}},{{.5,1.5},{5.5,1.5}}};
    for (const auto &edge:edges) {
        const auto a=field.edgeSummary(edge.first,edge.second),b=field.edgeSummary(edge.second,edge.first);
        check(a.supported&&a.maximum_risk&&a.reason.empty()&&a.touched_cells,"summary unavailable");
        near(*a.maximum_risk,oracleRisk(edge.first,edge.second,n,support,risk));
        near(*b.maximum_risk,*a.maximum_risk);check(a.touched_cells==b.touched_cells,"reverse cell count changed");
        const Point middle{(edge.first.x+edge.second.x)/2,(edge.first.y+edge.second.y)/2};
        near(*a.maximum_risk,std::max(*field.edgeSummary(edge.first,middle).maximum_risk,
                                    *field.edgeSummary(middle,edge.second).maximum_risk));
    }
    support[14]=0;risk[14]=0;
    SafetyField missing(n,n,0,0,1,n,n,support,risk);
    const auto absent=missing.edgeSummary({.5,2.5},{5.5,2.5});
    check(!absent.supported&&!absent.maximum_risk&&absent.reason=="missing_sem_support","NoData became zero risk");
    check(!missing.edgeSummary({2,2},{2,2}).supported,"closed corner skipped missing support");
    check(!missing.edgeSummary({std::nextafter(2.,0.),2.5},{std::nextafter(2.,0.),2.5}).supported,
          "16-ulp tie skipped missing cell");
    check(missing.edgeSummary({2.-1e-9,2.5},{2.-1e-9,2.5}).supported,"ulp tie grew into physical band");
    check(field.edgeSummary({-1,2},{2,2}).reason=="invalid_geometry","outside summary accepted");
    check(!field.edgeSummary({NAN,2},{2,2}).maximum_risk,"NaN produced a risk");
}

void graphOracleAndDeterminism() {
    constexpr int n=6;
    const auto terrain=flat(n);
    RiskCalibrationOptions options;options.grid_spacing_m=1.;options.endpoint_connection_radius_m=1.;
    const Point start{1,3},goal{5,3};
    for (int test=0;test<30;++test) {
        std::vector<int> support(n*n,1);std::vector<double> risk(n*n,1.);
        if (!test) {
            for (int j=2;j<4;++j) for (int i=2;i<4;++i) risk[j*n+i]=9.;
        } else {
            for (int j=0;j<n;++j) for (int i=0;i<n;++i) risk[j*n+i]=1.+(i*17+j*11+test*(i+3*j))%19;
        }
        SafetyField field(n,n,0,0,1,n,n,support,risk);
        const double expected=exhaustiveBottleneck(n,start,goal,1.,1.,support,risk);
        const auto result=calibrateRisk(terrain,field,start,goal,options);
        check(result.connected()&&result.b_graph&&result.effective_risk_cap,"connected test graph failed");
        near(*result.b_graph,expected);near(*result.effective_risk_cap,expected+.5);
        if (!test) near(*result.b_graph,1.); // Direct edge reaches risk9; detour is necessary.
        double witnessMaximum=0.;
        for (std::size_t k=1;k<result.witness.size();++k) {
            check(insideFlat(result.witness[k-1],n)&&insideFlat(result.witness[k],n),"witness footprint outside");
            witnessMaximum=std::max(witnessMaximum,oracleRisk(result.witness[k-1],result.witness[k],n,support,risk));
        }
        near(witnessMaximum,*result.b_graph);
        const auto again=calibrateRisk(terrain,field,start,goal,options);
        check(again.status==result.status&&again.b_graph==result.b_graph&&again.witness.size()==result.witness.size(),
              "calibration changed across calls");
        for (std::size_t i=0;i<result.witness.size();++i)
            check(again.witness[i].x==result.witness[i].x&&again.witness[i].y==result.witness[i].y,"witness nondeterministic");
        check(again.edge_evaluations==result.edge_evaluations&&again.state_evaluations==result.state_evaluations,
              "calibration work nondeterministic");
    }
}

void failureAndEndpoints() {
    const auto terrain=flat();
    std::vector<int> support(36,1);std::vector<double> risk(36,1.);
    risk[2*6]=8.;
    SafetyField highEndpoint(6,6,0,0,1,6,6,support,risk);
    RiskCalibrationOptions options;options.grid_spacing_m=.5;options.endpoint_connection_radius_m=.8;
    const auto high=calibrateRisk(terrain,highEndpoint,{.5,2.5},{5.5,2.5},options);
    check(high.connected(),"endpoint risk below40 wrongly rejected");near(*high.b_graph,8.);near(*high.effective_risk_cap,8.5);
    const auto singleton=calibrateRisk(terrain,highEndpoint,{.5,2.5},{.5,2.5},options);
    check(singleton.connected()&&singleton.witness.size()==1,"singleton calibration failed");near(*singleton.b_graph,8.);
    options.risk_slack=100.;near(*calibrateRisk(terrain,highEndpoint,{.5,2.5},{5.5,2.5},options).effective_risk_cap,40.);
    options.risk_slack=.5;
    for (int j=0;j<6;++j) { support[j*6+2]=0;risk[j*6+2]=0; }
    SafetyField blocked(6,6,0,0,1,6,6,support,risk);
    const auto no=calibrateRisk(terrain,blocked,{.5,2.5},{5.5,2.5},options);
    check(no.status=="CALIBRATION_NO_CONNECTION"&&!no.b_graph&&!no.effective_risk_cap&&no.witness.empty(),
          "disconnected graph fabricated threshold/witness");
    check(calibrateRisk(terrain,blocked,{2.5,2.5},{5.5,2.5},options).status=="INVALID_START","missing start accepted");
    check(calibrateRisk(terrain,blocked,{.5,2.5},{2.5,2.5},options).status=="INVALID_GOAL","missing goal accepted");
    check(calibrateRisk(terrain,blocked,{NAN,2.5},{5.5,2.5},options).status=="INVALID_START","NaN start accepted");
    for (int j=0;j<6;++j) { support[j*6+2]=1;risk[j*6+2]=41.; }
    SafetyField highWall(6,6,0,0,1,6,6,support,risk);
    check(calibrateRisk(terrain,highWall,{.5,2.5},{5.5,2.5},options).status=="CALIBRATION_NO_CONNECTION",
          "risk ceiling silently relaxed");
    const auto wallTerrain=flat(6,{{2.9,0,3.1,6}});
    SafetyField constant(6,6,0,0,1,6,6,std::vector<int>(36,1),std::vector<double>(36,1.));
    check(calibrateRisk(wallTerrain,constant,{.5,2.5},{5.5,2.5},options).status=="CALIBRATION_NO_CONNECTION",
          "geometry wall ignored by minimax");
    std::vector<double> heights;
    for (int j=0;j<=6;++j) for (int i=0;i<=6;++i) heights.push_back(j>=3?.5*(j-3):0.);
    const Terrain ramp(7,7,0,0,1,1,heights,std::vector<int>(36),{},.4,0,35);
    options.slope_cap_deg=20.;
    check(calibrateRisk(ramp,constant,{1,2.7},{5,2.7},options).status=="INVALID_START",
          "off-center footprint slope ignored");
    options.grid_spacing_m=0;rejects([&](){calibrateRisk(terrain,constant,{1,1},{5,5},options);});
    options.grid_spacing_m=.1;options.risk_slack=-1;rejects([&](){calibrateRisk(terrain,constant,{1,1},{5,5},options);});
}

void priCounterexample() {
    // Two synthetic corridors on a real field. Both touch endpoint risk8, but
    // the longer lower corridor has much less average unique-cell risk.
    constexpr int n=8;
    const auto terrain=flat(n);
    std::vector<int> support(n*n,1);std::vector<double> risk(n*n,7.5);
    for (int i=0;i<n;++i) risk[n+i]=1.;
    for (int j=1;j<=3;++j) { risk[j*n]=1.;risk[j*n+7]=1.; }
    risk[3*n]=8.;risk[3*n+7]=8.;
    SafetyField field(n,n,0,0,1,n,n,support,risk);
    const std::vector<Point> shortPath={{.5,3.5},{7.5,3.5}};
    const std::vector<Point> longPath={{.5,3.5},{.5,1.5},{7.5,1.5},{7.5,3.5}};
    const auto score=[&](const std::vector<Point> &path) {
        std::vector<bool> cells(n*n);double maximum=0.,sum=0.,length=0.;int count=0;
        for (std::size_t k=1;k<path.size();++k) {
            length+=std::hypot(path[k].x-path[k-1].x,path[k].y-path[k-1].y);
            for (int j=0;j<n;++j) for (int i=0;i<n;++i)
                if (touches(path[k-1],path[k],{double(i),double(j),double(i+1),double(j+1)})) cells[j*n+i]=true;
        }
        for (int i=0;i<n*n;++i) if (cells[i]) { maximum=std::max(maximum,risk[i]);sum+=risk[i];++count; }
        return std::vector<double>{maximum,sum/count,length};
    };
    const auto shortScore=score(shortPath),longScore=score(longPath);
    near(shortScore[0],longScore[0]);
    check(shortScore[1]>longScore[1]&&shortScore[2]<longScore[2],"corridor fixture lost PRI/length conflict");
    const auto result=calibrateRisk(terrain,field,shortPath.front(),shortPath.back());
    near(*result.b_graph,8.);
    check(result.witness.size()==2,"endpoint-bound certificate did not stop plateau traversal");
    check(result.edge_evaluations==1,"direct certificate evaluated unnecessary graph edges");
}

void radiusTies() {
    const auto terrain=flat();
    std::vector<double> risk(36,1.);risk[1*6+3]=41.;risk[2*6+3]=41.;
    SafetyField field(6,6,0,0,1,6,6,std::vector<int>(36,1),risk);
    RiskCalibrationOptions options;options.grid_spacing_m=1.;options.endpoint_connection_radius_m=.2;
    // 2.2-2 and 5.2-5 are slightly above binary64 .2. Both endpoints
    // require those radius-boundary links because the direct edge is blocked.
    const auto tied=calibrateRisk(terrain,field,{2.2,2.},{5.2,2.},options);
    check(tied.connected(),"decimal radius tie disconnected a graph endpoint");near(*tied.b_graph,1.);
    const auto outside=calibrateRisk(terrain,field,{2.2000001,2.},{5.2,2.},options);
    check(outside.status=="CALIBRATION_NO_CONNECTION","radius tie became a physical connection band");
}
}
int main() {
    try {
        summaryOracle();graphOracleAndDeterminism();failureAndEndpoints();priCounterexample();radiusTies();
        std::cout<<"PASS independent risk-cell oracle, exhaustive minimax thresholds, deterministic graph, full safety, failures and PRI counterexample\n";
        return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}
