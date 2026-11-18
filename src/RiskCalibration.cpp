#include "terrain25d/RiskCalibration.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace terrain25d {
namespace {
constexpr double infinity=std::numeric_limits<double>::infinity();
using Node=std::uint32_t;

bool same(Point a,Point b) { return a.x==b.x && a.y==b.y; }

class CalibrationGraph {
public:
    CalibrationGraph(const Terrain &terrain,const SafetyField &field,Point start,Point goal,
                     const RiskCalibrationOptions &options,RiskCalibrationResult &result)
        : terrain_(terrain),field_(field),options_(options),result_(result),bounds_(field.bounds()) {
        const auto dimensions=[&](double extent) {
            const double cells=std::floor(extent/options.grid_spacing_m);
            if (!std::isfinite(cells) || cells<0 || cells>2000000)
                throw std::invalid_argument("calibration grid is too large");
            return static_cast<Node>(cells)+1;
        };
        nx_=dimensions(bounds_.xmax-bounds_.xmin);
        ny_=dimensions(bounds_.ymax-bounds_.ymin);
        const std::uint64_t count=static_cast<std::uint64_t>(nx_)*ny_;
        if (count>2000000) throw std::invalid_argument("calibration grid exceeds 2000000 nodes");
        start_id_=static_cast<Node>(count);goal_id_=start_id_+1;
        points_.reserve(count+2);
        for (Node j=0;j<ny_;++j) for (Node i=0;i<nx_;++i)
            points_.push_back({bounds_.xmin+i*options.grid_spacing_m,bounds_.ymin+j*options.grid_spacing_m});
        points_.push_back(start);points_.push_back(goal);
        states_.assign(points_.size(),-1);state_risk_.assign(points_.size(),infinity);
        start_links_=endpointLinks(start);goal_links_=endpointLinks(goal);
        start_adjacency_.assign(count,0);goal_adjacency_.assign(count,0);
        for (const Node node:start_links_) start_adjacency_[node]=1;
        for (const Node node:goal_links_) goal_adjacency_[node]=1;
        result_.graph_nodes=points_.size();
        edges_.reserve(std::min<std::uint64_t>(count*4,1000000));
    }

    void solve() {
        if (!stateValid(start_id_)) { result_.status="INVALID_START";return; }
        if (!stateValid(goal_id_)) { result_.status="INVALID_GOAL";return; }
        if (same(points_[start_id_],points_[goal_id_])) {
            finish(state_risk_[start_id_],{points_[start_id_]});return;
        }
        const double endpoint_bound=std::max(state_risk_[start_id_],state_risk_[goal_id_]);
        const double direct=edgeRisk(start_id_,goal_id_);
        // Every graph path includes both endpoints. A valid direct edge meeting
        // that lower bound is an exact minimax certificate on this graph.
        if (direct==endpoint_bound) {
            finish(direct,{points_[start_id_],points_[goal_id_]});return;
        }
        const Node none=std::numeric_limits<Node>::max();
        std::vector<double> distance(points_.size(),infinity);
        std::vector<Node> parent(points_.size(),none);
        std::vector<unsigned char> settled(points_.size(),0);
        // Risk remains the only minimax key. Goal distance breaks equal-risk
        // plateaus deterministically without becoming an edge cost/heuristic.
        using Item=std::tuple<double,double,Node>;
        std::priority_queue<Item,std::vector<Item>,std::greater<Item>> queue;
        distance[start_id_]=state_risk_[start_id_];
        queue.push({distance[start_id_],distanceXY(points_[start_id_],points_[goal_id_]),start_id_});
        while (!queue.empty()) {
            const auto [value,goalDistance,node]=queue.top();queue.pop();
            (void)goalDistance;
            if (settled[node] || value!=distance[node]) continue;
            settled[node]=1;++result_.settled_nodes;
            if (node==goal_id_) {
                std::vector<Point> witness;
                for (Node at=goal_id_;at!=none;at=parent[at]) {
                    witness.push_back(points_[at]);
                    if (witness.size()>points_.size()) throw std::logic_error("cyclic calibration witness");
                }
                std::reverse(witness.begin(),witness.end());
                finish(value,std::move(witness));return;
            }
            for (const Node other:neighbours(node)) {
                if (settled[other]) continue;
                const double weight=edgeRisk(node,other);
                if (!std::isfinite(weight)) continue;
                const double candidate=std::max(value,weight);
                // Strict updates prevent parent cycles across equal-cost edges.
                if (candidate<distance[other]) {
                    distance[other]=candidate;parent[other]=node;
                    queue.push({candidate,distanceXY(points_[other],points_[goal_id_]),other});
                }
            }
        }
    }
private:
    std::vector<Node> endpointLinks(Point point) const {
        std::vector<Node> result;
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x<bounds_.xmin ||
            point.x>bounds_.xmax || point.y<bounds_.ymin || point.y>bounds_.ymax) return result;
        const double h=options_.grid_spacing_m,r=options_.endpoint_connection_radius_m;
        const auto lo=[&](double p,double origin) { return std::max(0.,std::floor((p-origin-r)/h)); };
        const auto hi=[&](double p,double origin,Node n) { return std::min(double(n-1),std::ceil((p-origin+r)/h)); };
        const int min_i=static_cast<int>(lo(point.x,bounds_.xmin));
        const int max_i=static_cast<int>(hi(point.x,bounds_.xmin,nx_));
        const int min_j=static_cast<int>(lo(point.y,bounds_.ymin));
        const int max_j=static_cast<int>(hi(point.y,bounds_.ymin,ny_));
        // Include numerical ties at the declared radius without an added
        // physical-width connection band. The edge itself remains fully checked.
        for (int j=min_j;j<=max_j;++j) for (int i=min_i;i<=max_i;++i) {
            const Node node=static_cast<Node>(j)*nx_+static_cast<Node>(i);
            const Point candidate=points_[node];
            const double magnitude=std::max({1.,std::abs(point.x),std::abs(point.y),
                                             std::abs(candidate.x),std::abs(candidate.y),r});
            const double limit=r+16.*(std::nextafter(magnitude,infinity)-magnitude);
            if (distanceXY(point,points_[node])<=limit) result.push_back(node);
        }
        return result;
    }
    std::vector<Node> neighbours(Node node) const {
        if (node==start_id_) { auto result=start_links_;result.push_back(goal_id_);return result; }
        if (node==goal_id_) { auto result=goal_links_;result.push_back(start_id_);return result; }
        std::vector<Node> result;result.reserve(10);
        const int i=static_cast<int>(node%nx_),j=static_cast<int>(node/nx_);
        for (int dj=-1;dj<=1;++dj) for (int di=-1;di<=1;++di) {
            if (!(di||dj) || i+di<0 || j+dj<0 || i+di>=static_cast<int>(nx_) || j+dj>=static_cast<int>(ny_)) continue;
            result.push_back(static_cast<Node>(j+dj)*nx_+static_cast<Node>(i+di));
        }
        if (start_adjacency_[node]) result.push_back(start_id_);
        if (goal_adjacency_[node]) result.push_back(goal_id_);
        return result;
    }
    bool stateValid(Node node) {
        if (states_[node]>=0) return states_[node]!=0;
        ++result_.state_evaluations;states_[node]=0;
        const Point point=points_[node];
        const auto summary=field_.edgeSummary(point,point);
        if (!summary.supported || !summary.maximum_risk || *summary.maximum_risk>options_.risk_ceiling ||
            !terrain_.validState(point)) return false;
        const double slope=terrain_.maximumSweptSlope(point,point);
        if (!std::isfinite(slope) || slope>options_.slope_cap_deg) return false;
        states_[node]=1;state_risk_[node]=*summary.maximum_risk;return true;
    }
    double edgeRisk(Node a,Node b) {
        if (a>b) std::swap(a,b);
        const std::uint64_t key=(static_cast<std::uint64_t>(a)<<32)|b;
        const auto found=edges_.find(key);
        if (found!=edges_.end()) { ++result_.edge_cache_hits;return found->second; }
        ++result_.edge_evaluations;
        double risk=infinity;
        if (stateValid(a) && stateValid(b)) {
            const auto summary=field_.edgeSummary(points_[a],points_[b]);
            if (summary.supported && summary.maximum_risk && *summary.maximum_risk<=options_.risk_ceiling &&
                terrain_.validMotion(points_[a],points_[b])) {
                const double slope=terrain_.maximumSweptSlope(points_[a],points_[b]);
                if (std::isfinite(slope) && slope<=options_.slope_cap_deg) risk=*summary.maximum_risk;
            }
        }
        edges_.emplace(key,risk);return risk;
    }
    void finish(double bottleneck,std::vector<Point> witness) {
        result_.status="CALIBRATED";result_.b_graph=bottleneck;
        result_.effective_risk_cap=std::min(options_.risk_ceiling,bottleneck+options_.risk_slack);
        result_.witness=std::move(witness);
    }
    const Terrain &terrain_;
    const SafetyField &field_;
    const RiskCalibrationOptions &options_;
    RiskCalibrationResult &result_;
    Rect bounds_;
    Node nx_,ny_,start_id_,goal_id_;
    std::vector<Point> points_;
    std::vector<signed char> states_;
    std::vector<double> state_risk_;
    std::vector<Node> start_links_,goal_links_;
    std::vector<unsigned char> start_adjacency_,goal_adjacency_;
    std::unordered_map<std::uint64_t,double> edges_;
};
}

RiskCalibrationResult calibrateRisk(const Terrain &terrain,const SafetyField &field,
                                   Point start,Point goal,const RiskCalibrationOptions &options) {
    const auto begin=std::chrono::steady_clock::now();
    if (!std::isfinite(options.grid_spacing_m) || options.grid_spacing_m<=0 ||
        !std::isfinite(options.endpoint_connection_radius_m) || options.endpoint_connection_radius_m<0 ||
        !std::isfinite(options.risk_ceiling) || options.risk_ceiling<0 || options.risk_ceiling>40 ||
        !std::isfinite(options.slope_cap_deg) || options.slope_cap_deg<0 || options.slope_cap_deg>35 ||
        !std::isfinite(options.risk_slack) || options.risk_slack<0)
        throw std::invalid_argument("invalid risk calibration options");
    const Rect a=terrain.bounds(),b=field.bounds();
    if (a.xmin!=b.xmin || a.ymin!=b.ymin || a.xmax!=b.xmax || a.ymax!=b.ymax)
        throw std::invalid_argument("calibration field/terrain bounds mismatch");
    RiskCalibrationResult result;
    CalibrationGraph graph(terrain,field,start,goal,options,result);
    graph.solve();
    result.preprocessing_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    return result;
}
} // namespace terrain25d
