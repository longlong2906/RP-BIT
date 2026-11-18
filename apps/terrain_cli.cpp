#include "terrain25d/Terrain.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace {
double argument(const char *text) {
    std::size_t used=0;const std::string value(text);const double result=std::stod(value,&used);
    if(used!=value.size() || !std::isfinite(result))throw std::runtime_error("invalid numeric argument");
    return result;
}
void number(double value) {if(std::isfinite(value))std::cout<<value;else std::cout<<"null";}
int traversabilityCommand(int argc,char **argv) {
    const bool cells=std::string(argv[1])=="features";
    if(argc!=(cells?7:11))throw std::runtime_error("usage: terrain_cli traversability MAP ax ay bx by lambda slope_weight clearance_weight comfort; or features MAP lambda slope_weight clearance_weight comfort");
    auto map=terrain25d::Terrain::load(argv[2]);
    const int base=cells?3:7;terrain25d::TraversabilityConfig config;
    config.lambda=argument(argv[base]);config.slope_weight=argument(argv[base+1]);
    config.clearance_weight=argument(argv[base+2]);config.clearance_comfort_m=argument(argv[base+3]);
    map.enableTraversability(config);std::cout<<std::setprecision(17);
    if(cells) {
        std::cout<<"{\"nx_cells\":"<<map.nx()-1<<",\"ny_cells\":"<<map.ny()-1<<",\"cells\":[";
        bool first=true;
        for(int j=0;j<map.ny()-1;++j)for(int i=0;i<map.nx()-1;++i) {
            const auto cell=map.featureCell(i,j);if(!first)std::cout<<',';first=false;
            std::cout<<"{\"available\":"<<(cell.available?"true":"false")<<",\"slope_penalty\":";
            if(cell.available)number(cell.slope_penalty);else std::cout<<"null";
            std::cout<<",\"clearance_penalty\":";
            if(cell.available)number(cell.clearance_penalty);else std::cout<<"null";
            std::cout<<'}';
        }
        std::cout<<"]}\n";
    } else {
        const terrain25d::Point a{argument(argv[3]),argument(argv[4])},b{argument(argv[5]),argument(argv[6])};
        const auto edge=map.edgeCostBreakdown(a,b);
        std::cout<<"{\"valid\":"<<(map.validMotion(a,b)?"true":"false")
            <<",\"has_data\":"<<(edge.has_data?"true":"false")<<",\"cost_terrain_m\":";
        number(edge.cost_terrain_m);std::cout<<",\"risk_slope_integral_m\":";number(edge.risk_slope_integral_m);
        std::cout<<",\"risk_clearance_integral_m\":";number(edge.risk_clearance_integral_m);
        std::cout<<",\"risk_integral_m\":";number(edge.risk_integral_m);
        std::cout<<",\"objective_value\":";number(edge.objective_value);std::cout<<"}\n";
    }
    return 0;
}
}
int main(int argc,char **argv) {
    try {
        if(argc>=2 && (std::string(argv[1])=="traversability" || std::string(argv[1])=="features"))
            return traversabilityCommand(argc,argv);
        if (argc==2 && std::string(argv[1])=="--version") {
            std::cout<<"terrain25d 0.1.0; state=xy; diagonal=SW_NE; objective=surface_length\n"; return 0;
        }
        if (argc<2) throw std::runtime_error("usage: terrain_cli MAP.t25d [--collision-index] [--scalar-cost] < edges.txt (ax ay bx by per line)");
        bool indexed=false,scalar=false;
        for (int k=2;k<argc;++k) {
            const std::string option=argv[k];
            if (option=="--collision-index") indexed=true;
            else if (option=="--scalar-cost") scalar=true;
            else throw std::runtime_error("unknown option: "+option);
        }
        auto map=terrain25d::Terrain::load(argv[1]);
        if (indexed) map.enableCollisionIndex();
        std::cout<<std::setprecision(17);
        terrain25d::Point a{},b{};
        while (std::cin>>a.x>>a.y>>b.x>>b.y) {
            auto surf=map.surfaceEdge(a,b);
            terrain25d::EdgeEvaluation e;
            if (scalar) {
                const bool clear=map.validMotion(a,b);
                const double cost=map.surfaceCost(a,b);
                e={clear && std::isfinite(cost),cost,surf.points.empty()?0:surf.points.size()-1,
                    !std::isfinite(cost)?surf.reason:clear?"ok":"forbidden_capsule"};
            } else e=map.evaluate(a,b);
            std::cout<<"{\"valid\":"<<(e.valid?"true":"false")<<",\"cost\":";
            if (std::isfinite(e.cost)) std::cout<<e.cost; else std::cout<<"null";
            std::cout<<",\"faces\":"<<e.faces<<",\"reason\":\""<<e.reason<<"\",\"surface\":[";
            bool first=true;
            for (auto p:surf.points) { if (!first) std::cout<<','; first=false;
                std::cout<<'['<<p.x<<','<<p.y<<','<<p.z<<']'; }
            std::cout<<"],\"collision_index_enabled\":"<<(map.collisionIndexEnabled()?"true":"false")
                     <<",\"scalar_cost\":"<<(scalar?"true":"false")<<"}\n";
        }
        if (!std::cin.eof()) throw std::runtime_error("invalid edge input");
        return 0;
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 2; }
}
