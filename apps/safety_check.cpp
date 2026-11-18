#include "terrain25d/Safety.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

using namespace terrain25d;
namespace {
double numeric(const std::string &text) {
    std::size_t used=0;
    const double value=std::stod(text,&used);
    if (used!=text.size() || !std::isfinite(value)) throw std::runtime_error("invalid finite number");
    return value;
}
std::string escaped(const std::string &text) {
    std::ostringstream out;
    for (unsigned char c:text) {
        if (c=='"' || c=='\\') out << '\\' << c;
        else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
        else out << c;
    }
    return out.str();
}
std::vector<Point> readPath(const std::string &path) {
    std::ifstream stream(path);
    std::string row;
    if (!stream || !std::getline(stream,row)) throw std::runtime_error("missing path CSV");
    if (row.size()>=3 && row.substr(0,3)=="\xef\xbb\xbf") row.erase(0,3);
    if (!row.empty() && row.back()=='\r') row.pop_back();
    if (row!="x,y") throw std::runtime_error("path CSV header must be x,y");
    std::vector<Point> points;
    while (std::getline(stream,row)) {
        if (!row.empty() && row.back()=='\r') row.pop_back();
        const auto comma=row.find(',');
        if (comma==std::string::npos || row.find(',',comma+1)!=std::string::npos)
            throw std::runtime_error("path CSV must contain two coordinates per row");
        points.push_back({numeric(row.substr(0,comma)),numeric(row.substr(comma+1))});
    }
    if (points.empty()) throw std::runtime_error("empty path CSV");
    return points;
}
}
int main(int argc,char **argv) {
    try {
        std::map<std::string,std::string> arguments;
        for (int k=1;k<argc;k+=2) {
            const std::string key=argv[k];
            if (k+1>=argc || (key!="--map" && key!="--field" && key!="--risk-cap" && key!="--slope-cap" && key!="--path") ||
                !arguments.emplace(key,argv[k+1]).second)
                throw std::runtime_error("usage: safety_check --map MAP --field FIELD --risk-cap NUMBER --slope-cap NUMBER --path CSV");
        }
        if (arguments.size()!=5) throw std::runtime_error("all safety_check options are required");
        const auto terrain=Terrain::load(arguments.at("--map"));
        const auto field=std::make_shared<const SafetyField>(SafetyField::load(arguments.at("--field")));
        const SafetyPolicy policy(terrain,field,numeric(arguments.at("--risk-cap")),numeric(arguments.at("--slope-cap")));
        const auto path=readPath(arguments.at("--path"));
        std::string reason;
        double maximum=0;
        double maximumRisk=0;
        bool supported=true;
        const std::size_t edges=std::max<std::size_t>(1,path.size()-1);
        for (std::size_t k=0;k<edges;++k) {
            const Point a=path[k],b=path[std::min(k+1,path.size()-1)];
            const auto current=policy.rejectionReason(a,b);
            if (reason.empty()) reason=current;
            maximum=std::max(maximum,terrain.maximumSweptSlope(a,b));
            const auto summary=field->edgeSummary(a,b);
            supported=supported&&summary.supported&&summary.maximum_risk.has_value();
            if(summary.maximum_risk)maximumRisk=std::max(maximumRisk,*summary.maximum_risk);
        }
        std::cout << std::setprecision(17) << "{\"valid\":" << (reason.empty()?"true":"false")
                  << ",\"reason\":\"" << escaped(reason) << "\",\"maximum_swept_slope_deg\":";
        if (std::isfinite(maximum)) std::cout << maximum; else std::cout << "null";
        std::cout << ",\"supported\":" << (supported?"true":"false") << ",\"maximum_risk\":";
        if(supported)std::cout << maximumRisk;else std::cout << "null";
        std::cout << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "{\"error\":\"" << escaped(error.what()) << "\"}\n";
        return 2;
    }
}
