#include "terrain25d/Terrain.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
double numeric(const std::string &text) {
    std::size_t used=0;const double value=std::stod(text,&used);
    if (used!=text.size() || !std::isfinite(value)) throw std::invalid_argument("invalid number");
    return value;
}
// This is a reproducible comparison fingerprint, not an integrity/security hash.
// Integers are mixed explicitly little endian so the representation is portable.
void hashWord(std::uint64_t &hash,std::uint64_t word) {
    for (unsigned k=0;k<8;++k) {hash^=(word>>(k*8))&255u;hash*=1099511628211ull;}
}
void hashDouble(std::uint64_t &hash,double value) {
    std::uint64_t word=0;static_assert(sizeof(word)==sizeof(value),"binary64 required");
    std::memcpy(&word,&value,sizeof(word));hashWord(hash,word);
}
}
int main(int argc,char **argv) {
    try {
        std::string filename,policy_name="legacy";
        terrain25d::TraversabilityConfig config;config.lambda=1.;
        bool dump=false;
        for (int k=1;k<argc;++k) {
            const std::string option=argv[k];
            if (option=="--dump-features") {dump=true;continue;}
            if (++k>=argc) throw std::invalid_argument("missing value for "+option);
            const std::string value=argv[k];
            if (option=="--map") filename=value;
            else if (option=="--policy") policy_name=value;
            else if (option=="--lambda") config.lambda=numeric(value);
            else if (option=="--slope-weight") config.slope_weight=numeric(value);
            else if (option=="--clearance-weight") config.clearance_weight=numeric(value);
            else if (option=="--clearance-comfort") config.clearance_comfort_m=numeric(value);
            else throw std::invalid_argument("unknown option: "+option);
        }
        if (filename.empty() || (policy_name!="legacy" && policy_name!="skip_dominated"))
            throw std::invalid_argument("usage: a3_preprocessing_benchmark --map MAP --policy legacy|skip_dominated [--dump-features]");
        const auto began=std::chrono::steady_clock::now();
        auto map=terrain25d::Terrain::load(filename);
        const double load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
        map.enableTraversability(config,policy_name=="legacy" ? terrain25d::TraversabilityBuildPolicy::Legacy :
            terrain25d::TraversabilityBuildPolicy::SkipDominated);
        const auto &s=map.traversabilityBuildStats();
        std::uint64_t hash=14695981039346656037ull;
        hashWord(hash,static_cast<std::uint64_t>(map.nx()));hashWord(hash,static_cast<std::uint64_t>(map.ny()));
        for (int j=0;j<map.ny()-1;++j) for (int i=0;i<map.nx()-1;++i) {
            const auto cell=map.featureCell(i,j);hashWord(hash,cell.available?1:0);
            hashDouble(hash,cell.slope_penalty);hashDouble(hash,cell.clearance_penalty);
        }
        std::ostringstream fingerprint;fingerprint<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;
        std::cout<<std::setprecision(17)<<"{\"build_policy\":\""<<policy_name
            <<"\",\"field_fingerprint\":\""<<fingerprint.str()<<"\",\"fingerprint_algorithm\":\"fnv1a64_binary64_le_v1\""
            <<",\"nx\":"<<map.nx()<<",\"ny\":"<<map.ny()<<",\"load_seconds\":"<<load_seconds
            <<",\"preprocessing_seconds\":"<<map.traversabilityPreprocessingSeconds()<<",\"build_stats\":{"
            <<"\"available_cells\":"<<s.available_cells<<",\"candidate_cells\":"<<s.candidate_cells
            <<",\"candidate_triangles\":"<<s.candidate_triangles<<",\"slope_distance_calls\":"<<s.slope_distance_calls
            <<",\"dominated_cell_skips\":"<<s.dominated_cell_skips
            <<",\"dominated_triangle_skips\":"<<s.dominated_triangle_skips
            <<",\"slope_saturated_cells\":"<<s.slope_saturated_cells
            <<",\"obstacle_distance_calls\":"<<s.obstacle_distance_calls
            <<",\"clearance_saturated_cells\":"<<s.clearance_saturated_cells
            <<",\"slope_extraction_seconds\":"<<s.slope_extraction_seconds
            <<",\"slope_envelope_seconds\":"<<s.slope_envelope_seconds
            <<",\"clearance_seconds\":"<<s.clearance_seconds<<",\"total_seconds\":"<<s.total_seconds<<'}';
        if (dump) {
            std::cout<<",\"cells\":[";bool first=true;
            for (int j=0;j<map.ny()-1;++j) for (int i=0;i<map.nx()-1;++i) {
                const auto cell=map.featureCell(i,j);if (!first) std::cout<<',';first=false;
                std::cout<<'['<<(cell.available?"true":"false")<<','<<cell.slope_penalty<<','<<cell.clearance_penalty<<']';
            }
            std::cout<<']';
        }
        std::cout<<"}\n";return 0;
    } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 2;}
}
