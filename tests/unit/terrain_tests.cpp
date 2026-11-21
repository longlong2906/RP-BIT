#include "terrain25d/Terrain.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace terrain25d;
void require(bool ok,const char *why) { if (!ok) throw std::runtime_error(why); }
void near(double a,double b) { require(std::isfinite(a)&&std::isfinite(b)&&
    std::abs(a-b)<=1e-8+1e-8*std::max(std::abs(a),std::abs(b)),"numerical mismatch"); }
Terrain make(std::function<double(double,double)> f, std::vector<Rect> obs={},
             double radius=.1,double scale=1,double offset=0) {
    std::vector<double> h;
    for(int j=0;j<=10;++j) for(int i=0;i<=10;++i) h.push_back(scale*f(i,j)+offset);
    for(auto &r:obs) {r.xmin*=scale;r.xmax*=scale;r.ymin*=scale;r.ymax*=scale;}
    return Terrain(11,11,0,0,scale,scale,h,std::vector<int>(100),obs,radius*scale,0,89);
}
int main() {
    try {
        auto flat=make([](double,double){return 3.;});
        Point a{1.2,2.3},b{8.6,7.1};
        near(flat.surfaceEdge(a,b).cost,distanceXY(a,b));
        std::cout<<"T01 PASS flat\n";
        auto plane=make([](double x,double y){return .3*x-.2*y+4;});
        near(plane.surfaceEdge(a,b).cost,std::hypot(distanceXY(a,b),.3*(b.x-a.x)-.2*(b.y-a.y)));
        std::cout<<"T02 PASS tilted plane\n";
        auto hill=make([](double x,double){return 5-std::abs(x-5);});
        Point c{1,5},d{9,5};
        near(hill.surfaceEdge(c,d).cost,8*std::sqrt(2.));
        require(hill.surfaceEdge(c,d).cost>hill.lowerBound(c,d,true),"chord mistaken for terrain");
        std::cout<<"T03 PASS folded hill\n";
        std::mt19937 rng(7351); std::uniform_real_distribution<double> u(.2,9.8);
        for(int k=0;k<300;++k) {
            Point p{u(rng),u(rng)},q{u(rng),u(rng)};
            Point mid{.37*p.x+.63*q.x,.37*p.y+.63*q.y};
            double cost=hill.surfaceEdge(p,q).cost;
            require(distanceXY(p,q)<=hill.lowerBound(p,q,true)+1e-8,"XY lower bound");
            require(hill.lowerBound(p,q,true)<=cost+1e-8,"phi lower bound");
            near(cost,hill.surfaceEdge(q,p).cost);
            near(cost,hill.surfaceEdge(p,mid).cost+hill.surfaceEdge(mid,q).cost);
        }
        std::cout<<"T04 PASS lower bounds/symmetry/split (300 edges)\n";
        for(auto p:std::vector<Point>{{1,1},{1,5},{5,5},{9,9},{2.5,2.5}}) {
            near(hill.surfaceEdge(p,p).cost,0);
            near(flat.surfaceEdge(p,{9,9}).cost,distanceXY(p,{9,9}));
        }
        std::cout<<"T05 PASS vertices/diagonals/zero length\n";
        auto wall=make([](double,double){return 0.;},{{4.99,0,5.01,10}});
        require(wall.validState(c)&&wall.validState(d)&&!wall.validMotion(c,d),"missed thin wall");
        require(wall.surfaceEdge(c,d).has_data,"geometry cost should be separate from validity");
        std::cout<<"T06 PASS thin obstacle\n";
        for(double width:{.19,.20,.21}) {
            auto corridor=make([](double,double){return 0.;},{{0,0,10,5-width/2},{0,5+width/2,10,10}});
            require(corridor.validMotion(c,d)==(width>.20),"corridor/contact failure");
        }
        std::cout<<"T07 PASS footprint and contact\n";
        auto missing=make([](double x,double y){return (x==5 && y==5)?
            std::numeric_limits<double>::quiet_NaN():0.;});
        double z=123.;
        require(!missing.height({5,5},z)&&!missing.validMotion(c,d),"NoData crossed");
        require(!flat.validState({-.1,5})&&!flat.validState({.1,5}),"outside/contact accepted");
        require(!flat.validMotion({-1,5},d),"invalid start");
        bool threw=false; try { Terrain bad(2,2,0,0,0,1,{0,0,0,0},{0},{},0,0,35); }
        catch(const std::exception&) {threw=true;} require(threw,"bad spacing accepted");
        auto steep=Terrain(2,2,0,0,10,10,{0,100,0,100},{0},{},.1,0,35);
        require(!steep.validState({5,5}),"slope threshold ignored");
        std::cout<<"T08 PASS NoData/bounds/input/slope\n";
        auto shifted=make([](double x,double){return 5-std::abs(x-5);},{},.1,1,10000);
        near(hill.surfaceEdge(a,b).cost,shifted.surfaceEdge(a,b).cost);
        near(hill.lowerBound(a,b,true),shifted.lowerBound(a,b,true));
        std::cout<<"T09 PASS height offset\n";
        auto scaled=make([](double x,double){return 5-std::abs(x-5);},{},.1,7);
        near(scaled.surfaceEdge({a.x*7,a.y*7},{b.x*7,b.y*7}).cost,7*hill.surfaceEdge(a,b).cost);
        require(scaled.validState({7*a.x,7*a.y})==hill.validState(a),"scale validity");
        std::cout<<"T10 PASS uniform scale\n";
        for (auto *terrain:std::vector<Terrain*>{&flat,&plane,&hill,&missing,&scaled}) {
            std::vector<std::pair<Point,Point>> edges{{a,b},{c,d},{{1,1},{9,9}},
                {{1,2},{9,2}},{{3,3},{3,3}},{{-1,5},{9,5}},{{5,5},{5,5}}};
            for (int k=0;k<300;++k) edges.push_back({{u(rng),u(rng)},{u(rng),u(rng)}});
            for (const auto &edge:edges) {
                terrain->resetCounters();
                auto reference=terrain->surfaceEdge(edge.first,edge.second);
                const auto original=terrain->counters();
                terrain->resetCounters();
                const auto scalar=terrain->surfaceCost(edge.first,edge.second);
                const auto counters=terrain->counters();
                require(scalar==reference.cost,"scalar changed exact cost/summation order");
                require(counters.height_queries==original.height_queries &&
                    counters.triangle_visits==original.triangle_visits &&
                    counters.full_surface_cost_calls==1,"scalar changed counter semantics");
                if (std::isfinite(scalar)) {
                    Point mid{.37*edge.first.x+.63*edge.second.x,.37*edge.first.y+.63*edge.second.y};
                    near(scalar,terrain->surfaceCost(edge.second,edge.first));
                    near(scalar,terrain->surfaceCost(edge.first,mid)+terrain->surfaceCost(mid,edge.second));
                }
            }
        }
        std::cout<<"G_SCALAR PASS exact agreement/counters/reverse/split/NoData (1535 edges)\n";
        std::vector<Rect> clustered;
        for (int j=1;j<8;++j) for (int i=1;i<8;++i)
            clustered.push_back({double(i),double(j),i+.05,j+.05});
        auto indexed_obstacles=make([](double,double){return 0.;},clustered);
        for (auto *terrain:std::vector<Terrain*>{&flat,&wall,&missing,&steep,&indexed_obstacles}) {
            require(!terrain->collisionIndexEnabled(),"index must be disabled by default");
            std::vector<std::pair<Point,Point>> edges{{a,b},{c,d},{{1,1},{9,9}},
                {{.1,5},{.1,5}},{{4.89,1},{4.89,9}},{{3.9,4},{3.9,6}},
                {{2,2},{2,2}},{{5,5},{5,5}},{{-.1,5},{9,5}}};
            for (int k=0;k<400;++k) edges.push_back({{u(rng),u(rng)},{u(rng),u(rng)}});
            std::vector<bool> reference;
            for (const auto &edge:edges) reference.push_back(terrain->validMotion(edge.first,edge.second));
            const auto calls_before=terrain->counters().motion_validity_calls;
            terrain->enableCollisionIndex();
            const double build_seconds=terrain->collisionIndexBuildSeconds();
            require(terrain->collisionIndexEnabled() && build_seconds>=0,"index was not built");
            terrain->enableCollisionIndex();
            require(build_seconds==terrain->collisionIndexBuildSeconds(),"index enable must be idempotent");
            require(calls_before==terrain->counters().motion_validity_calls,"building index changed counters");
            terrain->resetCounters();
            for (std::size_t k=0;k<edges.size();++k) {
                const auto &edge=edges[k];
                require(terrain->validMotion(edge.first,edge.second)==reference[k],"indexed collision mismatch");
                require(terrain->validMotion(edge.second,edge.first)==reference[k],"indexed reverse mismatch");
                if (reference[k]) {
                    Point mid{.37*edge.first.x+.63*edge.second.x,.37*edge.first.y+.63*edge.second.y};
                    require(terrain->validMotion(edge.first,mid) && terrain->validMotion(mid,edge.second),
                        "indexed split of valid edge rejected");
                }
            }
            if (terrain==&indexed_obstacles)
                require(terrain->counters().bvh_node_visits>edges.size() &&
                    terrain->counters().collision_narrow_checks>0,"BVH traversal was not exercised");
            require(terrain->counters().collision_candidates==terrain->counters().collision_narrow_checks,
                "collision candidate accounting mismatch");
            terrain->resetCounters();
            require(terrain->collisionIndexEnabled() && terrain->counters().bvh_node_visits==0,
                "counter reset changed index or left node visits");
        }
        for (double width:{.19,.20,.21}) {
            auto corridor=make([](double,double){return 0.;},{{0,0,10,5-width/2},{0,5+width/2,10,10}});
            corridor.enableCollisionIndex();
            require(corridor.validMotion(c,d)==(width>.20),"indexed corridor/contact failure");
        }
        std::cout<<"G_BVH PASS scan agreement/reverse/split/contact/NoData/empty/index lifecycle (2045 edges)\n";
        // Terrain::load: every accepted number form gives the exact value; malformed files are rejected.
        const auto folder=std::filesystem::temp_directory_path()/("terrain_load_"+std::to_string(std::random_device{}()));
        std::filesystem::create_directories(folder);
        const auto file=(folder/"map.t25d").string();
        const auto save=[&](const std::string &text) { std::ofstream(file,std::ios::binary)<<text; };
        const auto loads=[&](const std::string &text) {
            save(text);
            try { Terrain::load(file); } catch(const std::exception&) { return false; }
            return true;
        };
        const auto replaced=[](std::string text,const std::string &from,const std::string &to) {
            for (std::size_t at=0;(at=text.find(from,at))!=std::string::npos;at+=to.size()) text.replace(at,from.size(),to);
            return text;
        };
        const std::string head="T25D 1\nunits m\ndiagonal SW_NE\ngrid 3 3 0 0 1 1\nmodel 0.1 0 60\nheights\n";
        const std::string rows="+0.75 .5 1\n0x1p-2 0.30000000000000004 1.\n5e-1 -0.125 1\n";
        const std::string tail="mask\n0 0\n0 0\nrectangles 1\n0.2 0.2 0.4 0.4\nend\n";
        try {
            for (const auto &text:{head+rows+tail,replaced(replaced(head+rows+tail,"\n","\r\n")," ","\t")}) {
                save(text);
                const auto map=Terrain::load(file);
                // Read back at the nodes whose value Terrain::height returns without arithmetic.
                const double expected[2][2]={{.75,.5},{.25,.30000000000000004}};
                for (int j=0;j<2;++j) for (int i=0;i<2;++i) {
                    double value=-1.;
                    require(map.height({double(i),double(j)},value)&&value==expected[j][i],"load changed a height");
                }
                require(map.nx()==3&&map.ny()==3&&map.radius()==.1&&!map.validState({.3,.3})&&map.validState({1.5,1.5}),
                        "load changed the grid, the model or an obstacle");
            }
            save(head+replaced(rows,"0.30000000000000004","nodata")+tail);
            double value=0.;
            require(!Terrain::load(file).height({.5,.5},value),"nodata height was read as data");
            require(!loads(""),"empty terrain file accepted");
            require(!loads(replaced(head,"T25D","T25E")+rows+tail),"wrong magic accepted");
            require(!loads(head+rows+replaced(tail,"end\n","")),"truncated terrain file accepted");
            require(!loads(head+rows+tail+"0\n"),"trailing data accepted");
            for (const char *bad:{"inf","nan","abc","1.5x","1e999",""})
                require(!loads(head+replaced(rows,"-0.125",bad)+tail),"invalid height accepted");
            require(!loads(replaced(head,"grid 3 3","grid 3.0 3")+rows+tail),"fractional grid size accepted");
            require(!loads(replaced(head,"0 0 1 1","0 0 nodata 1")+rows+tail),"nodata spacing accepted");
            for (const char *bad:{"2","1.0","-1","x"})
                require(!loads(head+rows+replaced(tail,"mask\n0 0","mask\n0 "+std::string(bad))),"invalid mask accepted");
            require(!loads(head+rows+replaced(tail,"rectangles 1","rectangles -1")),"negative rectangle count accepted");
            require(!loads(head+rows+replaced(tail,"rectangles 1","rectangles 2")),"missing rectangle accepted");
        } catch(...) {std::filesystem::remove_all(folder);throw;}
        std::filesystem::remove_all(folder);
        bool missingFile=false; try { Terrain::load(file); } catch(const std::exception&) {missingFile=true;}
        require(missingFile,"missing terrain file accepted");
        std::cout<<"G_LOAD PASS number forms/CRLF/tabs/nodata/malformed files\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
