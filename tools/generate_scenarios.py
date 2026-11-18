import argparse
import math
from pathlib import Path
import random

from experiment_common import ROOT,digest,dump,load_config
from terrain_io import write_map
from independent_verifier import Oracle

GENERATOR_VERSION="2.0.0"
ADVANCED_GENERATOR_VERSION="3.0.0"
SEM_COMPARISON_GENERATOR_VERSION="4.0.0"
COMBINED_GENERATOR_VERSION="5.0.0"
CLOUD_GENERATOR_VERSION="6.0.0"

def pilot_geometry(family,seed,width,height):
    """Geometry fixed before observing planner results; witnesses never seed a planner."""
    shift=random.Random(seed).uniform(-.25,.25)
    rectangles=[]
    if family=="traversable_hill":
        # The direct edge clips the flank of a traversable cone. Moving outward
        # avoids enough ascent to compensate the slightly longer XY path.
        fn=lambda x,y:max(0.,4.-.5*math.hypot(x-20,y-(15+shift)))
        queries=[("q0",[4.,10+shift],[36.,10+shift],
            [[[4.,10+shift],[20.,8+shift],[36.,10+shift]],
             [[4.,10+shift],[36.,10+shift]]],"valid_direct_terrain_tradeoff"),
            ("q1",[4.,30.],[36.,30.],[[[4.,30.],[36.,30.]]],"easy_control")]
        params={"geometry":"traversable_conical_hill","height_m":4.,"radial_slope":.5,"center_xy":[20.,15+shift],"radius_m":8.}
    elif family in ("ridge_pass","narrow_passage"):
        if family=="ridge_pass":
            fn=lambda x,y:max(0.,3.-.5*abs(x-20.))
            low,high=11.5+shift,30.5+shift;gaps=[(10+shift,13+shift),(27+shift,34+shift)]
        else:
            fn=lambda x,y:.04*x
            low,high=16.6+shift,30.5+shift;gaps=[(16+shift,17.2+shift),(28+shift,33+shift)]
        rectangles=[(19.7,0,20.3,gaps[0][0]),(19.7,gaps[0][1],20.3,gaps[1][0]),(19.7,gaps[1][1],20.3,height)]
        queries=[]
        for query_id,y in (("q0",20.),("q1",23.)):
            start=[4.,y];goal=[36.,y]
            witnesses=[[start,[18.,cross],[22.,cross],goal] for cross in (low,high)]
            queries.append((query_id,start,goal,witnesses,"blocked_direct_competing_narrow_and_wide_routes"))
        params={"geometry":"two_pass_wall","gaps_y":gaps,"centerline_clearance_widths_m":[b-a-.8 for a,b in gaps],"ridge_height_m":3. if family=="ridge_pass" else 0.}
    elif family=="rolling_terrain":
        fn=lambda x,y:.8*math.sin((x+seed%10)/4.)*math.cos(y/6.)+.04*x
        rectangles=[(18+shift,14,22+shift,25)]
        queries=[("q0",[4.,20.],[36.,20.],
            [[[4.,20.],[16.,11.],[24.,11.],[36.,20.]],[[4.,20.],[16.,28.],[24.,28.],[36.,20.]]],"blocked_direct_competing_routes"),
            ("q1",[4.,7.],[36.,7.],[[[4.,7.],[36.,7.]]],"valid_direct_rolling_control")]
        params={"geometry":"rolling_sinusoid_with_central_obstacle","amplitude_m":.8,"x_scale_m":4.,"y_scale_m":6.,"phase_m":seed%10}
    elif family=="mixed_obstacles":
        fn=lambda x,y:None if 12.<=x<=14. and 18.<=y<=22. else .04*x+.45*math.sin(y/6.)
        rectangles=[(17+shift,12,23+shift,25),(10,26,13,34),(27,6,30,13)]
        queries=[]
        for query_id,y in (("q0",18.),("q1",21.)):
            start=[4.,y];goal=[36.,y]
            witnesses=[[start,[15.,9.],[25.,9.],[25.,16.],goal],
                [start,[15.,25.8],[25.,28.],goal]]
            queries.append((query_id,start,goal,witnesses,"blocked_direct_competing_routes"))
        params={"geometry":"three_obstacles_and_nodata_on_tilted_rolling_surface","central_x_shift_m":shift,
            "nodata_node_rectangle":[12.,18.,14.,22.],"nodata_cell_policy":"whole_cell_if_any_node_missing"}
    else:raise ValueError("unsupported pilot family: "+family)
    return fn,rectangles,queries,params

def generate_pilot(config,output):
    p=config["profiles"]["pilot"];model=config["model"]
    width,height=p["map_extent_m"];spacing=p["grid_spacing_m"]
    nx=round(width/spacing)+1;ny=round(height/spacing)+1
    maps=[]
    for family in p["families"]:
        for seed in p["map_seeds"]:
            map_id=f"{family}_{seed}";path=output/(map_id+".t25d")
            fn,rectangles,definitions,params=pilot_geometry(family,seed,width,height)
            heights=[[fn(i*spacing,j*spacing) for i in range(nx)] for j in range(ny)]
            meta=write_map(path,heights,spacing=(spacing,spacing),radius=model["robot_radius_m"],margin=model["safety_margin_m"],
                slope=model["max_slope_deg"],rectangles=rectangles,metadata={"map_id":map_id,"family":family,"generator_version":GENERATOR_VERSION,
                "generator_seed":seed,"parameters":params,"geometry_selection":"predeclared_deterministic_no_planner_results_used"})
            oracle=Oracle(path);queries=[]
            for query_id,start,goal,paths,label in definitions:
                witnesses=[]
                for index,xy in enumerate(paths):
                    checked=oracle.validate_path(xy,start,goal)
                    if not checked["valid"]:raise ValueError(f"invalid pilot witness {map_id}/{query_id}/{index}: {checked}")
                    witnesses.append({"witness_id":f"w{index}","path_xy":xy,"sha256":digest(xy),"validation":checked})
                direct=oracle.validate_path([start,goal],start,goal)
                best=min(witnesses,key=lambda w:w["validation"]["cost_terrain_m"])
                if label.startswith("blocked_direct") and direct["valid"]:raise ValueError("blocked-direct criterion not met")
                if label=="valid_direct_terrain_tradeoff" and not (direct["valid"] and best["validation"]["cost_terrain_m"]+1e-6<direct["cost_terrain_m"]):
                    raise ValueError(f"terrain tradeoff criterion not met: {best['validation']['cost_terrain_m']} vs {direct['cost_terrain_m']}")
                reference={"reference_type":"validated_witness","cost_terrain_m":best["validation"]["cost_terrain_m"],
                    "path_xy":best["path_xy"],"path_sha256":best["sha256"],"validation":best["validation"],"witness_id":best["witness_id"],
                    "map_sha256":meta["data_sha256"],"provenance":"generator geometric witness; independent Oracle; fixed before planner runs; not certified optimum"}
                query={"query_id":query_id,"start":start,"goal":goal,"geometry_label":label,"feasibility":"feasible_witness_validated",
                    "witnesses":witnesses,"direct_edge_validation":direct,"reference":reference}
                query["query_hash"]=digest({"start":start,"goal":goal,"map_hash":meta["data_sha256"],"model":model})
                queries.append(query)
            meta["queries"]=queries;dump(path.with_suffix(".json"),meta)
            maps.append({"map_id":map_id,"family":family,"map_seed":seed,"path":str(path.resolve()),"sha256":meta["data_sha256"],"queries":queries})
            print(f"Validated map witnesses: {map_id}",flush=True)
    manifest={"schema_version":2,"profile":"pilot","generator_version":GENERATOR_VERSION,"config_hash":digest(config),"maps":maps,
        "geometry_protocol":"fixed criteria: valid terrain tradeoff, blocked direct, competing routes and labeled easy controls; no planner-seed search"}
    dump(output/"manifest.json",manifest)
    return manifest


def generate_advanced(config,output):
    """Generate the frozen held-out A3-vs-advanced dataset.

    Every declared seed is attempted exactly once. A failed geometric
    qualification aborts generation and remains visible in the fresh output
    directory; it is never replaced by a more convenient seed.
    """
    p=config["profiles"]["advanced"];model=config["model"]
    width,height=p["map_extent_m"];spacing=p["grid_spacing_m"]
    nx=round(width/spacing)+1;ny=round(height/spacing)+1
    if abs((nx-1)*spacing-width)>1e-9 or abs((ny-1)*spacing-height)>1e-9:
        raise ValueError("extent must be divisible by spacing")
    declared_seeds=list(p["map_seeds"])
    maps=[]
    for family in p["families"]:
        for seed in declared_seeds:
            map_id=f"{family}_{seed}";path=output/(map_id+".t25d")
            fn,rectangles,definitions,params=pilot_geometry(family,seed,width,height)
            definitions=[definition for definition in definitions
                         if definition[0] in config["advanced_benchmark"]["required_query_ids"]]
            if [definition[0] for definition in definitions] != config["advanced_benchmark"]["required_query_ids"]:
                raise ValueError(f"advanced query set mismatch: {map_id}")
            heights=[[fn(i*spacing,j*spacing) for i in range(nx)] for j in range(ny)]
            meta=write_map(path,heights,spacing=(spacing,spacing),radius=model["robot_radius_m"],margin=model["safety_margin_m"],
                slope=model["max_slope_deg"],rectangles=rectangles,metadata={"map_id":map_id,"family":family,
                "generator_version":ADVANCED_GENERATOR_VERSION,"generator_seed":seed,"parameters":params,
                "dataset_role":"held_out_evaluation","seed_selection":"predeclared_no_replacement_no_planner_results_used"})
            oracle=Oracle(path);queries=[]
            for query_id,start,goal,paths,label in definitions:
                witnesses=[]
                for index,xy in enumerate(paths):
                    checked=oracle.validate_path(xy,start,goal)
                    if not checked["valid"]:
                        raise ValueError(f"advanced held-out qualification failed without replacement: {map_id}/{query_id}/{index}: {checked}")
                    witnesses.append({"witness_id":f"w{index}","path_xy":xy,"sha256":digest(xy),"validation":checked})
                direct=oracle.validate_path([start,goal],start,goal)
                best=min(witnesses,key=lambda witness:witness["validation"]["cost_terrain_m"])
                if label.startswith("blocked_direct") and direct["valid"]:
                    raise ValueError(f"advanced blocked-direct qualification failed without replacement: {map_id}/{query_id}")
                if label=="valid_direct_terrain_tradeoff" and not (direct["valid"] and
                        best["validation"]["cost_terrain_m"]+1e-6<direct["cost_terrain_m"]):
                    raise ValueError(f"advanced terrain-tradeoff qualification failed without replacement: {map_id}/{query_id}")
                reference={"reference_type":"validated_witness","cost_terrain_m":best["validation"]["cost_terrain_m"],
                    "path_xy":best["path_xy"],"path_sha256":best["sha256"],"validation":best["validation"],
                    "witness_id":best["witness_id"],"map_sha256":meta["data_sha256"],
                    "provenance":"held-out generator geometric witness; independent Oracle; fixed before planner runs; not certified optimum"}
                query={"query_id":query_id,"start":start,"goal":goal,"geometry_label":label,
                    "feasibility":"feasible_witness_validated","witnesses":witnesses,
                    "direct_edge_validation":direct,"reference":reference}
                query["query_hash"]=digest({"start":start,"goal":goal,"map_hash":meta["data_sha256"],"model":model})
                queries.append(query)
            meta["queries"]=queries;dump(path.with_suffix(".json"),meta)
            maps.append({"map_id":map_id,"family":family,"map_seed":seed,"path":str(path.resolve()),
                         "sha256":meta["data_sha256"],"queries":queries})
            print(f"Qualified held-out map without seed replacement: {map_id}",flush=True)
    expected={(family,seed) for family in p["families"] for seed in declared_seeds}
    realized={(item["family"],item["map_seed"]) for item in maps}
    if realized!=expected or len(maps)!=len(expected):
        raise ValueError("held-out generator did not realize every declared seed exactly once")
    manifest={"schema_version":3,"profile":"advanced","generator_version":ADVANCED_GENERATOR_VERSION,
        "config_hash":digest(config),"maps":maps,"held_out":True,"declared_map_seeds":declared_seeds,
        "seed_replacement_policy":"forbidden","seed_replacements":[],
        "qualification_policy":"all declared family/seed maps and q0 witnesses must pass before planner execution; abort, never replace"}
    dump(output/"manifest.json",manifest)
    return manifest


def generate_sem_comparison(config,output):
    """Generate the locked, fresh 50-map common-objective comparison set.

    Geometry-only qualification happens before planner execution.  Every
    declared family/seed is attempted exactly once; a failure aborts generation
    and is never repaired by selecting a replacement seed.
    """
    settings=config["profiles"]["sem_comparison"];model=config["model"]
    width,height=settings["map_extent_m"];spacing=settings["grid_spacing_m"]
    nx=round(width/spacing)+1;ny=round(height/spacing)+1
    if abs((nx-1)*spacing-width)>1e-9 or abs((ny-1)*spacing-height)>1e-9:
        raise ValueError("extent must be divisible by spacing")
    declared_seeds=list(settings["map_seeds"]);maps=[]
    required_queries=config["sem_comparison"]["required_query_ids"]
    for family in settings["families"]:
        for seed in declared_seeds:
            map_id=f"{family}_{seed}";path=output/(map_id+".t25d")
            fn,rectangles,definitions,params=pilot_geometry(family,seed,width,height)
            definitions=[definition for definition in definitions if definition[0] in required_queries]
            if [definition[0] for definition in definitions]!=required_queries:
                raise ValueError(f"sem comparison query set mismatch: {map_id}")
            heights=[[fn(i*spacing,j*spacing) for i in range(nx)] for j in range(ny)]
            meta=write_map(path,heights,spacing=(spacing,spacing),radius=model["robot_radius_m"],
                margin=model["safety_margin_m"],slope=model["max_slope_deg"],rectangles=rectangles,
                metadata={"map_id":map_id,"family":family,
                    "generator_version":SEM_COMPARISON_GENERATOR_VERSION,"generator_seed":seed,
                    "parameters":params,"dataset_role":"held_out_evaluation",
                    "seed_selection":"predeclared_no_replacement_no_planner_results_used"})
            oracle=Oracle(path);queries=[]
            for query_id,start,goal,paths,label in definitions:
                witnesses=[]
                for index,xy in enumerate(paths):
                    checked=oracle.validate_path(xy,start,goal)
                    if not checked["valid"]:
                        raise ValueError(f"sem comparison held-out qualification failed without replacement: "
                                         f"{map_id}/{query_id}/{index}: {checked}")
                    witnesses.append({"witness_id":f"w{index}","path_xy":xy,"sha256":digest(xy),
                                      "validation":checked})
                direct=oracle.validate_path([start,goal],start,goal)
                best=min(witnesses,key=lambda witness:witness["validation"]["cost_terrain_m"])
                if label.startswith("blocked_direct") and direct["valid"]:
                    raise ValueError(f"sem comparison blocked-direct qualification failed without replacement: "
                                     f"{map_id}/{query_id}")
                if label=="valid_direct_terrain_tradeoff" and not (direct["valid"] and
                        best["validation"]["cost_terrain_m"]+1e-6<direct["cost_terrain_m"]):
                    raise ValueError(f"sem comparison terrain-tradeoff qualification failed without replacement: "
                                     f"{map_id}/{query_id}")
                reference={"reference_type":"validated_witness",
                    "cost_terrain_m":best["validation"]["cost_terrain_m"],"path_xy":best["path_xy"],
                    "path_sha256":best["sha256"],"validation":best["validation"],
                    "witness_id":best["witness_id"],"map_sha256":meta["data_sha256"],
                    "provenance":"held-out generator geometric witness; independent Oracle; fixed before "
                                 "planner runs; not certified optimum"}
                query={"query_id":query_id,"start":start,"goal":goal,"geometry_label":label,
                    "feasibility":"feasible_witness_validated","witnesses":witnesses,
                    "direct_edge_validation":direct,"reference":reference}
                query["query_hash"]=digest({"start":start,"goal":goal,"map_hash":meta["data_sha256"],
                                             "model":model})
                queries.append(query)
            meta["queries"]=queries;dump(path.with_suffix(".json"),meta)
            maps.append({"map_id":map_id,"family":family,"map_seed":seed,"path":str(path.resolve()),
                         "sha256":meta["data_sha256"],"queries":queries})
            print(f"Qualified SEM-comparison held-out map without seed replacement: {map_id}",flush=True)
    expected={(family,seed) for family in settings["families"] for seed in declared_seeds}
    realized={(item["family"],item["map_seed"]) for item in maps}
    if realized!=expected or len(maps)!=len(expected):
        raise ValueError("SEM comparison generator did not realize every declared seed exactly once")
    manifest={"schema_version":4,"profile":"sem_comparison",
        "generator_version":SEM_COMPARISON_GENERATOR_VERSION,"config_hash":digest(config),"maps":maps,
        "held_out":True,"declared_map_seeds":declared_seeds,"seed_replacement_policy":"forbidden",
        "seed_replacements":[],
        "qualification_policy":"all declared family/seed maps and q0 witnesses must pass before planner "
                               "execution; abort, never replace"}
    dump(output/"manifest.json",manifest)
    return manifest


def generate_combined(config,output):
    """Generate the versioned held-out five-planner comparison set.

    The geometry and q0 qualification match the earlier held-out campaigns,
    but this is a fresh, separately versioned dataset.  Every configured seed
    is attempted once; failed qualification aborts instead of substituting a
    different map seed.
    """
    settings=config["profiles"]["combined"];model=config["model"]
    cloud=config.get("schema_version")==5
    generator_version=CLOUD_GENERATOR_VERSION if cloud else COMBINED_GENERATOR_VERSION
    width,height=settings["map_extent_m"];spacing=settings["grid_spacing_m"]
    nx=round(width/spacing)+1;ny=round(height/spacing)+1
    if abs((nx-1)*spacing-width)>1e-9 or abs((ny-1)*spacing-height)>1e-9:
        raise ValueError("extent must be divisible by spacing")
    declared_seeds=list(settings["map_seeds"]);maps=[]
    required_queries=config["combined_benchmark"]["required_query_ids"]
    for family in settings["families"]:
        for seed in declared_seeds:
            map_id=f"{family}_{seed}";path=output/(map_id+".t25d")
            fn,rectangles,definitions,params=pilot_geometry(family,seed,width,height)
            definitions=[definition for definition in definitions if definition[0] in required_queries]
            if [definition[0] for definition in definitions]!=required_queries:
                raise ValueError(f"combined query set mismatch: {map_id}")
            heights=[[fn(i*spacing,j*spacing) for i in range(nx)] for j in range(ny)]
            meta=write_map(path,heights,spacing=(spacing,spacing),radius=model["robot_radius_m"],
                margin=model["safety_margin_m"],slope=model["max_slope_deg"],rectangles=rectangles,
                metadata={"map_id":map_id,"family":family,
                    "generator_version":generator_version,"generator_seed":seed,
                    "parameters":params,"dataset_role":"held_out_evaluation",
                    "seed_selection":"predeclared_no_replacement_no_planner_results_used"})
            oracle=Oracle(path);queries=[]
            for query_id,start,goal,paths,label in definitions:
                witnesses=[]
                for index,xy in enumerate(paths):
                    checked=oracle.validate_path(xy,start,goal)
                    if not checked["valid"]:
                        raise ValueError(f"combined held-out qualification failed without replacement: "
                                         f"{map_id}/{query_id}/{index}: {checked}")
                    witnesses.append({"witness_id":f"w{index}","path_xy":xy,"sha256":digest(xy),
                                      "validation":checked})
                direct=oracle.validate_path([start,goal],start,goal)
                best=min(witnesses,key=lambda witness:witness["validation"]["cost_terrain_m"])
                if label.startswith("blocked_direct") and direct["valid"]:
                    raise ValueError(f"combined blocked-direct qualification failed without replacement: "
                                     f"{map_id}/{query_id}")
                if label=="valid_direct_terrain_tradeoff" and not (direct["valid"] and
                        best["validation"]["cost_terrain_m"]+1e-6<direct["cost_terrain_m"]):
                    raise ValueError(f"combined terrain-tradeoff qualification failed without replacement: "
                                     f"{map_id}/{query_id}")
                reference={"reference_type":"validated_witness",
                    "cost_terrain_m":best["validation"]["cost_terrain_m"],"path_xy":best["path_xy"],
                    "path_sha256":best["sha256"],"validation":best["validation"],
                    "witness_id":best["witness_id"],"map_sha256":meta["data_sha256"],
                    "provenance":"held-out generator geometric witness; independent Oracle; fixed before "
                                 "planner runs; not certified optimum"}
                query={"query_id":query_id,"start":start,"goal":goal,"geometry_label":label,
                    "feasibility":"feasible_witness_validated","witnesses":witnesses,
                    "direct_edge_validation":direct,"reference":reference}
                query["query_hash"]=digest({"start":start,"goal":goal,"map_hash":meta["data_sha256"],
                                             "model":model})
                queries.append(query)
            meta["queries"]=queries;dump(path.with_suffix(".json"),meta)
            maps.append({"map_id":map_id,"family":family,"map_seed":seed,"path":path.name if cloud else str(path.resolve()),
                         "sha256":meta["data_sha256"],"queries":queries})
            print(f"Qualified combined held-out map without seed replacement: {map_id}",flush=True)
    expected={(family,seed) for family in settings["families"] for seed in declared_seeds}
    realized={(item["family"],item["map_seed"]) for item in maps}
    if realized!=expected or len(maps)!=len(expected):
        raise ValueError("combined generator did not realize every declared seed exactly once")
    manifest={"schema_version":5,"profile":"combined",
        "generator_version":generator_version,"config_hash":digest(config),"maps":maps,
        "held_out":True,"declared_map_seeds":declared_seeds,"seed_replacement_policy":"forbidden",
        "seed_replacements":[],
        "qualification_policy":"all declared family/seed maps and q0 witnesses must pass before planner "
                               "execution; abort, never replace"}
    dump(output/"manifest.json",manifest)
    return manifest

def generate(config,profile="smoke",output=None):
    output=Path(output or ROOT/"data"/profile)
    if output.exists() and any(output.iterdir()):raise ValueError("generation output is not empty; use a new directory to preserve provenance")
    output.mkdir(parents=True,exist_ok=True)
    if config.get("schema_version")==9:
        from a3_traversability_perf_protocol import generate_maps
        return generate_maps(config,profile,output)
    if config.get("schema_version")==8:
        from a3_traversability_protocol import generate_maps
        return generate_maps(config,profile,output)
    if config.get("schema_version")==7:
        from a3_pilot_protocol import generate_maps
        return generate_maps(config,profile,output)
    if profile=="pilot":return generate_pilot(config,output)
    if profile=="advanced":return generate_advanced(config,output)
    if profile=="sem_comparison":return generate_sem_comparison(config,output)
    if profile=="combined":return generate_combined(config,output)
    p=config["profiles"][profile];model=config["model"]
    width,height=p["map_extent_m"];spacing=p["grid_spacing_m"]
    nx=round(width/spacing)+1;ny=round(height/spacing)+1
    if abs((nx-1)*spacing-width)>1e-9 or abs((ny-1)*spacing-height)>1e-9:raise ValueError("extent must be divisible by spacing")
    maps=[]
    for family in p["families"]:
        for seed in p["map_seeds"]:
            rng=random.Random(seed);map_id=f"{family}_{seed}";path=output/(map_id+".t25d")
            params={};rectangles=[];start=[width*.1,height*.2];goal=[width*.9,height*.8]
            witness=[start,goal]
            if family=="flat":
                params={"h0":2.};fn=lambda x,y:2.
            elif family=="tilted_plane":
                params={"a":.24,"b":.10,"c":2.};fn=lambda x,y:.24*x+.10*y+2
            elif family=="traversable_hill":
                params={"h0":0.,"A":2.5,"cx":width*.5+rng.uniform(-.4,.4),"cy":height*.5+rng.uniform(-.4,.4),"sx":4.,"sy":4.}
                fn=lambda x,y:params["A"]*math.exp(-.5*(((x-params["cx"])/params["sx"])**2+((y-params["cy"])/params["sy"])**2))
            else:
                params={"h0":0.,"corridor_width_m":1.8,"wall_thickness_m":.4,"gap_center_y":height/2}
                fn=lambda x,y:0.
                rectangles=[(width/2-.2,0,width/2+.2,height/2-.9),(width/2-.2,height/2+.9,width/2+.2,height)]
                witness=[start,[width/2-1,height/2],[width/2+1,height/2],goal]
                params["center_clearance_width_m"]=1.8-2*(model["robot_radius_m"]+model["safety_margin_m"])
            heights=[[fn(i*spacing,j*spacing) for i in range(nx)] for j in range(ny)]
            meta=write_map(path,heights,spacing=(spacing,spacing),radius=model["robot_radius_m"],margin=model["safety_margin_m"],
                slope=model["max_slope_deg"],rectangles=rectangles,metadata={"map_id":map_id,"family":family,"generator_version":GENERATOR_VERSION,
                "generator_seed":seed,"parameters":params,"query_seed":seed+100000,"queries":[{"query_id":"q0","start":start,"goal":goal,"witness_path":witness}]})
            oracle=Oracle(path);validation=oracle.validate_path(witness,start,goal)
            if not validation["valid"]:raise ValueError(f"invalid generated witness: {map_id}: {validation}")
            meta["witness_validation"]=validation
            if family in ("flat","tilted_plane"):
                meta["reference"]={"reference_type":"analytic","cost_terrain_m":validation["cost_terrain_m"],
                    "provenance":"straight lifted segment on unobstructed affine plane; chord lower bound attained"}
            else:meta["reference"]=None
            dump(path.with_suffix(".json"),meta)
            maps.append({"map_id":map_id,"family":family,"map_seed":seed,"path":str(path.resolve()),"sha256":meta["data_sha256"],"queries":meta["queries"]})
    manifest={"schema_version":1,"profile":profile,"generator_version":GENERATOR_VERSION,"config_hash":digest(config),"maps":maps}
    dump(output/"manifest.json",manifest)
    print(f"Generated {len(maps)} maps, with independently validated witness paths: {output}")
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--profile",default="smoke");p.add_argument("--output")
    args=p.parse_args();generate(load_config(args.config,args.profile),args.profile,args.output)

if __name__=="__main__":main()
