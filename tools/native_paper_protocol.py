"""Frozen schema 11: native objectives, 50 maps, 2,000 paired trials.

The user's native-objective experiment explicitly supersedes the baseline
surface-only objective for this protocol; schema 10 is never modified.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import random

from experiment_common import digest, dump
from a3_traversability_protocol import DEFAULT_TRAVERSABILITY, fixture_geometry
from g5_dataset import geometry_definition, qualify_queries
from independent_verifier import Oracle
from terrain_io import write_map
from terrain_compare_protocol import MODEL, sha256, source_identity

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3t_native_paper_kaggle_v2"
SCHEMA_VERSION = 11
BACKENDS = ("a3", "sem", "putn", "sgp")
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
SMOKE_FAMILIES = ("flat", "tilted", "unsafe_short_safe_detour", "two_corridors_slope",
                  "two_corridors_clearance", "nodata_thin_obstacle", "flat_40m")
OBJECTIVES = {"a3": "a3t_surface_slope_clearance_v1", "sem": "sem_length_turn_risk_paper_v2",
              "putn": "putn_com_traversability_paper_v2", "sgp": "sgp_xy_paper_v2"}
LABELS = {"a3": "A3-T", "sem": "SEM-RRT*-paper-reconstruction",
          "putn": "PUTN/PF-RRT*-paper-static", "sgp": "SGP-RRT*-paper-static"}


def default_config():
    return copy.deepcopy(dict(schema_version=SCHEMA_VERSION, protocol_id=PROTOCOL_ID,
        model=MODEL, native_objectives=OBJECTIVES, source_precedence="paper_then_pinned_author_source",
        native_parameters={
            "a3":dict(planner="OMPL_BITstar",objective="L_H+R",risk_lambda=1.,slope_weight=.5,clearance_weight=.5,comfort_clearance_m=.4),
            "sem":dict(weights=[.48,.16,.36],turn_threshold_deg=30.,risk_threshold=40.,step_m=1.,neighbor_radius_m=3.,raster_m=.1,
                gaussian_radius_cells=5,gaussian_sigma_cells=5/3,gaussian_padding="nearest",length_units="grid_cells",turn_units="radian",risk_slope_units="degree",
                directional_sobel="plane_exact_normalization",informed_bound="resolution_times_Cbest_divided_by_0.48",rolling_optimization=False),
            "putn":dict(cost="d_COM*(1+0.1*(1/(1-tau_u)+1/(1-tau_v)-2))",tau_limit=1.,flatness_denominator="N",horizontal_normal_slope_deg=0.,
                voxel_m=.1,fit_radius_m=.5,com_height_m=.7,step_m=.4,informed_proposal="XY_with_COM_lower_bound_filter"),
            "sgp":dict(kernel="RBF_SE",gradient="magnitude_minmax_0_1_constant_zero",flatness="plane_fit",step_height="relative_to_center_cell_5x5",
                weights=[.2,.4,.4],critical_values=[.5236,.5236,.3],traversability_limit=.4,objective="length_XY",uncertainty_endpoint_exemption=False,
                cpu_flush_denormal=True,subnormal_policy_scope="SGP_worker_CPU_FP32_and_FP64_FTZ_DAZ_not_a_model_hyperparameter")},
        profiles={"smoke": dict(families=list(SMOKE_FAMILIES), map_seeds=[101], planner_seeds=[11],
                  extent_m=20., spacing_m=.2, planning_budget_s=2.),
                  "main": dict(families=list(FAMILIES), map_seeds=list(range(8001,8011)),
                  planner_seeds=list(range(96001,96011)), extent_m=40., spacing_m=.2, planning_budget_s=15.)},
        representation=dict(master_cloud_spacing_m=.05, obstacle_height_m=2., sem_resolution_m=.1,
                            putn_voxel_m=.1, sgp_voxel_m=.2, observation_noise="none"),
        execution=dict(max_workers=4, threads_per_trial=1, shuffle_seed=73137,
                       setup_timeout_s=600., solve_grace_s=5., serialize_timeout_s=60.,
                       session_soft_stop_s=28800., session_finalize_s=32400., main_requires_explicit_flag=True),
        analysis=dict(checkpoints_s=[.1,.25,.5,1.,2.,5.,10.,15.], bootstrap_replicates=2000,
                      bootstrap_seed=73138, confidence=.95, sharp_turn_threshold_deg=30.,
                      shared_sem_evaluator=False, plot=False)))


def load_config(path):
    import yaml
    text = Path(path).read_text(encoding="utf-8")
    value = json.loads(text) if text.lstrip().startswith("{") else yaml.safe_load(text)
    if digest(value) != digest(default_config()):
        raise ValueError("schema 11 differs from frozen native-paper protocol; create a new version")
    return value


def generate_dataset(config, profile, output):
    import numpy as np
    from native_point_cloud import generate_cloud
    if digest(config) != digest(default_config()) or profile not in config["profiles"]:
        raise ValueError("unregistered configuration/profile")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite dataset")
    output.mkdir(parents=True, exist_ok=True)
    settings, maps = config["profiles"][profile], []
    try:
        for family in settings["families"]:
            for seed in settings["map_seeds"]:
                extent = 40. if family == "flat_40m" else settings["extent_m"]
                if profile == "smoke":
                    if family == "flat_40m":
                        fn, rects, start, goal, paths, label = (lambda x,y: 0.), [], [2.,20.], [38.,20.], [[[2.,20.],[38.,20.]]], "valid_direct_full_scale_flat_control"
                    else:
                        fn, rects, start, goal, paths, label = fixture_geometry(family)
                    definitions = [dict(query_id="q0",query_seed=seed,start=start,goal=goal,paths=paths,geometry_label=label)]
                    params = {"fixed_fixture": True}
                else:
                    fn, rects, definitions, params = geometry_definition(family,seed)
                    definitions = definitions[:1]
                spacing = settings["spacing_m"]
                n = round(extent/spacing)+1
                heights = [[fn(i*spacing,j*spacing) for i in range(n)] for j in range(n)]
                map_id = f"{family}_{seed}"
                path = output/(map_id+".t25d")
                meta = write_map(path,heights,spacing=(spacing,spacing),rectangles=rects,
                    metadata=dict(map_id=map_id,family=family,generator_seed=seed,
                                  generator_version="native_paper_2.0",parameters=params))
                queries = qualify_queries(Oracle(path),definitions,sha256(path),MODEL)
                meta["queries"] = queries
                dump(path.with_suffix(".json"),meta)
                cloud_path=path.with_suffix(".cloud.npy")
                points=generate_cloud(path)
                # The master cloud is shared data, never planner output or a fake trial.
                np.save(cloud_path,np.asarray(points,dtype=np.float64),allow_pickle=False)
                maps.append(dict(map_id=map_id,family=family,map_seed=seed,path=path.name,
                    sha256=sha256(path),metadata_sha256=sha256(path.with_suffix(".json")),queries=queries,
                    cloud_path=cloud_path.name,cloud_sha256=sha256(cloud_path),cloud_point_count=len(points)))
        manifest=dict(protocol_id=PROTOCOL_ID,profile=profile,config_hash=digest(config),maps=maps,
                      seed_replacements=[],qualification="independent_geometric_witness_not_optimum")
        dump(output/"manifest.json",manifest)
        return manifest
    except BaseException as error:
        dump(output/"qualification_failure.json",dict(error=str(error),seed_replacements=[],completed_maps=len(maps)))
        raise


def request_for(backend,q,map_path,cloud_path,cloud_hash,seed,budget,config):
    request=dict(map_path=str(Path(map_path).resolve()),cloud_path=str(Path(cloud_path).resolve()),
        cloud_sha256=cloud_hash,start=q["start"],goal=q["goal"],planner_seed=seed,
        planning_budget_s=budget,track="native",backend=backend,cloud_phase_handshake=True,
        native_objective_id=OBJECTIVES[backend],representation_config=copy.deepcopy(config["representation"]),
        protocol_id=PROTOCOL_ID,variant_id=f"{backend}_native_paper_v2",work_iterations=0)
    if backend=="a3":
        request.update(planner_id="bitstar_a3",objective_type="surface_length_traversability",
            traversability=copy.deepcopy(DEFAULT_TRAVERSABILITY),samples_per_batch=200,rewire_factor=1.1,
            sample_attempts=100,progress_interval_s=.05,collision_index=True,state_cache=True,scalar_cost=True,
            traversability_build_policy="skip_dominated",sampler_query_cache=True,
            exact_flat_certificate=False,batch_after_first_solution=0)
    else:
        request.update(planner_id=f"{backend}_native_paper_v2",objective_type="native")
    return request


def plan_trials(config,profile,dataset,dataset_root,provenance=None):
    settings=config["profiles"][profile]; dataset_root=Path(dataset_root)
    if dataset["config_hash"]!=digest(config) or dataset["profile"]!=profile:
        raise ValueError("dataset/config mismatch")
    expected={(f,s) for f in settings["families"] for s in settings["map_seeds"]}
    if len(dataset["maps"])!=len(expected) or {(m["family"],m["map_seed"]) for m in dataset["maps"]}!=expected:
        raise ValueError("dataset cardinality/identity mismatch")
    blocks=[]
    for mi,m in enumerate(dataset["maps"]):
        path=dataset_root/m["path"]
        for filename,expected_hash in ((path,m["sha256"]),(path.with_suffix(".json"),m["metadata_sha256"]),
                                       (dataset_root/m["cloud_path"],m["cloud_sha256"])):
            if sha256(filename)!=expected_hash: raise ValueError("dataset changed: "+str(filename))
        if len(m["queries"])!=1 or m["queries"][0]["query_id"]!="q0":raise ValueError("exactly q0 required")
        for si,seed in enumerate(settings["planner_seeds"]):
            offset=(mi+si)%4
            blocks.append((m,seed,BACKENDS[offset:]+BACKENDS[:offset]))
    random.Random(config["execution"]["shuffle_seed"]).shuffle(blocks)
    entries=[]
    for bi,(m,seed,backends) in enumerate(blocks):
        for backend in backends:
            request=request_for(backend,m["queries"][0],dataset_root/m["path"],dataset_root/m["cloud_path"],
                                m["cloud_sha256"],seed,settings["planning_budget_s"],config)
            entries.append(dict(trial_id=f"native__{m['map_id']}__s{seed}__{backend}",block_order=bi,
                planned_order=len(entries),backend=backend,track="native",family=m["family"],map_id=m["map_id"],
                map_seed=m["map_seed"],map_sha256=m["sha256"],query_id="q0",planner_seed=seed,request=request,
                execution_status="PLANNED",blocked_reason=None))
    return dict(schema_version=SCHEMA_VERSION,protocol_id=PROTOCOL_ID,profile=profile,entries=entries,
        provenance=dict(provenance or {},config_hash=digest(config),dataset_hash=digest(dataset)),
        planned_slots=len(entries),planned_executable=len(entries),runtime_policy="native_paper_sessions_v2",
        phase_timeouts=dict(setup_s=config["execution"]["setup_timeout_s"],serialize_s=config["execution"]["serialize_timeout_s"],
                            solve_grace_s=config["execution"]["solve_grace_s"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/native_paper_kaggle_v2.yaml")
    p.add_argument("--profile",choices=("smoke","main"),required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--plan-only",action="store_true")
    args=p.parse_args(); config=load_config(args.config)
    dataset=json.loads((args.output/"manifest.json").read_text()) if args.plan_only else generate_dataset(config,args.profile,args.output)
    plan=plan_trials(config,args.profile,dataset,args.output);dump(args.output/"trial_plan.json",plan)
    print(json.dumps(dict(slots=plan["planned_slots"],executable=plan["planned_executable"])))


if __name__=="__main__":main()
