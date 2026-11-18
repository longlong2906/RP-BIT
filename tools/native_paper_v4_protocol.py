"""Frozen schema 13: old A3-T, constrained A3-T and native SEM, 1,500 trials.

The two A3 variants retain their declared distinct objectives. The SEM baseline
is the unchanged static paper reconstruction; its native cost is never treated
as a cross-planner quality metric. This module prepares inputs, never runs a
planner or tunes the constrained policy using evaluation results.
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
from terrain_compare_protocol import MODEL, sha256, source_identity as _source_identity

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3t_old_safe_sem_kaggle_v4"
SCHEMA_VERSION = 13
BACKENDS = ("a3t_old", "a3t_safe", "sem_native")
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
SMOKE_FAMILIES = ("flat", "tilted", "unsafe_short_safe_detour", "two_corridors_slope",
                  "two_corridors_clearance", "nodata_thin_obstacle", "flat_40m")
OBJECTIVES = dict(a3t_old="a3t_surface_slope_clearance_v1",
                  a3t_safe="surface_length_subject_to_sem40_swept35_v1",
                  sem_native="sem_length_turn_risk_paper_v2")
LABELS = dict(a3t_old="A3-T old", a3t_safe="A3-T safe length",
              sem_native="SEM-RRT*-paper-reconstruction")


def source_identity(root=ROOT):
    value = _source_identity(root)
    files = list(value["source_files"])
    for path in sorted(Path(root).glob("requirements-native-paper-v[34]*")):
        files.append([path.name, sha256(path)])
    return dict(source_sha256=digest(files), source_files=files)


def default_config():
    from native_paper_v3_protocol import default_config as previous_config
    previous = previous_config()
    config = copy.deepcopy(previous)
    config.update(schema_version=SCHEMA_VERSION, protocol_id=PROTOCOL_ID,
                  native_objectives=copy.deepcopy(OBJECTIVES))
    config["native_parameters"] = dict(a3t_old=previous["native_parameters"]["a3"],
        a3t_safe=dict(planner="OMPL_BITstar", objective="surface_length", risk_cap=40., slope_cap_deg=35.,
                      atlas_sampling=True, initial_batch=100, batch_after_first_solution=200,
                      rewire_factor=1.1, sample_attempts=100, threshold_policy="immutable_no_relaxation"),
        sem_native=previous["native_parameters"]["sem"])
    config["profiles"]["main"].update(map_seeds=list(range(28001, 28011)),
        planner_seeds=list(range(296001, 296011)), planning_budget_s=20.)
    config["representation"].pop("putn_voxel_m", None)
    config["execution"].update(shuffle_seed=83137, required_cpu_capacity=4,
        runtime_policy="native_paper_sessions_v4", retry_started_trials=False,
        compaction="postsolve_coordinate_dedup_and_byte_identical_native_raster_sharing")
    config["analysis"].update(bootstrap_seed=83138, reference_backend="sem_native",
        contrasts=[["a3t_safe", "sem_native"], ["a3t_old", "sem_native"], ["a3t_safe", "a3t_old"]],
        first_solution_metrics=["first_geometry_s", "first_paper_s", "first_geometry_latency_s", "first_paper_latency_s"],
        restricted_first_safe=True)
    return config


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        value = json.loads(text)
    else:
        import yaml
        value = yaml.safe_load(text)
    if digest(value) != digest(default_config()):
        raise ValueError("schema 13 differs from the frozen comparison; register a new protocol")
    return value


def field_builder_key(metadata):
    keys = ("version", "format", "bounds", "resolution_m", "shape_yx", "cloud_sha256", "map_sha256",
            "builder_source_sha256", "field_sha256")
    return digest({key:metadata[key] for key in keys})


def generate_dataset(config, profile, output):
    import numpy as np
    from native_point_cloud import generate_cloud
    from safe_length_field import build_field_file

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
                        fn, rectangles, start, goal, paths, label = ((lambda x, y: 0.), [], [2., 20.], [38., 20.],
                            [[[2., 20.], [38., 20.]]], "valid_direct_full_scale_flat_control")
                    else:
                        fn, rectangles, start, goal, paths, label = fixture_geometry(family)
                    definitions = [dict(query_id="q0", query_seed=seed, start=start, goal=goal,
                                        paths=paths, geometry_label=label)]
                    parameters = dict(fixed_fixture=True)
                else:
                    fn, rectangles, definitions, parameters = geometry_definition(family, seed)
                    definitions = definitions[:1]
                spacing = settings["spacing_m"]
                n = round(extent/spacing)+1
                map_id = f"{family}_{seed}"
                path = output/(map_id+".t25d")
                metadata = write_map(path, [[fn(i*spacing, j*spacing) for i in range(n)] for j in range(n)],
                    spacing=(spacing, spacing), rectangles=rectangles,
                    metadata=dict(map_id=map_id, family=family, generator_seed=seed,
                                  generator_version="native_paper_4.0", parameters=parameters))
                queries = qualify_queries(Oracle(path), definitions, sha256(path), MODEL, clearance_metrics=False)
                metadata["queries"] = queries
                dump(path.with_suffix(".json"), metadata)
                cloud_path = path.with_suffix(".cloud.npy")
                points = generate_cloud(path)
                np.save(cloud_path, np.asarray(points, dtype=np.float64), allow_pickle=False)
                field_path = path.with_suffix(".safe_sem.txt")
                bounds = [0., 0., extent, extent]
                field_metadata = build_field_file(cloud_path, path, bounds, field_path)
                maps.append(dict(map_id=map_id, family=family, map_seed=seed, path=path.name,
                    sha256=sha256(path), metadata_sha256=sha256(path.with_suffix(".json")), queries=queries,
                    cloud_path=cloud_path.name, cloud_sha256=sha256(cloud_path), cloud_point_count=len(points),
                    bounds=bounds, safe_field_path=field_path.name, safe_field_sha256=sha256(field_path),
                    safe_field_metadata_sha256=sha256(field_path.with_suffix(".txt.meta.json")),
                    safety_field_builder_key=field_builder_key(field_metadata)))
        manifest = dict(protocol_id=PROTOCOL_ID, profile=profile, config_hash=digest(config), maps=maps,
            seed_replacements=[], qualification="independent_geometric_witness_not_optimum",
            field_storage="one_immutable_planner_field_per_map; rebuilt_and_hash_checked_in_every_safe_trial")
        dump(output/"manifest.json", manifest)
        return manifest
    except BaseException as error:
        dump(output/"qualification_failure.json", dict(error=str(error), seed_replacements=[], completed_maps=len(maps)))
        raise


def request_for(backend, q, map_path, cloud_path, cloud_hash, seed, budget, config):
    if backend not in BACKENDS:
        raise ValueError("unknown schema-13 backend")
    map_path, cloud_path = Path(map_path).resolve(), Path(cloud_path).resolve()
    request = dict(map_path=str(map_path), cloud_path=str(cloud_path), cloud_sha256=cloud_hash,
        map_sha256=sha256(map_path), start=q["start"], goal=q["goal"], planner_seed=seed,
        planning_budget_s=budget, track="native", backend=backend, cloud_phase_handshake=True,
        native_objective_id=OBJECTIVES[backend], representation_config=copy.deepcopy(config["representation"]),
        protocol_id=PROTOCOL_ID, campaign_schema_version=SCHEMA_VERSION, work_iterations=0,
        variant_id=backend+"_kaggle_v4", clearance_metrics=False)
    if backend in ("a3t_old", "a3t_safe"):
        request.update(planner_id="bitstar_a3", samples_per_batch=200, rewire_factor=1.1,
            sample_attempts=100, progress_interval_s=.05, collision_index=True, state_cache=True,
            scalar_cost=True, sampler_query_cache=True, exact_flat_certificate=False, batch_after_first_solution=0)
    if backend == "a3t_old":
        request.update(objective_type="surface_length_traversability",
                       traversability=copy.deepcopy(DEFAULT_TRAVERSABILITY), traversability_build_policy="skip_dominated")
    elif backend == "a3t_safe":
        field_path = map_path.with_suffix(".safe_sem.txt")
        metadata_path = field_path.with_suffix(".txt.meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["map_sha256"] != request["map_sha256"] or metadata["cloud_sha256"] != cloud_hash:
            raise ValueError("planner field input identity mismatch")
        if sha256(field_path) != metadata["field_sha256"]:
            raise ValueError("planner field bytes changed")
        request.update(objective_type="surface_length", samples_per_batch=100, batch_after_first_solution=200,
            shared_field_path=str(field_path), shared_field_metadata_sha256=sha256(metadata_path),
            safety_field_builder_key=field_builder_key(metadata),
            safe_length=dict(field_path=str(field_path), field_sha256=metadata["field_sha256"],
                cloud_sha256=cloud_hash, builder_source_sha256=metadata["builder_source_sha256"],
                policy_id="a3t_safe_length_v1", risk_cap=40., slope_cap_deg=35., atlas_sampling=True))
    else:
        request.update(planner_id="sem_native_paper_v2", objective_type="native")
    return request


def plan_trials(config, profile, dataset, dataset_root, provenance=None):
    if digest(config) != digest(default_config()) or profile not in config["profiles"]:
        raise ValueError("unregistered configuration/profile")
    settings, dataset_root = config["profiles"][profile], Path(dataset_root)
    if dataset.get("config_hash") != digest(config) or dataset.get("profile") != profile:
        raise ValueError("dataset/config mismatch")
    expected = {(family, seed) for family in settings["families"] for seed in settings["map_seeds"]}
    if len(dataset["maps"]) != len(expected) or {(m["family"], m["map_seed"]) for m in dataset["maps"]} != expected:
        raise ValueError("dataset cardinality/identity mismatch")
    blocks = []
    for mi, info in enumerate(dataset["maps"]):
        path, field = dataset_root/info["path"], dataset_root/info["safe_field_path"]
        for filename, expected_hash in ((path, info["sha256"]), (path.with_suffix(".json"), info["metadata_sha256"]),
            (dataset_root/info["cloud_path"], info["cloud_sha256"]), (field, info["safe_field_sha256"]),
            (field.with_suffix(".txt.meta.json"), info["safe_field_metadata_sha256"])):
            if sha256(filename) != expected_hash:
                raise ValueError("dataset changed: "+str(filename))
        if len(info["queries"]) != 1 or info["queries"][0]["query_id"] != "q0":
            raise ValueError("exactly q0 required")
        for si, seed in enumerate(settings["planner_seeds"]):
            offset = (mi+si) % len(BACKENDS)
            blocks.append((info, seed, BACKENDS[offset:]+BACKENDS[:offset]))
    random.Random(config["execution"]["shuffle_seed"]).shuffle(blocks)
    entries = []
    for bi, (info, seed, backends) in enumerate(blocks):
        for backend in backends:
            request = request_for(backend, info["queries"][0], dataset_root/info["path"],
                dataset_root/info["cloud_path"], info["cloud_sha256"], seed, settings["planning_budget_s"], config)
            if backend == "a3t_safe" and request["safety_field_builder_key"] != info["safety_field_builder_key"]:
                raise ValueError("planner field builder key changed")
            entries.append(dict(trial_id=f"v4__{info['map_id']}__s{seed}__{backend}", block_order=bi,
                planned_order=len(entries), backend=backend, track="native", family=info["family"], map_id=info["map_id"],
                map_seed=info["map_seed"], map_sha256=info["sha256"], query_id="q0", planner_seed=seed,
                request=request, execution_status="PLANNED", blocked_reason=None))
    return dict(schema_version=SCHEMA_VERSION, protocol_id=PROTOCOL_ID, profile=profile, entries=entries,
        provenance=dict(provenance or {}, config_hash=digest(config), dataset_hash=digest(dataset)),
        planned_slots=len(entries), planned_executable=len(entries), runtime_policy="native_paper_sessions_v4",
        phase_timeouts=dict(setup_s=config["execution"]["setup_timeout_s"], serialize_s=config["execution"]["serialize_timeout_s"],
                            solve_grace_s=config["execution"]["solve_grace_s"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/native_paper_kaggle_v4.yaml")
    parser.add_argument("--profile", choices=("smoke", "main"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = json.loads((args.output/"manifest.json").read_text()) if args.plan_only else generate_dataset(config, args.profile, args.output)
    plan = plan_trials(config, args.profile, dataset, args.output)
    dump(args.output/"trial_plan.json", plan)
    print(json.dumps(dict(slots=plan["planned_slots"], maps=len(dataset["maps"]), planner_executions=0)))


if __name__ == "__main__":
    main()
