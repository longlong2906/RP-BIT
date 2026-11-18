"""Frozen schema 10: static terrain planners; never changes legacy campaigns."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random

from experiment_common import digest, dump
from a3_traversability_protocol import DEFAULT_TRAVERSABILITY, fixture_geometry
from g5_dataset import geometry_definition, qualify_queries
from independent_verifier import Oracle
from terrain_io import write_map

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3_terrain_compare_kaggle_v1"
FAMILIES = ["traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage",
            "mixed_obstacles", "two_corridors_traversability"]
SMOKE_FAMILIES = ["flat", "tilted", "unsafe_short_safe_detour", "two_corridors_slope",
                  "two_corridors_clearance", "nodata_thin_obstacle"]
MODEL = {"state": "xy", "robot_radius_m": .3, "safety_margin_m": .1,
         "max_slope_deg": 35., "roughness_limit_m": None,
         "local_connector": "straight_xy_lifted_to_piecewise_affine_heightfield",
         "goal": "singleton", "goal_roundoff_tolerance_m": 1e-9}
BACKENDS = ["a3", "sem", "putn", "sgp"]
TRACKS = ["common", "own"]


def default_config():
    return copy.deepcopy({"schema_version": 10, "protocol_id": PROTOCOL_ID, "model": MODEL,
        "traversability": DEFAULT_TRAVERSABILITY,
        "profiles": {
            "smoke": {"families": SMOKE_FAMILIES, "map_seeds": [101], "planner_seeds": [11],
                      "extent_m": 20., "spacing_m": .2, "planning_budget_s": 2.},
            "main": {"families": FAMILIES, "map_seeds": list(range(7001, 7006)),
                     "planner_seeds": [95001, 95002, 95003], "extent_m": 40.,
                     "spacing_m": .2, "planning_budget_s": 10.}},
        "execution": {"max_workers": 4, "threads_per_trial": 1, "shuffle_seed": 73135,
                      "memory_limit_bytes": 8*1024**3, "admission_fraction": .8,
                      "pressure_fraction": .9, "pressure_duration_s": 5.,
                      "setup_timeout_s": 180., "solve_grace_s": 5., "serialize_timeout_s": 60.,
                      "session_soft_stop_s": 28800., "session_finalize_s": 32400.},
        "representation": {"ground_spacing_m": .2, "synthetic_obstacle_height_m": 2.,
                           "putn_voxel_m": .1, "sgp_grid_m": .2, "noise": "none"},
        "analysis": {"checkpoints_s": [.1, .25, .5, 1., 2., 5., 10.],
                     "bootstrap_replicates": 2000, "bootstrap_seed": 73136,
                     "confidence": .95, "risk_is_probability": False}})


def load_config(path):
    import yaml
    text = Path(path).read_text(encoding="utf-8")
    config = json.loads(text) if text.lstrip().startswith("{") else yaml.safe_load(text)
    if digest(config) != digest(default_config()):
        raise ValueError("schema 10 differs from frozen protocol; create a new protocol version")
    return config


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def source_identity(root=ROOT):
    files = []
    for directory in ("src", "include", "apps", "tools", "tests", "cmake", "configs", "patches", "external"):
        for p in sorted((root/directory).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                files.append([p.relative_to(root).as_posix(), sha256(p)])
    for p in sorted(root.glob("requirements-terrain-compare*")):
        files.append([p.name, sha256(p)])
    files.append(["CMakeLists.txt", sha256(root/"CMakeLists.txt")])
    return {"source_sha256": digest(files), "source_files": files}


def corridor_geometry(seed):
    rng = random.Random(seed + 500000)
    p = {k: rng.uniform(*v) for k, v in {
        "amplitude_m": (2.4, 3.2), "slope": (.35, .60), "center_y": (11., 13.),
        "center_x": (19., 21.), "half_width_m": (3.5, 4.5)}.items()}
    a, s, y, x, w = (p[k] for k in ("amplitude_m", "slope", "center_y", "center_x", "half_width_m"))
    fn = lambda xx, yy: max(0., a-s*abs(yy-y))
    start, goal = [4., y+6], [36., y+6]
    paths = [[start, [x-w-3, yy], [x+w+3, yy], goal] for yy in (y, y+16)]
    query = {"query_id": "q0", "query_seed": seed+1900000, "start": start, "goal": goal,
             "paths": paths, "geometry_label": "blocked_direct_short_sloped_long_flat_corridors"}
    return fn, [(x-w, y+2, x+w, y+12)], [query], p


def generate_dataset(config, profile, output):
    if digest(config) != digest(default_config()) or profile not in config["profiles"]:
        raise ValueError("unregistered configuration/profile")
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError("refusing to overwrite dataset")
    output.mkdir(parents=True, exist_ok=True)
    settings, maps = config["profiles"][profile], []
    try:
        for family in settings["families"]:
            for seed in settings["map_seeds"]:
                if profile == "smoke":
                    fn, rects, start, goal, paths, label = fixture_geometry(family)
                    definitions = [dict(query_id="q0", query_seed=101, start=start, goal=goal,
                                        paths=paths, geometry_label=label)]
                    params = {"fixed_fixture": True}
                else:
                    fn, rects, definitions, params = (corridor_geometry(seed) if family == FAMILIES[-1]
                                                      else geometry_definition(family, seed))
                    definitions = definitions[:1]
                spacing = settings["spacing_m"]
                n = round(settings["extent_m"]/spacing)+1
                heights = [[fn(i*spacing, j*spacing) for i in range(n)] for j in range(n)]
                map_id = f"{family}_{seed}"
                path = output/(map_id+".t25d")
                meta = write_map(path, heights, spacing=(spacing, spacing), rectangles=rects,
                    metadata={"map_id": map_id, "family": family, "generator_seed": seed,
                              "generator_version": "terrain_compare_1.0", "parameters": params})
                oracle = Oracle(path)
                queries = qualify_queries(oracle, definitions, sha256(path), MODEL)
                if family == FAMILIES[-1]:
                    measured = [oracle.validate_path(w["path_xy"], queries[0]["start"], queries[0]["goal"],
                                traversability=DEFAULT_TRAVERSABILITY) for w in queries[0]["witnesses"]]
                    if not (measured[0]["cost_terrain_m"] < measured[1]["cost_terrain_m"] and
                            measured[0]["risk_slope_integral_m"] > measured[1]["risk_slope_integral_m"]):
                        raise ValueError("predeclared corridor length-risk witness ordering failed")
                    for w, v in zip(queries[0]["witnesses"], measured): w["traversability_validation"] = v
                meta["queries"] = queries
                dump(path.with_suffix(".json"), meta)
                maps.append(dict(map_id=map_id, family=family, map_seed=seed, path=path.name,
                                 sha256=sha256(path), metadata_sha256=sha256(path.with_suffix(".json")), queries=queries))
        manifest = dict(protocol_id=PROTOCOL_ID, profile=profile, config_hash=digest(config), maps=maps,
                        seed_replacements=[], qualification="independent_geometric_witness_not_optimum")
        dump(output/"manifest.json", manifest)
        return manifest
    except BaseException as exc:
        dump(output/"qualification_failure.json", {"error": str(exc), "seed_replacements": [], "completed_maps": len(maps)})
        raise


def request_for(backend, track, q, map_path, seed, budget):
    request = dict(map_path=str(map_path.resolve()), planner_seed=seed, start=q["start"], goal=q["goal"],
        planning_budget_s=budget, track=track, backend=backend, cloud_phase_handshake=True,
        objective_type="surface_length_traversability", traversability=copy.deepcopy(DEFAULT_TRAVERSABILITY),
        samples_per_batch=200, rewire_factor=1.1, sample_attempts=100, progress_interval_s=.05,
        collision_index=True, state_cache=True, scalar_cost=True,
        traversability_build_policy="skip_dominated", sampler_query_cache=backend=="a3",
        exact_flat_certificate=backend=="a3", batch_after_first_solution=0)
    request["planner_id"] = {"a3": "bitstar_a3", "sem": "sem_rrtstar_common_t_v1",
                             "putn": "putn_static_v1", "sgp": "sgp_static_v1"}[backend]
    request["variant_id"] = f"{backend}_static_{track}_v1"
    if backend == "sem": request.update(sem_range_m=1., sem_neighbor_radius_m=3.)
    return request


def plan_trials(config, profile, dataset, dataset_root, provenance=None):
    settings = config["profiles"][profile]
    if dataset["config_hash"] != digest(config) or dataset["profile"] != profile:
        raise ValueError("dataset/config mismatch")
    expected = {(f, s) for f in settings["families"] for s in settings["map_seeds"]}
    if len(dataset["maps"]) != len(expected) or {(m["family"], m["map_seed"]) for m in dataset["maps"]} != expected:
        raise ValueError("dataset cardinality/identity mismatch")
    blocks = []
    for mi, m in enumerate(dataset["maps"]):
        path = Path(dataset_root)/m["path"]
        if sha256(path) != m["sha256"] or sha256(path.with_suffix(".json")) != m["metadata_sha256"]:
            raise ValueError("dataset changed")
        for si, seed in enumerate(settings["planner_seeds"]):
            for ti, track in enumerate(TRACKS):
                order = (mi+si+ti)%4
                blocks.append((m, seed, track, BACKENDS[order:]+BACKENDS[:order]))
    random.Random(config["execution"]["shuffle_seed"]).shuffle(blocks)
    entries = []
    for bi, (m, seed, track, backends) in enumerate(blocks):
        q = m["queries"][0]
        for backend in backends:
            blocked = backend == "sem" and track == "own"
            request = request_for(backend, track, q, Path(dataset_root)/m["path"], seed, settings["planning_budget_s"])
            entries.append(dict(trial_id=f"{track}__{m['map_id']}__s{seed}__{backend}",
                block_order=bi, planned_order=len(entries), backend=backend, track=track,
                family=m["family"], map_id=m["map_id"], map_seed=m["map_seed"], map_sha256=m["sha256"],
                query_id="q0", planner_seed=seed, request=request,
                execution_status="BLOCKED_SOURCE" if blocked else "PLANNED",
                blocked_reason="SEM own-cost source/specification unavailable" if blocked else None))
    return dict(schema_version=10, protocol_id=PROTOCOL_ID, profile=profile, entries=entries,
        provenance=dict(provenance or {}, config_hash=digest(config), dataset_hash=digest(dataset)),
        planned_slots=len(entries), planned_executable=sum(e["execution_status"]=="PLANNED" for e in entries))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default=str(ROOT/"configs/terrain_compare_kaggle_v1.yaml"))
    p.add_argument("--profile", choices=["smoke", "main"], required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args(); config = load_config(args.config)
    dataset = json.loads((args.output/"manifest.json").read_text()) if args.plan_only else generate_dataset(config, args.profile, args.output)
    planned = plan_trials(config, args.profile, dataset, args.output)
    dump(args.output/"trial_plan.json", planned)
    print(json.dumps({"slots": planned["planned_slots"], "executable": planned["planned_executable"]}))


if __name__ == "__main__": main()
