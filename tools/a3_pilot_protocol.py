"""Locked, portable A3 improvement pilot design; no planner execution here.

This schema is separate from the historical campaigns.  Its labels describe
implementations and two co-primary outcomes, not a claimed algorithm ranking.
"""
import copy
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import re

from experiment_common import digest, dump, read_json, resolved_planner_options

PROTOCOL_ID = "a3_improvements_pilot_v1"
GENERATOR_VERSION = "a3-improvements-pilot-1.0.0"
ORDER_POLICY = "canonical_block_cyclic_13_then_shuffle_v1"
PLANNERS = ("bitstar_a3", "a3_cache", "a3_scalar", "a3_bvh", "a3_optimized",
            "a3_batch50", "a3_batch100", "a3_batch400", "a3_two_phase50to200",
            "a3_optimized_two_phase50to200", "abitstar", "aitstar",
            "lazyprmstar_hauser2015")
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage",
            "mixed_obstacles")
SMOKE_FAMILIES = ("flat", "tilted_plane", "traversable_hill", "narrow_passage")
EXPECTED_TRIALS = {"smoke": 52, "a3_pilot": 780}
VARIANTS = {
    "a3_cache": {"planner_id": "bitstar_a3", "state_cache": True},
    "a3_scalar": {"planner_id": "bitstar_a3", "scalar_cost": True},
    "a3_bvh": {"planner_id": "bitstar_a3", "collision_index": True},
    "a3_optimized": {"planner_id": "bitstar_a3", "state_cache": True,
                     "scalar_cost": True, "collision_index": True},
    "a3_batch50": {"planner_id": "bitstar_a3", "samples_per_batch": 50},
    "a3_batch100": {"planner_id": "bitstar_a3", "samples_per_batch": 100},
    "a3_batch400": {"planner_id": "bitstar_a3", "samples_per_batch": 400},
    "a3_two_phase50to200": {"planner_id": "bitstar_a3", "samples_per_batch": 50,
                           "batch_after_first_solution": 200},
    "a3_optimized_two_phase50to200": {"planner_id": "bitstar_a3", "state_cache": True,
        "scalar_cost": True, "collision_index": True, "samples_per_batch": 50,
        "batch_after_first_solution": 200},
}


def comparison_pairs():
    """Ordered (candidate, reference) comparisons, fixed before observations."""
    return ([(p, "bitstar_a3") for p in PLANNERS[1:10]]
            + [("bitstar_a3", p) for p in PLANNERS[10:]]
            + [(a, p) for a in ("a3_optimized", "a3_optimized_two_phase50to200")
               for p in PLANNERS[10:]]
            + [("a3_optimized", "a3_optimized_two_phase50to200")])


def expected_config():
    """One immutable design under this protocol ID; no loaded-file defaults."""
    return {
        "schema_version": 7, "project": "bitstar_25d", "language": "vi",
        "model": {
            "state": "xy", "local_connector": "straight_xy_lifted_to_piecewise_affine_heightfield",
            "objective": "surface_length", "surface_interpolation": "fixed_diagonal_triangles",
            "cell_diagonal": "southwest_to_northeast", "height_units": "m", "coordinate_units": "m",
            "nodata_policy": "forbidden", "obstacle_contact_policy": "forbidden",
            "include_outside_map_as_forbidden": True, "robot_radius_m": .30,
            "safety_margin_m": .10, "max_slope_deg": 35., "roughness_limit_m": None,
            "roughness_window_m": 1., "goal": "singleton", "goal_roundoff_tolerance_m": 1e-9,
            "numerical_abs_tolerance_m": 1e-8, "numerical_rel_tolerance": 1e-8},
        "algorithm": {
            "nearest_neighbor_metric": "euclidean_xy", "bitstar_connection": "k_nearest",
            "bitstar_samples_per_batch": 200, "bitstar_rewire_factor": 1.1,
            "bitstar_just_in_time_sampling": False, "informed_sample_attempts_per_call": 100,
            "informed_measure_policy": "audit_and_use_documented_conservative_fallback",
            "edge_cache": False, "graph_pruning_requested": True,
            "fail_if_requested_variant_not_effective": True, "simplification": False,
            "approximate_solution_counts_as_success": False},
        "execution": {
            "jobs": 1, "one_process_per_trial": True, "seed_before_creating_planner": True,
            "shuffle_trial_order_seed": 73129, "save_trial_manifest": True,
            "resume_by_config_hash": True, "watchdog_grace_s": 5., "memory_limit_mb": 4096,
            "default_profile": "smoke", "full_requires_explicit_flag": True,
            "timing_origin": "immediately_before_solve", "record_preprocessing_separately": True,
            "record_setup_separately": True, "offline_validation_separately": True},
        "logging": {
            "progress_interval_s": .05, "save_improvement_events": True,
            "save_improvement_paths": True, "save_ompl_benchmark_log": True,
            "save_path_xy": True, "save_path_surface_breakpoints": True, "save_full_tree": False,
            "keep_failures": True, "strict_json_no_nan_or_infinity": True},
        "analysis": {
            "target_relative_tolerances": [.10, .05, .02], "require_reference_provenance": True,
            "confidence_level": .95, "bootstrap_unit_order": ["map", "query", "seed"],
            "ignore_unexecuted_trials_in_success_denominator": True,
            "separately_report_unexecuted_trials": True,
            "include_executed_timeouts_in_success_denominator": True},
        "profiles": {
            "smoke": {"map_extent_m": [20., 20.], "grid_spacing_m": .2,
                "families": list(SMOKE_FAMILIES), "maps_per_family": 1, "queries_per_map": 1,
                "map_seeds": [101], "planner_seeds": [11], "planners": list(PLANNERS),
                "planning_budget_s": 2., "purpose": "pipeline_validation_not_scientific_ranking"},
            "a3_pilot": {"map_extent_m": [40., 40.], "grid_spacing_m": .1,
                "families": list(FAMILIES), "maps_per_family": 2, "queries_per_map": 2,
                "map_seeds": [201, 202], "planner_seeds": [11, 22, 33], "planners": list(PLANNERS),
                "planning_budget_s": 10., "purpose": "exploratory_implementation_and_batch_pilot"}},
        "planner_variants": copy.deepcopy(VARIANTS),
        "a3_pilot_benchmark": {
            "protocol_id": PROTOCOL_ID, "planner_ids": list(PLANNERS),
            "required_query_ids": {"smoke": ["q0"], "a3_pilot": ["q0", "q1"]},
            "expected_trials": dict(EXPECTED_TRIALS), "planner_block_size": 13,
            "order_policy": ORDER_POLICY, "required_allow_flag": "--allow-a3-pilot",
            "seed_replacement_policy": "forbidden", "setup_watchdog_s": 60.,
            "serialization_watchdog_s": 60., "collection_soft_stop_s": 28800.,
            "hard_finalize_s": 32400., "timer_scope": "notebook_first_cell_to_export",
            "primary_metrics": ["cost_terrain_m_at_budget", "first_solution_time_s"],
            "first_solution_time_origin": "immediately_before_solve",
            "bootstrap_replicates": 2000, "bootstrap_seed": 73130,
            "comparison_pairs": [list(pair) for pair in comparison_pairs()]},
        "optional_ablations": {"enabled": False, "require_independent_policy_tests": True,
                               "planners": ["bitstar_a1", "bitstar_a2"]},
        "optional_extensions": {"real_maps": False, "cgal_reference": False, "aitstar": False,
            "abitstar": False, "physics_simulation": False, "constrained_3d_comparison": False},
    }


def _assert_locked(actual, expected, location="config"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{location}: missing/unknown fields in locked protocol")
        for key, value in expected.items():
            _assert_locked(actual[key], value, location + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{location}: differs from locked protocol")
        for index, value in enumerate(expected):
            _assert_locked(actual[index], value, f"{location}[{index}]")
    elif (actual != expected or
          (type(expected) in (bool, int) and type(actual) is not type(expected)) or
          (type(expected) is float and type(actual) not in (int, float))):
        raise ValueError(f"{location}: differs from locked protocol")


def validate_config(config, profile="smoke"):
    if profile not in EXPECTED_TRIALS:
        raise ValueError("schema v7 supports smoke/a3_pilot only; full is forbidden")
    _assert_locked(config, expected_config())
    return config


def relative_map_path(value):
    """Reject platform-specific and escaping paths before joining a dataset root."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("map path must be a relative POSIX path")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or
            any(part in ("", ".", "..") for part in value.split("/")) or
            path.suffix != ".t25d"):
        raise ValueError("map path must stay within the dataset root")
    return path.as_posix()


def plan_trials(config, profile, map_manifest):
    validate_config(config, profile)
    settings = config["profiles"][profile]
    if (map_manifest.get("config_hash") != digest(config) or
            map_manifest.get("profile") != profile or
            map_manifest.get("protocol_id") != PROTOCOL_ID or
            map_manifest.get("seed_replacements") != [] or
            map_manifest.get("seed_replacement_policy") != "forbidden"):
        raise ValueError("map manifest protocol/config/seed policy mismatch")
    maps = map_manifest["maps"]
    expected = [(family, seed) for family in settings["families"] for seed in settings["map_seeds"]]
    by_key = {(m["family"], m["map_seed"]): m for m in maps}
    if len(maps) != len(expected) or set(by_key) != set(expected):
        raise ValueError("map manifest cardinality/declared seed mismatch")
    query_ids = config["a3_pilot_benchmark"]["required_query_ids"][profile]
    blocks = []
    for family, map_seed in expected:
        item = by_key[family, map_seed]
        if item["map_id"] != f"{family}_{map_seed}":
            raise ValueError("map ID must match declared family and seed")
        path = relative_map_path(item["path"])
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("invalid map SHA-256")
        queries = {q["query_id"]: q for q in item["queries"]}
        if len(item["queries"]) != len(query_ids) or set(queries) != set(query_ids):
            raise ValueError("query cardinality/identity mismatch")
        for query_id in query_ids:
            query = queries[query_id]
            query_hash = digest({"start": query["start"], "goal": query["goal"],
                                 "map_hash": item["sha256"], "model": config["model"]})
            if query.get("query_hash") != query_hash:
                raise ValueError("query hash mismatch")
            reference = query.get("reference") or {}
            if (query.get("feasibility") != "feasible_witness_validated" or
                    reference.get("reference_type") != "validated_witness" or
                    reference.get("map_sha256") != item["sha256"] or
                    reference.get("validation", {}).get("valid") is not True):
                raise ValueError("query lacks independent validated witness reference")
            for seed in settings["planner_seeds"]:
                block_index = len(blocks)
                row = block_index % len(PLANNERS)
                ordering = PLANNERS[row:] + PLANNERS[:row]
                block_id = f'{item["map_id"]}_{query_id}_s{seed}'
                block = []
                for position, planner in enumerate(ordering):
                    options = resolved_planner_options(config, planner)
                    block.append({
                        "trial_id": f'{item["map_id"]}_{query_id}_{planner}_s{seed}',
                        "map_id": item["map_id"], "map_path": path, "map_hash": item["sha256"],
                        "family": family, "map_seed": map_seed, "query_id": query_id,
                        "start": query["start"], "goal": query["goal"], "query_hash": query_hash,
                        "reference": reference, "geometry_label": query.get("geometry_label", "smoke_control"),
                        "planner_id": planner, "planner_seed": seed,
                        "backend_planner_id": options["planner_id"], "resolved_optimizations": options,
                        "resolved_optimizations_sha256": digest(options),
                        "block_id": block_id, "block_index": block_index, "latin_row": row,
                        "planner_position": position, "problem_slot_id": block_id,
                        "within_slot_order": position})
                blocks.append(block)
    random.Random(config["execution"]["shuffle_trial_order_seed"]).shuffle(blocks)
    trials = []
    for order, block in enumerate(blocks):
        for entry in block:
            entry.update(block_order=order, problem_slot_order=order, shuffle_order=len(trials))
            trials.append(entry)
    audit_trial_plan(trials, profile)
    return trials


def audit_trial_plan(trials, profile="a3_pilot"):
    """Audit whole blocks, canonical rotations, exact design, and position balance."""
    if profile not in EXPECTED_TRIALS or len(trials) != EXPECTED_TRIALS[profile]:
        raise ValueError("trial cardinality mismatch")
    settings = expected_config()["profiles"][profile]
    size = len(PLANNERS)
    positions = {planner: [0] * size for planner in PLANNERS}
    family_positions = {family: {planner: [0] * size for planner in PLANNERS}
                        for family in settings["families"]}
    canonical = [(family, map_seed, f"q{query}", seed)
                 for family in settings["families"] for map_seed in settings["map_seeds"]
                 for query in range(settings["queries_per_map"]) for seed in settings["planner_seeds"]]
    expected_order = list(range(len(canonical)))
    random.Random(73129).shuffle(expected_order)
    seen = []
    for order, start in enumerate(range(0, len(trials), size)):
        block = trials[start:start + size]
        index = expected_order[order]
        family, map_seed, query_id, seed = canonical[index]
        expected_key = (family, map_seed, query_id, seed)
        row = index % size
        if tuple(e["planner_id"] for e in block) != PLANNERS[row:] + PLANNERS[:row]:
            raise ValueError("canonical block rotation mismatch")
        common = {(e["map_id"], e["map_hash"], e["query_hash"], e["block_id"],
                   tuple(e["start"]), tuple(e["goal"])) for e in block}
        if len(common) != 1:
            raise ValueError("block problem identity mismatch")
        for position, entry in enumerate(block):
            if ((entry["family"], entry["map_seed"], entry["query_id"], entry["planner_seed"]) != expected_key or
                    entry["block_index"] != index or entry["block_order"] != order or
                    entry["planner_position"] != position or entry["shuffle_order"] != start + position or
                    entry["latin_row"] != row or entry["problem_slot_order"] != order or
                    entry["within_slot_order"] != position or entry["problem_slot_id"] != entry["block_id"]):
                raise ValueError("block order/identity metadata mismatch")
            positions[entry["planner_id"]][position] += 1
            family_positions[family][entry["planner_id"]][position] += 1
            seen.append(entry["trial_id"])
    if len(set(seen)) != len(seen):
        raise ValueError("duplicate trial ID")
    for scope in [positions, *family_positions.values()]:
        if any(max(counts) - min(counts) > 1 for counts in scope.values()):
            raise ValueError("planner position imbalance exceeds one")
    return {"policy": ORDER_POLICY, "planned_trials": len(trials), "problem_slots": len(canonical),
            "trials_per_problem_slot": size, "complete_planner_blocks": True,
            "position_counts": positions, "family_position_counts": family_positions,
            "max_position_count_difference": max(max(v) - min(v) for v in positions.values())}


audit_plan = audit_trial_plan


def generate_maps(config, profile, output):
    """Reuse legacy geometry/oracle, then normalize the versioned dataset contract."""
    validate_config(config, profile)
    from generate_scenarios import generate, generate_pilot
    from independent_verifier import Oracle
    output = Path(output)
    projected = copy.deepcopy(config)
    projected["schema_version"] = 1
    try:
        if profile == "a3_pilot":
            projected["profiles"] = {"pilot": projected["profiles"][profile]}
            manifest = generate_pilot(projected, output)
        else:
            manifest = generate(projected, "smoke", output)
        for item in manifest["maps"]:
            path = Path(item["path"])
            item["path"] = relative_map_path(path.resolve().relative_to(output.resolve()).as_posix())
            metadata = read_json(path.with_suffix(".json"))
            if profile == "smoke":
                oracle = Oracle(path)
                for query in item["queries"]:
                    xy = query["witness_path"]
                    checked = oracle.validate_path(xy, query["start"], query["goal"])
                    if not checked["valid"]:
                        raise ValueError("smoke witness failed independent qualification")
                    query["feasibility"] = "feasible_witness_validated"
                    query["geometry_label"] = "smoke_control"
                    query["witnesses"] = [{"witness_id": "w0", "path_xy": xy,
                                           "sha256": digest(xy), "validation": checked}]
                    query["reference"] = {"reference_type": "validated_witness",
                        "cost_terrain_m": checked["cost_terrain_m"], "path_xy": xy,
                        "path_sha256": digest(xy), "validation": checked, "witness_id": "w0",
                        "map_sha256": item["sha256"],
                        "provenance": "fixed geometric smoke witness; independent Oracle; not certified optimum"}
                    query["query_hash"] = digest({"start": query["start"], "goal": query["goal"],
                        "map_hash": item["sha256"], "model": config["model"]})
            metadata.update(queries=item["queries"], generator_version=GENERATOR_VERSION,
                            dataset_role="correctness_smoke" if profile == "smoke" else "exploratory_pilot",
                            seed_selection="predeclared_no_replacement_no_planner_results_used")
            dump(path.with_suffix(".json"), metadata)
        manifest.update(schema_version=7, profile=profile, protocol_id=PROTOCOL_ID,
            generator_version=GENERATOR_VERSION, config_hash=digest(config), held_out=False,
            declared_map_seeds=config["profiles"][profile]["map_seeds"],
            seed_replacement_policy="forbidden", seed_replacements=[], path_base="manifest_directory",
            qualification_policy="every declared map/query witness must pass; abort without seed replacement")
        plan_trials(config, profile, manifest)
        dump(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        dump(output / "qualification_failure.json", {"protocol_id": PROTOCOL_ID, "profile": profile,
            "config_hash": digest(config), "status": "FAILED", "seed_replacements": [],
            "error_type": type(error).__name__, "error": str(error)})
        raise
