"""Opt-in, bounded correctness smoke for an additive traversability objective.

No planner run or parameter ranking is performed by this module. Legacy schemas
and their worker requests remain unchanged.
"""
import copy
import math
from pathlib import Path

from experiment_common import _validate_base_config, digest, dump

PROTOCOL_ID = "a3_traversability_smoke_v1"
GENERATOR_VERSION = "8.0.0"
OBJECTIVE_TYPE = "surface_length_traversability"
FAMILIES = ("flat", "tilted", "unsafe_short_safe_detour", "two_corridors_slope",
            "two_corridors_clearance", "nodata_thin_obstacle")
PLANNERS = ("a3_t_lambda0", "a3_t_lambda025", "a3_t_lambda1", "a3_t_lambda4",
            "a0_t_lambda1", "a3_t_lambda1_optimized")
TRAVERSABILITY_FIELDS = {"feature_version", "lambda", "slope_weight",
                         "clearance_weight", "clearance_comfort_m"}
DEFAULT_TRAVERSABILITY = {"feature_version": "cell_envelope_v1", "lambda": 1.,
                         "slope_weight": .5, "clearance_weight": .5,
                         "clearance_comfort_m": .4}
VARIANTS = {
    "a3_t_lambda0": {"planner_id": "bitstar_a3", "lambda": 0.},
    "a3_t_lambda025": {"planner_id": "bitstar_a3", "lambda": .25},
    "a3_t_lambda1": {"planner_id": "bitstar_a3", "lambda": 1.},
    "a3_t_lambda4": {"planner_id": "bitstar_a3", "lambda": 4.},
    "a0_t_lambda1": {"planner_id": "bitstar_a0", "lambda": 1.},
    "a3_t_lambda1_optimized": {"planner_id": "bitstar_a3", "lambda": 1.,
                               "state_cache": True, "scalar_cost": True,
                               "collision_index": True},
}


def validate_traversability(value):
    if not isinstance(value, dict) or set(value) != TRAVERSABILITY_FIELDS:
        raise ValueError("traversability: missing/unknown fields")
    if value["feature_version"] != "cell_envelope_v1":
        raise ValueError("unsupported traversability feature_version")
    for field in TRAVERSABILITY_FIELDS - {"feature_version"}:
        v = value[field]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError("invalid traversability." + field)
    if value["clearance_comfort_m"] <= 0:
        raise ValueError("clearance_comfort_m must be positive")
    if value["slope_weight"] + value["clearance_weight"] != 1.:
        raise ValueError("traversability feature weights must sum to one")
    return value


def resolved_traversability(config, planner):
    result = copy.deepcopy(config["traversability"])
    definition = config["planner_variants"].get(planner, {})
    if "lambda" in definition:
        result["lambda"] = definition["lambda"]
    return validate_traversability(result)


def objective_identity(config, planner):
    return {"objective_type": config["model"]["objective"],
            "traversability": resolved_traversability(config, planner)}


def validate_config(config, profile="smoke"):
    if profile != "smoke":
        raise ValueError("schema v8 authorizes smoke only; pilot/full are unsupported")
    _validate_base_config(config, profile, traversability=True)
    validate_traversability(config["traversability"])
    settings = config["profiles"][profile]
    expected = {"map_extent_m": [20., 20.], "grid_spacing_m": .2,
                "families": list(FAMILIES), "maps_per_family": 1,
                "queries_per_map": 1, "map_seeds": [101], "planner_seeds": [11],
                "planners": list(PLANNERS), "planning_budget_s": 2.}
    for name, value in expected.items():
        if settings[name] != value:
            raise ValueError("schema v8 smoke design differs: " + name)
    if set(config["planner_variants"]) != set(PLANNERS):
        raise ValueError("schema v8 requires the six declared variants")
    # These names denote fixed scientific comparisons, not arbitrary aliases.
    for planner, expected_variant in VARIANTS.items():
        if config["planner_variants"][planner] != expected_variant:
            raise ValueError("schema v8 variant differs from its declared comparison: " + planner)
        resolved_traversability(config, planner)
    if config["traversability"] != DEFAULT_TRAVERSABILITY:
        raise ValueError("schema v8 smoke feature configuration differs")
    if (config["model"]["robot_radius_m"] != .3 or
            config["model"]["safety_margin_m"] != .1 or
            config["model"]["max_slope_deg"] != 35.):
        raise ValueError("schema v8 fixtures require radius=.3, margin=.1, slope=35")
    if config["algorithm"]["bitstar_samples_per_batch"] != 200:
        raise ValueError("schema v8 comparisons require batch size 200")
    return config


def fixture_geometry(family):
    """Deterministic geometric fixtures; witnesses never seed a planner."""
    start, goal = [2., 10.], [18., 10.]
    rectangles = []
    label = "valid_direct_control"
    if family == "flat":
        height = lambda x, y: 2.
        paths = [[start, goal]]
    elif family == "tilted":
        height = lambda x, y: .24*x + .10*y + 2.
        paths = [[start, goal]]
    elif family == "unsafe_short_safe_detour":
        height = lambda x, y: max(0., 3.-1.2*math.hypot(x-10., y-10.))
        paths = [[start, [6., 5.], [14., 5.], goal]]
        label = "blocked_direct_excessive_slope_safe_detour"
    elif family == "two_corridors_slope":
        start, goal = [2., 9.], [18., 9.]
        height = lambda x, y: max(0., 1.5-.5*abs(y-6.))
        rectangles = [(8., 7., 12., 12.)]
        paths = [[start, [6.5, 6.], [13.5, 6.], goal],
                 [start, [6.5, 14.], [13.5, 14.], goal]]
        label = "blocked_direct_short_sloped_long_flat_corridors"
    elif family == "two_corridors_clearance":
        start, goal = [2., 8.], [18., 8.]
        height = lambda x, y: 0.
        rectangles = [(9.8, 0., 10.2, 7.4), (9.8, 8.6, 10.2, 12.),
                      (9.8, 16., 10.2, 20.)]
        paths = [[start, goal], [start, [8., 14.2], [12., 14.2], goal]]
        label = "valid_direct_narrow_short_wide_long_corridors"
    elif family == "nodata_thin_obstacle":
        height = lambda x, y: None if 9. <= x <= 10. and 7. <= y <= 13. else 0.
        rectangles = [(12., 6., 12.05, 14.)]
        paths = [[start, [7., 4.], [15., 4.], goal]]
        label = "blocked_direct_nodata_and_thin_obstacle"
    else:
        raise ValueError("unsupported traversability fixture: " + family)
    return height, rectangles, start, goal, paths, label


def generate_maps(config, profile, output):
    validate_config(config, profile)
    return _generate_maps(config, profile, output)


def _generate_maps(config, profile, output, *, schema_version=8,
                   protocol_id=PROTOCOL_ID, generator_version=GENERATOR_VERSION):
    """Shared fixture construction; callers validate their own frozen protocol."""
    from independent_verifier import Oracle
    from terrain_io import write_map
    output = Path(output)
    settings, model = config["profiles"][profile], config["model"]
    spacing = settings["grid_spacing_m"]
    nx, ny = (round(side/spacing)+1 for side in settings["map_extent_m"])
    maps = []
    for family in FAMILIES:
        seed = settings["map_seeds"][0]
        map_id = f"{family}_{seed}"
        path = output/(map_id+".t25d")
        height, rectangles, start, goal, paths, label = fixture_geometry(family)
        heights = [[height(i*spacing, j*spacing) for i in range(nx)] for j in range(ny)]
        meta = write_map(path, heights, spacing=(spacing, spacing),
            radius=model["robot_radius_m"], margin=model["safety_margin_m"],
            slope=model["max_slope_deg"], rectangles=rectangles,
            metadata={"map_id": map_id, "family": family, "generator_version": generator_version,
                      "generator_seed": seed, "protocol_id": protocol_id,
                      "geometry_selection": "fixed_correctness_fixture_no_planner_results_used",
                      "feature_model": config["traversability"],
                      "feature_config_sha256": digest(config["traversability"])})
        oracle = Oracle(path)
        witnesses = []
        for index, xy in enumerate(paths):
            checked = oracle.validate_path(xy, start, goal)
            if not checked["valid"]:
                raise ValueError(f"invalid declared witness {map_id}/{index}: {checked}")
            witnesses.append({"witness_id": f"w{index}", "path_xy": xy,
                              "sha256": digest(xy), "validation": checked})
            if family.startswith("two_corridors"):
                risk_checked = oracle.validate_path(xy, start, goal,
                    traversability=config["traversability"])
                if not risk_checked["valid"]:
                    raise ValueError(f"invalid traversability witness {map_id}/{index}: {risk_checked}")
                witnesses[-1]["traversability_validation"] = risk_checked
        if family.startswith("two_corridors"):
            short, long = (w["traversability_validation"] for w in witnesses)
            if not (short["cost_terrain_m"] < long["cost_terrain_m"] and
                    short["risk_integral_m"] > long["risk_integral_m"]):
                raise ValueError("declared length/risk tradeoff qualification failed: " + map_id)
        direct = oracle.validate_path([start, goal], start, goal)
        if direct["valid"] != label.startswith("valid_direct"):
            raise ValueError("fixture direct-edge qualification failed: " + map_id)
        best = min(witnesses, key=lambda item: item["validation"]["cost_terrain_m"])
        reference = {"reference_type": "validated_witness", "cost_terrain_m": best["validation"]["cost_terrain_m"],
                     "path_xy": best["path_xy"], "path_sha256": best["sha256"],
                     "validation": best["validation"], "map_sha256": meta["data_sha256"],
                     "provenance": "independent geometric witness; surface length only; not a weighted-objective optimum"}
        query = {"query_id": "q0", "start": start, "goal": goal, "geometry_label": label,
                 "feasibility": "feasible_witness_validated", "witnesses": witnesses,
                 "direct_edge_validation": direct, "reference": reference}
        if family.startswith("two_corridors"):
            query["tradeoff_qualification"] = {
                "status": "VALIDATED_WITNESS_TRADEOFF", "shorter_witness_id": "w0",
                "lower_risk_witness_id": "w1", "objective_config": config["traversability"],
                "method": "independent_oracle_cell_envelope_integrals",
                "interpretation": "two_declared_feasible_paths_not_planner_performance_or_optimum"}
        query["query_hash"] = digest({"start": start, "goal": goal,
                                    "map_hash": meta["data_sha256"], "model": model})
        meta["queries"] = [query]
        dump(path.with_suffix(".json"), meta)
        maps.append({"map_id": map_id, "family": family, "map_seed": seed,
                     "path": str(path.resolve()), "sha256": meta["data_sha256"], "queries": [query]})
    manifest = {"schema_version": schema_version, "profile": "smoke", "protocol_id": protocol_id,
                "generator_version": generator_version, "config_hash": digest(config),
                "maps": maps, "seed_replacement_policy": "forbidden", "seed_replacements": [],
                "purpose": "correctness_only_not_scientific_ranking"}
    dump(output/"manifest.json", manifest)
    return manifest
