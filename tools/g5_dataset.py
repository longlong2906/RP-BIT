"""Frozen, planner-independent G5 held-out geometry and witness qualification.

This module never invokes a planner. Independent Oracle geometry determines
qualification; a failed criterion stops generation without replacing any seed.
"""
import argparse
import gc
import math
from pathlib import Path
import random

import numpy as np

from experiment_common import digest, dump, file_hash, load_config
from independent_verifier import Oracle
from terrain_io import write_map

GENERATOR_VERSION = "g5-heldout-1.0.0"
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain",
            "narrow_passage", "mixed_obstacles")
MAP_SEEDS = tuple(range(1001, 1009))
QUERY_IDS = ("q0", "q1", "q2", "q3")
RASTER_SPACING_M = .25
PARAMETER_RANGES = {
    "traversable_hill": {"height_m": [2.5, 6.], "radial_slope": [.32, .44],
        "center_x": [18., 22.], "center_y": [17., 23.],
        "plane_x": [0., .04], "plane_y": [-.025, .025]},
    "ridge_pass": {"height_m": [2., 5.5], "radial_slope": [.20, .45],
        "center_x": [18., 22.], "wall_thickness_m": [.4, .9],
        "narrow_center_y": [10., 14.], "narrow_width_m": [1.2, 2.4],
        "wide_center_y": [27., 31.], "wide_width_m": [3., 6.],
        "plane_x": [0., .05], "plane_y": [-.03, .03]},
    "rolling_terrain": {"amplitude_m": [.5, 1.6], "x_scale_m": [4.5, 8.],
        "y_scale_m": [4.5, 9.], "phase_x_rad": [0., 2*math.pi],
        "phase_y_rad": [0., 2*math.pi], "plane_x": [0., .08],
        "plane_y": [-.02, .04], "obstacle_center_x": [18., 22.],
        "obstacle_center_y": [18., 22.], "obstacle_halfwidth_m": [1.5, 3.],
        "obstacle_halfheight_m": [3., 6.]},
    "narrow_passage": {"center_x": [18., 22.], "wall_thickness_m": [.35, .8],
        "narrow_center_y": [14., 18.], "narrow_width_m": [.95, 1.8],
        "wide_center_y": [28., 32.], "wide_width_m": [2.5, 5.],
        "plane_x": [-.03, .09], "plane_y": [-.03, .05],
        "amplitude_m": [.2, .7]},
    "mixed_obstacles": {"height_m": [1., 3.], "sigma_m": [5., 8.],
        "center_x": [18., 22.], "center_y": [17., 23.],
        "amplitude_m": [.2, .8], "y_scale_m": [5., 9.],
        "plane_x": [.01, .07], "plane_y": [-.025, .025],
        "obstacle_center_x": [19., 21.], "obstacle_center_y": [18., 22.],
        "obstacle_halfwidth_m": [2., 3.5], "obstacle_halfheight_m": [4., 5.5],
        "nodata_shift_x": [-.5, .5], "nodata_shift_y": [-.5, .5],
        "left_obstacle_shift_x": [-.8, .8], "right_obstacle_shift_x": [-.8, .8]},
}


def query_seed(family, map_seed, query_index):
    return 900000 + 10000*FAMILIES.index(family) + 100*map_seed + query_index


def geometry_definition(family, seed):
    """Return formula, obstacles, four fixed query constructions and parameters."""
    if family not in FAMILIES:
        raise ValueError("unsupported G5 family")
    rng = random.Random(seed + 100000*FAMILIES.index(family))
    params = {key: rng.uniform(*bounds) for key, bounds in PARAMETER_RANGES[family].items()}
    p = params
    ax, by = p["plane_x"], p["plane_y"]
    rectangles = []
    if family == "traversable_hill":
        cx, cy, a, slope = p["center_x"], p["center_y"], p["height_m"], p["radial_slope"]
        radius = a/slope
        params["radius_m"] = radius
        fn = lambda x, y: ax*x + by*y + max(0., a-slope*math.hypot(x-cx, y-cy))
        levels = [max(5., cy-.65*radius), cy, min(35., cy+.65*radius), 35.]
    elif family in ("ridge_pass", "narrow_passage"):
        cx, thickness = p["center_x"], p["wall_thickness_m"]
        centers = [p["narrow_center_y"], p["wide_center_y"]]
        widths = [p["narrow_width_m"], p["wide_width_m"]]
        gaps = [(c-w/2, c+w/2) for c, w in zip(centers, widths)]
        rectangles = [(cx-thickness/2, 0., cx+thickness/2, gaps[0][0]),
                      (cx-thickness/2, gaps[0][1], cx+thickness/2, gaps[1][0]),
                      (cx-thickness/2, gaps[1][1], cx+thickness/2, 40.)]
        params["gaps_y"] = gaps
        params["centerline_clearance_widths_m"] = [w-.8 for w in widths]
        if family == "ridge_pass":
            fn = lambda x, y: ax*x + by*y + max(0., p["height_m"]-p["radial_slope"]*abs(x-cx))
            levels = [19., 23., *centers]
        else:
            fn = lambda x, y: ax*x + by*y + p["amplitude_m"]*math.sin(x/5.)*math.cos(y/7.)
            params["x_scale_m"], params["y_scale_m"] = 5., 7.
            levels = [22., 24., *centers]
    elif family == "rolling_terrain":
        fn = lambda x, y: ax*x + by*y + p["amplitude_m"]*math.sin(x/p["x_scale_m"]+p["phase_x_rad"])*math.cos(y/p["y_scale_m"]+p["phase_y_rad"])
        cx, cy = p["obstacle_center_x"], p["obstacle_center_y"]
        hw, hh = p["obstacle_halfwidth_m"], p["obstacle_halfheight_m"]
        rectangles = [(cx-hw, cy-hh, cx+hw, cy+hh)]
        levels = [cy, cy-1., 7., 33.]
    else:
        sx, sy = p["nodata_shift_x"], p["nodata_shift_y"]
        nodata = [10.5+sx, 17.5+sy, 12.+sx, 20.5+sy]
        params["nodata_node_rectangle"] = nodata
        def fn(x, y):
            if nodata[0] <= x <= nodata[2] and nodata[1] <= y <= nodata[3]:
                return None
            return (ax*x + by*y + p["amplitude_m"]*math.sin(y/p["y_scale_m"]) +
                    p["height_m"]*math.exp(-.5*((x-p["center_x"])**2+(y-p["center_y"])**2)/p["sigma_m"]**2))
        cx, cy = p["obstacle_center_x"], p["obstacle_center_y"]
        hw, hh = p["obstacle_halfwidth_m"], p["obstacle_halfheight_m"]
        left, right = p["left_obstacle_shift_x"], p["right_obstacle_shift_x"]
        rectangles = [(cx-hw, cy-hh, cx+hw, cy+hh),
                      (9.+left, 27., 12.+left, 34.), (28.+right, 6., 31.+right, 12.)]
        levels = [19., 21., 5., 36.]
    definitions = []
    for index, level in enumerate(levels):
        qseed = query_seed(family, seed, index)
        qrng = random.Random(qseed)
        # Endpoints are generated solely from a separate, predeclared query RNG.
        y = level + qrng.uniform(-.02 if index >= 2 else -.2, .02 if index >= 2 else .2)
        start, goal = [qrng.uniform(3.5, 4.5), y], [qrng.uniform(35.5, 36.5), y]
        if family == "traversable_hill":
            candidates = [[start, [cx-2., cross], [cx+2., cross], goal]
                          for cross in (max(3., cy-radius-1.), min(37., cy+radius+1.))]
            label = "valid_direct_terrain_crossing" if index < 3 else "valid_direct_control"
        elif family in ("ridge_pass", "narrow_passage"):
            xl, xr = cx-thickness/2-1.2, cx+thickness/2+1.2
            if index < 2:
                candidates = [[start, [xl, cross], [xr, cross], goal] for cross in centers]
                label = "blocked_direct_two_pass_routes"
            else:
                cross = centers[index-2]
                candidates = [[start, [8., y+.75], [xl, cross], [xr, cross], goal]]
                label = "valid_direct_narrow_pass_control" if index == 2 else "valid_direct_wide_pass_control"
        elif family == "rolling_terrain":
            if index < 2:
                candidates = [[start, [cx-hw-1.2, cross], [cx+hw+1.2, cross], goal]
                              for cross in (cy-hh-1.2, cy+hh+1.2)]
                label = "blocked_direct_competing_routes"
            else:
                candidates = [[start, [20., y+.75], goal]]
                label = "valid_direct_rolling_control"
        else:
            if index < 2:
                candidates = [[start, [14., 8.], [26., 8.], [26., 15.], goal],
                              [start, [6., 35.5], [34., 35.5], goal]]
                label = "blocked_direct_mixed_routes"
            else:
                candidates = [[start, [20., y+(-.75 if index == 2 else .75)], goal]]
                label = "valid_direct_mixed_control"
        definitions.append({"query_id": QUERY_IDS[index], "query_seed": qseed,
                            "start": start, "goal": goal, "paths": candidates,
                            "geometry_label": label})
    return fn, rectangles, definitions, params


def raster_free_domain(oracle, spacing):
    """Midpoint quadrature of the exact footprint-valid domain, independent of C++."""
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("invalid raster spacing")
    x0, y0, x1, y1 = oracle.bounds
    nx, ny = round((x1-x0)/spacing), round((y1-y0)/spacing)
    if not nx or not ny or abs(nx*spacing-(x1-x0)) > 1e-10 or abs(ny*spacing-(y1-y0)) > 1e-10:
        raise ValueError("raster must exactly partition map bounds")
    xs, ys = x0+(np.arange(nx)+.5)*spacing, y0+(np.arange(ny)+.5)*spacing
    xx, yy = np.meshgrid(xs, ys)
    radius = oracle.radius+oracle.margin
    valid = (xx-radius > x0) & (xx+radius < x1) & (yy-radius > y0) & (yy+radius < y1)
    # Rectangle broad phase restricts only grid indices; distance is exact for a disk.
    for rx0, ry0, rx1, ry1 in oracle.forbidden:
        i0, i1 = np.searchsorted(xs, [rx0-radius-1e-13, rx1+radius+1e-13], side="left")
        j0, j1 = np.searchsorted(ys, [ry0-radius-1e-13, ry1+radius+1e-13], side="left")
        if i1 <= i0 or j1 <= j0:
            continue
        qx, qy = xx[j0:j1, i0:i1], yy[j0:j1, i0:i1]
        dx = np.maximum(np.maximum(rx0-qx, qx-rx1), 0.)
        dy = np.maximum(np.maximum(ry0-qy, qy-ry1), 0.)
        valid[j0:j1, i0:i1] &= np.hypot(dx, dy) > radius+1e-13
    points = np.column_stack((xx[valid], yy[valid]))
    return points, {"spacing_m": spacing, "grid_shape_xy": [nx, ny],
                    "total_midpoints": nx*ny, "free_midpoints": len(points),
                    "area_D_free_m2": len(points)*spacing**2,
                    "integration": "deterministic_midpoint_raster",
                    "error_note": "finite-resolution quadrature estimate, not exact continuous area; no adaptive refinement"}


def oracle_heights(oracle, points):
    """Evaluate Oracle's independently solved affine triangles at valid points."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    u = (points[:, 0]-oracle.x0)/oracle.dx
    v = (points[:, 1]-oracle.y0)/oracle.dy
    ix = np.minimum(np.floor(u).astype(int), oracle.nx-2)
    iy = np.minimum(np.floor(v).astype(int), oracle.ny-2)
    if np.any(ix < 0) or np.any(iy < 0) or np.any(u > oracle.nx-1) or np.any(v > oracle.ny-1):
        raise ValueError("height point outside map")
    triangle = ((v-iy) > (u-ix)).astype(int)
    ids = oracle.cell_triangle_ids[iy, ix, triangle]
    if np.any(ids < 0):
        raise ValueError("height point in NoData cell")
    coefficients = oracle.coefficients[ids]
    return np.sum(coefficients[:, :2]*points, axis=1)+coefficients[:, 2]


def informed_geometry(oracle, start, goal, c_ref, spacing=RASTER_SPACING_M, prepared=None):
    if not math.isfinite(c_ref) or c_ref <= 0:
        raise ValueError("c_ref must be finite and positive")
    if not oracle.valid_edge(start, start) or not oracle.valid_edge(goal, goal):
        raise ValueError("invalid geometry query endpoint")
    if prepared is None:
        points, raster = raster_free_domain(oracle, spacing)
        heights = oracle_heights(oracle, points)
    else:
        points, heights, raster = prepared
        if raster["spacing_m"] != spacing:
            raise ValueError("raster spacing mismatch")
    zstart, zgoal = oracle_heights(oracle, [start, goal])
    dstart, dgoal = np.linalg.norm(points-start, axis=1), np.linalg.norm(points-goal, axis=1)
    fxy = dstart+dgoal
    fphi = np.hypot(dstart, heights-zstart)+np.hypot(dgoal, heights-zgoal)
    e, informed = fxy < c_ref, fphi < c_ref
    if np.any(informed & ~e) or np.any(fphi < fxy-1e-12):
        raise ValueError("informed-set containment failed")
    delta = fphi[e]-fxy[e]
    area_e, area_i = int(e.sum())*spacing**2, int(informed.sum())*spacing**2
    return {"reference_type": "WITNESS_UPPER_BOUND", "c_ref_m": c_ref,
            "area_E": area_e, "area_I": area_i, "area_units": "m2",
            "informed_area_ratio": area_i/area_e if area_e else None,
            "mean_delta_f": float(np.mean(delta)) if len(delta) else None,
            "median_delta_f": float(np.median(delta)) if len(delta) else None,
            "p90_delta_f": float(np.quantile(delta, .9)) if len(delta) else None,
            "max_delta_f": float(np.max(delta)) if len(delta) else None,
            "delta_units": "m", "E_midpoint_count": int(e.sum()),
            "I_midpoint_count": int(informed.sum()), "raster": raster,
            "unavailable_reason": None if area_e else "no midpoint in E_xy at frozen raster resolution; ratio and delta summaries undefined",
            "set_definitions": "E={q in D_free:f_xy(q)<c_ref}; I={q in D_free:f_phi(q)<c_ref}",
            "selection_use": "descriptive prespecified covariates only; never query exclusion"}


def qualify_queries(oracle, definitions, map_hash, model, *, clearance_metrics=True):
    queries = []
    for definition in definitions:
        start, goal = definition["start"], definition["goal"]
        witnesses = []
        for index, xy in enumerate(definition["paths"]):
            validation = oracle.validate_path(xy, start, goal, clearance_metrics=clearance_metrics)
            if not validation["valid"]:
                raise ValueError(f"witness criterion failed: {definition['query_id']}/w{index}: {validation}")
            witnesses.append({"witness_id": f"w{index}", "path_xy": xy,
                              "sha256": digest(xy), "validation": validation})
        direct = oracle.validate_path([start, goal], start, goal, clearance_metrics=clearance_metrics)
        if direct["valid"] != definition["geometry_label"].startswith("valid_direct"):
            raise ValueError(f"direct-edge geometry criterion failed: {definition['query_id']}")
        best = min(witnesses, key=lambda witness: witness["validation"]["cost_terrain_m"])
        query = {key: definition[key] for key in ("query_id", "query_seed", "start", "goal", "geometry_label")}
        query.update({"feasibility": "feasible_witness_validated", "witnesses": witnesses,
            "direct_edge_validation": direct,
            "query_hash": digest({"start": start, "goal": goal, "map_hash": map_hash, "model": model}),
            "reference": {"reference_type": "WITNESS_UPPER_BOUND",
                "cost_terrain_m": best["validation"]["cost_terrain_m"],
                "path_xy": best["path_xy"], "path_sha256": best["sha256"],
                "map_sha256": map_hash, "witness_id": best["witness_id"],
                "validation": best["validation"],
                "provenance": "minimum of predeclared geometric witness polylines validated by independent Oracle before planners; feasible upper bound, not optimum; never provided as incumbent"}})
        queries.append(query)
    return queries


def validate_full_profile(config):
    p = config["profiles"]["full"]
    expected = {"map_extent_m": [40., 40.], "grid_spacing_m": .1,
        "families": list(FAMILIES), "map_seeds": list(MAP_SEEDS),
        "maps_per_family": 8, "queries_per_map": 4, "planner_seeds": list(range(1, 11)),
        "planners": ["bitstar_a0", "bitstar_a3", "informed_rrtstar", "rrtstar"]}
    if any(p.get(key) != value for key, value in expected.items()):
        raise ValueError("full profile differs from frozen G5 identities/design")
    model = config["model"]
    if any(model[key] != value for key, value in {"robot_radius_m": .3, "safety_margin_m": .1,
                                                "max_slope_deg": 35., "roughness_limit_m": None}.items()):
        raise ValueError("G5 model differs from frozen footprint/slope contract")


def generate(config, output, protocol):
    validate_full_profile(config)
    output, protocol = Path(output), Path(protocol)
    if not protocol.is_file():
        raise ValueError("G5 protocol must exist before held-out generation")
    if output.exists() and any(output.iterdir()):
        raise ValueError("generation output is not empty; never overwrite/reselect held-out inputs")
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = file_hash(protocol)
    p, model = config["profiles"]["full"], config["model"]
    spacing = p["grid_spacing_m"]
    nx, ny = [round(extent/spacing)+1 for extent in p["map_extent_m"]]
    maps, all_geometry = [], []
    dump(output/"generation_protocol.json", {"generator_version": GENERATOR_VERSION,
        "generator_sha256": file_hash(__file__), "protocol_sha256": protocol_hash,
        "config_sha256": digest(config), "parameter_ranges": PARAMETER_RANGES,
        "map_seeds": MAP_SEEDS, "query_ids": QUERY_IDS,
        "query_seed_formula": "900000+10000*family_index+100*map_seed+query_index",
        "raster_spacing_m": RASTER_SPACING_M,
        "qualification": "all predeclared witnesses valid; valid_direct or blocked_direct criterion; failure stops dataset, no replacement"})
    for family in FAMILIES:
        for seed in MAP_SEEDS:
            map_id = f"{family}_{seed}"
            path = output/(map_id+".t25d")
            try:
                fn, rectangles, definitions, params = geometry_definition(family, seed)
                heights = [[fn(i*spacing, j*spacing) for i in range(nx)] for j in range(ny)]
                meta = write_map(path, heights, spacing=(spacing, spacing),
                    radius=model["robot_radius_m"], margin=model["safety_margin_m"],
                    slope=model["max_slope_deg"], rectangles=rectangles,
                    metadata={"map_id": map_id, "family": family,
                        "generator_version": GENERATOR_VERSION, "generator_seed": seed,
                        "parameters": params, "geometry_selection": "frozen_no_planner_results_used"})
                oracle = Oracle(path)
                queries = qualify_queries(oracle, definitions, meta["data_sha256"], model)
                points, raster = raster_free_domain(oracle, RASTER_SPACING_M)
                prepared = points, oracle_heights(oracle, points), raster
                for query in queries:
                    geometry = informed_geometry(oracle, query["start"], query["goal"],
                                                 query["reference"]["cost_terrain_m"], prepared=prepared)
                    geometry.update({"map_id": map_id, "family": family, "map_seed": seed,
                                     "query_id": query["query_id"], "query_seed": query["query_seed"],
                                     "query_hash": query["query_hash"], "map_sha256": meta["data_sha256"]})
                    query["terrain_informed_geometry"] = geometry
                    all_geometry.append(geometry)
                meta["queries"] = queries
                dump(path.with_suffix(".json"), meta)
                maps.append({"map_id": map_id, "family": family, "map_seed": seed,
                    "path": str(path.resolve()), "sha256": meta["data_sha256"], "queries": queries})
                del oracle, prepared, points
                gc.collect()
                print(f"Qualified G5 geometry: {map_id} (4 witnesses/queries; no planner run)", flush=True)
            except Exception as error:
                dump(output/"qualification_failure.json", {"status": "BLOCKED",
                    "map_id": map_id, "error": str(error), "replacement_permitted": False,
                    "qualified_maps_before_failure": [record["map_id"] for record in maps]})
                raise
    if file_hash(protocol) != protocol_hash:
        raise ValueError("protocol changed during held-out generation")
    manifest = {"schema_version": 3, "profile": "full", "generator_version": GENERATOR_VERSION,
        "generator_sha256": file_hash(__file__), "config_hash": digest(config),
        "protocol_sha256": protocol_hash, "maps": maps,
        "planned_map_count": 40, "planned_query_count": 160,
        "geometry_protocol": "predeclared deterministic geometric qualification; no seed/query substitution; witness never seeds planners"}
    dump(output/"terrain_informed_geometry.json", {"schema_version": 1,
        "generator_version": GENERATOR_VERSION, "computed_before_planners": True,
        "raster_spacing_m": RASTER_SPACING_M, "queries": all_geometry})
    dump(output/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    if not args.allow_full:
        parser.error("held-out generation requires explicit --allow-full")
    config = load_config(args.config, "pilot")
    generate(config, args.output, args.protocol)


if __name__ == "__main__":
    main()
