"""A4 / RP-BIT* local protocol: phases, methods, trial schedule and worker requests.

Lean by design: one JSON config, an explicit trial inventory, and request
builders that reuse the registered SEM-native and SEM-exposure request shapes.
Dev maps are the already-seen pathsearch pilot inputs; they are not held-out.
"""
from __future__ import annotations

import math
from pathlib import Path
import struct

from corridor_v4_execution import file_hash, read

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a4_rpbitstar_v1"
CONFIG = ROOT / "configs/a4_rpbitstar_v1.json"
# Kaggle held-out: same protocol, 4 lanes, SEM-RRT* on a fixed iteration budget (recorded deviation).
KAGGLE_CONFIG = ROOT / "configs/a4_rpbitstar_v1_kaggle.json"
# Same-time comparison on Kaggle: the locked primary A4 against the C++ SEM-RRT* port, both 20 s.
FAIRTIME_CONFIG = ROOT / "configs/a4_rpbitstar_v1_fairtime_kaggle.json"
# Self-stop comparison on Kaggle: as fairtime, with SEM-RRT* (C++) also stopping on convergence by the A4 rule.
SELFSTOP_CONFIG = ROOT / "configs/a4_rpbitstar_v1_selfstop_kaggle.json"
# Refine-attribution factorial (dev preview): planner {SEM-RRT*, plain BIT*, A4} x rpRefine {off, on}.
FACTORIAL_DEV_CONFIG = ROOT / "configs/a4_rpbitstar_v1_refine_factorial_dev.json"
# PUTN baseline (dev preview): source slope weight and the declared 35-degree tau slope cap, both self-stopping.
PUTN_DEV_CONFIG = ROOT / "configs/a4_rpbitstar_v1_putn_dev.json"
# PUTN held-out on Kaggle: the dev design (both PUTN arms, self-stop SEM-RRT*, locked A4) at 10 s, fresh seeds.
PUTN_KAGGLE_CONFIG = ROOT / "configs/a4_rpbitstar_v1_putn_kaggle.json"
NATIVE = "sem_native"
# sem_refine: a derived trial that replays the finished SEM-RRT* trial of the same slot with A4's rpRefine.
DERIVED = "sem_refine"
# putn_native: PUTN/PF-RRT* paper-first static port on its own objective, run by the native worker like SEM.
PUTN = "putn_native"
# A5 (a5_rpbitstar_v1): A4 plus the sample window on growing batches, the slope shortcut, lazy refinement and the seed.
A5 = "rp_bit_a5"
A5_DEV_CONFIG = ROOT / "configs/a4_rpbitstar_v1_a5_dev.json"
# A5 held-out on Kaggle: the primary A5 arm, its C1 + C2 arm, the locked A4 and the self-stop SEM-RRT*, 20 s, fresh seeds.
A5_KAGGLE_CONFIG = ROOT / "configs/a4_rpbitstar_v1_a5_kaggle.json"
A5_SWITCHES = ("refine", "batch_growth", "fast_validity", "lazy_refine", "seed")
KINDS = ("sem_native", "sem_exposure", "rp_bit", DERIVED, PUTN, A5)
NATIVE_KINDS = ("sem_native", PUTN)
DECISION_RULE = "refine_attribution_v1"
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
SMOKE_FAMILIES = ("flat", "tilted", "unsafe_short_safe_detour", "two_corridors_slope",
                  "two_corridors_clearance", "nodata_thin_obstacle", "flat_40m")
DEV_MANIFEST = ROOT / "results/a3t_pri_pathsearch_v1_20261113/pilot/manifest.json"


def load_config(path=CONFIG):
    config = read(path)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("not an A4 RP-BIT* config")
    for name, method in config["methods"].items():
        if method.get("kind") not in KINDS or (method["kind"] == "sem_native") != (name == NATIVE):
            raise ValueError("invalid method definition: " + name)
        # SEM search engine: the Python reconstruction (default) or its bit-identical C++ port.
        if method.get("engine", "python") not in ("python", "cpp") or ("engine" in method and method["kind"] != "sem_native"):
            raise ValueError("invalid SEM engine: " + name)
        # Optional convergence stop of the C++ SEM search or of PUTN: {"relative_eps", "window_iterations"}.
        if "convergence_stop" in method:
            from sem_native_cpp import check_convergence_stop
            if (method["kind"] != PUTN and method.get("engine") != "cpp") or method["convergence_stop"] is None:
                raise ValueError("convergence stop needs the C++ SEM engine: " + name)
            check_convergence_stop(method["convergence_stop"])
        if method["kind"] == PUTN:
            cap = method.get("tau_slope_cap_deg")
            if (set(method) - {"kind", "convergence_stop", "tau_slope_cap_deg"} or ("tau_slope_cap_deg" in method
                    and (type(cap) is not float or not math.isfinite(cap) or not 0 < cap <= 90))):
                raise ValueError("invalid PUTN method: " + name)
        if method["kind"] == A5:
            # Every switch and size is registered explicitly; lazy refinement and the seed need refinement.
            sizes = [method.get("window_samples"), method.get("batch_min")]
            if (set(method) != {"kind", "lambda", "window_samples", "batch_min", *A5_SWITCHES}
                    or type(method["lambda"]) is not float or not math.isfinite(method["lambda"]) or method["lambda"] < 0
                    or any(type(method[key]) is not bool for key in A5_SWITCHES)
                    or any(type(size) is not int for size in sizes) or sizes[0] < 0 or sizes[1] <= 0
                    or ((method["seed"] or method["lazy_refine"]) and not method["refine"])):
                raise ValueError("invalid A5 method: " + name)
        if method["kind"] == DERIVED:
            times = method.get("refine_at_s")
            if (set(method) != {"kind", "source", "lambda", "refine_at_s"} or method["source"] != NATIVE
                    or config["methods"].get(NATIVE, {}).get("kind") != "sem_native"
                    or type(method["lambda"]) is not float or not math.isfinite(method["lambda"]) or method["lambda"] < 0
                    or type(times) is not list
                    or any(type(t) is not float or not math.isfinite(t) or t <= 0 for t in times)
                    or any(b <= a for a, b in zip(times, times[1:]))):
                raise ValueError("invalid derived method: " + name)
    # Optional validation lanes: validation is not timed, so it may use more cores than the timed collection.
    lanes = config["execution"].get("validation_cores")
    if lanes is not None and (type(lanes) is not list or not lanes or len(set(lanes)) != len(lanes)
                              or any(type(core) is not int or core < 0 for core in lanes)):
        raise ValueError("invalid validation cores")
    for name, phase in config["phases"].items():
        if phase["maps"] not in ("smoke", "dev", "generate") or set(phase["methods"]) - set(config["methods"]):
            raise ValueError("invalid phase definition: " + name)
        if any(config["methods"][m]["kind"] == DERIVED and config["methods"][m]["source"] not in phase["methods"]
               for m in phase["methods"]):
            raise ValueError("derived method without its source: " + name)
        # Optional fixed SEM-RRT* work: stop at this many iterations, with a wall-clock cap instead of the budget.
        work, cap = phase.get("sem_work_iterations"), phase.get("sem_budget_cap_s")
        if (work, cap) != (None, None) and not (type(work) is int and work > 0 and type(cap) is float
                                                and math.isfinite(cap) and cap >= phase["planning_budget_s"]):
            raise ValueError("invalid SEM work budget: " + name)
    # Optional extra paired comparisons [method, baseline] and the registered decision on one of them.
    analysis = config["analysis"]
    if "contrasts" in analysis:
        pairs = analysis["contrasts"]
        if (type(pairs) is not list or not pairs
                or any(type(p) is not list or len(p) != 2 or p[0] == p[1] or set(p) - set(config["methods"]) for p in pairs)
                or len({tuple(p) for p in pairs}) != len(pairs)):
            raise ValueError("invalid analysis contrasts")
        decision = analysis.get("decision")
        if decision is not None and (set(decision) != {"contrast", "rule"} or decision["contrast"] not in pairs
                                     or decision["rule"] != DECISION_RULE):
            raise ValueError("invalid analysis decision")
    elif "decision" in analysis:
        raise ValueError("an analysis decision needs contrasts")
    return config


def maps_for(config, phase):
    """Registered (family, map_seed) pairs of a phase, in schedule order."""
    settings = config["phases"][phase]
    families = SMOKE_FAMILIES if settings["maps"] == "smoke" else FAMILIES
    return [(family, seed) for seed in settings["map_seeds"] for family in families]


def inventory(config, phase):
    """One block per (map, planner seed); method order rotates between blocks.

    A derived method is not rotated: it replays its source's output, so it runs right after the
    source in the same block (same lane, sequential)."""
    settings = config["phases"][phase]
    methods, entries, block = settings["methods"], [], 0
    derived = [m for m in methods if config["methods"][m]["kind"] == DERIVED]
    base = [m for m in methods if m not in derived]
    for family, map_seed in maps_for(config, phase):
        map_id = f"{family}_{map_seed}"
        for planner_seed in settings["planner_seeds"]:
            offset = block % len(base)
            order = [name for m in base[offset:] + base[:offset]
                     for name in [m] + [d for d in derived if config["methods"][d]["source"] == m]]
            for method in order:
                entries.append(dict(trial_id=f"{map_id}__s{planner_seed}__{method}", map_id=map_id,
                    family=family, map_seed=map_seed, planner_seed=planner_seed, query_id="q0",
                    method=method, phase=phase, block_index=block))
            block += 1
    return entries


def request_for(entry, info, config, phase=None, field_path=None, field_metadata=None, bridge=None):
    phase = phase or entry["phase"]
    method = config["methods"][entry["method"]]
    budget = config["phases"][phase]["planning_budget_s"]
    if method["kind"] == "sem_native":
        from native_paper_v3_protocol import request_for as native_request, default_config as native_config
        work = config["phases"][phase].get("sem_work_iterations", 0)
        request = native_request("sem", dict(start=info["start"], goal=info["goal"]), info["map_path"],
            info["cloud_path"], info["cloud_sha256"], entry["planner_seed"],
            config["phases"][phase]["sem_budget_cap_s"] if work else budget, native_config())
        request.update(map_sha256=info["map_sha256"], bridge_library=str(Path(bridge).resolve()),
            library_search_paths=[str(Path(bridge).resolve().parent), "C:/mingw64/bin"],
            variant_id=NATIVE, algorithm_budget_policy="pri_portfolio_v1")
        if work:
            request["work_iterations"] = work
        if method.get("engine") == "cpp":
            from sem_native_cpp import library_beside
            library = library_beside(bridge).resolve()
            request.update(sem_engine="cpp", sem_library=str(library), sem_library_sha256=file_hash(library))
        if "convergence_stop" in method:
            request["sem_convergence_stop"] = dict(method["convergence_stop"])
        return request
    if method["kind"] == PUTN:
        # Same native request shape, master cloud and budget as SEM-RRT*; no SEM-only clock policy.
        from native_paper_v3_protocol import request_for as native_request, default_config as native_config
        request = native_request("putn", dict(start=info["start"], goal=info["goal"]), info["map_path"],
            info["cloud_path"], info["cloud_sha256"], entry["planner_seed"], budget, native_config())
        request.update(map_sha256=info["map_sha256"], bridge_library=str(Path(bridge).resolve()),
            library_search_paths=[str(Path(bridge).resolve().parent), "C:/mingw64/bin"], variant_id=entry["method"])
        if "convergence_stop" in method:
            request["putn_convergence_stop"] = dict(method["convergence_stop"])
        if "tau_slope_cap_deg" in method:
            request["putn_tau_slope_cap_deg"] = method["tau_slope_cap_deg"]
        return request
    from run_safe_length import request_for as safe_request
    request = safe_request("atlas_adaptive100_200", info, dict(planning_budget_s=budget,
        risk_cap=config["safety"]["risk_cap"], slope_cap_deg=config["safety"]["slope_cap_deg"]),
        field_path, field_metadata, bridge)
    if method["kind"] == DERIVED:
        trial = Path(field_path).parent
        if trial.name != entry["trial_id"]:
            raise ValueError("a derived request is built inside its own trial directory")
        source = trial.parent / f"{entry['map_id']}__s{entry['planner_seed']}__{method['source']}"
        replay = {key: request[key] for key in ("map_path", "start", "goal", "planning_budget_s", "collision_index",
                                                "cloud_phase_handshake", "safe_length")}
        replay.update(planner_id="rp_refine_replay", variant_id=entry["method"], planner_seed=entry["planner_seed"],
                      objective_type="surface_length_sem_exposure",
                      sem_exposure={"lambda": method["lambda"], "risk_scale": 40.},
                      rp_refine_replay=replay_spec(source, method))
        return replay
    request.update(variant_id=entry["method"], planner_seed=entry["planner_seed"],
        objective_type="surface_length_sem_exposure",
        sem_exposure={"lambda": method["lambda"], "risk_scale": 40.})
    if method["kind"] == "rp_bit":
        request["rp_bit"] = {key: value for key, value in method.items() if key not in ("kind", "lambda")}
    if method["kind"] == A5:
        request["rp_bit_a5"] = {key: value for key, value in method.items() if key not in ("kind", "lambda")}
        request["batch_after_first_solution"] = method["batch_min"]
    return request


def coordinate_bits(value):
    """IEEE-754 bit pattern of a coordinate (the C++ JSON parser is not exact on decimals)."""
    return struct.pack(">d", value).hex()


def coordinate_from_bits(bits):
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def replay_spec(source, method):
    """Replay input of a derived trial: every published incumbent of the finished SEM-RRT* trial `source`."""
    source = Path(source)
    if read(source / "execution.json").get("status") != "COMPLETED":
        raise ValueError("replay source did not complete: " + source.name)
    worker = read(source / "worker.json")
    timings = worker["phase_timings"]
    return dict(source_trial_id=source.name, source_worker_sha256=file_hash(source / "worker.json"),
                refine_at_s=list(method["refine_at_s"]), source_solve_s=timings["solve_s"],
                source_preprocessing_s=timings["total_before_solve_s"], source_planner_status=worker["planner_status"],
                source_termination_reason=worker["termination_reason"],
                incumbents=[dict(snapshot_id=item["snapshot_id"], elapsed_s=item["elapsed_s"],
                                 path_bits=[[coordinate_bits(x), coordinate_bits(y)] for x, y in item["path_xy"]])
                            for item in worker.get("snapshots") or []])
