"""Frozen registration for the exact-PRI, length-capped A3-T portfolio.

The native SEM comparison retains its paper reconstruction objective.  No
baseline path or metric is supplied to the portfolio planner.
"""
from __future__ import annotations

import math
from pathlib import Path
import random

from corridor_v4_execution import digest, read
from safe_length_v2_protocol import SMOKE_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_pri_portfolio_v1.json"
PROTOCOL_ID = "a3t_pri_portfolio_v1"
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
ALPHAS = (1., 1.025, 1.05, 1.1)
NATIVE = "sem_native"
PHASES = ("smoke_serial", "smoke_parallel", "pilot", "confirm")


def method_for(alpha):
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha not in ALPHAS:
        raise ValueError("unregistered length cap ratio")
    return "a3t_pri_a" + str(round(alpha * 1000))


METHODS = tuple(map(method_for, ALPHAS)) + (NATIVE,)


def default_config():
    smoke = dict(families=list(SMOKE_FAMILIES), map_seeds=[101], planner_seeds=[11],
                 planning_budget_s=2., extent_m=20., spacing_m=.2, query_id="q0", planned_trials=35)
    return dict(protocol_id=PROTOCOL_ID, schema_version=1,
        model=dict(state="xy", robot_radius_m=.3, safety_margin_m=.1, max_slope_deg=35.,
                   nearest_neighbor_metric="euclidean_xy", objective="surface_length"),
        portfolio=dict(length_cap_ratios=list(ALPHAS), episode_budget_fractions=[.25]*4,
            mask_quantiles=[.90, .75], component_connectivity=8, dilation_cells=0,
            component_rank="descending_anchor_risk_surplus_then_minimum_cell_id",
            selection_order=["PRI", "L_H", "episode_id", "callback_id"],
            no_external_incumbent=True, cap_used_for_candidate_admission_only=True),
        safety=dict(policy_id="a3t_safe_length_v1", risk_cap=40., slope_cap_deg=35., atlas_sampling=True),
        algorithm=dict(initial_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
                       sample_attempts=100, just_in_time_sampling=False),
        native_sem=dict(official_implementation=False, objective="native_length_turn_risk",
            weights=[.48, .16, .36], rolling_optimization=False,
            algorithm_budget_policy="pri_portfolio_v1"),
        profiles=dict(smoke_serial=dict(smoke), smoke_parallel=dict(smoke),
            pilot=dict(families=list(FAMILIES), map_seeds=list(range(41001, 41004)),
                planner_seeds=list(range(416001, 416006)), planning_budget_s=20.,
                extent_m=40., spacing_m=.2, query_id="q0", planned_trials=375),
            confirm=dict(families=list(FAMILIES), map_seeds=list(range(42001, 42011)),
                planner_seeds=list(range(426001, 426011)), planning_budget_s=20.,
                extent_m=40., spacing_m=.2, query_id="q0", planned_trials=1000)),
        execution=dict(max_workers=4, validation_workers=4, threads_per_trial=1, start_method="spawn",
            phase_order=["wave_collection", "wave_independent_validation"], wave_map_count=4,
            process_watchdog_s=600., reserve_bytes_per_collection_worker=512*1024**2,
            reserve_bytes_per_validation_worker=512*1024**2, minimum_free_disk_bytes=4*1024**3,
            retry_started_trials=False, seed_replacements=[], shuffle_seed=83177),
        analysis=dict(final_selection="registered_portfolio_selection_completed_in_budget_before_oracle",
            smoke_selects_alpha=False, tuning_common_cohort="SEM_and_all_four_alphas_final_safe",
            tuning_requires_every_map=True,
            tuning_gate="each_family:mean_delta_PRI<0;mean_delta_L_H<=0;observed_delta_PSR>=0",
            ranking_order=["maximum_family_delta_PRI", "mean_family_delta_L_H", "alpha"],
            confirmation_quality_gate="each_family:upper99.5_delta_PRI<0;upper99.5_delta_L_H<=0",
            psr_gate="observed_no_decrease_not_population_noninferiority",
            bootstrap_replicates=100000, bootstrap_seed=83178, confidence=.995,
            bootstrap_order=["family_fixed", "map", "paired_seed"],
            familywise_error_rate=.05, quality_comparisons=10, no_confirmation_runner_up=True))


def validate_config(config):
    if digest(config) != digest(default_config()):
        raise ValueError("unregistered portfolio configuration; create a new protocol for changes")
    return config


def load_config(path=CONFIG):
    return validate_config(read(path))


def entry_matches(actual, registered):
    """Ignore execution annotations, but distinguish JSON bools from numbers."""
    keys = [key for key in registered if key != "status"]
    return digest({key: actual.get(key) for key in keys}) == digest({key: registered[key] for key in keys})


def inventory(config, phase="pilot", selected_alpha=None):
    validate_config(config)
    if phase not in PHASES:
        raise ValueError("unknown registered phase")
    if phase == "confirm":
        methods = ((method_for(selected_alpha), float(selected_alpha)), (NATIVE, None))
    else:
        if selected_alpha is not None:
            raise ValueError("only confirm accepts a selected alpha")
        methods = tuple(zip(METHODS, (*ALPHAS, None)))
    settings = config["profiles"][phase]
    maps = [(seed, family) for seed in settings["map_seeds"] for family in settings["families"]]
    random.Random(config["execution"]["shuffle_seed"]).shuffle(maps)
    entries, block = [], 0
    for map_seed, family in maps:
        for planner_seed in settings["planner_seeds"]:
            offset = block % len(methods)
            for method, alpha in methods[offset:] + methods[:offset]:
                map_id = f"{family}_{map_seed}"
                entries.append(dict(trial_id=f"{map_id}__s{planner_seed}__{method}",
                    phase=phase, selected_alpha=selected_alpha, map_id=map_id, family=family,
                    map_seed=map_seed, planner_seed=planner_seed, query_id="q0", method=method,
                    alpha=alpha, sem_lambda=None, policy="native" if method == NATIVE else "pri_portfolio",
                    block_index=block, status="NOT_RUN"))
            block += 1
    if len(entries) != settings["planned_trials"]:
        raise AssertionError("registered trial count mismatch")
    return dict(protocol_id=PROTOCOL_ID, schema_version=1, phase=phase, profile=phase,
        status="PREPARED_INVENTORY", selected_alpha=selected_alpha, config_sha256=digest(config),
        entries=entries, planned_trials=len(entries), map_count=len(maps),
        planning_budget_s=settings["planning_budget_s"], planner_execution_count=0, seed_replacements=[])


def source_hashes():
    from safe_length_v2_protocol import source_hashes as scientific_sources
    return scientific_sources()


def request_for(entry, info, config, phase=None, field_path=None, field_metadata=None, bridge=None):
    phase = phase or entry.get("phase")
    if phase != entry.get("phase"):
        raise ValueError("request phase mismatch")
    expected = next((row for row in inventory(config, phase, entry.get("selected_alpha"))["entries"]
                     if row["trial_id"] == entry.get("trial_id")), None)
    if expected is None or not entry_matches(entry, expected):
        raise ValueError("unregistered portfolio map/query/seed/method slot")
    if info.get("map_id") != entry["map_id"] or info.get("query_id", "q0") != "q0":
        raise ValueError("request input differs from registered map/query")
    budget = config["profiles"][phase]["planning_budget_s"]
    if entry["method"] == NATIVE:
        from native_paper_v3_protocol import request_for as native_request, default_config as native_config
        request = native_request("sem", dict(start=info["start"], goal=info["goal"]),
            info["map_path"], info["cloud_path"], info["cloud_sha256"], entry["planner_seed"], budget, native_config())
        request.update(map_sha256=info["map_sha256"], bridge_library=str(Path(bridge).resolve()) if bridge else None,
            library_search_paths=[str(Path(bridge).resolve().parent), "C:/mingw64/bin"] if bridge else [],
            variant_id=NATIVE, algorithm_budget_policy="pri_portfolio_v1")
        return request
    from run_safe_length import request_for as safe_request
    request = safe_request("atlas_adaptive100_200", info,
        dict(planning_budget_s=budget, risk_cap=40., slope_cap_deg=35.), field_path, field_metadata, bridge)
    request.update(variant_id=entry["method"], planner_seed=entry["planner_seed"],
        objective_type="surface_length", pri_portfolio={"length_cap_ratio": entry["alpha"]})
    return request
